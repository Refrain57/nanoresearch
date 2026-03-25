"""Research runner — orchestrates the full research pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from nanobot.research.types import (
    ExecutionLog,
    ResearchConfig,
    ResearchPlan,
    ResearchResult,
    ResearchStatus,
    SearchIterationLog,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.research.knowledge_processor import KnowledgeProcessor
    from nanobot.research.knowledge_search import KnowledgeSearch

# Import sub-components lazily to avoid circular imports
_PLANNER: type | None = None
_SEARCHER: type | None = None
_SYNTHESIZER: type | None = None
_REFINER: type | None = None
_REPORTER: type | None = None
_KNOWLEDGE_PROCESSOR: type | None = None


def _lazy_imports() -> None:
    global _PLANNER, _SEARCHER, _SYNTHESIZER, _REFINER, _REPORTER, _KNOWLEDGE_PROCESSOR
    if _PLANNER is None:
        from nanobot.research.planner import ResearchPlanner
        from nanobot.research.searcher import SearchOrchestrator
        from nanobot.research.synthesizer import InformationSynthesizer
        from nanobot.research.refiner import ResearchRefiner
        from nanobot.research.reporter import ReportGenerator
        from nanobot.research.knowledge_processor import KnowledgeProcessor

        _PLANNER = ResearchPlanner
        _SEARCHER = SearchOrchestrator
        _SYNTHESIZER = InformationSynthesizer
        _REFINER = ResearchRefiner
        _REPORTER = ReportGenerator
        _KNOWLEDGE_PROCESSOR = KnowledgeProcessor


class ResearchRunner:
    """Main orchestrator for the Auto Research pipeline.

    Pipeline: Planner → Searcher → Synthesizer → [Refiner] → Reporter
              ↑_____________________________if needs more iterations____________↓

    The refiner decides whether to loop back for supplementary searches.
    After all iterations, the reporter generates the final report and self-evaluates it.

    Knowledge Loop: After completion, the result is processed by KnowledgeProcessor
    to extract claims and insights for future research.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        web_search_tool: Any,
        web_fetch_tool: Any,
        config: ResearchConfig | None = None,
        knowledge_search: KnowledgeSearch | None = None,
        knowledge_processor: KnowledgeProcessor | None = None,
        rag_store: Any = None,  # ChromaStore for user uploaded documents
    ) -> None:
        _lazy_imports()
        self.provider = provider
        self.model = model
        self.config = config or ResearchConfig()

        self.planner = _PLANNER(provider, model)
        self.searcher = _SEARCHER(web_search_tool, web_fetch_tool, self.config)
        self.synthesizer = _SYNTHESIZER(provider, model)
        self.refiner = _REFINER(provider, model)
        self.reporter = _REPORTER(provider, model)

        # Knowledge loop components (optional)
        self.knowledge_search = knowledge_search
        self.knowledge_processor = knowledge_processor
        self.rag_store = rag_store  # User uploaded documents

        self._results: dict[str, ResearchResult] = {}

    async def run(
        self,
        topic: str,
        depth: str | None = None,
        research_id: str | None = None,
    ) -> ResearchResult:
        """Execute the full research pipeline.

        Args:
            topic: Research topic or question.
            depth: One of "quick", "normal", "deep".
            research_id: Optional existing ID (for continuation).

        Returns:
            ResearchResult with report, metrics, and metadata.
        """
        rid = research_id or str(uuid.uuid4())[:8]
        depth = depth or self.config.default_depth

        # Adjust config based on depth
        max_iterations = self.config.max_iterations
        max_sources = self.config.max_sources_per_question
        if depth == "quick":
            max_iterations = 1
            max_sources = 5
        elif depth == "deep":
            max_iterations = 5
            max_sources = 20
        elif depth == "normal":
            max_iterations = 3
            max_sources = 10

        # Apply adjusted sources to searcher so each iteration uses depth-appropriate count
        adjusted_config = type(self.config)(
            max_iterations=max_iterations,
            max_sources_per_question=max_sources,
            min_coverage_threshold=self.config.min_coverage_threshold,
            search_timeout=self.config.search_timeout,
            default_depth=self.config.default_depth,
            enable_self_evaluation=self.config.enable_self_evaluation,
            evaluation_threshold=self.config.evaluation_threshold,
        )
        # Also update searcher's search_count to match
        self.searcher.search_count = max_sources

        result = ResearchResult(topic=topic, status=ResearchStatus.PLANNING, id=rid)
        self._results[rid] = result

        # 初始化执行日志
        execution_log = ExecutionLog(
            research_id=rid,
            topic=topic,
            depth=depth,
            config={
                "max_iterations": max_iterations,
                "max_sources_per_question": max_sources,
                "min_coverage_threshold": self.config.min_coverage_threshold,
                "rerank_enabled": self.config.rerank_enabled,
            },
        )

        logger.info(
            "ResearchRunner[{}]: starting topic='{}' depth='{}' max_iter={} max_sources={}",
            rid, topic, depth, max_iterations, max_sources,
        )

        try:
            # Phase 0: Pre-query existing knowledge
            existing_context = ""
            if self.knowledge_search:
                existing_context = await self._get_existing_knowledge(topic)
                if existing_context:
                    logger.info(
                        "ResearchRunner[{}]: found existing knowledge ({} chars)",
                        rid, len(existing_context),
                    )

            # Phase 0.5: Query user-uploaded documents
            document_context = ""
            if self.rag_store:
                document_context = await self._get_document_context(topic)
                if document_context:
                    logger.info(
                        "ResearchRunner[{}]: found relevant documents ({} chars)",
                        rid, len(document_context),
                    )

            # Combine contexts for planning
            combined_context = existing_context
            if document_context:
                if combined_context:
                    combined_context = f"{combined_context}\n\n## 用户上传的相关文档\n{document_context}"
                else:
                    combined_context = f"## 用户上传的相关文档\n{document_context}"

            # Phase 1: Planning (with existing knowledge context)
            plan = await self.planner.plan(topic, depth, existing_context=combined_context)
            plan.iteration = 0
            result.plan = plan
            logger.info("ResearchRunner[{}]: plan created with {} sub-questions", rid, len(plan.sub_questions))

            synthesis = None

            # Phase 2-4: Search + Synthesize + Iterate
            for iteration in range(max_iterations):
                result.status = ResearchStatus.SEARCHING
                logger.info("ResearchRunner[{}]: iteration {} — searching", rid, iteration)

                # 收集本轮搜索的关键词
                keywords_searched = []
                for sq in plan.sub_questions:
                    keywords_searched.extend(sq.keywords)

                search_results, rerank_details = await self.searcher.search(plan)
                result.total_sources += len(search_results)
                logger.info(
                    "ResearchRunner[{}]: iteration {} — found {} results (total {})",
                    rid, iteration, len(search_results), result.total_sources,
                )

                # 构建本轮���代日志
                iteration_log = SearchIterationLog(
                    iteration=iteration,
                    sub_questions_searched=keywords_searched,
                    search_results_count=len(search_results),
                    rerank_enabled=self.config.rerank_enabled,
                    rerank_details=rerank_details,
                )

                result.status = ResearchStatus.SYNTHESIZING
                logger.info("ResearchRunner[{}]: iteration {} — synthesizing", rid, iteration)
                synthesis = await self.synthesizer.synthesize(search_results, plan)

                # 记录本轮覆盖度
                iteration_log.coverage_score = synthesis.coverage_score

                # Check if we should continue iterating
                if not self.refiner.should_continue(synthesis, iteration, adjusted_config):
                    logger.info(
                        "ResearchRunner[{}]: stopping after {} iterations, coverage={:.2f}",
                        rid, iteration + 1, synthesis.coverage_score,
                    )
                    iteration_log.stopped = True
                    iteration_log.stop_reason = "coverage_threshold"
                    execution_log.iterations.append(iteration_log)
                    execution_log.final_coverage_score = synthesis.coverage_score
                    execution_log.stop_reason = "coverage_threshold"
                    break

                iteration_log.stopped = False
                execution_log.iterations.append(iteration_log)

                result.status = ResearchStatus.ITERATING
                logger.info("ResearchRunner[{}]: iteration {} — refining", rid, iteration)
                updated_plan = await self.refiner.refine(plan, synthesis, adjusted_config)

                if updated_plan is None:
                    logger.info("ResearchRunner[{}]: refiner returned None, stopping", rid)
                    iteration_log.stopped = True
                    iteration_log.stop_reason = "no_gaps"
                    execution_log.stop_reason = "no_gaps"
                    break

                plan = updated_plan
                plan.iteration = iteration + 1
                result.iterations = iteration + 1

            # 如果因为达到 max_iterations 而非 coverage 停止
            if not execution_log.stop_reason:
                execution_log.final_coverage_score = synthesis.coverage_score if synthesis else 0.0
                execution_log.stop_reason = "max_iterations"

            # Phase 5: Generate report
            result.status = ResearchStatus.COMPLETED
            if synthesis is None:
                synthesis = await self.synthesizer.synthesize([], plan)

            result.synthesis = synthesis
            result.report = await self.reporter.generate(topic, synthesis, plan)

            # Phase 6: Self-evaluation
            if self.config.enable_self_evaluation:
                metrics = await self.reporter.self_evaluate(result.report, synthesis, plan)
                result.metrics = metrics
                result.quality_score = metrics.overall

                # Retry if quality is below threshold (one time only)
                if metrics.overall < self.config.evaluation_threshold:
                    logger.warning(
                        "ResearchRunner[{}]: quality {:.1f} < threshold {:.1f}, retrying report",
                        rid, metrics.overall, self.config.evaluation_threshold,
                    )
                    result.report = await self.reporter.generate(topic, synthesis, plan)
                    metrics = await self.reporter.self_evaluate(result.report, synthesis, plan)
                    result.metrics = metrics
                    result.quality_score = metrics.overall

            result.completed_at = datetime.now()
            logger.info(
                "ResearchRunner[{}]: completed in {} iterations, {} sources, quality={:.1f}",
                rid, result.iterations, result.total_sources, result.quality_score,
            )

            # Phase 7: Process result into knowledge base
            if self.knowledge_processor:
                try:
                    knowledge_result = await self.knowledge_processor.process(result)
                    logger.info(
                        "ResearchRunner[{}]: knowledge processed - {} claims, {} insights",
                        rid, knowledge_result.claims_written, knowledge_result.insights_written,
                    )
                    # Store knowledge result in the research result for visibility
                    result.knowledge_result = knowledge_result
                    # 记录到执行日志
                    execution_log.knowledge_write = {
                        "claims": knowledge_result.claims_written,
                        "insights": knowledge_result.insights_written,
                        "duplicates": knowledge_result.duplicates_skipped,
                        "conflicts": knowledge_result.conflicts_detected,
                    }
                except Exception as e:
                    # Re-raise so the failure is visible to the caller
                    logger.error(
                        "ResearchRunner[{}]: knowledge processing failed: {}",
                        rid, e,
                    )
                    raise

            # 绑定执行日志到结果（白盒化）
            result.execution_log = execution_log

            return result

        except Exception:
            result.status = ResearchStatus.FAILED
            logger.exception("ResearchRunner[{}]: pipeline failed", rid)
            raise

    def get_result(self, research_id: str) -> ResearchResult | None:
        """Retrieve a cached research result by ID."""
        return self._results.get(research_id)

    def list_results(self) -> list[ResearchResult]:
        """List all cached results."""
        return list(self._results.values())

    # ============== Knowledge Loop Methods ==============

    async def _get_existing_knowledge(
        self, topic: str, token_budget: int = 1500
    ) -> str:
        """Pre-query existing knowledge for a topic with token budget management.

        Searches both claims and insights, formats them for the planner,
        and truncates to fit within the token budget.

        Args:
            topic: The research topic.
            token_budget: Maximum tokens to use for knowledge context (default 1500).

        Returns:
            Formatted context string with existing knowledge, or empty string.
        """
        if not self.knowledge_search:
            return ""

        try:
            claims, insights = await self.knowledge_search.search_all(topic)

            context = ""
            tokens_used = 0

            # Helper function to estimate tokens (rough: chars / 2)
            def estimate_tokens(text: str) -> int:
                return len(text) // 2

            # Helper function to check budget
            def can_add(text: str) -> bool:
                return tokens_used + estimate_tokens(text) < token_budget

            # Add insights first (higher priority for guiding research)
            # Sort by confidence, confirmed insights first
            if insights:
                # Sort: confirmed first (maturity="confirmed"), then by confidence
                sorted_insights = sorted(
                    insights,
                    key=lambda x: (
                        0 if x.get("metadata", {}).get("maturity") == "confirmed" else 1,
                        -x.get("metadata", {}).get("confidence", 0),
                    )
                )

                context += "## 已有相关规律 (Insights)\n"
                context += "以下是从过往研究中提炼的跨域规律：\n"

                for i in sorted_insights:
                    text = i.get("text", i.get("metadata", {}).get("text", ""))
                    if not text:
                        continue

                    maturity = i.get("metadata", {}).get("maturity", "candidate")
                    maturity_tag = "✓" if maturity == "confirmed" else "?"
                    confidence = i.get("metadata", {}).get("confidence", 0.5)
                    line = f"- [{maturity_tag}] {text} (置信度: {confidence:.0%})\n"

                    if can_add(line):
                        context += line
                        tokens_used += estimate_tokens(line)
                    else:
                        # Budget exhausted
                        context += "- [更多规律...] (预算限制)\n"
                        break

                context += "\n"

            # Add claims
            # Sort by confidence (high confidence first)
            if claims:
                sorted_claims = sorted(
                    claims,
                    key=lambda x: -x.get("metadata", {}).get("confidence", 0),
                )

                context += "## 已知相关事实 (Claims)\n"
                context += "以下是过往研究中的具体发现：\n"

                for c in sorted_claims:
                    text = c.get("text", c.get("metadata", {}).get("text", ""))
                    if not text:
                        continue

                    confidence = c.get("metadata", {}).get("confidence", 0.5)
                    line = f"- {text} (置信度: {confidence:.0%})\n"

                    if can_add(line):
                        context += line
                        tokens_used += estimate_tokens(line)
                    else:
                        # Budget exhausted
                        context += "- [更多事实...] (预算限制)\n"
                        break

            # Log token usage
            logger.info(
                f"ResearchRunner: knowledge context uses ~{tokens_used * 2} chars, "
                f"~{tokens_used} tokens (budget: {token_budget})"
            )

            return context

        except Exception as e:
            logger.warning(f"ResearchRunner: failed to get existing knowledge: {e}")
            return ""

    async def _get_document_context(self, topic: str, top_k: int = 10) -> str:
        """Query user-uploaded documents from document_store (default collection).

        Args:
            topic: The research topic.
            top_k: Number of chunks to retrieve.

        Returns:
            Formatted context string with relevant document chunks, or empty string.
        """
        if not self.rag_store:
            return ""

        try:
            # Get embedding from knowledge_search
            if not self.knowledge_search:
                return ""
            vector = self.knowledge_search._embed(topic)

            # Query default collection (user uploaded documents)
            results = self.rag_store.query(
                vector=vector,
                top_k=top_k,
            )

            if not results:
                return ""

            # Format results
            chunks = []
            for r in results:
                text = r.get("text", r.get("metadata", {}).get("text", ""))
                if text:
                    chunks.append(text)

            if not chunks:
                return ""

            return "\n\n---\n\n".join(chunks)

        except Exception as e:
            logger.warning(f"ResearchRunner: failed to query document_store: {e}")
            return ""

