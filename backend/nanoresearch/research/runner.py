"""Research runner — orchestrates the full research pipeline."""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from nanoresearch.research.types import (
    ExecutionLog,
    ResearchConfig,
    ResearchPlan,
    ResearchResult,
    ResearchStatus,
    SearchIterationLog,
)

if TYPE_CHECKING:
    from nanoresearch.providers.base import LLMProvider
    from nanoresearch.research.knowledge_search import KnowledgeSearch


# Import sub-components lazily to avoid circular imports
_PLANNER: type | None = None
_SEARCHER: type | None = None
_SYNTHESIZER: type | None = None
_REFINER: type | None = None
_REPORTER: type | None = None


def _lazy_imports() -> None:
    global _PLANNER, _SEARCHER, _SYNTHESIZER, _REFINER, _REPORTER
    if _PLANNER is None:
        from nanoresearch.research.planner import ResearchPlanner
        from nanoresearch.research.searcher import SearchOrchestrator
        from nanoresearch.research.synthesizer import InformationSynthesizer
        from nanoresearch.research.refiner import ResearchRefiner
        from nanoresearch.research.reporter import ReportGenerator

        _PLANNER = ResearchPlanner
        _SEARCHER = SearchOrchestrator
        _SYNTHESIZER = InformationSynthesizer
        _REFINER = ResearchRefiner
        _REPORTER = ReportGenerator


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
        rag_store: Any = None,  # ChromaStore for user uploaded documents
        *,
        settings: Any = None,  # RAG Settings for auto-ingest + HybridSearch
        uid: str | None = None,
        workspace: Path | None = None,
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

        self.knowledge_search = knowledge_search
        self.rag_store = rag_store
        self.settings = settings
        self.uid = uid
        self.workspace = workspace
        self._hs: Any = None  # lazy HybridSearch

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
            coverage_decline_threshold=self.config.coverage_decline_threshold,
            max_new_sub_questions_per_iteration=self.config.max_new_sub_questions_per_iteration,
            accumulated_results_cap=self.config.accumulated_results_cap,
            rerank_enabled=self.config.rerank_enabled,
            rerank_provider=self.config.rerank_provider,
            rerank_model=self.config.rerank_model,
            rerank_top_k=self.config.rerank_top_k,
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
            # Reset per-run searcher state so cross-iteration dedup doesn't bleed across topics
            self.searcher._global_seen_urls = set()
            self.searcher._failed_domains = {}

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
            prev_coverage: float | None = None
            accumulated_results: list = []  # cross-iteration result pool (Bug 2A)

            # Phase 2-4: Search + Synthesize + Iterate
            for iteration in range(max_iterations):
                result.status = ResearchStatus.SEARCHING
                logger.info("ResearchRunner[{}]: iteration {} — searching", rid, iteration)

                # 收集本轮搜索的关键词（仅 pending 子问题，与 searcher 的过滤对齐）
                keywords_searched = []
                for sq in plan.sub_questions:
                    if sq.status == "pending":
                        keywords_searched.extend(sq.keywords)

                search_results, rerank_details = await self.searcher.search(plan)
                result.total_sources += len(search_results)
                logger.info(
                    "ResearchRunner[{}]: iteration {} — found {} new results (total {})",
                    rid, iteration, len(search_results), result.total_sources,
                )

                # Accumulate results across iterations; keep top-N by final_score (Bug 2A)
                accumulated_results.extend(search_results)
                if len(accumulated_results) > adjusted_config.accumulated_results_cap:
                    accumulated_results = sorted(
                        accumulated_results, key=lambda r: r.final_score, reverse=True
                    )[:adjusted_config.accumulated_results_cap]

                # 构建本轮迭代日志
                iteration_log = SearchIterationLog(
                    iteration=iteration,
                    sub_questions_searched=keywords_searched,
                    search_results_count=len(search_results),
                    rerank_enabled=self.config.rerank_enabled,
                    rerank_details=rerank_details,
                )

                result.status = ResearchStatus.SYNTHESIZING
                logger.info("ResearchRunner[{}]: iteration {} — synthesizing {} accumulated results", rid, iteration, len(accumulated_results))
                synthesis = await self.synthesizer.synthesize(accumulated_results, plan)

                # 记录本轮覆盖度
                iteration_log.coverage_score = synthesis.coverage_score

                # Check if we should continue iterating (pass prev_coverage for trend detection)
                if not self.refiner.should_continue(synthesis, iteration, adjusted_config, prev_coverage=prev_coverage):
                    stop_reason = "coverage_threshold" if synthesis.coverage_score >= adjusted_config.min_coverage_threshold else "coverage_declining"
                    logger.info(
                        "ResearchRunner[{}]: stopping after {} iterations, coverage={:.2f}",
                        rid, iteration + 1, synthesis.coverage_score,
                    )
                    iteration_log.stopped = True
                    iteration_log.stop_reason = stop_reason
                    execution_log.iterations.append(iteration_log)
                    execution_log.final_coverage_score = synthesis.coverage_score
                    execution_log.stop_reason = stop_reason
                    break

                prev_coverage = synthesis.coverage_score
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
                    execution_log.final_coverage_score = synthesis.coverage_score
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
                synthesis = await self.synthesizer.synthesize(accumulated_results or [], plan)

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

            # Phase 7: Save report as MD file (auto-ingested into RAG pipeline)
            if result.report:
                try:
                    file_path = self._save_report_md(rid, topic, result.report, result.quality_score)
                    result.report_file_path = str(file_path)
                    logger.info("ResearchRunner[{}]: report saved to {}", rid, file_path)
                    # Phase 7b: Auto-ingest into RAG (fire-and-forget)
                    asyncio.create_task(self._auto_ingest(str(file_path)))
                except Exception as e:
                    logger.warning("ResearchRunner[{}]: failed to save report MD: {}", rid, e)

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

    @property
    def _hybrid_search(self) -> Any | None:
        """Lazy-initialized HybridSearch from settings."""
        if self._hs is None and self.settings is not None:
            try:
                from nanoresearch.rag.core.query_engine.hybrid_search import create_hybrid_search
                self._hs = create_hybrid_search(self.settings)
            except Exception as e:
                logger.warning("Failed to create HybridSearch: {}", e)
        return self._hs

    async def _get_existing_knowledge(self, topic: str, token_budget: int = 1500) -> str:
        """Retrieve relevant existing research reports from RAG."""
        hs = self._hybrid_search
        if not hs:
            return ""

        try:
            filters: dict = {"source": "research"}
            if self.uid:
                filters["uid"] = self.uid
            results = await hs.async_search(
                query=topic,
                top_k=5,
                filters=filters,
            )
            if not results:
                return ""

            # Group by document for readable formatting
            doc_groups: dict[str, list[tuple[float, str]]] = {}
            for r in results:
                src = r.metadata.get("source_path", "unknown")
                doc_groups.setdefault(src, []).append((r.score, r.text))

            parts: list[str] = []
            token_used = 0
            for src, snippets in doc_groups.items():
                head = f"### 来自 {Path(src).name}\n"
                body_parts: list[str] = []
                for score, text in snippets:
                    line = f"[relevance={score:.2f}] {text.strip()}"
                    estimated = len(line) // 4
                    if token_used + estimated > token_budget:
                        break
                    body_parts.append(line)
                    token_used += estimated

                if body_parts:
                    parts.append(head + "\n" + "\n".join(body_parts) + "\n")

            return "\n".join(parts) if parts else ""

        except Exception as e:
            logger.warning("_get_existing_knowledge failed: {}", e)
            return ""

    def _save_report_md(
        self, rid: str, topic: str, report: str, quality_score: float
    ) -> Path:
        """Save the research report as an MD file for optional user-confirmed ingest.

        The file includes YAML front-matter with source=research so the ingestion
        pipeline can tag it correctly in the KB document list.
        """
        notes_dir = (
            self.workspace / "research_notes"
            if self.workspace
            else Path("~/.nanoresearch/research_notes").expanduser()
        )
        notes_dir.mkdir(parents=True, exist_ok=True)

        slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic)[:40]
        filename = f"{rid}_{slug}.md"
        file_path = notes_dir / filename

        uid_line = f"uid: {self.uid}\n" if self.uid else ""
        content = f"""---
title: {topic}
research_id: {rid}
source: research
{uid_line}quality_score: {quality_score:.1f}
created_at: {datetime.now().isoformat()}
---

# {topic}

{report}
"""
        file_path.write_text(content, encoding="utf-8")
        return file_path

    async def _auto_ingest(self, file_path: str) -> None:
        """Background-ingest the research report MD into the RAG pipeline."""
        if not self.settings:
            return
        try:
            from nanoresearch.rag.ingestion.pipeline import IngestionPipeline

            loop = asyncio.get_running_loop()
            collection = self.settings.vector_store.collection_name
            pipeline = IngestionPipeline(
                self.settings,
                collection=collection,
                force=False,
            )
            result = await loop.run_in_executor(
                None, lambda: pipeline.run(file_path)
            )
            if result.success:
                logger.info("Auto-ingest OK: {} -> {} chunks", file_path, result.chunk_count)
            else:
                logger.warning("Auto-ingest failed: {}", result.error)
        except Exception as e:
            logger.warning("Auto-ingest error: {}", e)

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
            vector = self.knowledge_search.dense_encoder.embed([topic])[0]

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

