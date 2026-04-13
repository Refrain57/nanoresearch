"""Research runner — orchestrates the full research pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.research.types import (
    ResearchConfig,
    ResearchPlan,
    ResearchResult,
    ResearchStatus,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

# Import sub-components lazily to avoid circular imports
_PLANNER: type | None = None
_SEARCHER: type | None = None
_SYNTHESIZER: type | None = None
_REFINER: type | None = None
_REPORTER: type | None = None


def _lazy_imports() -> None:
    global _PLANNER, _SEARCHER, _SYNTHESIZER, _REFINER, _REPORTER
    if _PLANNER is None:
        from nanobot.research.planner import ResearchPlanner
        from nanobot.research.searcher import SearchOrchestrator
        from nanobot.research.synthesizer import InformationSynthesizer
        from nanobot.research.refiner import ResearchRefiner
        from nanobot.research.reporter import ReportGenerator

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
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        web_search_tool: Any,
        web_fetch_tool: Any,
        config: ResearchConfig | None = None,
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

        logger.info(
            "ResearchRunner[{}]: starting topic='{}' depth='{}' max_iter={} max_sources={}",
            rid, topic, depth, max_iterations, max_sources,
        )

        try:
            # Phase 1: Planning
            plan = await self.planner.plan(topic, depth)
            plan.iteration = 0
            result.plan = plan
            logger.info("ResearchRunner[{}]: plan created with {} sub-questions", rid, len(plan.sub_questions))

            synthesis = None

            # Phase 2-4: Search + Synthesize + Iterate
            for iteration in range(max_iterations):
                result.status = ResearchStatus.SEARCHING
                logger.info("ResearchRunner[{}]: iteration {} — searching", rid, iteration)

                search_results = await self.searcher.search(plan)
                result.total_sources += len(search_results)
                logger.info(
                    "ResearchRunner[{}]: iteration {} — found {} results (total {})",
                    rid, iteration, len(search_results), result.total_sources,
                )

                result.status = ResearchStatus.SYNTHESIZING
                logger.info("ResearchRunner[{}]: iteration {} — synthesizing", rid, iteration)
                synthesis = await self.synthesizer.synthesize(search_results, plan)

                # Check if we should continue iterating
                if not self.refiner.should_continue(synthesis, iteration, adjusted_config):
                    logger.info(
                        "ResearchRunner[{}]: stopping after {} iterations, coverage={:.2f}",
                        rid, iteration + 1, synthesis.coverage_score,
                    )
                    break

                result.status = ResearchStatus.ITERATING
                logger.info("ResearchRunner[{}]: iteration {} — refining", rid, iteration)
                updated_plan = await self.refiner.refine(plan, synthesis, adjusted_config)

                if updated_plan is None:
                    logger.info("ResearchRunner[{}]: refiner returned None, stopping", rid)
                    break

                plan = updated_plan
                plan.iteration = iteration + 1
                result.iterations = iteration + 1

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
