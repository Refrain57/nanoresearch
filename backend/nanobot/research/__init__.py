"""Auto Research Agent - 自主研究模块。

Usage:
    from nanobot.research import ResearchRunner, ResearchConfig, ResearchResult

    config = ResearchConfig(max_iterations=3, default_depth="normal")
    runner = ResearchRunner(
        provider=provider,
        model="anthropic/claude-opus-4-5",
        web_search_tool=web_search,
        web_fetch_tool=web_fetch,
        config=config,
    )
    result = await runner.run("LLM在医疗领域的应用", depth="normal")
    print(result.report)
    print(result.metrics.overall)

Knowledge Loop:
    from nanobot.research import KnowledgeSearch, KnowledgeProcessor, InsightTracker

    knowledge_search = KnowledgeSearch.from_settings(settings)
    tracker = InsightTracker()
    processor = KnowledgeProcessor(provider, model, knowledge_search, tracker)

    runner = ResearchRunner(
        provider=provider,
        model=model,
        web_search_tool=web_search,
        web_fetch_tool=web_fetch,
        knowledge_search=knowledge_search,
        knowledge_processor=processor,
    )
"""

from nanobot.research.runner import ResearchRunner
from nanobot.research.types import (
    BatchRefineResult,
    Claim,
    Contradiction,
    DepthLevel,
    Finding,
    Insight,
    KnowledgeGap,
    KnowledgeProcessResult,
    ReportMetrics,
    ResearchConfig,
    ResearchPlan,
    ResearchResult,
    ResearchStatus,
    SearchResult,
    SubQuestion,
    SynthesisResult,
)

# Optional imports for knowledge loop
from nanobot.research.knowledge_search import KnowledgeSearch
from nanobot.research.knowledge_processor import KnowledgeProcessor
from nanobot.research.insight_tracker import InsightTracker

__all__ = [
    "ResearchRunner",
    "ResearchConfig",
    "ResearchResult",
    "ResearchPlan",
    "ResearchStatus",
    "SubQuestion",
    "SearchResult",
    "SynthesisResult",
    "Contradiction",
    "Finding",
    "KnowledgeGap",
    "ReportMetrics",
    "DepthLevel",
    # Knowledge loop types
    "Claim",
    "Insight",
    "KnowledgeProcessResult",
    "BatchRefineResult",
    # Knowledge loop components
    "KnowledgeSearch",
    "KnowledgeProcessor",
    "InsightTracker",
]
