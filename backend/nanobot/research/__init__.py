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

Knowledge Search:
    from nanobot.research import KnowledgeSearch
    knowledge_search = KnowledgeSearch.from_settings(settings)
"""

from nanobot.research.runner import ResearchRunner
from nanobot.research.types import (
    Contradiction,
    DepthLevel,
    Finding,
    KnowledgeGap,
    ReportMetrics,
    ResearchConfig,
    ResearchPlan,
    ResearchResult,
    ResearchStatus,
    SearchResult,
    SubQuestion,
    SynthesisResult,
)

from nanobot.research.knowledge_search import KnowledgeSearch

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
    "KnowledgeSearch",
]
