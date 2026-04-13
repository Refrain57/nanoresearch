"""End-to-end tests for Research Agent.

Tests the complete research pipeline:
- Topic decomposition
- Multi-source search
- Report generation
- Quality evaluation via LLM-as-Judge
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.agent.evaluation.evaluator import (
    EvaluationResult,
    LLMasJudge,
    MetricResult,
)
from tests.agent.evaluation.test_datasets import (
    RESEARCH_TEST_CASES,
    ResearchTestCase,
)


# ============================================================================
# Mock Components
# ============================================================================

class MockSearchResult:
    """Mock search result for testing."""
    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content
        self.source_type = "web"
        self.credibility_score = 0.8
        self.relevance_score = 0.9
        self.recency_score = 0.7
        self.final_score = 0.8


class MockWebSearchTool:
    """Mock web search tool for testing."""

    async def execute(self, query: str, **kwargs) -> str:
        """Return mock search results."""
        return f"""1. {query} - 综合介绍
   https://example.com/intro
   这是一篇关于{query}的介绍文章，涵盖基础概念和应用场景。

2. {query} - 技术原理
   https://example.com/tech
   深入讲解{query}的技术原理和实现细节。

3. {query} - 最佳实践
   https://example.com/best-practices
   总结{query}的最佳实践和常见问题。"""


class MockWebFetchTool:
    """Mock web fetch tool for testing."""

    async def execute(self, url: str, **kwargs) -> str:
        """Return mock fetched content."""
        return json.dumps({
            "text": f"这是从 {url} 抓取的详细内容。包含技术细节、应用案例和分析讨论。"
        })


class MockLLMProvider:
    """Mock LLM provider with realistic responses."""

    def __init__(self, response_type: str = "research"):
        self.response_type = response_type
        self.call_count = 0

    async def chat(self, messages: list[dict], **kwargs) -> Any:
        """Return mock LLM response."""
        self.call_count += 1

        # Simulate research planning
        if "research_plan" in str(messages):
            return self._plan_response(messages)

        # Simulate synthesis
        if "synthesize" in str(messages):
            return self._synthesize_response(messages)

        # Simulate report generation
        if "report" in str(messages).lower() or "markdown" in str(messages).lower():
            return self._report_response(messages)

        # Simulate evaluation
        if "评估" in str(messages) or "evaluate" in str(messages).lower():
            return self._eval_response(messages)

        return MagicMock(content="Default response")

    def _plan_response(self, messages: list) -> Any:
        """Generate mock research plan."""
        return MagicMock(
            content="",
            has_tool_calls=True,
            tool_calls=[
                MagicMock(
                    id="plan_1",
                    name="research_plan",
                    arguments={
                        "sub_questions": [
                            {"question": "核心原理是什么", "keywords_zh": ["原理"], "keywords_en": ["principle"], "priority": 1},
                            {"question": "主要应用场景", "keywords_zh": ["应用"], "keywords_en": ["application"], "priority": 2},
                            {"question": "优缺点分析", "keywords_zh": ["优缺点"], "keywords_en": ["pros cons"], "priority": 3},
                        ]
                    }
                )
            ]
        )

    def _synthesize_response(self, messages: list) -> Any:
        """Generate mock synthesis."""
        return MagicMock(
            content="",
            has_tool_calls=True,
            tool_calls=[
                MagicMock(
                    id="synth_1",
                    name="synthesize",
                    arguments={
                        "findings": [
                            {"statement": "核心技术基于深度学习", "source_urls": ["https://example.com/1"]},
                            {"statement": "主要应用于图像处理", "source_urls": ["https://example.com/2"]},
                        ],
                        "contradictions": [],
                        "knowledge_gaps": [],
                        "coverage_assessment": [{"sub_question_id": 1, "coverage": "sufficient"}],
                    }
                )
            ]
        )

    def _report_response(self, messages: list) -> Any:
        """Generate mock report."""
        report = """# 研究报告

## 概述
本报告总结了相关技术的核心原理、应用场景和发展趋势。

## 核心原理
基于深度学习的技术框架，通过神经网络实现特征提取和模式识别。

## 应用场景
- 图像处理
- 自然语言处理
- 推荐系统

## 总结
该技术具有广泛的应用前景，但仍存在一些挑战需要解决。
"""
        return MagicMock(content=report, has_tool_calls=False)

    def _eval_response(self, messages: list) -> Any:
        """Generate mock evaluation response."""
        return MagicMock(
            content=json.dumps({
                "completeness": 8,
                "accuracy": 7,
                "readability": 8,
                "depth": 7,
                "overall": 7.5,
                "strengths": ["结构清晰", "覆盖全面"],
                "weaknesses": ["部分细节不够深入"],
                "suggestions": ["增加更多案例分析"]
            }),
            has_tool_calls=False
        )


# ============================================================================
# Tests
# ============================================================================

class TestResearchAgentPlanning:
    """Tests for research planning phase."""

    @pytest.mark.asyncio
    async def test_topic_decomposition(self):
        """Test that research topic is properly decomposed into sub-questions."""
        from nanobot.research.planner import ResearchPlanner
        from nanobot.research.types import ResearchConfig

        provider = MockLLMProvider()
        planner = ResearchPlanner(provider, "test-model")

        result = await planner.plan("RAG 技术原理与应用", depth="normal")

        # Verify plan structure
        assert result.topic == "RAG 技术原理与应用"
        assert len(result.sub_questions) >= 3
        assert len(result.sub_questions) <= 6

        # Verify sub-question quality
        for sq in result.sub_questions:
            assert sq.question, "Sub-question should not be empty"
            assert sq.priority >= 1 and sq.priority <= 5
            assert len(sq.keywords) > 0, "Should have keywords"

    @pytest.mark.asyncio
    async def test_planning_fallback(self):
        """Test fallback when LLM doesn't return proper tool call."""
        from nanobot.research.planner import ResearchPlanner

        # Provider that returns no tool calls
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(return_value=MagicMock(
            has_tool_calls=False,
            content="1. 核心原理\n2. 应用场景\n3. 发展趋势"
        ))

        planner = ResearchPlanner(provider, "test-model")
        result = await planner.plan("测试主题")

        assert result.topic == "测试主题"
        assert len(result.sub_questions) >= 2  # Fallback should generate at least 2


class TestResearchAgentSearch:
    """Tests for search orchestration."""

    @pytest.mark.asyncio
    async def test_parallel_search(self):
        """Test that searches run in parallel for multiple sub-questions."""
        from nanobot.research.searcher import SearchOrchestrator
        from nanobot.research.types import ResearchPlan, SubQuestion, ResearchConfig

        mock_search = MockWebSearchTool()
        mock_fetch = MockWebFetchTool()
        config = ResearchConfig()

        orchestrator = SearchOrchestrator(
            web_search_tool=mock_search,
            web_fetch_tool=mock_fetch,
            config=config,
            search_count=5,
        )

        plan = ResearchPlan(
            topic="测试主题",
            sub_questions=[
                SubQuestion(id=1, question="问题1", keywords=["关键词1"]),
                SubQuestion(id=2, question="问题2", keywords=["关键词2"]),
            ]
        )

        results = await orchestrator.search(plan)

        assert len(results) > 0
        # Each result should have required fields
        for r in results:
            assert r.url
            assert r.title
            assert r.content

    @pytest.mark.asyncio
    async def test_search_deduplication(self):
        """Test that duplicate URLs are removed."""
        from nanobot.research.searcher import SearchOrchestrator
        from nanobot.research.types import ResearchPlan, SubQuestion, ResearchConfig

        mock_search = MockWebSearchTool()
        mock_fetch = MockWebFetchTool()
        config = ResearchConfig()

        orchestrator = SearchOrchestrator(
            web_search_tool=mock_search,
            web_fetch_tool=mock_fetch,
            config=config,
        )

        # Create plan with overlapping keywords (should produce duplicates)
        plan = ResearchPlan(
            topic="测试主题",
            sub_questions=[
                SubQuestion(id=1, question="问题", keywords=["相同关键词"]),
                SubQuestion(id=2, question="问题", keywords=["相同关键词"]),
            ]
        )

        results = await orchestrator.search(plan)

        # Check no duplicate URLs
        urls = [r.url for r in results]
        assert len(urls) == len(set(urls)), "Should not have duplicate URLs"


class TestResearchAgentSynthesis:
    """Tests for information synthesis."""

    @pytest.mark.asyncio
    async def test_synthesis_structure(self):
        """Test that synthesis produces structured output."""
        from nanobot.research.synthesizer import InformationSynthesizer
        from nanobot.research.types import ResearchPlan, SubQuestion, SearchResult

        provider = MockLLMProvider()
        synthesizer = InformationSynthesizer(provider, "test-model")

        plan = ResearchPlan(
            topic="测试主题",
            sub_questions=[SubQuestion(id=1, question="问题1", keywords=["kw"])]
        )

        results = [
            SearchResult(
                url="https://example.com/1",
                title="文章1",
                content="这是测试内容",
                credibility_score=0.8,
                relevance_score=0.9,
                recency_score=0.7,
            )
        ]

        synthesis = await synthesizer.synthesize(results, plan)

        assert synthesis.findings, "Should have findings"
        assert synthesis.coverage_score >= 0


class TestResearchAgentEndToEnd:
    """End-to-end tests for the complete research pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests"
    )
    async def test_full_research_pipeline(self):
        """Test complete research pipeline with real LLM."""
        from nanobot.research.runner import ResearchRunner
        from nanobot.research.types import ResearchConfig

        # This test requires actual LLM API access
        # Skip in CI, run locally with API keys

        # Use real provider if available
        from nanobot.providers.base import get_provider

        provider = get_provider()  # Will use configured provider
        runner = ResearchRunner(
            provider=provider,
            model=provider.get_default_model(),
            web_search_tool=MockWebSearchTool(),
            web_fetch_tool=MockWebFetchTool(),
            config=ResearchConfig(max_iterations=1),
        )

        result = await runner.run("Python 异步编程最佳实践", depth="quick")

        assert result.status.value == "completed"
        assert result.report
        assert len(result.report) > 100

    @pytest.mark.asyncio
    async def test_research_evaluation_with_judge(self):
        """Test research report evaluation using LLM-as-Judge."""
        # Create mock judge provider
        judge_provider = MagicMock()
        judge_provider.chat = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "completeness": 8,
                "accuracy": 7,
                "readability": 8,
                "depth": 7,
                "overall": 7.5,
                "strengths": ["结构清晰"],
                "weaknesses": ["细节不足"],
                "suggestions": ["增加案例"]
            })
        ))

        judge = LLMasJudge(judge_provider, "gpt-4o")

        result = await judge.evaluate_research_report(
            report="# 测试报告\n\n这是内容...",
            research_topic="测试主题",
            sub_questions=["问题1", "问题2"],
            sources=["https://example.com"],
        )

        assert result.test_name
        assert len(result.metrics) == 5
        assert result.passed  # Overall 7.5 >= 6.0


class TestResearchAgentQuality:
    """Quality tests using benchmark datasets."""

    @pytest.mark.asyncio
    async def test_research_quality_benchmark(self):
        """Run research quality benchmark."""
        from nanobot.research.planner import ResearchPlanner

        results = []
        provider = MockLLMProvider()
        planner = ResearchPlanner(provider, "test-model")

        for test_case in RESEARCH_TEST_CASES[:3]:  # Test first 3
            plan = await planner.plan(test_case.topic, depth=test_case.depth)

            # Evaluate sub-question count
            sq_count = len(plan.sub_questions)
            count_ok = test_case.min_sub_questions <= sq_count <= test_case.max_sub_questions

            results.append({
                "topic": test_case.topic,
                "sub_questions_count": sq_count,
                "count_ok": count_ok,
            })

        # Aggregate results
        success_rate = sum(1 for r in results if r["count_ok"]) / len(results)
        assert success_rate >= 0.7, f"Sub-question count success rate {success_rate:.1%} < 70%"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
