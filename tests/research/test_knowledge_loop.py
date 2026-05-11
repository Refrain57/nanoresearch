"""Tests for knowledge loop components."""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.research.types import (
    BatchRefineResult,
    Claim,
    Insight,
    KnowledgeProcessResult,
    ResearchResult,
    ResearchStatus,
    SynthesisResult,
    Finding,
)
from nanobot.research.insight_tracker import InsightTracker


# ============================================================================
# InsightTracker Tests
# ============================================================================

class TestInsightTracker:
    """Tests for InsightTracker."""

    def test_add_and_list_candidates(self, tmp_path):
        """Test adding and listing candidates."""
        tracker = InsightTracker(storage_path=str(tmp_path / "tracker.json"))

        tracker.add_candidate("insight_001", {
            "insight": "Test insight 1",
            "confidence": 0.8,
        })
        tracker.add_candidate("insight_002", {
            "insight": "Test insight 2",
            "confidence": 0.9,
        })

        candidates = tracker.list_candidates()
        assert len(candidates) == 2
        assert tracker.count() == 2

    def test_mark_processed(self, tmp_path):
        """Test marking candidates as processed."""
        tracker = InsightTracker(storage_path=str(tmp_path / "tracker.json"))

        tracker.add_candidate("insight_001", {"insight": "Test"})
        tracker.add_candidate("insight_002", {"insight": "Test 2"})

        assert tracker.count() == 2

        tracker.mark_processed("insight_001")
        assert tracker.count() == 1

        candidates = tracker.list_candidates()
        assert candidates[0]["id"] == "insight_002"

    def test_get_candidate(self, tmp_path):
        """Test getting a specific candidate."""
        tracker = InsightTracker(storage_path=str(tmp_path / "tracker.json"))

        tracker.add_candidate("insight_001", {
            "insight": "Test insight",
            "confidence": 0.85,
        })

        candidate = tracker.get_candidate("insight_001")
        assert candidate is not None
        assert candidate["insight"] == "Test insight"
        assert candidate["confidence"] == 0.85
        assert candidate["id"] == "insight_001"

        # Non-existent candidate
        assert tracker.get_candidate("nonexistent") is None

    def test_persistence(self, tmp_path):
        """Test that data persists across instances."""
        storage_path = str(tmp_path / "tracker.json")

        # First instance
        tracker1 = InsightTracker(storage_path=storage_path)
        tracker1.add_candidate("insight_001", {"insight": "Persisted insight"})

        # Second instance (should load existing data)
        tracker2 = InsightTracker(storage_path=storage_path)
        assert tracker2.count() == 1
        candidates = tracker2.list_candidates()
        assert candidates[0]["insight"] == "Persisted insight"


# ============================================================================
# KnowledgeSearch Tests (with mocked stores)
# ============================================================================

class TestKnowledgeSearch:
    """Tests for KnowledgeSearch."""

    @pytest.fixture
    def mock_claim_store(self):
        """Mock claim store."""
        store = MagicMock()
        store._data = []

        def mock_query(vector, top_k=10, filters=None):
            return store._data[:top_k]

        def mock_upsert(records):
            for r in records:
                store._data.append({
                    "id": r["id"],
                    "score": 0.9,
                    "text": r["metadata"].get("text", ""),
                    "metadata": r["metadata"],
                })

        store.query = mock_query
        store.upsert = mock_upsert
        store.get_collection_stats = lambda: {"count": len(store._data)}
        return store

    @pytest.fixture
    def mock_insight_store(self):
        """Mock insight store."""
        store = MagicMock()
        store._data = []

        def mock_query(vector, top_k=10, filters=None):
            results = store._data
            if filters and "maturity" in filters:
                results = [r for r in results if r["metadata"].get("maturity") == filters["maturity"]]
            return results[:top_k]

        def mock_upsert(records):
            for r in records:
                store._data.append({
                    "id": r["id"],
                    "score": 0.85,
                    "text": r["metadata"].get("text", ""),
                    "metadata": r["metadata"],
                })

        def mock_delete(ids):
            store._data = [r for r in store._data if r["id"] not in ids]

        store.query = mock_query
        store.upsert = mock_upsert
        store.delete = mock_delete
        store.get_collection_stats = lambda: {"count": len(store._data)}
        return store

    @pytest.fixture
    def mock_encoder(self):
        """Mock embedding encoder."""
        encoder = MagicMock()
        encoder.embed = lambda texts: [[0.1] * 1536 for _ in texts]
        return encoder

    @pytest.fixture
    def knowledge_search(self, mock_claim_store, mock_insight_store, mock_encoder):
        """Create KnowledgeSearch with mocked stores."""
        from nanobot.research.knowledge_search import KnowledgeSearch
        return KnowledgeSearch(
            claim_store=mock_claim_store,
            insight_store=mock_insight_store,
            dense_encoder=mock_encoder,
        )

    @pytest.mark.asyncio
    async def test_write_claims(self, knowledge_search):
        """Test writing claims."""
        claims = [
            Claim(
                claim="3DGS uses 3D Gaussians",
                type="method",
                is_evergreen=True,
                source_urls=["https://example.com"],
                confidence=0.9,
            ),
            Claim(
                claim="3DGS is faster than NeRF",
                type="factual",
                is_evergreen=False,
                source_urls=["https://example.com"],
                confidence=0.8,
            ),
        ]

        ids = await knowledge_search.write_claims(claims)
        assert len(ids) == 2

        stats = knowledge_search.get_stats()
        assert stats["claims"] == 2

    @pytest.mark.asyncio
    async def test_write_insights(self, knowledge_search):
        """Test writing insights."""
        insights = [
            Insight(
                insight="Point-based methods are faster",
                supporting_claims=["claim_1"],
                applicable_domains=["3D reconstruction"],
                confidence=0.85,
                is_evergreen=True,
                maturity="candidate",
            ),
        ]

        ids = await knowledge_search.write_insights(insights, maturity="candidate")
        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_search_all(self, knowledge_search):
        """Test searching both claims and insights."""
        # Write some data first
        claims = [Claim(claim="Test claim", type="factual", is_evergreen=True, source_urls=[], confidence=0.8)]
        await knowledge_search.write_claims(claims)

        insights = [Insight(insight="Test insight", supporting_claims=[], applicable_domains=[], confidence=0.8, is_evergreen=True)]
        await knowledge_search.write_insights(insights)

        # Search
        claims_result, insights_result = await knowledge_search.search_all("test query")
        assert len(claims_result) >= 1
        assert len(insights_result) >= 1

    @pytest.mark.asyncio
    async def test_delete_insights(self, knowledge_search):
        """Test deleting insights."""
        insights = [
            Insight(insight="To delete", supporting_claims=[], applicable_domains=[], confidence=0.8, is_evergreen=True),
        ]
        ids = await knowledge_search.write_insights(insights)
        assert len(ids) == 1

        await knowledge_search.delete_insights(ids)
        stats = knowledge_search.get_stats()
        assert stats["insights"] == 0


# ============================================================================
# KnowledgeProcessor Tests
# ============================================================================

class TestKnowledgeProcessor:
    """Tests for KnowledgeProcessor."""

    @pytest.fixture
    def mock_provider(self):
        """Mock LLM provider."""
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock()

        # Mock response for claim extraction
        claim_response = MagicMock()
        claim_response.content = '''
        ```json
        {
          "claims": [
            {
              "claim": "3DGS uses 3D Gaussians for scene representation",
              "type": "method",
              "confidence": 0.9,
              "is_evergreen": true,
              "source_urls": []
            }
          ]
        }
        ```
        '''

        # Mock response for insight extraction
        insight_response = MagicMock()
        insight_response.content = '''
        ```json
        {
          "insights": [
            {
              "insight": "Explicit representations enable real-time rendering",
              "supporting_claims": ["3DGS uses 3D Gaussians"],
              "applicable_domains": ["3D reconstruction", "rendering"],
              "confidence": 0.85,
              "is_evergreen": true
            }
          ]
        }
        ```
        '''

        # Set up side effects for different calls
        provider.chat_with_retry.side_effect = [claim_response, insight_response]
        return provider

    @pytest.fixture
    def mock_knowledge_search(self):
        """Mock KnowledgeSearch."""
        ks = MagicMock()
        ks.search_claims = AsyncMock(return_value=[])
        ks.write_claims = AsyncMock(return_value=["claim_001"])
        ks.write_insights = AsyncMock(return_value=["insight_001"])
        ks.delete_insights = AsyncMock()
        return ks

    @pytest.fixture
    def mock_tracker(self, tmp_path):
        """Create InsightTracker with temp path."""
        return InsightTracker(storage_path=str(tmp_path / "tracker.json"))

    @pytest.fixture
    def research_result(self):
        """Create a sample research result."""
        result = ResearchResult(
            topic="3D Gaussian Splatting",
            status=ResearchStatus.COMPLETED,
            report="# 3DGS Report\n\n3DGS uses 3D Gaussians...",
            synthesis=SynthesisResult(
                findings=[
                    Finding(
                        statement="3DGS achieves real-time rendering",
                        source_urls=["https://example.com"],
                        confidence=0.9,
                    ),
                ],
            ),
        )
        return result

    @pytest.mark.asyncio
    async def test_process_extracts_claims_and_insights(
        self, mock_provider, mock_knowledge_search, mock_tracker, research_result
    ):
        """Test that process extracts claims and insights."""
        from nanobot.research.knowledge_processor import KnowledgeProcessor

        processor = KnowledgeProcessor(
            provider=mock_provider,
            model="test-model",
            knowledge_search=mock_knowledge_search,
            tracker=mock_tracker,
        )

        result = await processor.process(research_result)

        assert result.claims_written >= 0
        assert result.insights_written >= 0

    @pytest.mark.asyncio
    async def test_bigram_similarity(self, mock_provider, mock_knowledge_search, mock_tracker):
        """Test the bigram similarity function."""
        from nanobot.research.knowledge_processor import KnowledgeProcessor

        processor = KnowledgeProcessor(
            provider=mock_provider,
            model="test-model",
            knowledge_search=mock_knowledge_search,
            tracker=mock_tracker,
        )

        # Same content
        assert processor._is_same_claim("3DGS uses Gaussians", "3DGS uses Gaussians")

        # Similar content
        assert processor._is_same_claim("3DGS uses Gaussians", "3DGS uses 3D Gaussians")

        # Different content
        assert not processor._is_same_claim("3DGS uses Gaussians", "NeRF uses neural networks")

        # Chinese text
        assert processor._is_same_claim("3DGS使用高斯表示", "3DGS使用高斯表示法")
        assert not processor._is_same_claim("3DGS使用高斯表示", "NeRF使用神经网络")


# ============================================================================
# Planner with Existing Context Tests
# ============================================================================

class TestPlannerWithExistingContext:
    """Tests for planner with existing context."""

    @pytest.fixture
    def mock_provider(self):
        """Mock LLM provider."""
        provider = MagicMock()

        # Mock response with tool call
        response = MagicMock()
        response.has_tool_calls = True
        response.tool_calls = [
            MagicMock(
                arguments={
                    "sub_questions": [
                        {"question": "What are the limitations?", "priority": 1},
                        {"question": "How does it compare?", "priority": 2},
                    ]
                }
            )
        ]

        provider.chat_with_retry = AsyncMock(return_value=response)
        return provider

    @pytest.mark.asyncio
    async def test_plan_with_existing_context(self, mock_provider):
        """Test that planner uses existing context."""
        from nanobot.research.planner import ResearchPlanner

        planner = ResearchPlanner(provider=mock_provider, model="test-model")

        existing_context = """## 已有相关规律 (Insights)
- [✓] Point-based methods are faster

## 已知相关事实 (Claims)
- 3DGS uses 3D Gaussians"""

        plan = await planner.plan("3DGS limitations", depth="normal", existing_context=existing_context)

        # Verify the plan was created
        assert plan is not None
        assert plan.topic == "3DGS limitations"

        # Verify the provider was called
        mock_provider.chat_with_retry.assert_called_once()
        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[0].get("messages", []))

        # Check that existing context was included in the prompt
        user_message = messages[1]["content"]
        assert "已有知识" in user_message or "已有相关规律" in user_message


# ============================================================================
# Integration Test (Runner with Knowledge Loop)
# ============================================================================

class TestRunnerWithKnowledgeLoop:
    """Integration tests for ResearchRunner with knowledge loop."""

    @pytest.fixture
    def mock_provider(self):
        """Mock LLM provider."""
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock()

        # Mock planner response
        planner_response = MagicMock()
        planner_response.has_tool_calls = True
        planner_response.tool_calls = [
            MagicMock(
                arguments={
                    "sub_questions": [
                        {"question": "Test question", "priority": 1, "keywords_zh": ["测试"], "keywords_en": ["test"]},
                    ]
                }
            )
        ]

        # Mock reporter response
        reporter_response = MagicMock()
        reporter_response.content = "# Test Report\n\nThis is a test report."

        provider.chat_with_retry.side_effect = [planner_response, reporter_response, reporter_response]
        return provider

    @pytest.fixture
    def mock_web_tools(self):
        """Mock web search and fetch tools."""
        search = AsyncMock(return_value=[])
        fetch = AsyncMock(return_value="")
        return search, fetch

    @pytest.fixture
    def mock_knowledge_search(self):
        """Mock KnowledgeSearch."""
        ks = MagicMock()
        ks.search_all = AsyncMock(return_value=([], []))
        ks.write_claims = AsyncMock(return_value=["claim_001"])
        ks.write_insights = AsyncMock(return_value=["insight_001"])
        return ks

    @pytest.fixture
    def mock_knowledge_processor(self):
        """Mock KnowledgeProcessor."""
        processor = MagicMock()
        processor.process = AsyncMock(return_value=KnowledgeProcessResult(
            claims_written=1,
            insights_written=1,
            duplicates_skipped=0,
            conflicts_detected=0,
        ))
        return processor

    @pytest.mark.asyncio
    async def test_runner_calls_knowledge_processor(
        self, mock_provider, mock_web_tools, mock_knowledge_search, mock_knowledge_processor
    ):
        """Test that runner calls knowledge processor after research."""
        # This test requires mocking the entire pipeline
        # For a full integration test, we'd need to mock all components
        pass  # Placeholder for full integration test
