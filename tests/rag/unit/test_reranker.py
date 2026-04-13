"""Unit tests for CoreReranker.

Tests the reranking functionality including:
- Basic reranking with different backends
- Fallback behavior on backend errors
- Type conversion between RetrievalResult and candidate format
- Configuration extraction
- Edge cases (empty results, single result, disabled reranking)
"""

import pytest
from unittest.mock import MagicMock
from typing import List, Dict, Any, Optional

from nanobot.rag.core.query_engine.reranker import (
    CoreReranker,
    RerankConfig,
    RerankResult,
    RerankError,
    create_core_reranker,
)
from nanobot.rag.core.types import RetrievalResult


class MockSettings:
    """Mock Settings object for testing."""
    def __init__(self):
        from dataclasses import dataclass, field

        @dataclass
        class MockSplitter:
            provider: str = "recursive"
            chunk_size: int = 500
            chunk_overlap: int = 50

        @dataclass
        class MockEmbedding:
            provider: str = "openai"
            model: str = "text-embedding-3-small"
            dimension: int = 1536

        @dataclass
        class MockVectorStore:
            provider: str = "chroma"
            persist_directory: str = "./test_data/chroma"

        @dataclass
        class MockRerank:
            enabled: bool = True
            top_k: int = 5

        @dataclass
        class MockRetrieval:
            dense_top_k: int = 20
            sparse_top_k: int = 20
            fusion_top_k: int = 10
            rrf_k: int = 60

        @dataclass
        class MockIngestion:
            batch_size: int = 100

        @dataclass
        class MockLLM:
            provider: str = "openai"
            model: str = "gpt-4o-mini"

        self.splitter = MockSplitter()
        self.embedding = MockEmbedding()
        self.vector_store = MockVectorStore()
        self.rerank = MockRerank()
        self.retrieval = MockRetrieval()
        self.ingestion = MockIngestion()
        self.llm = MockLLM()


class MockBaseReranker:
    """Mock base reranker for testing."""

    def __init__(self, should_error: bool = False, error_type: Exception = Exception):
        self._should_error = should_error
        self._error_type = error_type
        self._call_count = 0

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        self._call_count += 1

        if self._should_error:
            raise self._error_type("Mock reranker error")

        # Mock reranking: reverse order and add rerank_score
        reranked = []
        for i, c in enumerate(reversed(candidates)):
            candidate = c.copy()
            candidate["rerank_score"] = 1.0 - (i * 0.1)
            reranked.append(candidate)

        return reranked


class TestRerankConfig:
    """Tests for RerankConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RerankConfig()
        assert config.enabled is True
        assert config.top_k == 5
        assert config.timeout == 30.0
        assert config.fallback_on_error is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RerankConfig(
            enabled=False,
            top_k=10,
            timeout=60.0,
            fallback_on_error=False,
        )
        assert config.enabled is False
        assert config.top_k == 10
        assert config.timeout == 60.0
        assert config.fallback_on_error is False


class TestCoreRerankerInit:
    """Tests for CoreReranker initialization."""

    def test_init_with_mock_reranker(self):
        """Test initialization with mock reranker."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        config = RerankConfig()

        core = CoreReranker(
            settings=settings,
            reranker=reranker,
            config=config,
        )

        assert core.settings is settings
        assert core._reranker is reranker
        assert core.config is config

    def test_init_extracts_config_from_settings(self):
        """Test config extraction from settings."""
        settings = MockSettings()
        settings.rerank.enabled = True
        settings.rerank.top_k = 8

        core = CoreReranker(settings=settings)

        assert core.config.enabled is True
        assert core.config.top_k == 8

    def test_init_disabled_rerank(self):
        """Test initialization when reranking disabled."""
        settings = MockSettings()
        settings.rerank.enabled = False

        core = CoreReranker(settings=settings)

        # Should use NoneReranker
        assert core._reranker.__class__.__name__ == "NoneReranker"
        assert core.config.enabled is False

    def test_init_detects_reranker_type_llm(self):
        """Test reranker type detection for LLM."""
        settings = MockSettings()
        reranker = MagicMock()
        reranker.__class__.__name__ = "LLMReranker"

        core = CoreReranker(settings=settings, reranker=reranker)

        assert core._reranker_type == "llm"

    def test_init_detects_reranker_type_cross_encoder(self):
        """Test reranker type detection for CrossEncoder."""
        settings = MockSettings()
        reranker = MagicMock()
        reranker.__class__.__name__ = "CrossEncoderReranker"

        core = CoreReranker(settings=settings, reranker=reranker)

        assert core._reranker_type == "cross_encoder"

    def test_init_detects_reranker_type_none(self):
        """Test reranker type detection for NoneReranker."""
        settings = MockSettings()
        reranker = MagicMock()
        reranker.__class__.__name__ = "NoneReranker"

        core = CoreReranker(settings=settings, reranker=reranker)

        assert core._reranker_type == "none"


class TestCoreRerankerBasicRerank:
    """Tests for basic reranking."""

    def test_rerank_basic(self):
        """Test basic reranking operation."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker)

        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="text a", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="text b", metadata={}),
            RetrievalResult(chunk_id="c", score=0.7, text="text c", metadata={}),
        ]

        result = core.rerank("test query", results)

        assert isinstance(result, RerankResult)
        assert len(result.results) == 3
        # Mock reranker reverses order: c (from reversed) gets rerank_score=1.0, b=0.9, a=0.8
        # Results are sorted by rerank_score descending
        assert result.results[0].chunk_id == "c"
        assert result.results[1].chunk_id == "b"
        assert result.results[2].chunk_id == "a"

    def test_rerank_with_top_k(self):
        """Test reranking with top_k limit."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker)

        results = [
            RetrievalResult(chunk_id=str(i), score=0.9 - i * 0.1, text="text", metadata={})
            for i in range(20)
        ]

        result = core.rerank("test query", results, top_k=5)

        assert len(result.results) == 5

    def test_rerank_empty_results(self):
        """Test reranking with empty results."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker)

        result = core.rerank("test query", [])

        assert result.results == []
        assert result.used_fallback is False

    def test_rerank_single_result(self):
        """Test reranking with single result."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker)

        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="text a", metadata={}),
        ]

        result = core.rerank("test query", results)

        # Single result doesn't need reranking
        assert len(result.results) == 1
        assert result.results[0].chunk_id == "a"
        assert result.used_fallback is False
        # Should not call reranker for single result
        assert reranker._call_count == 0


class TestCoreRerankerDisabled:
    """Tests for disabled reranking."""

    def test_disabled_rerank_returns_original(self):
        """Test that disabled reranking returns original order."""
        settings = MockSettings()
        config = RerankConfig(enabled=False)
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker, config=config)

        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="text a", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="text b", metadata={}),
        ]

        result = core.rerank("test query", results)

        # Original order preserved
        assert result.results[0].chunk_id == "a"
        assert result.results[1].chunk_id == "b"
        assert result.reranker_type == "none"

    def test_is_enabled_property(self):
        """Test is_enabled property."""
        enabled_core = CoreReranker(
            settings=MockSettings(),
            reranker=MockBaseReranker(),
            config=RerankConfig(enabled=True),
        )
        assert enabled_core.is_enabled is True

        disabled_core = CoreReranker(
            settings=MockSettings(),
            reranker=MagicMock(),
            config=RerankConfig(enabled=False),
        )
        assert disabled_core.is_enabled is False

    def test_reranker_type_property(self):
        """Test reranker_type property."""
        reranker = MagicMock()
        reranker.__class__.__name__ = "CustomReranker"
        core = CoreReranker(settings=MockSettings(), reranker=reranker)

        assert core.reranker_type == "customreranker"


class TestCoreRerankerFallback:
    """Tests for fallback behavior."""

    def test_fallback_on_error(self):
        """Test fallback to original order on error."""
        settings = MockSettings()
        reranker = MockBaseReranker(should_error=True)
        core = CoreReranker(
            settings=settings,
            reranker=reranker,
            config=RerankConfig(fallback_on_error=True),
        )

        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="text a", metadata={"key": "val"}),
            RetrievalResult(chunk_id="b", score=0.8, text="text b", metadata={}),
        ]

        result = core.rerank("test query", results)

        assert result.used_fallback is True
        assert result.fallback_reason is not None
        assert result.fallback_reason == "Mock reranker error"
        # Original order preserved
        assert result.results[0].chunk_id == "a"
        assert result.results[1].chunk_id == "b"
        # Metadata preserved
        assert result.results[0].metadata["key"] == "val"

    def test_fallback_metadata_flag(self):
        """Test that fallback adds metadata flag."""
        settings = MockSettings()
        # Use MagicMock directly with side_effect to trigger error
        reranker = MagicMock()
        reranker.rerank.side_effect = Exception("Reranker failed")

        core = CoreReranker(
            settings=settings,
            reranker=reranker,
            config=RerankConfig(fallback_on_error=True, enabled=True),
        )

        # Need at least 2 results to trigger reranking (single result returns early)
        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="text a", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="text b", metadata={}),
        ]

        result = core.rerank("test query", results)

        assert result.used_fallback is True
        assert result.results[0].metadata.get("rerank_fallback") is True
        assert result.results[0].metadata.get("reranked") is False

    def test_error_without_fallback(self):
        """Test error raised when fallback disabled."""
        settings = MockSettings()
        reranker = MagicMock()
        reranker.rerank.side_effect = Exception("Reranker failed")

        core = CoreReranker(
            settings=settings,
            reranker=reranker,
            config=RerankConfig(fallback_on_error=False, enabled=True),
        )

        # Need at least 2 results to trigger reranking
        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="text a", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="text b", metadata={}),
        ]

        with pytest.raises(RerankError, match="Reranking failed and fallback disabled"):
            core.rerank("test query", results)

    def test_original_order_preserved_in_result(self):
        """Test that original order is preserved in result."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker)

        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="text a", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="text b", metadata={}),
        ]

        result = core.rerank("test query", results)

        assert result.original_order is not None
        assert len(result.original_order) == 2
        assert result.original_order[0].chunk_id == "a"


class TestCoreRerankerTypeConversion:
    """Tests for type conversion between formats."""

    def test_results_to_candidates(self):
        """Test conversion from RetrievalResult to candidate dict."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker)

        results = [
            RetrievalResult(
                chunk_id="a",
                score=0.9,
                text="text content",
                metadata={"key": "val"},
            ),
        ]

        candidates = core._results_to_candidates(results)

        assert len(candidates) == 1
        assert candidates[0]["id"] == "a"
        assert candidates[0]["score"] == 0.9
        assert candidates[0]["text"] == "text content"
        assert candidates[0]["metadata"] == {"key": "val"}

    def test_candidates_to_results(self):
        """Test conversion from candidate dict to RetrievalResult."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker)

        original = [
            RetrievalResult(
                chunk_id="a",
                score=0.9,
                text="original text",
                metadata={"key": "val"},
            ),
        ]

        candidates = [
            {
                "id": "a",
                "score": 0.95,
                "rerank_score": 0.95,
                "text": "original text",
                "metadata": {"key": "val"},
            },
        ]

        results = core._candidates_to_results(candidates, original)

        assert len(results) == 1
        assert results[0].chunk_id == "a"
        assert results[0].score == 0.95
        assert results[0].metadata["original_score"] == 0.9
        assert results[0].metadata["rerank_score"] == 0.95
        assert results[0].metadata["reranked"] is True

    def test_candidates_not_in_original(self):
        """Test handling of candidates not found in original results."""
        settings = MockSettings()
        reranker = MockBaseReranker()
        core = CoreReranker(settings=settings, reranker=reranker)

        original = [
            RetrievalResult(chunk_id="a", score=0.9, text="text a", metadata={}),
        ]

        candidates = [
            {
                "id": "b",  # Not in original
                "rerank_score": 0.95,
                "text": "text b",
                "metadata": {"key": "val"},
            },
        ]

        results = core._candidates_to_results(candidates, original)

        assert len(results) == 1
        assert results[0].chunk_id == "b"
        assert results[0].score == 0.95


class TestCoreRerankerFactory:
    """Tests for create_core_reranker factory function."""

    def test_factory_basic(self):
        """Test factory creates CoreReranker."""
        settings = MockSettings()
        reranker = MockBaseReranker()

        core = create_core_reranker(settings=settings, reranker=reranker)

        assert isinstance(core, CoreReranker)
        assert core._reranker is reranker
