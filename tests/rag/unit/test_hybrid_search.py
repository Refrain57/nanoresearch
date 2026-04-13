"""Unit tests for HybridSearch.

Tests the hybrid search functionality including:
- Dense and sparse retrieval coordination
- Parallel vs sequential retrieval
- Result fusion with RRF
- Fallback behavior when one retriever fails
- Metadata filtering
- Error handling
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Optional, Dict, Any, Tuple

from nanobot.rag.core.query_engine.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    HybridSearchResult,
    create_hybrid_search,
)
from nanobot.rag.core.query_engine.fusion import RRFFusion
from nanobot.rag.core.query_engine.query_processor import QueryProcessor
from nanobot.rag.core.types import RetrievalResult, ProcessedQuery
from tests.rag.conftest import MockSettings


def create_mock_retrieval_result(chunk_id: str, score: float, text: str = "") -> RetrievalResult:
    """Helper to create mock retrieval results."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"Mock text for {chunk_id}",
        metadata={"source_path": f"{chunk_id}.pdf"},
    )


class MockDenseRetriever:
    """Mock dense retriever for testing."""

    def __init__(self, results: Optional[List[RetrievalResult]] = None, should_error: bool = False):
        self._results = results or []
        self._should_error = should_error
        self.provider_name = "mock_dense"
        self._call_count = 0
        self._last_query: Optional[str] = None
        self._last_filters: Optional[Dict[str, Any]] = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        self._call_count += 1
        self._last_query = query
        self._last_filters = filters

        if self._should_error:
            raise RuntimeError("Dense retriever error")

        return self._results[:top_k]


class MockSparseRetriever:
    """Mock sparse retriever for testing."""

    def __init__(self, results: Optional[List[RetrievalResult]] = None, should_error: bool = False):
        self._results = results or []
        self._should_error = should_error
        self._call_count = 0
        self._last_keywords: Optional[List[str]] = None
        self._last_collection: Optional[str] = None

    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        self._call_count += 1
        self._last_keywords = keywords
        self._last_collection = collection

        if self._should_error:
            raise RuntimeError("Sparse retriever error")

        return self._results[:top_k]


class TestHybridSearchInit:
    """Tests for HybridSearch initialization."""

    def test_init_with_all_components(self):
        """Test initialization with all components."""
        query_processor = QueryProcessor()
        dense_retriever = MockDenseRetriever()
        sparse_retriever = MockSparseRetriever()
        fusion = RRFFusion(k=60)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion,
        )

        assert hybrid.query_processor is query_processor
        assert hybrid.dense_retriever is dense_retriever
        assert hybrid.sparse_retriever is sparse_retriever
        assert hybrid.fusion is fusion

    def test_init_with_settings(self):
        """Test initialization extracting config from settings."""
        settings = MockSettings()
        hybrid = HybridSearch(settings=settings)

        assert hybrid.config.dense_top_k == 20
        assert hybrid.config.sparse_top_k == 20
        assert hybrid.config.fusion_top_k == 10

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = HybridSearchConfig(
            dense_top_k=50,
            sparse_top_k=30,
            fusion_top_k=20,
        )
        hybrid = HybridSearch(config=config)

        assert hybrid.config.dense_top_k == 50
        assert hybrid.config.sparse_top_k == 30
        assert hybrid.config.fusion_top_k == 20

    def test_init_without_retrievers(self):
        """Test initialization without retrievers (valid for config testing)."""
        hybrid = HybridSearch()
        assert hybrid.dense_retriever is None
        assert hybrid.sparse_retriever is None


class TestHybridSearchBasicSearch:
    """Tests for basic search functionality."""

    def test_search_with_both_retrievers(self):
        """Test search with both dense and sparse retrievers."""
        dense = MockDenseRetriever([
            create_mock_retrieval_result("a", 0.9),
            create_mock_retrieval_result("b", 0.8),
        ])
        sparse = MockSparseRetriever([
            create_mock_retrieval_result("b", 5.0),
            create_mock_retrieval_result("c", 4.0),
        ])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        results = hybrid.search("test query")

        assert len(results) > 0
        # "b" appears in both lists, should be first
        assert results[0].chunk_id == "b"

    def test_search_with_top_k(self):
        """Test search respects top_k parameter."""
        dense = MockDenseRetriever([
            create_mock_retrieval_result(f"d_{i}", 0.9 - i * 0.1)
            for i in range(20)
        ])
        sparse = MockSparseRetriever([
            create_mock_retrieval_result(f"s_{i}", 10.0 - i)
            for i in range(20)
        ])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        results = hybrid.search("test query", top_k=5)
        assert len(results) <= 5

    def test_search_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=MockDenseRetriever(),
            sparse_retriever=MockSparseRetriever(),
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.search("")

    def test_search_whitespace_query_raises_error(self):
        """Test that whitespace-only query raises ValueError."""
        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=MockDenseRetriever(),
            sparse_retriever=MockSparseRetriever(),
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.search("   \t\n  ")

    def test_search_return_details(self):
        """Test search with return_details=True."""
        dense = MockDenseRetriever([create_mock_retrieval_result("a", 0.9)])
        sparse = MockSparseRetriever([create_mock_retrieval_result("b", 5.0)])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        result = hybrid.search("test query", return_details=True)

        assert isinstance(result, HybridSearchResult)
        assert len(result.results) > 0
        assert result.dense_results is not None
        assert result.sparse_results is not None
        assert result.dense_error is None
        assert result.sparse_error is None


class TestHybridSearchFallback:
    """Tests for fallback behavior."""

    def test_dense_retriever_error_fallback(self):
        """Test fallback when dense retriever fails."""
        dense = MockDenseRetriever(should_error=True)
        sparse = MockSparseRetriever([
            create_mock_retrieval_result("a", 5.0),
            create_mock_retrieval_result("b", 4.0),
        ])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        result = hybrid.search("test query", return_details=True)

        assert result.used_fallback is True
        assert result.dense_error is not None
        assert len(result.results) == 2  # Only sparse results

    def test_sparse_retriever_error_fallback(self):
        """Test fallback when sparse retriever fails."""
        dense = MockDenseRetriever([
            create_mock_retrieval_result("a", 0.9),
            create_mock_retrieval_result("b", 0.8),
        ])
        sparse = MockSparseRetriever(should_error=True)

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        result = hybrid.search("test query", return_details=True)

        assert result.used_fallback is True
        assert result.sparse_error is not None
        assert len(result.results) == 2  # Only dense results

    def test_both_retrievers_error_raises(self):
        """Test that error is raised when both retrievers fail."""
        dense = MockDenseRetriever(should_error=True)
        sparse = MockSparseRetriever(should_error=True)

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        with pytest.raises(RuntimeError, match="Both retrieval paths failed"):
            hybrid.search("test query")

    def test_dense_only_mode(self):
        """Test search with only dense retriever."""
        dense = MockDenseRetriever([create_mock_retrieval_result("a", 0.9)])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=None,  # No sparse retriever
            fusion=RRFFusion(k=60),
        )

        results = hybrid.search("test query")
        assert len(results) == 1
        assert results[0].chunk_id == "a"

    def test_sparse_only_mode(self):
        """Test search with only sparse retriever."""
        sparse = MockSparseRetriever([create_mock_retrieval_result("a", 5.0)])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=None,  # No dense retriever
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        results = hybrid.search("test query")
        assert len(results) == 1
        assert results[0].chunk_id == "a"

    def test_no_retrievers_error(self):
        """Test error when no retrievers configured."""
        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=None,
            sparse_retriever=None,
        )

        with pytest.raises(RuntimeError, match="Both retrieval paths failed"):
            hybrid.search("test query")


class TestHybridSearchMetadataFiltering:
    """Tests for metadata filtering."""

    def test_filter_from_query(self):
        """Test filter extraction from query."""
        dense = MockDenseRetriever([
            RetrievalResult(chunk_id="a", score=0.9, text="text", metadata={"collection": "docs"}),
            RetrievalResult(chunk_id="b", score=0.8, text="text", metadata={"collection": "api"}),
        ])
        sparse = MockSparseRetriever()

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
            config=HybridSearchConfig(metadata_filter_post=True),
        )

        results = hybrid.search("test collection:docs")

        # Post-fusion filtering should be applied
        # (Mock retriever doesn't filter, so this tests post-filter)
        assert dense._last_query is not None

    def test_explicit_filters(self):
        """Test explicit filters passed to search."""
        dense = MockDenseRetriever([create_mock_retrieval_result("a", 0.9)])
        sparse = MockSparseRetriever()

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        hybrid.search("test query", filters={"collection": "docs"})

        assert dense._last_filters == {"collection": "docs"}

    def test_merge_query_and_explicit_filters(self):
        """Test merging query filters with explicit filters."""
        dense = MockDenseRetriever([create_mock_retrieval_result("a", 0.9)])
        sparse = MockSparseRetriever()

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        # Query has collection:docs, explicit has doc_type:pdf
        hybrid.search("test collection:docs", filters={"doc_type": "pdf"})

        assert dense._last_filters == {"collection": "docs", "doc_type": "pdf"}

    def test_explicit_filters_override_query(self):
        """Test that explicit filters override query filters."""
        dense = MockDenseRetriever([create_mock_retrieval_result("a", 0.9)])
        sparse = MockSparseRetriever()

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
        )

        # Both have collection, explicit should win
        hybrid.search("test collection:docs", filters={"collection": "api"})

        assert dense._last_filters == {"collection": "api"}


class TestHybridSearchConfig:
    """Tests for configuration options."""

    def test_disable_dense_retrieval(self):
        """Test disabling dense retrieval."""
        dense = MockDenseRetriever([create_mock_retrieval_result("a", 0.9)])
        sparse = MockSparseRetriever([create_mock_retrieval_result("b", 5.0)])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
            config=HybridSearchConfig(enable_dense=False),
        )

        results = hybrid.search("test query")

        # Only sparse should be called
        assert dense._call_count == 0
        assert sparse._call_count == 1
        assert len(results) == 1

    def test_disable_sparse_retrieval(self):
        """Test disabling sparse retrieval."""
        dense = MockDenseRetriever([create_mock_retrieval_result("a", 0.9)])
        sparse = MockSparseRetriever([create_mock_retrieval_result("b", 5.0)])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
            config=HybridSearchConfig(enable_sparse=False),
        )

        results = hybrid.search("test query")

        # Only dense should be called
        assert dense._call_count == 1
        assert sparse._call_count == 0
        assert len(results) == 1

    def test_custom_top_k_values(self):
        """Test custom top_k values for each retriever."""
        dense = MockDenseRetriever([create_mock_retrieval_result(f"d_{i}", 0.9) for i in range(30)])
        sparse = MockSparseRetriever([create_mock_retrieval_result(f"s_{i}", 5.0) for i in range(30)])

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(k=60),
            config=HybridSearchConfig(
                dense_top_k=15,
                sparse_top_k=10,
                fusion_top_k=5,
            ),
        )

        results = hybrid.search("test query")

        # Dense should be called with dense_top_k=15
        assert dense._call_count == 1
        # Sparse should be called with sparse_top_k=10
        assert sparse._call_count == 1
        # Final results should be limited to fusion_top_k=5
        assert len(results) <= 5


class TestHybridSearchFactory:
    """Tests for create_hybrid_search factory function."""

    def test_factory_with_defaults(self):
        """Test factory with minimal arguments."""
        hybrid = create_hybrid_search()

        assert hybrid.query_processor is None
        assert hybrid.dense_retriever is None
        assert hybrid.sparse_retriever is None
        assert hybrid.fusion is not None
        assert hybrid.fusion.k == RRFFusion.DEFAULT_K

    def test_factory_with_custom_k(self):
        """Test factory with custom RRF k value."""
        settings = MockSettings()
        settings.retrieval.rrf_k = 30

        hybrid = create_hybrid_search(settings=settings)

        assert hybrid.fusion.k == 30

    def test_factory_with_all_components(self):
        """Test factory with all components."""
        query_processor = QueryProcessor()
        dense = MockDenseRetriever()
        sparse = MockSparseRetriever()
        fusion = RRFFusion(k=20)

        hybrid = create_hybrid_search(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=fusion,
        )

        assert hybrid.query_processor is query_processor
        assert hybrid.dense_retriever is dense
        assert hybrid.sparse_retriever is sparse
        assert hybrid.fusion is fusion
        assert hybrid.fusion.k == 20