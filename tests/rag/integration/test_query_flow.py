"""Integration tests for Query Flow.

Tests the complete query flow:
- Query processing → hybrid search → reranking → response building
- End-to-end retrieval pipeline
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional, Dict, Any

from nanobot.rag.core.types import RetrievalResult, ProcessedQuery
from nanobot.rag.core.query_engine.fusion import RRFFusion
from nanobot.rag.core.query_engine.query_processor import QueryProcessor


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    def upsert(self, records: List) -> List[str]:
        ids = []
        for record in records:
            self._data[record.id] = {
                "id": record.id,
                "text": record.text,
                "metadata": record.metadata,
                "dense_vector": getattr(record, "dense_vector", None),
            }
            ids.append(record.id)
        return ids

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        results = []
        for chunk_id, data in list(self._data.items())[:top_k]:
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=0.9 - len(results) * 0.1,
                text=data["text"],
                metadata=data["metadata"],
            ))
        return results

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        return [self._data.get(id, {}) for id in ids if id in self._data]


class TestQueryFlowBasic:
    """Tests for basic query flow."""

    def test_query_processing_to_hybrid_search(self):
        """Test query processing feeds into hybrid search."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig

        # Process query
        processor = QueryProcessor()
        processed = processor.process("如何配置 Azure OpenAI collection:docs")

        assert "Azure" in processed.keywords or "OpenAI" in processed.keywords or "配置" in processed.keywords
        assert processed.filters.get("collection") == "docs"

    def test_full_hybrid_search_flow(self):
        """Test complete hybrid search flow."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch

        # Mock dense retriever
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="a", score=0.9, text="Azure config", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="OpenAI setup", metadata={}),
        ]

        # Mock sparse retriever
        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="b", score=5.0, text="OpenAI setup", metadata={}),
            RetrievalResult(chunk_id="c", score=4.0, text="Python SDK", metadata={}),
        ]

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        results = hybrid.search("Azure OpenAI configuration", top_k=5)

        # "b" appears in both, should rank first
        assert len(results) > 0
        assert results[0].chunk_id == "b"  # Highest RRF score

    def test_hybrid_search_returns_details(self):
        """Test hybrid search returns detailed results."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchResult

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="a", score=0.9, text="Azure", metadata={}),
        ]

        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="b", score=5.0, text="OpenAI", metadata={}),
        ]

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        result = hybrid.search("test", return_details=True)

        assert isinstance(result, HybridSearchResult)
        assert len(result.results) == 2
        assert result.dense_results is not None
        assert result.sparse_results is not None
        assert result.used_fallback is False


class TestQueryFlowWithReranking:
    """Tests for query flow with reranking."""

    def test_search_with_reranking(self):
        """Test hybrid search followed by reranking."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch
        from nanobot.rag.core.query_engine.reranker import CoreReranker, RerankConfig

        # Mock dense retriever
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="a", score=0.9, text="Azure configuration guide", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="OpenAI API reference", metadata={}),
            RetrievalResult(chunk_id="c", score=0.7, text="Python SDK", metadata={}),
        ]

        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = []

        # Mock reranker that reverses order
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = lambda query, candidates, **kwargs: [
            {"id": c["id"], "rerank_score": 1.0 - i * 0.1, "text": c["text"], "metadata": c["metadata"]}
            for i, c in enumerate(reversed(candidates))
        ]
        mock_reranker.__class__.__name__ = "MockReranker"

        # Hybrid search
        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        search_results = hybrid.search("Azure OpenAI", top_k=5)

        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.rerank.enabled = True
        mock_settings.rerank.top_k = 3

        # Rerank
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=mock_reranker,
            config=RerankConfig(enabled=True, top_k=3),
        )

        rerank_result = reranker.rerank("Azure OpenAI", search_results, top_k=3)

        assert len(rerank_result.results) <= 3
        assert "mock" in rerank_result.reranker_type.lower()


class TestQueryFlowFilters:
    """Tests for query flow with filters."""

    def test_filter_propagation_to_dense(self):
        """Test that filters propagate to dense retriever."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = []
        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = []

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        hybrid.search("test collection:api-docs type:pdf")

        # Check filters passed to dense retriever
        call_filters = mock_dense.retrieve.call_args[1].get("filters", {})
        assert call_filters.get("collection") == "api-docs"
        assert call_filters.get("doc_type") == "pdf"

    def test_explicit_filter_override(self):
        """Test explicit filters override query filters."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = []
        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = []

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        # Query has collection:docs, explicit filter overrides
        hybrid.search("test collection:docs", filters={"collection": "api"})

        call_filters = mock_dense.retrieve.call_args[1].get("filters", {})
        assert call_filters.get("collection") == "api"


class TestQueryFlowErrorHandling:
    """Tests for error handling in query flow."""

    def test_dense_failure_fallback(self):
        """Test fallback when dense retrieval fails."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch

        mock_dense = MagicMock()
        mock_dense.retrieve.side_effect = Exception("Dense retrieval failed")

        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="a", score=5.0, text="Result", metadata={}),
        ]

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        result = hybrid.search("test query", return_details=True)

        assert result.used_fallback is True
        assert result.dense_error is not None
        assert len(result.results) == 1

    def test_both_failures_raise_error(self):
        """Test error raised when both retrievers fail."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch

        mock_dense = MagicMock()
        mock_dense.retrieve.side_effect = Exception("Dense failed")

        mock_sparse = MagicMock()
        mock_sparse.retrieve.side_effect = Exception("Sparse failed")

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        with pytest.raises(RuntimeError, match="Both retrieval paths failed"):
            hybrid.search("test query")

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=MagicMock(),
            sparse_retriever=MagicMock(),
            fusion=RRFFusion(k=60),
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.search("")


class TestQueryFlowRRF:
    """Tests for RRF fusion in query flow."""

    def test_rrf_fusion_with_two_lists(self):
        """Test RRF fusion combines results from two lists."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch

        dense_results = [
            RetrievalResult(chunk_id="a", score=0.9, text="A", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="B", metadata={}),
        ]

        sparse_results = [
            RetrievalResult(chunk_id="b", score=5.0, text="B", metadata={}),
            RetrievalResult(chunk_id="c", score=4.0, text="C", metadata={}),
        ]

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = dense_results

        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = sparse_results

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        results = hybrid.search("test query")

        # "b" appears in both, should have highest RRF score
        assert results[0].chunk_id == "b"
        # All three should appear
        assert len(results) == 3

    def test_single_retriever_mode(self):
        """Test hybrid search with only one retriever."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="a", score=0.9, text="A", metadata={}),
            RetrievalResult(chunk_id="b", score=0.8, text="B", metadata={}),
        ]

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=None,  # No sparse
            fusion=RRFFusion(k=60),
        )

        results = hybrid.search("test query")

        assert len(results) == 2
        mock_dense.retrieve.assert_called_once()
