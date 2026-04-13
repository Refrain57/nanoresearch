"""Tests for RAG retrieval evaluation.

Tests retrieval quality using ground truth:
- Precision & Recall
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.agent.evaluation.evaluator import (
    RAGEvaluator,
    RetrievalTestCase,
    RetrievalEvaluationResult,
)
from tests.agent.evaluation.test_datasets import RAG_TEST_CASES


class TestRAGEvaluation:
    """Tests for RAG retrieval quality."""

    def test_evaluator_initialization(self):
        """Test that evaluator can be initialized with test cases."""
        evaluator = RAGEvaluator(RAG_TEST_CASES)

        assert len(evaluator.test_cases) == len(RAG_TEST_CASES)
        # Check first test case exists
        first_query = RAG_TEST_CASES[0].query
        assert first_query in evaluator.test_cases

    def test_single_retrieval_perfect_recall(self):
        """Test perfect recall when all relevant chunks are retrieved."""
        test_cases = [
            RetrievalTestCase(
                query="test query",
                relevant_chunks=["a", "b", "c"],
                relevant_sources=["source1"],
            )
        ]
        evaluator = RAGEvaluator(test_cases)

        result = evaluator.evaluate_retrieval(
            query="test query",
            retrieved_chunks=["a", "b", "c", "d", "e"],  # Includes all relevant
        )

        assert result.recall == 1.0, "Recall should be 100%"
        assert result.precision == 0.6, "Precision should be 3/5"

    def test_single_retrieval_partial_recall(self):
        """Test partial recall when only some relevant chunks are retrieved."""
        test_cases = [
            RetrievalTestCase(
                query="test query",
                relevant_chunks=["a", "b", "c", "d"],
                relevant_sources=["source1"],
            )
        ]
        evaluator = RAGEvaluator(test_cases)

        result = evaluator.evaluate_retrieval(
            query="test query",
            retrieved_chunks=["a", "b", "x"],  # Only 2 out of 4 relevant
        )

        assert result.recall == 0.5, "Recall should be 50% (2/4)"
        assert result.precision == pytest.approx(2/3, 0.01)

    def test_mrr_calculation(self):
        """Test Mean Reciprocal Rank calculation."""
        test_cases = [
            RetrievalTestCase(
                query="test",
                relevant_chunks=["target"],
                relevant_sources=[],
            )
        ]
        evaluator = RAGEvaluator(test_cases)

        # Target at position 1
        result = evaluator.evaluate_retrieval("test", ["target", "a", "b"])
        assert result.mrr == 1.0

        # Target at position 2
        result = evaluator.evaluate_retrieval("test", ["a", "target", "b"])
        assert result.mrr == 0.5

        # Target at position 3
        result = evaluator.evaluate_retrieval("test", ["a", "b", "target"])
        assert result.mrr == pytest.approx(1/3, 0.01)

        # Target not retrieved
        result = evaluator.evaluate_retrieval("test", ["a", "b", "c"])
        assert result.mrr == 0.0

    def test_ndcg_calculation(self):
        """Test NDCG calculation."""
        test_cases = [
            RetrievalTestCase(
                query="test",
                relevant_chunks=["a", "b", "c"],
                relevant_sources=[],
            )
        ]
        evaluator = RAGEvaluator(test_cases)

        # Perfect ranking
        result = evaluator.evaluate_retrieval("test", ["a", "b", "c"])
        assert result.ndcg == pytest.approx(1.0, 0.01)

        # Suboptimal ranking
        result = evaluator.evaluate_retrieval("test", ["x", "a", "y", "b"])
        assert 0 < result.ndcg < 1

        # No relevant results
        result = evaluator.evaluate_retrieval("test", ["x", "y", "z"])
        assert result.ndcg == 0.0

    def test_batch_evaluation(self):
        """Test batch evaluation across multiple queries."""
        evaluator = RAGEvaluator(RAG_TEST_CASES)

        # Simulate retrieval results
        retrieval_results = {}
        for tc in RAG_TEST_CASES:
            # Simulate: return 50-80% of relevant chunks plus some noise
            relevant = tc.relevant_chunks
            retrieved = relevant[:max(1, len(relevant) * 2 // 3)]  # 66% recall
            retrieved.extend(["noise1", "noise2"])  # Add some noise
            retrieval_results[tc.query] = retrieved

        batch_result = evaluator.evaluate_batch(retrieval_results)

        assert "overall" in batch_result
        assert "precision" in batch_result["overall"]
        assert "recall" in batch_result["overall"]
        assert "f1" in batch_result["overall"]
        assert "mrr" in batch_result["overall"]
        assert "ndcg" in batch_result["overall"]

    def test_category_breakdown(self):
        """Test per-category evaluation breakdown."""
        evaluator = RAGEvaluator(RAG_TEST_CASES)

        retrieval_results = {tc.query: tc.relevant_chunks for tc in RAG_TEST_CASES}

        batch_result = evaluator.evaluate_batch(retrieval_results)

        assert "by_category" in batch_result
        # Should have categories from test cases
        categories = set(tc.category for tc in RAG_TEST_CASES if tc.category)
        for cat in categories:
            if cat in batch_result["by_category"]:
                assert "avg_precision" in batch_result["by_category"][cat]
                assert "avg_recall" in batch_result["by_category"][cat]


class TestRAGIntegration:
    """Integration tests with actual hybrid search."""

    def test_hybrid_search_with_evaluator(self):
        """Test hybrid search output can be evaluated."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch
        from nanobot.rag.core.query_engine.fusion import RRFFusion
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor
        from nanobot.rag.core.types import RetrievalResult

        # Create mock retrievers
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="azure_openai_config", score=0.95, text="Azure config", metadata={}),
            RetrievalResult(chunk_id="api_key_setup", score=0.88, text="API setup", metadata={}),
            RetrievalResult(chunk_id="other_doc", score=0.75, text="Other", metadata={}),
        ]

        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="azure_openai_config", score=5.0, text="Azure config", metadata={}),
            RetrievalResult(chunk_id="azure_credentials", score=4.2, text="Credentials", metadata={}),
        ]

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
        )

        # Run search
        results = hybrid.search("test query azure", top_k=5)
        retrieved_ids = [r.chunk_id for r in results]

        # Evaluate - use a simple test case
        evaluator = RAGEvaluator([
            RetrievalTestCase(
                query="test query azure",
                relevant_chunks=["azure_openai_config"],
                relevant_sources=["example.com"],
            )
        ])
        eval_result = evaluator.evaluate_retrieval(
            query="test query azure",
            retrieved_chunks=retrieved_ids,
        )

        # Should have found relevant chunks
        assert eval_result.recall > 0, "Should retrieve some relevant chunks"


class TestRAGBenchmark:
    """Benchmark tests for RAG quality."""

    def test_rag_benchmark(self):
        """Run RAG quality benchmark against ground truth."""
        evaluator = RAGEvaluator(RAG_TEST_CASES)

        # Simulate realistic retrieval (70-90% recall, some noise)
        all_results = {}
        for tc in RAG_TEST_CASES:
            # Simulate varying quality
            relevant = tc.relevant_chunks
            n_retrieved = min(len(relevant) + 2, 10)

            # Return most relevant chunks plus some noise
            retrieved = relevant[:int(len(relevant) * 0.75)]  # ~75% recall
            for i in range(n_retrieved - len(retrieved)):
                retrieved.append(f"noise_{i}")

            all_results[tc.query] = retrieved[:n_retrieved]

        batch_result = evaluator.evaluate_batch(all_results)

        # Print results
        print("\n=== RAG Benchmark Results ===")
        print(f"Precision: {batch_result['overall']['precision']:.2%}")
        print(f"Recall: {batch_result['overall']['recall']:.2%}")
        print(f"F1: {batch_result['overall']['f1']:.2%}")
        print(f"MRR: {batch_result['overall']['mrr']:.2%}")
        print(f"NDCG: {batch_result['overall']['ndcg']:.2%}")

        # Quality threshold
        assert batch_result["overall"]["ndcg"] >= 0.5, \
            f"NDCG {batch_result['overall']['ndcg']:.2%} < 50%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])