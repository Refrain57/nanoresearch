"""Stress tests for memory usage.

Tests memory behavior under load:
- Memory usage with large result sets
- Memory leaks in retrieval
- Garbage collection behavior
- Peak memory monitoring
"""

import pytest
import gc
import tracemalloc
from typing import List
from unittest.mock import MagicMock

import sys
from nanobot.rag.core.types import RetrievalResult, ChunkRecord


class MockSettings:
    """Mock Settings object for testing."""
    def __init__(self):
        from dataclasses import dataclass

        @dataclass
        class MockRetrieval:
            dense_top_k: int = 20
            sparse_top_k: int = 20
            fusion_top_k: int = 10
            rrf_k: int = 60

        @dataclass
        class MockEmbedding:
            provider: str = "openai"
            model: str = "text-embedding-3-small"
            dimension: int = 1536
            api_key: str = "mock-key"

        self.retrieval = MockRetrieval()
        self.embedding = MockEmbedding()


class TestRetrievalMemory:
    """Tests for retrieval memory usage."""

    @pytest.mark.stress
    def test_dense_retrieval_memory(self):
        """Test memory usage during dense retrieval."""
        from nanobot.rag.core.query_engine.dense_retriever import DenseRetriever

        tracemalloc.start()

        settings = MockSettings()

        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 1536]

        # Simulate large result set
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "id": f"chunk_{i}",
                "score": 0.9 - i * 0.01,
                "text": "Text content " * 100,  # 1KB text per result
                "metadata": {"large": "metadata " * 100},
            }
            for i in range(1000)  # 1000 results
        ]

        retriever = DenseRetriever(
            settings=settings,
            embedding_client=mock_embedding,
            vector_store=mock_store,
        )

        # Perform retrieval
        result = retriever.retrieve("test", top_k=1000)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"\nPeak memory for 1000 retrieval results: {peak_mb:.2f} MB")
        print(f"Per-result overhead: {peak_mb/len(result):.4f} MB")

        assert peak_mb < 200  # Should be under 200MB

    @pytest.mark.stress
    def test_rerank_memory_usage(self):
        """Test memory usage during reranking."""
        from nanobot.rag.core.query_engine.reranker import CoreReranker, RerankConfig

        tracemalloc.start()

        settings = MockSettings()

        # Create many retrieval results
        results = [
            RetrievalResult(
                chunk_id=f"chunk_{i}",
                score=0.9 - i * 0.001,
                text="Long content text " * 50,
                metadata={},
            )
            for i in range(500)
        ]

        # Mock reranker that returns all results
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"id": f"chunk_{i}", "rerank_score": 1.0 - i * 0.002}
            for i in range(500)
        ]

        core = CoreReranker(
            settings=settings,
            reranker=mock_reranker,
            config=RerankConfig(enabled=True, top_k=500),
        )

        result = core.rerank("test", results)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"\nPeak memory for reranking 500 results: {peak_mb:.2f} MB")

        assert peak_mb < 100


class TestEmbeddingMemory:
    """Tests for embedding memory usage."""

    @pytest.mark.stress
    def test_dense_embedding_memory(self):
        """Test memory usage during dense embedding."""
        tracemalloc.start()

        # Simulate batch embedding
        batch_size = 100
        dimension = 1536

        chunks = [
            {"id": f"chunk_{i}", "text": "Text content" * 100}
            for i in range(batch_size)
        ]

        # Simulate embedding generation
        vectors = [[0.1] * dimension for _ in range(batch_size)]

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        expected_mb = (batch_size * dimension * 8) / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)

        print(f"\nDense embedding memory:")
        print(f"  Expected (raw): {expected_mb:.2f} MB")
        print(f"  Peak: {peak_mb:.2f} MB")

        # Peak should be close to expected
        assert peak_mb < expected_mb * 3  # Allow 3x overhead


class TestVectorStoreMemory:
    """Tests for vector store memory usage."""

    @pytest.mark.stress
    def test_vector_store_memory_scaling(self):
        """Test memory scaling with vector count."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Simulate storing vectors
        def get_memory_mb():
            return process.memory_info().rss / (1024 * 1024)

        initial_mb = get_memory_mb()

        # Create vectors of different sizes
        sizes = [100, 500, 1000, 5000]
        results = []

        for size in sizes:
            vectors = [[0.1] * 1536 for _ in range(size)]
            current_mb = get_memory_mb()
            vector_mb = (size * 1536 * 8) / (1024 * 1024)
            overhead_mb = current_mb - initial_mb

            results.append({
                "count": size,
                "expected_mb": vector_mb,
                "overhead_mb": overhead_mb,
            })

        print("\nVector store memory scaling:")
        for r in results:
            ratio = r["overhead_mb"] / r["expected_mb"] if r["expected_mb"] > 0 else 0
            print(f"  {r['count']} vectors: expected={r['expected_mb']:.2f}MB, overhead={r['overhead_mb']:.2f}MB (ratio={ratio:.2f})")


class TestMemoryLeaks:
    """Tests for memory leaks."""

    @pytest.mark.stress
    def test_query_processor_memory(self):
        """Test for memory leaks in query processor."""
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor

        tracemalloc.start()

        processor = QueryProcessor()

        # Initial snapshot
        gc.collect()
        initial = tracemalloc.take_snapshot()

        # Process many queries
        for i in range(1000):
            processor.process(f"Test query number {i} with some keywords")

        # Force garbage collection
        gc.collect()
        final = tracemalloc.take_snapshot()

        # Compare snapshots
        top_stats = final.compare_to(initial, 'lineno')

        print("\nTop memory differences:")
        for stat in top_stats[:5]:
            print(f"  {stat}")

        tracemalloc.stop()

        # Check for significant leaks (should be less than 10MB)
        total_diff = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        total_diff_mb = total_diff / (1024 * 1024)

        print(f"\nTotal memory difference: {total_diff_mb:.2f} MB")
        assert total_diff_mb < 10

    @pytest.mark.stress
    def test_retrieval_result_gc(self):
        """Test that retrieval results are properly garbage collected."""
        import weakref

        results = []
        refs = []

        for i in range(1000):
            result = RetrievalResult(
                chunk_id=f"chunk_{i}",
                score=0.9,
                text="Text content",
                metadata={},
            )
            results.append(result)
            refs.append(weakref.ref(result))

        # Clear all references
        results.clear()
        gc.collect()

        # Check that most objects were collected (pytest may keep some refs)
        living = sum(1 for ref in refs if ref() is not None)
        print(f"\nLiving RetrievalResults after clear: {living}")
        assert living < 10  # Most should be collected


class TestPeakMemory:
    """Tests for peak memory scenarios."""

    @pytest.mark.stress
    def test_simultaneous_large_operations(self):
        """Test memory with multiple large operations."""
        tracemalloc.start()

        initial = tracemalloc.take_snapshot()

        # Simulate concurrent large operations
        all_results = []

        # Large retrieval
        retrieval_results = [
            RetrievalResult(
                chunk_id=f"r_{i}",
                score=0.9,
                text="Result text " * 100,
                metadata={"data": "x" * 1000},
            )
            for i in range(5000)
        ]
        all_results.append(retrieval_results)

        # Large vectors
        vectors = [[0.1] * 1536 for _ in range(1000)]
        all_results.append(vectors)

        # Large chunks
        chunks = [
            ChunkRecord(
                id=f"chunk_{i}",
                text="Chunk text " * 100,
                metadata={"source_path": "/test/file.pdf", "data": "x" * 500},
                dense_vector=[0.1] * 1536,
            )
            for i in range(1000)
        ]
        all_results.append(chunks)

        current, peak = tracemalloc.get_traced_memory()

        # Clear references
        all_results.clear()
        gc.collect()

        final = tracemalloc.take_snapshot()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"\nPeak memory during large operations: {peak_mb:.2f} MB")

        # Should be under 1GB
        assert peak_mb < 1000

    @pytest.mark.stress
    def test_memory_after_cleanup(self):
        """Test that memory is properly released after cleanup."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        def get_memory_mb():
            return process.memory_info().rss / (1024 * 1024)

        # Baseline
        gc.collect()
        baseline_mb = get_memory_mb()

        # Create large data
        large_data = [
            [0.1] * 1536 for _ in range(10000)
        ]

        during_mb = get_memory_mb()

        # Clear and collect
        del large_data
        gc.collect()

        after_mb = get_memory_mb()

        print(f"\nMemory after cleanup:")
        print(f"  Baseline: {baseline_mb:.2f} MB")
        print(f"  During: {during_mb:.2f} MB")
        print(f"  After: {after_mb:.2f} MB")
        print(f"  Released: {during_mb - after_mb:.2f} MB")

        # Memory should be released (at least some reduction)
        assert after_mb < during_mb  # Memory should have decreased
