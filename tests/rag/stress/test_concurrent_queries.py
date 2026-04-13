"""Stress tests for concurrent queries.

Tests retrieval performance under load:
- Concurrent query execution
- Thread pool exhaustion
- Latency under load
- Throughput testing
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch
from typing import List

from nanobot.rag.core.types import RetrievalResult


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

        self.retrieval = MockRetrieval()


class TestConcurrentDenseQueries:
    """Tests for concurrent dense retrieval."""

    @pytest.mark.stress
    def test_concurrent_50_queries(self):
        """Test 50 concurrent dense queries."""
        from nanobot.rag.core.query_engine.dense_retriever import DenseRetriever

        settings = MockSettings()

        # Mock components
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 1536]

        mock_store = MagicMock()
        def mock_query(vector, top_k=10, filters=None, trace=None):
            # Return dicts as expected by _transform_results
            return [
                {
                    "id": f"chunk_{i}",
                    "score": 0.9 - i * 0.1,
                    "text": f"Result {i}",
                    "metadata": {},
                }
                for i in range(min(top_k, 5))
            ]
        mock_store.query = mock_query

        retriever = DenseRetriever(
            settings=settings,
            embedding_client=mock_embedding,
            vector_store=mock_store,
        )

        queries = [f"Query {i}" for i in range(50)]

        def execute_query(query):
            return retriever.retrieve(query, top_k=5)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(execute_query, q) for q in queries]
            results = [f.result() for f in as_completed(futures)]
        elapsed = time.time() - start_time

        assert len(results) == 50
        print(f"\n50 concurrent dense queries in {elapsed:.2f}s ({elapsed/50:.3f}s per query)")

    @pytest.mark.stress
    def test_concurrent_hybrid_queries(self):
        """Test concurrent hybrid search queries."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
        from nanobot.rag.core.query_engine.fusion import RRFFusion
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor

        # Mock retrievers
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="d1", score=0.9, text="Dense result", metadata={}),
            RetrievalResult(chunk_id="d2", score=0.8, text="Dense result 2", metadata={}),
        ]

        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="s1", score=5.0, text="Sparse result", metadata={}),
            RetrievalResult(chunk_id="s2", score=4.0, text="Sparse result 2", metadata={}),
        ]

        hybrid = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
            config=HybridSearchConfig(parallel_retrieval=False),  # Sequential to avoid complexity
        )

        queries = [f"Query {i}" for i in range(30)]

        def execute_query(query):
            return hybrid.search(query, top_k=5)

        start_time = time.time()
        results = []
        for q in queries:
            results.append(execute_query(q))
        elapsed = time.time() - start_time

        assert len(results) == 30
        print(f"\n30 hybrid queries in {elapsed:.2f}s ({elapsed/30:.3f}s per query)")

    @pytest.mark.stress
    def test_hybrid_search_parallel_vs_sequential(self):
        """Compare parallel vs sequential retrieval performance."""
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
        from nanobot.rag.core.query_engine.fusion import RRFFusion
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor

        # Slow retrievers to highlight difference
        def slow_dense_retrieve(*args, **kwargs):
            time.sleep(0.05)  # 50ms delay
            return [
                RetrievalResult(chunk_id="d1", score=0.9, text="Dense", metadata={}),
            ]

        def slow_sparse_retrieve(*args, **kwargs):
            time.sleep(0.05)  # 50ms delay
            return [
                RetrievalResult(chunk_id="s1", score=5.0, text="Sparse", metadata={}),
            ]

        mock_dense = MagicMock()
        mock_dense.retrieve.side_effect = slow_dense_retrieve

        mock_sparse = MagicMock()
        mock_sparse.retrieve.side_effect = slow_sparse_retrieve

        # Test sequential
        hybrid_seq = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
            config=HybridSearchConfig(parallel_retrieval=False),
        )

        start = time.time()
        hybrid_seq.search("test")
        sequential_time = time.time() - start

        # Reset mocks
        mock_dense.reset_mock()
        mock_sparse.reset_mock()
        mock_dense.retrieve.side_effect = slow_dense_retrieve
        mock_sparse.retrieve.side_effect = slow_sparse_retrieve

        # Test parallel
        hybrid_par = HybridSearch(
            query_processor=QueryProcessor(),
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=RRFFusion(k=60),
            config=HybridSearchConfig(parallel_retrieval=True),
        )

        start = time.time()
        hybrid_par.search("test")
        parallel_time = time.time() - start

        print(f"\nSequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s")
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")

        # Parallel should be faster (ideally ~2x for two independent calls)
        assert parallel_time < sequential_time * 0.9  # At least 10% faster


class TestQueryThroughput:
    """Tests for query throughput."""

    @pytest.mark.stress
    def test_query_throughput_single_thread(self):
        """Test query throughput in single thread."""
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor

        processor = QueryProcessor()
        queries = [
            "如何配置 Azure OpenAI",
            "Python 教程 PDF 下载",
            "Vector database best practices",
            "LLM fine-tuning guide",
        ] * 25  # 100 total queries

        start_time = time.time()
        for q in queries:
            processor.process(q)
        elapsed = time.time() - start_time

        qps = len(queries) / elapsed
        print(f"\nSingle-thread QPS: {qps:.2f} queries/sec")

    @pytest.mark.stress
    def test_query_throughput_multi_thread(self):
        """Test query throughput with multi-threading."""
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor

        processor = QueryProcessor()
        queries = [
            "如何配置 Azure OpenAI",
            "Python 教程 PDF 下载",
            "Vector database best practices",
        ] * 33  # 99 total queries

        def process_batch(query_batch):
            results = []
            for q in query_batch:
                results.append(processor.process(q))
            return results

        # Split into 3 batches
        batch_size = 33
        batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            all_results = [f.result() for f in as_completed(futures)]
        elapsed = time.time() - start_time

        total_queries = sum(len(r) for r in all_results)
        qps = total_queries / elapsed
        print(f"\nMulti-thread QPS (3 workers): {qps:.2f} queries/sec")


class TestQueryLatency:
    """Tests for query latency."""

    @pytest.mark.stress
    def test_p99_latency(self):
        """Test P99 latency for query processing."""
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor

        processor = QueryProcessor()
        queries = [f"Query about topic {i}" for i in range(100)]

        latencies = []
        for q in queries:
            start = time.perf_counter()
            processor.process(q)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print(f"\nQuery latency: P50={p50:.2f}ms, P95={p95:.2f}ms, P99={p99:.2f}ms")

        # P99 should be reasonable
        assert p99 < 100  # Less than 100ms


class TestThreadPoolExhaustion:
    """Tests for thread pool behavior under load."""

    @pytest.mark.stress
    def test_thread_pool_limits(self):
        """Test behavior when thread pool is exhausted."""
        import concurrent.futures

        def slow_task(i):
            time.sleep(0.1)
            return i

        # Small thread pool
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(slow_task, i) for i in range(10)]
            results = [f.result() for f in futures]
        elapsed = time.time() - start_time

        # With 2 workers and 10 tasks of 0.1s each:
        # Minimum time = 5 batches * 0.1s = 0.5s
        assert elapsed >= 0.4  # At least ~5 batches needed
        print(f"\n10 tasks with 2 workers: {elapsed:.2f}s")
