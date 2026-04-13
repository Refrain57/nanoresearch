"""Stress tests for large document ingestion.

Tests the pipeline under load:
- Large files (> 100MB)
- Many chunks (> 10,000)
- Batch processing performance
- Memory usage
"""

import pytest
import time
import sys
from unittest.mock import MagicMock, patch
from typing import List

from nanobot.rag.core.types import Chunk, Document, ChunkRecord


class MockSettings:
    """Mock Settings object for testing."""
    def __init__(self):
        from dataclasses import dataclass

        @dataclass
        class MockIngestion:
            batch_size: int = 100
            chunk_strategy: str = "fixed"
            min_chunk_length: int = 100
            max_chunk_length: int = 2000

        @dataclass
        class MockEmbedding:
            provider: str = "openai"
            model: str = "text-embedding-3-small"
            dimension: int = 1536
            api_key: str = "mock-key"

        @dataclass
        class MockVectorStore:
            provider: str = "chroma"
            persist_directory: str = "/tmp/test"

        self.ingestion = MockIngestion()
        self.embedding = MockEmbedding()
        self.vector_store = MockVectorStore()


class TestLargeDocumentIngestion:
    """Tests for handling large documents."""

    @pytest.mark.stress
    def test_ingest_1000_chunks(self):
        """Test ingestion of 1000 chunks."""
        from nanobot.rag.ingestion.pipeline import IngestionPipeline

        settings = MockSettings()
        settings.ingestion.batch_size = 50

        # Create many mock chunks
        chunks = []
        for i in range(1000):
            chunk = MagicMock()
            chunk.id = f"chunk_{i:04d}"
            chunk.text = f"Content of chunk {i}. " * 50  # ~250 chars per chunk
            chunk.metadata = {
                "source_path": "/test/large.pdf",
                "chunk_index": i,
            }
            chunks.append(chunk)

        with patch("nanobot.rag.ingestion.pipeline.SQLiteIntegrityChecker") as mock_ic_cls, \
             patch("nanobot.rag.ingestion.pipeline.PdfLoader") as mock_pdf_cls, \
             patch("nanobot.rag.ingestion.pipeline.DocumentChunker") as mock_chunker_cls, \
             patch("nanobot.rag.ingestion.pipeline.ChunkRefiner") as mock_refiner_cls, \
             patch("nanobot.rag.ingestion.pipeline.MetadataEnricher") as mock_enricher_cls, \
             patch("nanobot.rag.ingestion.pipeline.ImageCaptioner") as mock_captioner_cls, \
             patch("nanobot.rag.ingestion.pipeline.BatchProcessor") as mock_batch_cls, \
             patch("nanobot.rag.ingestion.pipeline.VectorUpserter") as mock_vector_cls, \
             patch("nanobot.rag.ingestion.pipeline.BM25Indexer") as mock_bm25_cls, \
             patch("nanobot.rag.ingestion.pipeline.ImageStorage") as mock_img_cls:

            mock_ic_cls.return_value = MagicMock(
                compute_sha256=lambda x: "hash_1000",
                should_skip=lambda x: False,
                mark_success=MagicMock(),
            )

            doc = MagicMock()
            doc.id = "doc_large"
            doc.text = "Large document text" * 10000
            doc.metadata = {"source_path": "/test/large.pdf", "images": []}
            mock_pdf = MagicMock()
            mock_pdf.load.return_value = doc
            mock_pdf_cls.return_value = mock_pdf

            mock_chunker = MagicMock()
            mock_chunker.split_document.return_value = chunks
            mock_chunker_cls.return_value = mock_chunker

            for mock_cls in [mock_refiner_cls, mock_enricher_cls, mock_captioner_cls]:
                m = MagicMock()
                m.transform.return_value = chunks
                mock_cls.return_value = m

            # Mock batch processor
            batch_result = MagicMock()
            batch_result.dense_vectors = [[0.1] * 1536 for _ in chunks]
            batch_result.sparse_stats = [{"chunk_id": c.id} for c in chunks]

            batch_processor = MagicMock()
            batch_processor.process.return_value = batch_result
            mock_batch_cls.return_value = batch_processor

            vector_upserter = MagicMock()
            vector_upserter.upsert.return_value = [c.id for c in chunks]
            mock_vector_cls.return_value = vector_upserter

            mock_bm25_cls.return_value = MagicMock(add_documents=MagicMock())
            mock_img_cls.return_value = MagicMock(close=MagicMock())

            pipeline = IngestionPipeline(settings)

            start_time = time.time()
            result = pipeline.run("/test/large.pdf")
            elapsed = time.time() - start_time

            assert result.success is True
            assert result.chunk_count == 1000
            print(f"\nIngested 1000 chunks in {elapsed:.2f}s ({elapsed/1000:.3f}s per chunk)")

    @pytest.mark.stress
    def test_ingest_large_file_simulation(self):
        """Test ingestion of simulated large file content."""
        # Create a very large text (simulating 10MB of text)
        large_text = "Sample text content. " * 500000  # ~10MB

        # Test that chunking handles it
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

            # Mock splitter to simulate realistic chunking
            chunk_size = 500
            overlap = 50
            num_chunks = len(large_text) // (chunk_size - overlap)

            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = [
                large_text[i:i+chunk_size]
                for i in range(0, len(large_text), chunk_size - overlap)
            ][:num_chunks]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(MockSettings())

            doc = Document(
                id="doc_large",
                text=large_text,
                metadata={"source_path": "/test/large.pdf"},
            )

            start_time = time.time()
            chunks = chunker.split_document(doc)
            elapsed = time.time() - start_time

            assert len(chunks) > 1000  # Should produce many chunks
            print(f"\nChunked 10MB text into {len(chunks)} chunks in {elapsed:.2f}s")


class TestBatchProcessing:
    """Tests for batch processing performance."""

    @pytest.mark.stress
    def test_batch_processor_large_batch(self):
        """Test batch processor with large batch."""
        from nanobot.rag.ingestion.embedding.batch_processor import BatchProcessor

        settings = MockSettings()
        batch_size = 100

        # Mock encoders
        mock_dense = MagicMock()
        mock_dense.encode.return_value = [[0.1] * 1536] * batch_size

        mock_sparse = MagicMock()
        mock_sparse.encode.return_value = [{"term1": 0.5, "term2": 0.3}] * batch_size

        batch_processor = BatchProcessor(
            dense_encoder=mock_dense,
            sparse_encoder=mock_sparse,
            batch_size=batch_size,
        )

        chunks = [
            MagicMock(
                id=f"chunk_{i}",
                text=f"Content of chunk {i}",
                metadata={},
            )
            for i in range(500)
        ]

        start_time = time.time()
        result = batch_processor.process(chunks)
        elapsed = time.time() - start_time

        assert len(result.dense_vectors) == 500
        assert len(result.sparse_stats) == 500
        # Should be called 5 times (500 / 100)
        assert mock_dense.encode.call_count == 5
        print(f"\nProcessed 500 chunks in {elapsed:.2f}s ({elapsed/500:.3f}s per chunk)")

    @pytest.mark.stress
    def test_batch_processor_scaling(self):
        """Test batch processor scaling with different batch sizes."""
        from nanobot.rag.ingestion.embedding.batch_processor import BatchProcessor

        chunk_counts = [100, 500, 1000]
        batch_sizes = [50, 100, 200]

        for chunk_count in chunk_counts:
            for batch_size in batch_sizes:
                mock_dense = MagicMock()
                mock_dense.encode.return_value = [[0.1] * 1536] * batch_size

                mock_sparse = MagicMock()
                mock_sparse.encode.return_value = [{"term1": 0.5}] * batch_size

                processor = BatchProcessor(
                    dense_encoder=mock_dense,
                    sparse_encoder=mock_sparse,
                    batch_size=batch_size,
                )

                chunks = [MagicMock(id=f"c_{i}", text="text") for i in range(chunk_count)]

                start = time.time()
                result = processor.process(chunks)
                elapsed = time.time() - start

                expected_calls = (chunk_count + batch_size - 1) // batch_size
                print(f"\nChunks: {chunk_count}, Batch: {batch_size}, Time: {elapsed:.3f}s, Calls: {expected_calls}")


class TestMemoryUsage:
    """Tests for memory usage under load."""

    @pytest.mark.stress
    def test_chunking_memory_usage(self):
        """Test memory usage during chunking."""
        import tracemalloc
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        tracemalloc.start()

        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            # Simulate many small chunks
            chunks_text = ["Chunk content " * 100] * 5000  # 5000 chunks
            mock_splitter.split_text.return_value = chunks_text
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(MockSettings())

            doc = Document(
                id="doc_memory_test",
                text="Large text " * 1000000,
                metadata={"source_path": "/test.pdf"},
            )

            chunks = chunker.split_document(doc)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / (1024 * 1024)
            print(f"\nPeak memory for 5000 chunks: {peak_mb:.2f} MB")
            print(f"Per-chunk memory: {peak_mb/5000:.4f} MB")

            assert peak_mb < 500  # Should be under 500MB

    @pytest.mark.stress
    def test_vector_storage_memory(self):
        """Test memory usage with large vector storage."""
        import tracemalloc

        tracemalloc.start()

        # Simulate storing 10000 vectors
        dimension = 1536
        num_vectors = 10000
        vectors = [[0.1] * dimension for _ in range(num_vectors)]

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        expected_mb = (dimension * num_vectors * 8) / (1024 * 1024)  # 8 bytes per float64

        print(f"\nPeak memory for {num_vectors} vectors: {peak_mb:.2f} MB")
        print(f"Expected (raw): {expected_mb:.2f} MB")

        # Actual memory should be close to expected
        assert peak_mb > expected_mb * 0.5  # At least half the expected
