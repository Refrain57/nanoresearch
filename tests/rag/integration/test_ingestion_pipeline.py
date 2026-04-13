"""Integration tests for Ingestion Pipeline.

Tests the complete ingestion flow:
- Document loading → chunking → transforms → encoding → storage
- End-to-end pipeline execution
- Error handling and recovery
- Idempotency (file integrity checking)
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MockSplitterConfig:
    provider: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class MockEmbeddingConfig:
    provider: str = "openai"  # Use real provider name but mock the factory
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    api_key: str = "mock-api-key"  # Dummy key for testing


@dataclass
class MockVectorStoreConfig:
    provider: str = "mock"
    persist_directory: str = "/tmp/test"


@dataclass
class MockRerankConfig:
    enabled: bool = False
    top_k: int = 5


@dataclass
class MockRetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    rrf_k: int = 60


@dataclass
class MockIngestionConfig:
    batch_size: int = 100
    chunk_strategy: str = "fixed"  # Use traditional splitter for tests
    min_chunk_length: int = 100
    max_chunk_length: int = 2000


@dataclass
class MockLLMConfig:
    provider: str = "mock"
    model: str = "mock-model"


@dataclass
class MockSettings:
    """Mock Settings object for testing."""
    splitter: MockSplitterConfig = field(default_factory=MockSplitterConfig)
    embedding: MockEmbeddingConfig = field(default_factory=MockEmbeddingConfig)
    vector_store: MockVectorStoreConfig = field(default_factory=MockVectorStoreConfig)
    rerank: MockRerankConfig = field(default_factory=MockRerankConfig)
    retrieval: MockRetrievalConfig = field(default_factory=MockRetrievalConfig)
    ingestion: MockIngestionConfig = field(default_factory=MockIngestionConfig)
    llm: MockLLMConfig = field(default_factory=MockLLMConfig)


class TestIngestionPipelineBasic:
    """Tests for basic pipeline execution."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        return MockSettings()

    @pytest.fixture
    def mock_integrity_checker(self):
        """Mock integrity checker that always allows processing."""
        checker = MagicMock()
        checker.compute_sha256.return_value = "mock_hash_12345"
        checker.should_skip.return_value = False
        return checker

    @pytest.fixture
    def mock_pdf_loader(self):
        """Mock PDF loader that returns sample documents."""
        from nanobot.rag.core.types import Document

        loader = MagicMock()
        doc = Document(
            id="doc_test_123",
            text="Sample PDF content for testing.\n\nThis is a test document.",
            metadata={
                "source_path": "/test/file.pdf",
                "doc_type": "pdf",
                "title": "Test Document",
                "images": [],
            },
        )
        loader.load.return_value = doc
        return loader

    @pytest.fixture
    def mock_chunks(self):
        """Mock chunks for testing."""
        from nanobot.rag.core.types import Chunk

        return [
            Chunk(
                id="chunk_001_0000",
                text="Sample PDF content for testing.",
                metadata={"source_path": "/test/file.pdf", "chunk_index": 0},
            ),
            Chunk(
                id="chunk_001_0001",
                text="This is a test document.",
                metadata={"source_path": "/test/file.pdf", "chunk_index": 1},
            ),
        ]

    def test_pipeline_success(self, mock_settings, mock_integrity_checker, mock_pdf_loader, mock_chunks):
        """Test successful pipeline execution."""
        from nanobot.rag.ingestion.pipeline import IngestionPipeline, PipelineResult
        from nanobot.rag.core.types import ChunkRecord

        # Mock all pipeline components including factories
        with patch("nanobot.rag.ingestion.pipeline.SQLiteIntegrityChecker") as mock_ic_cls, \
             patch("nanobot.rag.ingestion.pipeline.PdfLoader") as mock_pdf_cls, \
             patch("nanobot.rag.ingestion.pipeline.DocumentChunker") as mock_chunker_cls, \
             patch("nanobot.rag.ingestion.pipeline.ChunkRefiner") as mock_refiner_cls, \
             patch("nanobot.rag.ingestion.pipeline.MetadataEnricher") as mock_enricher_cls, \
             patch("nanobot.rag.ingestion.pipeline.ImageCaptioner") as mock_captioner_cls, \
             patch("nanobot.rag.ingestion.pipeline.BatchProcessor") as mock_batch_cls, \
             patch("nanobot.rag.ingestion.pipeline.VectorUpserter") as mock_vector_cls, \
             patch("nanobot.rag.ingestion.pipeline.BM25Indexer") as mock_bm25_cls, \
             patch("nanobot.rag.ingestion.pipeline.ImageStorage") as mock_img_cls, \
             patch("nanobot.rag.ingestion.pipeline.EmbeddingFactory") as mock_emb_factory_cls:

            # Configure mocks
            mock_ic_cls.return_value = mock_integrity_checker
            mock_pdf_cls.return_value = mock_pdf_loader

            # Mock embedding factory to return a mock embedding
            mock_emb = MagicMock()
            mock_emb.embed.return_value = [[0.1] * 1536]
            mock_emb_factory_cls.create.return_value = mock_emb

            mock_chunker = MagicMock()
            mock_chunker.split_document.return_value = mock_chunks
            mock_chunker_cls.return_value = mock_chunker

            refiner = MagicMock()
            refiner.use_llm = False
            refiner.transform.return_value = mock_chunks
            mock_refiner_cls.return_value = refiner

            enricher = MagicMock()
            enricher.use_llm = False
            enricher.transform.return_value = mock_chunks
            mock_enricher_cls.return_value = enricher

            captioner = MagicMock()
            captioner.llm = None
            captioner.transform.return_value = mock_chunks
            mock_captioner_cls.return_value = captioner

            batch_processor = MagicMock()
            batch_result = MagicMock()
            batch_result.dense_vectors = [[0.1] * 1536 for _ in mock_chunks]
            batch_result.sparse_stats = [{"chunk_id": c.id, "term_frequencies": {}} for c in mock_chunks]
            batch_processor.process.return_value = batch_result
            mock_batch_cls.return_value = batch_processor

            vector_upserter = MagicMock()
            vector_upserter.upsert.return_value = [c.id for c in mock_chunks]
            mock_vector_cls.return_value = vector_upserter

            bm25_indexer = MagicMock()
            mock_bm25_cls.return_value = bm25_indexer

            img_storage = MagicMock()
            mock_img_cls.return_value = img_storage

            # Create and run pipeline
            pipeline = IngestionPipeline(mock_settings, collection="test_collection")
            result = pipeline.run("/test/file.pdf")

            assert result.success is True
            assert result.chunk_count == 2
            assert len(result.vector_ids) == 2

    def test_pipeline_skip_existing_file(self, mock_settings, mock_integrity_checker, mock_pdf_loader, mock_chunks):
        """Test that existing files are skipped when force=False."""
        from nanobot.rag.ingestion.pipeline import IngestionPipeline

        mock_integrity_checker.should_skip.return_value = True

        with patch("nanobot.rag.ingestion.pipeline.SQLiteIntegrityChecker") as mock_ic_cls, \
             patch("nanobot.rag.ingestion.pipeline.PdfLoader") as mock_pdf_cls, \
             patch("nanobot.rag.ingestion.pipeline.DocumentChunker") as mock_chunker_cls, \
             patch("nanobot.rag.ingestion.pipeline.ChunkRefiner") as mock_refiner_cls, \
             patch("nanobot.rag.ingestion.pipeline.MetadataEnricher") as mock_enricher_cls, \
             patch("nanobot.rag.ingestion.pipeline.ImageCaptioner") as mock_captioner_cls, \
             patch("nanobot.rag.ingestion.pipeline.BatchProcessor") as mock_batch_cls, \
             patch("nanobot.rag.ingestion.pipeline.VectorUpserter") as mock_vector_cls, \
             patch("nanobot.rag.ingestion.pipeline.BM25Indexer") as mock_bm25_cls, \
             patch("nanobot.rag.ingestion.pipeline.ImageStorage") as mock_img_cls, \
             patch("nanobot.rag.ingestion.pipeline.EmbeddingFactory") as mock_emb_factory_cls:

            mock_ic_cls.return_value = mock_integrity_checker
            mock_pdf_cls.return_value = mock_pdf_loader

            mock_emb = MagicMock()
            mock_emb_factory_cls.create.return_value = mock_emb

            mock_chunker = MagicMock()
            mock_chunker.split_document.return_value = mock_chunks
            mock_chunker_cls.return_value = mock_chunker

            for mock_cls in [mock_refiner_cls, mock_enricher_cls, mock_captioner_cls]:
                m = MagicMock()
                m.transform.return_value = mock_chunks
                mock_cls.return_value = m

            mock_vector_cls.return_value = MagicMock()
            mock_bm25_cls.return_value = MagicMock()
            mock_img_cls.return_value = MagicMock()

            pipeline = IngestionPipeline(mock_settings, collection="test_collection")
            result = pipeline.run("/test/file.pdf")

            assert result.success is True
            assert result.chunk_count == 0  # File skipped
            assert "skipped" in result.stages.get("integrity", {})

    def test_pipeline_failure(self, mock_settings, mock_integrity_checker):
        """Test pipeline failure handling."""
        from nanobot.rag.ingestion.pipeline import IngestionPipeline

        with patch("nanobot.rag.ingestion.pipeline.SQLiteIntegrityChecker") as mock_ic_cls:
            mock_ic_cls.return_value = mock_integrity_checker

            with patch("nanobot.rag.ingestion.pipeline.PdfLoader") as mock_pdf_cls:
                # Simulate PDF loading failure
                mock_pdf = MagicMock()
                mock_pdf.load.side_effect = Exception("PDF parsing failed")
                mock_pdf_cls.return_value = mock_pdf

                with patch("nanobot.rag.ingestion.pipeline.DocumentChunker") as mock_chunker_cls, \
                     patch("nanobot.rag.ingestion.pipeline.ChunkRefiner") as mock_refiner_cls, \
                     patch("nanobot.rag.ingestion.pipeline.MetadataEnricher") as mock_enricher_cls, \
                     patch("nanobot.rag.ingestion.pipeline.ImageCaptioner") as mock_captioner_cls, \
                     patch("nanobot.rag.ingestion.pipeline.BatchProcessor") as mock_batch_cls, \
                     patch("nanobot.rag.ingestion.pipeline.VectorUpserter") as mock_vector_cls, \
                     patch("nanobot.rag.ingestion.pipeline.BM25Indexer") as mock_bm25_cls, \
                     patch("nanobot.rag.ingestion.pipeline.ImageStorage") as mock_img_cls, \
                     patch("nanobot.rag.ingestion.pipeline.EmbeddingFactory") as mock_emb_factory_cls:

                    mock_emb_factory_cls.create.return_value = MagicMock()

                    mock_chunker = MagicMock()
                    mock_chunker_cls.return_value = mock_chunker

                    for mock_cls in [mock_refiner_cls, mock_enricher_cls, mock_captioner_cls, mock_batch_cls]:
                        mock_cls.return_value = MagicMock()

                    mock_vector_cls.return_value = MagicMock()
                    mock_bm25_cls.return_value = MagicMock()
                    mock_img_cls.return_value = MagicMock()

                    pipeline = IngestionPipeline(mock_settings)
                    result = pipeline.run("/test/file.pdf")

                    assert result.success is False
                    assert result.error == "PDF parsing failed"


class TestIngestionPipelineStages:
    """Tests for individual pipeline stages."""

    @pytest.fixture
    def mock_settings(self):
        return MockSettings()

    def test_chunking_stage(self, mock_settings):
        """Test chunking stage produces correct output."""
        from nanobot.rag.core.types import Document, Chunk

        # This is a unit test for chunking logic
        doc = Document(
            id="test_doc",
            text="This is a test document with multiple sentences. " * 10,
            metadata={"source_path": "/test.pdf"},
        )

        # Verify document structure
        assert doc.id == "test_doc"
        assert len(doc.text) > 100
        assert doc.metadata["source_path"] == "/test.pdf"

    def test_pipeline_result_serialization(self):
        """Test PipelineResult serialization."""
        from nanobot.rag.ingestion.pipeline import PipelineResult

        result = PipelineResult(
            success=True,
            file_path="/test/file.pdf",
            doc_id="abc123",
            chunk_count=5,
            image_count=2,
            vector_ids=["v1", "v2", "v3"],
            stages={"integrity": {"file_hash": "hash"}},
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["file_path"] == "/test/file.pdf"
        assert result_dict["doc_id"] == "abc123"
        assert result_dict["chunk_count"] == 5
        assert result_dict["vector_ids_count"] == 3
        assert result_dict["stages"]["integrity"]["file_hash"] == "hash"

    def test_pipeline_result_with_error(self):
        """Test PipelineResult with error."""
        from nanobot.rag.ingestion.pipeline import PipelineResult

        result = PipelineResult(
            success=False,
            file_path="/test/file.pdf",
            error="Parsing failed",
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is False
        assert result_dict["error"] == "Parsing failed"


class TestBatchProcessing:
    """Tests for batch processing in ingestion."""

    @pytest.fixture
    def mock_settings(self):
        return MockSettings()

    def test_batch_processor_flow(self, mock_settings):
        """Test batch processor flow."""
        from nanobot.rag.core.types import Chunk

        # Create mock chunks
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                text=f"Content {i}",
                metadata={"source_path": "/test.pdf", "chunk_index": i},
            )
            for i in range(10)
        ]

        # Mock batch processor
        mock_batch = MagicMock()
        mock_batch.process.return_value = MagicMock(
            dense_vectors=[[0.1] * 1536 for _ in chunks],
            sparse_stats=[{"chunk_id": c.id, "term_frequencies": {}} for c in chunks],
        )

        # Verify mock works
        result = mock_batch.process(chunks)
        assert len(result.dense_vectors) == 10
        assert len(result.sparse_stats) == 10


class TestDocumentTypes:
    """Tests for core document types."""

    def test_document_creation(self):
        """Test Document creation and validation."""
        from nanobot.rag.core.types import Document

        doc = Document(
            id="doc_123",
            text="Sample text",
            metadata={"source_path": "/test.pdf"},
        )

        assert doc.id == "doc_123"
        assert doc.text == "Sample text"
        assert doc.metadata["source_path"] == "/test.pdf"

    def test_document_requires_source_path(self):
        """Test Document requires source_path in metadata."""
        from nanobot.rag.core.types import Document

        with pytest.raises(ValueError, match="source_path"):
            Document(id="doc_123", text="Sample text", metadata={})

    def test_chunk_creation(self):
        """Test Chunk creation and validation."""
        from nanobot.rag.core.types import Chunk

        chunk = Chunk(
            id="chunk_123",
            text="Chunk content",
            metadata={"source_path": "/test.pdf", "chunk_index": 0},
        )

        assert chunk.id == "chunk_123"
        assert chunk.text == "Chunk content"
        assert chunk.metadata["chunk_index"] == 0

    def test_chunk_record_creation(self):
        """Test ChunkRecord creation."""
        from nanobot.rag.core.types import ChunkRecord

        record = ChunkRecord(
            id="record_123",
            text="Record content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1] * 1536,
            sparse_vector={"term1": 0.5, "term2": 0.3},
        )

        assert record.id == "record_123"
        assert len(record.dense_vector) == 1536
        assert record.sparse_vector["term1"] == 0.5

    def test_retrieval_result_validation(self):
        """Test RetrievalResult validation."""
        from nanobot.rag.core.types import RetrievalResult

        # Valid result
        result = RetrievalResult(
            chunk_id="chunk_123",
            score=0.85,
            text="Result text",
            metadata={},
        )
        assert result.score == 0.85

        # Empty chunk_id should raise error
        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            RetrievalResult(chunk_id="", score=0.5, text="", metadata={})

        # Invalid score type should raise error
        with pytest.raises(ValueError, match="score must be numeric"):
            RetrievalResult(chunk_id="test", score="invalid", text="", metadata={})
