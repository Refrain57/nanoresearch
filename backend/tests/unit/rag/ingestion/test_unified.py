"""Unit tests for unified ingestion interface (unified.py).

Covers all 10 scenarios from the implementation plan:
path validation (3), KB not found, dedup skip, force reprocess,
processing residue, fresh ingest, pipeline failure rollback,
and empty chunks.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.rag.core.types import ChunkPayload
from nanobot.rag.ingestion.pipeline import PipelineResult
from nanobot.rag.ingestion.unified import (
    IngestFailedError,
    KbNotFoundError,
    _validate_file_path,
    ingest_document,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _mock_kb(chroma_collection: str = "test-collection"):
    kb = MagicMock()
    kb.chroma_collection = chroma_collection
    return kb


def _mock_doc(status: str = "indexed", file_path: str = "/old/doc.pdf",
              chunk_count: int = 5):
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.status = status
    doc.file_path = file_path
    doc.chunk_count = chunk_count
    return doc


def _make_payloads(n: int) -> list[ChunkPayload]:
    return [
        ChunkPayload(
            chroma_id=f"abc_{i:04d}_def",
            text=f"Chunk {i} text.",
            token_count=3,
            char_start=i * 100,
            char_end=(i + 1) * 100 - 1,
            metadata={"chunk_index": i},
        )
        for i in range(n)
    ]


def _setup_repo(kb=None, existing_doc=None):
    """Create a mock KnowledgeRepository with all async methods."""
    repo = MagicMock()
    repo.get = AsyncMock(return_value=kb)
    repo.find_by_content_hash = AsyncMock(return_value=existing_doc)
    repo.create_document = AsyncMock()
    repo.update_document_status = AsyncMock()
    repo.reset_for_reprocessing = AsyncMock()
    repo.delete_chunks_by_doc = AsyncMock()
    repo.create_chunks = AsyncMock()
    repo.increment_counts = AsyncMock()
    return repo


# ---------------------------------------------------------------------------
# Tests 1-3: Path validation (sync, test _validate_file_path directly)
# ---------------------------------------------------------------------------

def test_path_not_absolute_raises():
    with pytest.raises(ValueError, match="must be an absolute path"):
        _validate_file_path("relative/path/to/file.pdf")


def test_path_in_temp_raises():
    fd, tmp_path = tempfile.mkstemp()
    try:
        os.close(fd)
        with pytest.raises(ValueError, match="permanent path"):
            _validate_file_path(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_path_not_exists_raises():
    with pytest.raises(ValueError, match="does not exist"):
        _validate_file_path("/nonexistent/path/to/file.pdf")


# ---------------------------------------------------------------------------
# Tests 4-10: Flow logic (async, mock all dependencies)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kb_not_found_raises():
    repo = _setup_repo(kb=None)  # get() returns None
    settings = MagicMock()

    with patch("nanobot.rag.ingestion.unified._validate_file_path"):
        with pytest.raises(KbNotFoundError, match="not found"):
            await ingest_document(
                kb_id="00000000-0000-0000-0000-000000000001",
                file_path="/valid/file.pdf",
                original_filename="file.pdf",
                content_hash="abc123",
                repo=repo,
                settings=settings,
            )


@pytest.mark.asyncio
async def test_dedup_skips():
    kb = _mock_kb()
    existing = _mock_doc(status="indexed", chunk_count=5)
    repo = _setup_repo(kb=kb, existing_doc=existing)
    settings = MagicMock()

    with patch("nanobot.rag.ingestion.unified._validate_file_path"):
        result = await ingest_document(
            kb_id=str(uuid.uuid4()),
            file_path="/valid/file.pdf",
            original_filename="file.pdf",
            content_hash="abc123",
            force=False,
            repo=repo,
            settings=settings,
        )

    assert result.status == "skipped_duplicate"
    assert result.duplicate_of == str(existing.id)
    assert result.chunk_count == 5
    repo.create_document.assert_not_called()


@pytest.mark.asyncio
@patch("nanobot.rag.ingestion.unified.VectorStoreFactory")
@patch("nanobot.rag.ingestion.unified.IngestionPipeline")
async def test_force_reprocess(mock_pipeline_class, mock_vs_factory):
    kb = _mock_kb()
    existing = _mock_doc(status="indexed", file_path="/old/doc.pdf")
    repo = _setup_repo(kb=kb, existing_doc=existing)
    settings = MagicMock()
    payloads = _make_payloads(3)

    # Pipeline mock
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = PipelineResult(
        True, "/valid/file.pdf", chunk_count=3, chunk_payloads=payloads,
    )
    mock_pipeline_class.return_value = mock_pipeline

    # VS mock
    mock_vs = MagicMock()
    mock_vs.delete_by_metadata = MagicMock(return_value=5)
    mock_vs_factory.create.return_value = mock_vs

    with patch("nanobot.rag.ingestion.unified._validate_file_path"):
        result = await ingest_document(
            kb_id=str(uuid.uuid4()),
            file_path="/valid/file.pdf",
            original_filename="file.pdf",
            content_hash="abc123",
            force=True,
            repo=repo,
            settings=settings,
        )

    assert result.status == "replaced"
    assert result.chunk_count == 3
    repo.reset_for_reprocessing.assert_called_once()
    repo.delete_chunks_by_doc.assert_called_once_with(existing.id)
    mock_vs.delete_by_metadata.assert_called_once_with({"source_path": "/old/doc.pdf"})
    repo.create_chunks.assert_called_once()
    repo.increment_counts.assert_called_once()
    mock_pipeline_class.assert_called_once()


@pytest.mark.asyncio
@patch("nanobot.rag.ingestion.unified.VectorStoreFactory")
@patch("nanobot.rag.ingestion.unified.IngestionPipeline")
async def test_processing_residue_retries(mock_pipeline_class, mock_vs_factory):
    kb = _mock_kb()
    existing = _mock_doc(status="processing", file_path="/crashed/doc.pdf")
    repo = _setup_repo(kb=kb, existing_doc=existing)
    settings = MagicMock()
    payloads = _make_payloads(2)

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = PipelineResult(
        True, "/valid/file.pdf", chunk_count=2, chunk_payloads=payloads,
    )
    mock_pipeline_class.return_value = mock_pipeline

    mock_vs = MagicMock()
    mock_vs.delete_by_metadata = MagicMock(return_value=3)
    mock_vs_factory.create.return_value = mock_vs

    with patch("nanobot.rag.ingestion.unified._validate_file_path"):
        result = await ingest_document(
            kb_id=str(uuid.uuid4()),
            file_path="/valid/file.pdf",
            original_filename="file.pdf",
            content_hash="abc123",
            force=False,
            repo=repo,
            settings=settings,
        )

    assert result.status == "replaced"
    repo.reset_for_reprocessing.assert_called_once()
    repo.delete_chunks_by_doc.assert_called_once_with(existing.id)


@pytest.mark.asyncio
@patch("nanobot.rag.ingestion.unified.VectorStoreFactory")
@patch("nanobot.rag.ingestion.unified.IngestionPipeline")
async def test_fresh_ingest_success(mock_pipeline_class, mock_vs_factory):
    kb = _mock_kb()
    repo = _setup_repo(kb=kb, existing_doc=None)
    settings = MagicMock()
    payloads = _make_payloads(3)

    new_doc = MagicMock()
    new_doc.id = uuid.uuid4()
    repo.create_document.return_value = new_doc

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = PipelineResult(
        True, "/valid/file.pdf", chunk_count=3, chunk_payloads=payloads,
    )
    mock_pipeline_class.return_value = mock_pipeline

    mock_vs = MagicMock()
    mock_vs_factory.create.return_value = mock_vs

    with patch("nanobot.rag.ingestion.unified._validate_file_path"):
        result = await ingest_document(
            kb_id=str(uuid.uuid4()),
            file_path="/valid/file.pdf",
            original_filename="report.pdf",
            content_hash="abc123",
            repo=repo,
            settings=settings,
        )

    assert result.status == "created"
    assert result.chunk_count == 3

    # Verify document creation
    repo.create_document.assert_called_once()
    args, kwargs = repo.create_document.call_args
    assert args[1] == "report.pdf"  # filename is 2nd positional arg
    assert kwargs["content_hash"] == "abc123"

    # Verify chunks written
    repo.create_chunks.assert_called_once()
    chunks_arg = repo.create_chunks.call_args[0][0]
    assert len(chunks_arg) == 3
    assert chunks_arg[0].chunk_index == 0
    assert chunks_arg[0].chunk_metadata["source_path"] == "/valid/file.pdf"
    assert chunks_arg[0].chunk_metadata["original_filename"] == "report.pdf"

    # Verify finalization
    repo.update_document_status.assert_called()
    repo.increment_counts.assert_called_once()
    inc_kwargs = repo.increment_counts.call_args.kwargs
    assert inc_kwargs["doc_delta"] == 1
    assert inc_kwargs["chunk_delta"] == 3


@pytest.mark.asyncio
@patch("nanobot.rag.ingestion.unified.VectorStoreFactory")
@patch("nanobot.rag.ingestion.unified.IngestionPipeline")
async def test_pipeline_failure_rollback(mock_pipeline_class, mock_vs_factory):
    kb = _mock_kb()
    repo = _setup_repo(kb=kb, existing_doc=None)
    settings = MagicMock()

    new_doc = MagicMock()
    new_doc.id = uuid.uuid4()
    repo.create_document.return_value = new_doc

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = PipelineResult(
        False, "/valid/file.pdf", error="Embedding API timeout",
    )
    mock_pipeline_class.return_value = mock_pipeline

    mock_vs = MagicMock()
    mock_vs.delete_by_metadata = MagicMock(return_value=0)
    mock_vs_factory.create.return_value = mock_vs

    with patch("nanobot.rag.ingestion.unified._validate_file_path"):
        with pytest.raises(IngestFailedError, match="Embedding API timeout"):
            await ingest_document(
                kb_id=str(uuid.uuid4()),
                file_path="/valid/file.pdf",
                original_filename="file.pdf",
                content_hash="abc123",
                repo=repo,
                settings=settings,
            )

    # PG marked as failed
    assert any(
        call[0][1] == "failed" for call in repo.update_document_status.call_args_list
    )

    # ChromaDB cleaned
    mock_vs.delete_by_metadata.assert_called_with({"source_path": "/valid/file.pdf"})

    # No chunks written
    repo.create_chunks.assert_not_called()


@pytest.mark.asyncio
@patch("nanobot.rag.ingestion.unified.VectorStoreFactory")
@patch("nanobot.rag.ingestion.unified.IngestionPipeline")
async def test_empty_chunks(mock_pipeline_class, mock_vs_factory):
    kb = _mock_kb()
    repo = _setup_repo(kb=kb, existing_doc=None)
    settings = MagicMock()

    new_doc = MagicMock()
    new_doc.id = uuid.uuid4()
    repo.create_document.return_value = new_doc

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = PipelineResult(
        True, "/valid/file.pdf", chunk_count=0, chunk_payloads=[],
    )
    mock_pipeline_class.return_value = mock_pipeline

    mock_vs = MagicMock()
    mock_vs_factory.create.return_value = mock_vs

    with patch("nanobot.rag.ingestion.unified._validate_file_path"):
        result = await ingest_document(
            kb_id=str(uuid.uuid4()),
            file_path="/valid/file.pdf",
            original_filename="empty.pdf",
            content_hash="abc123",
            repo=repo,
            settings=settings,
        )

    assert result.status == "created"
    assert result.chunk_count == 0
    # create_chunks should not be called when there are zero chunks
    repo.create_chunks.assert_not_called()
    repo.increment_counts.assert_called_once()
    inc_kwargs = repo.increment_counts.call_args.kwargs
    assert inc_kwargs["chunk_delta"] == 0
