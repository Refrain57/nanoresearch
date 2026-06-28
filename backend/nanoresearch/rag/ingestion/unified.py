"""Unified document ingestion interface.

Single entry-point for ingesting documents into a knowledge base,
used by all three channels (Web UI worker, Agent MCP, CLI).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from nanoresearch.rag.core.settings import Settings
from nanoresearch.rag.ingestion.pipeline import IngestionPipeline
from nanoresearch.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
from nanoresearch.storage.models import KbChunk
from nanoresearch.storage.repositories.knowledge_repo import KnowledgeRepository


class KbNotFoundError(RuntimeError):
    """Raised when kb_id does not match any knowledge base."""


class IngestFailedError(RuntimeError):
    """Raised when pipeline processing fails after rollback."""

    def __init__(self, doc_id: str, cause: str) -> None:
        self.doc_id = doc_id
        self.cause = cause
        super().__init__(f"Ingestion failed for doc {doc_id}: {cause}")


@dataclass
class IngestResult:
    kb_id: str
    doc_id: str
    collection: str
    chunk_count: int
    status: Literal["created", "skipped_duplicate", "replaced"]
    duplicate_of: str | None = None


async def ingest_document(
    *,
    kb_id: str,
    file_path: str,
    original_filename: str,
    content_hash: str,
    pdf_parser: Literal["mineru", "marker", "markitdown"] = "mineru",
    chunk_strategy: str = "auto",
    force: bool = False,
    uid: str = "",
    metadata: dict[str, Any] | None = None,
    repo: KnowledgeRepository | None = None,
    settings: Settings | None = None,
    session_factory: Any = None,
) -> IngestResult:
    """Ingest a document into a knowledge base.

    This is the single entry-point for all three ingestion channels
    (Web UI worker, Agent MCP, CLI).  It handles path validation,
    deduplication, PG record management, pipeline execution, and
    chunk persistence with full rollback on pipeline failure.

    Args:
        kb_id: Knowledge base UUID (required).
        file_path: Absolute path to a permanently-stored file.
        original_filename: Display filename for UI/reporting.
        content_hash: SHA256 hex digest of file content (caller-computed).
        pdf_parser: PDF parser backend.
        chunk_strategy: Chunking strategy override.
        force: If True, re-process even when already indexed.
        uid: User ID (reserved for future use).
        metadata: Optional extra metadata (reserved for future use).
        repo: KnowledgeRepository instance (injected dependency).
        settings: Settings instance (injected dependency).
        session_factory: SQLAlchemy session factory (reserved).

    Returns:
        IngestResult summarizing the outcome.

    Raises:
        ValueError: If *file_path* is not a valid permanent path.
        KbNotFoundError: If *kb_id* does not match any knowledge base.
        RuntimeError: If the KB has no ``chroma_collection`` configured.
        IngestFailedError: If pipeline processing fails after rollback.
    """
    if repo is None:
        raise ValueError("repo is required")
    if settings is None:
        raise ValueError("settings is required")

    # ── Step 0: Path validation ──
    _validate_file_path(file_path)

    kb_uuid = uuid.UUID(kb_id)

    # ── Step A: Resolve KB ──
    kb = await repo.get(kb_uuid)
    if kb is None:
        raise KbNotFoundError(f"Knowledge base not found: {kb_id}")

    chroma_collection = kb.chroma_collection
    if not chroma_collection:
        raise RuntimeError(
            f"Knowledge base {kb_id} has no chroma_collection configured. "
            "Run the KB creation flow to fix this."
        )

    # ── Step B: Dedup ──
    existing = await repo.find_by_content_hash(kb_uuid, content_hash)

    if existing is not None and existing.status == "indexed" and not force:
        logger.info(
            "Duplicate document detected (kb={}, hash={}…), skipping",
            kb_id, content_hash[:12],
        )
        return IngestResult(
            kb_id=kb_id,
            doc_id=str(existing.id),
            collection=chroma_collection,
            chunk_count=existing.chunk_count or 0,
            status="skipped_duplicate",
            duplicate_of=str(existing.id),
        )

    # ── Step C: PG record ──
    # Statuses where ChromaDB is guaranteed empty (pipeline hasn't started writing yet).
    # "processing" is intentionally excluded: it means pipeline is running (or crashed
    # mid-run), so ChromaDB may already have partial chunks.  Treat it as an in-progress
    # guard rather than a safe "pre-created" state.
    _PRE_CREATED = {"uploaded", "parsing"}

    if existing is None:
        doc = await repo.create_document(
            kb_uuid,
            original_filename,
            file_path,
            content_hash=content_hash,
            pdf_parser=pdf_parser,
        )
        await repo.update_document_status(doc.id, "processing")
        doc_uuid = doc.id
        _result_status: Literal["created", "skipped_duplicate", "replaced"] = "created"
        logger.info("Created new document {}", doc_uuid)
    elif existing.status == "processing" and not force:
        # Pipeline is currently running (or crashed and left status stuck here).
        # Returning without running the pipeline is the correct idempotent behaviour.
        # To force a retry after a crash, the caller must pass force=True explicitly.
        logger.warning(
            "Document {} is already being processed (status=processing); ignoring duplicate call",
            existing.id,
        )
        return IngestResult(
            kb_id=kb_id,
            doc_id=str(existing.id),
            collection=chroma_collection,
            chunk_count=existing.chunk_count or 0,
            status="skipped_duplicate",
            duplicate_of=str(existing.id),
        )
    elif existing.status in _PRE_CREATED:
        # Pre-created by caller (e.g. Web UI router); pipeline never ran, nothing in ChromaDB.
        doc_uuid = existing.id
        if existing.file_path != file_path:
            await repo.update_document_file_path(doc_uuid, file_path)
        _result_status = "created"
        logger.info("Taking over pre-created document {}", doc_uuid)
    else:
        # Reprocessing: indexed+force=True, or status='reprocessing' from a crashed prior run.
        logger.info(
            "Reprocessing document {} (status={}, force={})",
            existing.id, existing.status, force,
        )
        await repo.delete_chunks_by_doc(existing.id)
        await repo.reset_for_reprocessing(existing.id, file_path, content_hash)
        old_source = existing.file_path or file_path
        vs = VectorStoreFactory.create(settings, collection_name=chroma_collection)
        if hasattr(vs, "delete_by_metadata"):
            vs.delete_by_metadata({"source_path": old_source})
        doc_uuid = existing.id
        _result_status = "replaced"

    # ── Step D: Pipeline ──
    pipeline = IngestionPipeline(
        settings,
        collection=chroma_collection,
        force=True,
        pdf_parser=pdf_parser,
        chunk_strategy_override=chunk_strategy,
    )
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipeline.run(file_path)
        )
    except Exception as exc:
        logger.error("Pipeline crashed for doc {}: {}", doc_uuid, exc)
        await repo.update_document_status(doc_uuid, "failed", error_msg=str(exc))
        vs = VectorStoreFactory.create(settings, collection_name=chroma_collection)
        if hasattr(vs, "delete_by_metadata"):
            vs.delete_by_metadata({"source_path": file_path})
        raise IngestFailedError(str(doc_uuid), str(exc)) from exc

    if not result.success:
        error_msg = result.error or "Unknown error"
        logger.error("Pipeline returned failure for doc {}: {}", doc_uuid, error_msg)
        await repo.update_document_status(doc_uuid, "failed", error_msg=error_msg)
        vs = VectorStoreFactory.create(settings, collection_name=chroma_collection)
        if hasattr(vs, "delete_by_metadata"):
            vs.delete_by_metadata({"source_path": file_path})
        raise IngestFailedError(str(doc_uuid), error_msg)

    # ── Step E: Write chunks & finalize ──
    chunk_rows: list[KbChunk] = []
    for idx, p in enumerate(result.chunk_payloads):
        meta = dict(p.metadata)
        meta["source_path"] = file_path
        meta["original_filename"] = original_filename
        chunk_rows.append(
            KbChunk(
                kb_id=kb_uuid,
                document_id=doc_uuid,
                chroma_id=p.chroma_id,
                chunk_index=idx,
                content=p.text,
                token_count=p.token_count,
                char_start=p.char_start,
                char_end=p.char_end,
                chunk_metadata=meta,
            )
        )

    if chunk_rows:
        await repo.create_chunks(chunk_rows)

    chunk_count = len(chunk_rows)
    new_status = _result_status
    await repo.update_document_status(doc_uuid, "indexed", chunk_count=chunk_count)
    await repo.increment_counts(kb_uuid, doc_delta=1, chunk_delta=chunk_count)

    logger.info(
        "Ingestion complete: doc={}, status={}, chunks={}",
        doc_uuid, new_status, chunk_count,
    )
    return IngestResult(
        kb_id=kb_id,
        doc_id=str(doc_uuid),
        collection=chroma_collection,
        chunk_count=chunk_count,
        status=new_status,
    )


def _validate_file_path(file_path: str) -> None:
    """Validate that *file_path* is absolute, exists, and is not a temp path."""
    if not os.path.isabs(file_path):
        raise ValueError(
            f"file_path must be an absolute path, got: {file_path}"
        )
    if not os.path.isfile(file_path):
        raise ValueError(
            f"file_path does not exist or is not a file: {file_path}"
        )
    tmpdir = tempfile.gettempdir()
    normalized = os.path.normcase(os.path.normpath(file_path))
    normalized_tmp = os.path.normcase(os.path.normpath(tmpdir))
    if normalized.startswith(normalized_tmp + os.sep) or normalized == normalized_tmp:
        raise ValueError(
            f"file_path must be a permanent path, not under a temp directory. "
            f"Got: {file_path}. Copy the file to a permanent location first."
        )
