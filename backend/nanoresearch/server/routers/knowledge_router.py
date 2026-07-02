"""Knowledge base, document, chunk, and retrieval test endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nanoresearch.config.loader import get_mode
from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.storage.models import KbChunk
from nanoresearch.storage.repositories.knowledge_repo import KnowledgeRepository

router = APIRouter()


def _kb_repo(request: Request) -> KnowledgeRepository:
    return KnowledgeRepository(request.app.state.session_factory)


def _rag_settings(request: Request):
    settings = getattr(request.app.state, "rag_settings", None)
    if settings is None:
        from nanoresearch.rag.core.settings import load_settings
        settings = load_settings()
        request.app.state.rag_settings = settings
    return settings


async def _resolve_rag_settings(uid: str, request: Request, role=None):
    """Unified replacement for _overlay_user_keys and _entity_extract_settings."""
    from nanoresearch.providers.model_factory import ModelFactory, ModelRole
    from nanoresearch.storage.repositories.user_settings_repo import UserSettingsRepository

    if role is None:
        role = ModelRole.INGESTION_LLM

    base = _rag_settings(request)
    config = getattr(request.app.state, "config", None)
    try:
        user_cfg = await UserSettingsRepository(request.app.state.session_factory).get(uid)
        user_model = user_cfg.model if user_cfg else None
        user_providers = (user_cfg.extra or {}).get("providers", []) if user_cfg else []
        user_roles = (user_cfg.extra or {}).get("roles") if user_cfg else None
    except Exception:
        user_model, user_providers, user_roles = None, [], None

    spec = ModelFactory.resolve(
        role,
        config=config,
        rag_settings=base,
        user_model=user_model,
        user_providers=user_providers,
        user_roles=user_roles,
        mode=get_mode(),
    )
    return ModelFactory.patch_settings(base, role, spec)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class KbCreate(BaseModel):
    name: str
    description: str | None = None
    embedding_model: str | None = None
    chunk_strategy: str = "auto"


class KbUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    embedding_model: str | None = None
    enable_graph_expansion: bool | None = None


class QueryTestRequest(BaseModel):
    query: str
    top_k: int = 5
    enable_dense: bool = True
    enable_sparse: bool = True


# ---------------------------------------------------------------------------
# Knowledge Base CRUD
# ---------------------------------------------------------------------------

@router.get("/api/knowledge")
async def list_knowledge(request: Request, uid: str = Depends(get_current_user)):
    kbs = await _kb_repo(request).list_by_uid(uid)
    return [_kb_to_dict(kb) for kb in kbs]


@router.post("/api/knowledge", status_code=201)
async def create_knowledge(request: Request, body: KbCreate, uid: str = Depends(get_current_user)):
    kb = await _kb_repo(request).create(uid, body.name, body.description, body.embedding_model, chunk_strategy=body.chunk_strategy)
    chroma_collection = f"{uid}_{kb.id}"
    kb = await _kb_repo(request).update(kb.id, chroma_collection=chroma_collection)
    return _kb_to_dict(kb)


@router.get("/api/knowledge/{kb_id}")
async def get_knowledge(kb_id: str, request: Request, uid: str = Depends(get_current_user)):
    kb = await _get_kb_or_404(kb_id, uid, request)
    return _kb_to_dict(kb)


@router.put("/api/knowledge/{kb_id}")
async def update_knowledge(kb_id: str, request: Request, body: KbUpdate, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    kb = await _kb_repo(request).update(uuid.UUID(kb_id), **fields)
    return _kb_to_dict(kb)


@router.delete("/api/knowledge/{kb_id}", status_code=204)
async def delete_knowledge(kb_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    await _kb_repo(request).delete(uuid.UUID(kb_id))


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@router.get("/api/knowledge/{kb_id}/documents")
async def list_documents(kb_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _kb_repo(request)
    kb_uuid = uuid.UUID(kb_id)
    docs, img_counts = await asyncio.gather(
        repo.list_documents(kb_uuid),
        repo.count_images_per_doc(kb_uuid),
    )
    return [_doc_to_dict(d, image_count=img_counts.get(d.id, 0)) for d in docs]


@router.post("/api/knowledge/{kb_id}/documents", status_code=202)
async def upload_document(
    kb_id: str,
    request: Request,
    file: UploadFile = File(...),
    pdf_parser: str = Form("mineru"),
    uid: str = Depends(get_current_user),
):
    kb = await _get_kb_or_404(kb_id, uid, request)
    repo = _kb_repo(request)

    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    # Dedup: if an indexed copy already exists, skip without creating a new record.
    existing = await repo.find_by_content_hash(uuid.UUID(kb_id), content_hash)
    if existing is not None and existing.status == "indexed":
        return JSONResponse(
            status_code=200,
            content={"status": "skipped_duplicate", **_doc_to_dict(existing)},
        )

    suffix = os.path.splitext(file.filename or "upload")[1] or ".bin"

    # Save to temp file; background task takes ownership and deletes it
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    tmp_path = tmp.name

    doc = await repo.create_document(
        kb_id=uuid.UUID(kb_id),
        filename=file.filename or "upload",
        file_path=tmp_path,
        file_size=len(content),
        mime_type=file.content_type,
        pdf_parser=pdf_parser,
        content_hash=content_hash,
    )

    await request.app.state.arq_pool.enqueue_job(
        "ingest_document_task",
        kb_id=kb_id,
        doc_id=str(doc.id),
        file_path=tmp_path,
        chroma_collection=kb.chroma_collection or kb_id,
        original_filename=file.filename or "upload",
        chunk_strategy=kb.chunk_strategy or "auto",
        pdf_parser=pdf_parser,
        uid=uid,
        content_hash=content_hash,
    )
    return _doc_to_dict(doc)


@router.delete("/api/knowledge/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_document(
    kb_id: str,
    doc_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    kb = await _get_kb_or_404(kb_id, uid, request)
    repo = _kb_repo(request)
    doc = await repo.get_document(uuid.UUID(doc_id))
    if doc is None or str(doc.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")

    doc_delta = -1 if doc.status == "indexed" else 0
    chunk_delta = -(doc.chunk_count or 0)
    await repo.delete_chunks_by_doc(uuid.UUID(doc_id))
    await repo.delete_document(uuid.UUID(doc_id))
    await repo.increment_counts(uuid.UUID(kb_id), doc_delta=doc_delta, chunk_delta=chunk_delta)

    if doc.file_path and os.path.exists(doc.file_path):
        try:
            os.unlink(doc.file_path)
        except Exception:
            pass

    # Best-effort: delete from ChromaDB
    try:
        settings = _rag_settings(request)
        from nanoresearch.rag.ingestion.document_manager import DocumentManager
        from nanoresearch.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
        from nanoresearch.rag.ingestion.storage.bm25_indexer import BM25Indexer
        from nanoresearch.rag.ingestion.storage.image_storage import ImageStorage
        from nanoresearch.rag.libs.loader.file_integrity import SQLiteIntegrityChecker
        from nanoresearch.rag.core.settings import resolve_path

        chroma_col = kb.chroma_collection or kb_id
        chroma = VectorStoreFactory.create(settings, collection_name=chroma_col)
        bm25 = BM25Indexer(index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{chroma_col}")))
        img_storage = ImageStorage(
            db_path=str(resolve_path("~/.nanoresearch/rag/images.db")),
            images_root=str(resolve_path("~/.nanoresearch/rag/images")),
        )
        integrity = SQLiteIntegrityChecker(db_path=str(resolve_path("~/.nanoresearch/rag/ingestion_history.db")))
        mgr = DocumentManager(chroma, bm25, img_storage, integrity)
        if doc.file_path:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: mgr.delete_document(doc.file_path, collection=chroma_col)
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# RAG image proxy — short-lived ticket auth
# ---------------------------------------------------------------------------

import secrets as _secrets
import time as _time
from threading import Lock as _Lock

_image_tickets: dict[str, tuple[str, float]] = {}  # ticket -> (uid, expires_at)
_tickets_lock = _Lock()


def _issue_image_ticket(uid: str) -> str:
    ticket = _secrets.token_urlsafe(32)
    expires_at = _time.time() + 60
    with _tickets_lock:
        now = _time.time()
        expired = [k for k, (_, exp) in _image_tickets.items() if exp < now]
        for k in expired:
            del _image_tickets[k]
        _image_tickets[ticket] = (uid, expires_at)
    return ticket


def _validate_image_ticket(ticket: str) -> str:
    """Validate ticket and return the associated uid."""
    with _tickets_lock:
        entry = _image_tickets.get(ticket)
        if not entry:
            raise HTTPException(status_code=401, detail="Invalid or expired ticket")
        uid, expires_at = entry
        if _time.time() > expires_at:
            del _image_tickets[ticket]
            raise HTTPException(status_code=401, detail="Ticket expired")
        return uid


@router.post("/api/rag/image-ticket")
async def create_image_ticket(uid: str = Depends(get_current_user)):
    """Issue a 60-second image access ticket. Frontend uses this instead of JWT in img URLs."""
    return {"ticket": _issue_image_ticket(uid)}


@router.get("/api/rag/images")
async def serve_rag_image(path: str, ticket: str, request: Request):
    from fastapi.responses import FileResponse
    import mimetypes
    from pathlib import Path as _Path
    from sqlalchemy import select as _select

    uid = _validate_image_ticket(ticket)

    base_dir = _Path(os.path.expanduser("~/.nanoresearch/rag/images")).resolve()
    try:
        candidate = _Path(path.replace("\\", "/")).resolve()
        rel = candidate.relative_to(base_dir)  # raises ValueError if outside
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    # rel.parts[0] is the collection name, which is "{uid}_{kb_id}"
    # Verify in DB that this KB belongs to the requesting user.
    collection_name = rel.parts[0] if rel.parts else ""
    if not collection_name:
        raise HTTPException(status_code=403, detail="Access denied")

    from nanoresearch.storage.models import KnowledgeBase as _KB
    async with request.app.state.session_factory() as db:
        result = await db.execute(
            _select(_KB).where(_KB.chroma_collection == collection_name)
        )
        kb = result.scalar_one_or_none()

    if kb is None or kb.uid != uid:
        raise HTTPException(status_code=403, detail="Access denied")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    media_type, _ = mimetypes.guess_type(str(candidate))
    return FileResponse(str(candidate), media_type=media_type or "image/jpeg")


# ---------------------------------------------------------------------------
# Document file download
# ---------------------------------------------------------------------------

@router.get("/api/knowledge/{kb_id}/documents/{doc_id}/file")
async def get_document_file(
    kb_id: str,
    doc_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    from fastapi.responses import FileResponse
    import mimetypes
    await _get_kb_or_404(kb_id, uid, request)
    doc = await _kb_repo(request).get_document(uuid.UUID(doc_id))
    if doc is None or str(doc.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="源文件不存在（文档需重新上传以启用预览）")
    media_type = doc.mime_type
    if not media_type:
        guessed, _ = mimetypes.guess_type(doc.filename or "")
        media_type = guessed or "application/octet-stream"
    from urllib.parse import quote
    encoded_name = quote(doc.filename or "", safe="")
    return FileResponse(
        doc.file_path,
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{encoded_name}"},
    )


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

@router.get("/api/knowledge/{kb_id}/chunks")
async def list_chunks(
    kb_id: str,
    request: Request,
    limit: int = 50,
    offset: int = 0,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    chunks = await _kb_repo(request).list_chunks_by_kb(uuid.UUID(kb_id), limit=limit, offset=offset)
    return [_chunk_to_dict(c) for c in chunks]


@router.get("/api/knowledge/{kb_id}/documents/{doc_id}/chunks")
async def list_document_chunks(
    kb_id: str,
    doc_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    chunks = await _kb_repo(request).list_chunks_by_doc(uuid.UUID(doc_id))
    return [_chunk_to_dict(c) for c in chunks]


# ---------------------------------------------------------------------------
# Test retrieval
# ---------------------------------------------------------------------------

@router.post("/api/knowledge/{kb_id}/query/test")
async def test_query(
    kb_id: str,
    request: Request,
    body: QueryTestRequest,
    uid: str = Depends(get_current_user),
):
    kb = await _get_kb_or_404(kb_id, uid, request)
    settings = _rag_settings(request)

    try:
        hybrid = _build_hybrid_search(request, kb, settings, body.top_k)

        result = await hybrid.async_search(body.query, top_k=body.top_k, return_details=True)

        # Look up full chunk content from PG using chroma_ids
        chroma_ids = [r.chunk_id for r in result.results]
        pg_chunks = await _kb_repo(request).get_chunks_by_chroma_ids(chroma_ids)
        chunk_map = {c.chroma_id: c for c in pg_chunks}

        return {
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "dense_score": r.dense_score,
                    "sparse_score": r.sparse_score,
                    "text": chunk_map[r.chunk_id].content if r.chunk_id in chunk_map else r.text,
                    "metadata": chunk_map[r.chunk_id].chunk_metadata if r.chunk_id in chunk_map else r.metadata,
                    "graph_via_entity": r.metadata.get("graph_via_entity"),
                    "is_graph_neighbor": r.metadata.get("is_graph_neighbor", False),
                }
                for r in result.results
            ],
            "used_fallback": result.used_fallback,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")


# ---------------------------------------------------------------------------
# RAG retrieval helpers
# ---------------------------------------------------------------------------

def _build_hybrid_search(request: Request, kb, settings, top_k: int):
    from nanoresearch.rag.core.query_engine.dense_retriever import DenseRetriever
    from nanoresearch.rag.core.query_engine.sparse_retriever import SparseRetriever
    from nanoresearch.rag.core.query_engine.query_processor import QueryProcessor
    from nanoresearch.rag.core.query_engine.fusion import RRFFusion
    from nanoresearch.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
    from nanoresearch.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
    from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory
    from nanoresearch.rag.ingestion.storage.bm25_indexer import BM25Indexer
    from nanoresearch.rag.core.settings import resolve_path
    chroma_col = kb.chroma_collection or str(kb.id)
    embedding = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=chroma_col)
    bm25 = BM25Indexer(index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{chroma_col}")))
    dense = DenseRetriever(settings=settings, embedding_client=embedding, vector_store=vector_store)
    sparse = SparseRetriever(settings=settings, bm25_indexer=bm25, vector_store=vector_store, default_collection=chroma_col)
    return HybridSearch(
        settings=settings, query_processor=QueryProcessor(),
        dense_retriever=dense, sparse_retriever=sparse, fusion=RRFFusion(),
        config=HybridSearchConfig(fusion_top_k=top_k, enable_dense=True, enable_sparse=True,
                                  enable_graph_expansion=kb.enable_graph_expansion),
        session_factory=request.app.state.session_factory, kb_id=kb.id,
    )


async def _retrieve_concept_evidence(request: Request, kb, settings, topic: str, top_k: int = 8) -> list[dict]:
    from nanoresearch.storage.models import KbDocument
    from sqlalchemy import select as _select
    hybrid = _build_hybrid_search(request, kb, settings, top_k)
    result = await hybrid.async_search(topic, top_k=top_k, return_details=True)
    chroma_ids = [r.chunk_id for r in result.results]
    pg_chunks = await _kb_repo(request).get_chunks_by_chroma_ids(chroma_ids)
    doc_ids = list({c.document_id for c in pg_chunks})
    name_map = {}
    if doc_ids:
        async with request.app.state.session_factory() as db:
            res = await db.execute(_select(KbDocument.id, KbDocument.filename).where(KbDocument.id.in_(doc_ids)))
            name_map = {row[0]: row[1] for row in res.all()}
    return [
        {"chunk_id": str(c.id), "content": c.content or "",
         "source": name_map.get(c.document_id, ""), "page": (c.chunk_metadata or {}).get("page")}
        for c in pg_chunks
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_kb_or_404(kb_id: str, uid: str, request: Request):
    try:
        kid = uuid.UUID(kb_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="知识库不存在")
    kb = await _kb_repo(request).get(kid)
    if kb is None or kb.uid != uid:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


def _kb_to_dict(kb) -> dict:
    return {
        "id": str(kb.id),
        "uid": kb.uid,
        "name": kb.name,
        "description": kb.description,
        "embedding_model": kb.embedding_model,
        "chunk_strategy": kb.chunk_strategy,
        "chunk_size": kb.chunk_size,
        "chunk_overlap": kb.chunk_overlap,
        "status": kb.status,
        "doc_count": kb.doc_count,
        "chunk_count": kb.chunk_count,
        "enable_graph_expansion": kb.enable_graph_expansion,
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
        "updated_at": kb.updated_at.isoformat() if kb.updated_at else None,
    }


def _doc_to_dict(doc, image_count: int = 0) -> dict:
    return {
        "id": str(doc.id),
        "kb_id": str(doc.kb_id),
        "filename": doc.filename,
        "file_size": doc.file_size,
        "mime_type": doc.mime_type,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "image_count": image_count,
        "error_msg": doc.error_msg,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


def _chunk_to_dict(chunk) -> dict:
    return {
        "id": str(chunk.id),
        "kb_id": str(chunk.kb_id),
        "document_id": str(chunk.document_id),
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "token_count": chunk.token_count,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "metadata": chunk.chunk_metadata,
    }




# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------

def _graph_repo(request: Request):
    from nanoresearch.storage.repositories.graph_repo import GraphRepository
    return GraphRepository(request.app.state.session_factory)




async def _extract_and_persist(chunk_rows, settings, session_factory) -> None:
    """Run EntityExtractor on KbChunk list, then persist to KG tables."""
    import asyncio as _asyncio
    from nanoresearch.rag.core.types import Chunk
    from nanoresearch.rag.ingestion.transform.entity_extractor import EntityExtractor
    from nanoresearch.rag.ingestion.graph.persist import persist_chunk_entities

    if not chunk_rows:
        return

    chunks = []
    for row in chunk_rows:
        meta = dict(row.chunk_metadata or {})
        meta.setdefault("source_path", getattr(row, "document_id", None) or str(row.id))
        try:
            chunks.append(Chunk(id=str(row.id), text=row.content, metadata=meta))
        except Exception as _ce:
            import sys as _sys
            print(f"[KG] Skipping chunk {row.id}: {_ce}", file=_sys.stderr)

    extractor = EntityExtractor(settings)
    loop = _asyncio.get_running_loop()
    extracted = await loop.run_in_executor(None, lambda: extractor.transform(chunks))

    extracted_map = {c.id: c for c in extracted}
    for row in chunk_rows:
        ec = extracted_map.get(str(row.id))
        if ec:
            row.chunk_metadata = {
                **(row.chunk_metadata or {}),
                "_kg_entities": ec.metadata.get("_kg_entities", []),
                "_kg_relations": ec.metadata.get("_kg_relations", []),
            }

    await persist_chunk_entities(chunk_rows, session_factory)


@router.post("/api/knowledge/{kb_id}/graph/build", status_code=202)
async def build_graph(
    kb_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    """Rebuild the knowledge graph for all indexed chunks in the KB.

    After completion, enable_graph_expansion is automatically set to true.
    """
    kb = await _get_kb_or_404(kb_id, uid, request)
    repo = _kb_repo(request)
    from nanoresearch.providers.model_factory import ModelRole
    settings = await _resolve_rag_settings(uid, request, ModelRole.INGESTION_LLM)

    async def _run_build():
        import sys

        kb_uuid = uuid.UUID(kb_id)
        graph_repo = _graph_repo(request)

        try:
            await graph_repo.delete_by_kb(kb_uuid)
        except Exception as e:
            print(f"[KG BUILD] Failed to clear existing graph: {e}", file=sys.stderr)

        PAGE_SIZE = 200
        NUM_WORKERS = 3
        queue: asyncio.Queue = asyncio.Queue()
        total_processed = 0

        async def producer():
            try:
                total = await repo.count_chunks_by_kb(kb_uuid)
                print(f"[KG BUILD] KB {kb_id}: {total} chunks, {NUM_WORKERS} workers", flush=True)
                for offset in range(0, total, PAGE_SIZE):
                    await queue.put(offset)
            finally:
                # Always send sentinels so workers are never left waiting
                for _ in range(NUM_WORKERS):
                    await queue.put(None)

        async def worker():
            nonlocal total_processed
            while True:
                offset = await queue.get()
                if offset is None:
                    break
                try:
                    chunks = await repo.list_chunks_by_kb(kb_uuid, limit=PAGE_SIZE, offset=offset)
                    if chunks:
                        await _extract_and_persist(chunks, settings, request.app.state.session_factory)
                        total_processed += len(chunks)
                except Exception as e:
                    print(f"[KG BUILD] Error at offset {offset}: {e}", file=sys.stderr)

        await asyncio.gather(producer(), *[worker() for _ in range(NUM_WORKERS)])
        await repo.update(kb_uuid, enable_graph_expansion=True)
        print(f"[KG BUILD] Completed for KB {kb_id}: {total_processed} chunks processed.", flush=True)

    asyncio.create_task(_run_build())
    return {"status": "building", "kb_id": kb_id}


@router.get("/api/knowledge/{kb_id}/graph/stats")
async def get_graph_stats(
    kb_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    """Return entity count, triple count, and top-20 most-mentioned entities."""
    await _get_kb_or_404(kb_id, uid, request)
    graph_repo = _graph_repo(request)
    stats = await graph_repo.get_stats(uuid.UUID(kb_id))
    return stats


@router.post("/api/knowledge/{kb_id}/documents/{doc_id}/graph/build", status_code=202)
async def build_doc_graph(
    kb_id: str,
    doc_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    """Run entity extraction on a single document's chunks and persist to KG tables."""
    await _get_kb_or_404(kb_id, uid, request)
    repo = _kb_repo(request)
    doc = await repo.get_document(uuid.UUID(doc_id))
    if doc is None or str(doc.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")

    from nanoresearch.providers.model_factory import ModelRole
    settings = await _resolve_rag_settings(uid, request, ModelRole.INGESTION_LLM)

    async def _run():
        import sys
        graph_repo = _graph_repo(request)
        doc_uuid = uuid.UUID(doc_id)
        try:
            await graph_repo.delete_by_doc(doc_uuid)
            chunks = await repo.list_chunks_by_doc(doc_uuid)
            if chunks:
                await _extract_and_persist(chunks, settings, request.app.state.session_factory)
            print(f"[KG DOC BUILD] doc={doc_id} done, {len(chunks)} chunks", flush=True)
        except Exception as e:
            print(f"[KG DOC BUILD] doc={doc_id} error: {e}", file=sys.stderr)

    asyncio.create_task(_run())
    return {"status": "building", "doc_id": doc_id}


@router.get("/api/knowledge/{kb_id}/documents/{doc_id}/graph/entities")
async def get_doc_entities(
    kb_id: str,
    doc_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    """Return entities extracted from a specific document."""
    await _get_kb_or_404(kb_id, uid, request)
    repo = _kb_repo(request)
    doc = await repo.get_document(uuid.UUID(doc_id))
    if doc is None or str(doc.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    graph_repo = _graph_repo(request)
    entities = await graph_repo.get_entities_by_doc(uuid.UUID(doc_id))
    return {"doc_id": doc_id, "entities": entities}


@router.get("/api/knowledge/{kb_id}/graph/entities")
async def list_graph_entities(
    kb_id: str,
    request: Request,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    entities = await _graph_repo(request).list_entities(
        uuid.UUID(kb_id), search=search, limit=limit, offset=offset
    )
    return {"entities": entities}


@router.get("/api/knowledge/{kb_id}/graph/entities/{name}")
async def get_graph_entity(
    kb_id: str,
    name: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _graph_repo(request)
    kb_uuid = uuid.UUID(kb_id)
    summary = await repo.get_entity_summary(kb_uuid, name)
    if summary is None:
        raise HTTPException(status_code=404, detail="entity not found")
    facts = await repo.get_entity_facts(kb_uuid, name)
    self_name = summary["name"]
    neighbors: list[str] = []
    seen = {self_name}
    for f in facts:
        other = f["target"] if f["source"] == self_name else f["source"]
        if other not in seen:
            seen.add(other)
            neighbors.append(other)
    return {**summary, "facts": facts, "neighbors": neighbors}


@router.get("/api/knowledge/{kb_id}/graph/triples/{triple_id}/chunks")
async def get_triple_chunks(
    kb_id: str,
    triple_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    chunks = await _graph_repo(request).get_chunks_by_triple(uuid.UUID(kb_id), uuid.UUID(triple_id))
    # Resolve document_id -> original filename (chunk carries only the ingest path context).
    from nanoresearch.storage.models import KbDocument
    from sqlalchemy import select as _select
    doc_ids = list({c.document_id for c in chunks})
    name_map: dict = {}
    if doc_ids:
        async with request.app.state.session_factory() as db:
            res = await db.execute(_select(KbDocument.id, KbDocument.filename).where(KbDocument.id.in_(doc_ids)))
            name_map = {r[0]: r[1] for r in res.all()}
    return {
        "chunks": [
            {
                "content": (c.content or "")[:500],
                "source": name_map.get(c.document_id, ""),
                "page": (c.chunk_metadata or {}).get("page"),
                "document_id": str(c.document_id),
            }
            for c in chunks
        ]
    }


def _article_dict(row, stale: bool) -> dict:
    return {
        "markdown": row.markdown,
        "citations": row.citations or [],
        "model": row.model,
        "generated_at": row.generated_at.isoformat() if row.generated_at else None,
        "stale": stale,
    }


@router.get("/api/knowledge/{kb_id}/graph/entities/{name}/article")
async def get_entity_article(kb_id: str, name: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _graph_repo(request)
    kb_uuid = uuid.UUID(kb_id)
    row = await repo.get_article(kb_uuid, name)
    if row is None:
        return {"article": None}
    from nanoresearch.rag.wiki.article_generator import evidence_signature
    evidence = await repo.get_entity_evidence(kb_uuid, name)
    stale = evidence_signature(evidence) != row.evidence_hash
    return {"article": _article_dict(row, stale)}


@router.post("/api/knowledge/{kb_id}/graph/entities/{name}/article")
async def generate_entity_article(kb_id: str, name: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _graph_repo(request)
    kb_uuid = uuid.UUID(kb_id)
    if await repo.get_entity_summary(kb_uuid, name) is None:
        raise HTTPException(status_code=404, detail="entity not found")
    from nanoresearch.rag.wiki.article_generator import generate_article, evidence_signature
    evidence = await repo.get_entity_evidence(kb_uuid, name)
    facts = await repo.get_entity_facts(kb_uuid, name)
    settings = await _resolve_rag_settings(uid, request)
    markdown, citations = await generate_article(settings, name, facts, evidence)
    model = getattr(getattr(settings, "llm", None), "model", None)
    row = await repo.upsert_article(kb_uuid, name, markdown, citations, evidence_signature(evidence), model)
    return {"article": _article_dict(row, stale=False)}
