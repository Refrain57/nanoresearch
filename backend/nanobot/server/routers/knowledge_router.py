"""Knowledge base, document, chunk, and retrieval test endpoints."""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

from nanobot.server.middleware.auth import get_current_user
from nanobot.storage.models import KbChunk
from nanobot.storage.repositories.knowledge_repo import KnowledgeRepository

router = APIRouter()


def _kb_repo(request: Request) -> KnowledgeRepository:
    return KnowledgeRepository(request.app.state.session_factory)


def _rag_settings(request: Request):
    settings = getattr(request.app.state, "rag_settings", None)
    if settings is None:
        from nanobot.rag.core.settings import load_settings
        settings = load_settings()
        request.app.state.rag_settings = settings
    return settings


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
    docs = await _kb_repo(request).list_documents(uuid.UUID(kb_id))
    return [_doc_to_dict(d) for d in docs]


@router.post("/api/knowledge/{kb_id}/documents", status_code=202)
async def upload_document(
    kb_id: str,
    request: Request,
    file: UploadFile = File(...),
    uid: str = Depends(get_current_user),
):
    kb = await _get_kb_or_404(kb_id, uid, request)
    repo = _kb_repo(request)

    content = await file.read()
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
    )

    asyncio.create_task(
        _ingest_document(request, uuid.UUID(kb_id), doc.id, tmp_path, kb.chroma_collection or str(kb_id), original_filename=file.filename or "upload", chunk_strategy=kb.chunk_strategy or "auto")
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

    chunk_delta = -(doc.chunk_count or 0)
    await repo.delete_chunks_by_doc(uuid.UUID(doc_id))
    await repo.delete_document(uuid.UUID(doc_id))
    await repo.increment_counts(uuid.UUID(kb_id), doc_delta=-1, chunk_delta=chunk_delta)

    # Best-effort: delete from ChromaDB
    try:
        settings = _rag_settings(request)
        from nanobot.rag.ingestion.document_manager import DocumentManager
        from nanobot.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
        from nanobot.rag.ingestion.storage.bm25_indexer import BM25Indexer
        from nanobot.rag.ingestion.storage.image_storage import ImageStorage
        from nanobot.rag.libs.loader.file_integrity import SQLiteIntegrityChecker
        from nanobot.rag.core.settings import resolve_path

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
    chroma_col = kb.chroma_collection or kb_id

    try:
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
        from nanobot.rag.core.query_engine.dense_retriever import DenseRetriever
        from nanobot.rag.core.query_engine.sparse_retriever import SparseRetriever
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor
        from nanobot.rag.core.query_engine.fusion import RRFFusion
        from nanobot.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
        from nanobot.rag.libs.embedding.embedding_factory import EmbeddingFactory
        from nanobot.rag.ingestion.storage.bm25_indexer import BM25Indexer
        from nanobot.rag.core.settings import resolve_path

        embedding = EmbeddingFactory.create(settings)
        vector_store = VectorStoreFactory.create(settings, collection_name=chroma_col)
        bm25 = BM25Indexer(index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{chroma_col}")))

        dense = DenseRetriever(settings=settings, embedding_client=embedding, vector_store=vector_store)
        sparse = SparseRetriever(settings=settings, bm25_indexer=bm25, vector_store=vector_store, default_collection=chroma_col)
        fusion = RRFFusion()
        hybrid = HybridSearch(
            settings=settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=fusion,
            config=HybridSearchConfig(fusion_top_k=body.top_k, enable_dense=body.enable_dense, enable_sparse=body.enable_sparse),
        )

        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: hybrid.search(body.query, top_k=body.top_k, return_details=True)
        )

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
                }
                for r in result.results
            ],
            "used_fallback": result.used_fallback,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")


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
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
        "updated_at": kb.updated_at.isoformat() if kb.updated_at else None,
    }


def _doc_to_dict(doc) -> dict:
    return {
        "id": str(doc.id),
        "kb_id": str(doc.kb_id),
        "filename": doc.filename,
        "file_size": doc.file_size,
        "mime_type": doc.mime_type,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
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


async def _ingest_document(request: Request, kb_id: uuid.UUID, doc_id: uuid.UUID, file_path: str, chroma_collection: str = "", original_filename: str = "", chunk_strategy: str = "auto") -> None:
    """Background task: run IngestionPipeline and persist chunks to DB."""
    repo = KnowledgeRepository(request.app.state.session_factory)
    settings = _rag_settings(request)
    collection = chroma_collection or str(kb_id)

    await repo.update_document_status(doc_id, "parsing")
    try:
        from nanobot.rag.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(settings, collection=collection, force=True, chunk_strategy_override=chunk_strategy)
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipeline.run(file_path)
        )


        if not result.success:
            await repo.update_document_status(doc_id, "error", error_msg=result.error or "ingestion failed")
            return

        # Fetch chunks from ChromaDB and persist to PostgreSQL
        await repo.update_document_status(doc_id, "indexing")
        try:
            from nanobot.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
            vector_store = VectorStoreFactory.create(settings, collection_name=collection)

            # Retrieve the chunks that were just stored using vector IDs from result
            chunk_rows: list[KbChunk] = []
            for idx, vector_id in enumerate(result.vector_ids):
                try:
                    items = vector_store.get_by_ids([vector_id])
                    if items:
                        item = items[0]
                        text = item.get("text", item.get("document", ""))
                        meta = dict(item.get("metadata") or {})
                        if original_filename:
                            meta["source_path"] = original_filename
                        chunk_rows.append(KbChunk(
                            kb_id=kb_id,
                            document_id=doc_id,
                            chroma_id=vector_id,
                            chunk_index=idx,
                            content=text,
                            token_count=meta.get("token_count"),
                            char_start=meta.get("char_start"),
                            char_end=meta.get("char_end"),
                            chunk_metadata=meta,
                        ))
                except Exception:
                    pass

            if chunk_rows:
                await repo.create_chunks(chunk_rows)

        except Exception:
            # Chunks from ChromaDB fetch failed; still mark indexed with result count
            pass

        chunk_count = result.chunk_count
        await repo.update_document_status(doc_id, "indexed", chunk_count=chunk_count)
        await repo.increment_counts(kb_id, doc_delta=1, chunk_delta=chunk_count)

    except Exception as exc:
        import traceback, sys
        print(f"[INGEST ERROR] doc={doc_id}: {exc}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        await repo.update_document_status(doc_id, "error", error_msg=str(exc))
    finally:
        try:
            os.unlink(file_path)
        except Exception:
            pass
