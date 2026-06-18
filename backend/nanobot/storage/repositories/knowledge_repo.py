"""Knowledge base, document, and chunk repository."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from loguru import logger

from nanobot.storage.models import KbChunk, KbDocument, KnowledgeBase


def _kb_to_hash(kb: KnowledgeBase) -> dict:
    return {
        "id": str(kb.id),
        "uid": kb.uid or "",
        "name": kb.name or "",
        "description": kb.description or "",
        "chroma_collection": kb.chroma_collection or "",
        "embedding_model": kb.embedding_model or "",
        "chunk_count": str(kb.chunk_count or 0),
        "doc_count": str(kb.doc_count or 0),
        "chunk_strategy": kb.chunk_strategy or "auto",
        "chunk_size": str(kb.chunk_size or 512),
        "chunk_overlap": str(kb.chunk_overlap or 50),
        "enable_graph_expansion": "true" if kb.enable_graph_expansion else "false",
        "status": kb.status or "active",
        "created_at": kb.created_at.isoformat() if kb.created_at else "",
        "updated_at": kb.updated_at.isoformat() if kb.updated_at else "",
    }


def _kb_from_hash(h: dict) -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.id = uuid.UUID(h["id"])
    kb.uid = h.get("uid") or ""
    kb.name = h.get("name") or ""
    kb.description = h.get("description") or None
    kb.chroma_collection = h.get("chroma_collection") or None
    kb.embedding_model = h.get("embedding_model") or None
    kb.chunk_count = int(h.get("chunk_count") or 0)
    kb.doc_count = int(h.get("doc_count") or 0)
    kb.chunk_strategy = h.get("chunk_strategy") or "auto"
    kb.chunk_size = int(h.get("chunk_size") or 512)
    kb.chunk_overlap = int(h.get("chunk_overlap") or 50)
    kb.enable_graph_expansion = h.get("enable_graph_expansion") == "true"
    kb.status = h.get("status") or "active"
    ts = h.get("created_at") or ""
    kb.created_at = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc) if ts else None
    ts = h.get("updated_at") or ""
    kb.updated_at = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc) if ts else None
    return kb


class KnowledgeRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    # ------------------------------------------------------------------
    # KnowledgeBase CRUD
    # ------------------------------------------------------------------

    async def list_by_uid(self, uid: str) -> list[KnowledgeBase]:
        async with self._factory() as db:
            result = await db.execute(
                select(KnowledgeBase)
                .where(KnowledgeBase.uid == uid)
                .order_by(KnowledgeBase.created_at.desc())
            )
            return list(result.scalars().all())

    async def get(self, kb_id: uuid.UUID) -> KnowledgeBase | None:
        from nanobot.bus.redis_client import get_redis
        from nanobot.bus.redis_keys import RedisKeys
        cache_key = RedisKeys.kb_meta(str(kb_id))
        try:
            cached = await get_redis().hgetall(cache_key)
            if cached:
                logger.bind(event="kb_meta_cache_hit", cache_layer="kb_meta_cache").debug(
                    "kb_meta cache hit for {}", kb_id
                )
                return _kb_from_hash(cached)
        except Exception:
            pass

        logger.bind(event="kb_meta_cache_miss", cache_layer="kb_meta_cache").debug(
            "kb_meta cache miss for {}", kb_id
        )
        async with self._factory() as db:
            result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
            kb = result.scalar_one_or_none()

        if kb is not None:
            try:
                r = get_redis()
                await r.hset(cache_key, mapping=_kb_to_hash(kb))
                await r.expire(cache_key, RedisKeys.KB_META_TTL)
            except Exception:
                pass
        return kb

    async def create(self, uid: str, name: str, description: str | None = None, embedding_model: str | None = None, chroma_collection: str | None = None, chunk_strategy: str = "auto") -> KnowledgeBase:
        kb = KnowledgeBase(uid=uid, name=name, description=description, embedding_model=embedding_model, chroma_collection=chroma_collection, chunk_strategy=chunk_strategy)
        async with self._factory() as db:
            db.add(kb)
            await db.commit()
            await db.refresh(kb)
        return kb

    async def update(self, kb_id: uuid.UUID, **fields) -> KnowledgeBase | None:
        async with self._factory() as db:
            result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
            kb = result.scalar_one_or_none()
            if kb is None:
                return None
            for key, value in fields.items():
                setattr(kb, key, value)
            kb.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(kb)
        try:
            from nanobot.bus.redis_client import get_redis
            from nanobot.bus.redis_keys import RedisKeys
            await get_redis().delete(RedisKeys.kb_meta(str(kb_id)))
        except Exception:
            pass
        return kb

    async def delete(self, kb_id: uuid.UUID) -> None:
        async with self._factory() as db:
            result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
            kb = result.scalar_one_or_none()
            if kb:
                await db.delete(kb)
                await db.commit()

    async def increment_counts(self, kb_id: uuid.UUID, doc_delta: int = 0, chunk_delta: int = 0) -> None:
        async with self._factory() as db:
            result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
            kb = result.scalar_one_or_none()
            if kb:
                kb.doc_count = max(0, kb.doc_count + doc_delta)
                kb.chunk_count = max(0, kb.chunk_count + chunk_delta)
                kb.updated_at = datetime.now(timezone.utc)
                await db.commit()
        try:
            from nanobot.bus.redis_client import get_redis
            from nanobot.bus.redis_keys import RedisKeys
            await get_redis().delete(RedisKeys.kb_meta(str(kb_id)))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # KbDocument CRUD
    # ------------------------------------------------------------------

    async def list_documents(self, kb_id: uuid.UUID) -> list[KbDocument]:
        async with self._factory() as db:
            result = await db.execute(
                select(KbDocument)
                .where(KbDocument.kb_id == kb_id)
                .order_by(KbDocument.created_at.desc())
            )
            return list(result.scalars().all())

    async def count_images_per_doc(self, kb_id: uuid.UUID) -> dict[uuid.UUID, int]:
        """Return {doc_id: image_count} by counting chunks whose content contains '!['."""
        async with self._factory() as db:
            result = await db.execute(
                text("""
                    SELECT c.document_id, COUNT(c.id)
                    FROM kb_chunks c
                    JOIN kb_documents d ON c.document_id = d.id
                    WHERE d.kb_id = :kb_id
                      AND c.content LIKE '%![%'
                    GROUP BY c.document_id
                """),
                {"kb_id": kb_id},
            )
            return {row[0]: row[1] for row in result.fetchall()}

    async def get_document(self, doc_id: uuid.UUID) -> KbDocument | None:
        async with self._factory() as db:
            result = await db.execute(select(KbDocument).where(KbDocument.id == doc_id))
            return result.scalar_one_or_none()

    async def create_document(
        self,
        kb_id: uuid.UUID,
        filename: str,
        file_path: str | None = None,
        file_size: int | None = None,
        mime_type: str | None = None,
        pdf_parser: str = "marker",
    ) -> KbDocument:
        doc = KbDocument(kb_id=kb_id, filename=filename, file_path=file_path, file_size=file_size, mime_type=mime_type, pdf_parser=pdf_parser)
        async with self._factory() as db:
            db.add(doc)
            await db.commit()
            await db.refresh(doc)
        return doc

    async def update_document_status(
        self,
        doc_id: uuid.UUID,
        status: str,
        chunk_count: int | None = None,
        error_msg: str | None = None,
    ) -> None:
        async with self._factory() as db:
            result = await db.execute(select(KbDocument).where(KbDocument.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = status
                if chunk_count is not None:
                    doc.chunk_count = chunk_count
                if error_msg is not None:
                    doc.error_msg = error_msg
                await db.commit()

    async def update_document_file_path(self, doc_id: uuid.UUID, file_path: str) -> None:
        async with self._factory() as db:
            result = await db.execute(select(KbDocument).where(KbDocument.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.file_path = file_path
                await db.commit()

    async def delete_document(self, doc_id: uuid.UUID) -> None:
        async with self._factory() as db:
            result = await db.execute(select(KbDocument).where(KbDocument.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                await db.delete(doc)
                await db.commit()

    # ------------------------------------------------------------------
    # KbChunk CRUD
    # ------------------------------------------------------------------

    async def create_chunks(self, chunks: list[KbChunk]) -> None:
        async with self._factory() as db:
            db.add_all(chunks)
            await db.commit()

    async def list_chunks_by_doc(self, document_id: uuid.UUID) -> list[KbChunk]:
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk)
                .where(KbChunk.document_id == document_id)
                .order_by(KbChunk.chunk_index)
            )
            return list(result.scalars().all())

    async def count_chunks_by_kb(self, kb_id: uuid.UUID) -> int:
        async with self._factory() as db:
            result = await db.execute(
                select(func.count()).select_from(KbChunk).where(KbChunk.kb_id == kb_id)
            )
            return result.scalar() or 0

    async def list_chunks_by_kb(self, kb_id: uuid.UUID, limit: int = 100, offset: int = 0) -> list[KbChunk]:
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk)
                .where(KbChunk.kb_id == kb_id)
                .order_by(KbChunk.document_id, KbChunk.chunk_index)
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def delete_chunks_by_doc(self, document_id: uuid.UUID) -> int:
        from sqlalchemy import delete as sa_delete
        async with self._factory() as db:
            result = await db.execute(sa_delete(KbChunk).where(KbChunk.document_id == document_id))
            await db.commit()
            return result.rowcount

    async def get_chunks_by_ids(self, chunk_ids: list[uuid.UUID]) -> list[KbChunk]:
        if not chunk_ids:
            return []
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk).where(KbChunk.id.in_(chunk_ids))
            )
            return list(result.scalars().all())

    async def get_chunks_by_chroma_ids(self, chroma_ids: list[str]) -> list[KbChunk]:
        if not chroma_ids:
            return []
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk).where(KbChunk.chroma_id.in_(chroma_ids))
            )
            return list(result.scalars().all())
