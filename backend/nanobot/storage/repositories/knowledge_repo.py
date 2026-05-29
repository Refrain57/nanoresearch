"""Knowledge base, document, and chunk repository."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanobot.storage.models import KbChunk, KbDocument, KnowledgeBase


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
        async with self._factory() as db:
            result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
            return result.scalar_one_or_none()

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
    ) -> KbDocument:
        doc = KbDocument(kb_id=kb_id, filename=filename, file_path=file_path, file_size=file_size, mime_type=mime_type)
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

    async def get_chunks_by_chroma_ids(self, chroma_ids: list[str]) -> list[KbChunk]:
        if not chroma_ids:
            return []
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk).where(KbChunk.chroma_id.in_(chroma_ids))
            )
            return list(result.scalars().all())
