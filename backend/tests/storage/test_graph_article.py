"""GraphRepository entity-evidence + article cache tests (Wiki Phase 2). Real PG."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from nanoresearch.storage.repositories.graph_repo import GraphRepository
from tests.conftest import make_factory, pg_conn


def run(coro):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.fixture(autouse=True)
def clean_graph():
    conn = pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE TABLE kg_triple_mentions, kg_entity_mentions, "
                "kg_triples, kg_entities, kb_chunks, kb_documents, knowledge_bases, users "
                "RESTART IDENTITY CASCADE"
            )
    finally:
        conn.close()


async def _seed(factory):
    from nanoresearch.storage.models import (
        KbChunk, KbDocument, KgEntity, KgEntityMention, KnowledgeBase, User,
    )
    kb_id = uuid.uuid4()
    d1 = uuid.uuid4()
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    e_gs = uuid.uuid4()
    async with factory() as db:
        db.add(User(uid="tester", email="tester@test.com", password_hash="dummy"))
        await db.flush()  # Ensure user is inserted first
        db.add(KnowledgeBase(id=kb_id, uid="tester", name="KB", chroma_collection="c"))
        await db.flush()  # Ensure KB is inserted before referencing it
        db.add(KbDocument(id=d1, kb_id=kb_id, filename="paperA.pdf", file_path="/tmp/a"))
        await db.flush()  # Ensure document is inserted before chunks
        db.add_all([
            KbChunk(id=c1, kb_id=kb_id, document_id=d1, chunk_index=0, content="3dgs uses explicit points"),
            KbChunk(id=c2, kb_id=kb_id, document_id=d1, chunk_index=1, content="3dgs renders fast"),
        ])
        await db.flush()  # Ensure chunks are inserted
        db.add(KgEntity(id=e_gs, kb_id=kb_id, name="3dgs", label="method"))
        await db.flush()  # Ensure entity is inserted
        db.add_all([
            KgEntityMention(entity_id=e_gs, chunk_id=c1, kb_id=kb_id),
            KgEntityMention(entity_id=e_gs, chunk_id=c2, kb_id=kb_id),
        ])
        await db.commit()
    return {"kb_id": kb_id}


def test_get_entity_evidence_returns_content_and_filename():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        ev = await repo.get_entity_evidence(s["kb_id"], "3DGS")
        assert len(ev) == 2
        contents = {e["content"] for e in ev}
        assert contents == {"3dgs uses explicit points", "3dgs renders fast"}
        assert all(e["source"] == "paperA.pdf" for e in ev)   # original filename, not path
        assert all("chunk_id" in e for e in ev)
    run(_())
