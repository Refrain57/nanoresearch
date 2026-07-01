"""GraphRepository read-layer tests (Wiki Phase 1). Real PG, sync psycopg2 cleanup."""
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
                "TRUNCATE TABLE kg_triple_mentions, kg_entity_mentions, kg_triples, "
                "kg_entities, kb_chunks, kb_documents, knowledge_bases, users RESTART IDENTITY CASCADE"
            )
    finally:
        conn.close()


async def _seed(factory):
    from nanoresearch.storage.models import (
        KbChunk, KbDocument, KgEntity, KgEntityMention, KgTriple, KgTripleMention, KnowledgeBase, User,
    )
    kb_id = uuid.uuid4()
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    c1, c2, c3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    e_gs, e_nerf = uuid.uuid4(), uuid.uuid4()
    tid = uuid.uuid4()
    async with factory() as db:
        db.add(User(uid="tester", email="tester@test.com", password_hash="dummy"))
        await db.flush()  # Ensure user is inserted first
        db.add(KnowledgeBase(id=kb_id, uid="tester", name="KB", chroma_collection="c"))
        await db.flush()  # Ensure KB is inserted before referencing it
        db.add_all([
            KbDocument(id=d1, kb_id=kb_id, filename="paperA.pdf", file_path="/tmp/a"),
            KbDocument(id=d2, kb_id=kb_id, filename="paperB.pdf", file_path="/tmp/b"),
        ])
        await db.flush()  # Ensure documents are inserted before chunks
        db.add_all([
            KbChunk(id=c1, kb_id=kb_id, document_id=d1, chunk_index=0, content="3dgs vs nerf"),
            KbChunk(id=c2, kb_id=kb_id, document_id=d2, chunk_index=0, content="3dgs faster"),
            KbChunk(id=c3, kb_id=kb_id, document_id=d1, chunk_index=1, content="nerf detail"),
        ])
        await db.flush()  # Ensure chunks are inserted
        db.add_all([
            KgEntity(id=e_gs, kb_id=kb_id, name="3dgs", label="method"),
            KgEntity(id=e_nerf, kb_id=kb_id, name="nerf", label="method"),
            KgTriple(id=tid, kb_id=kb_id, source_id=e_gs, target_id=e_nerf, label="faster_than"),
        ])
        await db.flush()  # Ensure entities and triples are inserted
        db.add_all([
            KgEntityMention(entity_id=e_gs, chunk_id=c1, kb_id=kb_id),
            KgEntityMention(entity_id=e_gs, chunk_id=c2, kb_id=kb_id),
            KgEntityMention(entity_id=e_nerf, chunk_id=c1, kb_id=kb_id),
            KgEntityMention(entity_id=e_nerf, chunk_id=c3, kb_id=kb_id),
            KgTripleMention(triple_id=tid, chunk_id=c1, kb_id=kb_id),  # doc d1
            KgTripleMention(triple_id=tid, chunk_id=c2, kb_id=kb_id),  # doc d2
            KgTripleMention(triple_id=tid, chunk_id=c3, kb_id=kb_id),  # c3 is doc d1 → 3 mentions but still 2 distinct docs
        ])
        await db.commit()
    return {"kb_id": kb_id, "tid": tid}


def test_list_entities_counts_and_search():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        rows = await repo.list_entities(s["kb_id"])
        by_name = {r["name"]: r for r in rows}
        assert by_name["3dgs"]["mentions"] == 2
        assert by_name["nerf"]["mentions"] == 2
        assert by_name["3dgs"]["label"] == "method"
        only = await repo.list_entities(s["kb_id"], search="3d")
        assert [r["name"] for r in only] == ["3dgs"]
    run(_())


def test_get_entity_summary():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        summ = await repo.get_entity_summary(s["kb_id"], "3DGS")
        assert summ == {"name": "3dgs", "label": "method", "mention_count": 2}
        assert await repo.get_entity_summary(s["kb_id"], "nope") is None
    run(_())


def test_get_entity_facts_doc_count_is_distinct_documents():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        facts = await repo.get_entity_facts(s["kb_id"], "3DGS")
        assert len(facts) == 1
        fact = facts[0]
        assert fact["source"] == "3dgs"
        assert fact["label"] == "faster_than"
        assert fact["target"] == "nerf"
        assert fact["doc_count"] == 2  # triple mentioned in chunks from 2 distinct docs
        assert fact["triple_id"] == str(s["tid"])
    run(_())


def test_get_chunks_by_triple():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        chunks = await repo.get_chunks_by_triple(s["tid"])
        assert len(chunks) == 3
        assert {c.content for c in chunks} == {"3dgs vs nerf", "3dgs faster", "nerf detail"}
    run(_())
