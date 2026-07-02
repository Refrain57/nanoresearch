import asyncio
import pytest

from nanoresearch.storage.repositories.memory_facts_repo import MemoryFactsRepository
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
def clean_memory_facts():
    conn = pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE memory_facts RESTART IDENTITY CASCADE")
    finally:
        conn.close()


def test_insert_and_list_active():
    async def _():
        repo = MemoryFactsRepository(make_factory())
        f = await repo.insert_extracted("u1", "facts", "偏好 Python", confidence=0.9)
        assert f.id and f.source == "extracted" and f.active
        rows = await repo.list_active("u1")
        assert [r.text for r in rows] == ["偏好 Python"]
    run(_())


def test_deactivate_hides_from_active():
    async def _():
        repo = MemoryFactsRepository(make_factory())
        f = await repo.insert_extracted("u1", "facts", "旧")
        await repo.deactivate(f.id)
        assert await repo.list_active("u1") == []
    run(_())


def test_manual_carries_audit_and_uid_scoped():
    async def _():
        repo = MemoryFactsRepository(make_factory())
        m = await repo.insert_manual("u1", "facts", "人工", edited_by="u1")
        assert m.source == "manual" and m.edited_by == "u1" and m.edited_at
        assert await repo.list_active("u2") == []  # uid isolation
    run(_())
