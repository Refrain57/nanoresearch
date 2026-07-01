import asyncio
import pytest

from nanoresearch.agent.memory import MemoryStore
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


def test_apply_populates_store_and_projects(tmp_path):
    async def _():
        factory = make_factory()
        store = MemoryStore(tmp_path, session_factory=factory)
        md = "# User Memory\n\n## FACTS\n- 偏好 Python\n- 用 Git\n"
        await store._apply_profile_update(md, uid="u1")
        rows = await MemoryFactsRepository(factory).list_active("u1")
        assert {r.text for r in rows} == {"偏好 Python", "用 Git"}
        projected = store.read_long_term()
        assert "偏好 Python" in projected and "用 Git" in projected
    run(_())


def test_apply_protects_manual_fact(tmp_path):
    async def _():
        factory = make_factory()
        repo = MemoryFactsRepository(factory)
        await repo.insert_manual("u1", "facts", "人工事实", edited_by="u1")
        store = MemoryStore(tmp_path, session_factory=factory)
        # LLM output omits the manual fact entirely
        await store._apply_profile_update("# User Memory\n\n## FACTS\n- 新事实\n", uid="u1")
        texts = {r.text for r in await repo.list_active("u1")}
        assert "人工事实" in texts   # protected, not removed
        assert "新事实" in texts     # new extracted added
    run(_())


def test_apply_falls_back_without_factory(tmp_path):
    async def _():
        store = MemoryStore(tmp_path, session_factory=None)
        await store._apply_profile_update("# User Memory\n\n## FACTS\n- x\n", uid="u1")
        assert "x" in store.read_long_term()  # legacy overwrite path
    run(_())
