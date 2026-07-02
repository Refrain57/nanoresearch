"""P4.1: rebuild chunking + per-uid serial convergence (C1 — real diff-apply, not mocked)."""
import asyncio

import pytest

from nanoresearch.agent.memory import MemoryStore
from nanoresearch.scripts.rebuild_memory_from_pg import plan_rebuild_chunks, rebuild_uid
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


def test_plan_rebuild_chunks_covers_all_contiguously():
    msgs = [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "A"},
        {"role": "user", "content": "b"}, {"role": "assistant", "content": "B"},
        {"role": "user", "content": "c"},
    ]
    chunks = plan_rebuild_chunks(msgs)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(msgs)
    # contiguous, non-overlapping
    for (a, b), (c, d) in zip(chunks, chunks[1:]):
        assert b == c
    assert plan_rebuild_chunks([]) == []


def test_rebuild_uid_processes_conversations_in_chronological_order():
    seen = []

    async def consolidate_fn(msgs, uid, conversation_id, s, e):
        seen.append(conversation_id)

    async def _():
        conversations = [
            ("cB", 2, [{"role": "user", "content": "b"}]),
            ("cA", 1, [{"role": "user", "content": "a"}]),  # earlier created_at
            ("cC", 3, [{"role": "user", "content": "c"}]),
        ]
        n = await rebuild_uid("u1", conversations, consolidate_fn)
        assert n == 3
        assert seen == ["cA", "cB", "cC"]  # chronological, not input order

    run(_())


class _ScriptedResp:
    def __init__(self, memory_update):
        self.finish_reason = "tool_calls"
        self.content = ""
        self.has_tool_calls = True
        self.tool_calls = [type("TC", (), {"arguments": {
            "history_entry": "[t] x", "memory_update": memory_update}})()]


class _ScriptedProvider:
    def __init__(self, memory_update):
        self._md = memory_update

    async def chat_with_retry(self, **kw):
        return _ScriptedResp(self._md)


def test_rebuild_uid_converges_contradictions_via_real_diff_apply(tmp_path):
    """C1: two conversations assert contradictory profiles; after a serial rebuild against the
    shared store, only the LATER one survives (real diff-apply, scripted LLM — not mocked)."""
    async def _():
        factory = make_factory()
        # No knowledge_search → events/summary skipped; profile diff applied to the real store.
        store = MemoryStore(tmp_path, session_factory=factory)
        md_by_conv = {
            "cA": "# User Memory\n\n## FACTS\n- 偏好 X\n",
            "cB": "# User Memory\n\n## FACTS\n- 偏好 not-X\n",  # drops X, asserts not-X
        }

        async def consolidate_fn(msgs, uid, conversation_id, s, e):
            prov = _ScriptedProvider(md_by_conv[conversation_id])
            return await store.consolidate(msgs, prov, "m", uid=uid,
                                           conversation_id=conversation_id, turn_start=s, turn_end=e)

        conversations = [
            ("cA", 1, [{"role": "user", "content": "a"}]),
            ("cB", 2, [{"role": "user", "content": "b"}]),
        ]
        await rebuild_uid("u1", conversations, consolidate_fn)

        texts = {f.text for f in await MemoryFactsRepository(factory).list_active("u1")}
        assert "偏好 not-X" in texts       # later conversation's fact
        assert "偏好 X" not in texts        # converged: earlier contradictory fact removed

    run(_())


def test_rebuild_uid_limit_processes_only_earliest():
    seen = []

    async def consolidate_fn(msgs, uid, cid, s, e):
        seen.append(cid)

    async def _():
        conversations = [
            ("cB", 2, [{"role": "user", "content": "b"}]),
            ("cA", 1, [{"role": "user", "content": "a"}]),
            ("cC", 3, [{"role": "user", "content": "c"}]),
        ]
        n = await rebuild_uid("u1", conversations, consolidate_fn, limit=1)
        assert seen == ["cA"]   # 试水: 只处理最早的一个对话
        assert n == 1

    run(_())


def test_rebuild_uid_dry_run_does_not_write():
    called = []

    async def consolidate_fn(msgs, uid, cid, s, e):
        called.append(cid)

    async def _():
        conversations = [
            ("cA", 1, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "A"},
                       {"role": "user", "content": "b"}]),
        ]
        n = await rebuild_uid("u1", conversations, consolidate_fn, dry_run=True)
        assert called == []   # 干跑不写
        assert n == 2         # 仍报告 chunk 数(2 个 user-turn 段)

    run(_())
