"""Phase 2 Task 1: agent_id 全程对齐 (continuation/join path) — real Redis + PG.

Foundation for serial-MVP conclusion ①: all session writes must funnel through the
SAME lock keyed by the primary main's real agent_id. Today the continuation/join path
hardcodes "none" (subagent.py:283, worker.py:328, stuck_run_watchdog.py:110), which
mismatches the dispatcher gate that parses the real agent_id from the mailbox key
(dispatcher.py:115,125). These tests pin the alignment.
"""
import pytest

from nanoresearch.bus.redis_keys import RedisKeys
from tests.conftest import make_factory, truncate_all


@pytest.fixture(autouse=True)
def _clean():
    truncate_all()


async def _seed_conv(uid="u1", agent_id=None):
    """Seed user + conversation whose session_key matches production: web:{conv.id}."""
    from nanoresearch.auth.password import hash_password
    from nanoresearch.storage.models import Conversation
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    factory = make_factory()
    await UserRepository(factory).create(uid, hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:tmp", uid=uid, agent_id=agent_id)
    async with factory() as db:
        c = await db.get(Conversation, conv.id)
        c.session_key = f"web:{conv.id}"
        await db.commit()
        await db.refresh(c)
        return factory, c


async def test_build_run_payload_threads_explicit_agent_id(redis_client):
    """An explicit agent_id kwarg overrides the conv-derived value; None falls back."""
    from nanoresearch.server.routers.chat_router import _build_run_payload
    factory, conv = await _seed_conv()  # conv.agent_id is None

    payload = await _build_run_payload(
        factory, str(conv.id), "u1", content="x", run_id="r1", agent_id="A")
    assert payload["agent_id"] == "A"

    payload_default = await _build_run_payload(
        factory, str(conv.id), "u1", content="x", run_id="r1")
    assert payload_default["agent_id"] == (str(conv.agent_id) if conv.agent_id else None)


async def _seed_conv_with_agent(uid="u1"):
    """Seed user + a real Agent + conversation bound to it (agent_id is a real FK)."""
    from nanoresearch.auth.password import hash_password
    from nanoresearch.storage.models import Conversation
    from nanoresearch.storage.repositories.agent_repo import AgentRepository
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    factory = make_factory()
    await UserRepository(factory).create(uid, hash_password("x"))
    agent = await AgentRepository(factory).create({"name": "Primary", "created_by": uid})
    conv = await ConversationRepository(factory).create(key="web:tmp", uid=uid, agent_id=agent.id)
    async with factory() as db:
        c = await db.get(Conversation, conv.id)
        c.session_key = f"web:{conv.id}"
        await db.commit()
        await db.refresh(c)
        return factory, c, agent


class _FakeArqPool:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, fn, **kw):
        self.jobs.append((fn, kw))


def _subagent_mgr(factory, conv_id, arq_pool, agent_id=None):
    from pathlib import Path
    from nanoresearch.agent.subagent import SubagentManager
    mgr = SubagentManager(provider=None, workspace=Path("."), bus=None, model="m",
                          uid="u1", session_factory=factory, arq_pool=arq_pool)
    mgr.set_run_context(conversation_id=conv_id, agent_id=agent_id)
    return mgr


async def test_continuation_lock_uses_real_agent_id(redis_client, monkeypatch):
    """join sets continuation_lock keyed by the owning main's REAL agent_id (not "none"),
    and the continuation payload carries that same agent_id."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    factory, conv = await _seed_conv()
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")
    pool = _FakeArqPool()
    mgr = _subagent_mgr(factory, str(conv.id), pool, agent_id="A")
    origin = {"channel": "web", "chat_id": str(conv.id), "run_id": "orig-1"}

    await mgr._report_and_join("t1", "L", "task", "result", origin, "ok", sk)

    assert len(pool.jobs) == 1
    kw = pool.jobs[0][1]
    cont_token = kw["_cont_lock_token"]
    assert await redis_client.get(RedisKeys.continuation_lock("A", str(conv.id))) == cont_token
    assert await redis_client.get(RedisKeys.continuation_lock("none", str(conv.id))) is None
    assert kw["agent_id"] == "A"
    assert kw["_cont_lock_key"] == RedisKeys.continuation_lock("A", str(conv.id))


async def test_continuation_acquire_uses_real_agent_id(redis_client):
    """The continuation acquires agent_lock keyed by the real agent_id (serial-MVP conclusion ①:
    session writes funnel through agent_lock:{primary}:{conv}, not agent_lock:none)."""
    import nanoresearch.worker as worker
    from nanoresearch.storage.repositories.run_repo import RunRepository
    factory, conv = await _seed_conv()
    run = await RunRepository(factory).create(conversation_id=conv.id, uid="u1")
    cont_lock = RedisKeys.continuation_lock("A", str(conv.id))
    await redis_client.set(cont_lock, "ctok", px=120_000)

    lk, tok = await worker._continuation_acquire(
        redis_client, RunRepository(factory), str(run.id),
        RedisKeys.run_events(str(run.id)), str(conv.id), cont_lock, "ctok",
        agent_id="A", timeout_s=2)

    assert lk == RedisKeys.agent_lock("A", str(conv.id))
    assert tok is not None


async def test_watchdog_continuation_lock_uses_real_agent_id(redis_client, monkeypatch):
    """The watchdog reaping a stale subagent keys continuation_lock by conv.agent_id, matching
    the dispatcher gate (not the legacy "none")."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    from nanoresearch.heartbeat.stuck_run_watchdog import StuckRunWatchdog
    factory, conv, agent = await _seed_conv_with_agent()
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "dead:1000000000")  # ts in 2001 → stale
    pool = _FakeArqPool()

    wd = StuckRunWatchdog(redis_client, factory, pool, subagent_stale=1)
    await wd._scan_once()

    aid = str(agent.id)
    assert await redis_client.get(RedisKeys.continuation_lock(aid, str(conv.id))) is not None
    assert await redis_client.get(RedisKeys.continuation_lock("none", str(conv.id))) is None
    assert pool.jobs[0][1]["_cont_lock_key"] == RedisKeys.continuation_lock(aid, str(conv.id))
    assert pool.jobs[0][1]["agent_id"] == aid


def test_loop_threads_agent_id_to_subagent_manager():
    """AgentLoop._set_tool_context forwards the run's real agent_id into the SubagentManager,
    so the join keys continuation routing by it at runtime (not just in tests)."""
    from pathlib import Path
    from nanoresearch.agent.loop import AgentLoop
    from nanoresearch.agent.subagent import SubagentManager
    from nanoresearch.agent.tools.registry import ToolRegistry
    loop = AgentLoop.__new__(AgentLoop)
    loop.subagents = SubagentManager(provider=None, workspace=Path("."), bus=None, model="m")
    loop.tools = ToolRegistry()
    loop._set_tool_context("web", "conv-1", agent_id="A")
    assert loop.subagents._conversation_id == "conv-1"
    assert loop.subagents._agent_id == "A"


def test_agent_id_roundtrip_invariant():
    """Regression guard: inbox/lock/continuation_lock all embed agent_id consistently, and the
    dispatcher parses the inbox key back to the SAME agent_id it was built with."""
    from nanoresearch.bus.dispatcher import _parse_inbox_key
    for aid in ("none", "A", "deedf011-8ce8-4f6d-bd13-898600ce54d1"):
        conv = "00000000-0000-0000-0000-0000000000ab"
        assert _parse_inbox_key(RedisKeys.agent_inbox(aid, conv)) == (aid, conv)
        assert RedisKeys.agent_lock(aid, conv).endswith(f"{aid}:{conv}")
        assert RedisKeys.continuation_lock(aid, conv).endswith(f"{aid}:{conv}")
