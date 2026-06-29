"""Phase 1 integration: subagent async return to the main agent (real Redis + PG)."""
import pytest

from nanoresearch.bus.redis_keys import RedisKeys
from tests.conftest import make_factory, truncate_all


@pytest.fixture(autouse=True)
def _clean():
    truncate_all()


async def _seed_conv(uid="u1"):
    """Seed a user + conversation whose session_key matches production: web:{conv.id}."""
    from nanoresearch.auth.password import hash_password
    from nanoresearch.storage.models import Conversation
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    factory = make_factory()
    await UserRepository(factory).create(uid, hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:tmp", uid=uid)
    async with factory() as db:
        c = await db.get(Conversation, conv.id)
        c.session_key = f"web:{conv.id}"
        await db.commit()
        await db.refresh(c)
        return factory, c


async def test_build_run_payload_rebuilds_config_from_conversation(redis_client):
    from nanoresearch.server.routers.chat_router import _build_run_payload
    factory, conv = await _seed_conv()
    payload = await _build_run_payload(factory, str(conv.id), "u1",
                                       content="请汇总", run_id="orig-run-1")
    assert payload["run_id"] == "orig-run-1"
    assert payload["conversation_id"] == str(conv.id)
    assert payload["content"] == "请汇总"
    assert payload["uid"] == "u1"
    assert payload["session_key"] == conv.session_key  # web:{conv.id} per seed
    assert "agent_id" in payload and "skill_names" in payload  # config keys present


async def test_has_pending_subagents(redis_client):
    """Phase 1 T5: main run defers run_end iff it has pending subagents."""
    import nanoresearch.worker as worker
    sk = "web:skip-c1"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")
    assert await worker._has_pending_subagents(redis_client, sk) is True
    await redis_client.delete(RedisKeys.pending(sk))
    assert await worker._has_pending_subagents(redis_client, sk) is False


class _FakeArqPool:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, fn, **kw):
        self.jobs.append((fn, kw))


def _subagent_mgr(factory, conv_id, arq_pool):
    from pathlib import Path
    from nanoresearch.agent.subagent import SubagentManager
    mgr = SubagentManager(provider=None, workspace=Path("."), bus=None, model="m",
                          uid="u1", session_factory=factory, arq_pool=arq_pool)
    mgr.set_run_context(conversation_id=conv_id)
    return mgr


async def test_subagent_completion_appends_and_fires_join(redis_client, monkeypatch):
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    factory, conv = await _seed_conv()
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000", "t2:1001")
    pool = _FakeArqPool()
    mgr = _subagent_mgr(factory, str(conv.id), pool)
    origin = {"channel": "web", "chat_id": str(conv.id), "run_id": "orig-1"}

    # t1 done → append + NOT fire (t2 still pending)
    await mgr._report_and_join("t1", "label1", "task1", "result-1", origin, "ok", sk)
    assert await redis_client.scard(RedisKeys.pending(sk)) == 1
    assert pool.jobs == []

    # t2 done → empties pending → fire → continuation enqueued reusing original run_id
    await mgr._report_and_join("t2", "label2", "task2", "result-2", origin, "ok", sk)
    assert await redis_client.scard(RedisKeys.pending(sk)) == 0
    assert len(pool.jobs) == 1
    fn, kw = pool.jobs[0]
    assert fn == "run_agent_job"
    assert kw["run_id"] == "orig-1"            # reused original run_id (SSE continuity)
    assert kw["_lock_token"] and kw.get("_entry_id") is None
    # both results appended to the session message list
    raw = await redis_client.lrange(RedisKeys.session_msg("u1", "web", str(conv.id)), 0, -1)
    assert len(raw) == 2


async def test_subagent_append_failure_does_not_advance_join(redis_client, monkeypatch):
    """必改 2: if append fails, the join is NOT advanced (member stays for the watchdog)."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    factory, conv = await _seed_conv()
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")
    pool = _FakeArqPool()
    mgr = _subagent_mgr(factory, str(conv.id), pool)
    # force append_message to fail
    import nanoresearch.session.manager as m
    monkeypatch.setattr(m.SessionManager, "append_message",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    origin = {"channel": "web", "chat_id": str(conv.id), "run_id": "orig-1"}

    await mgr._report_and_join("t1", "L", "task", "result", origin, "ok", sk)
    assert await redis_client.scard(RedisKeys.pending(sk)) == 1  # NOT removed
    assert pool.jobs == []                                       # NOT fired


async def test_watchdog_stale_pending_advances_join_and_wakes(redis_client, monkeypatch):
    """Phase 1 T7: a stale (crashed/stuck) subagent is reaped → join advanced + main woken."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    from nanoresearch.heartbeat.stuck_run_watchdog import StuckRunWatchdog
    factory, conv = await _seed_conv()
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "dead:1000000000")  # ts in year 2001 → stale
    pool = _FakeArqPool()

    wd = StuckRunWatchdog(redis_client, factory, pool, subagent_stale=1)
    await wd._scan_once()

    assert await redis_client.scard(RedisKeys.pending(sk)) == 0   # join advanced
    assert len(pool.jobs) == 1                                    # continuation woken (real run_id)
    assert pool.jobs[0][1]["run_id"]                              # non-empty run_id (修正)


async def test_watchdog_stuck_running_emits_run_end(redis_client):
    """Phase 1 T7: a run stuck in 'running' past the ceiling gets reaped + run_end (unblocks SSE)."""
    from datetime import datetime, timezone, timedelta
    from nanoresearch.bus.stream import xread_next
    from nanoresearch.heartbeat.stuck_run_watchdog import StuckRunWatchdog
    from nanoresearch.storage.repositories.run_repo import RunRepository
    factory, conv = await _seed_conv()
    run = await RunRepository(factory).create(conversation_id=conv.id, uid="u1")
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    await RunRepository(factory).update(run.id, status="running", started_at=old)

    wd = StuckRunWatchdog(redis_client, factory, _FakeArqPool(), run_stuck=1)
    await wd._scan_once()

    evs, _ = await xread_next(redis_client, RedisKeys.run_events(str(run.id)), "0-0", timeout_ms=300)
    assert any(e.get("type") == "run_end" and e.get("status") == "failed" for e in evs)
    assert (await RunRepository(factory).get(run.id)).status == "failed"
