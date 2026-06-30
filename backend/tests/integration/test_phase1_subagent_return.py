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


async def test_continuation_acquire_succeeds_when_free(redis_client):
    """T5: continuation acquires agent_lock when it is free → returns (agent_lock_key, token)."""
    import nanoresearch.worker as worker
    from nanoresearch.storage.repositories.run_repo import RunRepository
    factory, conv = await _seed_conv()
    run = await RunRepository(factory).create(conversation_id=conv.id, uid="u1")
    cont_lock = RedisKeys.continuation_lock("none", str(conv.id))
    await redis_client.set(cont_lock, "ctok", px=120_000)
    lk, tok = await worker._continuation_acquire(
        redis_client, RunRepository(factory), str(run.id),
        RedisKeys.run_events(str(run.id)), str(conv.id), cont_lock, "ctok", timeout_s=2)
    assert lk == RedisKeys.agent_lock("none", str(conv.id))
    assert tok is not None


async def test_continuation_acquire_self_cleans_on_timeout(redis_client):
    """T5/§3.5: agent_lock held → continuation self-cleans (DEL continuation_lock + run_end), no window."""
    import nanoresearch.worker as worker
    from nanoresearch.bus import dist_lock
    from nanoresearch.bus.stream import xread_next
    from nanoresearch.storage.repositories.run_repo import RunRepository
    factory, conv = await _seed_conv()
    run = await RunRepository(factory).create(conversation_id=conv.id, uid="u1")
    agent_lock = RedisKeys.agent_lock("none", str(conv.id))
    cont_lock = RedisKeys.continuation_lock("none", str(conv.id))
    await dist_lock.acquire(redis_client, agent_lock, px_ms=30_000)   # someone else holds agent_lock
    await redis_client.set(cont_lock, "ctok", px=120_000)

    lk, tok = await worker._continuation_acquire(
        redis_client, RunRepository(factory), str(run.id),
        RedisKeys.run_events(str(run.id)), str(conv.id), cont_lock, "ctok", timeout_s=1)

    assert tok is None                                               # could not acquire
    assert await redis_client.get(cont_lock) is None                # continuation_lock DEL'd (self-clean)
    evs, _ = await xread_next(redis_client, RedisKeys.run_events(str(run.id)), "0-0", timeout_ms=200)
    assert any(e.get("type") == "run_end" and e.get("status") == "failed" for e in evs)
    assert (await RunRepository(factory).get(run.id)).status == "failed"


async def test_continuation_drain_and_append(redis_client, monkeypatch, tmp_path):
    """T5/§4.2: continuation drains staging (atomic) + appends to the session list."""
    import json as _json
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    import nanoresearch.worker as worker
    from nanoresearch.session.manager import SessionManager
    factory, conv = await _seed_conv()
    sk = f"web:{conv.id}"
    rk = RedisKeys.subagent_results(sk)
    await redis_client.rpush(rk, _json.dumps({"label": "L1", "task": "t", "status": "ok", "result": "r1"}))
    await redis_client.rpush(rk, _json.dumps({"label": "L2", "task": "t", "status": "ok", "result": "r2"}))
    sessions = SessionManager(tmp_path, session_factory=factory, default_uid="u1")

    await worker._continuation_drain_and_append(redis_client, sessions, sk, "u1")

    assert await redis_client.exists(rk) == 0                        # staging drained (DEL)
    raw = await redis_client.lrange(RedisKeys.session_msg("u1", "web", str(conv.id)), 0, -1)
    assert len(raw) == 2                                            # appended to the session


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

    # t1 done → STAGE result + NOT fire (t2 still pending)
    await mgr._report_and_join("t1", "label1", "task1", "result-1", origin, "ok", sk)
    assert await redis_client.scard(RedisKeys.pending(sk)) == 1
    assert pool.jobs == []

    # t2 done → empties pending → fire → continuation enqueued reusing original run_id
    await mgr._report_and_join("t2", "label2", "task2", "result-2", origin, "ok", sk)
    assert await redis_client.scard(RedisKeys.pending(sk)) == 0
    assert len(pool.jobs) == 1
    fn, kw = pool.jobs[0]
    assert fn == "run_agent_job"
    assert kw["run_id"] == "orig-1"                       # reused original run_id (SSE continuity)
    assert kw["_cont_lock_token"] and kw.get("_entry_id") is None  # continuation lock, not agent lock
    # both results STAGED in subagent_results (not written to the session directly)
    raw = await redis_client.lrange(RedisKeys.subagent_results(sk), 0, -1)
    assert len(raw) == 2
    # and continuation_lock is set so the gate defers user messages until the continuation runs
    assert await redis_client.get(RedisKeys.continuation_lock("none", str(conv.id))) == kw["_cont_lock_token"]


async def test_subagent_staging_failure_does_not_advance_join(redis_client, monkeypatch):
    """必改 2: if staging RPUSH (even the marker) fully fails, the join is NOT advanced."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    factory, conv = await _seed_conv()
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")
    pool = _FakeArqPool()
    mgr = _subagent_mgr(factory, str(conv.id), pool)
    # force every staging RPUSH (full + marker) to fail
    async def _boom(*a, **k):
        raise RuntimeError("redis down")
    monkeypatch.setattr(redis_client, "rpush", _boom)
    origin = {"channel": "web", "chat_id": str(conv.id), "run_id": "orig-1"}

    await mgr._report_and_join("t1", "L", "task", "result", origin, "ok", sk)
    assert await redis_client.scard(RedisKeys.pending(sk)) == 1  # NOT removed
    assert pool.jobs == []                                       # NOT fired


async def test_staging_rpush_retry_then_marker_advances(redis_client, monkeypatch):
    """确认 2: full RPUSH fails, the minimal marker succeeds → THIS subagent advances the join
    (marker has real advancing effect, not just a log)."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    factory, conv = await _seed_conv()
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")
    pool = _FakeArqPool()
    mgr = _subagent_mgr(factory, str(conv.id), pool)
    real_rpush = redis_client.rpush
    calls = {"n": 0}
    async def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] <= 3:                  # the 3 full-result attempts fail
            raise RuntimeError("flaky")
        return await real_rpush(*a, **k)     # the 4th call (the marker) succeeds
    monkeypatch.setattr(redis_client, "rpush", _flaky)
    origin = {"channel": "web", "chat_id": str(conv.id), "run_id": "orig-1"}

    await mgr._report_and_join("t1", "L", "task", "result", origin, "ok", sk)

    assert await redis_client.scard(RedisKeys.pending(sk)) == 0   # join advanced (marker worked)
    assert len(pool.jobs) == 1                                    # continuation fired
    raw = await redis_client.lrange(RedisKeys.subagent_results(sk), 0, -1)
    assert len(raw) == 1 and "unavailable" in raw[0]             # the staged entry is the marker


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


async def test_ac5_subagent_still_streams_to_frontend(redis_client, monkeypatch):
    """AC5: a finished subagent still writes a subagent_result event to run_events:{orig run_id}
    so the frontend SSE display is unchanged."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    from nanoresearch.bus.stream import xread_next
    factory, conv = await _seed_conv()
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")
    mgr = _subagent_mgr(factory, str(conv.id), _FakeArqPool())
    origin = {"channel": "web", "chat_id": str(conv.id), "run_id": "orig-9"}

    await mgr._report_and_join("t1", "L", "task", "RESULT-BODY", origin, "ok", sk)

    evs, _ = await xread_next(redis_client, RedisKeys.run_events("orig-9"), "0-0", timeout_ms=300)
    assert any(e.get("type") == "subagent_result" for e in evs)
