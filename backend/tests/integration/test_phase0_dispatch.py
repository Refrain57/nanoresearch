"""Phase 0 integration: worker finalize + 4 acceptance criteria (real Redis)."""
import asyncio

import pytest

from nanoresearch.bus import dist_lock, mailbox
from nanoresearch.bus.dispatcher import AgentDispatcher
from nanoresearch.bus.redis_keys import RedisKeys


class _RecordingPool:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, fn, **kw):
        self.jobs.append((fn, kw))


async def _drain_notifies(redis, disp):
    """Consume all currently-unread notify entries through the group, like the real dispatcher."""
    await mailbox.ensure_group(redis)
    resp = await redis.xreadgroup(
        RedisKeys.DISPATCH_GROUP, "test-consumer", {RedisKeys.DISPATCH_NOTIFY: ">"}, count=100)
    decisions = []
    for _stream, entries in (resp or []):
        for entry_id, fields in entries:
            decisions.append(await disp._handle_notify(fields))
            await redis.xack(RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP, entry_id)
    return decisions


async def test_run_finally_releases_lock_advances_cursor_and_renotifies(redis_client):
    """run_agent_job finalize must atomically: advance cursor → re-notify backlog → release last."""
    import nanoresearch.worker as worker

    aid, cid = "none", "fin-c1"
    e1 = await mailbox.post_message(redis_client, aid, cid, {"content": "m1", "run_id": "r1"})
    await mailbox.post_message(redis_client, aid, cid, {"content": "m2", "run_id": "r2"})  # backlog
    lock_key = RedisKeys.agent_lock(aid, cid)
    token = await dist_lock.acquire(redis_client, lock_key, px_ms=30_000)
    notify_before = await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY)

    await worker._finalize_mailbox_run(
        redis_client, agent_id=aid, conversation_id=cid,
        lock_key=lock_key, lock_token=token, entry_id=e1)

    assert (await redis_client.get(RedisKeys.agent_inbox_cursor(aid, cid))) == e1   # advanced
    assert await dist_lock.acquire(redis_client, lock_key, px_ms=1000) is not None  # released
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) == notify_before + 1  # re-notified


async def test_http_helper_posts_to_inbox_and_notify(redis_client):
    """Task 6: _enqueue_via_mailbox posts one inbox entry + one notify (no direct enqueue)."""
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox

    aid, cid = "none", "http-c1"
    payload = {"content": "hello", "agent_id": None, "conversation_id": cid,
               "run_id": "r1", "session_key": f"web:{cid}", "uid": "u1"}
    await _enqueue_via_mailbox(redis_client, payload)

    got = await mailbox.read_next_after_cursor(redis_client, aid, cid)
    assert got is not None and got[1]["content"] == "hello"
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) >= 1


async def test_idempotency_gate_blocks_duplicate_inbox_entry(redis_client):
    """Must-fix 1: same job_id second SET NX fails → no second inbox entry posted."""
    aid, cid = "none", "dedup-c1"
    job_id = "dedupjob1"
    won1 = await redis_client.set(RedisKeys.job(job_id), "r1", nx=True, ex=3600)
    assert won1
    await mailbox.post_message(redis_client, aid, cid, {
        "content": "x", "agent_id": None, "conversation_id": cid, "run_id": "r1"})
    won2 = await redis_client.set(RedisKeys.job(job_id), "r2", nx=True, ex=3600)
    assert not won2  # duplicate blocked → caller returns existing run_id, does NOT post
    e1 = await mailbox.read_next_after_cursor(redis_client, aid, cid)
    await mailbox.advance_cursor(redis_client, aid, cid, e1[0])
    assert await mailbox.read_next_after_cursor(redis_client, aid, cid) is None  # only one entry


# ── 4 acceptance criteria ─────────────────────────────────────────────────────

def _msg(cid, rid, content="hi"):
    return {"content": content, "agent_id": None, "conversation_id": cid,
            "run_id": rid, "session_key": f"web:{cid}", "uid": "u1"}


async def test_ac1_single_message_one_run(redis_client):
    """验收1：单条消息 → 恰好一个 run 被入队。"""
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox
    pool = _RecordingPool()
    disp = AgentDispatcher(redis_client, pool)
    await _enqueue_via_mailbox(redis_client, _msg("ac1", "r1"))
    await _drain_notifies(redis_client, disp)
    assert [j[1]["run_id"] for j in pool.jobs] == ["r1"]


async def test_ac3_no_double_dispatch_while_locked(redis_client):
    """验收3：同信箱第二条通知在锁持有期被丢弃，不产生第二个 run。"""
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox
    pool = _RecordingPool()
    disp = AgentDispatcher(redis_client, pool)
    await _enqueue_via_mailbox(redis_client, _msg("ac3", "r1"))
    await _enqueue_via_mailbox(redis_client, _msg("ac3", "r2"))
    decisions = await _drain_notifies(redis_client, disp)
    assert [j[1]["run_id"] for j in pool.jobs] == ["r1"]      # only first enqueued
    assert "dropped_locked" in decisions                     # second dropped (lock held)


async def test_ac2_serialized_drain_after_finalize(redis_client):
    """验收2：第一个 run 收尾后链式拉起第二条，按序、无覆盖。"""
    import nanoresearch.worker as worker
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox
    pool = _RecordingPool()
    disp = AgentDispatcher(redis_client, pool)
    await _enqueue_via_mailbox(redis_client, _msg("ac2", "r1", "m1"))
    await _enqueue_via_mailbox(redis_client, _msg("ac2", "r2", "m2"))
    await _drain_notifies(redis_client, disp)
    assert [j[1]["run_id"] for j in pool.jobs] == ["r1"]      # only first; lock serializes

    # finalize r1 → atomically advance cursor + re-notify (backlog r2) + release lock
    j = pool.jobs[0][1]
    await worker._finalize_mailbox_run(
        redis_client, agent_id="none", conversation_id="ac2",
        lock_key=j["_lock_key"], lock_token=j["_lock_token"], entry_id=j["_entry_id"])

    await _drain_notifies(redis_client, disp)                 # process the re-notify
    assert [j[1]["run_id"] for j in pool.jobs] == ["r1", "r2"]


async def test_ac4_lock_auto_expires_on_worker_death(redis_client):
    """验收4：持锁者"死亡"（不续租）后 PX 到期，锁可被重新获取，不死锁。"""
    key = RedisKeys.agent_lock("none", "ac4")
    tok = await dist_lock.acquire(redis_client, key, px_ms=300)
    assert tok is not None
    await asyncio.sleep(0.4)  # no refresh = simulated worker death
    assert await dist_lock.acquire(redis_client, key, px_ms=300) is not None
