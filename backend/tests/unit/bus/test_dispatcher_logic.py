"""Phase 0: AgentDispatcher decision logic (uses real Redis + fake ARQ pool)."""
import pytest

from nanoresearch.bus import mailbox
from nanoresearch.bus.dispatcher import AgentDispatcher
from nanoresearch.bus.redis_keys import RedisKeys


class _FakePool:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, fn, **kw):
        self.jobs.append((fn, kw))


def _notify(aid, cid):
    return {
        "mailbox_key": RedisKeys.agent_inbox(aid, cid),
        "cursor_key": RedisKeys.agent_inbox_cursor(aid, cid),
        "lock_key": RedisKeys.agent_lock(aid, cid),
    }


async def test_handle_notify_enqueues_when_lock_acquired(redis_client):
    aid, cid = "none", "disp-c1"
    await mailbox.post_message(redis_client, aid, cid, {
        "content": "x", "agent_id": None, "conversation_id": cid, "run_id": "r1"})
    pool = _FakePool()
    disp = AgentDispatcher(redis_client, pool)

    decision = await disp._handle_notify(_notify(aid, cid))

    assert decision == "enqueued"
    assert pool.jobs and pool.jobs[0][0] == "run_agent_job"
    assert pool.jobs[0][1]["run_id"] == "r1"
    assert pool.jobs[0][1]["_lock_token"]      # lock metadata attached
    assert pool.jobs[0][1]["_entry_id"]


async def test_second_notify_dropped_while_locked(redis_client):
    aid, cid = "none", "disp-c2"
    await mailbox.post_message(redis_client, aid, cid, {
        "content": "x", "agent_id": None, "conversation_id": cid, "run_id": "r1"})
    pool = _FakePool()
    disp = AgentDispatcher(redis_client, pool)

    assert await disp._handle_notify(_notify(aid, cid)) == "enqueued"
    # lock still held (run hasn't finalized) → second notify must be dropped, no 2nd enqueue
    assert await disp._handle_notify(_notify(aid, cid)) == "dropped_locked"
    assert len(pool.jobs) == 1


async def test_reclaim_pending_reprocesses_unacked_notify(redis_client):
    """Adjustment 3: a notify left in the PEL by a crashed consumer is reclaimed on restart."""
    aid, cid = "none", "disp-reclaim"
    await mailbox.ensure_group(redis_client)
    await mailbox.post_message(redis_client, aid, cid, {
        "content": "x", "agent_id": None, "conversation_id": cid, "run_id": "r9"})
    await mailbox.post_notify(
        redis_client, mailbox_key=RedisKeys.agent_inbox(aid, cid),
        cursor_key=RedisKeys.agent_inbox_cursor(aid, cid),
        lock_key=RedisKeys.agent_lock(aid, cid))
    # a "crashed" consumer reads the notify into its PEL but never acks
    await redis_client.xreadgroup(
        RedisKeys.DISPATCH_GROUP, "crashed-consumer",
        {RedisKeys.DISPATCH_NOTIFY: ">"}, count=10)

    pool = _FakePool()
    disp = AgentDispatcher(redis_client, pool)
    await disp._reclaim_pending()

    assert any(j[1].get("run_id") == "r9" for j in pool.jobs)


async def test_handle_notify_defers_when_continuation_lock_present(redis_client):
    """Phase 1 redesign: pending empty but a continuation is pending/running → still defer
    (closes the pending-empty → continuation-running window)."""
    aid, cid = "none", "disp-cont"
    await mailbox.post_message(redis_client, aid, cid, {
        "content": "x", "agent_id": None, "conversation_id": cid, "run_id": "r1"})
    await redis_client.set(RedisKeys.continuation_lock(aid, cid), "ctok", px=120_000)
    pool = _FakePool()
    disp = AgentDispatcher(redis_client, pool)

    decision = await disp._handle_notify(_notify(aid, cid))

    assert decision == "deferred_batch"
    assert pool.jobs == []


async def test_handle_notify_defers_during_subagent_batch(redis_client):
    """Phase 1 必改 1: a user message must not run while a subagent batch is in flight."""
    from nanoresearch.bus import dist_lock
    aid, cid = "none", "disp-batch"
    await mailbox.post_message(redis_client, aid, cid, {
        "content": "x", "agent_id": None, "conversation_id": cid, "run_id": "r1"})
    await redis_client.sadd(RedisKeys.pending(f"web:{cid}"), "t1:1000")  # batch in flight
    pool = _FakePool()
    disp = AgentDispatcher(redis_client, pool)

    decision = await disp._handle_notify(_notify(aid, cid))

    assert decision == "deferred_batch"
    assert pool.jobs == []
    # lock released so the continuation can later acquire it
    assert await dist_lock.acquire(redis_client, RedisKeys.agent_lock(aid, cid), px_ms=1000) is not None


async def test_handle_notify_empty_inbox_releases_lock(redis_client):
    aid, cid = "none", "disp-c3"
    # advance cursor past everything (empty backlog), then notify
    pool = _FakePool()
    disp = AgentDispatcher(redis_client, pool)
    decision = await disp._handle_notify(_notify(aid, cid))
    assert decision == "empty_released"
    assert pool.jobs == []
    # lock was released → can acquire again
    from nanoresearch.bus import dist_lock
    assert await dist_lock.acquire(redis_client, RedisKeys.agent_lock(aid, cid), px_ms=1000) is not None
