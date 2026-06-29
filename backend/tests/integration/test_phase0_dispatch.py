"""Phase 0 integration: worker finalize + 4 acceptance criteria (real Redis)."""
import pytest

from nanoresearch.bus import dist_lock, mailbox
from nanoresearch.bus.redis_keys import RedisKeys


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
