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
