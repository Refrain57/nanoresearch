"""Phase 0: redis key helpers + inbox/cursor/notify primitives."""
import pytest

from nanoresearch.bus.redis_keys import RedisKeys

# asyncio_mode=auto runs bare `async def` tests; no module-level mark needed
# (this file mixes a sync key test with async Redis tests).


def test_inbox_keys_are_addressed_by_agent_and_conversation():
    assert RedisKeys.agent_inbox("a1", "c1") == "agent_inbox:a1:c1"
    assert RedisKeys.agent_inbox_cursor("a1", "c1") == "agent_inbox_cursor:a1:c1"
    assert RedisKeys.agent_lock("a1", "c1") == "agent_lock:a1:c1"
    assert RedisKeys.DISPATCH_NOTIFY == "dispatch_notify"
    assert RedisKeys.DISPATCH_GROUP == "dispatch_cg"


async def test_post_and_read_next_after_cursor(redis_client):
    from nanoresearch.bus import mailbox

    aid, cid = "a1", "conv-roundtrip"
    e1 = await mailbox.post_message(redis_client, aid, cid, {"content": "hi-1"})
    e2 = await mailbox.post_message(redis_client, aid, cid, {"content": "hi-2"})

    got1 = await mailbox.read_next_after_cursor(redis_client, aid, cid)
    assert got1 == (e1, {"content": "hi-1"})

    await mailbox.advance_cursor(redis_client, aid, cid, e1)
    got2 = await mailbox.read_next_after_cursor(redis_client, aid, cid)
    assert got2 == (e2, {"content": "hi-2"})

    await mailbox.advance_cursor(redis_client, aid, cid, e2)
    assert await mailbox.read_next_after_cursor(redis_client, aid, cid) is None


async def test_ensure_group_is_idempotent(redis_client):
    from nanoresearch.bus import mailbox

    await mailbox.ensure_group(redis_client)
    await mailbox.ensure_group(redis_client)  # second call must not raise


async def test_finalize_atomic_advances_renotifies_then_releases(redis_client):
    from nanoresearch.bus import dist_lock, mailbox

    aid, cid = "a1", "fin-atomic"
    e1 = await mailbox.post_message(redis_client, aid, cid, {"content": "m1"})
    await mailbox.post_message(redis_client, aid, cid, {"content": "m2"})  # backlog
    lock_key = RedisKeys.agent_lock(aid, cid)
    token = await dist_lock.acquire(redis_client, lock_key, px_ms=30_000)
    n0 = await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY)

    ok = await mailbox.finalize_and_release(
        redis_client, agent_id=aid, conversation_id=cid,
        lock_key=lock_key, token=token, entry_id=e1)
    assert ok is True
    assert (await redis_client.get(RedisKeys.agent_inbox_cursor(aid, cid))) == e1  # advanced
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) == n0 + 1            # re-notified
    assert await redis_client.get(lock_key) is None                               # released last


async def test_finalize_is_noop_when_token_lost(redis_client):
    from nanoresearch.bus import dist_lock, mailbox

    aid, cid = "a1", "fin-lost"
    e1 = await mailbox.post_message(redis_client, aid, cid, {"content": "m1"})
    lock_key = RedisKeys.agent_lock(aid, cid)
    await dist_lock.acquire(redis_client, lock_key, px_ms=30_000)  # someone else holds it
    ok = await mailbox.finalize_and_release(
        redis_client, agent_id=aid, conversation_id=cid,
        lock_key=lock_key, token="stale-token", entry_id=e1)
    assert ok is False
    assert await redis_client.get(RedisKeys.agent_inbox_cursor(aid, cid)) is None  # not advanced
    assert await redis_client.get(lock_key) is not None                            # not released
