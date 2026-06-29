"""Phase 1: atomic subagent join that also acquires the agent lock on completion."""
import pytest

from nanoresearch.bus import mailbox
from nanoresearch.bus.redis_keys import RedisKeys


async def test_join_acquire_fires_and_locks_when_last_member_removed(redis_client):
    sk = "web:ja-c1"
    pkey = RedisKeys.pending(sk)
    lock = RedisKeys.agent_lock("none", "ja-c1")
    await redis_client.sadd(pkey, "t1:1000", "t2:1001")
    # first done → not last → not fired, lock untouched
    assert await mailbox.join_and_acquire(redis_client, sk, "t1", lock, "tok-A") is False
    assert await redis_client.get(lock) is None
    # second done → empties pending → fired AND lock acquired atomically with tok-B
    assert await mailbox.join_and_acquire(redis_client, sk, "t2", lock, "tok-B") is True
    assert await redis_client.get(lock) == "tok-B"
    assert await redis_client.scard(pkey) == 0


async def test_join_acquire_removes_by_task_id_prefix(redis_client):
    sk = "web:ja-c2"
    await redis_client.sadd(RedisKeys.pending(sk), "abc:1700000000")
    lock = RedisKeys.agent_lock("none", "ja-c2")
    assert await mailbox.join_and_acquire(redis_client, sk, "abc", lock, "tok") is True
    assert await redis_client.get(lock) == "tok"


async def test_join_acquire_false_when_not_last(redis_client):
    sk = "web:ja-c3"
    await redis_client.sadd(RedisKeys.pending(sk), "x:1", "y:2", "z:3")
    lock = RedisKeys.agent_lock("none", "ja-c3")
    assert await mailbox.join_and_acquire(redis_client, sk, "x", lock, "t") is False
    assert await redis_client.scard(RedisKeys.pending(sk)) == 2
    assert await redis_client.get(lock) is None
