"""Phase 1: atomic subagent join that also acquires the agent lock on completion."""
import pytest

from nanoresearch.bus import dist_lock, mailbox
from nanoresearch.bus.redis_keys import RedisKeys


async def test_join_fires_even_when_parent_run_holds_agent_lock(redis_client):
    """Redesign red→green anchor: the last subagent must trigger the continuation EVEN WHILE
    the parent run (R1) still holds agent_lock. join must set continuation_lock (a key the
    parent never holds), NOT agent_lock — so it never contends with R1."""
    sk, cid = "web:hold-c1", "hold-c1"
    agent_lock = RedisKeys.agent_lock("none", cid)
    cont_lock = RedisKeys.continuation_lock("none", cid)
    await dist_lock.acquire(redis_client, agent_lock, px_ms=30_000)   # R1 still holds agent_lock
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")

    fired = await mailbox.join_and_fire(redis_client, sk, "t1", cont_lock, "ctok", 120_000)

    assert fired is True                                       # fires despite R1 holding agent_lock
    assert await redis_client.get(cont_lock) == "ctok"        # continuation_lock set
    assert await redis_client.get(agent_lock) is not None     # R1's agent_lock untouched
    assert await redis_client.scard(RedisKeys.pending(sk)) == 0


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
