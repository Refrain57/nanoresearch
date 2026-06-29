"""Phase 0: per-(agent,conversation) distributed lock primitive."""
import pytest

from nanoresearch.bus import dist_lock

pytestmark = pytest.mark.asyncio


async def test_acquire_then_second_acquire_fails(redis_client):
    key = "agent_lock:t:c1"
    tok1 = await dist_lock.acquire(redis_client, key, px_ms=5_000)
    assert tok1 is not None
    tok2 = await dist_lock.acquire(redis_client, key, px_ms=5_000)
    assert tok2 is None


async def test_release_with_wrong_token_is_noop(redis_client):
    key = "agent_lock:t:c2"
    tok = await dist_lock.acquire(redis_client, key, px_ms=5_000)
    assert await dist_lock.release(redis_client, key, "not-the-token") is False
    assert await dist_lock.release(redis_client, key, tok) is True
    assert await dist_lock.acquire(redis_client, key, px_ms=5_000) is not None


async def test_refresh_extends_only_with_matching_token(redis_client):
    key = "agent_lock:t:c3"
    tok = await dist_lock.acquire(redis_client, key, px_ms=2_000)
    assert await dist_lock.refresh(redis_client, key, tok, px_ms=10_000) is True
    assert await dist_lock.refresh(redis_client, key, "bad", px_ms=10_000) is False
    ttl = await redis_client.pttl(key)
    assert ttl > 2_000  # 续租生效
