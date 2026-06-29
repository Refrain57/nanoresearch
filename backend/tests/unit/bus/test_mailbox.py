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
