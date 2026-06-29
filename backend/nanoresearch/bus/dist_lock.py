"""Per-(agent, conversation) distributed lock: SET NX PX + token, Lua release/refresh.

The lock guarantees that at most one processor writes a given mailbox's session
state at a time, across worker processes. Release/refresh are token-gated via Lua
so a holder whose lease already expired (and was re-acquired by someone else) can
never delete or extend the new owner's lock.
"""
from __future__ import annotations

import uuid
from typing import Any

# token match → DEL; else no-op. Atomic (Redis runs the script single-threaded).
RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""

# token match → extend lease; else no-op.
REFRESH_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
else
    return 0
end
"""


async def acquire(redis: Any, key: str, *, px_ms: int = 30_000) -> str | None:
    """Try to acquire the lock. Returns a unique token on success, else None."""
    token = uuid.uuid4().hex
    ok = await redis.set(key, token, nx=True, px=px_ms)
    return token if ok else None


async def refresh(redis: Any, key: str, token: str, *, px_ms: int = 30_000) -> bool:
    """Extend the lease iff *token* still owns the lock."""
    res = await redis.eval(REFRESH_LUA, 1, key, token, str(px_ms))
    return bool(res)


async def release(redis: Any, key: str, token: str) -> bool:
    """Release the lock iff *token* still owns it."""
    res = await redis.eval(RELEASE_LUA, 1, key, token)
    return bool(res)
