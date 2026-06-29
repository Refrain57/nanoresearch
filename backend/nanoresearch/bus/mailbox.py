"""Per-(agent, conversation) inbox + cursor + global notify stream primitives (Phase 0).

- inbox  (Stream)  : one entry per inbound request payload, addressed by agent+conversation.
- cursor (String)  : last-processed inbox entry id; `XRANGE (cursor +]` = unprocessed backlog.
- notify (Stream)  : global "mailbox X has work" signal the dispatcher consumes (consumer group).

`finalize_and_release` is the atomic run-finish primitive: in one Lua it verifies the lock
token, advances the cursor, re-notifies if backlog remains, and releases the lock LAST — so the
lock is never freed before the next notify is in place (no exposure window). Token mismatch is a
full no-op (the lock is owned by someone else now).
"""
from __future__ import annotations

import json
from typing import Any

from nanoresearch.bus.redis_keys import RedisKeys


def _next_stream_id(sid: str) -> str:
    """Smallest stream id strictly greater than *sid* (= ms-(seq+1)).

    Redis 5.0 has no exclusive "(" range interval (6.2+), so we read inclusively
    from the next possible id to get entries strictly after the cursor.
    """
    ms, _, seq = sid.partition("-")
    return f"{ms}-{int(seq or 0) + 1}"


async def post_message(redis: Any, agent_id: str, conversation_id: str, payload: dict) -> str:
    key = RedisKeys.agent_inbox(agent_id, conversation_id)
    entry_id = await redis.xadd(key, {"data": json.dumps(payload, ensure_ascii=False)})
    await redis.expire(key, RedisKeys.AGENT_INBOX_TTL)
    return entry_id


async def post_notify(redis: Any, *, mailbox_key: str, cursor_key: str, lock_key: str) -> None:
    await redis.xadd(RedisKeys.DISPATCH_NOTIFY, {
        "mailbox_key": mailbox_key,
        "cursor_key": cursor_key,
        "lock_key": lock_key,
    })


async def read_next_after_cursor(
    redis: Any, agent_id: str, conversation_id: str
) -> tuple[str, dict] | None:
    inbox = RedisKeys.agent_inbox(agent_id, conversation_id)
    cursor_key = RedisKeys.agent_inbox_cursor(agent_id, conversation_id)
    cursor = await redis.get(cursor_key) or "0-0"
    # Redis 5.0: no exclusive "(" interval — read inclusively from the next id.
    res = await redis.xrange(inbox, min=_next_stream_id(cursor), max="+", count=1)
    if not res:
        return None
    entry_id, fields = res[0]
    return entry_id, json.loads(fields["data"])


async def advance_cursor(redis: Any, agent_id: str, conversation_id: str, entry_id: str) -> None:
    await redis.set(
        RedisKeys.agent_inbox_cursor(agent_id, conversation_id),
        entry_id, ex=RedisKeys.AGENT_INBOX_TTL,
    )


async def ensure_group(redis: Any) -> None:
    try:
        await redis.xgroup_create(
            RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP, id="0", mkstream=True)
    except Exception as e:  # redis.exceptions.ResponseError: BUSYGROUP if already exists
        if "BUSYGROUP" not in str(e):
            raise


# Atomic finalize — token-check → advance cursor → (if backlog) re-notify → release, all in ONE
# Lua. The lock DEL is LAST and only after the next notify is already in the stream, so there is
# no "lock freed but next not notified" window. token mismatch → no-op (return 0).
# KEYS[1]=lock KEYS[2]=inbox KEYS[3]=cursor KEYS[4]=notify_stream
# ARGV[1]=token ARGV[2]=entry_id ARGV[3]=cursor_ttl
# ARGV[4]=mailbox_key ARGV[5]=cursor_key ARGV[6]=lock_key ARGV[7]=next_id (Redis-5 inclusive scan)
FINALIZE_LUA = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('SET', KEYS[3], ARGV[2], 'EX', ARGV[3])
local nxt = redis.call('XRANGE', KEYS[2], ARGV[7], '+', 'COUNT', 1)
if #nxt > 0 then
    redis.call('XADD', KEYS[4], '*',
               'mailbox_key', ARGV[4], 'cursor_key', ARGV[5], 'lock_key', ARGV[6])
end
redis.call('DEL', KEYS[1])
return 1
"""


async def finalize_and_release(
    redis: Any, *, agent_id: str, conversation_id: str,
    lock_key: str, token: str, entry_id: str, ttl: int = RedisKeys.AGENT_INBOX_TTL,
) -> bool:
    inbox = RedisKeys.agent_inbox(agent_id, conversation_id)
    cursor = RedisKeys.agent_inbox_cursor(agent_id, conversation_id)
    res = await redis.eval(
        FINALIZE_LUA, 4,
        lock_key, inbox, cursor, RedisKeys.DISPATCH_NOTIFY,                  # KEYS
        token, entry_id, str(ttl), inbox, cursor, lock_key,                 # ARGV[1..6]
        _next_stream_id(entry_id),                                          # ARGV[7]
    )
    return bool(res)
