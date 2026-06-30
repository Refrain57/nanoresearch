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


# Phase 1: release-only finalize for the continuation run (it bypasses the inbox, so there is
# NO entry to advance the cursor past). Atomic: token-check → (if inbox backlog after the
# current cursor) re-notify → release lock LAST. Reads/derives next-id from the cursor in Lua so
# the whole thing is one atomic step (no exposure window before the queued user message runs).
# KEYS[1]=lock KEYS[2]=inbox KEYS[3]=cursor KEYS[4]=notify
# ARGV[1]=token ARGV[2]=mailbox_key ARGV[3]=cursor_key ARGV[4]=lock_key
RELEASE_ONLY_LUA = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
local cursor = redis.call('GET', KEYS[3])
if not cursor then cursor = '0-0' end
local dash = string.find(cursor, '-')
local ms = string.sub(cursor, 1, dash - 1)
local seq = tonumber(string.sub(cursor, dash + 1))
local nextid = ms .. '-' .. tostring(seq + 1)
local nxt = redis.call('XRANGE', KEYS[2], nextid, '+', 'COUNT', 1)
if #nxt > 0 then
    redis.call('XADD', KEYS[4], '*', 'mailbox_key', ARGV[2], 'cursor_key', ARGV[3], 'lock_key', ARGV[4])
end
redis.call('DEL', KEYS[1])
return 1
"""


async def finalize_and_release(
    redis: Any, *, agent_id: str, conversation_id: str,
    lock_key: str, token: str, entry_id: str | None, ttl: int = RedisKeys.AGENT_INBOX_TTL,
) -> bool:
    inbox = RedisKeys.agent_inbox(agent_id, conversation_id)
    cursor = RedisKeys.agent_inbox_cursor(agent_id, conversation_id)
    if entry_id is None:
        # Continuation run: no inbox entry was consumed — release lock + re-notify backlog only.
        res = await redis.eval(
            RELEASE_ONLY_LUA, 4,
            lock_key, inbox, cursor, RedisKeys.DISPATCH_NOTIFY,             # KEYS
            token, inbox, cursor, lock_key,                                # ARGV
        )
        return bool(res)
    res = await redis.eval(
        FINALIZE_LUA, 4,
        lock_key, inbox, cursor, RedisKeys.DISPATCH_NOTIFY,                  # KEYS
        token, entry_id, str(ttl), inbox, cursor, lock_key,                 # ARGV[1..6]
        _next_stream_id(entry_id),                                          # ARGV[7]
    )
    return bool(res)


# Phase 1 必改 1: atomic subagent join that ALSO acquires the agent lock the instant it
# empties the pending set. Removing the last pending member and taking the lock happen in one
# atomic EVAL, so a dispatcher cannot grab the (otherwise free) lock in the gap between "batch
# done" and "continuation started" — closing the user-message-vs-continuation race.
# KEYS[1]=pending_key KEYS[2]=lock_key  ARGV[1]=task_id ARGV[2]=lock_token ARGV[3]=lock_px_ms
# Returns 1 iff this call removed the last member AND acquired the lock (→ fire the
# continuation with the given token); else 0.
JOIN_ACQUIRE_LUA = """
local members = redis.call('SMEMBERS', KEYS[1])
local prefix = ARGV[1] .. ':'
for i = 1, #members do
    local m = members[i]
    if m == ARGV[1] or string.sub(m, 1, #prefix) == prefix then
        redis.call('SREM', KEYS[1], m)
        break
    end
end
if redis.call('SCARD', KEYS[1]) == 0 then
    if redis.call('SET', KEYS[2], ARGV[2], 'NX', 'PX', ARGV[3]) then
        return 1
    end
end
return 0
"""


async def join_and_acquire(
    redis: Any, session_key: str, task_id: str,
    lock_key: str, lock_token: str, lock_px_ms: int = 30_000,
) -> bool:
    res = await redis.eval(
        JOIN_ACQUIRE_LUA, 2,
        RedisKeys.pending(session_key), lock_key,        # KEYS
        task_id, lock_token, str(lock_px_ms),            # ARGV
    )
    return bool(res)


# Phase 1 redesign: atomic join that, on emptying pending, SETs the **continuation_lock** (a key
# the parent run NEVER holds) — so it never contends with the parent's agent_lock (fixes the
# "R1 holds agent_lock → join SET NX fails → lost wakeup" hole). Overwrite SET (not NX): only the
# pending-emptying subagent reaches it within a batch; across layers a lingering continuation_lock
# is overwritten, and the previous continuation's token-gated DEL won't delete the new value.
# KEYS[1]=pending_key KEYS[2]=continuation_lock_key  ARGV[1]=task_id ARGV[2]=cont_token ARGV[3]=cont_px_ms
JOIN_FIRE_LUA = """
local members = redis.call('SMEMBERS', KEYS[1])
local prefix = ARGV[1] .. ':'
for i = 1, #members do
    local m = members[i]
    if m == ARGV[1] or string.sub(m, 1, #prefix) == prefix then
        redis.call('SREM', KEYS[1], m)
        break
    end
end
if redis.call('SCARD', KEYS[1]) == 0 then
    redis.call('SET', KEYS[2], ARGV[2], 'PX', ARGV[3])
    return 1
end
return 0
"""


async def join_and_fire(
    redis: Any, session_key: str, task_id: str,
    cont_lock_key: str, cont_token: str, cont_px_ms: int = 120_000,
) -> bool:
    """Remove this subagent's pending member; iff that empties the batch, SET continuation_lock
    and return True (fire the continuation with cont_token). Never touches agent_lock."""
    res = await redis.eval(
        JOIN_FIRE_LUA, 2,
        RedisKeys.pending(session_key), cont_lock_key,   # KEYS
        task_id, cont_token, str(cont_px_ms),            # ARGV
    )
    return bool(res)
