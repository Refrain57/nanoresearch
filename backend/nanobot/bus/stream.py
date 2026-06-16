"""Redis Stream helpers: chunked write + xread wrapper."""
from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis

import logging

logger = logging.getLogger(__name__)

MAX_CHUNK_BYTES = 8_192       # Problem 5: Big Key 防护
STREAM_TTL_SECS = 86_400      # 24h replay 窗口 (Problem 6)


async def xadd_event(redis: "aioredis.Redis", stream_key: str, event: dict) -> None:
    """XADD to stream. Chunks payloads larger than MAX_CHUNK_BYTES. (Problem 5)"""
    key = stream_key
    payload = json.dumps(event, ensure_ascii=False)
    encoded = payload.encode()

    if len(encoded) <= MAX_CHUNK_BYTES:
        await redis.xadd(key, {"data": payload})
    else:
        chunks = _chunk_bytes(encoded, MAX_CHUNK_BYTES)
        total = len(chunks)
        group_id = str(uuid.uuid4())   # explicit group id — avoids same-ms collision
        for i, chunk in enumerate(chunks):
            await redis.xadd(key, {
                "data": chunk.decode(errors="replace"),
                "chunk_group_id": group_id,
                "chunk_index": str(i),
                "total_chunks": str(total),
            })

    await redis.expire(key, STREAM_TTL_SECS)


async def xread_next(
    redis: "aioredis.Redis",
    stream_key: str,
    last_id: str,
    timeout_ms: int = 5_000,
) -> tuple[list[dict], str]:
    """XREAD one batch starting after last_id.

    Returns (events, new_last_id).  Caller MUST update its cursor with
    new_last_id — never use "+" as a cursor (Problem 2).
    Chunked messages are reassembled by chunk_group_id (Problem 5).
    """
    key = stream_key
    # block=0 in redis-py means "block forever"; None means non-blocking.
    _block = timeout_ms if timeout_ms > 0 else None
    result = await redis.xread({key: last_id}, count=20, block=_block)

    if not result:
        return [], last_id

    pending_chunks: dict[str, list[str]] = {}
    events: list[dict] = []
    new_last_id = last_id

    for _, messages in result:
        for msg_id, fields in messages:
            new_last_id = msg_id

            group_id = fields.get("chunk_group_id")
            if group_id is None:
                # Complete message
                try:
                    events.append(json.loads(fields["data"]))
                except Exception:
                    pass
            else:
                # Chunked message — accumulate
                total = int(fields["total_chunks"])
                idx = int(fields["chunk_index"])
                bucket = pending_chunks.setdefault(group_id, [""] * total)
                bucket[idx] = fields["data"]
                if all(bucket):
                    full_payload = "".join(pending_chunks.pop(group_id))
                    try:
                        events.append(json.loads(full_payload))
                    except Exception:
                        pass

    return events, new_last_id


async def get_last_id(redis: "aioredis.Redis", stream_key: str) -> str:
    """Return the current tail ID of the stream (or '0-0' if empty/absent).

    Call this BEFORE process_direct starts so the xread cursor covers the
    entire window when waiting for sub-agent results. (Problem 2)

    This function is called outside _run_agent's try block, so errors
    must be caught here rather than propagated.
    """
    try:
        msgs = await redis.xrevrange(stream_key, count=1)
        return msgs[0][0] if msgs else "0-0"
    except Exception:
        logger.warning("Redis get_last_id failed for stream %s, falling back to 0-0", stream_key)
        return "0-0"


def _chunk_bytes(data: bytes, size: int) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]
