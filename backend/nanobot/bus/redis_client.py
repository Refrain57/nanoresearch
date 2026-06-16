"""Redis client singleton — decode_responses=True enforced globally."""
from __future__ import annotations

import os

import redis.asyncio as aioredis

from nanobot.bus.redis_keys import RedisKeys

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _client


# Compatibility shims — callers import these names from here; logic lives in RedisKeys
stream_key   = RedisKeys.chat_events
job_key      = RedisKeys.job
cancel_key   = RedisKeys.cancel
pending_key  = RedisKeys.pending
run_events_key = RedisKeys.run_events
