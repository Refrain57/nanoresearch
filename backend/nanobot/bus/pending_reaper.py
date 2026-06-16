"""Background reaper for orphan pending:{session_key} keys.

Members are stored as '{task_id}:{unix_ts}' (see subagent.py spawn).
Age is determined by parsing the embedded timestamp, NOT by OBJECT IDLETIME
(which is reset by Phase 3 MEMORY USAGE monitoring).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from nanobot.bus.redis_client import get_redis
from nanobot.bus.redis_keys import RedisKeys

logger = logging.getLogger(__name__)


class PendingReaper:
    """Periodically scan and remove stale entries from pending:* sets.

    A member is considered stale when:
      1. Its embedded timestamp is older than ``idle_threshold`` seconds, AND
      2. The corresponding ``chat_events:{chat_id}`` stream no longer exists.

    The age guard (condition 1) is mandatory: stream absence alone is a false
    positive for freshly started sessions whose previous stream already expired.
    """

    def __init__(self, interval: int = 300, idle_threshold: int = 7200) -> None:
        self._interval = interval
        self._idle_threshold = idle_threshold
        self._task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self._reap()
            except Exception:
                logger.exception("PendingReaper iteration failed")
            await asyncio.sleep(self._interval)

    async def _reap(self) -> int:
        """``SCAN`` all ``pending:*`` keys and remove stale members.

        Returns the number of members removed.
        """
        redis = get_redis()
        cursor: Any = "0"
        now = time.time()
        reaped = 0

        while True:
            cursor, keys = await redis.scan(cursor, match="pending:*", count=100)
            for key in keys:
                stale = await self._stale_members(redis, key, now)
                if stale:
                    await redis.srem(key, *stale)
                    remaining = await redis.scard(key)
                    if remaining == 0:
                        await redis.delete(key)
                    reaped += len(stale)
                    logger.info(
                        "PendingReaper: removed %d stale member(s) from %s",
                        len(stale),
                        key,
                    )
            if cursor == 0:
                break

        if reaped:
            logger.info("PendingReaper: total %d stale member(s) cleaned", reaped)
        return reaped

    async def _stale_members(self, redis: Any, key: str, now: float) -> list[str]:
        """Return members whose embedded timestamp exceeds *idle_threshold* and whose stream is gone."""
        # Derive session_key from pending:{session_key}
        session_key = key[len("pending:"):]  # noqa: E203
        if ":" not in session_key:
            return []
        _ch, chat_id = session_key.split(":", 1)

        members: list[str] = await redis.smembers(key)
        stale: list[str] = []

        for member in members:
            parts = member.rsplit(":", 1)
            if len(parts) != 2:
                continue  # malformed member, skip
            try:
                ts = int(parts[1])
            except ValueError:
                continue

            if now - ts < self._idle_threshold:
                continue  # age guard — still within threshold

            # Double-check: stream must also be gone
            stream_exists = await redis.exists(RedisKeys.chat_events(chat_id))
            if not stream_exists:
                stale.append(member)

        return stale
