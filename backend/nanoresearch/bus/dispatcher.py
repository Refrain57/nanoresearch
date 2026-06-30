"""Long-lived, stateless dispatcher: the ONLY job enqueuer (Phase 0).

Consumes the global notify stream via a consumer group, acquires the per-mailbox
distributed lock, and enqueues `run_agent_job` for the next unprocessed inbox
entry. Extra notifies for a locked mailbox are dropped; the running run re-posts a
notify on finish (atomically, in finalize) if the inbox still has backlog.

The dispatcher holds NO per-conversation state — continuity lives in Redis
(mailbox/cursor) and the DB. Restarting it self-heals via a one-shot PEL reclaim.
"""
from __future__ import annotations

import asyncio
import socket
from typing import Any

from loguru import logger

from nanoresearch.bus import dist_lock, mailbox
from nanoresearch.bus.redis_keys import RedisKeys


def _parse_inbox_key(mailbox_key: str) -> tuple[str, str]:
    # "agent_inbox:{agent_id}:{conversation_id}"
    _, agent_id, conversation_id = mailbox_key.split(":", 2)
    return agent_id, conversation_id


class AgentDispatcher:
    def __init__(self, redis: Any, arq_pool: Any, *, lock_px_ms: int = 30_000) -> None:
        self._redis = redis
        self._arq = arq_pool
        self._lock_px_ms = lock_px_ms
        self._consumer = f"disp-{socket.gethostname()}-{id(self)}"
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        await mailbox.ensure_group(self._redis)
        await self._reclaim_pending()          # restart self-heal for un-ACKed notifies
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("AgentDispatcher started (consumer={})", self._consumer)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _reclaim_pending(self) -> None:
        """One-shot reclaim of notify entries left un-ACKed by a previous dispatcher
        instance (PEL). Redis 5.0 has no XAUTOCLAIM (6.2+) → XPENDING + XCLAIM.
        Continuous reclaim = Phase 0.1. Best-effort, never raises."""
        try:
            summary = await self._redis.xpending(
                RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP)
            if not summary or not summary.get("pending"):
                return
            detail = await self._redis.xpending_range(
                RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP,
                min="-", max="+", count=500)
            ids = [item["message_id"] for item in detail]
            if not ids:
                return
            claimed = await self._redis.xclaim(
                RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP, self._consumer,
                min_idle_time=0, message_ids=ids)
            for entry_id, fields in claimed:
                try:
                    await self._handle_notify(fields)
                except Exception:
                    logger.exception("reclaim _handle_notify failed")
                finally:
                    await self._redis.xack(
                        RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP, entry_id)
            logger.info("AgentDispatcher reclaimed {} pending notify(s)", len(claimed))
        except Exception:
            logger.exception("dispatcher _reclaim_pending failed (non-fatal)")

    async def _run(self) -> None:
        while self._running:
            try:
                resp = await self._redis.xreadgroup(
                    RedisKeys.DISPATCH_GROUP, self._consumer,
                    {RedisKeys.DISPATCH_NOTIFY: ">"}, count=20, block=5_000)
                if not resp:
                    continue
                for _stream, entries in resp:
                    for entry_id, fields in entries:
                        try:
                            await self._handle_notify(fields)
                        except Exception:
                            logger.exception("dispatcher _handle_notify failed")
                        finally:
                            await self._redis.xack(
                                RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP, entry_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("dispatcher loop error; retrying")
                await asyncio.sleep(1)

    async def _handle_notify(self, fields: dict) -> str:
        """Acquire the mailbox lock and enqueue the next unprocessed entry.

        Returns one of: "enqueued" | "dropped_locked" | "empty_released" (for tests/telemetry).
        """
        mailbox_key = fields["mailbox_key"]
        lock_key = fields["lock_key"]
        agent_id, conversation_id = _parse_inbox_key(mailbox_key)

        token = await dist_lock.acquire(self._redis, lock_key, px_ms=self._lock_px_ms)
        if token is None:
            return "dropped_locked"  # a run is already processing this mailbox

        # Phase 1 redesign batch gate. Defer a user message while a subagent batch is in flight
        # (pending>0) OR a continuation is pending/running (continuation_lock set). The two are
        # switched atomically by the join (SREM-to-zero + SET continuation_lock in one EVAL), so
        # there is no gap. (Platform session_key is "web:{conversation_id}".)
        cont_lock_key = RedisKeys.continuation_lock(agent_id, conversation_id)
        if (await self._redis.scard(RedisKeys.pending(f"web:{conversation_id}"))
                or await self._redis.exists(cont_lock_key)):
            await dist_lock.release(self._redis, lock_key, token)
            return "deferred_batch"

        nxt = await mailbox.read_next_after_cursor(self._redis, agent_id, conversation_id)
        if nxt is None:
            await dist_lock.release(self._redis, lock_key, token)
            return "empty_released"

        entry_id, payload = nxt
        await self._arq.enqueue_job(
            "run_agent_job",
            **payload,
            _lock_key=lock_key,
            _lock_token=token,
            _entry_id=entry_id,
        )
        return "enqueued"
