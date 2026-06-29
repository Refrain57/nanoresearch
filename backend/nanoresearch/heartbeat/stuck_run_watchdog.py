"""Phase 1 watchdog: recover crashed/stuck subagents and timed-out runs.

Two periodic scans:
- Stale pending member → treat the subagent as failed: append a failure note to the main
  conversation, advance the atomic join, and (if it completes the batch) wake the main agent.
  Stops a dead subagent from hanging the conversation forever.
- AgentRun stuck in 'running' past a hard ceiling (and with no pending subagents) → mark failed
  and emit run_end so the SSE connection is not left hanging.

Threshold note (必改 3): pending members are timestamped at SPAWN time (no subagent heartbeat),
so `subagent_stale` is set on the same order as `run_stuck` (default 7800s ≈ job_timeout) to only
catch "the process is actually gone", never a long-but-alive deep-research subagent.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanoresearch.bus import mailbox
from nanoresearch.bus.redis_keys import RedisKeys
from nanoresearch.bus.stream import xadd_event


class StuckRunWatchdog:
    def __init__(self, redis: Any, session_factory: Any, arq_pool: Any, *,
                 interval: int = 120, subagent_stale: int = 7800, run_stuck: int = 7800) -> None:
        self._redis = redis
        self._factory = session_factory
        self._arq = arq_pool
        self._interval = interval
        self._subagent_stale = subagent_stale
        self._run_stuck = run_stuck
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("StuckRunWatchdog started (interval={}s)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while self._running:
            try:
                await self._scan_once()
            except Exception:
                logger.exception("StuckRunWatchdog scan failed")
            await asyncio.sleep(self._interval)

    async def _scan_once(self) -> None:
        await self._scan_stale_pending()
        await self._scan_stuck_running()

    async def _scan_stale_pending(self) -> None:
        now = time.time()
        cursor: Any = "0"
        while True:
            cursor, keys = await self._redis.scan(cursor, match="pending:*", count=100)
            for key in keys:
                session_key = key[len("pending:"):]  # noqa: E203
                for member in await self._redis.smembers(key):
                    parts = member.rsplit(":", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        ts = int(parts[1])
                    except ValueError:
                        continue
                    if now - ts < self._subagent_stale:
                        continue  # still within the (large) grace window — assume alive
                    await self._fail_subagent(session_key, parts[0])
            if str(cursor) == "0":
                break

    async def _fail_subagent(self, session_key: str, task_id: str) -> None:
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        conv = await ConversationRepository(self._factory).get_by_session_key(session_key)
        if conv is None:
            return  # orphan pending (no conversation) — PendingReaper's job, not ours
        conv_id = str(conv.id)

        # append a failure marker (best-effort — we still advance the join either way here,
        # because the subagent is presumed dead and would otherwise hang the conversation)
        try:
            from nanoresearch.session.manager import SessionManager
            mgr = SessionManager(Path("."), session_factory=self._factory, default_uid=conv.uid)
            await mgr.append_message(
                session_key, {"role": "user", "content": f"[Subagent {task_id} timed out / crashed]"},
                uid=conv.uid)
        except Exception:
            logger.warning("watchdog append failure marker failed (non-fatal)")

        token = uuid.uuid4().hex
        lock_key = RedisKeys.agent_lock("none", conv_id)
        fired = await mailbox.join_and_acquire(self._redis, session_key, task_id, lock_key, token)
        if not fired:
            return

        # wake the main — a fresh, valid run_id (修正: never run_id="").
        from nanoresearch.server.routers.chat_router import _build_run_payload
        from nanoresearch.storage.repositories.run_repo import RunRepository
        run = await RunRepository(self._factory).create(
            conversation_id=conv.id, uid=conv.uid, agent_id=conv.agent_id)
        payload = await _build_run_payload(
            self._factory, conv_id, conv.uid,
            content="部分子任务超时/失败，结果已并入对话。请基于已有结果尽力汇总并回复用户。",
            run_id=str(run.id))
        await self._arq.enqueue_job(
            "run_agent_job", **payload, _lock_key=lock_key, _lock_token=token)
        logger.info("watchdog reaped stale subagent {} → woke continuation run {}", task_id, run.id)

    async def _scan_stuck_running(self) -> None:
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        from nanoresearch.storage.repositories.run_repo import RunRepository
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._run_stuck)
        for run in await RunRepository(self._factory).list_stuck_running(cutoff):
            conv = await ConversationRepository(self._factory).get_by_id(run.conversation_id)
            session_key = conv.session_key if conv else None
            # a run legitimately waiting on in-flight subagents must NOT be reaped
            if session_key and await self._redis.scard(RedisKeys.pending(session_key)):
                continue
            await RunRepository(self._factory).update(
                run.id, status="failed", finished_at=datetime.now(timezone.utc),
                error_message="stuck run reaped by watchdog")
            await xadd_event(self._redis, RedisKeys.run_events(str(run.id)),
                             {"type": "run_end", "status": "failed", "error": "stuck run reaped"})
            logger.info("watchdog reaped stuck run {} (emitted run_end)", run.id)
