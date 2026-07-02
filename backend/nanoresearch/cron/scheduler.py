"""CronScheduler — the time-triggered dispatcher (cron production redesign).

A long-lived sentinel (started in the FastAPI lifespan, alongside AgentDispatcher /
StuckRunWatchdog) that periodically scans the cron_jobs table for due jobs and
CLAIMS + DISPATCHES each one — it never executes the job itself. Execution goes
through the existing mailbox → dispatcher → worker path.

Exactly-once across N sentinels (spec §3.3) rests on three layers:
  1. cron_lock:{job_id}  — dist_lock serializes sentinels within a tick.
  2. deterministic occurrence key job:{sha(job_id, occurrence_ms)} SET NX — dedupes
     a re-post after a crash between post and schedule-advance.
  3. CronJobRepository.advance/finish_one_shot — CAS on next_run_at defines when the
     next occurrence becomes eligible; advances exactly once.
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from loguru import logger

from nanoresearch.bus import dist_lock
from nanoresearch.bus.redis_keys import RedisKeys
from nanoresearch.cron.schedule import resolve_misfire
from nanoresearch.storage.repositories.cron_repo import CronJobRepository

# Injectable dispatch: given a due CronJob, create the run + post to the mailbox.
DispatchFn = Callable[[Any], Awaitable[str | None]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CronScheduler:
    def __init__(
        self,
        redis: Any,
        session_factory: Any,
        *,
        interval_s: int = 30,
        lock_px_ms: int = 60_000,
        idem_ttl_s: int = 3600,
        dispatch_fn: DispatchFn | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._redis = redis
        self._factory = session_factory
        self._repo = CronJobRepository(session_factory)
        self._interval_s = interval_s
        self._lock_px_ms = lock_px_ms
        self._idem_ttl_s = idem_ttl_s
        self._dispatch_fn = dispatch_fn or self._default_dispatch
        self._now_fn = now_fn or _utcnow
        self._task: asyncio.Task | None = None
        self._running = False

    # ── lifecycle (mirrors StuckRunWatchdog) ─────────────────────────────────
    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("CronScheduler started (interval={}s)", self._interval_s)

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
        # Sleep before the first scan: avoids hammering the DB the instant the process starts
        # (lets startup settle) and keeps the scheduler inert during short-lived test lifespans.
        while self._running:
            await asyncio.sleep(self._interval_s)
            if not self._running:
                break
            try:
                await self._scan_once()
            except Exception:
                logger.exception("CronScheduler scan failed")

    # ── scan + per-job claim/dispatch ────────────────────────────────────────
    async def _scan_once(self) -> list[str]:
        now = self._now_fn()
        due = await self._repo.list_due(now)
        outcomes: list[str] = []
        for job in due:
            try:
                outcomes.append(await self._dispatch_due_job(job))
            except Exception:
                logger.exception("CronScheduler dispatch failed for job {}", job.id)
                outcomes.append("error")
        return outcomes

    @staticmethod
    def _det_key(job_id: uuid.UUID, observed: datetime) -> str:
        occ_ms = int(observed.timestamp() * 1000)
        digest = hashlib.sha256(f"{job_id}:{occ_ms}".encode()).hexdigest()[:20]
        return f"cron-{digest}"

    async def _dispatch_due_job(self, job: Any) -> str:
        """Claim one due job under cron_lock and dispatch it (or skip per misfire policy).

        Returns a telemetry outcome: fired|deduped|missed|skipped|locked|gone|not_due.
        """
        now = self._now_fn()
        lock_key = f"cron_lock:{job.id}"
        token = await dist_lock.acquire(self._redis, lock_key, px_ms=self._lock_px_ms)
        if token is None:
            return "locked"  # another sentinel is handling this job this tick
        try:
            fresh = await self._repo.get(job.id)
            if fresh is None or not fresh.enabled or fresh.next_run_at is None:
                return "gone"
            observed = fresh.next_run_at
            if observed > now:
                return "not_due"  # already advanced by another sentinel

            decision = resolve_misfire(
                fresh.misfire_policy, fresh.schedule_kind, observed, now, fresh.misfire_grace_s,
                every_s=fresh.schedule_every_s, expr=fresh.schedule_expr,
                tz=fresh.schedule_tz, at=fresh.schedule_at,
            )
            is_one_shot = fresh.schedule_kind == "at"

            outcome = "missed" if decision.missed else "skipped"
            if decision.action == "fire":
                det = self._det_key(job.id, observed)
                won = await self._redis.set(
                    RedisKeys.job(det), "pending", nx=True, ex=self._idem_ttl_s)
                if won:
                    try:
                        run_id = None
                        for _ in range(max(1, decision.fire_count)):
                            run_id = await self._dispatch_fn(fresh)
                        outcome = "fired"
                        if run_id:  # breadcrumb: replace the placeholder with the run id
                            await self._redis.set(
                                RedisKeys.job(det), str(run_id), ex=self._idem_ttl_s)
                    except Exception as e:
                        # Not (fully) posted → clear the occurrence key so the next scan retries,
                        # record the error, and DO NOT advance the schedule (job stays due).
                        await self._redis.delete(RedisKeys.job(det))
                        await self._repo.advance(
                            job.id, observed, observed, last_status="error",
                            last_error=str(e)[:500], run_at=now,
                            run_record={"run_at": now.isoformat(), "status": "error"})
                        return "error"
                else:
                    outcome = "deduped"  # occurrence already posted (crash-recovery)

            status = "ok" if outcome in ("fired", "deduped") else outcome
            run_record = {"run_at": now.isoformat(), "status": status}
            if is_one_shot:
                # Delete only when the occurrence actually fired; a missed/skipped one-shot is
                # disabled-with-record so the dropped reminder stays visible.
                delete = fresh.delete_after_run and outcome in ("fired", "deduped")
                await self._repo.finish_one_shot(
                    job.id, observed, delete=delete,
                    last_status=status, run_at=now, run_record=run_record)
            else:
                await self._repo.advance(
                    job.id, observed, decision.next_run,
                    last_status=status, run_at=now, run_record=run_record)
            return outcome
        finally:
            await dist_lock.release(self._redis, lock_key, token)

    async def _default_dispatch(self, job: Any) -> str | None:
        """Production dispatch: create a run + post to the mailbox (wired in Task 5/8)."""
        from nanoresearch.cron.dispatch import dispatch_cron_job

        return await dispatch_cron_job(self._redis, self._factory, job)
