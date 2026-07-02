"""CronJob repository — persistence + atomic occurrence claim (cron production redesign).

The scheduler reads due jobs with ``list_due`` and, after dispatching, calls
``advance`` / ``finish_one_shot``. Both are compare-and-swap on ``next_run_at``:
the update only applies when the row still holds the *observed* value, so a given
occurrence advances exactly once even if two schedulers race. Under PostgreSQL
READ COMMITTED, ``SELECT ... FOR UPDATE`` re-checks the WHERE against the latest
committed row after the lock is granted, so the loser sees a mismatch and no-ops.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanoresearch.storage.models import CronJob

_MAX_RUN_HISTORY = 20


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _append_capped(history: list | None, record: dict | None) -> list:
    hist = list(history or [])
    if record is not None:
        hist.append(record)
    return hist[-_MAX_RUN_HISTORY:]


class CronJobRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def create(
        self,
        *,
        uid: str,
        name: str,
        agent_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        schedule_kind: str,
        message: str,
        next_run_at: datetime | None,
        schedule_at: datetime | None = None,
        schedule_every_s: int | None = None,
        schedule_expr: str | None = None,
        schedule_tz: str | None = None,
        misfire_policy: str = "fire_once",
        misfire_grace_s: int = 3600,
        deliver: bool = False,
        deliver_channel: str | None = None,
        deliver_to: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        job = CronJob(
            uid=uid, name=name, agent_id=agent_id, conversation_id=conversation_id,
            schedule_kind=schedule_kind, schedule_at=schedule_at,
            schedule_every_s=schedule_every_s, schedule_expr=schedule_expr,
            schedule_tz=schedule_tz, message=message,
            misfire_policy=misfire_policy, misfire_grace_s=misfire_grace_s,
            deliver=deliver, deliver_channel=deliver_channel, deliver_to=deliver_to,
            next_run_at=next_run_at, delete_after_run=delete_after_run,
        )
        async with self._factory() as db:
            db.add(job)
            await db.commit()
            await db.refresh(job)
        return job

    async def get(self, job_id: uuid.UUID) -> CronJob | None:
        async with self._factory() as db:
            res = await db.execute(select(CronJob).where(CronJob.id == job_id))
            return res.scalar_one_or_none()

    async def list_by_uid(self, uid: str, *, include_disabled: bool = False) -> list[CronJob]:
        async with self._factory() as db:
            stmt = select(CronJob).where(CronJob.uid == uid)
            if not include_disabled:
                stmt = stmt.where(CronJob.enabled.is_(True))
            stmt = stmt.order_by(CronJob.created_at)
            res = await db.execute(stmt)
            return list(res.scalars().all())

    async def list_due(self, now: datetime, *, limit: int = 100) -> list[CronJob]:
        async with self._factory() as db:
            res = await db.execute(
                select(CronJob)
                .where(
                    CronJob.enabled.is_(True),
                    CronJob.next_run_at.is_not(None),
                    CronJob.next_run_at <= now,
                )
                .order_by(CronJob.next_run_at)
                .limit(limit)
            )
            return list(res.scalars().all())

    async def advance(
        self,
        job_id: uuid.UUID,
        observed_next_run_at: datetime | None,
        new_next_run_at: datetime | None,
        *,
        last_status: str,
        last_error: str | None = None,
        run_at: datetime,
        run_record: dict | None = None,
    ) -> bool:
        """CAS: move next_run_at observed→new. Returns True iff this call won the occurrence."""
        async with self._factory() as db:
            res = await db.execute(
                select(CronJob)
                .where(CronJob.id == job_id, CronJob.next_run_at == observed_next_run_at)
                .with_for_update()
            )
            job = res.scalar_one_or_none()
            if job is None:
                return False
            job.next_run_at = new_next_run_at
            job.last_status = last_status
            job.last_error = last_error
            job.last_run_at = run_at
            job.run_history = _append_capped(job.run_history, run_record)
            job.updated_at = _utcnow()
            await db.commit()
            return True

    async def finish_one_shot(
        self,
        job_id: uuid.UUID,
        observed_next_run_at: datetime | None,
        *,
        delete: bool,
        last_status: str,
        run_at: datetime,
        run_record: dict | None = None,
    ) -> bool:
        """CAS terminal transition for a one-shot job: delete the row, or disable + clear next_run."""
        async with self._factory() as db:
            res = await db.execute(
                select(CronJob)
                .where(CronJob.id == job_id, CronJob.next_run_at == observed_next_run_at)
                .with_for_update()
            )
            job = res.scalar_one_or_none()
            if job is None:
                return False
            if delete:
                await db.delete(job)
            else:
                job.enabled = False
                job.next_run_at = None
                job.last_status = last_status
                job.last_run_at = run_at
                job.run_history = _append_capped(job.run_history, run_record)
                job.updated_at = _utcnow()
            await db.commit()
            return True

    async def remove(self, job_id: uuid.UUID) -> bool:
        async with self._factory() as db:
            res = await db.execute(select(CronJob).where(CronJob.id == job_id))
            job = res.scalar_one_or_none()
            if job is None:
                return False
            await db.delete(job)
            await db.commit()
            return True

    async def set_enabled(self, job_id: uuid.UUID, enabled: bool) -> CronJob | None:
        async with self._factory() as db:
            res = await db.execute(select(CronJob).where(CronJob.id == job_id))
            job = res.scalar_one_or_none()
            if job is None:
                return None
            job.enabled = enabled
            job.updated_at = _utcnow()
            await db.commit()
            await db.refresh(job)
            return job
