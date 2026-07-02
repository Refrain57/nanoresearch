"""Task 4: CronScheduler sentinel — claim + lock + idempotency + CAS.

test_two_schedulers_fire_once is the cross-process exactly-once command post:
N sentinels scanning the same due job dispatch it exactly once.
Redis + asyncpg share one SelectorEventLoop (Windows-safe) via run().
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import redis.asyncio as aioredis

from nanoresearch.bus.redis_keys import RedisKeys
from nanoresearch.cron.scheduler import CronScheduler
from nanoresearch.storage.models import User
from nanoresearch.storage.repositories.cron_repo import CronJobRepository
from tests.conftest import make_factory, pg_conn

UTC = timezone.utc
NOW = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
REDIS_URL = "redis://localhost:6379/15"


def run(coro):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def clean_cron():
    conn = pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE TABLE cron_jobs, agent_runs, messages, conversations, agents, users "
                "RESTART IDENTITY CASCADE"
            )
    finally:
        conn.close()


async def _redis():
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    await r.flushdb()
    return r


async def _seed_user(factory, uid="u"):
    async with factory() as db:
        db.add(User(uid=uid, email=f"{uid}@t.com", password_hash="x"))
        await db.commit()


async def _mk(repo, **over):
    kw = dict(
        uid="u", name="job", agent_id=None, conversation_id=None,
        schedule_kind="every", schedule_every_s=1200, schedule_expr=None,
        message="do it", next_run_at=NOW - timedelta(minutes=10),
    )
    kw.update(over)
    return await repo.create(**kw)


def _recorder():
    calls: list = []

    async def stub(job):
        calls.append(job.id)
        await asyncio.sleep(0)  # yield so concurrent scans interleave
        return f"run-{len(calls)}"

    return calls, stub


def test_due_job_dispatched_once():
    async def _body():
        r = await _redis()
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        job = await _mk(repo)
        calls, stub = _recorder()
        sched = CronScheduler(r, factory, dispatch_fn=stub, now_fn=lambda: NOW)
        await sched._scan_once()
        assert calls == [job.id]
        got = await repo.get(job.id)
        assert got.next_run_at > NOW      # advanced forward
        assert got.last_status == "ok"
        assert len(got.run_history) == 1
        await r.aclose()

    run(_body())


def test_two_schedulers_fire_once():
    async def _body():
        r = await _redis()
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        job = await _mk(repo)
        calls, stub = _recorder()
        s1 = CronScheduler(r, factory, dispatch_fn=stub, now_fn=lambda: NOW)
        s2 = CronScheduler(r, factory, dispatch_fn=stub, now_fn=lambda: NOW)
        await asyncio.gather(s1._scan_once(), s2._scan_once())
        assert calls == [job.id]           # exactly one dispatch across both sentinels
        got = await repo.get(job.id)
        assert len(got.run_history) == 1   # exactly one advance recorded
        await r.aclose()

    run(_body())


def test_crash_between_post_and_advance_no_double():
    """A prior attempt posted (det key set) but crashed before advancing → re-scan must
    not re-dispatch, but must still complete the advance."""
    async def _body():
        r = await _redis()
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        observed = NOW - timedelta(minutes=10)
        job = await _mk(repo, next_run_at=observed)
        det = CronScheduler._det_key(job.id, observed)
        await r.set(RedisKeys.job(det), "prior-run", nx=True, ex=3600)  # simulate posted, not advanced
        calls, stub = _recorder()
        sched = CronScheduler(r, factory, dispatch_fn=stub, now_fn=lambda: NOW)
        await sched._scan_once()
        assert calls == []                 # not re-dispatched (idempotency dedupe)
        got = await repo.get(job.id)
        assert got.next_run_at > NOW       # advance still ran
        await r.aclose()

    run(_body())


def test_stale_job_applies_misfire_skip():
    async def _body():
        r = await _redis()
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        job = await _mk(repo, next_run_at=NOW - timedelta(hours=3))  # outside 1h grace
        calls, stub = _recorder()
        sched = CronScheduler(r, factory, dispatch_fn=stub, now_fn=lambda: NOW)
        await sched._scan_once()
        assert calls == []                 # stale → not fired
        got = await repo.get(job.id)
        assert got.last_status == "missed"
        assert got.next_run_at > NOW       # rolled forward, not stuck

    run(_body())


def test_one_shot_fired_then_deleted():
    async def _body():
        r = await _redis()
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        observed = NOW - timedelta(minutes=5)
        job = await _mk(repo, schedule_kind="at", schedule_every_s=None,
                        schedule_at=observed, next_run_at=observed, delete_after_run=True)
        calls, stub = _recorder()
        sched = CronScheduler(r, factory, dispatch_fn=stub, now_fn=lambda: NOW)
        await sched._scan_once()
        assert calls == [job.id]
        assert await repo.get(job.id) is None  # one-shot deleted after firing
        await r.aclose()

    run(_body())


def test_disabled_job_not_dispatched():
    async def _body():
        r = await _redis()
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        job = await _mk(repo)
        await repo.set_enabled(job.id, False)
        calls, stub = _recorder()
        sched = CronScheduler(r, factory, dispatch_fn=stub, now_fn=lambda: NOW)
        await sched._scan_once()
        assert calls == []
        await r.aclose()

    run(_body())
