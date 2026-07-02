"""Task 2: CronJobRepository — CRUD, due-scan, and the atomic CAS advance.

test_advance_cas_wins_once is the exactly-once command post: the schedule may
only be advanced once per observed occurrence, across processes.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from nanoresearch.storage.models import User
from nanoresearch.storage.repositories.cron_repo import CronJobRepository
from tests.conftest import make_factory, pg_conn

UTC = timezone.utc


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


async def _seed_user(factory, uid="u"):
    async with factory() as db:
        db.add(User(uid=uid, email=f"{uid}@t.com", password_hash="x"))
        await db.commit()


async def _mk(repo, **over):
    kw = dict(
        uid="u", name="job", agent_id=None, conversation_id=None,
        schedule_kind="cron", schedule_expr="0 9 * * *", message="do it",
        next_run_at=datetime(2026, 7, 1, 9, 0, tzinfo=UTC),
    )
    kw.update(over)
    return await repo.create(**kw)


def test_create_get_and_list_by_uid():
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        job = await _mk(repo, name="daily")
        got = await repo.get(job.id)
        assert got is not None and got.name == "daily"
        assert got.misfire_policy == "fire_once" and got.misfire_grace_s == 3600
        listed = await repo.list_by_uid("u")
        assert [j.id for j in listed] == [job.id]

    run(_body())


def test_list_due_returns_only_past_enabled():
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
        j_past = await _mk(repo, next_run_at=now - timedelta(minutes=5))
        j_future = await _mk(repo, next_run_at=now + timedelta(hours=1))
        j_disabled = await _mk(repo, next_run_at=now - timedelta(minutes=5))
        await repo.set_enabled(j_disabled.id, False)

        due_ids = {j.id for j in await repo.list_due(now)}
        assert j_past.id in due_ids
        assert j_future.id not in due_ids
        assert j_disabled.id not in due_ids

    run(_body())


def test_advance_cas_wins_once():
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        observed = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
        new = datetime(2026, 7, 2, 9, 0, tzinfo=UTC)
        job = await _mk(repo, next_run_at=observed)

        ok1 = await repo.advance(
            job.id, observed, new, last_status="ok", run_at=observed,
            run_record={"run_at": observed.isoformat(), "status": "ok"})
        ok2 = await repo.advance(
            job.id, observed, new, last_status="ok", run_at=observed,
            run_record={"run_at": observed.isoformat(), "status": "ok"})

        assert ok1 is True   # first observer claims the occurrence
        assert ok2 is False  # second observer sees next_run already advanced → no-op
        got = await repo.get(job.id)
        assert got.next_run_at == new
        assert len(got.run_history) == 1  # only the winning advance recorded a run

    run(_body())


def test_advance_appends_run_history_capped():
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        cur = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
        job = await _mk(repo, next_run_at=cur)
        for i in range(25):
            nxt = cur + timedelta(days=1)
            ok = await repo.advance(
                job.id, cur, nxt, last_status="ok", run_at=cur,
                run_record={"i": i, "status": "ok"})
            assert ok is True
            cur = nxt
        got = await repo.get(job.id)
        assert len(got.run_history) == 20
        assert got.run_history[-1]["i"] == 24  # most recent retained

    run(_body())


def test_finish_one_shot_delete_removes_row():
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        observed = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
        job = await _mk(repo, schedule_kind="at",
                        schedule_at=observed, schedule_expr=None,
                        next_run_at=observed, delete_after_run=True)
        ok = await repo.finish_one_shot(
            job.id, observed, delete=True, last_status="ok", run_at=observed,
            run_record={"status": "ok"})
        assert ok is True
        assert await repo.get(job.id) is None

    run(_body())


def test_finish_one_shot_disable_keeps_row_next_null():
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        observed = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
        job = await _mk(repo, schedule_kind="at",
                        schedule_at=observed, schedule_expr=None,
                        next_run_at=observed)
        ok = await repo.finish_one_shot(
            job.id, observed, delete=False, last_status="missed", run_at=observed,
            run_record={"status": "missed"})
        assert ok is True
        got = await repo.get(job.id)
        assert got is not None
        assert got.enabled is False
        assert got.next_run_at is None
        assert got.last_status == "missed"

    run(_body())


def test_remove_deletes_row():
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        repo = CronJobRepository(factory)
        job = await _mk(repo)
        assert await repo.remove(job.id) is True
        assert await repo.get(job.id) is None
        assert await repo.remove(job.id) is False  # already gone

    run(_body())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
