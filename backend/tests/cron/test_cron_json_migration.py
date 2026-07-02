"""Task 9: migrate legacy <workspace>/cron/jobs.json into the cron_jobs table."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from nanoresearch.cron.migrate_json import migrate_jobs_json
from nanoresearch.storage.models import User
from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
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


def _sample(path):
    data = {
        "version": 1,
        "jobs": [
            {
                "id": "a1", "name": "one-time", "enabled": True,
                "schedule": {"kind": "at", "atMs": int(datetime(2026, 8, 1, 9, 0, tzinfo=UTC).timestamp() * 1000)},
                "payload": {"kind": "agent_turn", "message": "remind meeting",
                            "deliver": True, "channel": "telegram", "to": "999"},
                "deleteAfterRun": True,
            },
            {
                "id": "e1", "name": "every10m", "enabled": True,
                "schedule": {"kind": "every", "everyMs": 600000},
                "payload": {"message": "check stars", "deliver": False},
            },
            {
                "id": "c1", "name": "standup", "enabled": True,
                "schedule": {"kind": "cron", "expr": "0 9 * * *", "tz": "America/Vancouver"},
                "payload": {"message": "standup", "deliver": False},
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


async def _seed_user(factory, uid="admin"):
    async with factory() as db:
        db.add(User(uid=uid, email=f"{uid}@t.com", password_hash="x"))
        await db.commit()


def test_migrate_sample_jobs(tmp_path):
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        jf = tmp_path / "jobs.json"
        _sample(jf)

        n = await migrate_jobs_json(jf, factory, default_uid="admin")
        assert n == 3

        jobs = {j.name: j for j in await CronJobRepository(factory).list_by_uid("admin", include_disabled=True)}
        assert set(jobs) == {"one-time", "every10m", "standup"}

        at_job = jobs["one-time"]
        assert at_job.schedule_kind == "at"
        assert at_job.schedule_at is not None
        assert at_job.delete_after_run is True
        assert at_job.deliver is True
        assert at_job.deliver_channel == "telegram"
        assert at_job.deliver_to == "999"

        every_job = jobs["every10m"]
        assert every_job.schedule_kind == "every"
        assert every_job.schedule_every_s == 600

        cron_job = jobs["standup"]
        assert cron_job.schedule_kind == "cron"
        assert cron_job.schedule_expr == "0 9 * * *"
        assert cron_job.schedule_tz == "America/Vancouver"
        assert cron_job.next_run_at is not None

        # each migrated job has its own conversation (web:{id}) marked as cron
        conv_repo = ConversationRepository(factory)
        for j in jobs.values():
            assert j.conversation_id is not None
            conv = await conv_repo.get_by_id(j.conversation_id)
            assert conv is not None
            assert conv.session_key == f"web:{j.conversation_id}"
            assert conv.conv_metadata.get("cron_job_id") == str(j.id)

    run(_body())


def test_migrate_idempotent(tmp_path):
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        jf = tmp_path / "jobs.json"
        _sample(jf)
        first = await migrate_jobs_json(jf, factory, default_uid="admin")
        second = await migrate_jobs_json(jf, factory, default_uid="admin")
        assert first == 3
        assert second == 0  # nothing new on re-run
        jobs = await CronJobRepository(factory).list_by_uid("admin", include_disabled=True)
        assert len(jobs) == 3  # no duplicates

    run(_body())


def test_migrate_missing_file_noop(tmp_path):
    async def _body():
        factory = make_factory()
        await _seed_user(factory)
        n = await migrate_jobs_json(tmp_path / "nope.json", factory, default_uid="admin")
        assert n == 0

    run(_body())
