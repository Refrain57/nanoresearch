"""Task 1: cron_jobs table + CronJob ORM model (production-grade cron redesign)."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, select

from nanoresearch.storage.models import CronJob, User
from tests.conftest import TEST_DB_URL, make_factory, pg_conn


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


def test_cron_jobs_tablename():
    assert CronJob.__tablename__ == "cron_jobs"


def test_cron_job_has_required_columns():
    cols = {c.name for c in inspect(CronJob).columns}
    required = {
        "id", "uid", "agent_id", "conversation_id", "name", "enabled",
        "schedule_kind", "schedule_at", "schedule_every_s", "schedule_expr", "schedule_tz",
        "message", "misfire_policy", "misfire_grace_s",
        "deliver", "deliver_channel", "deliver_to",
        "next_run_at", "last_run_at", "last_status", "last_error", "run_history",
        "delete_after_run", "created_at", "updated_at",
    }
    missing = required - cols
    assert not missing, f"CronJob missing columns: {missing}"


def test_cron_jobs_has_scan_index():
    idx_names = {idx.name for idx in CronJob.__table__.indexes}
    assert "ix_cron_jobs_enabled_next" in idx_names


def test_cron_job_roundtrip_and_defaults():
    async def _body():
        factory = make_factory()
        job_id = uuid.uuid4()
        async with factory() as db:
            db.add(User(uid="cronu", email="c@t.com", password_hash="x"))
            await db.flush()
            db.add(CronJob(
                id=job_id, uid="cronu", name="daily standup", enabled=True,
                schedule_kind="cron", schedule_expr="0 9 * * *",
                schedule_tz="America/Vancouver", message="standup",
                next_run_at=datetime(2026, 7, 3, 9, 0, tzinfo=timezone.utc),
            ))
            await db.commit()
        async with factory() as db:
            got = (await db.execute(select(CronJob).where(CronJob.id == job_id))).scalar_one()
            assert got.schedule_kind == "cron"
            assert got.schedule_expr == "0 9 * * *"
            assert got.schedule_tz == "America/Vancouver"
            # defaults locked in the spec
            assert got.misfire_policy == "fire_once"
            assert got.misfire_grace_s == 3600
            assert got.delete_after_run is False
            assert got.deliver is False
            assert got.run_history == []
            assert got.enabled is True

    run(_body())


def test_cron_jobs_table_create_idempotent():
    """The migration uses Table.create(checkfirst=True); creating an existing table is a no-op."""
    from sqlalchemy import create_engine

    sync_url = TEST_DB_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    eng = create_engine(sync_url, echo=False)
    try:
        CronJob.__table__.create(eng, checkfirst=True)  # already exists → must not raise
        CronJob.__table__.create(eng, checkfirst=True)
    finally:
        eng.dispose()
