"""Task 7: CronTool (DB-backed) — add/list/remove write cron_jobs + dedicated conversation."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from nanoresearch.agent.tools.cron import CronTool
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


async def _tool(factory, uid="u"):
    async with factory() as db:
        db.add(User(uid=uid, email=f"{uid}@t.com", password_hash="x"))
        await db.commit()
    return CronTool(CronJobRepository(factory), ConversationRepository(factory), uid=uid)


def test_add_creates_job_and_dedicated_conversation():
    async def _body():
        factory = make_factory()
        tool = await _tool(factory)
        res = await tool.execute(action="add", message="Check GitHub stars", every_seconds=1200)
        assert "Created job" in res
        jobs = await CronJobRepository(factory).list_by_uid("u")
        assert len(jobs) == 1
        j = jobs[0]
        assert j.conversation_id is not None
        conv = await ConversationRepository(factory).get_by_id(j.conversation_id)
        assert conv is not None
        assert conv.session_key == f"web:{j.conversation_id}"        # conclusion ②
        assert conv.conv_metadata.get("cron_job_id") == str(j.id)     # hideable from chat list

    run(_body())


def test_add_computes_next_run_for_every():
    async def _body():
        factory = make_factory()
        tool = await _tool(factory)
        await tool.execute(action="add", message="ping", every_seconds=1200)
        j = (await CronJobRepository(factory).list_by_uid("u"))[0]
        delta = (j.next_run_at - datetime.now(UTC)).total_seconds()
        assert 1150 < delta <= 1201

    run(_body())


def test_add_cron_expr_sets_schedule_and_next_run():
    async def _body():
        factory = make_factory()
        tool = await _tool(factory)
        await tool.execute(action="add", message="standup", cron_expr="0 9 * * *",
                           tz="America/Vancouver")
        j = (await CronJobRepository(factory).list_by_uid("u"))[0]
        assert j.schedule_kind == "cron"
        assert j.schedule_expr == "0 9 * * *"
        assert j.schedule_tz == "America/Vancouver"
        assert j.next_run_at is not None and j.next_run_at > datetime.now(UTC)

    run(_body())


def test_list_reads_from_db():
    async def _body():
        factory = make_factory()
        tool = await _tool(factory)
        await tool.execute(action="add", message="job A", every_seconds=600)
        await tool.execute(action="add", message="job B", every_seconds=1200)
        out = await tool.execute(action="list")
        assert "job A" in out and "job B" in out

    run(_body())


def test_remove_deletes_and_is_uid_scoped():
    async def _body():
        factory = make_factory()
        tool = await _tool(factory)
        await tool.execute(action="add", message="temp", every_seconds=600)
        job = (await CronJobRepository(factory).list_by_uid("u"))[0]
        res = await tool.execute(action="remove", job_id=str(job.id))
        assert "Removed" in res
        assert await CronJobRepository(factory).list_by_uid("u") == []
        # removing a non-existent / other id → not found
        assert "not found" in await tool.execute(action="remove", job_id=str(job.id))

    run(_body())


def test_self_schedule_blocked():
    async def _body():
        factory = make_factory()
        tool = await _tool(factory)
        token = tool.set_cron_context(True)
        try:
            res = await tool.execute(action="add", message="loop", every_seconds=60)
        finally:
            tool.reset_cron_context(token)
        assert "cannot schedule new jobs from within a cron job" in res
        assert await CronJobRepository(factory).list_by_uid("u") == []

    run(_body())


def test_tz_only_with_cron_expr():
    async def _body():
        factory = make_factory()
        tool = await _tool(factory)
        res = await tool.execute(action="add", message="x", every_seconds=60,
                                 tz="America/Vancouver")
        assert "tz can only be used with cron_expr" in res

    run(_body())


def test_unknown_tz_rejected():
    async def _body():
        factory = make_factory()
        tool = await _tool(factory)
        res = await tool.execute(action="add", message="x", cron_expr="0 9 * * *",
                                 tz="Not/AZone")
        assert "unknown timezone" in res

    run(_body())
