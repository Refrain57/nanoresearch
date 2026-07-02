"""Task 5: cron payload builder + real mailbox dispatch (dedicated cron conversation)."""
from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import redis.asyncio as aioredis

from nanoresearch.bus import mailbox
from nanoresearch.bus.redis_keys import RedisKeys
from nanoresearch.cron.dispatch import dispatch_cron_job
from nanoresearch.cron.payload import build_cron_run_payload
from nanoresearch.storage.models import Conversation, User
from nanoresearch.storage.repositories.cron_repo import CronJobRepository
from nanoresearch.storage.repositories.run_repo import RunRepository
from nanoresearch.worker import run_agent_job
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


async def _seed(factory, *, deliver=False):
    """Seed a user + dedicated cron conversation + a cron job; return (conv_id, job)."""
    conv_id = uuid.uuid4()
    async with factory() as db:
        db.add(User(uid="u", email="u@t.com", password_hash="x"))
        await db.flush()
        db.add(Conversation(
            id=conv_id, uid="u", session_key=f"web:{conv_id}", channel="web",
            conv_metadata={"cron_job_id": "pending"}))
        await db.commit()
    repo = CronJobRepository(factory)
    job = await repo.create(
        uid="u", name="check stars", agent_id=None, conversation_id=conv_id,
        schedule_kind="every", schedule_every_s=600, message="Check GitHub stars and report",
        next_run_at=NOW - timedelta(minutes=1),
        deliver=deliver, deliver_channel=("telegram" if deliver else None),
        deliver_to=("123" if deliver else None))
    return conv_id, job


def test_payload_targets_cron_conversation():
    async def _body():
        factory = make_factory()
        conv_id, job = await _seed(factory)
        payload = await build_cron_run_payload(factory, job, run_id="r1")
        assert payload["session_key"] == f"web:{conv_id}"
        assert payload["conversation_id"] == str(conv_id)
        assert payload["run_id"] == "r1"
        assert payload["agent_id"] is None  # job has no agent

    run(_body())


def test_payload_wraps_task_message():
    async def _body():
        factory = make_factory()
        _conv_id, job = await _seed(factory)
        payload = await build_cron_run_payload(factory, job, run_id="r1")
        assert "[Scheduled Task]" in payload["content"]
        assert "Check GitHub stars and report" in payload["content"]

    run(_body())


def test_payload_carries_cron_metadata():
    async def _body():
        factory = make_factory()
        _conv_id, job = await _seed(factory, deliver=True)
        payload = await build_cron_run_payload(factory, job, run_id="r1")
        cron = payload["cron"]
        assert cron["deliver"] is True
        assert cron["channel"] == "telegram"
        assert cron["to"] == "123"
        assert cron["task_context"] == "Check GitHub stars and report"

    run(_body())


def test_cron_payload_keys_accepted_by_run_agent_job():
    """Every payload key the dispatcher splats must be a run_agent_job param (no TypeError)."""
    async def _body():
        factory = make_factory()
        _conv_id, job = await _seed(factory, deliver=True)
        payload = await build_cron_run_payload(factory, job, run_id="r1")
        accepted = set(inspect.signature(run_agent_job).parameters)
        extra = set(payload) - accepted
        assert not extra, f"payload keys rejected by run_agent_job: {extra}"

    run(_body())


def test_dispatch_creates_run_and_posts_to_mailbox():
    async def _body():
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        await r.flushdb()
        factory = make_factory()
        conv_id, job = await _seed(factory)

        run_id = await dispatch_cron_job(r, factory, job)

        # a run row was created for the cron conversation
        run_row = await RunRepository(factory).get(uuid.UUID(run_id))
        assert run_row is not None
        assert str(run_row.conversation_id) == str(conv_id)

        # the run was posted to the (agent_id=none, conv) inbox + a notify emitted
        got = await mailbox.read_next_after_cursor(r, "none", str(conv_id))
        assert got is not None
        assert got[1]["run_id"] == run_id
        assert "[Scheduled Task]" in got[1]["content"]
        assert await r.xlen(RedisKeys.DISPATCH_NOTIFY) >= 1
        await r.aclose()

    run(_body())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
