"""Real cron dispatch: create a run + post it to the mailbox (default_dispatch target).

Routes a due cron job through the exact same path as a user message —
_enqueue_via_mailbox → AgentDispatcher → worker — so cron reuses the mailbox's
concurrency protection and crash recovery instead of a second runtime.
"""
from __future__ import annotations

from typing import Any


async def dispatch_cron_job(redis: Any, factory: Any, job: Any) -> str:
    """Create the AgentRun for this occurrence and post it to the cron conversation's inbox.

    Returns the new run_id. Assumes the caller (CronScheduler) already holds the
    per-job cron_lock and passed the deterministic idempotency gate.
    """
    from nanoresearch.cron.payload import build_cron_run_payload
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox
    from nanoresearch.storage.repositories.run_repo import RunRepository

    run = await RunRepository(factory).create(
        conversation_id=job.conversation_id, uid=job.uid, agent_id=job.agent_id)
    run_id = str(run.id)
    payload = await build_cron_run_payload(factory, job, run_id)
    await _enqueue_via_mailbox(redis, payload)
    return run_id
