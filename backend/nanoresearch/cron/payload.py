"""Build a run payload for a cron job, targeting its dedicated conversation.

Reuses chat_router._build_run_payload so a cron run is indistinguishable from a
normal user run to the dispatcher/worker (spec conclusion ②): same agent config,
session_key = web:{cron_conversation_id}, so its agent_lock never contends with
user chat. Adds a cron_delivery block consumed by the phase-2 delivery gate.
"""
from __future__ import annotations

from typing import Any


def _wrap_cron_task(job: Any) -> str:
    """Wrap the scheduled instruction as a turn-starting message (cf. legacy on_cron_job)."""
    return (
        "[Scheduled Task] Timer finished.\n\n"
        f"Task '{job.name}' has been triggered.\n"
        f"Scheduled instruction: {job.message}"
    )


async def build_cron_run_payload(factory: Any, job: Any, run_id: str) -> dict:
    from nanoresearch.server.routers.chat_router import _build_run_payload

    payload = await _build_run_payload(
        factory,
        str(job.conversation_id),
        job.uid,
        content=_wrap_cron_task(job),
        run_id=run_id,
        agent_id=(str(job.agent_id) if job.agent_id else None),
    )
    payload["cron_delivery"] = {
        "deliver": job.deliver,
        "channel": job.deliver_channel,
        "to": job.deliver_to,
        "task_context": job.message,
    }
    return payload
