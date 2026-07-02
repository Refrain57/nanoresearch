"""One-time migration of the legacy JSON cron store into the cron_jobs table.

Reads <workspace>/cron/jobs.json (the CronService on-disk format) and creates a
cron_jobs row + a dedicated conversation per job. Idempotent: a job is skipped if
a row with the same (uid, name, schedule fingerprint) already exists, so re-running
is safe.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoresearch.cron.schedule import compute_next_run
from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
from nanoresearch.storage.repositories.cron_repo import CronJobRepository


def _fingerprint(kind: str, *, at_ms, every_s, expr, tz) -> tuple:
    return (kind, at_ms, every_s, expr, tz)


def _job_fingerprint_from_row(job: Any) -> tuple:
    at_ms = int(job.schedule_at.timestamp() * 1000) if job.schedule_at else None
    return _fingerprint(job.schedule_kind, at_ms=at_ms, every_s=job.schedule_every_s,
                        expr=job.schedule_expr, tz=job.schedule_tz)


async def migrate_jobs_json(json_path: Path, factory: Any, *, default_uid: str) -> int:
    """Migrate jobs.json → cron_jobs. Returns the number of NEW jobs created."""
    json_path = Path(json_path)
    if not json_path.is_file():
        return 0
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    cron_repo = CronJobRepository(factory)
    conv_repo = ConversationRepository(factory)

    existing = await cron_repo.list_by_uid(default_uid, include_disabled=True)
    seen = {(j.name, _job_fingerprint_from_row(j)) for j in existing}

    now = datetime.now(timezone.utc)
    created = 0
    for j in data.get("jobs", []):
        name = j.get("name") or "cron"
        sched = j.get("schedule", {})
        kind = sched.get("kind")
        at_ms = sched.get("atMs")
        every_ms = sched.get("everyMs")
        expr = sched.get("expr")
        tz = sched.get("tz")

        schedule_at = every_s = None
        if kind == "at":
            schedule_at = datetime.fromtimestamp(at_ms / 1000, tz=timezone.utc) if at_ms else None
            next_run = schedule_at
        elif kind == "every":
            every_s = int(every_ms / 1000) if every_ms else None
            next_run = compute_next_run("every", every_s=every_s, after=now)
        elif kind == "cron":
            next_run = compute_next_run("cron", expr=expr, tz=tz, after=now)
        else:
            continue  # unknown schedule kind — skip

        fp = (name, _fingerprint(kind, at_ms=at_ms, every_s=every_s, expr=expr, tz=tz))
        if fp in seen:
            continue
        seen.add(fp)

        payload = j.get("payload", {})
        conv_id = uuid.uuid4()
        await conv_repo.create(
            key=f"web:{conv_id}", uid=default_uid, conv_id=conv_id,
            title=f"[cron] {name[:40]}", metadata={"cron": True})
        job = await cron_repo.create(
            uid=default_uid, name=name, agent_id=None, conversation_id=conv_id,
            schedule_kind=kind, schedule_at=schedule_at, schedule_every_s=every_s,
            schedule_expr=expr, schedule_tz=tz, message=payload.get("message", ""),
            deliver=bool(payload.get("deliver", False)),
            deliver_channel=payload.get("channel"), deliver_to=payload.get("to"),
            next_run_at=next_run, delete_after_run=bool(j.get("deleteAfterRun", False)))
        await conv_repo.update_meta(
            conv_id, 0, {"cron": True, "cron_job_id": str(job.id)}, datetime.now(timezone.utc))
        if not j.get("enabled", True):
            await cron_repo.set_enabled(job.id, False)
        created += 1

    return created
