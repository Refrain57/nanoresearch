"""Cron tool — schedule reminders and tasks (DB-backed, production redesign).

add/list/remove now persist to the cron_jobs table (via CronJobRepository) instead
of the legacy JSON store. Each `add` also creates a dedicated conversation for the
job (session_key = web:{conversation_id}) so its runs never contend with user chat
(spec conclusion ②). The CronScheduler sentinel picks jobs up from the DB.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from nanoresearch.agent.tools.base import Tool
from nanoresearch.cron.schedule import compute_next_run
from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
from nanoresearch.storage.repositories.cron_repo import CronJobRepository

_NON_DELIVERABLE = {"", "web", "cli", "system"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CronTool(Tool):
    """Tool to schedule reminders and recurring tasks (writes the cron_jobs table)."""

    def __init__(
        self,
        cron_repo: CronJobRepository,
        conv_repo: ConversationRepository,
        uid: str,
        default_timezone: str = "UTC",
    ):
        self._cron = cron_repo
        self._conv = conv_repo
        self._uid = uid
        self._default_timezone = default_timezone
        self._channel = ""
        self._chat_id = ""
        self._in_cron_context: ContextVar[bool] = ContextVar("cron_in_context", default=False)

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current session context (used to pick a delivery target)."""
        self._channel = channel
        self._chat_id = chat_id

    def set_cron_context(self, active: bool):
        """Mark whether the tool is executing inside a cron job callback."""
        return self._in_cron_context.set(active)

    def reset_cron_context(self, token) -> None:
        self._in_cron_context.reset(token)

    @staticmethod
    def _validate_timezone(tz: str) -> str | None:
        try:
            ZoneInfo(tz)
        except (KeyError, Exception):
            return f"Error: unknown timezone '{tz}'"
        return None

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Schedule reminders and recurring tasks. Actions: add, list, remove. "
            f"If tz is omitted, cron expressions and naive ISO times default to {self._default_timezone}."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],
                    "description": "Action to perform",
                },
                "message": {"type": "string", "description": "Reminder message (for add)"},
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds (for recurring tasks)",
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression like '0 9 * * *' (for scheduled tasks)",
                },
                "tz": {
                    "type": "string",
                    "description": (
                        "Optional IANA timezone for cron expressions "
                        f"(e.g. 'America/Vancouver'). Defaults to {self._default_timezone}."
                    ),
                },
                "at": {
                    "type": "string",
                    "description": (
                        "ISO datetime for one-time execution "
                        f"(e.g. '2026-02-12T10:30:00'). Naive values default to {self._default_timezone}."
                    ),
                },
                "job_id": {"type": "string", "description": "Job ID (for remove)"},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "add":
            if self._in_cron_context.get():
                return "Error: cannot schedule new jobs from within a cron job execution"
            return await self._add_job(message, every_seconds, cron_expr, tz, at)
        if action == "list":
            return await self._list_jobs()
        if action == "remove":
            return await self._remove_job(job_id)
        return f"Unknown action: {action}"

    async def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
    ) -> str:
        if not message:
            return "Error: message is required for add"
        if tz and not cron_expr:
            return "Error: tz can only be used with cron_expr"
        if tz:
            if err := self._validate_timezone(tz):
                return err

        now = _now()
        kind = None
        schedule_at = every_s = expr = eff_tz = None
        delete_after = False

        if every_seconds:
            kind, every_s = "every", every_seconds
            next_run = now + timedelta(seconds=every_seconds)
        elif cron_expr:
            eff_tz = tz or self._default_timezone
            if err := self._validate_timezone(eff_tz):
                return err
            kind, expr = "cron", cron_expr
            next_run = compute_next_run("cron", expr=expr, tz=eff_tz, after=now)
        elif at:
            try:
                dt = datetime.fromisoformat(at)
            except ValueError:
                return f"Error: invalid ISO datetime format '{at}'. Expected format: YYYY-MM-DDTHH:MM:SS"
            if dt.tzinfo is None:
                if err := self._validate_timezone(self._default_timezone):
                    return err
                dt = dt.replace(tzinfo=ZoneInfo(self._default_timezone))
            kind, schedule_at, next_run = "at", dt, dt
            delete_after = True
        else:
            return "Error: either every_seconds, cron_expr, or at is required"

        deliver = self._channel not in _NON_DELIVERABLE

        conv_id = uuid.uuid4()
        await self._conv.create(
            key=f"web:{conv_id}", uid=self._uid, conv_id=conv_id,
            title=f"[cron] {message[:40]}", metadata={"cron": True})
        job = await self._cron.create(
            uid=self._uid, name=message[:30], agent_id=None, conversation_id=conv_id,
            schedule_kind=kind, schedule_at=schedule_at, schedule_every_s=every_s,
            schedule_expr=expr, schedule_tz=eff_tz, message=message,
            deliver=deliver,
            deliver_channel=(self._channel if deliver else None),
            deliver_to=(self._chat_id if deliver else None),
            next_run_at=next_run, delete_after_run=delete_after)
        await self._conv.update_meta(
            conv_id, 0, {"cron": True, "cron_job_id": str(job.id)}, _now())
        return f"Created job '{job.name}' (id: {job.id})"

    def _format_timing(self, job) -> str:
        if job.schedule_kind == "cron":
            tz = f" ({job.schedule_tz})" if job.schedule_tz else ""
            return f"cron: {job.schedule_expr}{tz}"
        if job.schedule_kind == "every" and job.schedule_every_s:
            s = job.schedule_every_s
            if s % 3600 == 0:
                return f"every {s // 3600}h"
            if s % 60 == 0:
                return f"every {s // 60}m"
            return f"every {s}s"
        if job.schedule_kind == "at" and job.schedule_at:
            return f"at {job.schedule_at.isoformat()}"
        return job.schedule_kind

    async def _list_jobs(self) -> str:
        jobs = await self._cron.list_by_uid(self._uid)
        if not jobs:
            return "No scheduled jobs."
        lines = []
        for j in jobs:
            parts = [f"- {j.name} (id: {j.id}, {self._format_timing(j)})"]
            if j.next_run_at:
                parts.append(f"  Next run: {j.next_run_at.isoformat()}")
            if j.last_run_at:
                info = f"  Last run: {j.last_run_at.isoformat()} — {j.last_status or 'unknown'}"
                if j.last_error:
                    info += f" ({j.last_error})"
                parts.append(info)
            lines.append("\n".join(parts))
        return "Scheduled jobs:\n" + "\n".join(lines)

    async def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"
        try:
            jid = uuid.UUID(job_id)
        except ValueError:
            return f"Job {job_id} not found"
        job = await self._cron.get(jid)
        if job is None or job.uid != self._uid:  # uid-scoped: no cross-tenant removal
            return f"Job {job_id} not found"
        await self._cron.remove(jid)
        return f"Removed job {job_id}"
