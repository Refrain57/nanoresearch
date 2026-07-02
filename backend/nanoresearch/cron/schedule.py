"""Pure schedule math for the cron production redesign.

No I/O — just next-run computation and the misfire decision (spec §0 conclusion ①).
The scheduler sentinel calls these; keeping them pure makes the tricky timezone
and misfire logic unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

CRON_FIRE_ALL_CAP = 10


def compute_next_run(
    kind: str,
    *,
    at: datetime | None = None,
    every_s: int | None = None,
    expr: str | None = None,
    tz: str | None = None,
    after: datetime,
) -> datetime | None:
    """Next scheduled run strictly after *after* (tz-aware), or None if there is none."""
    if kind == "every":
        if not every_s or every_s <= 0:
            return None
        return after + timedelta(seconds=every_s)

    if kind == "cron":
        if not expr:
            return None
        from croniter import croniter

        zone = ZoneInfo(tz) if tz else timezone.utc
        base = after.astimezone(zone)
        return croniter(expr, base).get_next(datetime)

    if kind == "at":
        if at is None:
            return None
        return at if at > after else None

    return None


def _count_occurrences(
    kind: str, scheduled: datetime, now: datetime,
    *, every_s: int | None, expr: str | None, tz: str | None,
) -> int:
    """How many scheduled occurrences fall in [scheduled, now] (capped by the caller)."""
    if kind == "every":
        if not every_s or every_s <= 0:
            return 1
        elapsed = (now - scheduled).total_seconds()
        return int(elapsed // every_s) + 1

    if kind == "cron" and expr:
        from croniter import croniter

        zone = ZoneInfo(tz) if tz else timezone.utc
        it = croniter(expr, scheduled.astimezone(zone))
        count = 1  # the scheduled occurrence itself
        while count <= CRON_FIRE_ALL_CAP:
            nxt = it.get_next(datetime)
            if nxt > now:
                break
            count += 1
        return count

    return 1


@dataclass
class MisfireDecision:
    action: Literal["fire", "skip"]
    fire_count: int
    next_run: datetime | None
    missed: bool


def resolve_misfire(
    policy: str,
    kind: str,
    scheduled: datetime,
    now: datetime,
    grace_s: int,
    *,
    every_s: int | None = None,
    expr: str | None = None,
    tz: str | None = None,
    at: datetime | None = None,
) -> MisfireDecision:
    """Decide whether a due occurrence fires, how many times, and the next run time.

    - fire_once (default): fire once iff the occurrence is within the grace window,
      else skip and mark missed. Recurring jobs roll forward; one-shots do not.
    - skip: never fire on recovery; roll forward / disable.
    - fire_all: fire each missed occurrence up to CRON_FIRE_ALL_CAP (one-shots behave
      like fire_once).
    """
    missed = scheduled < (now - timedelta(seconds=grace_s))
    is_one_shot = kind == "at"
    next_run = None if is_one_shot else compute_next_run(
        kind, every_s=every_s, expr=expr, tz=tz, after=now)

    if policy == "skip":
        return MisfireDecision(action="skip", fire_count=0, next_run=next_run, missed=missed)

    if policy == "fire_all" and not is_one_shot:
        count = min(
            _count_occurrences(kind, scheduled, now, every_s=every_s, expr=expr, tz=tz),
            CRON_FIRE_ALL_CAP,
        )
        if count >= 1:
            return MisfireDecision(action="fire", fire_count=count, next_run=next_run, missed=missed)
        return MisfireDecision(action="skip", fire_count=0, next_run=next_run, missed=missed)

    # fire_once (and fire_all for one-shots)
    if missed:
        return MisfireDecision(action="skip", fire_count=0, next_run=next_run, missed=True)
    return MisfireDecision(action="fire", fire_count=1, next_run=next_run, missed=False)
