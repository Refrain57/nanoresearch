"""Task 3: cron/schedule.py — next-run computation + misfire policy (pure functions)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from nanoresearch.cron.schedule import (
    CRON_FIRE_ALL_CAP,
    compute_next_run,
    resolve_misfire,
)

UTC = timezone.utc


# ── compute_next_run ──────────────────────────────────────────────────────────

def test_compute_next_every():
    after = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    assert compute_next_run("every", every_s=1200, after=after) == after + timedelta(seconds=1200)


def test_compute_next_cron_with_tz():
    after = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    nxt = compute_next_run("cron", expr="0 9 * * *", tz="America/Vancouver", after=after)
    local = nxt.astimezone(ZoneInfo("America/Vancouver"))
    assert (local.hour, local.minute) == (9, 0)
    assert nxt > after


def test_compute_next_at_past_returns_none():
    after = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    past = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    assert compute_next_run("at", at=past, after=after) is None


def test_compute_next_at_future():
    after = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    fut = datetime(2026, 7, 1, 15, 0, tzinfo=UTC)
    assert compute_next_run("at", at=fut, after=after) == fut


def test_cron_dst_boundary():
    """A 09:00 daily cron near Vancouver spring-forward still yields local 09:00."""
    tz = "America/Vancouver"
    before = datetime(2026, 3, 7, 20, 0, tzinfo=UTC)
    nxt = compute_next_run("cron", expr="0 9 * * *", tz=tz, after=before)
    assert nxt.astimezone(ZoneInfo(tz)).hour == 9


# ── resolve_misfire ───────────────────────────────────────────────────────────

def test_misfire_within_grace_fires_once():
    now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    scheduled = now - timedelta(minutes=10)
    d = resolve_misfire("fire_once", "every", scheduled, now, 3600, every_s=1200)
    assert d.action == "fire"
    assert d.fire_count == 1
    assert d.missed is False
    assert d.next_run > now


def test_misfire_outside_grace_skips_and_rolls_forward():
    now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    scheduled = now - timedelta(hours=3)
    d = resolve_misfire("fire_once", "every", scheduled, now, 3600, every_s=1200)
    assert d.action == "skip"
    assert d.missed is True
    assert d.next_run > now  # recurring: rolled forward, not stuck


def test_misfire_policy_skip_never_fires():
    now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    scheduled = now - timedelta(minutes=1)  # within grace, but policy=skip
    d = resolve_misfire("skip", "every", scheduled, now, 3600, every_s=1200)
    assert d.action == "skip"
    assert d.fire_count == 0
    assert d.next_run > now


def test_misfire_policy_fire_all_capped_at_10():
    now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    scheduled = now - timedelta(seconds=1200 * 50)  # 50 missed intervals
    d = resolve_misfire("fire_all", "every", scheduled, now, 3600, every_s=1200)
    assert d.action == "fire"
    assert d.fire_count == CRON_FIRE_ALL_CAP == 10
    assert d.next_run > now


def test_misfire_at_outside_grace_marks_missed_next_none():
    now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    scheduled = now - timedelta(hours=5)
    d = resolve_misfire("fire_once", "at", scheduled, now, 3600, at=scheduled)
    assert d.action == "skip"
    assert d.missed is True
    assert d.next_run is None  # one-shot never rolls forward


def test_misfire_at_within_grace_fires_once_next_none():
    now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    scheduled = now - timedelta(minutes=5)
    d = resolve_misfire("fire_once", "at", scheduled, now, 3600, at=scheduled)
    assert d.action == "fire"
    assert d.fire_count == 1
    assert d.next_run is None
