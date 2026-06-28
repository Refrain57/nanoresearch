"""C1: updated_at must round-trip as tz-aware UTC so the idle gate math is correct."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from nanoresearch.utils.helpers import as_aware_utc, utcnow_aware
from nanoresearch.session.manager import Session, SessionManager
from tests.unit.session.test_redis_roundtrip import _FakeRedis


def test_utcnow_aware_is_timezone_aware():
    now = utcnow_aware()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_as_aware_utc_passes_through_aware():
    aware = datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)
    assert as_aware_utc(aware) == aware


def test_as_aware_utc_converts_offset_to_utc():
    plus8 = datetime(2026, 6, 29, 20, 0, tzinfo=timezone(timedelta(hours=8)))
    assert as_aware_utc(plus8) == datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)


def test_as_aware_utc_treats_naive_as_utc():
    naive = datetime(2026, 6, 29, 12, 0)
    assert as_aware_utc(naive) == datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)


def test_idle_delta_is_consistent_regardless_of_source_tz():
    """A 5-minute-old session must read as ~5 minutes idle whether the stored
    timestamp came back naive-UTC (Lua path) or aware-UTC (Redis path)."""
    now = utcnow_aware()
    five_min_ago_naive = (now - timedelta(minutes=5)).replace(tzinfo=None)
    five_min_ago_aware = now - timedelta(minutes=5)

    delta_naive = now - as_aware_utc(five_min_ago_naive)
    delta_aware = now - as_aware_utc(five_min_ago_aware)

    assert abs(delta_naive.total_seconds() - 300) < 1
    assert abs(delta_aware.total_seconds() - 300) < 1


async def test_redis_roundtrip_updated_at_is_aware_utc(tmp_path: Path):
    fake = _FakeRedis()
    manager = SessionManager(workspace=tmp_path)
    session = Session(key="web:tz-1", messages=[{"role": "user", "content": "hi"}])

    with patch("nanoresearch.bus.redis_client.get_redis", return_value=fake):
        await manager._redis_save(session)
        loaded = await manager._redis_load("web:tz-1")

    assert loaded is not None
    assert loaded.updated_at.tzinfo is not None
    delta = utcnow_aware() - loaded.updated_at
    assert -1 < delta.total_seconds() < 5
