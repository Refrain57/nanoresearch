"""Task 8: the FastAPI serve lifespan starts (and stops) the CronScheduler.

This is the fix for the fatal 'web never fires' bug — under serve+worker the
scheduler must be a long-lived sentinel like the dispatcher/watchdog. Heavy
startup deps are stubbed so we test only the wiring.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nanoresearch.server.main import create_app


class _Stub:
    def __init__(self, *a, **k):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass

    async def aclose(self):
        pass


@pytest.fixture
def cron_events(monkeypatch):
    events: dict = {}

    class _CronStub:
        def __init__(self, redis, factory, **kw):
            events["constructed"] = True

        async def start(self):
            events["started"] = True

        async def stop(self):
            events["stopped"] = True

    async def _create_pool(*a, **k):
        return _Stub()

    monkeypatch.setattr("nanoresearch.config.migration.migrate_llm_keys", lambda **k: None)
    monkeypatch.setattr("nanoresearch.storage.database._engine", None)
    monkeypatch.setattr("nanoresearch.bus.redis_client.get_redis", lambda: _Stub())
    monkeypatch.setattr("nanoresearch.bus.pending_reaper.PendingReaper", _Stub)
    monkeypatch.setattr("nanoresearch.bus.redis_monitor.RedisMonitor", _Stub)
    monkeypatch.setattr("arq.create_pool", _create_pool)
    monkeypatch.setattr("nanoresearch.bus.dispatcher.AgentDispatcher", _Stub)
    monkeypatch.setattr("nanoresearch.heartbeat.stuck_run_watchdog.StuckRunWatchdog", _Stub)
    monkeypatch.setattr("nanoresearch.cron.scheduler.CronScheduler", _CronStub)
    return events


async def test_serve_lifespan_starts_and_stops_cron_scheduler(cron_events):
    app = create_app(MagicMock(), MagicMock(), channel_manager=None)
    async with app.router.lifespan_context(app):
        assert cron_events.get("started") is True
        assert getattr(app.state, "cron_scheduler", None) is not None
    assert cron_events.get("stopped") is True
