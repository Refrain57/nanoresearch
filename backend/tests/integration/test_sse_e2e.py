"""SSE end-to-end sensors for the run_events → /api/runs/{id}/events chain (Phase 0/1).

Controllable: we seed events into `run_events:{run_id}` with a SYNC redis client (DB 0, the app's
db) so the SSE chain can be exercised without a worker/LLM. Covers normal streaming, the Phase 1
reuse-run_id continuity, and probes the suspected "terminal run + empty stream → hang" gap.
"""
from __future__ import annotations

import asyncio
import json
import threading

import pytest
import redis as _sync_redis
from fastapi.testclient import TestClient

from nanoresearch.auth.password import hash_password
from nanoresearch.bus.redis_client import REDIS_URL
from nanoresearch.bus.redis_keys import RedisKeys
from tests.conftest import make_factory, truncate_all


def run(coro):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


class FakeAgentLoop:
    model = "fake-model"
    _last_usage: dict = {}

    async def process_direct(self, content, *, on_stream, on_progress, **kwargs):
        return None


@pytest.fixture(autouse=True)
def clean_tables():
    truncate_all()


@pytest.fixture(autouse=True)
def _reset_redis_singleton():
    # The async redis client is a module singleton bound to the loop where it was created. Each
    # TestClient runs the app lifespan in its own loop, so reset the singleton around every test.
    import nanoresearch.bus.redis_client as rc
    rc._client = None
    yield
    rc._client = None


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanoresearch.server.main import create_app
    return create_app(channel_loop=FakeAgentLoop(), session_factory=make_factory())


@pytest.fixture
def auth_headers(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanoresearch.auth.jwt import create_token

    async def _seed():
        from nanoresearch.storage.repositories.user_repo import UserRepository
        await UserRepository(make_factory()).create("testadmin", hash_password("secret123"))
    run(_seed())
    return {"Authorization": f"Bearer {create_token('testadmin')}"}


async def _seed_run(status="running"):
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.run_repo import RunRepository
    factory = make_factory()
    conv = await ConversationRepository(factory).create(key="web:sse", uid="testadmin")
    run_obj = await RunRepository(factory).create(conversation_id=conv.id, uid="testadmin")
    if status != "pending":
        await RunRepository(factory).update(run_obj.id, status=status)
    return str(run_obj.id)


def _seed_events_sync(run_id: str, events: list[dict]) -> None:
    """Append events to run_events:{run_id} matching xadd_event's small-payload format."""
    r = _sync_redis.from_url(REDIS_URL)
    key = RedisKeys.run_events(run_id)
    for ev in events:
        r.xadd(key, {"data": json.dumps(ev, ensure_ascii=False)})
    r.expire(key, 86400)
    r.close()


def _collect_sse(app, headers, run_id, timeout=8):
    """Read the SSE stream in a worker thread; returns (events, timed_out)."""
    out: dict = {"events": [], "done": False}

    def _read():
        try:
            with TestClient(app) as client:
                with client.stream("GET", f"/api/runs/{run_id}/events",
                                   headers=headers, timeout=timeout) as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            out["events"].append(json.loads(line[6:]))
        except Exception:
            pass
        finally:
            out["done"] = True

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=timeout + 4)
    return out["events"], t.is_alive() or not out["done"]


def test_sse_streams_events_and_terminates_on_run_end(app, auth_headers):
    """Sensor: deltas then run_end on the stream → SSE delivers them and terminates."""
    run_id = run(_seed_run("running"))
    _seed_events_sync(run_id, [
        {"type": "message_delta", "chunk": "Hello "},
        {"type": "message_delta", "chunk": "world"},
        {"type": "run_end", "status": "completed", "duration_ms": 5},
    ])
    events, timed_out = _collect_sse(app, auth_headers, run_id)
    assert not timed_out, "SSE did not terminate on run_end"
    deltas = [e for e in events if e["type"] == "message_delta"]
    assert "".join(e["chunk"] for e in deltas) == "Hello world"
    assert events[-1]["type"] == "run_end" and events[-1]["status"] == "completed"


def test_sse_phase1_reuse_run_id_continuity(app, auth_headers):
    """Phase 1: R1 phase deltas + subagent_result + continuation deltas + run_end all arrive on
    ONE SSE connection because the continuation REUSES the original run_id."""
    run_id = run(_seed_run("running"))
    _seed_events_sync(run_id, [
        {"type": "message_delta", "chunk": "spawning..."},            # R1 phase
        {"type": "subagent_result", "task_id": "t1", "label": "L", "status": "ok",
         "content": "sub done"},                                       # subagent (same run_id)
        {"type": "message_delta", "chunk": "summary"},                 # continuation phase (reused run_id)
        {"type": "run_end", "status": "completed", "duration_ms": 9},  # continuation emits run_end
    ])
    events, timed_out = _collect_sse(app, auth_headers, run_id)
    assert not timed_out
    types = [e["type"] for e in events]
    assert "subagent_result" in types
    assert types[-1] == "run_end"
    assert any(e.get("chunk") == "summary" for e in events)  # continuation output reached the client


def test_sse_terminal_completed_empty_stream_synthesizes_run_end(app, auth_headers):
    """Fix: a terminal (completed) run with an empty/expired stream → SSE detects the DB status
    and synthesizes a run_end (no hang). Was the root cause of the long-hanging completed-run test."""
    run_id = run(_seed_run("completed"))   # DB completed, NO events in the stream
    events, timed_out = _collect_sse(app, auth_headers, run_id, timeout=12)
    assert not timed_out, "SSE hung on a terminal run with an empty stream"
    assert events and events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "completed"   # concern 2: real terminal status


def test_sse_terminal_failed_run_synthesizes_failed_run_end(app, auth_headers):
    """Concern 2: a failed run must synthesize status=failed (not completed), else the frontend
    would show a failed run as success."""
    run_id = run(_seed_run("failed"))
    events, timed_out = _collect_sse(app, auth_headers, run_id, timeout=12)
    assert not timed_out
    assert events and events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "failed"


def test_sse_prefers_real_run_end_over_synthetic(app, auth_headers):
    """Concern 3: when the stream already carries a real run_end (e.g. written by the watchdog),
    the SSE delivers THAT one and does not also synthesize → frontend gets exactly one run_end."""
    run_id = run(_seed_run("failed"))
    _seed_events_sync(run_id, [
        {"type": "message_delta", "chunk": "x"},
        {"type": "run_end", "status": "failed", "duration_ms": 7, "error": "real"},
    ])
    events, timed_out = _collect_sse(app, auth_headers, run_id, timeout=12)
    assert not timed_out
    run_ends = [e for e in events if e["type"] == "run_end"]
    assert len(run_ends) == 1                       # exactly one (no duplicate synthetic)
    assert run_ends[0].get("duration_ms") == 7      # the REAL one (synthetic has no duration_ms)
    assert run_ends[0].get("error") == "real"
