"""Integration tests for Phase 2 chat API — conversations, runs, SSE."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from nanoresearch.auth.password import hash_password
from tests.conftest import truncate_all, make_factory


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


class FakeAgentLoop:
    """Minimal stand-in for AgentLoop — emits two stream chunks then returns."""

    model = "fake-model"
    _last_usage: dict = {}

    async def process_direct(self, content, *, on_stream, on_progress, **kwargs):
        await on_stream("Hello ")
        await on_stream("world")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_tables():
    truncate_all()


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanoresearch.server.main import create_app
    return create_app(channel_loop=FakeAgentLoop(), session_factory=make_factory())


@pytest.fixture
def seeded_user():
    async def _():
        from nanoresearch.storage.repositories.user_repo import UserRepository
        repo = UserRepository(make_factory())
        return await repo.create("testadmin", hash_password("secret123"))
    return run(_())


@pytest.fixture
def other_user():
    async def _():
        from nanoresearch.storage.repositories.user_repo import UserRepository
        repo = UserRepository(make_factory())
        return await repo.create("other", hash_password("other123"))
    return run(_())


@pytest.fixture
def auth_headers(seeded_user, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanoresearch.auth.jwt import create_token
    return {"Authorization": f"Bearer {create_token('testadmin')}"}


@pytest.fixture
def other_auth_headers(other_user, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanoresearch.auth.jwt import create_token
    return {"Authorization": f"Bearer {create_token('other')}"}


# ── Auth guard ────────────────────────────────────────────────────────────────

def test_auth_me(app, auth_headers):
    with TestClient(app) as client:
        resp = client.get("/api/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["uid"] == "testadmin"


def test_conversations_requires_auth(app):
    with TestClient(app) as client:
        assert client.get("/api/conversations").status_code == 401
        assert client.post("/api/conversations", json={}).status_code == 401
        assert client.post("/api/runs", json={"conversation_id": "x", "content": "y"}).status_code == 401


# ── Conversation CRUD ─────────────────────────────────────────────────────────

def test_create_conversation(app, auth_headers):
    with TestClient(app) as client:
        resp = client.post("/api/conversations", json={"title": "My Chat"}, headers=auth_headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "My Chat"
    assert "id" in body
    assert "session_key" in body


def test_list_conversations_empty(app, auth_headers):
    with TestClient(app) as client:
        resp = client.get("/api/conversations", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_conversations(app, auth_headers):
    with TestClient(app) as client:
        client.post("/api/conversations", json={"title": "Conv A"}, headers=auth_headers)
        client.post("/api/conversations", json={"title": "Conv B"}, headers=auth_headers)
        resp = client.get("/api/conversations", headers=auth_headers)
    assert resp.status_code == 200
    titles = {c["title"] for c in resp.json()}
    assert titles == {"Conv A", "Conv B"}


def test_get_conversation(app, auth_headers):
    with TestClient(app) as client:
        create_resp = client.post("/api/conversations", json={"title": "Detail"}, headers=auth_headers)
        conv_id = create_resp.json()["id"]
        resp = client.get(f"/api/conversations/{conv_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == conv_id
    assert resp.json()["title"] == "Detail"


def test_get_conversation_not_found(app, auth_headers):
    with TestClient(app) as client:
        resp = client.get("/api/conversations/00000000-0000-0000-0000-000000000000", headers=auth_headers)
    assert resp.status_code == 404


def test_get_conversation_invalid_id(app, auth_headers):
    with TestClient(app) as client:
        resp = client.get("/api/conversations/not-a-uuid", headers=auth_headers)
    assert resp.status_code == 404


def test_delete_conversation(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        del_resp = client.delete(f"/api/conversations/{conv_id}", headers=auth_headers)
        assert del_resp.status_code == 204
        get_resp = client.get(f"/api/conversations/{conv_id}", headers=auth_headers)
        assert get_resp.status_code == 404


def test_conversation_isolation(app, auth_headers, other_auth_headers):
    """User cannot see or delete another user's conversation."""
    with TestClient(app) as client:
        conv_id = client.post(
            "/api/conversations", json={"title": "Private"}, headers=auth_headers
        ).json()["id"]
        assert client.get(f"/api/conversations/{conv_id}", headers=other_auth_headers).status_code == 404
        assert client.delete(f"/api/conversations/{conv_id}", headers=other_auth_headers).status_code == 404


# ── Messages ──────────────────────────────────────────────────────────────────

def test_get_messages_empty(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        resp = client.get(f"/api/conversations/{conv_id}/messages", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_messages_pagination(app, auth_headers):
    async def seed(conv_id_str):
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        import uuid
        repo = ConversationRepository(make_factory())
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"} for i in range(5)]
        await repo.replace_messages(uuid.UUID(conv_id_str), msgs)

    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        run(seed(conv_id))
        all_resp = client.get(f"/api/conversations/{conv_id}/messages", headers=auth_headers)
        paged_resp = client.get(
            f"/api/conversations/{conv_id}/messages?limit=2&offset=1", headers=auth_headers
        )
    assert len(all_resp.json()) == 5
    paged = paged_resp.json()
    assert len(paged) == 2
    assert paged[0]["seq"] == 1


# ── Run lifecycle ─────────────────────────────────────────────────────────────

def test_create_run(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        resp = client.post(
            "/api/runs",
            json={"conversation_id": conv_id, "content": "hello"},
            headers=auth_headers,
        )
    assert resp.status_code == 201
    body = resp.json()
    assert "run_id" in body
    assert body["conversation_id"] == conv_id
    assert body["status"] == "pending"


def test_get_run(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        run_id = client.post(
            "/api/runs",
            json={"conversation_id": conv_id, "content": "hello"},
            headers=auth_headers,
        ).json()["run_id"]
        resp = client.get(f"/api/runs/{run_id}", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == run_id
    assert "status" in body
    assert "tool_calls" in body
    assert "tokens_used" in body


def test_get_run_not_found(app, auth_headers):
    with TestClient(app) as client:
        resp = client.get("/api/runs/00000000-0000-0000-0000-000000000000", headers=auth_headers)
    assert resp.status_code == 404


def test_conversation_runs_list(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        client.post("/api/runs", json={"conversation_id": conv_id, "content": "first"}, headers=auth_headers)
        client.post("/api/runs", json={"conversation_id": conv_id, "content": "second"}, headers=auth_headers)
        resp = client.get(f"/api/conversations/{conv_id}/runs", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ── SSE events ────────────────────────────────────────────────────────────────

def test_run_events_already_completed(app, auth_headers):
    """Completed run (no active queue) returns a single terminal run_end event."""
    async def seed_completed_run():
        import uuid
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        from nanoresearch.storage.repositories.run_repo import RunRepository
        from datetime import datetime, timezone
        factory = make_factory()
        conv = await ConversationRepository(factory).create(key="web:done-conv", uid="testadmin")
        run_obj = await RunRepository(factory).create(conversation_id=conv.id, uid="testadmin")
        await RunRepository(factory).update(run_obj.id, status="completed")
        return str(run_obj.id)

    run_id = run(seed_completed_run())

    with TestClient(app) as client:
        events = []
        with client.stream("GET", f"/api/runs/{run_id}/events", headers=auth_headers) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

    assert len(events) == 1
    assert events[0]["type"] == "run_end"
    assert events[0]["status"] == "completed"


def test_run_events_live_stream(app, auth_headers):
    """SSE stream terminates with run_end/completed.

    The background task may finish before or after the SSE connection opens —
    both paths are valid.  If we captured live message_delta events, we also
    verify their content.
    """
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        run_id = client.post(
            "/api/runs",
            json={"conversation_id": conv_id, "content": "stream me"},
            headers=auth_headers,
        ).json()["run_id"]

        events = []
        with client.stream("GET", f"/api/runs/{run_id}/events", headers=auth_headers) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        # Stream must end with a terminal run_end event
        assert events, "SSE stream returned no events"
        assert events[-1]["type"] == "run_end"
        assert events[-1]["status"] == "completed"

        # If live message_delta events arrived, verify their content
        deltas = [e for e in events if e["type"] == "message_delta"]
        if deltas:
            assert "".join(e["chunk"] for e in deltas) == "Hello world"

        # DB must reflect completed status
        run_resp = client.get(f"/api/runs/{run_id}", headers=auth_headers)
        assert run_resp.json()["status"] == "completed"
