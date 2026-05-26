"""Integration tests for agent_router — GET/PUT /api/agents."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from nanobot.auth.password import hash_password
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
    model = "fake-model"
    _last_usage: dict = {}

    async def process_direct(self, content, *, on_stream, on_progress, **kwargs):
        await on_stream("ok")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_tables():
    truncate_all()


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanobot.server.main import create_app
    return create_app(channel_loop=FakeAgentLoop(), session_factory=make_factory())


@pytest.fixture
def seeded_user():
    async def _():
        from nanobot.storage.repositories.user_repo import UserRepository
        repo = UserRepository(make_factory())
        return await repo.create("testadmin", hash_password("secret123"))
    return run(_())


@pytest.fixture
def auth_headers(seeded_user, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanobot.auth.jwt import create_token
    return {"Authorization": f"Bearer {create_token('testadmin')}"}


@pytest.fixture
def seeded_agent():
    async def _():
        from nanobot.storage.repositories.agent_repo import AgentRepository
        repo = AgentRepository(make_factory())
        return await repo.create({
            "name": "Nano Research",
            "description": "深度研究智能体",
            "version": "1.0.0",
            "is_default": True,
            "default_model": "anthropic/claude-opus-4-5",
            "provider": "anthropic",
            "capabilities": {"streaming": True, "web_search": True, "deep_research": True},
            "skills_config": [{"id": "deep-research", "name": "Deep Research", "enabled": True}],
            "tools_config": [{"name": "web_search", "enabled": True}],
        })
    return run(_())


# ── GET /api/agents ────────────────────────────────────────────────────────────

def test_list_agents_empty(app, auth_headers):
    with TestClient(app) as client:
        resp = client.get("/api/agents", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_agents_requires_auth(app):
    with TestClient(app) as client:
        assert client.get("/api/agents").status_code == 401


def test_list_agents_returns_card(app, auth_headers, seeded_agent):
    with TestClient(app) as client:
        resp = client.get("/api/agents", headers=auth_headers)
    assert resp.status_code == 200
    agents = resp.json()
    assert len(agents) == 1
    card = agents[0]
    assert card["name"] == "Nano Research"
    assert card["is_default"] is True
    assert "capabilities" in card
    assert "skills" in card
    assert "tools" in card
    assert "stats" in card
    stats = card["stats"]
    assert stats["total_runs"] == 0
    assert stats["total_conversations"] == 0
    assert stats["avg_run_duration_ms"] is None


# ── GET /api/agents/{id} ───────────────────────────────────────────────────────

def test_get_agent_detail(app, auth_headers, seeded_agent):
    with TestClient(app) as client:
        resp = client.get(f"/api/agents/{seeded_agent.id}", headers=auth_headers)
    assert resp.status_code == 200
    card = resp.json()
    assert card["id"] == str(seeded_agent.id)
    assert card["description"] == "深度研究智能体"
    assert card["skills"] == [{"id": "deep-research", "name": "Deep Research", "enabled": True}]
    assert card["tools"] == [{"name": "web_search", "enabled": True}]


def test_get_agent_not_found(app, auth_headers):
    with TestClient(app) as client:
        resp = client.get("/api/agents/00000000-0000-0000-0000-000000000000", headers=auth_headers)
    assert resp.status_code == 404


def test_get_agent_invalid_id(app, auth_headers):
    with TestClient(app) as client:
        resp = client.get("/api/agents/not-a-uuid", headers=auth_headers)
    assert resp.status_code == 404


# ── PUT /api/agents/{id} ───────────────────────────────────────────────────────

def test_update_agent(app, auth_headers, seeded_agent):
    with TestClient(app) as client:
        resp = client.put(
            f"/api/agents/{seeded_agent.id}",
            json={"name": "Updated Name", "max_iterations": 20},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "Updated Name"


def test_update_agent_not_found(app, auth_headers):
    with TestClient(app) as client:
        resp = client.put(
            "/api/agents/00000000-0000-0000-0000-000000000000",
            json={"name": "X"},
            headers=auth_headers,
        )
    assert resp.status_code == 404


# ── stats with actual runs ─────────────────────────────────────────────────────

def test_stats_with_runs(app, auth_headers, seeded_agent, seeded_user):
    async def _seed_runs():
        from nanobot.storage.repositories.conversation_repo import ConversationRepository
        from nanobot.storage.repositories.run_repo import RunRepository
        factory = make_factory()
        conv = await ConversationRepository(factory).create(
            key="web:stats-test", uid="testadmin", agent_id=seeded_agent.id
        )
        run_repo = RunRepository(factory)
        await run_repo.create(conversation_id=conv.id, uid="testadmin", agent_id=seeded_agent.id)
        r2 = await run_repo.create(conversation_id=conv.id, uid="testadmin", agent_id=seeded_agent.id)
        await run_repo.update(r2.id, duration_ms=1000)

    run(_seed_runs())

    with TestClient(app) as client:
        resp = client.get(f"/api/agents/{seeded_agent.id}", headers=auth_headers)
    stats = resp.json()["stats"]
    assert stats["total_runs"] == 2
    assert stats["total_conversations"] == 1
