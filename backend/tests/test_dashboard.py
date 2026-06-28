"""Integration tests for the dashboard FastAPI server.

Uses httpx.TestClient (sync) + psycopg2 for DB setup.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nanoresearch.auth.password import hash_password
from tests.conftest import TEST_DB_URL, truncate_all, make_factory


def run(coro):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.fixture(autouse=True)
def clean_tables():
    truncate_all()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    # Inject test session factory directly so dashboard uses the test DB
    import nanoresearch.storage.database as db_mod
    monkeypatch.setattr(db_mod, "_AsyncSessionLocal", make_factory())
    from nanoresearch.dashboard.server import create_app
    return create_app(workspace=tmp_path)


@pytest.fixture
def seeded_user():
    async def _():
        from nanoresearch.storage.repositories.user_repo import UserRepository
        repo = UserRepository(make_factory())
        return await repo.create("testadmin", hash_password("secret123"))
    return run(_())


# ── Auth endpoint ─────────────────────────────────────────────────────────────

def test_login_success(app, seeded_user):
    with TestClient(app) as client:
        resp = client.post(
            "/api/auth/token",
            data={"username": "testadmin", "password": "secret123"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password(app, seeded_user):
    with TestClient(app) as client:
        resp = client.post(
            "/api/auth/token",
            data={"username": "testadmin", "password": "wrongpass"},
        )
    assert resp.status_code == 401


def test_login_unknown_user(app):
    with TestClient(app) as client:
        resp = client.post(
            "/api/auth/token",
            data={"username": "nobody", "password": "secret123"},
        )
    assert resp.status_code == 401


def test_state_requires_auth(app):
    with TestClient(app) as client:
        resp = client.get("/api/state")
    assert resp.status_code == 401


def test_full_auth_flow(app, seeded_user):
    """Login → get token → use token to access /api/state."""
    with TestClient(app) as client:
        login_resp = client.post(
            "/api/auth/token",
            data={"username": "testadmin", "password": "secret123"},
        )
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]

        state_resp = client.get(
            "/api/state",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert state_resp.status_code == 200
    body = state_resp.json()
    assert "sessions" in body
    assert "timestamp" in body
