"""Integration tests for GET /api/skills per-user pool scan."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_factory


class FakeAgentLoop:
    model = "fake-model"
    async def process_direct(self, content, *, on_stream, on_progress, **kwargs):
        await on_stream("ok")


@pytest.fixture
def app(monkeypatch, tmp_path):
    # Auth is pure-JWT (no DB lookup) and these endpoints touch neither DB nor
    # Redis, so we skip DB seeding / truncate and the manual run() event loop —
    # that pattern triggers pre-existing "bound to a different event loop" flakes.
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanoresearch.server.main import create_app
    app = create_app(channel_loop=FakeAgentLoop(), session_factory=make_factory())
    app.state.loop_config = {"base_workspace": str(tmp_path)}
    return app


@pytest.fixture
def auth_headers(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanoresearch.auth.jwt import create_token
    return {"Authorization": f"Bearer {create_token('testadmin')}"}


def _write_skill(tmp_path, uid, name, description):
    d = tmp_path / "users" / uid / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f'---\nname: {name}\ndescription: {description}\n---\n# {name}\n', encoding="utf-8"
    )


def test_skills_requires_auth(app):
    with TestClient(app) as client:
        assert client.get("/api/skills").status_code == 401


def test_skills_includes_workspace_skill(app, auth_headers, tmp_path):
    _write_skill(tmp_path, "testadmin", "my-scraper", "scrape things")
    with TestClient(app) as client:
        resp = client.get("/api/skills", headers=auth_headers)
    assert resp.status_code == 200
    names = {s["name"]: s for s in resp.json()}
    assert names["my-scraper"]["source"] == "workspace"
    assert names["my-scraper"]["description"] == "scrape things"
    assert "clawhub" in names  # a built-in skill is still present


def test_skills_isolated_per_user(app, auth_headers, tmp_path):
    _write_skill(tmp_path, "someone-else", "secret-skill", "not yours")
    with TestClient(app) as client:
        resp = client.get("/api/skills", headers=auth_headers)
    assert "secret-skill" not in {s["name"] for s in resp.json()}
