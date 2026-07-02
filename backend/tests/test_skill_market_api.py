"""Integration tests for the skill market router (ClawHub mocked)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nanoresearch.server import clawhub
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


def test_search_requires_auth(app):
    with TestClient(app) as client:
        assert client.get("/api/skills/market/search?q=x").status_code == 401


def test_search_happy(app, auth_headers, monkeypatch):
    async def fake_search(q, limit=20):
        assert q == "scrape"
        return [{"slug": "@bob/s", "name": "S", "summary": "", "version": "1", "owner": "bob", "score": 1}]
    monkeypatch.setattr(clawhub, "search", fake_search)
    with TestClient(app) as client:
        resp = client.get("/api/skills/market/search?q=scrape", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()[0]["slug"] == "@bob/s"


def test_search_upstream_error_502(app, auth_headers, monkeypatch):
    async def boom(q, limit=20):
        raise clawhub.ClawHubError("down")
    monkeypatch.setattr(clawhub, "search", boom)
    with TestClient(app) as client:
        resp = client.get("/api/skills/market/search?q=x", headers=auth_headers)
    assert resp.status_code == 502


def test_readme_happy(app, auth_headers, monkeypatch):
    async def fake_readme(slug):
        return "# Hello"
    monkeypatch.setattr(clawhub, "get_readme", fake_readme)
    with TestClient(app) as client:
        resp = client.get("/api/skills/market/@bob/s/readme", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["content"] == "# Hello"


def test_install_uses_per_user_workdir(app, auth_headers, monkeypatch, tmp_path):
    seen = {}
    async def fake_install(slug, workdir):
        seen["slug"] = slug
        seen["workdir"] = Path(workdir)
    monkeypatch.setattr(clawhub, "install", fake_install)
    with TestClient(app) as client:
        resp = client.post("/api/skills/install", json={"slug": "@bob/s"}, headers=auth_headers)
    assert resp.status_code == 200
    assert seen["slug"] == "@bob/s"
    assert seen["workdir"] == tmp_path / "users" / "testadmin"


def test_install_cli_missing_500(app, auth_headers, monkeypatch):
    async def boom(slug, workdir):
        raise clawhub.ClawHubCLINotFound()
    monkeypatch.setattr(clawhub, "install", boom)
    with TestClient(app) as client:
        resp = client.post("/api/skills/install", json={"slug": "@bob/s"}, headers=auth_headers)
    assert resp.status_code == 500


def test_delete_removes_workspace_skill(app, auth_headers, tmp_path):
    d = tmp_path / "users" / "testadmin" / "skills" / "foo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: foo\n---\n", encoding="utf-8")
    with TestClient(app) as client:
        resp = client.delete("/api/skills/foo", headers=auth_headers)
    assert resp.status_code == 204
    assert not d.exists()


def test_delete_missing_skill_404(app, auth_headers):
    with TestClient(app) as client:
        resp = client.delete("/api/skills/does-not-exist", headers=auth_headers)
    assert resp.status_code == 404


def test_delete_rejects_traversal(app, auth_headers):
    with TestClient(app) as client:
        resp = client.delete("/api/skills/..%2f..%2fsecret", headers=auth_headers)
    assert resp.status_code in (403, 404)
