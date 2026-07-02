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
    async def fake_get_skill(slug):
        return {"moderation": {"state": "clean"}}
    async def fake_install(slug, workdir):
        seen["slug"] = slug
        seen["workdir"] = Path(workdir)
    monkeypatch.setattr(clawhub, "get_skill", fake_get_skill)
    monkeypatch.setattr(clawhub, "install", fake_install)
    with TestClient(app) as client:
        resp = client.post("/api/skills/install", json={"slug": "@bob/s"}, headers=auth_headers)
    assert resp.status_code == 200
    assert seen["slug"] == "@bob/s"
    assert seen["workdir"] == tmp_path / "users" / "testadmin"


def test_install_cli_missing_500(app, auth_headers, monkeypatch):
    async def fake_get_skill(slug):
        return {"moderation": {"state": "clean"}}
    async def boom(slug, workdir):
        raise clawhub.ClawHubCLINotFound()
    monkeypatch.setattr(clawhub, "get_skill", fake_get_skill)
    monkeypatch.setattr(clawhub, "install", boom)
    with TestClient(app) as client:
        resp = client.post("/api/skills/install", json={"slug": "@bob/s"}, headers=auth_headers)
    assert resp.status_code == 500


def test_install_rejects_invalid_slug(app, auth_headers, monkeypatch):
    def fail_if_called(*a, **kw):
        raise AssertionError("clawhub.install should not be called for invalid slug")
    monkeypatch.setattr(clawhub, "install", fail_if_called)
    with TestClient(app) as client:
        for bad in ["--evil", "../x"]:
            resp = client.post("/api/skills/install", json={"slug": bad}, headers=auth_headers)
            assert resp.status_code == 400, f"expected 400 for {bad!r}, got {resp.status_code}"


def test_install_rejects_flagged_moderation(app, auth_headers, monkeypatch):
    async def fake_get_skill(slug):
        return {"moderation": {"state": "flagged"}}
    def fail_if_called(*a, **kw):
        raise AssertionError("clawhub.install should not be called for flagged skill")
    monkeypatch.setattr(clawhub, "get_skill", fake_get_skill)
    monkeypatch.setattr(clawhub, "install", fail_if_called)
    with TestClient(app) as client:
        resp = client.post("/api/skills/install", json={"slug": "@bob/s"}, headers=auth_headers)
    assert resp.status_code == 403


def test_install_allows_clean_moderation(app, auth_headers, monkeypatch, tmp_path):
    install_calls = []
    async def fake_get_skill(slug):
        return {"moderation": {"state": "clean"}}
    async def fake_install(slug, workdir):
        install_calls.append((slug, workdir))
    monkeypatch.setattr(clawhub, "get_skill", fake_get_skill)
    monkeypatch.setattr(clawhub, "install", fake_install)
    with TestClient(app) as client:
        resp = client.post("/api/skills/install", json={"slug": "@bob/s"}, headers=auth_headers)
    assert resp.status_code == 200
    assert install_calls, "install should have been called"
    assert install_calls[0][1] == tmp_path / "users" / "testadmin"


def test_market_skill_rejects_invalid_slug(app, auth_headers, monkeypatch):
    def fail_if_called(*a, **kw):
        raise AssertionError("clawhub.get_skill should not be called for invalid slug")
    monkeypatch.setattr(clawhub, "get_skill", fail_if_called)
    with TestClient(app) as client:
        resp = client.get("/api/skills/market/--evil", headers=auth_headers)
    assert resp.status_code == 400


def test_slug_re_accepts_bare_and_scoped():
    """Real clawhub slugs are single-segment ('toby-pptx'); scoped '@owner/name' also valid.

    Regression: the original regex required a '/', so every bare slug 400'd.
    """
    from nanoresearch.server.routers.skill_market_router import SLUG_RE
    for good in ["pdf", "toby-pptx", "slidepro", "paper-anonymizer-pdf", "@bob/s"]:
        assert SLUG_RE.match(good), f"should accept {good!r}"
    for bad in ["--evil", "../x", "a/../b", "", "a/b/c"]:
        assert not SLUG_RE.match(bad), f"should reject {bad!r}"


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
    # httpx normalizes %2F → / so the path may not reach the DELETE handler (405),
    # or safe_resolve raises 403, or skill not found (404) — all are rejected.
    assert resp.status_code in (403, 404, 405)
