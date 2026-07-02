"""Unit tests for the ClawHub integration service (no network / no real npx)."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from nanoresearch.server import clawhub


def _mock_client(handler):
    return httpx.AsyncClient(base_url=clawhub.REGISTRY, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_search_maps_fields(monkeypatch):
    def handler(request):
        assert request.url.path == "/api/v1/search"
        assert request.url.params["q"] == "scrape"
        return httpx.Response(200, json={"results": [
            {"slug": "@bob/web-scraper", "displayName": "Web Scraper",
             "summary": "scrape sites", "version": "1.2.0",
             "ownerHandle": "bob", "score": 0.9},
        ]})
    monkeypatch.setattr(clawhub, "_client", lambda: _mock_client(handler))
    items = await clawhub.search("scrape", limit=5)
    assert items == [{
        "slug": "@bob/web-scraper", "name": "Web Scraper", "summary": "scrape sites",
        "version": "1.2.0", "owner": "bob", "score": 0.9,
    }]


@pytest.mark.asyncio
async def test_search_raises_on_upstream_error(monkeypatch):
    monkeypatch.setattr(clawhub, "_client",
                        lambda: _mock_client(lambda req: httpx.Response(500)))
    with pytest.raises(clawhub.ClawHubError):
        await clawhub.search("x")


@pytest.mark.asyncio
async def test_get_skill_flags_scripts(monkeypatch):
    def handler(request):
        if request.url.path == "/api/v1/skills/@bob/tool":
            return httpx.Response(200, json={
                "slug": "@bob/tool", "displayName": "Tool", "summary": "does things",
                "topics": [], "tags": [], "stats": {"stars": 3},
                "latestVersion": {"version": "0.1.0", "changelog": "init"},
                "metadata": {}, "moderation": {"state": "clean"},
            })
        if request.url.path == "/api/v1/skills/@bob/tool/versions/0.1.0":
            return httpx.Response(200, json={"files": [
                {"path": "SKILL.md"}, {"path": "scripts/run.py"},
            ]})
        return httpx.Response(404)
    monkeypatch.setattr(clawhub, "_client", lambda: _mock_client(handler))
    skill = await clawhub.get_skill("@bob/tool")
    assert skill["version"] == "0.1.0"
    assert skill["moderation"] == {"state": "clean"}
    assert "scripts/run.py" in skill["files"]
    assert skill["has_scripts"] is True


@pytest.mark.asyncio
async def test_install_builds_argv(monkeypatch):
    calls = {}
    class FakeProc:
        returncode = 0
        async def communicate(self):
            return (b"ok", b"")
    async def fake_exec(*argv, **kw):
        calls["argv"] = argv
        calls["cwd"] = kw.get("cwd")
        return FakeProc()
    monkeypatch.setattr(clawhub.asyncio, "create_subprocess_exec", fake_exec)
    await clawhub.install("@bob/tool", Path("/ws/users/alice"))
    assert "install" in calls["argv"]
    assert "@bob/tool" in calls["argv"]
    assert "--workdir" in calls["argv"]
    assert str(Path("/ws/users/alice")) in calls["argv"]


@pytest.mark.asyncio
async def test_install_nonzero_raises_cli_error(monkeypatch):
    class FakeProc:
        returncode = 1
        async def communicate(self):
            return (b"", b"boom")
    monkeypatch.setattr(clawhub.asyncio, "create_subprocess_exec",
                        lambda *a, **k: _await(FakeProc()))
    with pytest.raises(clawhub.ClawHubCLIError):
        await clawhub.install("@bob/tool", Path("/ws"))


@pytest.mark.asyncio
async def test_install_missing_npx_raises_not_found(monkeypatch):
    async def fake_exec(*a, **k):
        raise FileNotFoundError("npx")
    monkeypatch.setattr(clawhub.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(clawhub.ClawHubCLINotFound):
        await clawhub.install("@bob/tool", Path("/ws"))


async def _await(v):  # helper: make a plain value awaitable for lambda use
    return v
