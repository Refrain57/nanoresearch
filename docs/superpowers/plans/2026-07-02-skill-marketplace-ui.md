# F1 — Skill 市场 / ClawHub 安装 UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a logged-in user search ClawHub, preview a skill, install it into their own workspace (where it enters the available pool), and remove workspace-installed skills — all inside the agent「添加 Skill」modal.

**Architecture:** Backend proxies ClawHub's public HTTP API for read-only ops (search / detail / SKILL.md), shells out to the `clawhub` CLI for install, and does a guarded filesystem delete for remove. The available pool is a per-user filesystem scan of `users/{uid}/skills`. Frontend adds a two-tab modal (可用池 / 从市场安装) with a preview drawer.

**Tech Stack:** Python 3 / FastAPI / httpx / pytest (backend); Vue 3 / Ant Design Vue 4 / pinia / marked, built with vite via pnpm (frontend).

## Global Constraints

- Backend HTTP client: `httpx` (already a dep, `>=0.28.0,<1.0.0`). Do not add new deps.
- ClawHub registry base URL: `os.environ.get("CLAWHUB_REGISTRY", "https://clawhub.ai")`. All API paths under `/api/v1/...`.
- Install/uninstall CLI: `npx --yes clawhub@latest ...` (matches `backend/nanoresearch/skills/clawhub/SKILL.md`).
- Auth: every new endpoint depends on `nanoresearch.server.middleware.auth.get_current_user`. `uid` comes from the JWT only — never a request parameter. Install/remove target `users/{uid}/skills/` exclusively.
- Per-user workspace root: `{base_workspace}/users/{uid}` where `base_workspace = app.state.loop_config["base_workspace"]` or `get_workspace_path()`.
- Frontend: Ant Design Vue 4 components + the existing Anthropic warm-clay theme (do not hardcode colors; use theme tokens / existing classes). Package manager is **pnpm**. There is **no** JS test runner — verify frontend tasks with `pnpm --dir web build` and the manual E2E checklist. Do not add a test framework.
- SKILL.md preview rendering uses `marked` (already a dep).

---

### Task 1: Shared workspace path helpers

Extract the per-user workspace + path-traversal guard out of `workspace_router.py` into a small module so the skills endpoints can reuse them without a cross-router private import.

**Files:**
- Create: `backend/nanoresearch/server/routers/workspace_paths.py`
- Modify: `backend/nanoresearch/server/routers/workspace_router.py:23-38` (replace the two local helper defs with imports)
- Test: `backend/tests/unit/server/test_workspace_paths.py`

**Interfaces:**
- Produces:
  - `user_workspace(request: Request, uid: str) -> Path` — returns `{base}/users/{uid}`.
  - `safe_resolve(workspace: Path, rel_path: str) -> Path` — resolves `rel_path` under `workspace`, raising `HTTPException(403, "非法路径")` on escape.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/server/test_workspace_paths.py`:

```python
"""Unit tests for shared workspace path helpers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from nanoresearch.server.routers.workspace_paths import safe_resolve, user_workspace


def _fake_request(base: str | None):
    state = SimpleNamespace(loop_config={"base_workspace": base} if base else {})
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_user_workspace_builds_per_user_path(tmp_path):
    req = _fake_request(str(tmp_path))
    assert user_workspace(req, "alice") == tmp_path / "users" / "alice"


def test_safe_resolve_allows_inside(tmp_path):
    assert safe_resolve(tmp_path, "skills/foo") == (tmp_path / "skills" / "foo").resolve()


def test_safe_resolve_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException) as exc:
        safe_resolve(tmp_path, "../../etc/passwd")
    assert exc.value.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/unit/server/test_workspace_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: nanoresearch.server.routers.workspace_paths`

- [ ] **Step 3: Create the helpers module**

Create `backend/nanoresearch/server/routers/workspace_paths.py`:

```python
"""Shared per-user workspace path helpers used by workspace + skills routers."""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request


def user_workspace(request: Request, uid: str) -> Path:
    cfg = getattr(request.app.state, "loop_config", None) or {}
    base = cfg.get("base_workspace")
    if base is None:
        from nanoresearch.config.paths import get_workspace_path
        base = get_workspace_path()
    return Path(base) / "users" / uid


def safe_resolve(workspace: Path, rel_path: str) -> Path:
    resolved = (workspace / rel_path).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="非法路径")
    return resolved
```

- [ ] **Step 4: Rewire `workspace_router.py` to use them**

In `backend/nanoresearch/server/routers/workspace_router.py`, delete the local `_user_workspace` (lines ~23-29) and `_safe_resolve` (lines ~32-38) function definitions and replace them with an import near the other imports at the top:

```python
from nanoresearch.server.routers.workspace_paths import (
    safe_resolve as _safe_resolve,
    user_workspace as _user_workspace,
)
```

Leave all existing call sites (`_user_workspace(...)`, `_safe_resolve(...)`) unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/unit/server/test_workspace_paths.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/server/routers/workspace_paths.py backend/nanoresearch/server/routers/workspace_router.py backend/tests/unit/server/test_workspace_paths.py
git commit -m "refactor(server): extract shared per-user workspace path helpers"
```

---

### Task 2: Per-user `/api/skills` pool scan

`GET /api/skills` currently scans the built-in skills dir, so workspace-installed skills never appear in the pool. Fix it to scan the authenticated user's workspace + built-in.

**Files:**
- Modify: `backend/nanoresearch/server/routers/agent_router.py:61-76` (`list_skills`)
- Test: `backend/tests/test_skills_api.py`

**Interfaces:**
- Consumes: `user_workspace` (Task 1), `SkillsLoader`, `get_current_user`.
- Produces: `GET /api/skills` → `list[{name, description, source}]` where `source` ∈ {`workspace`, `builtin`}, scoped to the caller.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_skills_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_skills_api.py -v`
Expected: FAIL — `test_skills_includes_workspace_skill` fails because `my-scraper` is absent (current endpoint scans the built-in dir).

- [ ] **Step 3: Fix `list_skills`**

In `backend/nanoresearch/server/routers/agent_router.py`, replace the existing `list_skills` (lines ~61-76). Ensure `Request` is imported (it is used elsewhere in this file) and add the `workspace_paths` import at the top:

```python
from nanoresearch.server.routers.workspace_paths import user_workspace
```

Replace the endpoint with:

```python
@router.get("/api/skills")
async def list_skills(request: Request, uid: str = Depends(get_current_user)):
    from nanoresearch.agent.skills import BUILTIN_SKILLS_DIR, SkillsLoader
    loader = SkillsLoader(
        workspace=user_workspace(request, uid),
        builtin_skills_dir=BUILTIN_SKILLS_DIR,
    )
    result = []
    for s in loader.list_skills(filter_unavailable=False):
        meta = loader.get_skill_metadata(s["name"]) or {}
        result.append({
            "name": s["name"],
            "description": meta.get("description", s["name"]),
            "source": s["source"],
        })
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_skills_api.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/server/routers/agent_router.py backend/tests/test_skills_api.py
git commit -m "fix(skills): GET /api/skills scans the per-user workspace so installed skills appear in the pool"
```

---

### Task 3: ClawHub integration service

A single module wrapping ClawHub: HTTP proxy for read ops, subprocess for install, guarded delete for remove.

**Files:**
- Create: `backend/nanoresearch/server/clawhub.py`
- Test: `backend/tests/unit/server/test_clawhub.py`

**Interfaces:**
- Produces (all `async` unless noted):
  - `search(q: str, limit: int = 20) -> list[dict]` → items `{slug, name, summary, version, owner, score}`.
  - `get_skill(slug: str) -> dict` → `{slug, name, summary, topics, tags, version, changelog, moderation, stats, os_restrictions, files: list[str], has_scripts: bool}`.
  - `get_readme(slug: str) -> str` (raw SKILL.md text).
  - `install(slug: str, workdir: Path) -> None`.
  - `uninstall(name: str, workdir: Path) -> None` (best-effort CLI lockfile tidy; the router owns the authoritative filesystem delete).
  - Exceptions: `ClawHubError` (registry/network/non-2xx), `ClawHubCLIError(message, stderr)`, `ClawHubCLINotFound`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/server/test_clawhub.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/unit/server/test_clawhub.py -v`
Expected: FAIL — `ModuleNotFoundError: nanoresearch.server.clawhub`

- [ ] **Step 3: Implement the service**

Create `backend/nanoresearch/server/clawhub.py`:

```python
"""ClawHub integration: HTTP proxy for read ops + CLI for install."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx

REGISTRY = os.environ.get("CLAWHUB_REGISTRY", "https://clawhub.ai")
_TIMEOUT = 15.0
_CLI_TIMEOUT = 180.0


class ClawHubError(Exception):
    """Registry unreachable or returned a non-2xx response."""


class ClawHubCLIError(Exception):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


class ClawHubCLINotFound(Exception):
    """npx / node not available on the server."""


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=REGISTRY, timeout=_TIMEOUT)


async def _get_json(path: str, params: dict | None = None):
    try:
        async with _client() as c:
            resp = await c.get(path, params=params)
    except httpx.HTTPError as e:  # network/DNS/timeout
        raise ClawHubError(f"ClawHub unreachable: {e}") from e
    if resp.status_code >= 400:
        raise ClawHubError(f"ClawHub returned {resp.status_code} for {path}")
    return resp.json()


async def search(q: str, limit: int = 20) -> list[dict]:
    data = await _get_json("/api/v1/search", {"q": q, "limit": limit})
    items = data.get("results", data) if isinstance(data, dict) else data
    out = []
    for it in items or []:
        out.append({
            "slug": it.get("slug"),
            "name": it.get("displayName") or it.get("slug"),
            "summary": it.get("summary", ""),
            "version": it.get("version"),
            "owner": it.get("ownerHandle") or (it.get("owner") or {}).get("handle"),
            "score": it.get("score"),
        })
    return out


async def get_skill(slug: str) -> dict:
    meta = await _get_json(f"/api/v1/skills/{slug}")
    latest = meta.get("latestVersion") or {}
    version = latest.get("version")
    files: list[str] = []
    if version:
        try:
            v = await _get_json(f"/api/v1/skills/{slug}/versions/{version}")
            files = [f.get("path") for f in (v.get("files") or []) if f.get("path")]
        except ClawHubError:
            files = []
    return {
        "slug": meta.get("slug", slug),
        "name": meta.get("displayName") or slug,
        "summary": meta.get("summary", ""),
        "topics": meta.get("topics") or [],
        "tags": meta.get("tags") or [],
        "version": version,
        "changelog": latest.get("changelog", ""),
        "moderation": meta.get("moderation") or {},
        "stats": meta.get("stats") or {},
        "os_restrictions": meta.get("metadata") or {},
        "files": files,
        "has_scripts": any(p != "SKILL.md" for p in files),
    }


async def get_readme(slug: str) -> str:
    try:
        async with _client() as c:
            resp = await c.get(f"/api/v1/skills/{slug}/file", params={"path": "SKILL.md"})
    except httpx.HTTPError as e:
        raise ClawHubError(f"ClawHub unreachable: {e}") from e
    if resp.status_code >= 400:
        raise ClawHubError(f"ClawHub returned {resp.status_code} for SKILL.md")
    return resp.text


async def _run_cli(*args: str) -> None:
    argv = ["npx", "--yes", "clawhub@latest", *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise ClawHubCLINotFound("npx/node not found on server") from e
    try:
        _out, err = await asyncio.wait_for(proc.communicate(), timeout=_CLI_TIMEOUT)
    except asyncio.TimeoutError as e:
        proc.kill()
        raise ClawHubCLIError("clawhub CLI timed out") from e
    if proc.returncode != 0:
        raise ClawHubCLIError(
            f"clawhub {args[0]} failed (rc={proc.returncode})",
            stderr=(err or b"").decode("utf-8", "replace")[:2000],
        )


async def install(slug: str, workdir: Path) -> None:
    await _run_cli("install", slug, "--workdir", str(workdir))


async def uninstall(name: str, workdir: Path) -> None:
    await _run_cli("uninstall", name, "--yes", "--workdir", str(workdir))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/unit/server/test_clawhub.py -v`
Expected: PASS (6 passed). If `pytest-asyncio` auto-mode is not enabled, add `@pytest.mark.asyncio` handling per repo convention (check `backend/pyproject.toml` `[tool.pytest.ini_options] asyncio_mode`); the repo already runs async tests, so mirror its setting.

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/server/clawhub.py backend/tests/unit/server/test_clawhub.py
git commit -m "feat(skills): add ClawHub integration service (HTTP proxy + CLI install)"
```

---

### Task 4: Skill market router (endpoints + registration)

**Files:**
- Create: `backend/nanoresearch/server/routers/skill_market_router.py`
- Modify: `backend/nanoresearch/server/main.py:146,154` (import + `include_router`)
- Test: `backend/tests/test_skill_market_api.py`

**Interfaces:**
- Consumes: `clawhub` service (Task 3), `user_workspace` + `safe_resolve` (Task 1), `get_current_user`.
- Produces endpoints: `GET /api/skills/market/search`, `GET /api/skills/market/{slug:path}`, `GET /api/skills/market/{slug:path}/readme`, `POST /api/skills/install`, `DELETE /api/skills/{name}`.

> Note: ClawHub slugs contain `@` and `/` (e.g. `@bob/web-scraper`), so the `{slug:path}` converter is required. `readme` is a distinct sibling route registered *before* the bare `{slug:path}` route to avoid greedy capture — or expressed as `/market/{slug:path}` with a trailing `/readme` handled by ordering; register `readme` first.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_skill_market_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_skill_market_api.py -v`
Expected: FAIL — router not registered (404s / import error).

- [ ] **Step 3: Implement the router**

Create `backend/nanoresearch/server/routers/skill_market_router.py`:

```python
"""ClawHub skill marketplace + workspace install/remove endpoints."""
from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from nanoresearch.server import clawhub
from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.server.routers.workspace_paths import safe_resolve, user_workspace

router = APIRouter()


class InstallBody(BaseModel):
    slug: str


@router.get("/api/skills/market/search")
async def market_search(q: str, limit: int = 20, _uid: str = Depends(get_current_user)):
    try:
        return await clawhub.search(q, limit=limit)
    except clawhub.ClawHubError:
        raise HTTPException(status_code=502, detail="技能市场暂时不可用")


# NOTE: register the readme route before the bare {slug} route so it is not swallowed.
@router.get("/api/skills/market/{slug:path}/readme")
async def market_readme(slug: str, _uid: str = Depends(get_current_user)):
    try:
        return {"content": await clawhub.get_readme(slug)}
    except clawhub.ClawHubError:
        raise HTTPException(status_code=502, detail="技能市场暂时不可用")


@router.get("/api/skills/market/{slug:path}")
async def market_skill(slug: str, _uid: str = Depends(get_current_user)):
    try:
        return await clawhub.get_skill(slug)
    except clawhub.ClawHubError:
        raise HTTPException(status_code=502, detail="技能市场暂时不可用")


@router.post("/api/skills/install")
async def install_skill(body: InstallBody, request: Request, uid: str = Depends(get_current_user)):
    workdir = user_workspace(request, uid)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        await clawhub.install(body.slug, workdir)
    except clawhub.ClawHubCLINotFound:
        raise HTTPException(status_code=500, detail="服务器未安装 Node/clawhub CLI")
    except clawhub.ClawHubCLIError as e:
        raise HTTPException(status_code=502, detail=f"安装失败: {e}")
    return {"installed": body.slug}


@router.delete("/api/skills/{name}", status_code=204)
async def remove_skill(name: str, request: Request, uid: str = Depends(get_current_user)):
    skills_dir = user_workspace(request, uid) / "skills"
    target = safe_resolve(skills_dir, name)  # raises 403 on traversal
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="工作区中未找到该 skill")
    shutil.rmtree(target)
    return None
```

> Remove is a guarded filesystem delete rather than `clawhub uninstall`, because the pool is filesystem-based and must handle skills not tracked by ClawHub's lockfile; the lockfile only matters for `update`, which is out of scope. This is a deliberate refinement over the spec's "CLI uninstall".

- [ ] **Step 4: Register the router in `main.py`**

In `backend/nanoresearch/server/main.py`, add next to the other router imports (~line 146):

```python
from nanoresearch.server.routers.skill_market_router import router as skill_market_router
```

and next to the other `include_router` calls (~line 154):

```python
app.include_router(skill_market_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_skill_market_api.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/server/routers/skill_market_router.py backend/nanoresearch/server/main.py backend/tests/test_skill_market_api.py
git commit -m "feat(skills): skill market router (search/preview/install/remove)"
```

---

### Task 5: Frontend API client

**Files:**
- Create: `web/src/apis/skills.js`

**Interfaces:**
- Produces: `searchMarket(q, limit)`, `getMarketSkill(slug)`, `getMarketReadme(slug)`, `installSkill(slug)`, `uninstallSkill(name)`, `listSkills()`.

- [ ] **Step 1: Create the API client**

Create `web/src/apis/skills.js`:

```javascript
import { apiGet, apiPost, apiDelete } from './base'

// ClawHub marketplace (backend proxies clawhub.ai)
export const searchMarket   = (q, limit = 20) =>
  apiGet(`/api/skills/market/search?q=${encodeURIComponent(q)}&limit=${limit}`)
export const getMarketSkill  = (slug) =>
  apiGet(`/api/skills/market/${encodeURIComponent(slug)}`)
export const getMarketReadme = (slug) =>
  apiGet(`/api/skills/market/${encodeURIComponent(slug)}/readme`)

// Workspace skill pool
export const installSkill   = (slug) => apiPost('/api/skills/install', { slug })
export const uninstallSkill = (name) => apiDelete(`/api/skills/${encodeURIComponent(name)}`)
export const listSkills     = ()     => apiGet('/api/skills')
```

> `encodeURIComponent` turns `@bob/web-scraper` into `%40bob%2Fweb-scraper`; the backend `{slug:path}` route + FastAPI unquoting handles it. Verify during E2E that a scoped slug previews correctly; if the `/` must stay literal, drop `encodeURIComponent` for the slug segment only.

- [ ] **Step 2: Verify the build**

Run: `pnpm --dir web build`
Expected: build succeeds (no import errors).

- [ ] **Step 3: Commit**

```bash
git add web/src/apis/skills.js
git commit -m "feat(web): skills marketplace API client"
```

---

### Task 6: Skill preview drawer component

**Files:**
- Create: `web/src/components/SkillPreviewDrawer.vue`

**Interfaces:**
- Consumes: `getMarketSkill`, `getMarketReadme`, `installSkill` (Task 5).
- Props: `open: Boolean`, `slug: String`. Emits: `update:open`, `installed` (payload: slug).
- Behavior: on open, fetch skill metadata + readme; render trust signals + `marked(readme)`; install button disabled when `moderation.state` ∈ {`flagged`,`removed`}; on install success emit `installed` and close.

- [ ] **Step 1: Create the component**

Create `web/src/components/SkillPreviewDrawer.vue`:

```vue
<template>
  <a-drawer
    :open="open"
    :title="skill?.name || slug"
    width="600"
    @close="$emit('update:open', false)"
  >
    <a-spin :spinning="loading">
      <template v-if="skill">
        <div class="trust-row">
          <a-tag>作者 @{{ skill.owner || skill.slug }}</a-tag>
          <a-tag v-if="skill.version">v{{ skill.version }}</a-tag>
          <a-tag v-if="skill.stats?.stars != null">★ {{ skill.stats.stars }}</a-tag>
          <a-tag :color="blocked ? 'red' : 'green'">
            审核: {{ skill.moderation?.state || 'unknown' }}
          </a-tag>
          <a-tag v-if="skill.has_scripts" color="orange">包含可执行脚本</a-tag>
        </div>

        <a-alert
          v-if="blocked"
          type="error"
          show-icon
          message="该 skill 已被市场标记，禁止安装"
          style="margin: 12px 0"
        />
        <a-alert
          v-else-if="skill.has_scripts"
          type="warning"
          show-icon
          message="此 skill 附带脚本文件，安装后 Agent 可能执行它们。请先阅读下方内容。"
          style="margin: 12px 0"
        />

        <div v-if="skill.files?.length" class="files">
          <div class="section-label">文件</div>
          <a-tag v-for="f in skill.files" :key="f" size="small">{{ f }}</a-tag>
        </div>

        <div class="section-label">SKILL.md</div>
        <div class="readme" v-html="readmeHtml"></div>
      </template>
    </a-spin>

    <template #footer>
      <a-space>
        <a-button @click="$emit('update:open', false)">取消</a-button>
        <a-popconfirm
          title="安装到你的工作区？安装后需新开会话才会加载。"
          ok-text="安装"
          @confirm="doInstall"
        >
          <a-button type="primary" :disabled="blocked" :loading="installing">安装</a-button>
        </a-popconfirm>
      </a-space>
    </template>
  </a-drawer>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { marked } from 'marked'
import { message } from 'ant-design-vue'
import { getMarketSkill, getMarketReadme, installSkill } from '@/apis/skills'

const props = defineProps({ open: Boolean, slug: String })
const emit = defineEmits(['update:open', 'installed'])

const loading = ref(false)
const installing = ref(false)
const skill = ref(null)
const readmeHtml = ref('')

const blocked = computed(() =>
  ['flagged', 'removed'].includes(skill.value?.moderation?.state)
)

watch(
  () => [props.open, props.slug],
  async ([open, slug]) => {
    if (!open || !slug) return
    loading.value = true
    skill.value = null
    readmeHtml.value = ''
    try {
      const [meta, readme] = await Promise.all([
        getMarketSkill(slug),
        getMarketReadme(slug).catch(() => ({ content: '' })),
      ])
      skill.value = meta
      readmeHtml.value = marked(readme.content || '')
    } catch (e) {
      message.error(e.message || '加载 skill 详情失败')
      emit('update:open', false)
    } finally {
      loading.value = false
    }
  },
  { immediate: true }
)

async function doInstall() {
  installing.value = true
  try {
    await installSkill(props.slug)
    message.success('已安装，新开会话后生效')
    emit('installed', props.slug)
    emit('update:open', false)
  } catch (e) {
    message.error(e.message || '安装失败')
  } finally {
    installing.value = false
  }
}
</script>

<style scoped>
.trust-row { display: flex; flex-wrap: wrap; gap: 6px; }
.section-label { font-weight: 600; margin: 14px 0 6px; }
.files { margin-top: 12px; }
.readme { font-size: 13px; line-height: 1.6; }
.readme :deep(pre) { background: rgba(0,0,0,0.04); padding: 10px; border-radius: 6px; overflow: auto; }
</style>
```

- [ ] **Step 2: Verify the build**

Run: `pnpm --dir web build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/SkillPreviewDrawer.vue
git commit -m "feat(web): skill preview drawer with trust signals + SKILL.md render"
```

---

### Task 7: Skill market panel (从市场安装 tab)

**Files:**
- Create: `web/src/components/SkillMarket.vue`

**Interfaces:**
- Consumes: `searchMarket` (Task 5), `SkillPreviewDrawer` (Task 6).
- Emits: `installed` (bubbled up from the drawer) so the parent can refresh the pool.
- Behavior: search box → result cards; each card 「预览」opens the drawer for that slug; drawer's install bubbles `installed`.

- [ ] **Step 1: Create the component**

Create `web/src/components/SkillMarket.vue`:

```vue
<template>
  <div class="skill-market">
    <a-input-search
      v-model:value="query"
      placeholder="搜索技能市场，例如 web scraping"
      enter-button="搜索"
      :loading="loading"
      @search="doSearch"
    />

    <a-spin :spinning="loading">
      <div v-if="results.length" class="results">
        <div v-for="r in results" :key="r.slug" class="market-card">
          <div class="market-info">
            <div class="market-name">{{ r.name }}</div>
            <div class="market-meta">@{{ r.owner }} · v{{ r.version }}</div>
            <div v-if="r.summary" class="market-summary">{{ r.summary }}</div>
          </div>
          <a-button size="small" @click="preview(r.slug)">预览</a-button>
        </div>
      </div>
      <a-empty v-else-if="searched" description="没有找到匹配的 skill" />
      <div v-else class="hint">搜索 ClawHub 公共技能市场并安装到你的工作区。</div>
    </a-spin>

    <skill-preview-drawer
      v-model:open="drawerOpen"
      :slug="activeSlug"
      @installed="onInstalled"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { message } from 'ant-design-vue'
import { searchMarket } from '@/apis/skills'
import SkillPreviewDrawer from './SkillPreviewDrawer.vue'

const emit = defineEmits(['installed'])

const query = ref('')
const results = ref([])
const loading = ref(false)
const searched = ref(false)
const drawerOpen = ref(false)
const activeSlug = ref('')

async function doSearch() {
  if (!query.value.trim()) return
  loading.value = true
  try {
    results.value = await searchMarket(query.value.trim())
    searched.value = true
  } catch (e) {
    message.error(e.message || '搜索失败')
  } finally {
    loading.value = false
  }
}

function preview(slug) {
  activeSlug.value = slug
  drawerOpen.value = true
}

function onInstalled(slug) {
  emit('installed', slug)
}
</script>

<style scoped>
.results { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }
.market-card {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 12px; border: 1px solid var(--nr-border, #e6e3da); border-radius: 8px;
}
.market-name { font-weight: 600; }
.market-meta { font-size: 12px; opacity: 0.7; }
.market-summary { font-size: 12px; margin-top: 2px; }
.hint { margin-top: 16px; font-size: 13px; opacity: 0.7; }
</style>
```

- [ ] **Step 2: Verify the build**

Run: `pnpm --dir web build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/SkillMarket.vue
git commit -m "feat(web): skill market search panel"
```

---

### Task 8: Wire tabs + uninstall into the agent modal

Turn the「添加 Skill」modal into two tabs and add an uninstall action on workspace-source pool items.

**Files:**
- Modify: `web/src/views/AgentDetailView.vue` (modal `:204-223`, script `:285-288`, `:352-364`)

**Interfaces:**
- Consumes: `SkillMarket` (Task 7), `uninstallSkill` (Task 5), `agentStore.fetchSkills()`.

- [ ] **Step 1: Import the pieces in the `<script setup>` block**

In `web/src/views/AgentDetailView.vue`, add to the imports:

```javascript
import SkillMarket from '@/components/SkillMarket.vue'
import { uninstallSkill } from '@/apis/skills'
```

- [ ] **Step 2: Add an uninstall handler + market-install refresh (script)**

Add these functions near `addSkill` (~line 352):

```javascript
async function uninstallPoolSkill(name) {
  try {
    await uninstallSkill(name)
    await agentStore.fetchSkills()
    message.success(`已从工作区移除 ${name}`)
  } catch (e) {
    message.error(e.message || '移除失败')
  }
}

async function onMarketInstalled() {
  await agentStore.fetchSkills()
}
```

- [ ] **Step 3: Replace the「添加 Skill」modal body with tabs (template)**

Replace the modal at `:204-223` with:

```vue
<a-modal v-model:open="addSkillOpen" title="添加 / 安装 Skill" :footer="null" width="640">
  <a-tabs>
    <a-tab-pane key="pool" tab="可用池">
      <div v-if="skillsToAdd.length" class="add-skill-list">
        <div v-for="s in skillsToAdd" :key="s.name" class="add-skill-row">
          <div class="add-skill-main" @click="addSkill(s)">
            <div class="add-skill-name">{{ s.name }}</div>
            <div class="add-skill-desc">{{ s.description }}</div>
          </div>
          <div class="add-skill-actions">
            <a-tag size="small" :color="s.source === 'builtin' ? 'blue' : 'green'">
              {{ s.source === 'builtin' ? '内置' : '自定义' }}
            </a-tag>
            <a-popconfirm
              v-if="s.source === 'workspace'"
              title="从工作区卸载该 skill？使用它的 Agent 将失去该能力。"
              @confirm="uninstallPoolSkill(s.name)"
            >
              <a-button size="small" danger type="link">卸载</a-button>
            </a-popconfirm>
          </div>
        </div>
      </div>
      <a-empty v-else description="所有可用 Skill 已添加" />
    </a-tab-pane>

    <a-tab-pane key="market" tab="从市场安装">
      <skill-market @installed="onMarketInstalled" />
    </a-tab-pane>
  </a-tabs>
</a-modal>
```

> The 可用池 tab now shows an item's `添加`(click the main area, attaches to this agent) and, for `source === 'workspace'` items, a `卸载` (removes from the workspace/pool entirely). Built-in items have no 卸载. This is the installed-skills manager, satisfying the "remove installed" scope inside the modal.

- [ ] **Step 4: Add minimal styles for the new row layout**

In the `<style scoped>` block, add / adjust:

```css
.add-skill-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.add-skill-main { flex: 1; cursor: pointer; }
.add-skill-actions { display: flex; align-items: center; gap: 8px; }
```

(If `.add-skill-row` already exists, merge these properties rather than duplicating the selector.)

- [ ] **Step 5: Verify the build**

Run: `pnpm --dir web build`
Expected: build succeeds.

- [ ] **Step 6: Manual E2E (record results)**

With backend + frontend running and a logged-in user:
1. Open an agent → 编辑/详情 → click 「添加」 (Skills card). Modal opens with two tabs.
2. Tab 从市场安装 → search e.g. `pdf` → results render.
3. Click 预览 on a result → drawer shows trust signals + rendered SKILL.md; a script-bundling skill shows the orange warning; a flagged skill disables 安装.
4. Click 安装 → confirm → success toast「已安装，新开会话后生效」.
5. Tab 可用池 → the installed skill now appears with source 自定义 and a 卸载 button; built-in skills have none.
6. Click the installed skill's main area → it attaches to the agent (appears in the Skills card).
7. Click 卸载 on the installed skill → confirm → it disappears from the pool.
8. Break-glass: stop the backend, retry search → friendly「技能市场暂时不可用」toast (502).

- [ ] **Step 7: Commit**

```bash
git add web/src/views/AgentDetailView.vue
git commit -m "feat(web): two-tab add-skill modal with ClawHub market + workspace uninstall"
```

---

## Self-Review

**Spec coverage:**
- Search + browse → Tasks 3,4,7. ✅
- Preview (SKILL.md + trust signals) before install → Tasks 3,4,6. ✅
- Install into caller's own workspace → Task 4 (`user_workspace`, uid from JWT). ✅
- List installed → Task 2 (per-user `/api/skills`, `source: workspace`) surfaced in 可用池 tab (Task 8). ✅
- Remove installed → Task 4 (`DELETE`) + Task 8 (卸载 button). ✅
- Fix `/api/skills` per-user → Task 2. ✅
- Security: self-serve + mandatory preview/confirm → install lives in the preview drawer (Task 6); moderation gating + script warning (Task 6); own-workspace only + path guard (Tasks 1,4). ✅
- Error handling: 502 unreachable, 500 no-npx, 404 missing, traversal guard → Tasks 3,4. ✅
- UI placement inside the modal → Task 8. ✅
- Out of scope (update, create-skill) → not present. ✅

**Deviation from spec (documented):** Remove is a guarded filesystem delete, not `clawhub uninstall` — rationale in Task 4. Search cards omit stars/moderation (search API doesn't return them); those appear on preview — accurate to the API.

**Placeholder scan:** No TBD/TODO; every code step has full content; test code is concrete.

**Type consistency:** `clawhub.search/get_skill/get_readme/install/uninstall` signatures match between Task 3 (definition), Task 4 (router calls + test mocks). Frontend `searchMarket/getMarketSkill/getMarketReadme/installSkill/uninstallSkill` defined in Task 5 and consumed in Tasks 6-8. `installed`/`update:open` events consistent between Tasks 6-8. Skill dict keys (`moderation`, `has_scripts`, `files`, `stats.stars`, `version`, `owner`, `name`, `slug`) consistent between Task 3 output and Task 6 template.
