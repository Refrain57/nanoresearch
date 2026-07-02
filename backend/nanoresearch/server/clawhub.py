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
