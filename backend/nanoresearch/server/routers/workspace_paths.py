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
