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


def build_attachment_descriptors(
    media: list[str] | None, workspace_root: Path
) -> list[dict]:
    """Map absolute media paths to workspace-relative attachment descriptors.

    Drops any path outside workspace_root, non-existent, or not a file.
    """
    root = workspace_root.resolve()
    out: list[dict] = []
    for p in media or []:
        try:
            rp = Path(p).resolve()
            rel = rp.relative_to(root)
        except (ValueError, OSError):
            continue
        try:
            if not rp.is_file():
                continue
            size = rp.stat().st_size
        except OSError:
            continue
        out.append({"path": rel.as_posix(), "name": rp.name, "size": size})
    return out
