"""Workspace file browser and download endpoints."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from nanobot.server.middleware.auth import get_current_user

router = APIRouter()

EDITABLE_FILES = {"SOUL.md", "AGENTS.md", "USER.md", "TOOLS.md"}


class FileWriteBody(BaseModel):
    content: str


def _user_workspace(request: Request, uid: str) -> Path:
    cfg = getattr(request.app.state, "loop_config", None) or {}
    base = cfg.get("base_workspace")
    if base is None:
        from nanobot.config.paths import get_workspace_path
        base = get_workspace_path()
    return Path(base) / "users" / uid


def _safe_resolve(workspace: Path, rel_path: str) -> Path:
    resolved = (workspace / rel_path).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="非法路径")
    return resolved


def _file_entry(path: Path, workspace: Path) -> dict:
    stat = path.stat()
    return {
        "path": path.relative_to(workspace).as_posix(),
        "name": path.name,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "is_dir": path.is_dir(),
    }


@router.get("/api/workspace/files")
async def list_files(
    request: Request,
    dir: str = "",
    uid: str = Depends(get_current_user),
):
    workspace = _user_workspace(request, uid)
    target = _safe_resolve(workspace, dir) if dir else workspace.resolve()

    if not target.exists():
        return []

    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    return [_file_entry(p, workspace) for p in entries]


@router.put("/api/workspace/files/{file_path:path}")
async def write_file(
    file_path: str,
    body: FileWriteBody,
    request: Request,
    uid: str = Depends(get_current_user),
):
    if Path(file_path).name not in EDITABLE_FILES:
        raise HTTPException(status_code=403, detail="该文件不允许通过 API 编辑")
    workspace = _user_workspace(request, uid)
    resolved = _safe_resolve(workspace, file_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(body.content, encoding="utf-8")
    return {"path": file_path, "size": len(body.content.encode("utf-8"))}


@router.get("/api/workspace/files/{file_path:path}")
async def download_file(
    file_path: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    workspace = _user_workspace(request, uid)
    resolved = _safe_resolve(workspace, file_path)

    if not resolved.exists() or resolved.is_dir():
        raise HTTPException(status_code=404, detail="文件不存在")

    media_type, _ = mimetypes.guess_type(str(resolved))
    return FileResponse(
        path=str(resolved),
        filename=resolved.name,
        media_type=media_type or "application/octet-stream",
    )
