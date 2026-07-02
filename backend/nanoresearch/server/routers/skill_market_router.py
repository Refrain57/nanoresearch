"""ClawHub skill marketplace + workspace install/remove endpoints."""
from __future__ import annotations

import re
import shutil

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from nanoresearch.server import clawhub
from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.server.routers.workspace_paths import safe_resolve, user_workspace

SLUG_RE = re.compile(r"^@?[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

router = APIRouter()


def _validate_slug(slug: str) -> None:
    if not SLUG_RE.match(slug):
        raise HTTPException(status_code=400, detail="非法的 skill 标识")


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
    _validate_slug(slug)
    try:
        return {"content": await clawhub.get_readme(slug)}
    except clawhub.ClawHubError:
        raise HTTPException(status_code=502, detail="技能市场暂时不可用")


@router.get("/api/skills/market/{slug:path}")
async def market_skill(slug: str, _uid: str = Depends(get_current_user)):
    _validate_slug(slug)
    try:
        return await clawhub.get_skill(slug)
    except clawhub.ClawHubError:
        raise HTTPException(status_code=502, detail="技能市场暂时不可用")


@router.post("/api/skills/install")
async def install_skill(body: InstallBody, request: Request, uid: str = Depends(get_current_user)):
    _validate_slug(body.slug)
    workdir = user_workspace(request, uid)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        skill = await clawhub.get_skill(body.slug)
    except clawhub.ClawHubError:
        raise HTTPException(status_code=502, detail="技能市场暂时不可用")
    if (skill.get("moderation") or {}).get("state") in {"flagged", "removed"}:
        raise HTTPException(status_code=403, detail="该 skill 已被市场标记，禁止安装")
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
