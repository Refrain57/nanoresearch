"""Dashboard and auth API server."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

SKILLS_DIR = Path(__file__).parent.parent / "skills"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


async def _get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    from nanoresearch.auth.jwt import verify_token
    return verify_token(token)


def create_app(workspace: Path) -> FastAPI:
    app = FastAPI()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    @app.post("/api/auth/token")
    async def login(form: OAuth2PasswordRequestForm = Depends()):
        from nanoresearch.auth.password import verify_password
        from nanoresearch.auth.jwt import create_token
        from nanoresearch.storage.database import get_session_factory
        from nanoresearch.storage.repositories.user_repo import UserRepository

        try:
            factory = get_session_factory()
            repo = UserRepository(factory)
            user = await repo.get_by_uid(form.username)
        except RuntimeError:
            # DB not configured — single-user fallback via env vars
            import os
            expected_user = os.environ.get("ADMIN_USERNAME", "admin")
            expected_hash = os.environ.get("ADMIN_PASSWORD_HASH", "")
            if form.username != expected_user or not verify_password(form.password, expected_hash):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
            return {"access_token": create_token(form.username), "token_type": "bearer"}

        if user is None or not verify_password(form.password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
        return {"access_token": create_token(user.uid), "token_type": "bearer"}

    # ------------------------------------------------------------------
    # Dashboard state (now requires auth)
    # ------------------------------------------------------------------

    @app.get("/api/state")
    async def get_state(uid: str = Depends(_get_current_user)):
        sessions = _read_sessions(workspace)
        return {
            "sessions": sessions,
            "agents": _derive_agents(sessions),
            "skills": _read_skills(workspace),
            "cron_jobs": _read_cron_jobs(workspace),
            "recent_activity": _read_recent_activity(workspace),
            "memory_preview": _read_memory_preview(workspace),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app


def _derive_agents(sessions: list[dict]) -> list[dict]:
    return [
        {
            "name": s["display_name"],
            "session_key": s["key"],
            "channel": s["channel"],
            "chat_id": s["chat_id"],
            "display_name": s["display_name"],
            "status": "idle",
            "last_active": s.get("updated_at"),
        }
        for s in sessions
    ]


def _read_sessions(workspace: Path) -> list[dict]:
    results = []
    sessions_dir = workspace / "sessions"
    if not sessions_dir.exists():
        return results
    for path in sorted(sessions_dir.glob("*.jsonl")):
        try:
            with open(path, encoding="utf-8") as f:
                first_line = f.readline()
                count = sum(1 for _ in f)
            meta = json.loads(first_line)
            key = meta.get("key", path.stem.replace("_", ":", 1))
            channel, _, chat_id = key.partition(":")
            user_meta = meta.get("metadata", {})
            display = user_meta.get("user_name") or user_meta.get("chat_name") or chat_id
            results.append({
                "key": key,
                "channel": channel,
                "chat_id": chat_id,
                "display_name": display,
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "message_count": count,
                "last_consolidated": meta.get("last_consolidated", 0),
            })
        except Exception:
            continue
    results.sort(key=lambda s: s.get("updated_at") or "", reverse=True)
    return results


def _parse_frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    result = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip().strip('"')
    return result


def _read_skills(workspace: Path) -> list[dict]:
    seen: set[str] = set()
    results: list[dict] = []
    for source_label, base in [("user", workspace / "skills"), ("builtin", SKILLS_DIR)]:
        if not base.exists():
            continue
        for skill_md in sorted(base.glob("*/SKILL.md")):
            try:
                text = skill_md.read_text(encoding="utf-8")
                fm = _parse_frontmatter(text)
                name = fm.get("name") or skill_md.parent.name
                if name in seen:
                    continue
                seen.add(name)
                results.append({
                    "name": name,
                    "description": fm.get("description", ""),
                    "always": fm.get("always", "false").lower() == "true",
                    "source": source_label,
                })
            except Exception:
                continue
    return results


def _ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _read_cron_jobs(workspace: Path) -> list[dict]:
    # Cron production redesign: jobs live in the cron_jobs DB table, not the legacy JSON store.
    # This legacy gateway dashboard has no DB session; the cron panel is served by the web UI.
    return []


def _read_recent_activity(workspace: Path) -> list[dict]:
    return []


def _read_memory_preview(workspace: Path) -> str:
    path = workspace / "memory" / "MEMORY.md"
    try:
        text = path.read_text(encoding="utf-8")
        return text[:800] + ("..." if len(text) > 800 else "")
    except Exception:
        return ""
