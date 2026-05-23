"""FastAPI app factory for the nanobot API server."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from nanobot.server.middleware.auth import get_current_user
from nanobot.server.routers.chat_router import router as chat_router


def create_app(agent_loop, session_factory) -> FastAPI:
    app = FastAPI(title="Nanobot API", version="2.0.0")
    app.state.agent_loop = agent_loop
    app.state.session_factory = session_factory
    app.state.run_queues = {}  # run_id (str) -> asyncio.Queue

    @app.post("/api/auth/token")
    async def login(form: OAuth2PasswordRequestForm = Depends()):
        from nanobot.auth.password import verify_password
        from nanobot.auth.jwt import create_token
        from nanobot.storage.repositories.user_repo import UserRepository

        repo = UserRepository(session_factory)
        user = await repo.get_by_uid(form.username)
        if user is None or not verify_password(form.password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
        return {"access_token": create_token(user.uid), "token_type": "bearer"}

    @app.get("/api/auth/me")
    async def me(uid: str = Depends(get_current_user)):
        return {"uid": uid}

    app.include_router(chat_router)
    return app
