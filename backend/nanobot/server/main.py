"""FastAPI app factory for the nanobot API server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from nanobot.server.middleware.auth import get_current_user
from nanobot.server.routers.agent_router import router as agent_router
from nanobot.server.routers.chat_router import router as chat_router


def create_app(agent_loop, session_factory, channel_manager=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = []
        if channel_manager:
            # Channels route inbound messages via bus → agent_loop.run() must be active
            tasks.append(asyncio.create_task(agent_loop.run()))
            tasks.append(asyncio.create_task(channel_manager.start_all()))
        yield
        if channel_manager:
            agent_loop.stop()
            await channel_manager.stop_all()
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="Nanobot API", version="2.0.0", lifespan=lifespan)
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
    app.include_router(agent_router)

    # 生产静态文件服务（pnpm build 产物），放在所有路由之后
    import os
    from pathlib import Path
    dist = Path(__file__).parent.parent.parent.parent / "web" / "dist"
    if dist.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")

    return app
