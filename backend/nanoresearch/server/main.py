"""FastAPI app factory for the nanoresearch API server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request as _Req, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm

from nanoresearch.utils.env_compat import apply_legacy_env_compat

apply_legacy_env_compat()

from nanoresearch.providers.model_factory import ModelResolutionError  # noqa: E402
from nanoresearch.server.middleware.auth import get_current_user  # noqa: E402


async def _missing_provider_handler(request: _Req, exc: ModelResolutionError) -> JSONResponse:
    """Global handler: ModelResolutionError → structured 422 JSON response."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "missing_provider",
            "role": exc.missing_role or "",
            "message": str(exc),
        },
    )


def create_app(channel_loop, session_factory, loop_config=None, channel_manager=None, rag_settings=None, allowed_models=None, config=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One-time migration: ensure settings.yaml api_keys exist in config.json
        try:
            from nanoresearch.config.migration import migrate_llm_keys
            migrate_llm_keys(dry_run=False)
        except Exception:
            pass

        tasks = []
        # Dispose stale asyncpg connections from any previous process before accepting requests.
        # Without this, the first request after a restart fails on Windows (ProactorEventLoop
        # proactor is None on old connections).
        from nanoresearch.storage.database import _engine
        if _engine is not None:
            await _engine.dispose()

        # Redis client — shared across all requests (Problem 7: decode_responses=True enforced)
        from nanoresearch.bus.redis_client import get_redis
        app.state.redis = get_redis()

        # PendingReaper — background orphan cleanup, attached to this process's lifespan
        from nanoresearch.bus.pending_reaper import PendingReaper
        from nanoresearch.bus.redis_monitor import RedisMonitor
        app.state.pending_reaper = PendingReaper()
        await app.state.pending_reaper.start()

        # RedisMonitor — eviction alerting (3-A) and memory sampling (3-B)
        app.state.redis_monitor = RedisMonitor()
        await app.state.redis_monitor.start()

        # ARQ pool — enqueues agent jobs into the worker process
        from arq import create_pool
        from arq.connections import RedisSettings as ArqRedisSettings
        from nanoresearch.bus.redis_client import REDIS_URL
        app.state.arq_pool = await create_pool(ArqRedisSettings.from_dsn(REDIS_URL))

        # AgentDispatcher — the SOLE job enqueuer: consumes the notify stream and enqueues
        # run_agent_job per mailbox. Depends on redis + arq_pool (both initialised above).
        from nanoresearch.bus.dispatcher import AgentDispatcher
        app.state.dispatcher = AgentDispatcher(app.state.redis, app.state.arq_pool)
        await app.state.dispatcher.start()

        if channel_manager:
            # Channels route inbound messages via bus → channel_loop.run() must be active
            tasks.append(asyncio.create_task(channel_loop.run()))
            tasks.append(asyncio.create_task(channel_manager.start_all()))
        yield
        if channel_manager:
            channel_loop.stop()
            await channel_manager.stop_all()
        for loop in list(app.state.web_loops.values()):
            loop.stop()
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if getattr(app.state, "dispatcher", None):
            await app.state.dispatcher.stop()
        await app.state.pending_reaper.stop()
        await app.state.redis_monitor.stop()
        try:
            await app.state.arq_pool.aclose()
        except Exception:
            pass
        await app.state.redis.aclose()

    app = FastAPI(title="Nanoresearch API", version="2.0.0", lifespan=lifespan)
    app.add_exception_handler(ModelResolutionError, _missing_provider_handler)
    app.state.channel_loop = channel_loop
    app.state.web_loops = {}              # legacy placeholder — agent loops live in worker
    app.state.web_loops_lock = asyncio.Lock()
    app.state.loop_config = loop_config or {}
    app.state.session_factory = session_factory
    app.state.arq_pool = None      # initialised in lifespan before first request
    app.state.redis = None         # initialised in lifespan before first request
    app.state.redis_monitor = None # initialised in lifespan before first request
    app.state.dispatcher = None    # initialised in lifespan before first request
    app.state.rag_settings = rag_settings  # loaded lazily if None
    app.state.allowed_models = allowed_models or []
    app.state.config = config

    @app.post("/api/auth/token")
    async def login(form: OAuth2PasswordRequestForm = Depends()):
        from nanoresearch.auth.password import verify_password
        from nanoresearch.auth.jwt import create_token
        from nanoresearch.storage.repositories.user_repo import UserRepository

        repo = UserRepository(session_factory)
        user = await repo.get_by_uid(form.username)
        if user is None or not verify_password(form.password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
        return {"access_token": create_token(user.uid), "token_type": "bearer"}

    @app.get("/api/auth/me")
    async def me(uid: str = Depends(get_current_user)):
        return {"uid": uid}

    from nanoresearch.server.routers.agent_router import router as agent_router
    from nanoresearch.server.routers.agent_eval_router import router as agent_eval_router
    from nanoresearch.server.routers.chat_router import router as chat_router
    from nanoresearch.server.routers.knowledge_router import router as knowledge_router
    from nanoresearch.server.routers.eval_router import router as eval_router
    from nanoresearch.server.routers.settings_router import router as settings_router
    from nanoresearch.server.routers.workspace_router import router as workspace_router

    app.include_router(chat_router)
    app.include_router(agent_router)
    app.include_router(agent_eval_router)
    app.include_router(knowledge_router)
    app.include_router(eval_router)
    app.include_router(settings_router)
    app.include_router(workspace_router)

    # RAG 图片静态文件服务（必须在前端 "/" 挂载之前）
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from nanoresearch.config.loader import get_nanoresearch_home
    rag_images_dir = get_nanoresearch_home() / "rag" / "images"
    rag_images_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/rag-images", StaticFiles(directory=str(rag_images_dir)), name="rag-images")

    # 生产静态文件服务（pnpm build 产物），放在所有路由之后
    dist = Path(__file__).parent.parent.parent.parent / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")

    return app
