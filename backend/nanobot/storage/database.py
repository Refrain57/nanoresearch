"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


_engine = None
_AsyncSessionLocal: async_sessionmaker | None = None


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL 环境变量未设置。"
            "示例：postgresql+asyncpg://postgres:postgres@localhost:5432/nanoresearch"
        )
    return url


def init_engine(database_url: str | None = None) -> None:
    global _engine, _AsyncSessionLocal
    url = database_url or get_database_url()
    _engine = create_async_engine(url, echo=False, pool_pre_ping=True)
    _AsyncSessionLocal = async_sessionmaker(_engine, expire_on_commit=False)


def get_session_factory() -> async_sessionmaker:
    if _AsyncSessionLocal is None:
        raise RuntimeError("DB engine not initialized. Call init_engine() first.")
    return _AsyncSessionLocal


async def init_db() -> None:
    """Create all tables if they don't exist."""
    from nanobot.storage import models as _  # noqa: F401 — ensure models are registered
    if _engine is None:
        raise RuntimeError("DB engine not initialized. Call init_engine() first.")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield an async DB session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
