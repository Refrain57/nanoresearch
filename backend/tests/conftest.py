"""Shared test fixtures — DB setup via psycopg2 (sync), bypassing asyncpg/event-loop issues."""

import os
import psycopg2
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import pytest

from nanoresearch.storage import models as _  # noqa: F401 — register all ORM models
from nanoresearch.storage.database import Base

# ── Connection config ─────────────────────────────────────────────────────────

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:123456@localhost:5432/nanoresearch_test",
)

# psycopg2 uses a different URL scheme
_PG_DSN = os.environ.get(
    "TEST_DATABASE_DSN",
    "host=localhost port=5432 dbname=nanoresearch_test user=postgres password=123456",
)


# ── Sync DB helpers (psycopg2) ────────────────────────────────────────────────

def pg_conn():
    """Return a new psycopg2 connection with autocommit."""
    conn = psycopg2.connect(_PG_DSN)
    conn.autocommit = True
    return conn


def create_tables() -> None:
    """Drop and recreate all tables using SQLAlchemy metadata + psycopg2."""
    from sqlalchemy import create_engine
    sync_url = TEST_DB_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    engine = create_engine(sync_url, echo=False)
    try:
        Base.metadata.drop_all(engine)
    except Exception:
        # FK constraints from tables outside Base.metadata (e.g. cron_jobs) may
        # prevent drop_all. Fall through — create_all is idempotent and truncate_all
        # handles per-test data isolation.
        pass
    Base.metadata.create_all(engine)
    engine.dispose()


def truncate_all() -> None:
    """Truncate all tables between tests."""
    conn = pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE TABLE agent_runs, messages, conversations, agents, users "
                "RESTART IDENTITY CASCADE"
            )
    finally:
        conn.close()


# ── Async session factory (for repository tests) ──────────────────────────────

def make_factory(url: str = TEST_DB_URL) -> async_sessionmaker:
    """Create an async session factory for use in async test helpers."""
    engine = create_async_engine(url, echo=False)
    return async_sessionmaker(engine, expire_on_commit=False)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create all tables once per session."""
    create_tables()


# ── Redis fixture (Phase 0 bus tests) ─────────────────────────────────────────

import pytest_asyncio  # noqa: E402


@pytest_asyncio.fixture
async def redis_client():
    """Async Redis client against test DB 15; skip if Redis unreachable.

    Flushes DB 15 before and after each test so cases never bleed into each other.
    DB 15 keeps tests isolated from any dev/prod Redis data on DB 0.
    """
    import os

    import redis.asyncio as aioredis

    url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
    r = aioredis.from_url(url, decode_responses=True)
    try:
        await r.ping()
    except Exception:
        pytest.skip("Redis not reachable for tests")
    await r.flushdb()
    yield r
    try:
        await r.flushdb()
    finally:
        await r.aclose()
