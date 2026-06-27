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


async def check_schema_migrations() -> None:
    """Warn at startup if pending schema migrations are detected.

    This catches the case where the DB has the old schema but the code expects
    new columns — fail-fast with a clear message instead of a cryptic 500 at
    runtime.
    """
    if _engine is None:
        return
    CHECKS = [
        ("eval_runs", "eval_type"),
        ("eval_runs", "error_message"),
        ("eval_run_items", "retrieved_contexts"),
        ("user_settings", "uid"),
        ("knowledge_bases", "chunk_strategy"),
        ("agents", "harness"),
        ("agents", "persona"),
        # KG tables (migrate_add_kg_tables.py)
        ("knowledge_bases", "enable_graph_expansion"),
        ("kg_entities", "id"),
        ("kg_entity_mentions", "id"),
        ("kg_triples", "id"),
        ("kg_triple_mentions", "id"),
        # Agent evaluation tables
        ("agent_run_snapshots", "id"),
        ("agent_test_cases", "id"),
        ("agent_eval_runs", "id"),
        ("agent_run_snapshots", "eval_run_id"),
        # question_type for eval dataset items and run items
        ("eval_dataset_items", "question_type"),
        ("eval_run_items", "question_type"),
        # Agent eval v2 (migrate_agent_eval_v2.py)
        ("agent_run_snapshots", "tool_recordings"),
        ("agent_run_snapshots", "semantic_category"),
        ("agent_run_snapshots", "judge_metadata"),
        ("agent_eval_runs", "baseline_eval_run_id"),
        ("agent_eval_runs", "has_regression"),
        ("judge_calibration_logs", "id"),
        ("optimization_proposals", "id"),
        # Phase 0: context trace (migrate_phase0_context_trace.sql)
        ("agent_run_snapshots", "context_trace"),
        # Phase 1: structured root-cause pointer + tunable version registry
        ("agent_run_snapshots", "classification_layer"),
        ("agent_run_snapshots", "classification_target_kind"),
        ("agent_run_snapshots", "classification_target_id"),
        ("tunable_object_versions", "id"),
        # Phase 2: regression set separation (migrate_phase2_health_set.sql)
        ("agent_test_cases", "set_kind"),
        ("agent_test_cases", "tool_recordings"),
        ("kb_documents", "content_hash"),
        # Phase 5: baseline anchor + deployment gate (migrate_phase5_baseline_gate.sql)
        ("optimization_proposals", "baseline_score"),
        ("optimization_proposals", "baseline_version_id"),
    ]
    missing = []
    async with _engine.connect() as conn:
        for table, column in CHECKS:
            result = await conn.execute(
                __import__("sqlalchemy").text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = :c"
                ),
                {"t": table, "c": column},
            )
            if result.scalar() is None:
                missing.append(f"{table}.{column}")
    if missing:
        import sys
        print(
            "\n⚠️  Schema migration required before starting the server.\n"
            f"   Missing columns: {', '.join(missing)}\n"
            "   Run the relevant migration script in scripts/\n",
            file=sys.stderr,
        )
        raise SystemExit(1)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield an async DB session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
