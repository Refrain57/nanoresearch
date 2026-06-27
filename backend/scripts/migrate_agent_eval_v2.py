"""Add v2 agent eval schema: recordings/semantic_category/judge_metadata on snapshots,
regression fields on eval runs, and two new tables: judge_calibration_logs,
optimization_proposals.

Usage (from backend/ directory):
    python scripts/migrate_agent_eval_v2.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

from nanobot.storage.database import init_engine
from sqlalchemy import text, inspect as sa_inspect


async def _has_column(conn, table: str, column: str) -> bool:
    result = await conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    )
    return result.scalar() is not None


async def _has_table(conn, table: str) -> bool:
    result = await conn.execute(
        text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = :t"
        ),
        {"t": table},
    )
    return result.scalar() is not None


async def main() -> None:
    init_engine()
    from nanobot.storage.database import _engine

    async with _engine.begin() as conn:

        # 1. agent_run_snapshots — new columns
        for col, ddl in [
            ("tool_recordings",  "JSONB"),
            ("semantic_category", "VARCHAR"),
            ("judge_metadata",   "JSONB"),
        ]:
            if await _has_column(conn, "agent_run_snapshots", col):
                print(f"Already exists: agent_run_snapshots.{col}")
            else:
                await conn.execute(text(
                    f"ALTER TABLE agent_run_snapshots ADD COLUMN {col} {ddl}"
                ))
                print(f"Added: agent_run_snapshots.{col}")

        # 2. agent_eval_runs — regression fields
        for col, ddl in [
            ("baseline_eval_run_id",
             "UUID REFERENCES agent_eval_runs(id) ON DELETE SET NULL"),
            ("regression_diffs", "JSONB"),
            ("has_regression",   "BOOLEAN NOT NULL DEFAULT FALSE"),
        ]:
            if await _has_column(conn, "agent_eval_runs", col):
                print(f"Already exists: agent_eval_runs.{col}")
            else:
                await conn.execute(text(
                    f"ALTER TABLE agent_eval_runs ADD COLUMN {col} {ddl}"
                ))
                print(f"Added: agent_eval_runs.{col}")

        # 3. judge_calibration_logs table
        if await _has_table(conn, "judge_calibration_logs"):
            print("Already exists: judge_calibration_logs")
        else:
            await conn.execute(text("""
                CREATE TABLE judge_calibration_logs (
                    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    judge_model   VARCHAR NOT NULL,
                    mad_value     DOUBLE PRECISION NOT NULL,
                    passed        BOOLEAN NOT NULL,
                    sample_count  INTEGER NOT NULL DEFAULT 0,
                    checked_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    eval_run_id   UUID REFERENCES agent_eval_runs(id) ON DELETE SET NULL
                )
            """))
            await conn.execute(text(
                "CREATE INDEX ix_judge_calibration_logs_checked_at "
                "ON judge_calibration_logs (checked_at DESC)"
            ))
            print("Created: judge_calibration_logs")

        # 4. optimization_proposals table
        if await _has_table(conn, "optimization_proposals"):
            print("Already exists: optimization_proposals")
        else:
            await conn.execute(text("""
                CREATE TABLE optimization_proposals (
                    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    category    VARCHAR NOT NULL,
                    proposals   JSONB NOT NULL DEFAULT '[]',
                    status      VARCHAR NOT NULL DEFAULT 'pending',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    approved_at TIMESTAMPTZ,
                    created_by  VARCHAR REFERENCES users(uid)
                )
            """))
            await conn.execute(text(
                "CREATE INDEX ix_optimization_proposals_status "
                "ON optimization_proposals (status)"
            ))
            await conn.execute(text(
                "CREATE INDEX ix_optimization_proposals_created_at "
                "ON optimization_proposals (created_at DESC)"
            ))
            print("Created: optimization_proposals")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
