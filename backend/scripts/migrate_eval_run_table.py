"""Create agent_eval_runs table and add eval_run_id column to agent_run_snapshots.

Usage (from backend/ directory):
    python scripts/migrate_eval_run_table.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

from nanoresearch.storage.database import init_engine, get_session_factory
from sqlalchemy import text, inspect as sa_inspect


async def main() -> None:
    init_engine()
    from nanoresearch.storage.database import _engine
    from nanoresearch.storage import models as _  # register all models
    from nanoresearch.storage.database import Base

    async with _engine.begin() as conn:
        existing = await conn.run_sync(
            lambda sync_conn: sa_inspect(sync_conn).get_table_names()
        )

        # 1. Create agent_eval_runs table if missing
        if "agent_eval_runs" not in existing:
            print("Creating agent_eval_runs ...")
            await conn.run_sync(
                lambda sync_conn: Base.metadata.create_all(
                    sync_conn,
                    tables=[Base.metadata.tables["agent_eval_runs"]],
                )
            )
            print("  agent_eval_runs: OK")
        else:
            print("Already exists: agent_eval_runs")

        # 2. Add eval_run_id column to agent_run_snapshots if missing
        async with _engine.connect() as check_conn:
            cols_result = await check_conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'agent_run_snapshots' AND column_name = 'eval_run_id'"
                )
            )
            has_col = cols_result.fetchone() is not None

        if not has_col:
            print("Adding eval_run_id column to agent_run_snapshots ...")
            await conn.execute(
                text(
                    "ALTER TABLE agent_run_snapshots "
                    "ADD COLUMN eval_run_id UUID REFERENCES agent_eval_runs(id) ON DELETE SET NULL"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_agent_run_snapshots_eval_run_id "
                    "ON agent_run_snapshots (eval_run_id)"
                )
            )
            print("  eval_run_id column: OK")
        else:
            print("Already exists: agent_run_snapshots.eval_run_id")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
