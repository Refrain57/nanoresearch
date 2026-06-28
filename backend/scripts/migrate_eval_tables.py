"""Create agent_run_snapshots and agent_test_cases tables.

Usage (from backend/ directory):
    python scripts/migrate_eval_tables.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Allow running from backend/ directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

from nanoresearch.storage.database import init_engine, get_session_factory
from sqlalchemy import text, inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncConnection


NEW_TABLES = ["agent_run_snapshots", "agent_test_cases"]


async def main() -> None:
    init_engine()
    from nanoresearch.storage.database import _engine
    from nanoresearch.storage import models as _  # register all models

    async with _engine.begin() as conn:
        # Check which tables already exist
        existing = await conn.run_sync(
            lambda sync_conn: sa_inspect(sync_conn).get_table_names()
        )
        to_create = [t for t in NEW_TABLES if t not in existing]
        already = [t for t in NEW_TABLES if t in existing]

        if already:
            print(f"Already exists: {', '.join(already)}")

        if not to_create:
            print("Nothing to do — both tables already present.")
            return

        print(f"Creating: {', '.join(to_create)} ...")
        from nanoresearch.storage.database import Base
        # Create only missing tables
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(
                sync_conn,
                tables=[Base.metadata.tables[t] for t in to_create],
            )
        )
        print("Done.")

        # Quick verify
        async with _engine.connect() as verify_conn:
            for table in to_create:
                result = await verify_conn.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                )
                print(f"  {table}: OK (0 rows)")


if __name__ == "__main__":
    asyncio.run(main())
