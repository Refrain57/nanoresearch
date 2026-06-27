"""Add question_type column to eval_dataset_items and eval_run_items.

Usage (from backend/ directory):
    python scripts/migrate_question_type.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

from nanobot.storage.database import init_engine
from sqlalchemy import text

COLUMNS = [
    ("eval_dataset_items", "question_type", "TEXT"),
    ("eval_run_items",     "question_type", "TEXT"),
]


async def main() -> None:
    init_engine()
    from nanobot.storage.database import _engine

    async with _engine.begin() as conn:
        for table, col, dtype in COLUMNS:
            result = await conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = :c"
                ),
                {"t": table, "c": col},
            )
            if result.scalar() is None:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
                )
                print(f"Added {table}.{col}")
            else:
                print(f"Already exists: {table}.{col}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
