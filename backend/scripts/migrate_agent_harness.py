#!/usr/bin/env python
"""Add harness JSONB column to agents table.

Usage:
    cd backend
    DATABASE_URL=... uv run scripts/migrate_agent_harness.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL 未设置", file=sys.stderr)
        sys.exit(1)

    from nanobot.storage.database import init_engine
    import sqlalchemy as sa

    init_engine(database_url)

    from nanobot.storage.database import _engine as engine
    async with engine.begin() as conn:
        result = await conn.execute(sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'agents' AND column_name = 'harness'"
        ))
        if result.scalar():
            print("Column agents.harness already exists — skipping.")
            return
        await conn.execute(sa.text(
            "ALTER TABLE agents ADD COLUMN harness JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        print("Added agents.harness column.")


if __name__ == "__main__":
    asyncio.run(main())
