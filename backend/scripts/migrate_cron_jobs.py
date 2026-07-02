#!/usr/bin/env python
"""Create the cron_jobs table (production cron redesign).

Idempotent: uses SQLAlchemy Table.create(checkfirst=True), so re-running is a
no-op. The ORM model is the single source of truth for the schema.

Usage:
    cd backend
    DATABASE_URL=... uv run scripts/migrate_cron_jobs.py
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

    from nanoresearch.storage.database import init_engine
    from nanoresearch.storage.models import CronJob

    init_engine(database_url)
    from nanoresearch.storage.database import _engine as engine
    async with engine.begin() as conn:
        await conn.run_sync(CronJob.__table__.create, checkfirst=True)
    print("cron_jobs table ensured.")


if __name__ == "__main__":
    asyncio.run(main())
