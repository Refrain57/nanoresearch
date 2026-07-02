#!/usr/bin/env python
"""Migrate the legacy JSON cron store into the cron_jobs table.

Usage:
    cd backend
    DATABASE_URL=... uv run scripts/migrate_cron_jobs_from_json.py <workspace>/cron/jobs.json <admin_uid>
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: migrate_cron_jobs_from_json.py <jobs.json> <default_uid>", file=sys.stderr)
        sys.exit(1)
    json_path, default_uid = Path(sys.argv[1]), sys.argv[2]

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL 未设置", file=sys.stderr)
        sys.exit(1)

    from nanoresearch.cron.migrate_json import migrate_jobs_json
    from nanoresearch.storage.database import get_session_factory, init_engine

    init_engine(database_url)
    factory = get_session_factory()
    n = await migrate_jobs_json(json_path, factory, default_uid=default_uid)
    print(f"Migrated {n} cron job(s) from {json_path}.")


if __name__ == "__main__":
    asyncio.run(main())
