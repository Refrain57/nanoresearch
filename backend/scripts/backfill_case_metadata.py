"""Backfill B4 metadata fields on existing agent_test_cases rows.

Default values applied:
- origin_badcase_id: NULL (cases pre-dating the badcase pipeline have no origin)
- target_dimension: "legacy_pre_b4" (sentinel value — distinguishable from real future values)
- added_at: rows.created_at if present, else now()
- added_by: "system:backfill_2026_06"
- coverage_tags: []  (empty array)

Usage:
    python -m scripts.backfill_case_metadata --dry-run
    python -m scripts.backfill_case_metadata --apply
"""
import argparse
import asyncio
import os
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from nanoresearch.storage.models import AgentTestCase

LEGACY_DIMENSION = "legacy_pre_b4"
BACKFILL_AUTHOR = "system:backfill_2026_06"


async def main(apply: bool) -> int:
    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with AsyncSession(engine) as session:
        # I2: safety guard — if enforce migration already ran, target_dimension is NOT NULL.
        # Re-running backfill after enforce would silently find 0 rows, which is ambiguous.
        nullable_check = await session.execute(
            text(
                "SELECT is_nullable FROM information_schema.columns"
                " WHERE table_name = 'agent_test_cases' AND column_name = 'target_dimension'"
            )
        )
        nullable_row = nullable_check.fetchone()
        if nullable_row is not None and nullable_row[0] == "NO":
            print(
                "WARNING: target_dimension is already NOT NULL — enforce migration has already run."
                " If you intended to re-backfill, you must drop the constraint first. No-op."
            )
            return 0

        result = await session.execute(
            select(AgentTestCase).where(AgentTestCase.target_dimension.is_(None))
        )
        rows = result.scalars().all()
        print(f"Found {len(rows)} rows missing B4 metadata.")
        if not apply:
            for r in rows[:5]:
                print(f"  would backfill id={r.id} name={r.name}")
            print("(dry run — no writes)")
            return 0

        now = datetime.now(timezone.utc)
        for r in rows:
            r.target_dimension = LEGACY_DIMENSION
            r.added_at = r.created_at or now
            r.added_by = BACKFILL_AUTHOR
            r.coverage_tags = []
        await session.commit()
        print(f"Backfilled {len(rows)} rows.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(apply=args.apply)))
