"""Add replay-lineage columns to agent_run_snapshots (route B replay-as-run).

The server does not run Base.metadata.create_all at startup, so these columns
must be added explicitly (mirrors migrate_add_kg_entity_articles.py).

Usage:
    python -m nanoresearch.scripts.migrate_replay_lineage
"""

from __future__ import annotations

import asyncio


async def migrate() -> None:
    from nanoresearch.storage.database import init_engine, get_session_factory
    import sqlalchemy as sa

    init_engine()
    factory = get_session_factory()

    ddl_statements = [
        "ALTER TABLE agent_run_snapshots ADD COLUMN IF NOT EXISTS origin VARCHAR NOT NULL DEFAULT 'live'",
        "ALTER TABLE agent_run_snapshots ADD COLUMN IF NOT EXISTS parent_snapshot_id UUID "
        "REFERENCES agent_run_snapshots(id) ON DELETE SET NULL",
        "ALTER TABLE agent_run_snapshots ADD COLUMN IF NOT EXISTS root_snapshot_id UUID "
        "REFERENCES agent_run_snapshots(id) ON DELETE SET NULL",
        "ALTER TABLE agent_run_snapshots ADD COLUMN IF NOT EXISTS replay_config JSONB",
        "CREATE INDEX IF NOT EXISTS ix_agent_run_snapshots_root ON agent_run_snapshots (root_snapshot_id)",
        "CREATE INDEX IF NOT EXISTS ix_agent_run_snapshots_parent ON agent_run_snapshots (parent_snapshot_id)",
    ]

    async with factory() as session:
        for stmt in ddl_statements:
            await session.execute(sa.text(stmt))
        await session.commit()

    print("[migrate_replay_lineage] Done — replay-lineage columns added (idempotent).")


if __name__ == "__main__":
    asyncio.run(migrate())
