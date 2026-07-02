"""Add knowledge graph entity article cache table: kg_entity_articles.

Backfills the wiki Phase-2 (LLM entity pages) table on existing databases.
The server does not run Base.metadata.create_all at startup, so this table
must be created explicitly (mirrors migrate_add_kg_tables.py).

Usage:
    python -m nanoresearch.scripts.migrate_add_kg_entity_articles
"""

from __future__ import annotations

import asyncio


async def migrate() -> None:
    from nanoresearch.storage.database import init_engine, get_session_factory
    import sqlalchemy as sa

    init_engine()
    factory = get_session_factory()

    ddl_statements = [
        # kg_entity_articles — cached LLM-generated entity article, keyed by (kb_id, entity_name)
        """
        CREATE TABLE IF NOT EXISTS kg_entity_articles (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            kb_id         UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            entity_name   VARCHAR NOT NULL,
            markdown      TEXT NOT NULL,
            citations     JSONB NOT NULL DEFAULT '[]',
            evidence_hash VARCHAR NOT NULL,
            model         VARCHAR,
            generated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_kg_entity_article UNIQUE (kb_id, entity_name)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_kg_entity_articles_kb_id ON kg_entity_articles (kb_id)",
    ]

    async with factory() as session:
        for stmt in ddl_statements:
            await session.execute(sa.text(stmt.strip()))
        await session.commit()

    print("[migrate_add_kg_entity_articles] Done — kg_entity_articles created (idempotent).")


if __name__ == "__main__":
    asyncio.run(migrate())
