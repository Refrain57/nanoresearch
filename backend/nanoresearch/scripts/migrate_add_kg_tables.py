"""Add knowledge graph tables: kg_entities, kg_entity_mentions, kg_triples, kg_triple_mentions.

Usage:
    python -m nanoresearch.scripts.migrate_add_kg_tables
"""

from __future__ import annotations

import asyncio
import sys


async def migrate() -> None:
    from nanoresearch.storage.database import init_engine, get_session_factory
    import sqlalchemy as sa

    init_engine()
    factory = get_session_factory()

    ddl_statements = [
        # knowledge_bases: graph expansion flag
        """
        ALTER TABLE knowledge_bases
        ADD COLUMN IF NOT EXISTS enable_graph_expansion BOOLEAN NOT NULL DEFAULT FALSE
        """,

        # kg_entities
        """
        CREATE TABLE IF NOT EXISTS kg_entities (
            id          UUID PRIMARY KEY,
            kb_id       UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            name        VARCHAR NOT NULL,
            label       VARCHAR NOT NULL DEFAULT 'Entity',
            attributes  JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_kg_entity_kb_name_label UNIQUE (kb_id, name, label)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_kg_entities_kb_id ON kg_entities (kb_id)",

        # kg_entity_mentions
        """
        CREATE TABLE IF NOT EXISTS kg_entity_mentions (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_id   UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
            chunk_id    UUID NOT NULL REFERENCES kb_chunks(id) ON DELETE CASCADE,
            kb_id       UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_kg_entity_mention UNIQUE (entity_id, chunk_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_kg_entity_mentions_entity_id ON kg_entity_mentions (entity_id)",
        "CREATE INDEX IF NOT EXISTS ix_kg_entity_mentions_chunk_id  ON kg_entity_mentions (chunk_id)",
        "CREATE INDEX IF NOT EXISTS ix_kg_entity_mentions_kb_id     ON kg_entity_mentions (kb_id)",

        # kg_triples
        """
        CREATE TABLE IF NOT EXISTS kg_triples (
            id          UUID PRIMARY KEY,
            kb_id       UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            source_id   UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
            target_id   UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
            label       VARCHAR NOT NULL DEFAULT 'RELATED_TO',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_kg_triple UNIQUE (kb_id, source_id, target_id, label)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_kg_triples_kb_id     ON kg_triples (kb_id)",
        "CREATE INDEX IF NOT EXISTS ix_kg_triples_source_id ON kg_triples (source_id)",
        "CREATE INDEX IF NOT EXISTS ix_kg_triples_target_id ON kg_triples (target_id)",

        # kg_triple_mentions
        """
        CREATE TABLE IF NOT EXISTS kg_triple_mentions (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            triple_id   UUID NOT NULL REFERENCES kg_triples(id) ON DELETE CASCADE,
            chunk_id    UUID NOT NULL REFERENCES kb_chunks(id) ON DELETE CASCADE,
            kb_id       UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_kg_triple_mention UNIQUE (triple_id, chunk_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_kg_triple_mentions_triple_id ON kg_triple_mentions (triple_id)",
        "CREATE INDEX IF NOT EXISTS ix_kg_triple_mentions_chunk_id  ON kg_triple_mentions (chunk_id)",
        "CREATE INDEX IF NOT EXISTS ix_kg_triple_mentions_kb_id     ON kg_triple_mentions (kb_id)",
    ]

    async with factory() as session:
        for stmt in ddl_statements:
            await session.execute(sa.text(stmt.strip()))
        await session.commit()

    print("[migrate_add_kg_tables] Done — 4 KG tables created (idempotent).")


if __name__ == "__main__":
    asyncio.run(migrate())
