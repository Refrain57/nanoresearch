"""Knowledge Graph repository — entities, triples, mentions."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanobot.storage.models import KgEntity, KgEntityMention, KgTriple, KgTripleMention


def _normalize(name: str) -> str:
    """Normalize entity name for dedup: lowercase, collapse spaces, strip parens."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"（[^）]*）|\([^)]*\)", "", name).strip()
    return name


def _entity_id(kb_id: uuid.UUID, name: str, label: str) -> uuid.UUID:
    key = f"{kb_id}:{_normalize(name)}:{label.lower()}"
    digest = hashlib.sha256(key.encode()).digest()[:16]
    return uuid.UUID(bytes=digest)


def _triple_id(kb_id: uuid.UUID, src_id: uuid.UUID, tgt_id: uuid.UUID, label: str) -> uuid.UUID:
    key = f"{kb_id}:{src_id}:{tgt_id}:{label.lower()}"
    digest = hashlib.sha256(key.encode()).digest()[:16]
    return uuid.UUID(bytes=digest)


class GraphRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    async def upsert_entities(
        self, kb_id: uuid.UUID, entities: list[dict]
    ) -> list[KgEntity]:
        """Upsert entities by deterministic ID. Returns persisted rows."""
        if not entities:
            return []
        rows = []
        seen: set[uuid.UUID] = set()
        for e in entities:
            name = e.get("text") or e.get("name", "")
            label = e.get("label", "Entity")
            if not name:
                continue
            eid = _entity_id(kb_id, name, label)
            if eid in seen:
                continue
            seen.add(eid)
            rows.append({
                "id": eid,
                "kb_id": kb_id,
                "name": _normalize(name),
                "label": label.lower(),
                "attributes": e.get("attributes", {}),
            })
        if not rows:
            return []
        stmt = (
            pg_insert(KgEntity)
            .values(rows)
            .on_conflict_do_nothing()
            .returning(KgEntity)
        )
        async with self._factory() as db:
            result = await db.execute(stmt)
            await db.commit()
            inserted = list(result.scalars().all())

        # For rows that conflicted (not returned), fetch them
        if len(inserted) < len(rows):
            inserted_ids = {r.id for r in inserted}
            missing_ids = [r["id"] for r in rows if r["id"] not in inserted_ids]
            if missing_ids:
                async with self._factory() as db:
                    res = await db.execute(select(KgEntity).where(KgEntity.id.in_(missing_ids)))
                    inserted.extend(res.scalars().all())
        return inserted

    async def upsert_triples(
        self, kb_id: uuid.UUID, triples: list[dict], entity_map: dict[str, uuid.UUID]
    ) -> list[KgTriple]:
        """Upsert triples. entity_map: normalized_name -> entity_id."""
        if not triples:
            return []
        rows = []
        seen: set[uuid.UUID] = set()
        for t in triples:
            src_name = _normalize((t.get("source") or {}).get("text", ""))
            tgt_name = _normalize((t.get("target") or {}).get("text", ""))
            label = t.get("label", "RELATED_TO")
            src_id = entity_map.get(src_name)
            tgt_id = entity_map.get(tgt_name)
            if not src_id or not tgt_id:
                continue
            tid = _triple_id(kb_id, src_id, tgt_id, label)
            if tid in seen:
                continue
            seen.add(tid)
            rows.append({"id": tid, "kb_id": kb_id, "source_id": src_id, "target_id": tgt_id, "label": label.lower()})
        if not rows:
            return []
        stmt = (
            pg_insert(KgTriple)
            .values(rows)
            .on_conflict_do_nothing()
            .returning(KgTriple)
        )
        async with self._factory() as db:
            result = await db.execute(stmt)
            await db.commit()
            return list(result.scalars().all())

    async def upsert_mentions(
        self,
        chunk_id: uuid.UUID,
        kb_id: uuid.UUID,
        entity_ids: list[uuid.UUID],
        triple_ids: list[uuid.UUID],
    ) -> None:
        """Insert entity/triple mentions for a chunk, ignore duplicates."""
        em_rows = [{"entity_id": eid, "chunk_id": chunk_id, "kb_id": kb_id} for eid in entity_ids]
        tm_rows = [{"triple_id": tid, "chunk_id": chunk_id, "kb_id": kb_id} for tid in triple_ids]
        async with self._factory() as db:
            if em_rows:
                await db.execute(
                    pg_insert(KgEntityMention).values(em_rows).on_conflict_do_nothing()
                )
            if tm_rows:
                await db.execute(
                    pg_insert(KgTripleMention).values(tm_rows).on_conflict_do_nothing()
                )
            await db.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get_chunks_by_entity_name(
        self, kb_id: uuid.UUID, name: str
    ) -> list[uuid.UUID]:
        """Return chunk IDs that mention the given entity name."""
        norm = _normalize(name)
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntityMention.chunk_id)
                .join(KgEntity, KgEntityMention.entity_id == KgEntity.id)
                .where(KgEntity.kb_id == kb_id, KgEntity.name == norm)
                .distinct()
            )
            return [r[0] for r in result.all()]

    async def get_entity_names_for_chunks(
        self, chunk_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, list[str]]:
        """Return {chunk_id: [entity_names]} for the given chunks."""
        if not chunk_ids:
            return {}
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntityMention.chunk_id, KgEntity.name)
                .join(KgEntity, KgEntityMention.entity_id == KgEntity.id)
                .where(KgEntityMention.chunk_id.in_(chunk_ids))
            )
            mapping: dict[uuid.UUID, list[str]] = defaultdict(list)
            for chunk_id, name in result.all():
                mapping[chunk_id].append(name)
            return dict(mapping)

    async def get_neighbor_chunks_via_entities(
        self,
        seed_chunk_ids: list[uuid.UUID],
        kb_id: uuid.UUID,
        top_k: int = 5,
    ) -> list[tuple[uuid.UUID, str]]:
        """Find chunks that share entities with seed chunks (cross-document expansion)."""
        if not seed_chunk_ids:
            return []
        async with self._factory() as db:
            # Find entities mentioned in seed chunks
            seed_entity_q = (
                select(KgEntityMention.entity_id)
                .where(KgEntityMention.chunk_id.in_(seed_chunk_ids))
                .distinct()
            )
            # Find other chunks that mention those entities, excluding seeds
            result = await db.execute(
                select(KgEntityMention.chunk_id, KgEntity.name)
                .join(KgEntity, KgEntityMention.entity_id == KgEntity.id)
                .where(
                    KgEntityMention.entity_id.in_(seed_entity_q),
                    KgEntityMention.kb_id == kb_id,
                    KgEntityMention.chunk_id.notin_(seed_chunk_ids),
                )
                .distinct()
                .limit(top_k)
            )
            return [(row[0], row[1]) for row in result.all()]

    async def get_stats(self, kb_id: uuid.UUID) -> dict:
        """Return entity count, edge count, and top-20 most-mentioned entities."""
        async with self._factory() as db:
            entity_count = (await db.execute(
                select(func.count()).where(KgEntity.kb_id == kb_id)
                .select_from(KgEntity)
            )).scalar_one()

            triple_count = (await db.execute(
                select(func.count()).where(KgTriple.kb_id == kb_id)
                .select_from(KgTriple)
            )).scalar_one()

            top_entities_q = await db.execute(
                select(KgEntity.name, func.count(KgEntityMention.id).label("mentions"))
                .join(KgEntityMention, KgEntityMention.entity_id == KgEntity.id)
                .where(KgEntity.kb_id == kb_id)
                .group_by(KgEntity.name)
                .order_by(text("mentions DESC"))
                .limit(20)
            )
            top_entities = [{"name": r[0], "mentions": r[1]} for r in top_entities_q.all()]

        return {
            "entity_count": entity_count,
            "triple_count": triple_count,
            "top_entities": top_entities,
        }

    async def get_entities_by_doc(self, doc_id: uuid.UUID) -> list[dict]:
        """Return entities (with mention counts) for chunks belonging to a document."""
        from nanobot.storage.models import KbChunk
        async with self._factory() as db:
            result = await db.execute(
                select(
                    KgEntity.name,
                    KgEntity.label,
                    func.count(KgEntityMention.id).label("mentions"),
                )
                .join(KgEntityMention, KgEntityMention.entity_id == KgEntity.id)
                .join(KbChunk, KbChunk.id == KgEntityMention.chunk_id)
                .where(KbChunk.document_id == doc_id)
                .group_by(KgEntity.name, KgEntity.label)
                .order_by(text("mentions DESC"))
                .limit(100)
            )
            return [{"name": r[0], "label": r[1], "mentions": r[2]} for r in result.all()]

    async def delete_by_doc(self, doc_id: uuid.UUID) -> None:
        """Delete entity mentions for all chunks of a document."""
        async with self._factory() as db:
            await db.execute(
                text("""
                    DELETE FROM kg_entity_mentions
                    WHERE chunk_id IN (SELECT id FROM kb_chunks WHERE document_id = :doc_id)
                """),
                {"doc_id": doc_id},
            )
            await db.execute(
                text("""
                    DELETE FROM kg_triple_mentions
                    WHERE chunk_id IN (SELECT id FROM kb_chunks WHERE document_id = :doc_id)
                """),
                {"doc_id": doc_id},
            )
            await db.commit()

    async def delete_by_kb(self, kb_id: uuid.UUID) -> None:
        """Delete all graph data for a KB (cascade handles mentions)."""
        async with self._factory() as db:
            await db.execute(
                text("DELETE FROM kg_entities WHERE kb_id = :kb_id"), {"kb_id": kb_id}
            )
            await db.commit()
