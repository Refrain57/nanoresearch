"""Knowledge Graph repository — entities, triples, mentions."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict

from sqlalchemy import distinct, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import aliased

from nanoresearch.storage.models import KgEntity, KgEntityMention, KgTriple, KgTripleMention


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

    async def list_entities(
        self, kb_id: uuid.UUID, search: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """Paginated/searchable entity list with mention counts (desc)."""
        conds = [KgEntity.kb_id == kb_id]
        if search:
            conds.append(KgEntity.name.ilike(f"%{_normalize(search)}%"))
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntity.name, KgEntity.label, func.count(KgEntityMention.id).label("mentions"))
                .join(KgEntityMention, KgEntityMention.entity_id == KgEntity.id)
                .where(*conds)
                .group_by(KgEntity.name, KgEntity.label)
                .order_by(text("mentions DESC"))
                .limit(limit)
                .offset(offset)
            )
            return [{"name": r[0], "label": r[1], "mentions": r[2]} for r in result.all()]

    async def get_entity_summary(self, kb_id: uuid.UUID, name: str) -> dict | None:
        """Header info for one entity (by normalized name), or None if absent."""
        norm = _normalize(name)
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntity.name, KgEntity.label, func.count(KgEntityMention.id).label("mentions"))
                .join(KgEntityMention, KgEntityMention.entity_id == KgEntity.id, isouter=True)
                .where(KgEntity.kb_id == kb_id, KgEntity.name == norm)
                .group_by(KgEntity.name, KgEntity.label)
            )
            row = result.first()
            if not row:
                return None
            return {"name": row[0], "label": row[1], "mention_count": row[2]}

    async def get_entity_facts(self, kb_id: uuid.UUID, name: str) -> list[dict]:
        """Triples where the entity is source OR target, with distinct-document corroboration."""
        from nanoresearch.storage.models import KbChunk
        norm = _normalize(name)
        SrcE = aliased(KgEntity)
        TgtE = aliased(KgEntity)
        async with self._factory() as db:
            ids_res = await db.execute(
                select(KgEntity.id).where(KgEntity.kb_id == kb_id, KgEntity.name == norm)
            )
            entity_ids = [r[0] for r in ids_res.all()]
            if not entity_ids:
                return []
            doc_count_sq = (
                select(
                    KgTripleMention.triple_id.label("tid"),
                    func.count(distinct(KbChunk.document_id)).label("doc_count"),
                )
                .join(KbChunk, KbChunk.id == KgTripleMention.chunk_id)
                .where(KgTripleMention.kb_id == kb_id)
                .group_by(KgTripleMention.triple_id)
                .subquery()
            )
            result = await db.execute(
                select(
                    KgTriple.id, SrcE.name, KgTriple.label, TgtE.name,
                    func.coalesce(doc_count_sq.c.doc_count, 0).label("doc_count"),
                )
                .join(SrcE, SrcE.id == KgTriple.source_id)
                .join(TgtE, TgtE.id == KgTriple.target_id)
                .outerjoin(doc_count_sq, doc_count_sq.c.tid == KgTriple.id)
                .where(
                    KgTriple.kb_id == kb_id,
                    or_(KgTriple.source_id.in_(entity_ids), KgTriple.target_id.in_(entity_ids)),
                )
                .order_by(text("doc_count DESC"))
            )
            return [
                {"triple_id": str(r[0]), "source": r[1], "label": r[2], "target": r[3], "doc_count": r[4]}
                for r in result.all()
            ]

    async def get_chunks_by_triple(self, kb_id: uuid.UUID, triple_id: uuid.UUID) -> list[KbChunk]:
        """Evidence chunks for a fact (triple), via triple mentions."""
        from nanoresearch.storage.models import KbChunk
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk)
                .join(KgTripleMention, KgTripleMention.chunk_id == KbChunk.id)
                .where(KgTripleMention.triple_id == triple_id, KgTripleMention.kb_id == kb_id)
                .distinct()
            )
            return list(result.scalars().all())

    async def get_entities_by_doc(self, doc_id: uuid.UUID) -> list[dict]:
        """Return entities (with mention counts) for chunks belonging to a document."""
        from nanoresearch.storage.models import KbChunk
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

    async def get_entity_evidence(self, kb_id: uuid.UUID, name: str, limit: int = 20) -> list[dict]:
        """Evidence chunks for an entity: content + original filename, for article generation."""
        from nanoresearch.storage.models import KbChunk, KbDocument
        norm = _normalize(name)
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk.id, KbChunk.content, KbChunk.chunk_metadata, KbDocument.filename)
                .join(KgEntityMention, KgEntityMention.chunk_id == KbChunk.id)
                .join(KgEntity, KgEntity.id == KgEntityMention.entity_id)
                .join(KbDocument, KbDocument.id == KbChunk.document_id)
                .where(KgEntity.kb_id == kb_id, KgEntity.name == norm)
                .distinct()
                .order_by(KbChunk.id)
                .limit(limit)
            )
            out = []
            for r in result.all():
                out.append({
                    "chunk_id": str(r[0]),
                    "content": r[1] or "",
                    "page": (r[2] or {}).get("page"),
                    "source": r[3] or "",
                })
            return out

    async def get_article(self, kb_id: uuid.UUID, entity_name: str):
        from nanoresearch.storage.models import KgEntityArticle
        norm = _normalize(entity_name)
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntityArticle).where(
                    KgEntityArticle.kb_id == kb_id, KgEntityArticle.entity_name == norm
                )
            )
            return result.scalar_one_or_none()

    async def upsert_article(self, kb_id: uuid.UUID, entity_name: str, markdown: str,
                             citations: list, evidence_hash: str, model: str | None):
        from nanoresearch.storage.models import KgEntityArticle
        norm = _normalize(entity_name)
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntityArticle).where(
                    KgEntityArticle.kb_id == kb_id, KgEntityArticle.entity_name == norm
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                row = KgEntityArticle(kb_id=kb_id, entity_name=norm)
                db.add(row)
            row.markdown = markdown
            row.citations = citations
            row.evidence_hash = evidence_hash
            row.model = model
            from datetime import datetime, timezone
            row.generated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(row)
            return row
