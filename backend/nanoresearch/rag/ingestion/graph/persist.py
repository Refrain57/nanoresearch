"""Persist entity/relation extraction results from chunks into KG tables."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from nanoresearch.storage.repositories.graph_repo import _entity_id, _normalize, _triple_id

from sqlalchemy.ext.asyncio import async_sessionmaker

if TYPE_CHECKING:
    from nanoresearch.storage.models import KbChunk

logger = logging.getLogger(__name__)


async def persist_chunk_entities(
    chunk_rows: list[KbChunk],
    session_factory: async_sessionmaker,
) -> None:
    """Read _kg_entities/_kg_relations from chunk metadata, write to KG tables.

    Designed to be called non-blocking after create_chunks(). Errors are logged
    but never raised — KG failure must not block document ingestion.
    """
    from nanoresearch.storage.repositories.graph_repo import GraphRepository

    if not chunk_rows:
        return

    # Group chunks by KB to minimize repo calls
    kb_chunks: dict[uuid.UUID, list[KbChunk]] = {}
    for chunk in chunk_rows:
        kb_chunks.setdefault(chunk.kb_id, []).append(chunk)

    repo = GraphRepository(session_factory)

    for kb_id, chunks in kb_chunks.items():
        for chunk in chunks:
            meta = chunk.chunk_metadata or {}
            raw_entities: list[dict] = meta.get("_kg_entities", [])
            raw_relations: list[dict] = meta.get("_kg_relations", [])

            if not raw_entities and not raw_relations:
                continue

            try:
                # Collect all entity dicts: nodes + relation endpoints
                all_entity_dicts: list[dict] = list(raw_entities)
                for rel in raw_relations:
                    src = rel.get("source")
                    tgt = rel.get("target")
                    if src:
                        all_entity_dicts.append(src)
                    if tgt:
                        all_entity_dicts.append(tgt)

                await repo.upsert_entities(kb_id, all_entity_dicts)

                # Deterministic ID computation — no dependency on RETURNING.
                # _entity_id() is the same function used inside upsert_entities,
                # so IDs are guaranteed consistent regardless of concurrent inserts.
                entity_map: dict[str, uuid.UUID] = {}
                for e_dict in all_entity_dicts:
                    name = _normalize(e_dict.get("text") or e_dict.get("name", ""))
                    label = e_dict.get("label", "Entity")
                    if name:
                        entity_map[name] = _entity_id(kb_id, name, label)

                entity_ids = list(set(entity_map.values()))

                await repo.upsert_triples(kb_id, raw_relations, entity_map)

                triple_ids: list[uuid.UUID] = []
                for rel in raw_relations:
                    src_name = _normalize((rel.get("source") or {}).get("text", ""))
                    tgt_name = _normalize((rel.get("target") or {}).get("text", ""))
                    label = rel.get("label", "RELATED_TO")
                    src_id = entity_map.get(src_name)
                    tgt_id = entity_map.get(tgt_name)
                    if src_id and tgt_id:
                        triple_ids.append(_triple_id(kb_id, src_id, tgt_id, label))

                if entity_ids or triple_ids:
                    await repo.upsert_mentions(chunk.id, kb_id, entity_ids, triple_ids)

            except Exception as exc:
                logger.warning(
                    f"[KG PERSIST] Failed for chunk {chunk.id} in kb {kb_id}: {exc}"
                )
