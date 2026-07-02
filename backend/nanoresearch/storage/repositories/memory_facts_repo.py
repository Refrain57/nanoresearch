from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanoresearch.agent.memory_facts import Fact
from nanoresearch.storage.models import MemoryFact


def _to_fact(row: MemoryFact) -> Fact:
    return Fact(
        id=str(row.id), uid=row.uid, section=row.section, text=row.text,
        source=row.source, derived_from=list(row.derived_from or []),
        confidence=row.confidence, edited_by=row.edited_by,
        edited_at=row.edited_at.isoformat() if row.edited_at else None,
        active=row.active,
    )


class MemoryFactsRepository:
    """Persistence for the 画像 fact store. All reads are uid-scoped and active-only."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def list_active(self, uid: str) -> list[Fact]:
        async with self._factory() as db:
            res = await db.execute(
                select(MemoryFact)
                .where(MemoryFact.uid == uid, MemoryFact.active.is_(True))
                .order_by(MemoryFact.created_at)
            )
            return [_to_fact(r) for r in res.scalars().all()]

    async def insert_extracted(self, uid, section, text, derived_from=None, confidence=None) -> Fact:
        async with self._factory() as db:
            row = MemoryFact(uid=uid, section=section, text=text, source="extracted",
                             derived_from=derived_from or [], confidence=confidence)
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return _to_fact(row)

    async def insert_manual(self, uid, section, text, edited_by) -> Fact:
        async with self._factory() as db:
            row = MemoryFact(uid=uid, section=section, text=text, source="manual",
                             edited_by=edited_by, edited_at=datetime.now(timezone.utc))
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return _to_fact(row)

    async def deactivate(self, fact_id: str) -> None:
        async with self._factory() as db:
            await db.execute(
                sa_update(MemoryFact).where(MemoryFact.id == uuid.UUID(fact_id)).values(active=False)
            )
            await db.commit()
