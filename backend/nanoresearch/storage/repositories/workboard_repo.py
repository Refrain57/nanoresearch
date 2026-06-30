"""Workboard card repository (Phase 2 serial-MVP)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanoresearch.storage.models import WorkboardCard

# Legal card state transitions. Status CAS rejects anything not listed here.
_LEGAL_TRANSITIONS = {
    ("backlog", "todo"),
    ("todo", "ready"),
    ("ready", "running"),
    ("running", "done"),
    ("running", "blocked"),
    ("running", "ready"),   # release returns the card to the board
}


class WorkboardRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def create_card(
        self,
        *,
        conversation_id: uuid.UUID,
        title: str,
        spec: str | None = None,
        status: str = "backlog",
        target_agent_id: uuid.UUID | None = None,
        created_by_agent_id: uuid.UUID | None = None,
        depth: int = 0,
    ) -> WorkboardCard:
        card = WorkboardCard(
            conversation_id=conversation_id,
            title=title,
            spec=spec,
            status=status,
            target_agent_id=target_agent_id,
            created_by_agent_id=created_by_agent_id,
            depth=depth,
        )
        async with self._factory() as db:
            db.add(card)
            await db.commit()
            await db.refresh(card)
        return card

    async def get(self, card_id: uuid.UUID) -> WorkboardCard | None:
        async with self._factory() as db:
            return (await db.execute(
                select(WorkboardCard).where(WorkboardCard.id == card_id)
            )).scalar_one_or_none()

    async def list_by_conversation(
        self, conversation_id: uuid.UUID, statuses: set[str] | None = None
    ) -> list[WorkboardCard]:
        async with self._factory() as db:
            stmt = select(WorkboardCard).where(WorkboardCard.conversation_id == conversation_id)
            if statuses:
                stmt = stmt.where(WorkboardCard.status.in_(statuses))
            return list((await db.execute(stmt.order_by(WorkboardCard.created_at))).scalars().all())

    async def transition(
        self, card_id: uuid.UUID, *, expect_status: str, to_status: str, **fields
    ) -> bool:
        """Status CAS: move card expect_status → to_status iff the transition is legal AND the
        card's current status equals expect_status. Returns True on success, else False (illegal
        transition or status mismatch / lost race)."""
        if (expect_status, to_status) not in _LEGAL_TRANSITIONS:
            return False
        async with self._factory() as db:
            res = await db.execute(
                update(WorkboardCard)
                .where(WorkboardCard.id == card_id, WorkboardCard.status == expect_status)
                .values(status=to_status, updated_at=datetime.now(timezone.utc), **fields)
            )
            await db.commit()
            return res.rowcount == 1
