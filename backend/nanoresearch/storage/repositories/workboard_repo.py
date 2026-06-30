"""Workboard card repository (Phase 2 serial-MVP)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanoresearch.storage.models import WorkboardCard, WorkboardCardLink

# Cap a single card's spec so one oversized instruction can't blow the card-working context window.
WORKBOARD_MAX_SPEC_CHARS = 16_000

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
        if spec is not None and len(spec) > WORKBOARD_MAX_SPEC_CHARS:
            _marker = "\n... (spec truncated)"
            spec = spec[:WORKBOARD_MAX_SPEC_CHARS - len(_marker)] + _marker
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

    async def attach_result(self, card_id: uuid.UUID, result: str, artifacts: list, *, token: str) -> bool:
        """Token-gated: write the card's product (result + artifacts) iff *token* owns the claim.
        Status is left unchanged (the caller transitions running→done separately)."""
        async with self._factory() as db:
            res = await db.execute(
                update(WorkboardCard)
                .where(WorkboardCard.id == card_id, WorkboardCard.claim_token == token)
                .values(result=result, artifacts=artifacts, updated_at=datetime.now(timezone.utc))
            )
            await db.commit()
            return res.rowcount == 1

    async def touch_heartbeat(self, card_id: uuid.UUID) -> None:
        """Refresh heartbeat_at (claim liveness; the watchdog reaps cards whose lease lapsed)."""
        async with self._factory() as db:
            await db.execute(
                update(WorkboardCard).where(WorkboardCard.id == card_id)
                .values(heartbeat_at=datetime.now(timezone.utc))
            )
            await db.commit()

    # ---- Task 4: dependency links + promote ----

    async def link(self, parent_card_id: uuid.UUID, child_card_id: uuid.UUID) -> None:
        """Add a parent → child dependency (child waits in todo until all parents done)."""
        async with self._factory() as db:
            db.add(WorkboardCardLink(parent_card_id=parent_card_id, child_card_id=child_card_id))
            await db.commit()

    async def parents_all_done(self, child_card_id: uuid.UUID) -> bool:
        """True if the child has no parents, or every parent card is done."""
        async with self._factory() as db:
            parent_ids = (await db.execute(
                select(WorkboardCardLink.parent_card_id)
                .where(WorkboardCardLink.child_card_id == child_card_id)
            )).scalars().all()
            if not parent_ids:
                return True
            not_done = (await db.execute(
                select(func.count()).select_from(WorkboardCard)
                .where(WorkboardCard.id.in_(parent_ids), WorkboardCard.status != "done")
            )).scalar()
            return not_done == 0

    async def promote_ready_children(self, done_card_id: uuid.UUID) -> list[uuid.UUID]:
        """For each todo child of *done_card_id* whose parents are all done, transition it to
        ready. Returns the promoted child ids. Card-level analog of the Phase 1 join SCARD==0."""
        async with self._factory() as db:
            child_ids = (await db.execute(
                select(WorkboardCardLink.child_card_id)
                .where(WorkboardCardLink.parent_card_id == done_card_id)
            )).scalars().all()
        promoted: list[uuid.UUID] = []
        for cid in child_ids:
            if await self.parents_all_done(cid):
                if await self.transition(cid, expect_status="todo", to_status="ready"):
                    promoted.append(cid)
        return promoted
