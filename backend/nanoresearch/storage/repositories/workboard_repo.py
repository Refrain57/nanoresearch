"""Workboard card repository (Phase 2 serial-MVP)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanoresearch.storage.models import WorkboardCard, WorkboardCardLink

# Cap a single card's spec so one oversized instruction can't blow the card-working context window.
WORKBOARD_MAX_SPEC_CHARS = 16_000

# Termination guarantees (serial-MVP): bound the collaboration graph so a successor-creating loop
# can't expand forever → the board is guaranteed to quiesce.
MAX_CARDS_PER_ROUND = 50
MAX_SUCCESSOR_DEPTH = 8

# Legal card state transitions. Status CAS rejects anything not listed here.
_LEGAL_TRANSITIONS = {
    ("backlog", "todo"),
    ("todo", "ready"),
    ("ready", "running"),
    ("ready", "blocked"),   # no member agent accepted the card (reroute exhausted) — terminate
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

    # ---- Task 8: termination caps + watchdog ----

    async def can_create_successor(self, conversation_id: uuid.UUID, parent_depth: int) -> bool:
        """Cost cap: reject a new successor card if the round已达卡片上限 or the successor depth would
        exceed MAX_SUCCESSOR_DEPTH. Guarantees the board quiesces (no infinite successor loop)."""
        if parent_depth + 1 > MAX_SUCCESSOR_DEPTH:
            return False
        async with self._factory() as db:
            count = (await db.execute(
                select(func.count()).select_from(WorkboardCard)
                .where(WorkboardCard.conversation_id == conversation_id)
            )).scalar()
        return count < MAX_CARDS_PER_ROUND

    async def list_running_cards(self) -> list[WorkboardCard]:
        """All running cards across conversations (watchdog stale-card scan)."""
        async with self._factory() as db:
            return list((await db.execute(
                select(WorkboardCard).where(WorkboardCard.status == "running")
            )).scalars().all())

    # ---- Task 7: collector quiescence ----

    async def list_done_cards_for_collection(self, conversation_id: uuid.UUID) -> list[WorkboardCard]:
        async with self._factory() as db:
            return list((await db.execute(
                select(WorkboardCard).where(
                    WorkboardCard.conversation_id == conversation_id,
                    WorkboardCard.status == "done",
                    WorkboardCard.collected.is_(False),
                ).order_by(WorkboardCard.created_at)
            )).scalars().all())

    async def mark_collected(self, conversation_id: uuid.UUID) -> None:
        async with self._factory() as db:
            await db.execute(
                update(WorkboardCard).where(
                    WorkboardCard.conversation_id == conversation_id,
                    WorkboardCard.status == "done",
                    WorkboardCard.collected.is_(False),
                ).values(collected=True, updated_at=datetime.now(timezone.utc))
            )
            await db.commit()

    async def is_board_quiesced(self, conversation_id: uuid.UUID) -> bool:
        """True iff there is ≥1 uncollected done card, AND no ready/running card, AND no promotable
        todo card (a todo whose parents are all done). Serial-MVP: WIP=1 → this read is settled."""
        async with self._factory() as db:
            uncollected_done = (await db.execute(
                select(func.count()).select_from(WorkboardCard).where(
                    WorkboardCard.conversation_id == conversation_id,
                    WorkboardCard.status == "done",
                    WorkboardCard.collected.is_(False),
                )
            )).scalar()
            if not uncollected_done:
                return False
            active = (await db.execute(
                select(func.count()).select_from(WorkboardCard).where(
                    WorkboardCard.conversation_id == conversation_id,
                    WorkboardCard.status.in_(("ready", "running")),
                )
            )).scalar()
            if active:
                return False
            todo_ids = (await db.execute(
                select(WorkboardCard.id).where(
                    WorkboardCard.conversation_id == conversation_id,
                    WorkboardCard.status == "todo",
                )
            )).scalars().all()
        for tid in todo_ids:
            if await self.parents_all_done(tid):
                return False  # a todo can still be promoted → board not quiesced
        return True

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

    # ---- Task 1 (collab layer): pass tracking ----

    async def set_target(self, card_id: uuid.UUID, agent_id: uuid.UUID) -> None:
        """Set target_agent_id on a card (reroute without changing status)."""
        async with self._factory() as db:
            await db.execute(
                update(WorkboardCard).where(WorkboardCard.id == card_id)
                .values(target_agent_id=agent_id, updated_at=datetime.now(timezone.utc))
            )
            await db.commit()

    async def record_pass(self, card_id: uuid.UUID, passed_agent_id: str) -> int:
        """Increment pass_count and append {"passed": passed_agent_id} to artifacts.
        Returns the new pass_count."""
        async with self._factory() as db:
            card = (await db.execute(
                select(WorkboardCard).where(WorkboardCard.id == card_id)
            )).scalar_one()
            card.pass_count = (card.pass_count or 0) + 1
            card.artifacts = list(card.artifacts or []) + [{"passed": passed_agent_id}]
            card.updated_at = datetime.now(timezone.utc)
            await db.commit()
            return card.pass_count

    async def reset_pass(self, card_id: uuid.UUID) -> None:
        """Reset pass_count to 0 (optional; used by collector after merging pass results)."""
        async with self._factory() as db:
            await db.execute(
                update(WorkboardCard).where(WorkboardCard.id == card_id)
                .values(pass_count=0, updated_at=datetime.now(timezone.utc))
            )
            await db.commit()

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
