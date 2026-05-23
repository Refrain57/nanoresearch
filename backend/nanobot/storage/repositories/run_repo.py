"""AgentRun repository."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanobot.storage.models import AgentRun


class RunRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def create(
        self,
        conversation_id: uuid.UUID,
        uid: str,
        agent_id: uuid.UUID | None = None,
    ) -> AgentRun:
        run = AgentRun(
            conversation_id=conversation_id,
            uid=uid,
            agent_id=agent_id,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        async with self._factory() as db:
            db.add(run)
            await db.commit()
            await db.refresh(run)
        return run

    async def get(self, run_id: uuid.UUID) -> AgentRun | None:
        async with self._factory() as db:
            result = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
            return result.scalar_one_or_none()

    async def update(self, run_id: uuid.UUID, **fields) -> None:
        async with self._factory() as db:
            result = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
            run = result.scalar_one_or_none()
            if run:
                for key, value in fields.items():
                    setattr(run, key, value)
                await db.commit()

    async def list_by_conversation(self, conversation_id: uuid.UUID) -> list[AgentRun]:
        async with self._factory() as db:
            result = await db.execute(
                select(AgentRun)
                .where(AgentRun.conversation_id == conversation_id)
                .order_by(AgentRun.created_at.desc())
            )
            return list(result.scalars().all())
