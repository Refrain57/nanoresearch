"""Agent repository."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanobot.storage.models import Agent


class AgentRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def get_default(self) -> Agent | None:
        async with self._factory() as db:
            result = await db.execute(select(Agent).where(Agent.is_default == True))  # noqa: E712
            return result.scalar_one_or_none()

    async def get_by_id(self, agent_id: uuid.UUID) -> Agent | None:
        async with self._factory() as db:
            result = await db.execute(select(Agent).where(Agent.id == agent_id))
            return result.scalar_one_or_none()

    async def list_all(self) -> list[Agent]:
        async with self._factory() as db:
            result = await db.execute(select(Agent).order_by(Agent.created_at))
            return list(result.scalars().all())

    async def list_by_user(self, uid: str) -> list[Agent]:
        from sqlalchemy import or_
        async with self._factory() as db:
            result = await db.execute(
                select(Agent)
                .where(or_(Agent.created_by == uid, Agent.is_default == True))  # noqa: E712
                .order_by(Agent.created_at)
            )
            return list(result.scalars().all())

    async def create(self, data: dict) -> Agent:
        agent = Agent(**data)
        async with self._factory() as db:
            db.add(agent)
            await db.commit()
            await db.refresh(agent)
        return agent

    async def update(self, agent_id: uuid.UUID, **fields) -> Agent | None:
        async with self._factory() as db:
            result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = result.scalar_one_or_none()
            if agent:
                for key, value in fields.items():
                    setattr(agent, key, value)
                await db.commit()
                await db.refresh(agent)
            return agent

    async def delete(self, agent_id: uuid.UUID) -> bool:
        async with self._factory() as db:
            result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = result.scalar_one_or_none()
            if agent is None:
                return False
            await db.delete(agent)
            await db.commit()
            return True

    async def default_exists(self) -> bool:
        async with self._factory() as db:
            result = await db.execute(select(Agent.id).where(Agent.is_default == True))  # noqa: E712
            return result.scalar_one_or_none() is not None
