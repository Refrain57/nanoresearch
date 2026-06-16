"""Agent repository."""

from __future__ import annotations

import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from loguru import logger

from nanobot.storage.models import Agent, AgentKnowledgeBinding, KnowledgeBase


def _agent_to_hash(agent: Agent) -> dict:
    return {
        "id": str(agent.id),
        "name": agent.name or "",
        "description": agent.description or "",
        "persona": agent.persona or "",
        "default_model": agent.default_model or "",
        "is_default": "true" if agent.is_default else "false",
        "skills_config": json.dumps(agent.skills_config or [], ensure_ascii=False),
        "tools_config": json.dumps(agent.tools_config or [], ensure_ascii=False),
        "harness": json.dumps(agent.harness or {}, ensure_ascii=False),
        "capabilities": json.dumps(agent.capabilities or {}, ensure_ascii=False),
        "max_iterations": str(agent.max_iterations or 40),
        "version": agent.version or "1.0.0",
        "provider": agent.provider or "",
        "created_by": agent.created_by or "",
    }


def _agent_from_hash(h: dict) -> Agent:
    agent = Agent()
    agent.id = uuid.UUID(h["id"])
    agent.name = h.get("name") or ""
    agent.description = h.get("description") or None
    agent.persona = h.get("persona") or None
    agent.default_model = h.get("default_model") or None
    agent.is_default = h.get("is_default") == "true"
    agent.skills_config = json.loads(h.get("skills_config") or "[]")
    agent.tools_config = json.loads(h.get("tools_config") or "[]")
    agent.harness = json.loads(h.get("harness") or "{}")
    agent.capabilities = json.loads(h.get("capabilities") or "{}")
    agent.max_iterations = int(h.get("max_iterations") or 40)
    agent.version = h.get("version") or "1.0.0"
    agent.provider = h.get("provider") or None
    agent.created_by = h.get("created_by") or None
    return agent


class AgentRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def get_default(self) -> Agent | None:
        async with self._factory() as db:
            result = await db.execute(select(Agent).where(Agent.is_default == True))  # noqa: E712
            return result.scalar_one_or_none()

    async def get_by_id(self, agent_id: uuid.UUID) -> Agent | None:
        from nanobot.bus.redis_client import get_redis
        from nanobot.bus.redis_keys import RedisKeys
        cache_key = RedisKeys.agent(str(agent_id))
        try:
            cached = await get_redis().hgetall(cache_key)
            if cached:
                logger.bind(event="agent_cache_hit", cache_layer="agent_cache").debug(
                    "agent cache hit for {}", agent_id
                )
                return _agent_from_hash(cached)
        except Exception:
            pass

        logger.bind(event="agent_cache_miss", cache_layer="agent_cache").debug(
            "agent cache miss for {}", agent_id
        )
        async with self._factory() as db:
            result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = result.scalar_one_or_none()

        if agent is not None:
            try:
                r = get_redis()
                await r.hset(cache_key, mapping=_agent_to_hash(agent))
                await r.expire(cache_key, RedisKeys.AGENT_TTL)
            except Exception:
                pass
        return agent

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
        if agent is not None:
            try:
                from nanobot.bus.redis_client import get_redis
                from nanobot.bus.redis_keys import RedisKeys
                await get_redis().delete(RedisKeys.agent(str(agent_id)))
            except Exception:
                pass
        return agent

    async def delete(self, agent_id: uuid.UUID) -> bool:
        async with self._factory() as db:
            result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = result.scalar_one_or_none()
            if agent is None:
                return False
            await db.delete(agent)
            await db.commit()
        try:
            from nanobot.bus.redis_client import get_redis
            from nanobot.bus.redis_keys import RedisKeys
            await get_redis().delete(RedisKeys.agent(str(agent_id)))
        except Exception:
            pass
        return True

    async def default_exists(self) -> bool:
        async with self._factory() as db:
            result = await db.execute(select(Agent.id).where(Agent.is_default == True))  # noqa: E712
            return result.scalar_one_or_none() is not None

    # ---- Knowledge Base bindings ----

    async def list_bound_kbs(self, agent_id: uuid.UUID) -> list[KnowledgeBase]:
        """List knowledge bases bound to an agent."""
        from sqlalchemy import select as _select
        async with self._factory() as db:
            result = await db.execute(
                _select(KnowledgeBase)
                .join(AgentKnowledgeBinding, AgentKnowledgeBinding.kb_id == KnowledgeBase.id)
                .where(AgentKnowledgeBinding.agent_id == agent_id)
            )
            return list(result.scalars().all())

    async def bind_kb(self, agent_id: uuid.UUID, kb_id: uuid.UUID) -> None:
        """Bind a knowledge base to an agent."""
        async with self._factory() as db:
            db.add(AgentKnowledgeBinding(agent_id=agent_id, kb_id=kb_id))
            await db.commit()

    async def unbind_kb(self, agent_id: uuid.UUID, kb_id: uuid.UUID) -> bool:
        """Unbind a knowledge base from an agent."""
        async with self._factory() as db:
            result = await db.execute(
                select(AgentKnowledgeBinding).where(
                    AgentKnowledgeBinding.agent_id == agent_id,
                    AgentKnowledgeBinding.kb_id == kb_id,
                )
            )
            binding = result.scalar_one_or_none()
            if binding is None:
                return False
            await db.delete(binding)
            await db.commit()
            return True
