"""AgentRun repository."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, distinct, select
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

    async def get_stats_by_agent(self, agent_id: uuid.UUID) -> dict:
        async with self._factory() as db:
            row = await db.execute(
                select(
                    func.count(AgentRun.id).label("total_runs"),
                    func.count(distinct(AgentRun.conversation_id)).label("total_conversations"),
                    func.avg(AgentRun.duration_ms).label("avg_duration_ms"),
                ).where(AgentRun.agent_id == agent_id)
            )
            r = row.one()
            return {
                "total_runs": r.total_runs or 0,
                "total_conversations": r.total_conversations or 0,
                "avg_run_duration_ms": round(r.avg_duration_ms) if r.avg_duration_ms else None,
            }

    async def get_tool_stats_by_agent(self, agent_id: uuid.UUID) -> list[dict]:
        """Aggregate tool call stats across all runs for an agent."""
        async with self._factory() as db:
            from sqlalchemy import text
            result = await db.execute(text("""
                SELECT
                    tc->>'name' AS tool_name,
                    COUNT(*) AS total,
                    SUM(CASE WHEN tc->>'status' = 'success' THEN 1 ELSE 0 END) AS success_count,
                    SUM(CASE WHEN tc->>'status' = 'error'   THEN 1 ELSE 0 END) AS error_count
                FROM agent_runs,
                     jsonb_array_elements(tool_calls) AS tc
                WHERE agent_id = :agent_id
                  AND jsonb_typeof(tool_calls) = 'array'
                  AND jsonb_array_length(tool_calls) > 0
                GROUP BY tc->>'name'
                ORDER BY total DESC
            """), {"agent_id": str(agent_id)})
            rows = result.fetchall()
        return [
            {
                "tool_name": r.tool_name,
                "total": r.total,
                "success": r.success_count,
                "error": r.error_count,
                "success_rate": round(r.success_count / r.total * 100) if r.total else 0,
            }
            for r in rows
        ]

    async def list_by_agent(self, agent_id: uuid.UUID, limit: int = 20) -> list[AgentRun]:
        async with self._factory() as db:
            result = await db.execute(
                select(AgentRun)
                .where(AgentRun.agent_id == agent_id)
                .order_by(AgentRun.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def list_by_conversation(self, conversation_id: uuid.UUID) -> list[AgentRun]:
        async with self._factory() as db:
            result = await db.execute(
                select(AgentRun)
                .where(AgentRun.conversation_id == conversation_id)
                .order_by(AgentRun.created_at.desc())
            )
            return list(result.scalars().all())
