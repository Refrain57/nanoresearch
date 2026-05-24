"""Agent card endpoints — list, detail, update."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from nanobot.server.middleware.auth import get_current_user
from nanobot.storage.repositories.agent_repo import AgentRepository
from nanobot.storage.repositories.run_repo import RunRepository

router = APIRouter()


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    capabilities: dict | None = None
    skills_config: list | None = None
    tools_config: list | None = None
    system_prompt: str | None = None
    max_iterations: int | None = None
    default_model: str | None = None
    provider: str | None = None


async def _agent_to_card(agent, stats: dict) -> dict:
    return {
        "id": str(agent.id),
        "name": agent.name,
        "description": agent.description,
        "version": agent.version,
        "capabilities": agent.capabilities or {},
        "skills": agent.skills_config or [],
        "tools": agent.tools_config or [],
        "model": agent.default_model,
        "provider": agent.provider,
        "is_default": agent.is_default,
        "stats": stats,
        "created_at": agent.created_at.isoformat() if agent.created_at else None,
        "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
    }


@router.get("/api/agents")
async def list_agents(
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agents = await AgentRepository(factory).list_all()
    run_repo = RunRepository(factory)
    result = []
    for agent in agents:
        stats = await run_repo.get_stats_by_agent(agent.id)
        result.append(await _agent_to_card(agent, stats))
    return result


@router.get("/api/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    stats = await RunRepository(factory).get_stats_by_agent(agent.id)
    return await _agent_to_card(agent, stats)


@router.put("/api/agents/{agent_id}")
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    updated = await AgentRepository(factory).update(agent.id, **fields)
    stats = await RunRepository(factory).get_stats_by_agent(updated.id)
    return await _agent_to_card(updated, stats)


async def _get_agent_or_404(agent_id: str, factory):
    try:
        aid = uuid.UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    agent = await AgentRepository(factory).get_by_id(aid)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    return agent
