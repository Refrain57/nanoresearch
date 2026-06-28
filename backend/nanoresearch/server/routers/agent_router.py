"""Agent card endpoints — list, detail, update."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.storage.repositories.agent_repo import AgentRepository
from nanoresearch.storage.repositories.knowledge_repo import KnowledgeRepository
from nanoresearch.storage.repositories.run_repo import RunRepository

router = APIRouter()


class AgentCreate(BaseModel):
    name: str
    description: str = ""
    default_model: str | None = None
    provider: str | None = None
    max_iterations: int = 40
    capabilities: dict = {}


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    capabilities: dict | None = None
    skills_config: list | None = None
    tools_config: list | None = None
    persona: str | None = None
    harness: dict | None = None
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
        "max_iterations": agent.max_iterations,
        "persona": agent.persona or "",
        "harness": agent.harness or {},
        "stats": stats,
        "created_at": agent.created_at.isoformat() if agent.created_at else None,
        "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
    }


@router.get("/api/skills")
async def list_skills(_uid: str = Depends(get_current_user)):
    from nanoresearch.agent.skills import BUILTIN_SKILLS_DIR, SkillsLoader
    loader = SkillsLoader(
        workspace=BUILTIN_SKILLS_DIR.parent,
        builtin_skills_dir=BUILTIN_SKILLS_DIR,
    )
    result = []
    for s in loader.list_skills(filter_unavailable=False):
        meta = loader.get_skill_metadata(s["name"]) or {}
        result.append({
            "name": s["name"],
            "description": meta.get("description", s["name"]),
            "source": s["source"],
        })
    return result


@router.post("/api/agents", status_code=201)
async def create_agent(
    body: AgentCreate,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await AgentRepository(factory).create({
        "name": body.name,
        "description": body.description,
        "default_model": body.default_model,
        "provider": body.provider,
        "max_iterations": body.max_iterations,
        "capabilities": body.capabilities,
        "skills_config": [],
        "tools_config": [],
        "is_default": False,
    })
    stats = await RunRepository(factory).get_stats_by_agent(agent.id)
    return await _agent_to_card(agent, stats)


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
    # Use model_fields_set for partial update; allow empty string to clear persona
    set_fields = body.model_fields_set
    fields = {k: v for k, v in body.model_dump().items() if k in set_fields}
    updated = await AgentRepository(factory).update(agent.id, **fields)
    stats = await RunRepository(factory).get_stats_by_agent(updated.id)
    return await _agent_to_card(updated, stats)


@router.get("/api/agents/{agent_id}/prompt-preview")
async def get_agent_prompt_preview(
    agent_id: str,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    """Return the static portion of the system prompt for this agent (no memory/knowledge)."""
    from nanoresearch.agent.context import ContextBuilder
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    workspace = request.app.state.workspace if hasattr(request.app.state, "workspace") else None
    if workspace is None:
        from pathlib import Path
        import os
        workspace = Path(os.environ.get("NANORESEARCH_WORKSPACE", "~/.nanoresearch/workspace")).expanduser()
    builder = ContextBuilder(workspace)
    skill_names = [s["name"] for s in (agent.skills_config or []) if s.get("enabled", True)]
    harness = agent.harness or {}
    # Load bound KBs for prompt preview
    _kbs = await AgentRepository(factory).list_bound_kbs(agent.id)
    _kb_bindings = [
        {"id": str(kb.id), "name": kb.name, "description": kb.description or ""}
        for kb in _kbs
    ] if _kbs else None
    workspace_text = builder._build_workspace_block(tool_names=None)
    agent_text = builder._build_agent_block(
        skill_names=skill_names,
        custom_persona=agent.persona or None,
        agents_registry=None,
        kb_bindings=_kb_bindings,
    )
    static = "\n\n---\n\n".join(p for p in [workspace_text, agent_text] if p)
    return {
        "static_prompt": static,
        "harness": harness,
        "persona": agent.persona or "",
        "total_token_budget": harness.get("total_token_budget", 3000),
        "memory_budget_ratio": harness.get("memory_budget_ratio", 0.6),
    }


@router.delete("/api/agents/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: str,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    if agent.is_default:
        raise HTTPException(status_code=400, detail="默认 Agent 不能删除")
    deleted = await AgentRepository(factory).delete(agent.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent 不存在")


@router.get("/api/agents/{agent_id}/tool-stats")
async def get_agent_tool_stats(
    agent_id: str,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    return await RunRepository(factory).get_tool_stats_by_agent(agent.id)


@router.get("/api/agents/{agent_id}/runs")
async def list_agent_runs(
    agent_id: str,
    request: Request,
    limit: int = 20,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    runs = await RunRepository(factory).list_by_agent(agent.id, limit=limit)
    return [_run_to_dict(r) for r in runs]


def _run_to_dict(run) -> dict:
    return {
        "id": str(run.id),
        "conversation_id": str(run.conversation_id),
        "status": run.status,
        "model_used": run.model_used,
        "tool_calls": run.tool_calls or [],
        "tokens_used": run.tokens_used or {},
        "duration_ms": run.duration_ms,
        "error_message": run.error_message,
        "artifacts": run.artifacts or [],
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


class BindKnowledgeRequest(BaseModel):
    kb_id: str


@router.get("/api/agents/{agent_id}/knowledge")
async def list_bound_knowledge(
    agent_id: str,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    kbs = await AgentRepository(factory).list_bound_kbs(agent.id)
    return [
        {"id": str(kb.id), "name": kb.name, "description": kb.description or ""}
        for kb in kbs
    ]


@router.post("/api/agents/{agent_id}/knowledge", status_code=201)
async def bind_knowledge(
    agent_id: str,
    body: BindKnowledgeRequest,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    kb_id = uuid.UUID(body.kb_id)
    # Verify KB exists
    kb = await KnowledgeRepository(factory).get(kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    await AgentRepository(factory).bind_kb(agent.id, kb_id)
    return {"status": "ok"}


@router.delete("/api/agents/{agent_id}/knowledge/{kb_id}", status_code=204)
async def unbind_knowledge(
    agent_id: str,
    kb_id: str,
    request: Request,
    _uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    agent = await _get_agent_or_404(agent_id, factory)
    ok = await AgentRepository(factory).unbind_kb(agent.id, uuid.UUID(kb_id))
    if not ok:
        raise HTTPException(status_code=404, detail="绑定不存在")


async def _get_agent_or_404(agent_id: str, factory):
    try:
        aid = uuid.UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    agent = await AgentRepository(factory).get_by_id(aid)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    return agent
