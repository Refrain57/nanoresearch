"""Chat, conversation, and run endpoints."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nanobot.server.middleware.auth import get_current_user
from nanobot.storage.repositories.agent_repo import AgentRepository
from nanobot.storage.repositories.conversation_repo import ConversationRepository
from nanobot.storage.repositories.run_repo import RunRepository

router = APIRouter()


async def _get_web_loop(uid: str, state, model_override: str | None = None):
    """Return (or lazily create) the per-uid AgentLoop. Double-checked locking.

    Falls back to state.channel_loop when loop_config is absent (e.g. tests).
    model_override: conversation-level model override; triggers loop eviction if different.
    """
    from nanobot.agent.loop import AgentLoop
    from nanobot.session.manager import SessionManager

    cfg = getattr(state, "loop_config", None) or {}
    if "base_workspace" not in cfg:
        return state.channel_loop

    # Evict loop if the conversation needs a different model than what's cached
    cached = state.web_loops.get(uid)
    if cached and model_override and getattr(cached, "model", None) != model_override:
        state.web_loops.pop(uid, None)

    if uid in state.web_loops:          # fast path (no lock)
        return state.web_loops[uid]

    async with state.web_loops_lock:
        if uid not in state.web_loops:  # double-check after acquiring lock
            from nanobot.storage.repositories.user_settings_repo import UserSettingsRepository
            user_cfg = await UserSettingsRepository(state.session_factory).get(uid)
            # Precedence: conversation override > user settings > system default
            model = model_override or (user_cfg.model if user_cfg else None) or cfg.get("model")

            base: Path = cfg["base_workspace"]
            ws = base / "users" / uid
            ws.mkdir(parents=True, exist_ok=True)

            session_manager = SessionManager(ws, session_factory=state.session_factory, default_uid=uid)

            # Per-user provider: find the provider whose models list contains the selected model
            providers = ((user_cfg.extra or {}).get("providers") or []) if user_cfg else []
            matched = next(
                (p for p in providers if model and model in p.get("models", [])),
                providers[0] if providers else None,  # fallback: first provider if any
            )
            user_api_key = (matched or {}).get("api_key") or None
            if user_api_key:
                from nanobot.providers.openai_compat_provider import OpenAICompatProvider
                user_api_base = (matched or {}).get("api_base") or None
                provider = OpenAICompatProvider(
                    api_key=user_api_key,
                    api_base=user_api_base,
                    default_model=model or "gpt-4o",
                )
            else:
                provider = cfg["provider"]

            loop = AgentLoop(
                bus=cfg["bus"],
                provider=provider,
                workspace=ws,
                model=model,
                max_iterations=cfg.get("max_iterations", 40),
                context_window_tokens=cfg.get("context_window_tokens", 65536),
                web_search_config=cfg.get("web_search_config"),
                web_proxy=cfg.get("web_proxy"),
                exec_config=cfg.get("exec_config"),
                cron_service=cfg.get("cron_service"),
                restrict_to_workspace=True,
                session_manager=session_manager,
                mcp_servers=cfg.get("mcp_servers"),
                channels_config=cfg.get("channels_config"),
                timezone=cfg.get("timezone"),
                research_config=cfg.get("research_config"),
                knowledge_search=cfg.get("knowledge_search"),
                rag_store=cfg.get("rag_store"),
            )
            state.web_loops[uid] = loop

    return state.web_loops[uid]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConversationCreate(BaseModel):
    title: str | None = None
    agent_id: uuid.UUID | None = None


class RunCreate(BaseModel):
    conversation_id: str
    content: str


class AgentOverrideUpdate(BaseModel):
    model: str | None = None          # "" clears override
    max_iterations: int | None = None  # None keeps existing
    skills: list[str] | None = None   # None = keep existing; [] = clear to agent default


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@router.get("/api/conversations")
async def list_conversations(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    repo = ConversationRepository(factory)
    convs = await repo.list_conversations(uid, limit=limit, offset=offset)
    result = []
    for c in convs:
        # N+1 查询：MVP 可接受（limit 默认 20），后续可用子查询优化
        last_msg = await repo.get_last_message(c.id)
        preview = None
        if last_msg and isinstance(last_msg.content, dict):
            text = last_msg.content.get("text") or last_msg.content.get("content", "")
            if isinstance(text, str) and text:
                preview = text[:100] + ("..." if len(text) > 100 else "")
        result.append({
            "id": str(c.id),
            "title": c.title,
            "channel": c.channel,
            "agent_id": str(c.agent_id) if c.agent_id else None,
            "agent_override": (c.conv_metadata or {}).get("agent_override") or {},
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "last_message_preview": preview,
        })
    return result


@router.post("/api/conversations", status_code=201)
async def create_conversation(
    request: Request,
    body: ConversationCreate,
    uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    repo = ConversationRepository(factory)
    conv_id = uuid.uuid4()
    conv = await repo.create(
        key=f"web:{conv_id}",
        uid=uid,
        agent_id=body.agent_id,
        title=body.title,
    )
    return {
        "id": str(conv.id),
        "title": conv.title,
        "session_key": conv.session_key,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


@router.get("/api/conversations/{conv_id}")
async def get_conversation(
    conv_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    conv = await _get_conv_or_404(conv_id, uid, request)
    return {
        "id": str(conv.id),
        "title": conv.title,
        "session_key": conv.session_key,
        "agent_id": str(conv.agent_id) if conv.agent_id else None,
        "agent_override": (conv.conv_metadata or {}).get("agent_override") or {},
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


@router.get("/api/conversations/{conv_id}/messages")
async def get_messages(
    conv_id: str,
    request: Request,
    limit: int = 50,
    offset: int = 0,
    uid: str = Depends(get_current_user),
):
    conv = await _get_conv_or_404(conv_id, uid, request)
    factory = request.app.state.session_factory
    repo = ConversationRepository(factory)
    msgs = await repo.get_messages_paged(conv.id, limit=limit, offset=offset)
    return [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "seq": m.seq,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in msgs
    ]


@router.delete("/api/conversations/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    conv = await _get_conv_or_404(conv_id, uid, request)
    factory = request.app.state.session_factory
    repo = ConversationRepository(factory)
    await repo.delete(conv.id)


@router.put("/api/conversations/{conv_id}/agent-override")
async def update_agent_override(
    conv_id: str,
    request: Request,
    body: AgentOverrideUpdate,
    uid: str = Depends(get_current_user),
):
    conv = await _get_conv_or_404(conv_id, uid, request)
    repo = ConversationRepository(request.app.state.session_factory)

    # Build new override from existing + updates
    current = dict((conv.conv_metadata or {}).get("agent_override") or {})
    sent = body.model_fields_set if hasattr(body, "model_fields_set") else body.__fields_set__
    if "model" in sent:
        if body.model:
            current["model"] = body.model
        else:
            current.pop("model", None)
    if "max_iterations" in sent:
        if body.max_iterations is not None:
            current["max_iterations"] = body.max_iterations
        else:
            current.pop("max_iterations", None)
    if "skills" in sent:
        if body.skills is not None:
            current["skills"] = body.skills  # [] means "clear override, use agent default"
        else:
            current.pop("skills", None)

    await repo.update_agent_override(conv.id, current)
    # Evict loop so next message picks up new model if changed
    request.app.state.web_loops.pop(uid, None)
    return {"agent_override": current}


@router.get("/api/conversations/{conv_id}/runs")
async def get_conversation_runs(
    conv_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    conv = await _get_conv_or_404(conv_id, uid, request)
    factory = request.app.state.session_factory
    run_repo = RunRepository(factory)
    runs = await run_repo.list_by_conversation(conv.id)
    return [_run_to_dict(r) for r in runs]


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@router.post("/api/runs", status_code=201)
async def create_run(
    request: Request,
    body: RunCreate,
    uid: str = Depends(get_current_user),
):
    factory = request.app.state.session_factory
    run_repo = RunRepository(factory)

    conv = await _get_conv_or_404(body.conversation_id, uid, request)
    agent_override: dict = (conv.conv_metadata or {}).get("agent_override") or {}

    agent_loop = await _get_web_loop(uid, request.app.state, model_override=agent_override.get("model"))
    run = await run_repo.create(conversation_id=conv.id, uid=uid, agent_id=conv.agent_id)
    run_id = run.id

    # 取该 agent 的 enabled skills、persona、harness，再应用对话级别覆盖
    # None = 不过滤（CLI 模式）；[] = agent 存在但未配置 skill，显示空
    skill_names: list[str] | None = None
    custom_persona: str | None = None
    agent_harness: dict = {}
    agents_registry: list[dict] = []
    if conv.agent_id:
        agent = await AgentRepository(factory).get_by_id(conv.agent_id)
        if agent is not None:
            agent_skill_names = [s["name"] for s in (agent.skills_config or []) if s.get("enabled", True)]
            override_skills = agent_override.get("skills")
            if override_skills is not None:
                skill_names = [s for s in override_skills if s in agent_skill_names]
            else:
                skill_names = agent_skill_names
            custom_persona = agent.persona or None
            agent_harness = agent.harness or {}

    # Build agent registry: user's agents + default agents (uid-scoped, multi-user safe)
    all_agents = await AgentRepository(factory).list_by_user(uid)
    agents_registry = [{"id": str(a.id), "name": a.name, "description": a.description or ""} for a in all_agents]

    queue: asyncio.Queue = asyncio.Queue()
    request.app.state.run_queues[str(run_id)] = queue

    session_key = conv.session_key or f"web:{conv.id}"
    asyncio.create_task(
        _run_agent(
            loop=agent_loop,
            run_id=run_id,
            content=body.content,
            session_key=session_key,
            queue=queue,
            factory=factory,
            run_queues=request.app.state.run_queues,
            skill_names=skill_names,
            agent_id=str(conv.agent_id) if conv.agent_id else None,
            agent_override=agent_override or None,
            custom_persona=custom_persona,
            harness=agent_harness or None,
            agents_registry=agents_registry or None,
        )
    )

    return {
        "run_id": str(run_id),
        "conversation_id": str(conv.id),
        "status": "pending",
    }


@router.get("/api/runs/{run_id}")
async def get_run(
    run_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    run = await _get_run_or_404(run_id, uid, request)
    return _run_to_dict(run)


@router.get("/api/runs/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    run = await _get_run_or_404(run_id, uid, request)
    queue: asyncio.Queue | None = request.app.state.run_queues.get(run_id)

    if queue is None:
        # Run already completed — return a single terminal event.
        async def _done():
            yield f"data: {json.dumps({'type': 'run_end', 'status': run.status}, ensure_ascii=False)}\n\n"
        return StreamingResponse(_done(), media_type="text/event-stream")

    # NOTE: 已知限制 —— 客户端断线后重连无法补发已消费的 delta 事件。
    # Phase 3 前端开发时注意：断线即视为丢失历史流式内容。
    async def _stream():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            request.app.state.run_queues.pop(run_id, None)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Background agent task
# ---------------------------------------------------------------------------

async def _run_agent(
    loop,
    run_id: uuid.UUID,
    content: str,
    session_key: str,
    queue: asyncio.Queue,
    factory,
    run_queues: dict,
    skill_names: list[str] | None = None,
    agent_id: str | None = None,
    agent_override: dict | None = None,
    custom_persona: str | None = None,
    harness: dict | None = None,
    agents_registry: list[dict] | None = None,
) -> None:
    run_repo = RunRepository(factory)
    start = _utcnow()
    await run_repo.update(run_id, status="running", started_at=start)

    tool_calls_log: list[dict] = []

    async def on_stream(delta: str) -> None:
        await queue.put({"type": "message_delta", "chunk": delta})

    async def on_progress(text: str, *, tool_hint: bool = False) -> None:
        if tool_hint:
            await queue.put({"type": "tool_hint", "content": text})

    async def on_tool_call(record: dict) -> None:
        tool_calls_log.append(record)

    try:
        await loop.process_direct(
            content,
            session_key=session_key,
            channel="web",
            chat_id=session_key.split(":", 1)[-1],
            on_stream=on_stream,
            on_progress=on_progress,
            on_tool_call=on_tool_call,
            skill_names=skill_names,
            agent_id=agent_id,
            agent_override=agent_override,
            custom_persona=custom_persona,
            harness=harness,
            agents_registry=agents_registry,
        )
        finished = _utcnow()
        duration_ms = int((finished - start).total_seconds() * 1000)
        usage = loop._last_usage or {}
        tokens_used = {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "cache_read": usage.get("cache_read_input_tokens", 0),
            "cache_write": usage.get("cache_creation_input_tokens", 0),
        }
        await run_repo.update(
            run_id,
            status="completed",
            finished_at=finished,
            duration_ms=duration_ms,
            model_used=loop.model,
            tokens_used=tokens_used,
            tool_calls=tool_calls_log,
        )
        await queue.put({"type": "run_end", "status": "completed", "duration_ms": duration_ms})
    except Exception as e:
        await run_repo.update(run_id, status="failed", finished_at=_utcnow(), error_message=str(e))
        await queue.put({"type": "run_end", "status": "failed", "error": str(e)})
    finally:
        await queue.put(None)  # sentinel — 保证 SSE 端无论如何都能退出
        run_queues.pop(str(run_id), None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_conv_or_404(conv_id: str, uid: str, request: Request):
    factory = request.app.state.session_factory
    repo = ConversationRepository(factory)
    try:
        cid = uuid.UUID(conv_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="对话不存在")
    conv = await repo.get_by_id(cid)
    if conv is None or conv.uid != uid:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conv


async def _get_run_or_404(run_id: str, uid: str, request: Request):
    factory = request.app.state.session_factory
    repo = RunRepository(factory)
    try:
        rid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run 不存在")
    run = await repo.get(rid)
    if run is None or run.uid != uid:
        raise HTTPException(status_code=404, detail="Run 不存在")
    return run


def _run_to_dict(run) -> dict:
    return {
        "id": str(run.id),
        "conversation_id": str(run.conversation_id),
        "agent_id": str(run.agent_id) if run.agent_id else None,
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
