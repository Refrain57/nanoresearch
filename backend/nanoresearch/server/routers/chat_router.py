"""Chat, conversation, and run endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.storage.repositories.agent_repo import AgentRepository
from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
from nanoresearch.storage.repositories.run_repo import RunRepository

router = APIRouter()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConversationCreate(BaseModel):
    title: str | None = None
    agent_id: uuid.UUID | None = None
    model: str | None = None  # initial model override


class RunCreate(BaseModel):
    conversation_id: str
    content: str
    rag_mode: str = "agentic"   # agentic | simple
    kb_id: str | None = None    # simple 模式使用


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
    metadata = None
    if body.model:
        metadata = {"agent_override": {"model": body.model}}
    conv = await repo.create(
        key=f"web:{conv_id}",
        uid=uid,
        agent_id=body.agent_id,
        title=body.title,
        metadata=metadata,
    )
    return {
        "id": str(conv.id),
        "title": conv.title,
        "session_key": conv.session_key,
        "agent_id": str(conv.agent_id) if conv.agent_id else None,
        "agent_override": (conv.conv_metadata or {}).get("agent_override") or {},
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
        # Hide internal orchestration turns (subagent results + continuation instruction).
        # They stay in the session for the LLM; they must not render as user bubbles.
        if not (isinstance(m.content, dict) and m.content.get("internal"))
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
    import hashlib
    from nanoresearch.bus.redis_keys import RedisKeys

    factory = request.app.state.session_factory
    run_repo = RunRepository(factory)

    conv = await _get_conv_or_404(body.conversation_id, uid, request)
    agent_override: dict = (conv.conv_metadata or {}).get("agent_override") or {}

    # 取该 agent 的 enabled skills、persona、harness、default_model，再应用对话级别覆盖
    # None = 不过滤（CLI 模式）；[] = agent 存在但未配置 skill，显示空
    skill_names: list[str] | None = None
    custom_persona: str | None = None
    agent_harness: dict = {}
    agents_registry: list[dict] = []
    agent_default_model: str | None = None
    if conv.agent_id:
        agent = await AgentRepository(factory).get_by_id(conv.agent_id)
        if agent is not None:
            agent_default_model = agent.default_model or None
            agent_skill_names = [s["name"] for s in (agent.skills_config or []) if s.get("enabled", True)]
            override_skills = agent_override.get("skills")
            if override_skills is not None:
                skill_names = [s for s in override_skills if s in agent_skill_names]
            else:
                skill_names = agent_skill_names
            custom_persona = agent.persona or None
            agent_harness = agent.harness or {}

    session_key = conv.session_key or f"web:{conv.id}"

    # Problem 4: idempotency check BEFORE creating the run record
    _redis = request.app.state.redis
    _job_id = hashlib.sha256(
        f"{uid}:{session_key}:{body.content}".encode()
    ).hexdigest()[:24]
    try:
        existing_run_id = await _redis.get(RedisKeys.job(_job_id))
    except Exception:
        existing_run_id = None
        import logging
        logging.getLogger(__name__).warning(
            "Redis GET(job) failed — skipping idempotency check, "
            "may produce duplicate run"
        )
    if existing_run_id:
        return {"run_id": existing_run_id, "conversation_id": str(conv.id), "status": "dedup"}

    run = await run_repo.create(conversation_id=conv.id, uid=uid, agent_id=conv.agent_id)
    run_id = run.id

    # Resolve kb_id for agentic RAG: frontend first, agent config overrides
    agent_kb_id = body.kb_id
    if conv.agent_id and agent is not None:
        if agent_harness.get("kb_id"):
            agent_kb_id = agent_harness["kb_id"]
        else:
            for _t in (agent.tools_config or []):
                _k = _t.get("kb_id") or _t.get("collection")
                if _k:
                    agent_kb_id = _k
                    break

    # Build agent registry: user's agents + default agents (uid-scoped, multi-user safe)
    all_agents = await AgentRepository(factory).list_by_user(uid)
    agents_registry = [{"id": str(a.id), "name": a.name, "description": a.description or ""} for a in all_agents]

    # Phase 0 (Must-fix 1): atomic idempotency gate BEFORE posting to the inbox. A duplicate
    # submit/retry must NOT create a second inbox entry (→ second run + LLM call + write). On a
    # lost race, return the winner's run_id without posting.
    _JOB_TTL = 3600
    _won = await _redis.set(RedisKeys.job(_job_id), str(run_id), nx=True, ex=_JOB_TTL)
    if not _won:
        _existing = await _redis.get(RedisKeys.job(_job_id))
        return {"run_id": _existing or str(run_id),
                "conversation_id": str(conv.id), "status": "dedup"}

    # Sole-enqueuer path: post to the mailbox + notify; the dispatcher does the actual enqueue.
    payload = {
        "run_id": str(run_id),
        "session_key": session_key,
        "conversation_id": str(conv.id),
        "content": body.content,
        "uid": uid,
        "rag_mode": body.rag_mode,
        "kb_id": body.kb_id,
        "skill_names": skill_names,
        "agent_id": str(conv.agent_id) if conv.agent_id else None,
        "agent_override": agent_override or None,
        "custom_persona": custom_persona,
        "harness": agent_harness or None,
        "agents_registry": agents_registry or None,
        "agent_kb_id": agent_kb_id,
        "job_id": _job_id,
    }
    await _enqueue_via_mailbox(_redis, payload)

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
    last_id: str = "0-0",
    uid: str = Depends(get_current_user),
):
    """SSE: stream events directly from `run_events:{run_id}` Redis Stream.

    Reconnect with `?last_id=<cursor>` to resume from a checkpoint within the
    24h replay window.
    """
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.bus.stream import xread_next

    run = await _get_run_or_404(run_id, uid, request)
    redis = request.app.state.redis
    factory = request.app.state.session_factory
    stream_key = RedisKeys.run_events(run_id)
    _session_key = run.conversation_id and f"web:{run.conversation_id}"
    _DB_CHECK_INTERVAL = 15.0  # only re-check the run's DB status after this much idle (rate-limit)

    async def _stream():
        import time as _time

        from nanoresearch.storage.repositories.run_repo import RunRepository
        cursor = last_id
        _normal_exit = False
        # Backdate so the FIRST idle batch checks immediately (fast terminal detection), then
        # at most once per interval (so a quiet-but-running run doesn't hammer the DB).
        _last_db_check = _time.monotonic() - _DB_CHECK_INTERVAL
        try:
            while True:
                events, cursor = await xread_next(redis, stream_key, cursor, timeout_ms=5_000)
                for ev in events:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev.get("type") == "run_end":
                        _normal_exit = True
                        return
                if events:
                    _last_db_check = _time.monotonic()   # activity → reset idle timer
                    continue
                # Idle batch (no new events). Concern 1: only hit the DB once per interval.
                if _time.monotonic() - _last_db_check < _DB_CHECK_INTERVAL:
                    continue
                _last_db_check = _time.monotonic()
                try:
                    fresh = await RunRepository(factory).get(uuid.UUID(run_id))
                except Exception:
                    fresh = None
                if fresh is None or fresh.status not in ("completed", "failed"):
                    continue   # still running / unknown → legit idle, keep waiting
                # Terminal in the DB but no run_end seen on the stream (worker crashed, stream
                # expired, etc.). Concern 3: a real run_end (e.g. from the watchdog) may have landed
                # during this idle window — drain once non-blocking and PREFER it over synthesizing.
                events, cursor = await xread_next(redis, stream_key, cursor, timeout_ms=0)
                for ev in events:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev.get("type") == "run_end":
                        _normal_exit = True
                        return
                # Still none → synthesize a run_end with the REAL terminal status (concern 2: a
                # failed run must not be reported as completed).
                synthetic = {"type": "run_end", "status": fresh.status}
                if fresh.status == "failed" and fresh.error_message:
                    synthetic["error"] = fresh.error_message
                yield f"data: {json.dumps(synthetic, ensure_ascii=False)}\n\n"
                _normal_exit = True
                return
        finally:
            # Problem 3: only set cancel flag when client disconnected mid-stream,
            # NOT when the run completed normally (would poison the next request).
            if not _normal_exit and _session_key:
                import logging as _logging
                try:
                    await redis.set(RedisKeys.cancel(_session_key), "1")
                except Exception:
                    _logging.getLogger(__name__).warning(
                        "Redis SET(cancel) failed — cancel flag not set"
                    )

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _enqueue_via_mailbox(redis, payload: dict) -> None:
    """Phase 0: post to the per-(agent, conversation) inbox + signal the dispatcher.

    Replaces the direct ARQ enqueue so the long-lived dispatcher is the sole enqueuer
    (no "HTTP enqueues + dispatcher enqueues" double source). agent_id is "none" until
    Phase 2 fills real identity — so the inbox is effectively per-conversation today.
    """
    from nanoresearch.bus import mailbox
    from nanoresearch.bus.redis_keys import RedisKeys

    agent_id = payload.get("agent_id") or "none"
    conversation_id = payload["conversation_id"]
    await mailbox.post_message(redis, agent_id, conversation_id, payload)
    await mailbox.post_notify(
        redis,
        mailbox_key=RedisKeys.agent_inbox(agent_id, conversation_id),
        cursor_key=RedisKeys.agent_inbox_cursor(agent_id, conversation_id),
        lock_key=RedisKeys.agent_lock(agent_id, conversation_id),
    )


async def _build_run_payload(factory, conversation_id: str, uid: str, content: str, run_id: str,
                             *, agent_id: str | None = None) -> dict:
    """Phase 1: rebuild a run payload (agent config from the conversation). Shared by the HTTP
    entry and the subagent/watchdog continuation wakeup so the continuation has identical agent
    config — no persisted 'intent' state needed.

    Phase 2 Task 1: an explicit *agent_id* (the owning/primary main) overrides the conv-derived
    value so the continuation/collector routes through the same identity the dispatcher gate sees
    (dispatcher.py:115,125). None keeps the Phase 1 conv.agent_id behaviour (single-main)."""
    conv = await ConversationRepository(factory).get_by_id(uuid.UUID(conversation_id))
    skill_names = None
    custom_persona = None
    agent_harness: dict = {}
    agent_kb_id = None
    if conv is not None and conv.agent_id:
        agent = await AgentRepository(factory).get_by_id(conv.agent_id)
        if agent is not None:
            skill_names = [s["name"] for s in (agent.skills_config or []) if s.get("enabled", True)]
            custom_persona = agent.persona or None
            agent_harness = agent.harness or {}
            if agent_harness.get("kb_id"):
                agent_kb_id = agent_harness["kb_id"]
    agents_registry = [
        {"id": str(a.id), "name": a.name, "description": a.description or ""}
        for a in await AgentRepository(factory).list_by_user(uid)
    ]
    return {
        "run_id": run_id,
        "session_key": (conv.session_key if conv else None) or f"web:{conversation_id}",
        "conversation_id": conversation_id,
        "content": content,
        "uid": uid,
        "rag_mode": "agentic",
        "kb_id": None,
        "skill_names": skill_names,
        "agent_id": agent_id if agent_id is not None else (str(conv.agent_id) if conv and conv.agent_id else None),
        "agent_override": None,
        "custom_persona": custom_persona,
        "harness": agent_harness or None,
        "agents_registry": agents_registry or None,
        "agent_kb_id": agent_kb_id,
        "job_id": None,
    }


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
