"""Chat, conversation, and run endpoints."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nanobot.server.middleware.auth import get_current_user
from nanobot.storage.repositories.conversation_repo import ConversationRepository
from nanobot.storage.repositories.run_repo import RunRepository

router = APIRouter()


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
    agent_loop = request.app.state.agent_loop
    run_repo = RunRepository(factory)

    conv = await _get_conv_or_404(body.conversation_id, uid, request)
    run = await run_repo.create(conversation_id=conv.id, uid=uid, agent_id=conv.agent_id)
    run_id = run.id

    queue: asyncio.Queue = asyncio.Queue()
    request.app.state.run_queues[str(run_id)] = queue

    # 使用 conv.session_key（创建时存储的），而非 f"web:{conv.id}"（两者 UUID 不同）
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
) -> None:
    run_repo = RunRepository(factory)
    start = _utcnow()
    await run_repo.update(run_id, status="running", started_at=start)

    async def on_stream(delta: str) -> None:
        await queue.put({"type": "message_delta", "chunk": delta})

    async def on_progress(text: str, *, tool_hint: bool = False) -> None:
        if tool_hint:
            await queue.put({"type": "tool_hint", "content": text})

    try:
        await loop.process_direct(
            content,
            session_key=session_key,
            channel="web",
            chat_id=session_key.split(":", 1)[-1],
            on_stream=on_stream,
            on_progress=on_progress,
        )
        finished = _utcnow()
        duration_ms = int((finished - start).total_seconds() * 1000)
        await run_repo.update(run_id, status="completed", finished_at=finished, duration_ms=duration_ms)
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
