"""ARQ Worker entry — Agent execution process.

Launch:
    arq nanobot.worker.WorkerSettings

The worker subscribes to Redis for jobs enqueued by the web process and runs
the AgentLoop in isolation. All run events are written to
`run_events:{run_id}` Redis Streams, which the SSE endpoint XREADs directly.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arq.connections import RedisSettings

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Provider construction (mirrors chat_router._get_web_loop)
# ---------------------------------------------------------------------------

def _build_provider_from_spec(spec, fallback):
    """Instantiate a per-user provider from a ModelFactory spec, or return fallback."""
    if not spec or not spec.api_key:
        return fallback

    from nanobot.providers.registry import find_by_name

    p_spec = find_by_name(spec.provider) if spec.provider else None
    backend = p_spec.backend if p_spec else "openai_compat"

    if backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=spec.api_key,
            api_base=spec.base_url or (p_spec.default_api_base if p_spec else None),
            default_model=spec.model or "claude-sonnet-4-20250514",
            extra_headers=spec.extra_headers,
        )
    if backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
        return AzureOpenAIProvider(
            api_key=spec.api_key,
            api_base=spec.base_url,
            default_model=spec.model or "gpt-4o",
        )
    from nanobot.providers.openai_compat_provider import OpenAICompatProvider
    return OpenAICompatProvider(
        api_key=spec.api_key,
        api_base=spec.base_url,
        default_model=spec.model or "gpt-4o",
        extra_headers=spec.extra_headers,
        spec=p_spec,
    )


async def _build_agent_loop(
    uid: str,
    ctx: dict,
    model_override: str | None = None,
    agent_model: str | None = None,
):
    """Construct a per-uid AgentLoop inside the worker process."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.providers.model_factory import ModelFactory, ModelRole
    from nanobot.session.manager import SessionManager
    from nanobot.storage.repositories.user_settings_repo import UserSettingsRepository

    cfg = ctx["loop_config"]
    factory = ctx["session_factory"]
    user_cfg = await UserSettingsRepository(factory).get(uid)

    model = (
        model_override
        or agent_model
        or (user_cfg.model if user_cfg else None)
        or cfg.get("model")
    )

    base: Path = cfg["base_workspace"]
    ws = base / "users" / uid
    ws.mkdir(parents=True, exist_ok=True)

    session_manager = SessionManager(ws, session_factory=factory, default_uid=uid)

    providers = ((user_cfg.extra or {}).get("providers") or []) if user_cfg else []
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        config=cfg.get("config"),
        rag_settings=None,
        user_model=user_cfg.model if user_cfg else None,
        user_providers=providers,
        model_override=model_override or agent_model,
    )
    provider = _build_provider_from_spec(spec, cfg["provider"])

    return AgentLoop(
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
        uid=uid,
        session_factory=factory,
    )


# ---------------------------------------------------------------------------
# ARQ lifecycle
# ---------------------------------------------------------------------------

async def startup(ctx: dict) -> None:
    # Load .env from repo root (two levels up from this file) before anything else
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=False)

    from nanobot.cli.commands import build_loop_config

    ctx["loop_config"] = await build_loop_config()
    ctx["session_factory"] = ctx["loop_config"]["session_factory"]
    ctx["rag_settings"] = ctx["loop_config"].get("rag_settings")
    logger.info("ARQ worker startup complete")


async def shutdown(ctx: dict) -> None:
    from nanobot.storage.database import _engine

    if _engine is not None:
        try:
            await _engine.dispose()
        except Exception:
            pass
    logger.info("ARQ worker shutdown complete")


# ---------------------------------------------------------------------------
# Simple RAG path (port of chat_router._run_simple_rag)
# ---------------------------------------------------------------------------

async def _run_simple_rag_job(
    *,
    run_id: str,
    run_stream_key: str,
    content: str,
    kb_id: str,
    factory,
    settings,
    redis,
) -> None:
    import os as _os

    from nanobot.bus.stream import xadd_event
    from nanobot.server.routers.eval_router import _build_hybrid_search
    from nanobot.storage.repositories.knowledge_repo import KnowledgeRepository
    from nanobot.storage.repositories.run_repo import RunRepository

    run_repo = RunRepository(factory)
    start = _utcnow()
    await run_repo.update(run_id, status="running", started_at=start)
    try:
        kb = await KnowledgeRepository(factory).get(uuid.UUID(kb_id))
        chroma_col = (kb.chroma_collection if kb else None) or kb_id

        hybrid = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _build_hybrid_search(settings, chroma_col, top_k=5)
        )
        search_result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: hybrid.search(content, top_k=5, return_details=True)
        )
        if isinstance(search_result, list):
            results = search_result
        elif hasattr(search_result, "results"):
            results = search_result.results
        else:
            results = []
        ctx_text = "\n\n---\n\n".join(r.text for r in results[:5])

        from openai import AsyncOpenAI
        llm_cfg = getattr(settings, "llm", None)
        client = AsyncOpenAI(
            base_url=getattr(llm_cfg, "base_url", None) or "https://api.openai.com/v1",
            api_key=getattr(llm_cfg, "api_key", None) or _os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
        )
        model = getattr(llm_cfg, "model", None) or "gpt-4o-mini"
        stream = await client.chat.completions.create(
            model=model, temperature=0, stream=True,
            messages=[
                {"role": "system", "content": "基于提供的上下文信息准确回答问题。"},
                {"role": "user", "content": f"上下文:\n{ctx_text}\n\n问题: {content}"},
            ],
        )
        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if delta:
                await xadd_event(redis, run_stream_key, {"type": "message_delta", "chunk": delta})

        finished = _utcnow()
        duration_ms = int((finished - start).total_seconds() * 1000)
        await run_repo.update(run_id, status="completed", finished_at=finished, duration_ms=duration_ms)
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "completed", "duration_ms": duration_ms})
    except Exception as e:
        logger.error("Simple RAG failed: %s", e, exc_info=True)
        await run_repo.update(run_id, status="failed", finished_at=_utcnow(), error_message=str(e))
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "failed", "error": str(e)})


# ---------------------------------------------------------------------------
# Main ARQ job
# ---------------------------------------------------------------------------

async def run_agent_job(
    ctx: dict,
    *,
    run_id: str,
    session_key: str,
    content: str,
    uid: str,
    rag_mode: str = "agentic",
    kb_id: str | None = None,
    skill_names: list[str] | None = None,
    agent_id: str | None = None,
    agent_override: dict | None = None,
    custom_persona: str | None = None,
    harness: dict | None = None,
    agents_registry: list[dict] | None = None,
    agent_kb_id: str | None = None,
    job_id: str | None = None,
) -> None:
    """ARQ job: execute an Agent run inside the worker; events stream via Redis."""
    from nanobot.bus.redis_client import get_redis
    from nanobot.bus.redis_keys import RedisKeys
    from nanobot.bus.stream import xadd_event
    from nanobot.storage.repositories.run_repo import RunRepository

    redis = get_redis()
    run_stream_key = RedisKeys.run_events(run_id)
    factory = ctx["session_factory"]
    run_repo = RunRepository(factory)
    start = _utcnow()

    # Register idempotency lock (set in create_run, refreshed here for safety)
    if job_id:
        try:
            await redis.set(RedisKeys.job(job_id), run_id)
        except Exception:
            logger.warning("Redis SET(job) failed — idempotency unavailable")

    await run_repo.update(run_id, status="running", started_at=start)

    async def _check_cancel() -> bool:
        try:
            return bool(await redis.exists(RedisKeys.cancel(session_key)))
        except Exception:
            return False

    loop = None
    try:
        # Simple RAG path
        if rag_mode == "simple" and kb_id:
            await _run_simple_rag_job(
                run_id=run_id,
                run_stream_key=run_stream_key,
                content=content,
                kb_id=kb_id,
                factory=factory,
                settings=ctx["rag_settings"],
                redis=redis,
            )
            return

        # Agent path
        loop = await _build_agent_loop(
            uid, ctx,
            model_override=(agent_override or {}).get("model"),
        )

        tool_calls_log: list[dict] = []

        async def on_stream(delta: str) -> None:
            await xadd_event(redis, run_stream_key, {"type": "message_delta", "chunk": delta})

        async def on_progress(text: str, *, tool_hint: bool = False) -> None:
            if tool_hint:
                await xadd_event(redis, run_stream_key, {"type": "tool_hint", "content": text})

        async def on_tool_call(record: dict) -> None:
            tool_calls_log.append(record)
            raw_output = record.get("output")
            if isinstance(raw_output, dict):
                summary = (raw_output.get("text") or raw_output.get("content")
                           or raw_output.get("results") or str(raw_output))[:300]
            elif isinstance(raw_output, str):
                summary = raw_output[:300]
            else:
                summary = str(raw_output)[:300] if raw_output else ""
            await xadd_event(redis, run_stream_key, {
                "type": "tool_call",
                "name": record.get("name"),
                "input": record.get("input"),
                "output_summary": summary,
                "status": record.get("status", "success"),
            })

        # Build kb_bindings / kb_map for agentic RAG (same logic as old _run_agent)
        kb_bindings: list[dict] = []
        kb_map: dict[str, str] = {}
        if agent_id:
            try:
                from nanobot.storage.repositories.agent_repo import AgentRepository
                bound_kbs = await AgentRepository(factory).list_bound_kbs(uuid.UUID(agent_id))
                for _kb in bound_kbs:
                    _kid = str(_kb.id)
                    kb_bindings.append({
                        "id": _kid, "name": _kb.name,
                        "description": _kb.description or "",
                    })
                    if _kb.chroma_collection:
                        kb_map[_kid] = _kb.chroma_collection
            except Exception as kb_err:
                logger.warning("Failed to build kb_map for agent %s: %s", agent_id, kb_err)

        _chat_id = session_key.split(":", 1)[-1]
        await loop.process_direct(
            content,
            session_key=session_key,
            channel="web",
            chat_id=_chat_id,
            run_id=run_id,
            on_stream=on_stream,
            on_progress=on_progress,
            on_tool_call=on_tool_call,
            skill_names=skill_names,
            agent_id=agent_id,
            agent_override=agent_override,
            custom_persona=custom_persona,
            harness=harness,
            agents_registry=agents_registry,
            kb_bindings=kb_bindings,
            kb_map=kb_map,
        )

        # Wait for sub-agents: SCARD-only polling.  Subagents write directly to
        # run_events:{run_id}, so SSE picks them up without a relay.
        _MAX_WAIT, _waited = 1800, 0
        while _waited < _MAX_WAIT:
            if await _check_cancel():
                break
            try:
                _pending = await redis.scard(RedisKeys.pending(session_key))
            except Exception:
                logger.warning("Redis SCARD(pending) failed — exiting wait loop")
                break
            if not _pending:
                break
            await asyncio.sleep(5)
            _waited += 5
            if _waited % 30 == 0:
                await xadd_event(redis, run_stream_key, {"type": "heartbeat"})

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
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "completed", "duration_ms": duration_ms})
    except Exception as e:
        logger.error("run_agent_job failed: %s", e, exc_info=True)
        await run_repo.update(
            run_id, status="failed",
            finished_at=_utcnow(), error_message=str(e),
        )
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "failed", "error": str(e)})
    finally:
        # Close MCP stdio connections before the job task exits to avoid anyio cancel-scope error
        if loop is not None:
            try:
                await loop.close_mcp()
            except Exception:
                pass
        try:
            if job_id:
                await redis.delete(RedisKeys.job(job_id))
            await redis.delete(RedisKeys.cancel(session_key))
        except Exception:
            logger.warning("Redis cleanup in finally failed (non-fatal)")


class WorkerSettings:
    functions = [run_agent_job]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10
    job_timeout = 7200   # 2h — deep research subagents can run up to 30 min each
    keep_result = 3600
