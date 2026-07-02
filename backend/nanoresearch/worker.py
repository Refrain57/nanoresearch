"""ARQ Worker entry — Agent execution process.

Launch:
    arq nanoresearch.worker.WorkerSettings

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

from nanoresearch.config.loader import get_mode
from nanoresearch.utils.env_compat import apply_legacy_env_compat

apply_legacy_env_compat()

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

    from nanoresearch.providers.registry import find_by_name

    p_spec = find_by_name(spec.provider) if spec.provider else None
    backend = p_spec.backend if p_spec else "openai_compat"

    if backend == "anthropic":
        from nanoresearch.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=spec.api_key,
            api_base=spec.base_url or (p_spec.default_api_base if p_spec else None),
            default_model=spec.model or "claude-sonnet-4-20250514",
            extra_headers=spec.extra_headers,
        )
    if backend == "azure_openai":
        from nanoresearch.providers.azure_openai_provider import AzureOpenAIProvider
        return AzureOpenAIProvider(
            api_key=spec.api_key,
            api_base=spec.base_url,
            default_model=spec.model or "gpt-4o",
        )
    from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider
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
    from nanoresearch.agent.loop import AgentLoop
    from nanoresearch.providers.model_factory import ModelFactory, ModelRole
    from nanoresearch.session.manager import SessionManager
    from nanoresearch.storage.repositories.user_settings_repo import UserSettingsRepository

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
    roles = ((user_cfg.extra or {}).get("roles") or None) if user_cfg else None
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        config=cfg.get("config"),
        rag_settings=None,
        user_model=user_cfg.model if user_cfg else None,
        user_providers=providers,
        user_roles=roles,
        model_override=model_override or agent_model,
        mode=get_mode(),
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
        rag_settings=cfg.get("rag_settings"),
        arq_pool=ctx.get("arq_pool"),
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

    from nanoresearch.cli.commands import build_loop_config

    ctx["loop_config"] = await build_loop_config()
    ctx["session_factory"] = ctx["loop_config"]["session_factory"]
    ctx["rag_settings"] = ctx["loop_config"].get("rag_settings")

    # Phase 1: ARQ pool so subagents can enqueue the continuation run directly (bypassing the
    # inbox). The continuation reuses the original run_id and is gated by the agent lock.
    from arq import create_pool
    from arq.connections import RedisSettings as ArqRedisSettings
    ctx["arq_pool"] = await create_pool(ArqRedisSettings.from_dsn(REDIS_URL))
    logger.info("ARQ worker startup complete")


async def shutdown(ctx: dict) -> None:
    from nanoresearch.storage.database import _engine

    try:
        if ctx.get("arq_pool") is not None:
            await ctx["arq_pool"].aclose()
    except Exception:
        pass
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

    from nanoresearch.bus.stream import xadd_event
    from nanoresearch.server.routers.eval_router import _build_hybrid_search
    from nanoresearch.storage.repositories.knowledge_repo import KnowledgeRepository
    from nanoresearch.storage.repositories.run_repo import RunRepository

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
        from nanoresearch.config.loader import env_key_or_raise
        llm_cfg = getattr(settings, "llm", None)
        client = AsyncOpenAI(
            base_url=getattr(llm_cfg, "base_url", None) or "https://api.openai.com/v1",
            api_key=getattr(llm_cfg, "api_key", None) or env_key_or_raise("OPENAI_API_KEY", role="ingestion_llm"),
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

async def _has_pending_subagents(redis, session_key: str) -> bool:
    """Phase 1: True if this run spawned subagents that haven't all completed yet."""
    from nanoresearch.bus.redis_keys import RedisKeys
    try:
        return bool(await redis.scard(RedisKeys.pending(session_key)))
    except Exception:
        return False


async def _finalize_mailbox_run(redis, *, agent_id, conversation_id, lock_key, lock_token, entry_id):
    """Atomic finalize via one Lua (advance cursor → re-notify if backlog → release LAST,
    token-gated). The lock is freed only after the next notify is already in place, so there is
    no exposure window. token mismatch → whole thing no-ops. Best-effort, never raises."""
    from nanoresearch.bus import mailbox
    try:
        await mailbox.finalize_and_release(
            redis, agent_id=agent_id, conversation_id=conversation_id,
            lock_key=lock_key, token=lock_token, entry_id=entry_id,
        )
    except Exception:
        logger.warning("finalize_mailbox_run failed (non-fatal)")


async def _lock_refresher(redis, lock_key, lock_token, px_ms, stop_evt, abort_evt, proc_task,
                          cont_lock_key=None, cont_lock_token=None, cont_px_ms=120_000):
    """Extend the agent_lock lease while the run is alive. If the lease is lost (token no longer
    ours), set abort_evt and cancel the processing task so the run cannot keep writing unguarded —
    the token-gated finalize then no-ops. For a continuation run also best-effort refresh the
    continuation_lock each tick (over-refresh is harmless; its loss is backstopped by PX + watchdog)."""
    from nanoresearch.bus import dist_lock
    interval = px_ms / 3 / 1000.0
    try:
        while not stop_evt.is_set():
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=interval)
                return  # stop_evt set → run finished normally
            except asyncio.TimeoutError:
                pass
            if not await dist_lock.refresh(redis, lock_key, lock_token, px_ms=px_ms):
                logger.error(
                    "lock lease LOST for {} — aborting run to prevent unguarded writes", lock_key)
                abort_evt.set()
                if not proc_task.done():
                    proc_task.cancel()
                return
            if cont_lock_key:
                try:
                    await dist_lock.refresh(redis, cont_lock_key, cont_lock_token, px_ms=cont_px_ms)
                except Exception:
                    pass
    except asyncio.CancelledError:
        return


async def _continuation_acquire(redis, run_repo, run_id, run_stream_key, conversation_id,
                                cont_lock_key, cont_lock_token, *, agent_id=None, timeout_s=30):
    """Continuation run: acquire agent_lock (bounded retry) before any session read/write so it
    is mutually exclusive with a parent that may still be saving. On timeout, SELF-CLEAN (token-
    gated DEL continuation_lock + run_end failed + re-notify) and return (None, None). No reliance
    on PX/watchdog for the timeout path → no exposure window (§3.5)."""
    import time as _time

    from nanoresearch.bus import dist_lock, mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.bus.stream import xadd_event
    _aid = agent_id or "none"
    agent_lock_key = RedisKeys.agent_lock(_aid, conversation_id)
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        tok = await dist_lock.acquire(redis, agent_lock_key, px_ms=30_000)
        if tok:
            return agent_lock_key, tok
        await asyncio.sleep(0.5)
    logger.error("continuation %s could not acquire agent_lock in %ss — self-cleaning", run_id, timeout_s)
    try:
        await dist_lock.release(redis, cont_lock_key, cont_lock_token)  # token-gated DEL
        await run_repo.update(run_id, status="failed", finished_at=_utcnow(),
                              error_message="continuation could not acquire agent_lock")
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "failed", "error": "continuation lock timeout"})
        await mailbox.post_notify(
            redis,
            mailbox_key=RedisKeys.agent_inbox(_aid, conversation_id),
            cursor_key=RedisKeys.agent_inbox_cursor(_aid, conversation_id),
            lock_key=agent_lock_key)
    except Exception:
        logger.warning("continuation %s self-clean cleanup failed (non-fatal)", run_id)
    return None, None


async def _continuation_drain_and_append(redis, sessions, session_key, uid):
    """Drain subagent_results (atomic LRANGE+DEL) and append each into the session — INSIDE the
    agent_lock (caller holds it), so the baseline read in process_direct sees a single, latest state."""
    from nanoresearch.bus import mailbox
    staged = await mailbox.drain_subagent_results(redis, session_key)
    for item in staged:
        body = (f"[Subagent '{item.get('label', '')}' "
                f"{'completed' if item.get('status') == 'ok' else 'failed'}]\n\n"
                f"Task: {item.get('task', '')}\n\nResult:\n{item.get('result', '')}")
        try:
            # internal=True → LLM-visible (continuation rebuilds context from this history) but
            # hidden from the frontend chat view, which must not render orchestration text as a
            # user bubble (get_messages filters it; chat.js filters as defense-in-depth).
            await sessions.append_message(
                session_key, {"role": "user", "content": body, "internal": True}, uid=uid)
        except Exception as e:
            logger.warning("continuation append staged result failed (non-fatal): {}", e)


def _make_on_citations(redis, run_stream_key: str):
    """Factory: return an on_citations callback that emits a citations SSE event.

    Extracted as a module-level factory (mirroring on_tool_call's inline closure)
    so it can be unit-tested with a fake Redis without needing a running worker.
    xadd_event JSON-encodes the payload synchronously, so the items list is
    serialized on the call — no deep-copy needed even if the accumulator mutates.
    """
    from nanoresearch.bus.stream import xadd_event

    async def on_citations(items: list[dict]) -> None:
        await xadd_event(redis, run_stream_key, {"type": "citations", "items": items})

    return on_citations


def _apply_cron_run_context(loop, cron: dict | None) -> None:
    """Mark a cron-triggered run so its agent can't schedule further cron jobs.

    The self-schedule guard lives on the CronTool as a ContextVar; the loop is fresh per run,
    so setting it once here holds for the whole run. No-op for ordinary (non-cron) runs.
    """
    if cron is None:
        return
    ct = loop.tools.get("cron")
    if ct is not None and hasattr(ct, "set_cron_context"):
        ct.set_cron_context(True)


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
    conversation_id: str | None = None,
    # Cron production redesign: present iff this run was posted by the CronScheduler. Carries
    # {deliver, channel, to, task_context}; marks the run as cron-triggered (self-schedule guard).
    cron: dict | None = None,
    # Phase 0 mailbox lock lifecycle (set by the dispatcher; absent for CLI/legacy).
    _lock_key: str | None = None,
    _lock_token: str | None = None,
    _entry_id: str | None = None,
    # Phase 1 continuation (set by the join / watchdog; bypasses the inbox, holds continuation_lock).
    _cont_lock_key: str | None = None,
    _cont_lock_token: str | None = None,
) -> None:
    """ARQ job: execute an Agent run inside the worker; events stream via Redis."""
    from nanoresearch.bus.redis_client import get_redis
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.bus.stream import xadd_event
    from nanoresearch.storage.repositories.run_repo import RunRepository

    redis = get_redis()
    run_stream_key = RedisKeys.run_events(run_id)
    factory = ctx["session_factory"]
    run_repo = RunRepository(factory)
    start = _utcnow()

    # Phase 0/1: mailbox lock lifecycle. entry_id is present for dispatcher-enqueued runs and
    # absent (None) for the Phase 1 continuation (it bypasses the inbox but still owns the lock,
    # so it still needs the refresher + a release-only finalize).
    _mailbox_enabled = bool(_lock_key and _lock_token)
    _is_continuation = bool(_cont_lock_key and _cont_lock_token)
    _refresh_stop = asyncio.Event()
    _abort_evt = asyncio.Event()
    _refresh_task = None

    # Register idempotency lock (set in create_run, refreshed here for safety)
    if job_id:
        try:
            await redis.set(RedisKeys.job(job_id), run_id)
        except Exception:
            logger.warning("Redis SET(job) failed — idempotency unavailable")

    await run_repo.update(run_id, status="running", started_at=start)

    # --------------- perf-test bypass (no LLM) ---------------
    if content.startswith("__PERF_TEST__"):
        _n = int(content.split(":")[1]) if ":" in content else 5
        for _i in range(_n):
            await xadd_event(redis, run_stream_key, {"type": "message_delta", "chunk": f"c{_i}"})
        _fin = _utcnow()
        _dur = int((_fin - start).total_seconds() * 1000)
        await run_repo.update(run_id, status="completed", finished_at=_fin, duration_ms=_dur)
        await xadd_event(redis, run_stream_key, {"type": "run_end", "status": "completed", "duration_ms": _dur})
        return
    # ----------------------------------------------------------

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
        _apply_cron_run_context(loop, cron)

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

        on_citations = _make_on_citations(redis, run_stream_key)

        # Build kb_bindings / kb_map for agentic RAG (same logic as old _run_agent)
        kb_bindings: list[dict] = []
        kb_map: dict[str, str] = {}
        if agent_id:
            try:
                from nanoresearch.storage.repositories.agent_repo import AgentRepository
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

        # Build citation_source_map: {file_path: filename} across ALL of this uid's KBs
        # (not just agent-bound), so whichever KB kb_search actually queries is covered —
        # the agent may query a KB via kb_id that isn't in its bound list. The chunk's
        # source_path equals KbDocument.file_path (both are the ingest-time path, typically
        # a temp file); remapping to KbDocument.filename gives the original document name.
        # Injected onto loop; loop.py remaps citation["source"] in after_iteration.
        citation_source_map: dict[str, str] = {}
        if uid:
            try:
                from nanoresearch.storage.repositories.knowledge_repo import KnowledgeRepository
                _kb_repo = KnowledgeRepository(factory)
                for _kb in await _kb_repo.list_by_uid(uid):
                    try:
                        for _doc in await _kb_repo.list_documents(_kb.id):
                            if _doc.file_path and _doc.filename:
                                citation_source_map[_doc.file_path] = _doc.filename
                    except Exception as _doc_err:
                        logger.warning("citation_source_map: failed for kb %s: %s", _kb.id, _doc_err)
            except Exception as _csm_err:
                logger.warning("citation_source_map build failed (non-fatal): %s", _csm_err)
        loop._citation_source_map = citation_source_map

        # Phase 1 continuation: acquire agent_lock (bounded retry; self-clean on timeout) BEFORE any
        # session read/write, then drain staged subagent results into the session — all in-lock, so
        # process_direct's baseline read sees the single latest state (§4.2 / §5).
        if _is_continuation:
            _lock_key, _lock_token = await _continuation_acquire(
                redis, run_repo, run_id, run_stream_key, conversation_id,
                _cont_lock_key, _cont_lock_token,
                agent_id=str(agent_id) if agent_id else None)
            if _lock_token is None:
                return  # self-cleaned (run_end emitted, continuation_lock released, re-notified)
            _mailbox_enabled = True
            await _continuation_drain_and_append(redis, loop.sessions, session_key, uid)

        _chat_id = session_key.split(":", 1)[-1]

        # Web bridge: the `message` tool is wired to bus.publish_outbound, whose outbound queue
        # nothing consumes in the worker → its content would be silently dropped (the frontend
        # only reads the run_events SSE stream). Re-point it to run_events so agent-sent messages
        # reach the web UI. Persistence across the run_end reload is handled in _process_message
        # (folds the sends into the saved turn as assistant messages).
        from nanoresearch.agent.tools.message import MessageTool as _MessageTool
        _mt = loop.tools.get("message")
        if isinstance(_mt, _MessageTool):
            async def _web_message_sink(m: Any) -> None:
                if getattr(m, "channel", None) == "web":
                    await xadd_event(redis, run_stream_key,
                                     {"type": "agent_message", "content": m.content})
            _mt.set_send_callback(_web_message_sink)

        _proc_task = asyncio.create_task(loop.process_direct(
            content,
            session_key=session_key,
            channel="web",
            chat_id=_chat_id,
            run_id=run_id,
            on_stream=on_stream,
            on_progress=on_progress,
            on_tool_call=on_tool_call,
            on_citations=on_citations,
            skill_names=skill_names,
            agent_id=agent_id,
            agent_override=agent_override,
            custom_persona=custom_persona,
            harness=harness,
            agents_registry=agents_registry,
            kb_bindings=kb_bindings,
            kb_map=kb_map,
            conversation_id=conversation_id,
            # Continuation runs carry an internal summarize instruction as `content`; mark it so
            # the persisted user turn is hidden from the frontend (kept for the LLM).
            internal_content=_is_continuation,
        ))
        if _mailbox_enabled:
            _refresh_task = asyncio.create_task(_lock_refresher(
                redis, _lock_key, _lock_token, 30_000, _refresh_stop, _abort_evt, _proc_task,
                cont_lock_key=_cont_lock_key if _is_continuation else None,
                cont_lock_token=_cont_lock_token if _is_continuation else None))
        outbound = await _proc_task   # refresher cancels this on lease loss → CancelledError

        # Phase 1: if this run spawned subagents (pending non-empty), do NOT finish the turn.
        # Leave status=running and emit NO run_end — the run_events stream stays open so the
        # frontend keeps streaming subagent output, and the continuation run (reusing this
        # run_id, enqueued by the join when the batch completes) emits run_end after summarizing.
        # The Phase 0 `finally` still runs on this return (releases the inbox lock etc.).
        if await _has_pending_subagents(redis, session_key):
            logger.info("run_agent_job %s spawned subagents — deferring run_end to continuation", run_id)
            return

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

        # Cron delivery (web): a cron run executes in its dedicated conversation, so push its
        # result back into the origin conversation + that conversation's live SSE stream. No-op
        # for ordinary runs (cron is None) and for non-web / non-deliver jobs.
        if cron is not None:
            try:
                from nanoresearch.cron.delivery import deliver_cron_result_web
                await deliver_cron_result_web(
                    redis, loop.sessions, uid=uid, cron=cron,
                    response_text=(outbound.content if outbound is not None else None),
                )
            except Exception:
                logger.warning("cron delivery failed (non-fatal)")
    except Exception as e:
        logger.error("run_agent_job failed: %s", e, exc_info=True)
        await run_repo.update(
            run_id, status="failed",
            finished_at=_utcnow(), error_message=str(e),
        )
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "failed", "error": str(e)})
    finally:
        # Phase 0: stop the lease refresher, then atomically finalize the mailbox run
        # (advance cursor → re-notify backlog → release lock). Runs even on cancel/error.
        if _mailbox_enabled:
            _refresh_stop.set()
            if _refresh_task is not None:
                try:
                    await _refresh_task
                except Exception:
                    pass
            await _finalize_mailbox_run(
                redis,
                agent_id=str(agent_id) if agent_id else "none",
                conversation_id=conversation_id,
                lock_key=_lock_key, lock_token=_lock_token, entry_id=_entry_id,
            )
            # Phase 1 continuation: also DEL the continuation_lock (token-gated) so the gate clears
            # and deferred user messages can run. (finalize above already released agent_lock + re-notified.)
            if _is_continuation:
                try:
                    from nanoresearch.bus import dist_lock as _dl
                    await _dl.release(redis, _cont_lock_key, _cont_lock_token)
                except Exception:
                    logger.warning("continuation_lock release failed (non-fatal); PX will clear it")
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


async def ingest_document_task(
    ctx: dict,
    *,
    kb_id: str,
    doc_id: str,
    file_path: str,
    chroma_collection: str = "",
    original_filename: str = "",
    chunk_strategy: str = "auto",
    pdf_parser: str = "mineru",
    uid: str = "",
    content_hash: str = "",
    file_is_permanent: bool = False,
) -> None:
    """ARQ job: copy to permanent storage (unless already there), then run unified.ingest_document."""
    import os
    import shutil
    from pathlib import Path as _Path

    from nanoresearch.storage.repositories.knowledge_repo import KnowledgeRepository
    from nanoresearch.rag.ingestion.unified import ingest_document, IngestFailedError

    factory = ctx["session_factory"]
    settings = ctx["rag_settings"]
    doc_uuid = uuid.UUID(doc_id)
    repo = KnowledgeRepository(factory)

    if uid:
        from nanoresearch.providers.model_factory import (
            ModelFactory,
            ModelResolutionError,
            ModelRole,
        )
        from nanoresearch.storage.repositories.user_settings_repo import UserSettingsRepository

        user_cfg = None
        try:
            user_cfg = await UserSettingsRepository(factory).get(uid)
        except Exception:
            # Settings lookup is best-effort; fall through to factory defaults.
            pass

        if user_cfg is not None or get_mode() == "server":
            user_model = user_cfg.model if user_cfg else None
            user_providers = (user_cfg.extra or {}).get("providers", []) if user_cfg else []
            user_roles = (user_cfg.extra or {}).get("roles") if user_cfg else None
            spec = ModelFactory.resolve(
                ModelRole.INGESTION_LLM,
                config=ctx["loop_config"].get("config"),
                rag_settings=settings,
                user_model=user_model,
                user_providers=user_providers,
                user_roles=user_roles,
                mode=get_mode(),
            )
            settings = ModelFactory.patch_settings(settings, ModelRole.INGESTION_LLM, spec)

    await repo.update_document_status(doc_uuid, "parsing")

    perm_path = file_path
    if not file_is_permanent:
        # Web UI path: file_path is a temp file; copy to permanent storage then delete.
        try:
            doc_dir = _Path(os.path.expanduser(f"~/.nanoresearch/rag/documents/{kb_id}"))
            doc_dir.mkdir(parents=True, exist_ok=True)
            ext = _Path(file_path).suffix or ""
            perm_path = str(doc_dir / f"{doc_id}{ext}")
            shutil.copy2(file_path, perm_path)
            await repo.update_document_file_path(doc_uuid, perm_path)
        except Exception as _e:
            logger.warning("ingest_document_task: 持久化源文件失败: %s", _e)

    try:
        result = await ingest_document(
            kb_id=kb_id,
            file_path=perm_path,
            original_filename=original_filename,
            content_hash=content_hash,
            pdf_parser=pdf_parser,
            chunk_strategy=chunk_strategy,
            uid=uid,
            repo=repo,
            settings=settings,
        )
        logger.info(
            "ingest_document_task: doc=%s status=%s chunks=%d",
            doc_id, result.status, result.chunk_count,
        )
    except IngestFailedError as exc:
        logger.error("ingest_document_task: doc=%s pipeline failed: %s", doc_id, exc)
    except Exception as exc:
        logger.error("ingest_document_task: doc=%s unexpected error: %s", doc_id, exc, exc_info=True)
        await repo.update_document_status(doc_uuid, "failed", error_msg=str(exc))
    finally:
        if not file_is_permanent:
            try:
                os.unlink(file_path)
            except Exception:
                pass


class WorkerSettings:
    functions = [run_agent_job, ingest_document_task]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10
    job_timeout = 7200   # 2h — deep research subagents can run up to 30 min each
    keep_result = 3600
