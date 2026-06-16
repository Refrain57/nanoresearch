# SDD — ARQ Worker + Run Events Stream（异步流解耦 + Worker 进程化）

## 1. 背景与目标

### 问题

当前架构中，Agent 执行和 SSE 连接通过同进程的 `asyncio.Queue` 绑定：

```
POST /api/runs → asyncio.Queue → asyncio.create_task(_run_agent)
GET  /api/runs/{run_id}/events → SSE 消费 asyncio.Queue
```

`app.state.run_queues[run_id]` 是进程本地状态。负载均衡把两个请求路由到不同服务器时，第二台找不到 Queue，SSE 无事件。

### 目标

两个改造必须合并一次完成：

- **Run Events Stream**：把主 Agent token 从 `asyncio.Queue` 迁移到 `Redis Stream run_events:{run_id}`，SSE 端直接 XREAD，两端彻底解耦
- **ARQ Worker**：把 Agent 执行从 web 进程的 `asyncio.create_task` 移到独立 ARQ Worker 进程

原因：如果分两次做，中间状态是 Worker 写 Stream 但 SSE 还在读 Queue，SSE 无事件。

---

## 2. 当前状态

### 事件流向（改造前）

| 事件类型 | 写入方 | 传递路径 | 读取方 |
|---|---|---|---|
| `message_delta` | `_run_agent.on_stream` | `asyncio.Queue` | SSE `_stream()` |
| `tool_hint` | `_run_agent.on_progress` | `asyncio.Queue` | SSE `_stream()` |
| `tool_call` | `_run_agent.on_tool_call` | `asyncio.Queue` | SSE `_stream()` |
| `run_end` / `heartbeat` | `_run_agent` | `asyncio.Queue` | SSE `_stream()` |
| `subagent_result` | `SubagentManager._announce_result` | `xadd chat_events:{chat_id}` | `_run_agent` XREAD → Queue → SSE |
| `message_complete` | `_run_agent`（process_direct 后） | `xadd chat_events:{chat_id}` | `_run_agent` XREAD → Queue → SSE |

### chat_router.py queue.put 调用清单（全部需迁移）

| 位置 | 当前调用 | 处理 |
|---|---|---|
| `_run_agent on_stream`（~592） | `queue.put({"type": "message_delta", ...})` | → `xadd_event` |
| `_run_agent on_progress`（~596） | `queue.put({"type": "tool_hint", ...})` | → `xadd_event` |
| `_run_agent on_tool_call`（~608） | `queue.put({"type": "tool_call", ...})` | → `xadd_event` |
| `_run_agent wait loop heartbeat`（~689） | `queue.put({"type": "heartbeat"})` | → `xadd_event` |
| `_run_agent 正常结束`（~709） | `queue.put({"type": "run_end", "status": "completed", ...})` | → `xadd_event` |
| `_run_agent 异常结束`（~712） | `queue.put({"type": "run_end", "status": "failed", ...})` | → `xadd_event` |
| `_run_agent finally sentinel`（~714） | `queue.put(None)` | **删除** |
| `_run_simple_rag token`（~787） | `queue.put({"type": "message_delta", ...})` | → `xadd_event` |
| `_run_simple_rag 正常结束`（~792） | `queue.put({"type": "run_end", "status": "completed", ...})` | → `xadd_event` |
| `_run_simple_rag 异常结束`（~797） | `queue.put({"type": "run_end", "status": "failed", ...})` | → `xadd_event` |
| `_run_simple_rag finally sentinel`（~799） | `queue.put(None)` | **删除** |

### chat_router.py 其他需删除/替换的调用

| 位置 | 当前 | 处理 |
|---|---|---|
| `~381` | `agent_loop = await _get_web_loop(...)` | **删除**，loop 在 worker 进程构建 |
| `~405-406` | `queue = asyncio.Queue()` + `run_queues[run_id] = queue` | **删除** |
| `~409-415` | `await _redis.set(RedisKeys.run_chat(run_id), chat_id)` | **删除** |
| `~419-450` | `asyncio.create_task(_run_simple_rag/run_agent)` | → `arq_pool.enqueue_job(...)` |
| `~481` | `run_queues.get(run_id)` | **删除**，统一 XREAD |
| `~486` | `redis.get(RedisKeys.run_chat(run_id))` | **删除** |
| `~495` | `xread_next(redis, chat_id, ...)` replay 分支 | **删除**，整个分支替换为 XREAD |
| `~587` | `stream_last_id = await get_last_id(redis, _chat_id)` | **删除** |
| `~666` | `xadd_event(redis, _chat_id, {"type": "message_complete"})` | **删除** |
| `~676` | `xread_next(redis, _chat_id, ...)` relay loop | **删除** |
| `~715` | `run_queues.pop(str(run_id), None)` | **删除** |
| `~717` | `redis.delete(RedisKeys.run_chat(str(run_id)))` | **删除** |
| `~800` | `run_queues.pop(str(run_id), None)` | **删除** |

### stream.py 签名变更后调用方全清单

`chat_id: str` → `stream_key: str`，调用方必须传完整 key：

| 文件 | 当前调用 | 改后 |
|---|---|---|
| `chat_router.py:587` | `get_last_id(redis, _chat_id)` | **DELETE** |
| `chat_router.py:666` | `xadd_event(redis, _chat_id, ...)` | **DELETE** |
| `chat_router.py:676` | `xread_next(redis, _chat_id, ...)` | **DELETE** |
| `chat_router.py:495` | `xread_next(redis, chat_id, ...)` | **DELETE**（整个 replay 分支删掉） |
| `subagent.py:241` (非 web) | `xadd_event(redis, origin["chat_id"], ...)` | → `xadd_event(redis, RedisKeys.chat_events(origin["chat_id"]), ...)` |
| `worker.py` (新) | 所有新增调用 | 直接传 `RedisKeys.run_events(run_id)` 或 `RedisKeys.chat_events(chat_id)` |

### Redis Key 变化

| Key | 现状 | 新状态 |
|---|---|---|
| `asyncio.Queue`（进程内存） | 主流事件载体 | **废弃** |
| `run_events:{run_id}` | 不存在 | **新增**，TTL 86400s |
| `run_chat:{run_id}` | 映射 run_id→chat_id | **废弃** |
| `chat_events:{chat_id}` | 子 Agent + message_complete | **部分保留**：非 web channel 的子 Agent 结果仍写这里 |
| `cancel:{session_key}` | 断连信号 | 不变 |
| `pending:{session_key}` | 子 Agent 计数 | 不变 |
| `job:{job_id}` | 幂等锁 | 不变 |

### 关键决策

- **job 幂等锁**：保留原 `job:{job_id}` 逻辑，不用 ARQ 原生 `_job_id`。原因：ARQ 的 `_job_id` 去重无法返回已有 run_id，当前 `create_run` 在命中时返回 `{run_id, status: "dedup"}`
- **Redis 连接**：ARQ 用独立 `RedisSettings` 连接池，不复用 `decode_responses=True` 单例；worker 内部用项目的 `get_redis()` 做业务操作
- **AgentLoop 构建**：提取 `_build_agent_loop(uid, ctx, ...)` 供 worker 使用，不依赖 `app.state`
- **`_run_simple_rag`**：也迁移到 `worker.py`，作为 `run_agent_job` 里的 `rag_mode == "simple"` 分支

---

## 3. 实现计划

### Phase 0：依赖 + 基础设施（无前置）

#### 0-A `backend/pyproject.toml`

```toml
"arq>=0.26.0,<1.0.0",
```

#### 0-B `backend/nanobot/bus/redis_keys.py`

在 `chat_events` 定义之后添加：

```python
# Run events stream — cross-process event delivery, 24h TTL
RUN_EVENTS_TTL = 86400

@staticmethod
def run_events(run_id: str) -> str:
    return f"run_events:{run_id}"
```

#### 0-C `backend/nanobot/bus/stream.py`

三个函数的 `chat_id: str` 参数改为 `stream_key: str`，内部直接 `key = stream_key`（删掉 `RedisKeys.chat_events` 调用和 import）：

```python
# 改前
async def xadd_event(redis, chat_id: str, event: dict) -> None:
    key = RedisKeys.chat_events(chat_id)

# 改后
async def xadd_event(redis, stream_key: str, event: dict) -> None:
    key = stream_key
```

同样修改 `xread_next` 和 `get_last_id`。

#### 0-D `backend/nanobot/bus/redis_monitor.py`

`_PREFIXES` 列表：删 `"run_chat:"`，加 `"run_events:"`

---

### Phase 1：新建 worker.py 框架（前置：Phase 0）

**新文件**：`backend/nanobot/worker.py`

```python
"""ARQ Worker 入口 — Agent 执行进程。

启动方式：
    arq nanobot.worker.WorkerSettings
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from arq.connections import RedisSettings

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


async def startup(ctx: dict) -> None:
    from nanobot.storage.database import get_session_factory
    from nanobot.cli.commands import build_loop_config   # Phase 2-A 提取
    ctx["session_factory"] = await get_session_factory()
    ctx["loop_config"] = await build_loop_config()
    ctx["rag_settings"] = ctx["loop_config"].get("rag_settings")


async def shutdown(ctx: dict) -> None:
    if engine := ctx.get("engine"):
        await engine.dispose()


async def run_agent_job(ctx: dict, *, run_id: str, session_key: str,
                        content: str, uid: str,
                        rag_mode: str = "agentic",
                        kb_id: str | None = None,
                        skill_names: list[str] | None = None,
                        agent_id: str | None = None,
                        agent_override: dict | None = None,
                        custom_persona: str | None = None,
                        harness: dict | None = None,
                        agents_registry: list[dict] | None = None,
                        agent_kb_id: str | None = None,
                        job_id: str | None = None) -> None:
    """ARQ job：在 Worker 进程执行 Agent，事件写 run_events:{run_id}。"""
    ...  # Phase 2 填充


class WorkerSettings:
    functions = [run_agent_job]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10
    job_timeout = 1800
    keep_result = 3600
```

---

### Phase 2：worker.py 核心逻辑（前置：Phase 1）

#### 2-A 提取 `build_loop_config`（`nanobot/cli/commands.py`）

把当前 `main()` 里组装 `loop_config` dict 的逻辑提取为顶层函数，供 web server lifespan 和 worker startup 共用：

```python
async def build_loop_config() -> dict:
    """构建 AgentLoop 所需配置 dict。"""
    ...
```

#### 2-B `_build_agent_loop` helper（`worker.py` 内部）

```python
async def _build_agent_loop(uid: str, ctx: dict,
                             model_override: str | None = None,
                             agent_model: str | None = None):
    from nanobot.agent.loop import AgentLoop
    from nanobot.session.manager import SessionManager
    from nanobot.providers.model_factory import ModelFactory, ModelRole
    from nanobot.storage.repositories.user_settings_repo import UserSettingsRepository

    cfg = ctx["loop_config"]
    factory = ctx["session_factory"]
    user_cfg = await UserSettingsRepository(factory).get(uid)
    model = model_override or agent_model or (user_cfg.model if user_cfg else None) or cfg.get("model")

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
    provider = _build_provider(spec) if spec.api_key else cfg["provider"]

    return AgentLoop(
        bus=cfg["bus"], provider=provider, workspace=ws, model=model,
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
```

#### 2-C `run_agent_job` 完整实现

Agent path（原 `_run_agent` 迁移）：

```python
async def run_agent_job(ctx, *, run_id, session_key, content, uid, ...):
    from nanobot.bus.redis_keys import RedisKeys
    from nanobot.bus.stream import xadd_event
    from nanobot.storage.repositories.run_repo import RunRepository
    from nanobot.bus.redis_client import get_redis

    redis = get_redis()
    run_stream_key = RedisKeys.run_events(run_id)
    run_repo = RunRepository(ctx["session_factory"])
    start = datetime.now(timezone.utc)

    if job_id:
        try:
            await redis.set(RedisKeys.job(job_id), run_id)
        except Exception:
            pass

    await run_repo.update(run_id, status="running", started_at=start)

    async def _check_cancel() -> bool:
        try:
            return bool(await redis.exists(RedisKeys.cancel(session_key)))
        except Exception:
            return False

    try:
        if rag_mode == "simple" and kb_id:
            await _run_simple_rag_job(...)   # Phase 2-D
            return

        # Agent path
        loop = await _build_agent_loop(uid, ctx, model_override=(agent_override or {}).get("model"))
        tool_calls_log = []

        async def on_stream(delta):
            await xadd_event(redis, run_stream_key, {"type": "message_delta", "chunk": delta})

        async def on_progress(text, *, tool_hint=False):
            if tool_hint:
                await xadd_event(redis, run_stream_key, {"type": "tool_hint", "content": text})

        async def on_tool_call(record):
            tool_calls_log.append(record)
            raw = record.get("output")
            summary = (raw.get("text") or raw.get("content") or str(raw))[:300] if isinstance(raw, dict) else str(raw or "")[:300]
            await xadd_event(redis, run_stream_key, {
                "type": "tool_call",
                "name": record.get("name"), "input": record.get("input"),
                "output_summary": summary, "status": record.get("status", "success"),
            })

        # kb_bindings / kb_map 构建（与原 _run_agent 相同逻辑）
        ...

        _chat_id = session_key.split(":", 1)[-1]
        await loop.process_direct(
            content, session_key=session_key, channel="web", chat_id=_chat_id,
            run_id=run_id,   # Phase 5 新增
            on_stream=on_stream, on_progress=on_progress, on_tool_call=on_tool_call,
            skill_names=skill_names, agent_id=agent_id, agent_override=agent_override,
            custom_persona=custom_persona, harness=harness,
            agents_registry=agents_registry, kb_bindings=kb_bindings, kb_map=kb_map,
        )

        # 等待子 Agent — 纯 SCARD 轮询（子 Agent 直接写 run_events，SSE 自己读，无需 relay）
        _MAX_WAIT, _waited = 1800, 0
        while _waited < _MAX_WAIT:
            if await _check_cancel(): break
            try:
                _pending = await redis.scard(RedisKeys.pending(session_key))
            except Exception:
                break
            if not _pending: break
            await asyncio.sleep(5)
            _waited += 5
            if _waited % 30 == 0:
                await xadd_event(redis, run_stream_key, {"type": "heartbeat"})

        finished = datetime.now(timezone.utc)
        duration_ms = int((finished - start).total_seconds() * 1000)
        await run_repo.update(run_id, status="completed", finished_at=finished,
                              duration_ms=duration_ms, model_used=loop.model,
                              tokens_used={...}, tool_calls=tool_calls_log)
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "completed", "duration_ms": duration_ms})

    except Exception as e:
        await run_repo.update(run_id, status="failed",
                              finished_at=datetime.now(timezone.utc), error_message=str(e))
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "failed", "error": str(e)})
    finally:
        try:
            if job_id:
                await redis.delete(RedisKeys.job(job_id))
            await redis.delete(RedisKeys.cancel(session_key))
            # 注意：不再有 run_queues.pop 和 run_chat DEL
        except Exception:
            pass
```

#### 2-D `_run_simple_rag_job` helper（原 `_run_simple_rag` 迁移）

所有 `queue.put(...)` 改为 `await xadd_event(redis, run_stream_key, ...)`，删除 `queue.put(None)` 和 `run_queues.pop`。

---

### Phase 3：chat_router.py（前置：Phase 2）

#### 3-A `create_run`（POST /api/runs）

删除：
- `agent_loop = await _get_web_loop(...)` （~381）
- `queue = asyncio.Queue()` + `run_queues[run_id] = queue` （~405-406）
- `await _redis.set(RedisKeys.run_chat(...), chat_id)` 整块（~409-415）
- `asyncio.create_task(_run_agent/simple_rag(...))` 两个分支（~419-450）

替换为：
```python
await request.app.state.arq_pool.enqueue_job(
    "run_agent_job",
    run_id=str(run_id),
    session_key=session_key,
    content=body.content,
    uid=uid,
    rag_mode=body.rag_mode,
    kb_id=body.kb_id,
    skill_names=skill_names,
    agent_id=str(conv.agent_id) if conv.agent_id else None,
    agent_override=agent_override or None,
    custom_persona=custom_persona,
    harness=agent_harness or None,
    agents_registry=agents_registry or None,
    agent_kb_id=agent_kb_id,
    job_id=_job_id,
)
```

#### 3-B `run_events`（GET /api/runs/{run_id}/events）

整个函数替换为纯 XREAD 循环（删除 queue 分支、replay 分支、run_chat GET）：

```python
@router.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, request: Request,
                     last_id: str = "0-0", uid=Depends(get_current_user)):
    from nanobot.bus.redis_keys import RedisKeys
    from nanobot.bus.stream import xread_next

    run = await _get_run_or_404(run_id, uid, request)
    redis = request.app.state.redis
    stream_key = RedisKeys.run_events(run_id)
    _session_key = f"web:{run.conversation_id}"

    async def _stream():
        cursor = last_id
        _normal_exit = False
        try:
            while True:
                events, cursor = await xread_next(redis, stream_key, cursor, timeout_ms=5000)
                for ev in events:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev.get("type") == "run_end":
                        _normal_exit = True
                        return
        finally:
            if not _normal_exit:
                try:
                    await redis.set(RedisKeys.cancel(_session_key), "1")
                except Exception:
                    pass

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

断线重连：客户端带 `?last_id=<上次cursor>` 从断点续读，24h 内有效。

#### 3-C 删除 `_run_agent` 函数（lines ~540-723）

#### 3-D 删除 `_run_simple_rag` 函数（lines ~725-800）

---

### Phase 4：main.py（前置：Phase 3）

```python
# lifespan startup 添加
from arq import create_pool
from arq.connections import RedisSettings as ArqRedisSettings
from nanobot.bus.redis_client import REDIS_URL
app.state.arq_pool = await create_pool(ArqRedisSettings.from_dsn(REDIS_URL))

# lifespan shutdown 添加
await app.state.arq_pool.aclose()

# 删除
app.state.run_queues = {}
```

---

### Phase 5：run_id 透传链路（前置：Phase 2）

**目的**：让 subagent 知道当前 run_id，从而把结果写到 `run_events:{run_id}` 而非 `chat_events:{chat_id}`。

**完整链路：**

```
worker.py run_agent_job(run_id=run_id)
  └─ loop.process_direct(..., run_id=run_id)        ← 新增参数
       └─ loop._process_message(..., run_id=run_id)  ← 新增参数
            └─ loop._set_tool_context(channel, chat_id, run_id=run_id) ← 新增参数
                 └─ SpawnTool.set_context(channel, chat_id, run_id)    ← 新增参数，存 self._run_id
                      └─ SpawnTool.execute()
                           └─ self._manager.spawn(..., run_id=self._run_id)
                                └─ SubagentManager.spawn(run_id=run_id) ← 新增参数，存入 origin
                                     └─ _announce_result(..., origin)
```

**各文件改动：**

`agent/loop.py`：
- `process_direct(..., run_id: str | None = None)` → 透传给 `_process_message`
- `_process_message(..., run_id: str | None = None)` → 透传给 `_set_tool_context`
- `_set_tool_context(channel, chat_id, run_id: str | None = None)` → 传给 `SpawnTool.set_context`

`agent/tools/spawn.py`：
- `set_context(self, channel, chat_id, run_id: str | None = None)` → `self._run_id = run_id`
- `execute(self, ...)` → `self._manager.spawn(..., run_id=self._run_id)`

`agent/subagent.py`：
- `spawn(self, task, ..., run_id: str | None = None)` → `origin = {"channel": ..., "chat_id": ..., "run_id": run_id}`
- `_announce_result` web 分支：
  ```python
  stream_key = (RedisKeys.run_events(origin["run_id"]) if origin.get("run_id")
                else RedisKeys.chat_events(origin["chat_id"]))  # CLI fallback
  await xadd_event(redis, stream_key, {"type": "subagent_result", ...})
  ```
- `_announce_result` 非 web 分支（stream.py 签名变更，必须同步更新）：
  ```python
  # 改前（line 241）：xadd_event(redis, origin["chat_id"], {...})
  # 改后：
  await xadd_event(redis, RedisKeys.chat_events(origin["chat_id"]), {...})
  ```

---

## 4. 文件改动一览

| 文件 | 性质 | Phase |
|---|---|---|
| `backend/pyproject.toml` | 添加 `arq` 依赖 | 0 |
| `backend/nanobot/bus/redis_keys.py` | 新增 `run_events()` + `RUN_EVENTS_TTL` | 0 |
| `backend/nanobot/bus/stream.py` | `chat_id` → `stream_key`；删 RedisKeys import | 0 |
| `backend/nanobot/bus/redis_monitor.py` | prefix：删 `run_chat:`，加 `run_events:` | 0 |
| `backend/nanobot/cli/commands.py` | 提取 `build_loop_config()` 顶层函数 | 2-A |
| `backend/nanobot/worker.py` | **新建**：WorkerSettings + run_agent_job + helpers | 1, 2 |
| `backend/nanobot/server/routers/chat_router.py` | create_run 改 enqueue；run_events 改 XREAD；删 _run_agent + _run_simple_rag | 3 |
| `backend/nanobot/server/main.py` | 初始化 arq_pool；删 run_queues | 4 |
| `backend/nanobot/agent/loop.py` | `process_direct` + `_process_message` + `_set_tool_context` 加 run_id | 5 |
| `backend/nanobot/agent/tools/spawn.py` | `set_context` + `execute` 加 run_id | 5 |
| `backend/nanobot/agent/subagent.py` | `spawn` + `_announce_result` 用 run_events；非 web 路径更新 stream_key | 5 |

---

## 5. 不变的部分

- `cancel:{session_key}`、`pending:{session_key}`、`job:{job_id}` 键逻辑不变
- CLI / 非 web channel 路径：无 run_id 时 subagent fallback 到 `chat_events:{chat_id}`
- `PendingReaper`、`RedisMonitor`、session 层、RAG 层全部不变
- `chat_events:{chat_id}` 保留用于非 web channel 的 subagent 结果

---

## 6. 内存估算

| 新增 Key 层 | 单位大小 | 活跃数 | 合计 |
|---|---|---|---|
| `run_events:{run_id}` | ~100B/event × 1000 events = ~100KB | 100 并发 run | ~10MB |

新增约 10MB，在现有 Redis 预算（512MB）内无压力。`asyncio.Queue` 从进程堆移出，进程内存略降。

---

## 7. 执行顺序

| Phase | 内容 | 前置 |
|---|---|---|
| 0 | pyproject + redis_keys + stream.py + monitor | 无 |
| 1 | 新建 worker.py 框架 | Phase 0 |
| 2-A | 提取 `build_loop_config` | Phase 1 |
| 2-B/C/D | worker.py 核心逻辑（agent + simple_rag） | Phase 2-A |
| 3 | chat_router.py（enqueue + XREAD + 删函数） | Phase 2 |
| 4 | main.py（arq_pool；删 run_queues） | Phase 3 |
| 5 | run_id 透传链路（loop + spawn + subagent） | Phase 2 |

---

## 8. 启动方式

```bash
uvicorn nanobot.server.main:app --host 0.0.0.0 --port 8000  # web 进程
arq nanobot.worker.WorkerSettings                            # worker 进程
```

两个进程通过 Redis 通信：web enqueue，worker 执行，SSE XREAD 事件流。

---

## 9. 验证清单

1. `arq nanobot.worker.WorkerSettings` 启动无报错
2. 发消息，SSE 收到 `message_delta` → `run_end`
3. `redis-cli XRANGE run_events:{run_id} - +` 确认全部事件写入
4. 断开 SSE，带 `?last_id=<cursor>` 重连，从断点续读
5. 触发子 Agent（spawn），`subagent_result` 出现在 `run_events:{run_id}`
6. CLI 模式发消息，子 Agent 结果正常（走 `chat_events` fallback）
7. `grep -r "run_queues" backend/` → 0 匹配
8. `grep -r "asyncio.create_task.*_run_agent" backend/` → 0 匹配
9. `grep -r "run_chat" backend/` → 0 匹配（除注释外）
