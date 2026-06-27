# Redis SDD — Phase 3: Observability

> 基于 `redis-sdd.md` 的 Phase 3 实现计划。
> Phases 0/1/2 已完成：控制信号去 TTL、Session/Config/RAG 三层缓存、稳定性兜底。
> 编写日期: 2026-06-15

---

## 目标

补齐监控盲区：

| 任务 | 目标 |
|---|---|
| 3-A | 后台轮询 Redis `INFO stats`，`evicted_keys` 增量 > 0 触发 WARNING |
| 3-B | 定期 SCAN + pipeline MEMORY USAGE，按 prefix 采样，写 JSON Lines |
| 3-C | Session / Config / RAG 缓存调用点加结构化 hit/miss 日志 |

---

## 关键设计决策

### 日志框架选择

| 包 | 框架 | 依据 |
|---|---|---|
| `bus/` | stdlib `logging.getLogger(__name__)` | 与 `pending_reaper.py` 一致 |
| `session/`, `storage/repositories/` | loguru `logger.bind(event=...)` | 与 `session/manager.py` 一致 |
| `rag/` | stdlib `logger.debug(..., extra={...})` | 与 `hybrid_search.py`, `retrieval.py` 已有 logger 一致 |

### metrics path 计算

```python
# redis_monitor.py 位于 backend/nanobot/bus/redis_monitor.py
# parents[0]=bus/, parents[1]=nanobot/, parents[2]=backend/
_DEFAULT_METRICS_PATH = Path(__file__).resolve().parents[2] / "logs" / "redis_metrics.jsonl"
# → backend/logs/redis_metrics.jsonl（与 traces.jsonl 同目录）
# 环境变量 REDIS_METRICS_PATH 优先，方便运维覆盖
```

### MEMORY USAGE 用 pipeline

每个 prefix 最多采样 50 个 key，12 个 prefix 最多 600 次调用。用 `pipeline(transaction=False)` 批量一次 round-trip，避免串行发送的 Redis 压力。

```python
async with redis.pipeline(transaction=False) as pipe:
    for key in sample_keys:
        pipe.memory_usage(key)
mem_results = await pipe.execute()
for mem in mem_results:
    if mem is not None:   # key 可能在 SCAN 和 USAGE 之间过期
        total_bytes += mem
```

### session L1 / Redis 事件分离

in-process dict 命中（纳秒级）和 Redis 命中（毫秒级）用不同 event 区分，便于计算 Redis 层实际命中率：

| 场景 | event |
|---|---|
| `self._cache` 命中 | `session_l1_hit` |
| Redis list/hash 命中 | `session_redis_hit` |
| 全部未命中（从 DB 加载） | `session_cache_miss` |

### `_write_metrics` 用 asyncio.to_thread

`_write_metrics` 是同步文件 IO，用 `asyncio.to_thread(self._write_metrics, ts, results)` 包装，避免 300s 写一次时短暂阻塞事件循环。

---

## 新建文件

### `backend/nanobot/bus/redis_monitor.py`

遵循 `pending_reaper.py` 模式：`class RedisMonitor` + `start()`/`stop()` + 单个 `asyncio.Task`。

```
RedisMonitor
  ├── start()              # asyncio.create_task(self._run())
  ├── stop()               # cancel + await
  ├── _run()               # 主循环：每 stats_interval 秒运行一次
  │                        # 用 time.monotonic() 控制 memory_interval
  ├── _check_stats()       # redis.info("stats") → evicted_keys delta → WARNING
  ├── _scan_memory()       # SCAN 12 prefix → pipeline MEMORY USAGE → to_thread write
  └── _write_metrics()     # 同步：path.open("a") 追加 JSON Lines
```

**JSON Lines schema（`logs/redis_metrics.jsonl`）：**

```json
{
  "timestamp": 1718438400.123,
  "type": "redis_memory_scan",
  "prefixes": [
    {"prefix": "session:msg:", "sampled_keys": 42, "total_sample_bytes": 172032, "avg_bytes_per_key": 4096},
    {"prefix": "embedding:",   "sampled_keys": 0,  "total_sample_bytes": 0,      "avg_bytes_per_key": 0}
  ]
}
```

prefix 恒为 12 条（无 key 时填零），schema 稳定。

---

## 修改文件

### `backend/nanobot/server/main.py`

在 PendingReaper import 后立即 import RedisMonitor（同在 lifespan 局部 import，与 PendingReaper 一致）：

```python
from nanobot.bus.pending_reaper import PendingReaper
from nanobot.bus.redis_monitor import RedisMonitor
```

lifespan 启动（PendingReaper.start 后）：
```python
app.state.redis_monitor = RedisMonitor()
await app.state.redis_monitor.start()
```

lifespan 关闭（pending_reaper.stop 后，redis.aclose 前）：
```python
await app.state.redis_monitor.stop()
```

app.state 初始化块：
```python
app.state.redis_monitor = None
```

### `backend/nanobot/session/manager.py`

`get_or_create()` — loguru 已在文件头 import，直接 bind：

```python
if key in self._cache:
    logger.bind(event="session_l1_hit",    cache_layer="session_cache").debug(...)
    return self._cache[key]
session = await self._redis_load(key)
if session is not None:
    logger.bind(event="session_redis_hit", cache_layer="session_cache").debug(...)
else:
    logger.bind(event="session_cache_miss", cache_layer="session_cache").debug(...)
    ...
```

### `backend/nanobot/storage/repositories/{agent,user_settings,knowledge}_repo.py`

三个文件各新增 `from loguru import logger`，在 HGETALL hit/miss 点插入 bind 日志：

| 文件 | hit event | miss event |
|---|---|---|
| `agent_repo.py` | `agent_cache_hit` | `agent_cache_miss` |
| `user_settings_repo.py` | `user_settings_cache_hit` | `user_settings_cache_miss` |
| `knowledge_repo.py` | `kb_meta_cache_hit` | `kb_meta_cache_miss` |

miss log 放在 `except Exception: pass` 后、DB 查询前。

### `backend/nanobot/rag/core/query_engine/hybrid_search.py`

embedding cache 块（约 line 915-936），用 stdlib `extra=`：

```python
# hit
logger.debug("embedding cache hit ...", extra={"event": "embedding_cache_hit", ...})

# miss — 放在 if precomputed_embedding is None and self.dense_retriever is not None: 内
logger.debug("embedding cache miss ...", extra={"event": "embedding_cache_miss", ...})
```

### `backend/nanobot/rag/mcp_server/tools/agentic/retrieval.py`

`_batch_fetch_chunks_cached()` line 56（uncached_ids 计算后），用 stdlib `extra=`：

```python
hit_count = len(cached)
miss_count = len(uncached_ids)
if hit_count:
    logger.debug("chunk cache: %d hit(s) ...", extra={"event": "chunk_cache_hit", ...})
if miss_count:
    logger.debug("chunk cache: %d miss(es) ...", extra={"event": "chunk_cache_miss", ...})
```

---

## hit/miss event 字段一览

| 文件 | event（hit） | event（miss） | logger |
|---|---|---|---|
| `session/manager.py` | `session_l1_hit` / `session_redis_hit` | `session_cache_miss` | loguru bind |
| `storage/repositories/agent_repo.py` | `agent_cache_hit` | `agent_cache_miss` | loguru bind |
| `storage/repositories/user_settings_repo.py` | `user_settings_cache_hit` | `user_settings_cache_miss` | loguru bind |
| `storage/repositories/knowledge_repo.py` | `kb_meta_cache_hit` | `kb_meta_cache_miss` | loguru bind |
| `rag/core/query_engine/hybrid_search.py` | `embedding_cache_hit` | `embedding_cache_miss` | stdlib extra= |
| `rag/mcp_server/tools/agentic/retrieval.py` | `chunk_cache_hit` | `chunk_cache_miss` | stdlib extra= |

---

## 文件变更清单

| 操作 | 文件 |
|---|---|
| 新建 | `backend/nanobot/bus/redis_monitor.py` |
| 修改 | `backend/nanobot/server/main.py` |
| 修改 | `backend/nanobot/session/manager.py` |
| 修改 | `backend/nanobot/storage/repositories/agent_repo.py` |
| 修改 | `backend/nanobot/storage/repositories/user_settings_repo.py` |
| 修改 | `backend/nanobot/storage/repositories/knowledge_repo.py` |
| 修改 | `backend/nanobot/rag/core/query_engine/hybrid_search.py` |
| 修改 | `backend/nanobot/rag/mcp_server/tools/agentic/retrieval.py` |

---

## 验证

### 3-A — eviction 告警
```bash
redis-cli CONFIG SET maxmemory 1mb
# 触发后恢复
redis-cli CONFIG SET maxmemory 512mb
# 日志中查找（message 包含 "eviction alert" 字样）
grep "eviction alert" logs/app.log
```

### 3-B — 内存快照
```bash
python -c "
import json
for line in open('backend/logs/redis_metrics.jsonl'):
    rec = json.loads(line)
    for p in rec['prefixes']:
        print(p['prefix'], p['sampled_keys'], 'keys,', p['total_sample_bytes'], 'bytes')
"
```

### 3-C — hit/miss 日志
```bash
# 发两次相同 chat 请求（第二次期望命中缓存）
grep "session_redis_hit\|agent_cache_hit" logs/app.log
```

### smoke test
```bash
cd backend
python -c "from nanobot.bus.redis_monitor import RedisMonitor; m = RedisMonitor(); print(m._metrics_path)"
```
