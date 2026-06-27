# Redis Phase 3 — QA Report

> 审查日期: 2026-06-15
> 审查方式: 静态代码阅读，逐条对照 `redis-phase3-plan.md` + 实现代码

---

## 总体结论

**P3 实现逻辑正确，功能完整。** 发现 1 个实施期 bug（import 缩进错误，已当场修复），2 个可接受的小问题。

---

## Task 3-A: RedisMonitor — eviction 告警

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| `redis.info("stats")` 取 evicted_keys | `redis_monitor.py:95-96` | ✅ |
| 首次调用设基线，不告警（`_last_evicted = None`） | `redis_monitor.py:97` | ✅ |
| delta > 0 → `logger.warning(...eviction alert...)` | `redis_monitor.py:100-106` | ✅ |
| warning 消息包含 "eviction alert" 字样（与验证 grep 对齐） | `"RedisMonitor: eviction alert — ..."` | ✅ |
| Redis 异常 → except → logger.exception | `redis_monitor.py:79-80` | ✅ |

---

## Task 3-B: RedisMonitor — 内存快照

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| 12 个 prefix 全覆盖 | `_PREFIXES` 列表，12 项 | ✅ |
| SCAN 每 prefix 至多 50 key | `sample_keys[:self._sample_size]` | ✅ |
| pipeline(transaction=False) 批量 MEMORY USAGE | `redis_monitor.py:132-135` | ✅ |
| `if mem is not None` 防过期 key | `redis_monitor.py:137` | ✅ |
| `asyncio.to_thread` 包装文件 IO | `redis_monitor.py:150` | ✅ |
| `path.open("a")` 追加 JSON Lines | `redis_monitor.py:162-163` | ✅ |
| 每 prefix 结果含 `sampled_keys=0` 时仍写入 | `results.append(...)` 无条件执行 | ✅ |
| 路径 `parents[2]` → `backend/logs/` | `redis_monitor.py:40`，smoke test 验证输出正确 | ✅ |
| 环境变量 `REDIS_METRICS_PATH` 优先 | `redis_monitor.py:56-57` | ✅ |
| `_run()` 单任务双间隔（monotonic 跟踪） | `redis_monitor.py:75-90` | ✅ |

---

## Task 3-A/B: main.py 集成

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| import 在 lifespan 内（与 PendingReaper 一致） | `main.py:45` | ✅ |
| lifespan 启动：`RedisMonitor().start()` | `main.py:50-51` | ✅ |
| lifespan 关闭：`redis_monitor.stop()` 在 `redis.aclose()` 前 | `main.py:68-69` | ✅ |
| `app.state.redis_monitor = None` 初始化 | `main.py:79` | ✅ |

---

## Task 3-C: hit/miss 结构化日志

### session/manager.py

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| L1 命中 → `session_l1_hit` | `manager.py:203` | ✅ |
| Redis 命中 → `session_redis_hit` | `manager.py:209` | ✅ |
| 全未命中 → `session_cache_miss` | `manager.py:213` | ✅ |
| loguru `logger.bind(event=..., cache_layer=...)` | ✅ | ✅ |
| loguru 已在文件头 import，无需新增 | `manager.py:12` | ✅ |

### storage/repositories/agent_repo.py

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| `from loguru import logger` | `agent_repo.py:11` | ✅ |
| HGETALL hit → `agent_cache_hit` bind | `agent_repo.py:70-72` | ✅ |
| except 后 → `agent_cache_miss` bind | `agent_repo.py:77-79` | ✅ |

### storage/repositories/user_settings_repo.py

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| `from loguru import logger` | `user_settings_repo.py:11` | ✅ |
| HGETALL hit → `user_settings_cache_hit` bind | ✅ | ✅ |
| except 后 → `user_settings_cache_miss` bind | ✅ | ✅ |

### storage/repositories/knowledge_repo.py

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| `from loguru import logger` | `knowledge_repo.py:11` | ✅ |
| HGETALL hit → `kb_meta_cache_hit` bind | ✅ | ✅ |
| except 后 → `kb_meta_cache_miss` bind | ✅ | ✅ |

### rag/core/query_engine/hybrid_search.py

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| hit log 在 `if cached:` 内，`precomputed_embedding` 赋值后 | `hybrid_search.py:918-922` | ✅ |
| miss log 在 `if precomputed_embedding is None and self.dense_retriever is not None:` 内 | `hybrid_search.py:928-932` | ✅ |
| stdlib `extra={"event": ..., "cache_layer": ...}` | ✅ | ✅ |
| 无新增 import | ✅ | ✅ |

### rag/mcp_server/tools/agentic/retrieval.py

| 计划行为 | 实际代码 | 状态 |
|---|---|---|
| `hit_count` / `miss_count` 在 try 块内、uncached_ids 计算后 | `retrieval.py:57-72` | ✅ |
| `if hit_count:` / `if miss_count:` 各自独立（两者可同时触发） | ✅ | ✅ |
| `extra` 包含 `hit_count` / `miss_count` 字段 | `retrieval.py:64,71` | ✅ |
| except 后 cached={}/uncached_ids=all，不触发日志（在 except 外） | ✅ | ✅ |

---

## 发现的问题

### Bug 1 — main.py import 缩进错误（已修复）

**严重程度**: Critical（会导致 Python IndentationError / SyntaxError）

**现象**: `from nanobot.bus.redis_monitor import RedisMonitor` 被 Edit 工具以 0 缩进插入，而该行应在 `lifespan` 函数内（需 8 空格缩进）。

**原因**: Edit 工具的 `new_string` 中第二行未带缩进，导致 Python 解析时脱离函数体。

**修复**: 已修正为 8 空格缩进，`python -c "import ast; ..."` smoke test 通过。

**后续**: 此类多行 Edit 需在 new_string 内每行手动保持缩进，工具不会自动对齐。

---

### 问题 2 — repo miss log 在 Redis 异常时也触发（可接受）

**位置**: `agent_repo.py:77`、`user_settings_repo.py` 同位置、`knowledge_repo.py` 同位置

**现象**:

```python
try:
    cached = await get_redis().hgetall(cache_key)
    if cached:
        logger.bind(event="agent_cache_hit", ...).debug(...)
        return _agent_from_hash(cached)
except Exception:
    pass

logger.bind(event="agent_cache_miss", ...).debug(...)   # ← 也在 Redis 异常路径触发
```

当 Redis 不可用时，except 捕获异常后继续执行，miss log 被记录，但实际是 Redis 错误而非 cache miss。

**影响**: 命中率统计会将 Redis 错误计为 miss，命中率偏低但不影响功能正确性（行为相同：fallback 到 DB）。

**处置**: 如需精确区分，可引入 `redis_error` 事件。当前不必要，保留现状。

---

### 问题 3 — 首次内存扫描在启动时立即触发（可接受）

**位置**: `redis_monitor.py:75`（`last_memory_check = 0.0`）

**现象**: `time.monotonic()` 启动时返回系统运行秒数（通常数千秒），必然 `>= 300`，导致第一次 `_run()` 迭代就执行 `_scan_memory()`。启动后 Redis 尚无热数据，12 个 prefix 全部返回 `sampled_keys=0`，写入一条全零记录。

**影响**: 额外一次无意义 I/O，JSON Lines 第一条记录全为零。功能完全正常。

**处置**: 可在 `__init__` 改为 `self._last_memory_check = time.monotonic()` 延迟首次扫描 300s。当前不影响功能，不需紧急修复。

---

## 小结

| 检查项 | 结果 |
|---|---|
| 3-A eviction 告警逻辑 | ✅ |
| 3-B pipeline MEMORY USAGE | ✅ |
| 3-B `if mem is not None` 防空值 | ✅ |
| 3-B asyncio.to_thread 文件 IO | ✅ |
| 3-B parents[2] 路径正确 | ✅ smoke test 验证 |
| 3-C session L1/Redis 事件分离 | ✅ |
| 3-C 三个 repo loguru bind | ✅ |
| 3-C embedding/chunk stdlib extra= | ✅ |
| main.py lifespan 集成 | ✅（修复缩进 bug 后） |
| main.py 语法 | ✅ ast.parse 通过 |
| 功能性 bug | **无**（修复后） |
| 可接受小问题 | 2 处，均无需紧急处理 |
