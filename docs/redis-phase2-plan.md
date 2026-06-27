# Redis SDD — Phase 2: Stability

> 基于 `redis-sdd.md` 的 Phase 2 实现计划。
> Phase 0（控制信号去 TTL）和 Phase 1（缓存核心功能）已完成。
> 编写日期: 2026-06-15

## 状态评估

Phase 1 实现质量较高，大部分 Redis 调用点已有 try/except 保护。

### 已覆盖（无需改动）

| 文件 | 说明 |
|---|---|
| `agent_repo.py` | 读/写/删缓存全部 try/except |
| `user_settings_repo.py` | 读/写/删缓存全部 try/except |
| `knowledge_repo.py` | 读/写/删缓存全部 try/except |
| `session/manager.py` | _redis_load / _redis_save 全部 try/except |
| `agent/memory.py` | Lua LTRIM try/except |
| `subagent.py` | SADD/SREM/EXISTS/xadd_event 全部 try/except |
| `hybrid_search.py` | embedding GET/SET try/except |
| `retrieval.py` | chunk HGETALL/HSET pipeline try/except |

### 关键设计决策

1. **reaper 年龄用 member 时间戳，不用 OBJECT IDLETIME** — Phase 3 的 MEMORY USAGE 会重置 IDLETIME
2. **xread_next/xadd_event 不加 try/except** — 异常传播到 _run_agent 外层，SSE 断连后客户端重连
3. **Line 369 降级打 WARNING 日志** — 标识"可能产生重复 run"风险

---

## Task 2-A: 补齐 try/except

### chat_router.py

| 行号 | 调用 | 降级策略 |
|---|---|---|
| 369 | `_redis.get(job_key(...))` | log WARNING "可能产生重复run"，允许通过 |
| 401 | `_redis.set(run_chat_key(...), chat_id)` | log warning，继续执行 |
| 471 | `redis.get(run_chat_key(...))` | chat_id=None → replay 降级 DB |
| 508 | `redis.set(cancel_key(...), "1")` | log warning，cancel 失效 |
| 554 | `redis.set(job_key(...), str(run_id))` | log warning，幂等性失效 |
| 631 | `xadd_event(redis, ...)` | **不加 try/except** → 异常传播到外层 |
| 638 | `redis.exists(cancel_key(...))` | return False（允许继续） |
| 645 | `redis.scard(pending_key(...))` | break（退出等待循环） |
| 676-679 | `redis.delete(...)` 三个 key | log warning，orphan 留待清扫 |

### stream.py

| 函数 | 策略 |
|---|---|
| `xadd_event` | **不加 try/except** → 异常传播到 _run_agent 外层 |
| `xread_next` | **不加 try/except** → 异常传播到 _run_agent 外层；SSE 断连后客户端重连 |
| `get_last_id` | **加 try/except** → 返回 `"0-0"`（在 try 块外调用） |

---

## Task 2-B: session/manager.py DB-only

**✅ 已实现，无需改动。** _redis_save log warning → DB save 继续。

---

## Task 2-C: Redis Sentinel / Managed Redis

**基础设施任务，不涉及代码变更。**

---

## Task 2-D: bus/pending_reaper.py + subagent.py

### 背景

OBJECT IDLETIME 被 Phase 3 MEMORY USAGE 重置 → 改用 member 嵌入时间戳。

### subagent.py 改动

- **SADD**: `{task_id}:{int(time.time())}` 而非裸 `{task_id}`
- **SREM**: 加辅助函数 `_remove_pending_member()`，SMEMBERS 匹配前缀后 SREM

### pending_reaper.py

```python
class PendingReaper:
    # interval=300s, idle_threshold=7200s
    # SCAN pending:*, 解析 member ts, 检查 stream 是否存在
    # 二者都满足 → SREM stale members (set 空则 DEL key)
```

### 集成

在 `server/main.py` 的 lifespan 中 start/stop。

---

## 文件变更清单

| 操作 | 文件 |
|---|---|
| 修改 | `backend/nanobot/server/routers/chat_router.py` |
| 修改 | `backend/nanobot/agent/subagent.py` |
| 修改 | `backend/nanobot/bus/stream.py` |
| 新建 | `backend/nanobot/bus/pending_reaper.py` |
| 修改 | `backend/nanobot/server/main.py` |

## 验证

1. **优雅降级**: 停掉 Redis → 发 chat 请求 → 返回 200，无 500
2. **Reaper**: orphan key（ts > 2h, no stream）→ 被删；active key → 保留
3. **chat_router**: 每个操作点 Redis 异常时吐 warning 日志，不抛异常
4. **SSE 断连**: xread_next 抛异常 → SSE 断开 → 客户端重连
5. **REDIS_URL=""**: session 写正常走 DB
