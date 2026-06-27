# Redis Phase 2 — QA Report

> 审查日期: 2026-06-15
> 审查方式: 静态代码阅读，逐条对照 `redis-phase2-plan.md`

---

## 总体结论

**P2 实现完整，无功能性 bug。** 发现 2 个小问题，均不影响正确性。

---

## Task 2-A: chat_router.py — try/except 覆盖

| 计划行为 | 实际代码位置 | 状态 |
|---|---|---|
| `GET(job)` 失败 → log WARNING + allow | L369-377 | ✅ |
| `SET(run_chat)` 失败 → log warning + continue | L409-415 | ✅ |
| `GET(run_chat)` 失败 → `chat_id=None` → replay 降级 DB | L485-488 | ✅ |
| `SET(cancel)` 失败 → log warning | L526-531 | ✅ |
| `SET(job)` 失败 → log warning + 幂等性失效 | L578-583 | ✅ |
| `xadd_event` — **不加** try/except | L665（在外层 try 内，异常传播到 run 失败处理） | ✅ |
| `EXISTS(cancel)` 失败 → return False | L618-622 | ✅ |
| `SCARD(pending)` 失败 → break 退出等待循环 | L679-683 | ✅ |
| `DELETE` 三个 key 失败 → log warning | L715-721 | ✅ |

## Task 2-A: stream.py

| 函数 | 计划 | 实际 | 状态 |
|---|---|---|---|
| `xadd_event` | 不加 try/except | 无 try/except，异常向上传播 | ✅ |
| `xread_next` | 不加 try/except | 无 try/except，异常向上传播 | ✅ |
| `get_last_id` | 加 try/except → 返回 `"0-0"` | L104-110 有 try/except + fallback | ✅ |

## Task 2-B: session/manager.py

已在 P1 实现，无需改动。✅

## Task 2-D: subagent.py + pending_reaper.py + main.py

| 项目 | 计划 | 实际 | 状态 |
|---|---|---|---|
| SADD member 格式 | `{task_id}:{unix_ts}` | `f"{task_id}:{int(time.time())}"` (L82) | ✅ |
| SREM 辅助函数 | `_remove_pending_member()` SMEMBERS + 前缀匹配 | L302-313，`member.startswith(task_id + ":")` | ✅ |
| 所有 SREM 调用点 try/except | 全覆盖 | CancelledError 路径 L200-204，Exception 路径 L211-216，web announce L236-252，non-web L272-277 | ✅ |
| PendingReaper: SCAN pending:* | ✅ | L69 | ✅ |
| PendingReaper: age guard | `now - ts < idle_threshold` | L110-111 | ✅ |
| PendingReaper: stream 存在性检查 | `EXISTS(chat_events:{chat_id})` | L114-116 | ✅ |
| PendingReaper: SREM + DEL empty key | ✅ | L73-76 | ✅ |
| main.py lifespan 集成 | start/stop | L44-46 start，L62 stop | ✅ |

---

## 发现的问题

### 问题 1 — 风格：`import logging` 在 except 块内（非 bug）

**位置**: `chat_router.py` L373、L412

```python
except Exception:
    existing_run_id = None
    import logging          # ← 在 except 块内 import
    logging.getLogger(__name__).warning(...)
```

同一函数的 L624 已经有 `import logging as _logging`，两处写法不一致。Python 缓存 import，不影响性能或正确性，纯风格问题。

**处置**: 可在后续清理时统一到函数头部 import，不需紧急修复。

### 问题 2 — 设计：error path 双重 SREM（有意，无需修改）

**位置**: `subagent.py` Exception 路径 L211-217 + `_announce_result` L250/L275

subagent 执行失败时：
1. except 块先调用 `_remove_pending_member`（crash-safety，防止 announce 自身失败导致 key 泄漏）
2. `_announce_result` 内再次调用 `_remove_pending_member`

第二次 SREM 目标 member 已不存在，Redis SREM 对不存在的 member 返回 0，是 no-op。代码注释已说明意图（"Crash-safety SREM before _announce_result in case announce itself fails"），属于有意设计。

**处置**: 无需修改。

---

## 小结

| 检查项 | 结果 |
|---|---|
| 所有计划 try/except 点 | ✅ 全覆盖 |
| stream.py 异常传播策略 | ✅ 正确 |
| subagent SADD/SREM 格式 | ✅ 正确 |
| PendingReaper 逻辑 | ✅ 正确 |
| main.py 生命周期集成 | ✅ 正确 |
| 功能性 bug | **无** |
| 风格问题 | 1 处，低优先级 |
