# Redis SDD — Phase 4: 遗留问题收尾

> 基于对 Phase 0-3 实现的实际审查，梳理出两个需要执行的任务。
> 编写日期: 2026-06-15

---

## 问题清单

| 编号 | 问题 | 状态 |
|---|---|---|
| 4-A | Pub/Sub subscriber 未实现 | **关闭** — 选方案 2，接受 TTL 自然失效，SDD 已加 accepted risk |
| 4-B | pending_reaper 仍用 OBJECT IDLETIME，被 RedisMonitor SCAN 持续重置 | **待执行** |
| 4-C | Redis Sentinel 未落地 | **待确认** — 需确认是否即将上生产 |
| 4-D | 控制信号 key 缺 uid | **关闭** — 无 group chat 场景，R5 不适用 |

---

## 4-A: 已关闭

当前 `_cache` 是 turn-scoped，不存在多 worker 一致性问题。引入持久 _cache 是明确的架构决策，到时候再补 subscriber 有足够机会。接受 TTL 自然失效（agent 1800s、kb 600s），零实现成本。

已在 SDD §7 Phase 1-D 加 accepted risk 说明。

---

## 4-B: pending_reaper IDLETIME → member 时间戳（功能性 bug）

### 问题根因

`pending_reaper.py`（Phase 2-D）用 `OBJECT IDLETIME key > 7200` 判断孤儿 key 年龄。
`redis_monitor.py`（Phase 3-B）每 300s `SCAN + MEMORY USAGE`，MEMORY USAGE 会重置 IDLETIME。
两个模块互相干扰：孤儿 `pending:*` key 的 IDLETIME 被持续重置，reaper 判断永远失败，孤儿 key 永远不清理。

Phase 2-D 计划里已经写了"改用 member 时间戳"，实际实现没有跟上。

### 修法

**文件**: `backend/nanobot/agent/subagent.py`

SADD 写入带 unix 时间戳的 member：

```python
# 修改前
await redis.sadd(pending_key, task_id)

# 修改后
ts = int(time.time())
await redis.sadd(pending_key, f"{task_id}:{ts}")
```

SREM 改为辅助函数按前缀匹配（处理旧格式 member 的兼容问题）：

```python
async def _remove_pending_member(redis, pending_key: str, task_id: str):
    members = await redis.smembers(pending_key)
    for m in members:
        if m == task_id or m.startswith(f"{task_id}:"):
            await redis.srem(pending_key, m)
            break
```

> Set 在最后一个 member 被 SREM 后自动删除 key，不需要显式 DEL。

**文件**: `backend/nanobot/bus/pending_reaper.py`

解析 member 里的时间戳，跳过旧格式（无时间戳）的 member：

```python
members = await redis.smembers(key)
now = int(time.time())
stale = []
for m in members:
    try:
        ts = int(m.rsplit(":", 1)[1])
        if now - ts > self._idle_threshold:
            stale.append(m)
    except (ValueError, IndexError):
        pass  # 旧格式 member（不带时间戳），跳过不处理

# 只 SREM stale members
for m in stale:
    await redis.srem(key, m)
```

> 旧格式 member 不会被清理，等下次重启后旧 member 自然消失（不写新旧格式混合）。
> 需要同时在 `subagent.py` 的 SREM 路径里也兼容旧格式（见上方 `_remove_pending_member`）。

### 文件变更

| 操作 | 文件 |
|---|---|
| 修改 | `backend/nanobot/agent/subagent.py` |
| 修改 | `backend/nanobot/bus/pending_reaper.py` |

### 验证

```bash
# 写入旧格式 member（验证兼容性，不崩）
redis-cli SADD "pending:admin:test:C1" "old_task_id"

# 写入新格式但时间戳已过期的 member（验证清理逻辑）
redis-cli SADD "pending:admin:test:C1" "new_task:$(( $(date +%s) - 8000 ))"

# 等 reaper 下次运行
redis-cli SMEMBERS "pending:admin:test:C1"
# 期望：old_task_id 保留（跳过），new_task:... 被 SREM
```

---

## 4-C: Redis Sentinel（待确认）

**基础设施任务，无代码变动。** 需确认生产上线时间线后决定是否纳入本阶段。
