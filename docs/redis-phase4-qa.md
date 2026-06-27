# Redis SDD — Phase 4 QA 报告

> 编写日期: 2026-06-15

---

## 4-A：已关闭

无代码变动，接受 TTL 自然失效（agent 1800s / kb 600s）。

**结论**：✅ 关闭，无需验证。

---

## 4-B：pending_reaper IDLETIME → member 时间戳

### 变更文件

| 文件 | 变更内容 |
|---|---|
| `backend/nanobot/agent/subagent.py` | SADD 写带时间戳 member；`_remove_pending_member` 兼容旧格式 |
| `backend/nanobot/bus/pending_reaper.py` | 解析 member 时间戳判断年龄，不再依赖 OBJECT IDLETIME |

### 代码审查

**SADD 路径（subagent.py:82）**

```python
await get_redis().sadd(pending_key(session_key), f"{task_id}:{int(time.time())}")
```

- 格式 `{task_id}:{unix_ts}` 正确，时间戳精度秒级，与 reaper 的阈值（7200s）匹配。✅

**SREM 路径（subagent.py:311）**

```python
if member == task_id or member.startswith(task_id + ":"):
```

- `member == task_id`：兼容旧格式（无时间戳）member，部署切换期间写入的旧数据可被正常清理。✅
- `member.startswith(task_id + ":")` ：匹配新格式。`:` 作为分隔符，避免 task_id 前缀误匹配。✅
- 调用点覆盖完整：`_announce_result` web 路径、非 web 路径、`CancelledError` 异常路径，三处均调用同一方法，无遗漏。✅

**reaper 路径（pending_reaper.py:101-116）**

```python
parts = member.rsplit(":", 1)
if len(parts) != 2:
    continue  # 旧格式跳过
ts = int(parts[1])
if now - ts < self._idle_threshold:
    continue  # age guard
stream_exists = await redis.exists(RedisKeys.chat_events(chat_id))
if not stream_exists:
    stale.append(member)
```

- 旧格式 member（`len(parts) != 2`）跳过不清理，等任务正常结束时由 `_remove_pending_member` 负责。✅
- age guard（7200s）优先，防止误清理活跃任务的孤儿记录。✅
- 双重验证（age + stream 不存在）降低误删率。✅

### 行为一致性验证

| 场景 | SREM 路径 | reaper 路径 | 预期结果 |
|---|---|---|---|
| 新格式 member，任务正常完成 | `startswith` 匹配，SREM ✅ | — | member 被清理 |
| 旧格式 member，任务正常完成 | `== task_id` 匹配，SREM ✅ | — | member 被清理 |
| 新格式 member，进程崩溃成孤儿 | 无法执行 | age > 7200 且 stream 不存在，SREM ✅ | reaper 清理 |
| 旧格式 member，进程崩溃成孤儿 | 无法执行 | `len(parts) != 2` 跳过 | 重启后自然消失 |
| 新格式 member，任务仍在运行 | — | age < 7200，跳过 ✅ | 不误删 |

**结论**：✅ 实现完整，逻辑正确，与 Plan 4-B 设计一致。

---

## 4-C：Redis Sentinel

**结论**：✅ 单机部署不适用，关闭。Sentinel 解决主从高可用场景，单实例无主从结构，无意义。生产单机建议：开启 `appendonly yes` + systemd 自动重启。

---

## 4-D：已关闭

无 group chat 场景，控制信号 key 无需加 uid。

**结论**：✅ 关闭，无需验证。

---

## 总结

Phase 4 唯一代码变动（4-B）已实现并通过审查。所有路径覆盖完整，旧格式兼容逻辑正确，reaper 与 SREM 路径分工清晰，不存在竞态或误删风险。
