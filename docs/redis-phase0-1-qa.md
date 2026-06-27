# Redis 实现 QA 报告

> 审核范围：Phase 0 + Phase 1 全部改动（redis_keys.py → retrieval.py）

---

## 已修复 Bug

### B1 — agent_repo 缓存缺失 `version` / `provider`

**严重性**：高（API 返回错误数据）

**发现**：`_agent_to_card()` (`agent_router.py:45,50`) 读取 `agent.version` 和 `agent.provider`，但 `_agent_to_hash` / `_agent_from_hash` 未包含这两字段。缓存命中时 API 返回 `"1.0.0"`（ORM 默认值）和 `null` 而非真实值。

**修复**：在 `_agent_to_hash` 和 `_agent_from_hash` 中补充 `version` 和 `provider`。

### B2 — knowledge_repo 缓存缺字段导致 API 返回 `null`

**严重性**：高（API 返回错误/空白数据）

**发现**：`KnowledgeRepository.get()` 同时服务于：
- **热路径**（chat_router — 少量字段：`name`, `chroma_collection`, `embedding_model`, `chunk_count`）
- **API 路径**（knowledge_router — `_kb_to_dict` 读取 `uid`, `chunk_size`, `chunk_overlap`, `enable_graph_expansion`, `created_at`, `updated_at`）

原缓存 Hash 仅包含前 4 字段，API 缓存命中时返回 `null`/默认值。

**修复**：扩展 `_kb_to_hash` / `_kb_from_hash` 覆盖全部 ORM 字段，DateTime 序列化为 ISO 字符串。

---

## 未修复问题（在现有设计容忍范围内）

### F1 — `session/manager.py:save()` Redis 先于 DB 写，无回滚

**位置**：`session/manager.py:215`

**场景**：`await self._redis_save(session)` 成功后 DB 写入失败 → Redis 持有截至旧边界的数据。下次 `get_or_create` 时 Redis miss（2h TTL）→ 从 DB 重新加载 → 自愈。

**结论**：按 SDD §R4 设计，不处理。

### F2 — full consolidation 后 Redis 缓存清空，无性能收益

**位置**：`session/manager.py:179`

**场景**：`last_consolidated == len(messages)` 时 `window == []` → `pipe.delete(msg_key)` + 无 RPUSH → 下次 `_redis_load` 时 `not raw_msgs` 成立（空列表在 Python 中为 `False`）→ 返回 `None` → 回退 DB。

**结论**：首次 miss 后 DB 加载 + `_redis_save` 回写，仅一次 DB 开销。容忍。

### F3 — `datetime.utcnow()` 已弃用

**位置**：`agent/memory.py:631`

```
datetime.utcnow().isoformat()
```

应替换为 `datetime.now(timezone.utc).isoformat()`。功能无影响。

### F4 — async_search RedisKeys import 重复

**位置**：`hybrid_search.py:913` + `hybrid_search.py:929`

两次 `from nanobot.bus.redis_keys import RedisKeys`（分处不同 try 块）。功能无影响。

### F5 — `_batch_fetch_chunks_cached` import 重复

**位置**：`retrieval.py:38` + `retrieval.py:81`

与 F4 同样的问题，模块内不同函数/路径分别 import。

---

## 设计决策验证

### D1 — 控制信号 TTL 已全部移除

| Key | 位置 | 之前 | 之后 |
|---|---|---|---|
| `cancel:{session_key}` | chat_router.py:508 | SETEX 1800 | SET |
| `job:{job_id}` | chat_router.py:554 | SET ex=7200 | SET |
| `run_chat:{run_id}` | chat_router.py:401 | SET ex=86400 | SET |

`finally` 块（676-679）已有 DEL。通过。

### D2 — Session rolling window 一致性

- `save()` 写 `messages[last_consolidated:]`（Redis list = 未压缩后缀）
- Redis meta `last_consolidated` 固定为 `"0"`（list 起点即压缩边界）
- `_redis_load` 读 `int(meta["last_consolidated"])` → 始终为 0
- Lua LTRIM 参数 `keep_from_idx = end_idx - old_last_consolidated`，语义为"保留从该索引开始"的元素

验证：全部对齐。

### D3 — Embedding cache 无竞态

DenseRetriever 实例在 `SelfHostedRetriever` / `batch_retrieval.py` 中跨请求共享。`async_search()` 在 `run_in_executor` **之前** 做 Redis GET（async）和预计算 embedding（executor），再将向量传入 `search()`。DenseRetriever 实例无状态变异。通过。

---

## 未实现（Phase 2 覆盖）

- **Pub/Sub 无效化通道**：`INVALIDATE_SESSION`、`INVALIDATE_AGENT`、`INVALIDATE_KB` 已定义但无订阅/发布者。当前使用 DEL+EXPIRE 刷新，功能正常但多实例热部署时有 ~600s（TTL）的窗口期读到旧数据。
- **Pending reaper**：孤儿 `pending:*` 键清理（Phase 2-D）。

---

## 汇总

| 类别 | 数量 |
|---|---|
| 已修复 Bug | 2 |
| 未修复（设计容忍）| 5 |
| 设计决策正确性验证 | 3 |
| 遗漏实现（Phase 2）| 2 |
