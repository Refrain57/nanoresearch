# Redis 实现计划

> 基于 `redis-sdd.md`，包含执行前分析修正。

## 文件影响清单

| 文件 | Phase | 改动摘要 |
|---|---|---|
| `bus/redis_keys.py` (新建) | 0-A | RedisKeys 类 |
| `server/routers/chat_router.py` | 0-B/C, 1-D | 控制信号去 TTL；config cache-aside |
| `bus/redis_client.py` | 1-A | 删旧 helper，改为从 redis_keys 导入 |
| `bus/stream.py` | 1-A | `stream_key` → `RedisKeys.chat_events` |
| `agent/subagent.py` | 1-A | 6 处 import 改为 RedisKeys |
| `session/manager.py` | 1-B | Redis 读写层 |
| `agent/memory.py` | 1-C | Lua LTRIM |
| `storage/repositories/agent_repo.py` | 1-D | cache-aside + DEL on write |
| `storage/repositories/user_settings_repo.py` | 1-D | cache-aside + DEL on write |
| `storage/repositories/knowledge_repo.py` | 1-D | cache-aside + DEL on write |
| `rag/core/query_engine/dense_retriever.py` | 1-E | 接受 precomputed_query_embedding 参数 |
| `rag/core/query_engine/hybrid_search.py` | 1-E | async_search 做 embedding cache |
| `rag/mcp_server/tools/agentic/retrieval.py` | 1-F | FetchSection/Neighbors chunk cache |

---

## Phase 0 — 先决条件

### 0-A `bus/redis_keys.py`（新建）

照 SDD §4.1 完整实现 `RedisKeys` 类，包含：
- 控制信号 key 方法（无 TTL）
- Session key 方法 + `SESSION_TTL = 7200`
- Config cache key 方法 + `AGENT_TTL = 1800`, `USER_SETTINGS_TTL = 1800`, `KB_META_TTL = 600`
- RAG cache key 方法 + `CHUNK_TTL = 21600`, `EMBEDDING_TTL = 3600`
- SSE stream key 方法
- Pub/Sub channel 常量

### 0-B/C `chat_router.py` — 控制信号去 TTL

| 行 | 改动前 | 改动后 |
|---|---|---|
| 401 | `_redis.set(run_chat_key(...), chat_id, ex=86400)` | `_redis.set(run_chat_key(...), chat_id)` |
| 508 | `redis.setex(cancel_key(_session_key), 1800, "1")` | `redis.set(cancel_key(_session_key), "1")` |
| 554 | `redis.set(job_key(job_id), str(run_id), ex=7200)` | `redis.set(job_key(job_id), str(run_id))` |

`finally` 块（676-679）已有 DEL，无需改动。

### 0-D Redis 配置（用户手动执行）

```bash
redis-cli CONFIG SET maxmemory 512mb
redis-cli CONFIG SET maxmemory-policy volatile-lru
```

### 0-E 验证（已确认）

`subagent.py` 已确认：`pending` 只有 SADD/SREM，无 EXPIRE。✓

### Phase 0 验收检查（执行完后跑）

```bash
# 1. 控制信号无 SETEX / ex=（期望 0 结果）
grep -n "setex\|SETEX" backend/nanobot/server/routers/chat_router.py
grep -n "ex=7200\|ex=86400\|ex=1800" backend/nanobot/server/routers/chat_router.py

# 2. pending 无 EXPIRE
grep -n "expire\|EXPIRE" backend/nanobot/agent/subagent.py

# 3. redis_keys.py smoke test
python -c "from nanobot.bus.redis_keys import RedisKeys; print(RedisKeys.session_msg('u1','slack','C123'))"
# 期望: session:msg:u1:slack:C123
```

---

## Phase 1-A — 统一 Key Namespace

### `bus/redis_client.py`

删除 5 个旧 helper 函数（lines 20-38），末尾加 lambda 转发兼容层：

```python
from nanobot.bus.redis_keys import RedisKeys

stream_key   = RedisKeys.chat_events
job_key      = lambda job_id:      RedisKeys.job(job_id)
cancel_key   = lambda session_key: RedisKeys.cancel(session_key)
pending_key  = lambda session_key: RedisKeys.pending(session_key)
run_chat_key = lambda run_id:      RedisKeys.run_chat(run_id)
```

调用方（chat_router、subagent、stream）的 import 路径暂不变，通过转发层兼容。

### `bus/stream.py`

`stream_key(chat_id)` → `RedisKeys.chat_events(chat_id)`，import 改为 redis_keys。

---

## Phase 1-B — Session Redis 层

### `session/manager.py`

**`get_or_create(key)`** — 三层查找：

```
1. _cache hit → return
2. Redis HGETALL session:meta + LRANGE session:msg → 命中 → 重建 Session → _cache
3. Miss → _load(DB/file) → _redis_save() → _cache → return
```

**`save(session)`** — MULTI/EXEC：

```
DEL    session:msg:{uid}:{ch}:{chat_id}
RPUSH  session:msg:{uid}:{ch}:{chat_id}  [json(m) for m in messages[last_consolidated:]]
HSET   session:meta:{uid}:{ch}:{chat_id}
         updated_at <ts>
         created_at <ts>
         metadata   <json>
         last_consolidated "0"        ← 必须写入，重建时用
EXPIRE session:msg   7200
EXPIRE session:meta  7200
```

**重建 Session 时**：

```python
session.last_consolidated = int(meta.get("last_consolidated", 0))
```

两侧对齐，避免 get_or_create 重建后 last_consolidated 不确定。

**uid 来源**：`self._default_uid`（已有）  
**key 分割**：`ch, chat_id = key.split(":", 1)`（session.key 格式保证有 `:`）  
**降级**：Redis 失败 `except` + log，继续 DB/file 路径。

---

## Phase 1-C — Memory 压缩 Lua

### `agent/memory.py`

**uid 来源确认**：`maybe_consolidate_by_tokens(self, session, agent_id=None, uid=None)` 有 uid 参数，调用方 `loop.py` 传 `uid=self._uid`，来源可靠。

**注入位置**：在 `session.last_consolidated = end_idx` **之前**（需先捕获 `old_last_consolidated`），LLM 压缩成功后：

```python
old_last_consolidated = session.last_consolidated
end_idx = boundary[0]
chunk = session.messages[old_last_consolidated:end_idx]
...
if not await self.consolidate_messages(chunk, ...):
    return

# Lua LTRIM（fast-path，失败不影响正确性）
if uid is not None:
    try:
        parts = session.key.split(":", 1)
        if len(parts) == 2:
            from nanobot.bus.redis_keys import RedisKeys
            from nanobot.bus.redis_client import get_redis
            _redis = get_redis()
            _msg_key  = RedisKeys.session_msg(uid, parts[0], parts[1])
            _meta_key = RedisKeys.session_meta(uid, parts[0], parts[1])
            keep_from_idx = end_idx - old_last_consolidated  # LTRIM 保留起点
            await _redis.eval(
                _LUA_LTRIM, 2, _msg_key, _meta_key,
                str(keep_from_idx), datetime.utcnow().isoformat()
            )
    except Exception as _lua_err:
        logger.warning("Lua LTRIM failed (non-fatal): {}", _lua_err)

session.last_consolidated = end_idx
self.sessions.save(session)   # 注：此处缺 await 是预存在 bug，本次不修
```

变量命名：`keep_from_idx`（不用 `_drop`，语义更准确：LTRIM 保留从该索引开始的元素）。

Lua 脚本按 SDD §4.4 定义为模块常量 `_LUA_LTRIM`。

---

## Phase 1-D — Config Cache-Aside

### 策略

各 repo 的 read 方法内部 `try/except` 包裹 Redis，不改方法签名。`get_redis()` 直接调用。

### `AgentRepository.get_by_id()`

- Redis key: `agent:{agent_id}`，TTL: 1800
- Hash fields: `name`, `persona`, `skills_config`(JSON), `harness`(JSON), `tools_config`(JSON), `default_model`, `is_default`
- `update()` 末尾加 DEL

### `UserSettingsRepository.get()` / `upsert()`

- Redis key: `user_settings:{uid}`，TTL: 1800
- Hash fields: `model`, `extra`(JSON)
- `upsert()` 末尾加 DEL

### `KnowledgeRepository.get()` / `update()`

- Redis key: `kb:meta:{kb_id}`，TTL: 600
- Hash fields: `name`, `chroma_collection`, `embedding_model`, `chunk_count`
- `update()` 末尾加 DEL

---

## Phase 1-E — Embedding Cache

### 生命周期确认

`batch_retrieval.py` 中 `_retriever_cache[cache_key]` 跨请求缓存 HybridSearch 和 DenseRetriever 实例。**实例是共享的**，`_last_query_vector` 方案有竞态，禁止使用。

### 正确方案：embedding 在 async 上下文预计算后传入

**`DenseRetriever.retrieve()`** 新增参数：

```python
def retrieve(self, query, top_k=None, filters=None, trace=None,
             precomputed_query_embedding: list[float] | None = None):
    ...
    if precomputed_query_embedding is not None:
        query_vector = precomputed_query_embedding
    else:
        query_vectors = self.embedding_client.embed([query], trace=trace)
        query_vector = query_vectors[0]
    # 后续 vector_store.query 不变
```

**`HybridSearch.async_search()`** 在 `run_in_executor` 之前：

```python
import asyncio, hashlib, json
from nanobot.bus.redis_client import get_redis
from nanobot.bus.redis_keys import RedisKeys

text_hash = hashlib.sha256(query.encode()).hexdigest()[:32]
precomputed_embedding = None
_redis = None

try:
    _redis = get_redis()
    cached = await _redis.get(RedisKeys.embedding(text_hash))
    if cached:
        precomputed_embedding = json.loads(cached)
except Exception: pass

# Cache miss：在 executor 里单独计算 embedding，再存 Redis
if precomputed_embedding is None and self.dense_retriever is not None:
    try:
        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(
            None, lambda: self.dense_retriever.embedding_client.embed([query])
        )
        precomputed_embedding = vecs[0]
        if _redis is not None:
            await _redis.set(
                RedisKeys.embedding(text_hash),
                json.dumps(precomputed_embedding),
                ex=RedisKeys.EMBEDDING_TTL,
            )
    except Exception: pass

# 原 run_in_executor，传入预计算向量
base = await loop.run_in_executor(
    None,
    lambda: self.search(query, top_k, filters, trace, return_details,
                        precomputed_query_embedding=precomputed_embedding),
)
```

`search()` 需要透传 `precomputed_query_embedding` 到 `_run_dense_retrieval()` → `DenseRetriever.retrieve()`。

---

## Phase 1-F — Chunk Cache

### `rag/mcp_server/tools/agentic/retrieval.py`

**collection name vs kb_id**：`self._current_collection` 是 chroma 集合名，**不是** kb_id UUID。chunk cache key 改用 collection_name 作命名空间：`chunk:{collection_name}:{chunk_id}`（与 SDD `chunk:{kb_id}:{chunk_id}` 结构一致，只是 namespace 改为 collection name，保证缓存隔离正确）。

**`FetchSectionTool.execute()` 和 `FetchNeighborsTool.execute()`**：

对 `collection.get(ids=[...])` 的个别 chunk 查询：
1. 批量 `HGETALL chunk:{collection}:{chunk_id}` → 命中用缓存
2. 缺失 ID 从 vector store 取
3. 取回后 `HSET ... EXPIRE CHUNK_TTL`

全部包在 `try/except`，失败回退原始路径。
