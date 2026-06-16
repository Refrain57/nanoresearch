# Redis SDD — Nanobot

## 1. Overview

### Objective

Introduce three new Redis usage layers on top of the existing event-streaming infrastructure:

1. **Session short-term memory** — store conversation messages in Redis (2 h TTL) to eliminate per-turn DB round-trips and enable multi-instance session sharing.
2. **Config hot-cache** — cache Agent config / User settings / KB metadata in Redis Hash to avoid repeated DB reads on every request.
3. **RAG retrieval cache** — cache chunk text and query embeddings to reduce ChromaDB + embedding API calls.

### In Scope

- `redis_keys.py` — unified key namespace (new file)
- `session/manager.py` — Redis backend for session messages + meta
- `agent/memory.py` — Lua-based atomic LTRIM on compaction
- Config hot-cache layer in `server/routers/chat_router.py` and relevant repos
- RAG chunk + embedding cache in `rag/libs/` and retrieval path
- Graceful degradation (try/except → DB fallback) for all three layers
- Monitoring instrumentation (eviction alerts, hit/miss metrics)

### Out of Scope

- SSE Redis Stream (`chat_events:{chat_id}`) — already implemented; no changes
- Control signal keys (`pending`, `cancel`, `job`, `run_chat`) write-path logic — structure preserved, TTL behaviour adjusted per spec
- Consumer groups (XGROUP) — not planned
- Redis Cluster — Sentinel / Managed Redis addressed in Phase 2

---

## 2. Current State

### 2.1 Redis Client

**File**: `backend/nanobot/bus/redis_client.py`

- Library: `redis.asyncio`
- Connection: singleton, `REDIS_URL` env var (default `redis://localhost:6379`)
- `decode_responses=True` globally — all keys/values are strings
- No TLS, no auth in default config
- Helper functions defined here: `stream_key`, `job_key`, `cancel_key`, `pending_key`, `run_chat_key`

**Known issue**: Key helper functions are scattered in `redis_client.py` rather than a unified namespace; no TTL constants defined anywhere.

### 2.2 Control Signal Keys — Current Behaviour

| Key pattern | Redis type | Current TTL | Producer | Consumers |
|---|---|---|---|---|
| `pending:{session_key}` | Set | **NONE** | `subagent.py:78` SADD | `chat_router.py:645` SCARD; `subagent.py:198,210,246,271` SREM |
| `cancel:{session_key}` | String | **1800 s** (SETEX) | `chat_router.py:508` | `chat_router.py:588`; `subagent.py:158` EXISTS |
| `job:{job_id}` | String | **7200 s** (SET ex=) | `chat_router.py:554` | `chat_router.py:369` GET |
| `run_chat:{run_id}` | String | **86400 s** (SET ex=) | `chat_router.py:401` | `chat_router.py:471` GET |

**Critical finding**: `cancel`, `job`, and `run_chat` currently carry TTL. Under `volatile-lru` policy these keys are eviction candidates — silently dropping a cancellation signal or allowing a duplicate job to slip through. The design spec mandates **no TTL** on all four control signal key types so they are non-evictable.

`pending:{session_key}` already has no TTL, but this creates an orphan-key risk: if a subagent crashes after `SADD` but before `SREM`, the set member leaks indefinitely.

### 2.3 SSE Event Stream

**File**: `backend/nanobot/bus/stream.py`

- Key: `chat_events:{chat_id}` — Redis Stream, EXPIRE 86400 s after every XADD
- Producers: `subagent.py:237` (subagent_result), `chat_router.py:631` (message_complete)
- Consumer: `chat_router.py` xread_next loop (lines 637–649), cursor recorded before agent starts
- Chunking: payloads > 8 192 B split into multiple XADD entries, reassembled on read
- No consumer groups; per-client cursor tracking

Status: **working as designed; no changes planned**.

### 2.4 Session Management

**File**: `backend/nanobot/session/manager.py`

- Session key format: `"{channel}:{chat_id}"` — **uid is absent from the key**
- In-process cache: `_cache: dict[str, Session]` — process-local, lost on restart
- Primary backend: PostgreSQL (`conversations` + `messages` tables)
- Fallback backend: JSONL files under `{workspace}/sessions/`
- `save()`: calls `ConversationRepository.replace_messages()` (DELETE + bulk INSERT) + `update_meta()`
- `get_or_create()`: checks `_cache` → loads from DB/file → creates empty Session if missing

**Known issues**:
- No shared state between multiple worker processes (each has its own `_cache`)
- `replace_messages` deletes and re-inserts all messages on every save — expensive for long conversations
- Session key lacks `uid`, making multi-tenant Redis namespace collision possible

### 2.5 Memory Compaction

**File**: `backend/nanobot/agent/memory.py`

Triggered by `maybe_consolidate_by_tokens()` when estimated prompt tokens exceed `context_window - max_completion - 1024`.

Post-compaction behaviour:
1. A slice `messages[last_consolidated:end_idx]` is sent to LLM for summarisation → written to `MEMORY.md`
2. `session.last_consolidated` is advanced to `end_idx`
3. `sessions.save(session)` is called — DB write
4. **Messages are never removed from `session.messages`**; `get_history()` returns `messages[last_consolidated:]`

No Lua scripts or MULTI/EXEC patterns currently; the offset update + DB save are sequential (not atomic).

### 2.6 RAG Retrieval

**Files**: `backend/nanobot/rag/libs/`, `backend/nanobot/rag/mcp_server/`

- Dense retrieval: ChromaDB vector search, `dense_top_k = 20`
- Sparse retrieval: BM25 in-memory index, `sparse_top_k = 20`
- Hybrid fusion: RRF, `fusion_top_k = 10`
- Embedding: `EmbeddingFactory` → OpenAI / Azure / Ollama / DashScope (per `settings.embedding.provider`)
- **No application-level caching** for chunk text or query embeddings — every request hits ChromaDB + embedding API
- ChromaDB client instance caching (`_client_cache`) exists but is process-local

### 2.7 Configuration Loading

No caching layer exists for any of the three config types:

| Config | Source table | Loading point | Cache? |
|---|---|---|---|
| Agent config | `agents` | `chat_router.py:349` + `:597` | ❌ fresh DB query |
| User settings | `user_settings` | `chat_router.py:48` | ❌ fresh DB query |
| KB metadata | `knowledge_bases` | `chat_router.py:702`, `:595` | ❌ fresh DB query |

---

## 3. Key Space Design

All keys defined in `backend/nanobot/bus/redis_keys.py` (new file).

| Key pattern | Type | TTL | volatile-lru evictable? | Invalidation | Notes |
|---|---|---|---|---|---|
| `pending:{session_key}` | Set | **None** | No | Manual SREM / DEL | session_key = `{ch}:{chat_id}` |
| `cancel:{session_key}` | String | **None** | No | Manual DEL in finally | Remove SETEX; add explicit DEL |
| `job:{job_id}` | String | **None** | No | Manual DEL in finally | Remove SET ex=; add explicit DEL |
| `run_chat:{run_id}` | String | **None** | No | Manual DEL in finally | Remove SET ex=; add explicit DEL |
| `chat_events:{chat_id}` | Stream | 86400 s (unchanged) | Yes | Auto-expire | No changes |
| `session:msg:{uid}:{ch}:{chat_id}` | List | **7200 s** | Yes | EXPIRE refresh on save; Pub/Sub `invalidate:session` on logout | Rolling window: list starts at compaction boundary |
| `session:meta:{uid}:{ch}:{chat_id}` | Hash | **7200 s** | Yes | Same as msg key | Fields: created_at, updated_at, metadata (JSON); last_consolidated always 0 |
| `agent:{agent_id}` | Hash | **1800 s** | Yes | DEL on update; Pub/Sub `invalidate:agent` | All agent fields as hash fields |
| `user_settings:{uid}` | Hash | **1800 s** | Yes | DEL on upsert | Fields: model, extra (JSON) |
| `kb:meta:{kb_id}` | Hash | **600 s** | Yes | DEL on update; Pub/Sub `invalidate:kb` | Fields: name, chroma_collection, embedding_model, chunk_count |
| `chunk:{kb_id}:{chunk_id}` | Hash | **21600 s** | Yes | DEL on KB re-index | Fields: text, section_path, seq |
| `embedding:{text_hash}` | String | **3600 s** | Yes | Auto-expire | Value: JSON array of float; hash = SHA256(text)[:32] |

### Pub/Sub Channels (invalidation notifications)

```
invalidate:session     — payload: "{uid}:{ch}:{chat_id}"
invalidate:agent       — payload: "{agent_id}"
invalidate:kb          — payload: "{kb_id}"
```

User-settings invalidation is handled by DEL + EXPIRE refresh; no dedicated Pub/Sub channel needed (low-frequency writes).

---

## 4. Component Changes

### 4.1 `backend/nanobot/bus/redis_keys.py` (new file)

**Before**: Key strings inlined in `redis_client.py` helper functions; no TTL constants.

**After**:
```python
class RedisKeys:
    # Control signals — no TTL, manual DEL, non-evictable under volatile-lru
    @staticmethod
    def pending(session_key: str) -> str: return f"pending:{session_key}"
    @staticmethod
    def cancel(session_key: str) -> str: return f"cancel:{session_key}"
    @staticmethod
    def job(job_id: str) -> str: return f"job:{job_id}"
    @staticmethod
    def run_chat(run_id: str) -> str: return f"run_chat:{run_id}"

    # Session short-term memory — 2 h TTL, MULTI/EXEC write
    SESSION_TTL = 7200
    @staticmethod
    def session_msg(uid: str, ch: str, chat_id: str) -> str:
        return f"session:msg:{uid}:{ch}:{chat_id}"
    @staticmethod
    def session_meta(uid: str, ch: str, chat_id: str) -> str:
        return f"session:meta:{uid}:{ch}:{chat_id}"

    # Config hot-cache — DEL on write
    AGENT_TTL, USER_SETTINGS_TTL, KB_META_TTL = 1800, 1800, 600
    @staticmethod
    def agent(agent_id: str) -> str: return f"agent:{agent_id}"
    @staticmethod
    def user_settings(uid: str) -> str: return f"user_settings:{uid}"
    @staticmethod
    def kb_meta(kb_id: str) -> str: return f"kb:meta:{kb_id}"

    # RAG cache — volatile-lru evictable
    CHUNK_TTL, EMBEDDING_TTL = 21600, 3600
    @staticmethod
    def chunk(kb_id: str, chunk_id: str) -> str: return f"chunk:{kb_id}:{chunk_id}"
    @staticmethod
    def embedding(text_hash: str) -> str: return f"embedding:{text_hash}"

    # SSE stream (unchanged)
    @staticmethod
    def chat_events(chat_id: str) -> str: return f"chat_events:{chat_id}"

    # Pub/Sub channels
    INVALIDATE_SESSION = "invalidate:session"
    INVALIDATE_AGENT   = "invalidate:agent"
    INVALIDATE_KB      = "invalidate:kb"
```

Old helper functions in `redis_client.py` (lines 20–39) are replaced by imports from `RedisKeys`.

### 4.2 `backend/nanobot/bus/redis_client.py`

**Before**: Inline key helper functions (lines 20–39); no TTL handling.

**After**: Remove inline key functions; import `RedisKeys`; existing `get_redis()` singleton unchanged.

### 4.3 `backend/nanobot/session/manager.py`

**Before**: `get_or_create()` → in-process `_cache` → DB/JSONL. `save()` → DB replace_messages (bulk DELETE + INSERT).

**After**: Redis backend added as priority tier. The Redis list is a **rolling window** — it always holds `session.messages[session.last_consolidated:]` (the unconsolidated suffix), not the full history. `last_consolidated` in Redis meta is always `0` because the list's first element is already at the compaction boundary; the absolute offset lives only in the DB and in the Python `Session` object.

```
get_or_create():
  1. Check _cache (turn-scoped; cleared after save returns)
  2. HGETALL session:meta + LRANGE session:msg → Redis hit → reconstruct Session
     (session.last_consolidated = 0 from meta; messages = list contents)
  3. Miss → load from DB → MULTI/EXEC warm-up (see save() below) → cache in _cache → return

save():
  1. MULTI/EXEC:
       DEL    session:msg  {uid}:{ch}:{chat_id}
       RPUSH  session:msg  {uid}:{ch}:{chat_id}  <messages[session.last_consolidated:]  as JSON>
       HSET   session:meta {uid}:{ch}:{chat_id}  updated_at {ts}  metadata {json}
       EXPIRE session:msg  7200
       EXPIRE session:meta 7200
  2. DB save (unchanged, write-through)
  3. Update _cache
  4. On Redis error: log + continue (DB is source of truth)
```

**Why DEL + full RPUSH (not incremental RPUSH)**: Under `volatile-lru`, `session:msg` can be evicted at any time. If the key disappears between two `save()` calls, an incremental RPUSH would write only the newest messages into an empty list, silently losing all prior context. DEL + full RPUSH is idempotent regardless of eviction.

**Why `messages[last_consolidated:]` not `messages[:]`**: Writing the full history would make the Lua LTRIM in 4.4 a no-op — the very next `save()` call would overwrite the trimmed list with the full history again, yielding zero memory benefit. Writing only the unconsolidated window ensures LTRIM and `save()` are consistent: both maintain the invariant that the Redis list starts at the compaction boundary.

**uid resolution**: `self._default_uid` is already available (`manager.py:126`); `chat_router.py:57` passes `default_uid=uid`. No API change needed. Key is constructed as `f"session:msg:{self._default_uid}:{ch}:{chat_id}"` by splitting `session.key` on `":"`.

In-process `_cache` is retained but scoped to the current turn only.

### 4.4 `backend/nanobot/agent/memory.py`

**Before**: After compaction, `session.last_consolidated` is advanced and `sessions.save(session)` is called sequentially. No Redis coordination.

**After**: After LLM consolidation succeeds, run a Lua script **before** calling `sessions.save()` to atomically trim the Redis list to match the new compaction boundary:

```lua
-- KEYS[1] = session:msg key, KEYS[2] = session:meta key
-- ARGV[1] = number of messages to drop from the front (= end_idx - old_last_consolidated)
-- ARGV[2] = ISO timestamp
local drop = tonumber(ARGV[1])
local len  = redis.call('LLEN', KEYS[1])
if len > drop then
    redis.call('LTRIM', KEYS[1], drop, -1)
else
    redis.call('DEL', KEYS[1])
end
redis.call('HSET', KEYS[2], 'updated_at', ARGV[2])
return 1
```

`last_consolidated` is **not written** to Redis meta here — it is always `0` by convention (list start = compaction boundary). The Lua LTRIM advances that boundary by dropping `drop` entries from the front.

Sequence:
1. Lua LTRIM succeeds → Redis list now starts at new boundary
2. `session.last_consolidated = end_idx` (Python object)
3. `sessions.save(session)` → DB write + Redis DEL+RPUSH of `messages[end_idx:]`

Step 3 re-writes the list from the new boundary, which is consistent with the Lua result. If Lua fails: skip it, proceed to step 2+3 — `save()` will write the correct window anyway. Lua is a fast-path optimisation to shed memory before `save()` runs, not a correctness requirement.

If `session.msg` key is absent (evicted) when Lua runs, `LLEN` returns 0 and the script exits cleanly; `save()` will repopulate the list correctly.

### 4.5 `backend/nanobot/server/routers/chat_router.py`

**Before**: Lines 48–51 query `user_settings` DB fresh. Lines 349, 597–607 query `agents` and `knowledge_bases` DB fresh. Lines 401, 508, 554 use `SET ex=` / `SETEX` for control signals.

**After**:
- Remove `ex=` from `job` SET (line 554) and `run_chat` SET (line 401); confirm DEL in finally block.
- Replace `SETEX cancel_key` (line 508) with `SET cancel_key "1"` (no TTL); DEL already at line 679.
- Wrap agent/user/KB reads in cache-aside: HGETALL → miss → DB → HSET + EXPIRE.

### 4.6 `backend/nanobot/storage/repositories/` (agent_repo, user_settings_repo, knowledge_repo)

**Before**: `get()` / `upsert()` methods query/write DB directly, no cache.

**After**: Each write method adds `redis.delete(RedisKeys.xxx(...))` after the DB commit (write-invalidation). Cache is refilled on the next read miss.

### 4.7 RAG retrieval path

**Files**: `backend/nanobot/rag/core/query_engine/hybrid_search.py`, `backend/nanobot/rag/mcp_server/tools/agentic/retrieval.py`

**Before**: Every retrieval hits ChromaDB for vectors + PostgreSQL for chunk text.

**After**:
1. Compute `text_hash = SHA256(query_text)[:32]`
2. `GET RedisKeys.embedding(text_hash)` → hit: skip embedding API call
3. After vector search returns `chunk_id` list, batch `HGETALL RedisKeys.chunk(kb_id, chunk_id)` for all IDs
4. Fetch missing chunk IDs from PostgreSQL `kb_chunks`; write back with HSET + EXPIRE
5. Cache query embedding: `SET RedisKeys.embedding(text_hash) {json_vector} EX 3600`

All Redis calls wrapped in try/except; on error → fall through to ChromaDB/DB path.

---

## 5. Memory Estimation

### Baseline Assumptions (from code)

- `fusion_top_k = 10` per query (final chunks returned to LLM)
- Typical chunk: ~512 tokens ≈ 2 KB text
- Session unconsolidated window: ~20 messages avg × 500 bytes = 10 KB per session (rolling window, not full history)
- Embedding vector: 1 536 dimensions × 4 bytes + JSON overhead ≈ 8 KB

### Estimate by Layer

| Layer | Unit size | Expected active count | Total |
|---|---|---|---|
| Session messages (List, rolling window) | 10 KB | 200 active sessions | 2 MB |
| Session meta (Hash) | 0.5 KB | 200 | 0.1 MB |
| Agent config | 5 KB | 50 distinct agents | 0.25 MB |
| User settings | 1 KB | 200 active users | 0.2 MB |
| KB metadata | 2 KB | 50 KBs | 0.1 MB |
| Chunk cache (6 h window) | 2 KB | 5 000 unique chunks | 10 MB |
| Embedding cache (1 h window) | 8 KB | 500 unique queries/h | 4 MB |
| SSE streams (24 h) | ~50 KB | 100 active streams | 5 MB |
| Control signals | < 1 KB | 200 concurrent | 0.2 MB |
| Redis overhead (~2×) | — | — | ~22 MB |

**Subtotal ≈ 44 MB**

**Recommended `maxmemory`**: Start at **512 MB** (>10× headroom for spikes — batch ingestion can temporarily spike chunk cache). Scale to **1 GB** if KB count grows past 200 or concurrent sessions exceed 500.

---

## 6. Risk Analysis

### R1 — volatile-lru EXPIRE prerequisite not met for control signals

**Current state**: `cancel` uses SETEX 1800 s; `job` uses SET ex=7200; `run_chat` uses SET ex=86400.
Under `volatile-lru`, these keys are eviction candidates. If Redis evicts `cancel:{session_key}` mid-run, subagents will miss the disconnect signal; if it evicts `job:{job_id}`, duplicate jobs can slip through.

**Fix (Phase 0 blocker)**: Remove TTL from all three SET calls. Ensure the `finally` block in `chat_router.py` (lines 676–679) DELetes all three keys unconditionally. Orphaned keys (process crash before finally) are cleaned up by the background reaper in Phase 2 (task 2-D).

### R2 — Memory estimate deviation

If a knowledge base has unusually large chunks (e.g. 4 KB average from marker-extracted PDFs), the chunk cache estimate doubles. Monitor `MEMORY USAGE chunk:*` in production and adjust `maxmemory` accordingly.

**Mitigation**: Phase 3 prefix-level memory monitoring will alert before OOM.

### R3 — Redis SPOF (Single Point of Failure)

Current deployment uses a single Redis instance (localhost). If it crashes: SSE streams are unavailable; session reads fall back to DB; config reads fall back to DB. All non-stream fallback paths are implemented in Phase 2.

**SSE streams have no fallback** — Redis outage = SSE disconnect. Client-side reconnect with exponential backoff handles transient failures. For production: use Redis Sentinel or a managed Redis service (Phase 2).

### R4 — Compaction Lua script atomicity edge cases

If Lua LTRIM succeeds but `sessions.save()` DB write fails, Redis holds the trimmed window while DB still has full history. On the next `get_or_create()` Redis miss (TTL expiry), the DB state is reloaded and `save()` writes the correct window. No data is lost; the discrepancy is self-healing.

**Mitigation**: Log a warning when Lua succeeds but DB save fails so the discrepancy is visible.

### R5 — Multi-tenant key collision in control signals

Control signal keys use `{channel}:{chat_id}` without uid. In practice `chat_id` is a platform-specific ID (Slack channel ID, Telegram chat ID) that is globally unique within its channel type, so collision is unlikely.

**Recommended fix**: Prefix control signal keys with `uid` in a follow-up, or document the uniqueness assumption explicitly.

### R6 — Session key format and uid resolution

New Redis session keys require `uid` which is absent from the Python `Session.key` (`{ch}:{chat_id}`). `SessionManager.__init__` already accepts `default_uid` (line 126, `manager.py`) stored as `self._default_uid`. `chat_router.py:57` already passes `default_uid=uid`. No API signature change needed.

Backward compat: `default_uid="admin"` (default for tests / CLI) produces `session:msg:admin:{ch}:{chat_id}` — still unique per session.

---

## 7. Implementation Plan

### Phase 0 — Prerequisites (blocking; must complete before any code changes)

| Task | File | Action | Verification |
|---|---|---|---|
| 0-A | `backend/nanobot/bus/redis_keys.py` | Create file with full `RedisKeys` class | `python -c "from nanobot.bus.redis_keys import RedisKeys; print(RedisKeys.session_msg('u','slack','C1'))"` |
| 0-B | `backend/nanobot/server/routers/chat_router.py` | Remove `ex=` from `job` SET (line 554) and `run_chat` SET (line 401) | grep returns 0 matches for these two lines |
| 0-C | `backend/nanobot/server/routers/chat_router.py` | Replace `SETEX cancel_key` (line 508) with `SET cancel_key "1"` | `grep -n "setex\|SETEX" chat_router.py` → 0 results |
| 0-D | Redis instance | Set `maxmemory 512mb` and `maxmemory-policy volatile-lru` | `redis-cli CONFIG GET maxmemory-policy` → `volatile-lru` |
| 0-E | Verify `pending` has no EXPIRE | Grep subagent.py | `grep -n "expire\|EXPIRE" subagent.py` → confirm no EXPIRE on pending key |

**Phase 0 checkpoint**: All four control signal key types must have zero EXPIRE/SETEX/ex= calls before proceeding.

### Phase 1 — Core Implementation

| Task | File | Action | Verification |
|---|---|---|---|
| 1-A | `bus/redis_client.py` | Replace inline key helpers with `RedisKeys` imports | All callers updated; import test passes |
| 1-B | `session/manager.py` | Add Redis backend: `get_or_create()` reads from Redis first; `save()` uses MULTI/EXEC DEL+RPUSH(`messages[last_consolidated:]`)+HSET+EXPIRE | Integration test: start session, restart process, session loaded from Redis with correct unconsolidated window |
| 1-C | `agent/memory.py` | Add Lua LTRIM before `sessions.save()`; Lua drops `end_idx - old_last_consolidated` entries from list front | Unit test: 20 msgs in Redis list, compact at index 10 → list has 10 entries; next `save()` writes `messages[10:]` → list still 10 entries |
| 1-D | `chat_router.py` + repos | Add cache-aside for agent config, user settings, KB meta | `redis-cli HGETALL agent:{id}` returns data after first request |

> **Pub/Sub invalidation (accepted risk)**: SDD §3 defines `invalidate:agent` / `invalidate:kb` / `invalidate:session` channels. No subscriber is implemented. Write path uses DEL only (same-worker invalidation). Multi-worker consistency relies on TTL (agent 1800s, kb 600s, session 7200s). This is acceptable because the in-process `_cache` is turn-scoped and not a persistent store; introducing a persistent _cache is an explicit architectural decision that will trigger a subscriber implementation at that time.
| 1-E | `rag/core/query_engine/hybrid_search.py` | Add embedding cache (GET/SET on text_hash) | `redis-cli GET embedding:{hash}` returns vector after first query |
| 1-F | `rag/mcp_server/tools/agentic/retrieval.py` | Add chunk cache (batch HGETALL → miss → DB → HSET) | `redis-cli HGETALL chunk:{kb_id}:{chunk_id}` populated after retrieval |

### Phase 2 — Stability

| Task | File | Action | Verification |
|---|---|---|---|
| 2-A | All Redis call sites | Wrap in `try/except RedisError` → log + fall back to DB | Kill Redis mid-request; verify 200 response, no 500 |
| 2-B | `session/manager.py` | On Redis error in `save()`, continue with DB-only mode | Set `REDIS_URL=""` → session writes succeed |
| 2-C | Infrastructure | Configure Redis Sentinel or switch to Managed Redis | Failover test: kill primary, sentinel promotes replica within 30 s |
| 2-D | New `bus/pending_reaper.py` | Background task: SCAN `pending:*`; for each key check `OBJECT IDLETIME > 7200` AND `EXISTS chat_events:{chat_id}` is false; DELETE only if both conditions met. **Age guard is mandatory** — stream absence alone can be a false positive for a freshly started session whose previous stream already expired. | Unit test: orphan key idle > 2 h with no stream is reaped; active key with live stream is untouched |

### Phase 3 — Observability

| Task | File | Action | Verification |
|---|---|---|---|
| 3-A | New monitoring module | Poll `redis-cli INFO stats`; alert if `evicted_keys` delta > 0 per minute | Simulate memory pressure; alert fires |
| 3-B | New scheduled task | Periodic `SCAN + MEMORY USAGE` per prefix; store in metrics DB | Log shows per-prefix byte breakdown |
| 3-C | Session / config / RAG call sites | Add hit/miss counters (`session_cache_hit`, `agent_cache_miss`, etc.) via structlog | Hit rate > 80% for session layer after warmup |

---

## Appendix: Phase 0 Grep-Check Reference

```bash
# 1. No SETEX / ex= on cancel/job/run_chat (must return 0 matches):
grep -n "setex\|SETEX" backend/nanobot/server/routers/chat_router.py
grep -n "\.set(cancel_key\|\.set(job_key\|\.set(run_chat_key" backend/nanobot/server/routers/chat_router.py | grep "ex="

# 2. No EXPIRE on pending key in subagent:
grep -n "expire\|EXPIRE" backend/nanobot/agent/subagent.py

# 3. No TTL in control signal methods of redis_keys.py:
grep -A5 "def pending\|def cancel\|def job\|def run_chat" backend/nanobot/bus/redis_keys.py | grep -i "expire\|setex\|ex="
# Expected: (empty)

# 4. Smoke-test key construction:
python -c "from nanobot.bus.redis_keys import RedisKeys; print(RedisKeys.session_msg('u1','slack','C123'))"
# Expected: session:msg:u1:slack:C123
```
