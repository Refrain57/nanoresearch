# Memory Layering P2+P3+P4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** On top of P1 (画像 store), add the events layer (P2), the conversation-summary sliding-window layer (P3), and the one-time PG rebuild + old-store retirement (P4).

**Architecture:** `consolidate()` becomes a single LLM pass whose `save_memory` tool returns `events[]` + `summary` + (P1) `profile_diff`-equivalent. Events go to a new Chroma collection `mem_events` (semantic recall replaces the flat `user_memory` `<history>`). Conversation summaries go to `mem_conv_summaries` keyed by `conversation_id`+turn range; injection uses a **bounded sliding window** — recent segments deterministically re-injected (~60% budget), earlier segments via semantic recall filtered to the same conversation (~40%). Migration rebuilds all derived layers from PG and retires `user_memory` + the old extractor.

**Tech Stack:** Python 3.12, Chroma (`ChromaStore`), `KnowledgeSearch` hybrid (BM25+vector+RRF+rerank+decay), SQLAlchemy async, pytest 9.

## Global Constraints

- Reuse `KnowledgeSearch`/`ChromaStore` machinery (mirror the proven `user_memory` methods); do NOT invent a new vector layer.
- Test DB recipe (P1): fresh `nanoresearch_test_mem` via `TEST_DATABASE_URL`/`TEST_DATABASE_DSN`; run with main `backend/.venv` python + `PYTHONPATH=<worktree>/backend`.
- Each live-path change keeps a safe fallback; branch stays unmerged for review.

## Verification Boundary (READ FIRST)

Unit tests do **not** spin up real Chroma/embeddings (repo convention: consolidation is tested with a fake `KnowledgeSearch` + mocked LLM tool-calls). Therefore:

- **Unit-verified (green tests required):** consolidation routing (events/summary parsed → correct store method called with correct args), sliding-window selection (pure fn), injection assembly (`<conversation_summary>`/`<history>` blocks from a fake store), migration chunking/orchestration (mocked consolidator).
- **Implemented-by-analogy, integration-verify-on-merge:** the actual `ChromaStore` write/search adapters in `KnowledgeSearch` (mirror `write_user_memory_sync`/`search_user_memory_sync`, which are already proven). Verified by import/type + a smoke call with a fake encoder where possible; real embedding recall confirmed after merge in a running env.
- **Manual-run:** P4 rebuild over real PG history invokes the real consolidation LLM — cannot run autonomously without API + data. Its chunking/orchestration is unit-tested with a mocked consolidator; the actual bulk run is a documented manual step.

---

## Pre-execution Corrections (MANDATORY — override any conflicting task text below)

Confirmed with the user after the first draft. Where a task section conflicts, THIS section wins.

**C1 (P4.1, top priority — data correctness): rebuild is per-uid serial + shared state; the test must prove convergence.**
`rebuild_uid` processes a uid's conversations in **chronological order**, applying each chunk's consolidation onto the **same already-built fact store** (NOT one independent run per conversation). Later conversations' diffs land on the earlier-built 画像 so contradictions **converge** (A "prefers X" then later B "prefers not-X" → store ends with the later only, not both). The test must **not** mock the consolidator to count calls — it runs the real diff-apply against a real `memory_facts` store with a scripted LLM emitting contradictory profile lines across two conversations, and asserts the final store converged (only the later fact active).

**C2 (P1 dependency — confirm): remove matching is normalized, not bare text.**
`compute_profile_diff` locates remove victims by **normalized** (section, casefold+whitespace-collapsed text) key — already implemented in P1 via `normalize()`, NOT raw equality. Rebuild relies on this so rephrasing doesn't strand stale contradictory facts. (Semantic matching is a later upgrade; normalized is the accepted baseline.)

**C3 (P3.2/P3.3 — restore the design the user rejected in the draft): minimal recent slice + 60% CAP + protected growth floor.**
`select_recent_window` must NOT greedily fill to 60%. It takes only the **smallest most-recent segment(s) adjacent to the compaction boundary** — just enough to catch back-references — and 60% is a **hard circuit-breaker ceiling, not a fill target**. Add a **reserved new-message growth floor**: injected blocks (`<memory>`+`<conversation_summary>`+`<history>`) must stay within their bounded budget and never grow toward the context window; if the conv-summary sub-budget would be breached, shrink **far** (semantic) first, then near. Goal: compaction is triggered by real conversation volume, never forced by injection bloat.
Revised signature: `select_recent_window(summaries, cap_tokens, est_fn) -> (near, far_remainder)` — `near` = minimal recent slice bounded by `cap_tokens` (a ceiling); `_build_dynamic_suffix` keeps total conv-summary injection within its sub-budget, far-then-near shrink on overflow.

**C4 (P2.2 — traceability): backfill `derived_from` with the just-written event ids.**
`consolidate()` writes `events` FIRST, captures their ids, THEN applies the profile diff with `derived_from=<those event ids>`. `MemoryStore._apply_profile_update` gains `derived_from_event_ids: list[str]` threaded to `repo.insert_extracted(..., derived_from=...)`. (P1-era facts keep empty `derived_from` — expected.)

**C5 (data foundation — confirm): events are append-only, no TTL/cleanup.**
The events line introduces **no** `cleanup_*`/TTL/expiry. `search_events_sync` decay is a **recall ranking weight only** — never deletes/hides events (else `fact.derived_from` dangles). No `cleanup_old_*` analog for `mem_events`.

**C6 (option-1 premise — minimize analogy surface): reuse the proven hybrid chain, don't rewrite it.**
Factor the existing `search_user_memory_sync` pipeline (over-retrieve → uid filter → BM25 → RRF → rerank → decay) into a shared `_hybrid_search(store, query, top_k, uid, extra_filters=None, exclude_ids=None)` helper; `search_events_sync`/`search_conv_summaries_sync` CALL it, differing only by store + metadata filter. Do NOT duplicate/rewrite the ranking chain per collection — fake-store unit tests can't catch subtle deviations, so keep divergence to store+filter only.

---

## Phase P2 — Events layer

### Task P2.1: `KnowledgeSearch` events collection + write/search

**Files:**
- Modify: `backend/nanoresearch/research/knowledge_search.py` (add `mem_events_store`; `write_events_sync`, `search_events_sync`)
- Test: `backend/tests/unit/research/test_events_store_contract.py` (contract test with a fake `ChromaStore` double)

**Interfaces:**
- Produces: `write_events_sync(events: list[dict], uid: str) -> int` (each event `{time, topic, action, result, conversation_id}`; embed text = `"{topic} | {action} | {result}"`; metadata mirrors user_memory + `conversation_id`); `search_events_sync(query, top_k=5, uid=None) -> list[dict]`.

- [ ] **Step 1:** Write contract test: inject a fake store capturing `insert_batch`; assert `write_events_sync` embeds the composed text, sets `metadata.type=="event"`, carries `conversation_id`/`topic`/`action`/`result`/`created_at`, and filters `search_events_sync` results by `uid`. (Fake store returns canned rows; assert uid filter + top_k.)
- [ ] **Step 2:** Run → FAIL (methods absent).
- [ ] **Step 3:** Implement in `KnowledgeSearch`:
  - `from_settings`: `self.mem_events_store = ChromaStore(settings=settings, collection_name=f"mem_events{collection_suffix}")` (thread through `__init__` param `mem_events_store=None`).
  - `write_events_sync`: build texts `f"{e['topic']} | {e['action']} | {e['result']}"`, `self.dense_encoder.embed(texts)`, `insert_batch` with metadata `{type:"event", uid, conversation_id, created_at:e['time'], topic, action, result, text}`. (Mirror `write_user_memory_sync` structure; no 0.7 gate — events aren't confidence-scored.)
  - `search_events_sync`: mirror `search_user_memory_sync` (vector over-retrieve → uid filter → BM25 → RRF → rerank → decay) against `mem_events_store`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(memory): mem_events collection + write/search in KnowledgeSearch (P2 t1)`.

### Task P2.2: consolidation emits events; retire the extractor

**Files:**
- Modify: `backend/nanoresearch/agent/memory.py` (`_SAVE_MEMORY_TOOL` add `events`; `_CONSOLIDATION_SYSTEM_PROMPT` instruct events; `consolidate()` write events; drop `history_entry`→`user_memory` `consolidation_summary` write — moves to P3)
- Modify: `backend/nanoresearch/agent/memory.py` `MemoryConsolidator._extract_conversation_knowledge` → remove (delete `ConversationKnowledgeExtractor` usage)
- Delete: `backend/nanoresearch/agent/conversation_knowledge_extractor.py`
- Test: `backend/tests/unit/agent/test_consolidation_events.py`

**Interfaces:**
- Consumes: `write_events_sync` (P2.1).
- Produces: `save_memory` tool arg `events: [{topic, action, result}]`; `consolidate()` calls `knowledge_search.write_events_sync(events, uid)` with `conversation_id` + `time` filled from the chunk/session.

- [ ] **Step 1:** Write test: fake provider returns tool args with `events:[{topic,action,result}]`; fake `KnowledgeSearch` captures `write_events_sync`; assert events written with `conversation_id` + `time`. Assert extractor no longer imported/called.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Add `events` to `_SAVE_MEMORY_TOOL` params + prompt section (extract atomic `{topic, action, result}`; ignore agent replies). In `consolidate()`, after parsing args, call `write_events_sync`. Remove the `_extract_conversation_knowledge` call in `consolidate_messages` and delete the method + the extractor module + its tests. `conversation_id` derived from `session.key`/passed through (thread a `conversation_id` param into `consolidate`/`consolidate_messages` from the loop, defaulting to parsing `session.key`).
- [ ] **Step 4:** Run → PASS; update `test_consolidation_confidence.py` (its `consolidation_summary` assertion moves to P3 — temporarily assert events written).
- [ ] **Step 5:** Commit `feat(memory): consolidation emits atomic events; delete ConversationKnowledgeExtractor (P2 t2)`.

### Task P2.3: repoint `<history>` recall to events

**Files:**
- Modify: `backend/nanoresearch/agent/context.py` (`build_history_context` → `search_events_sync`)
- Test: `backend/tests/unit/agent/test_history_context_events.py`

- [ ] **Step 1:** Test: fake `KnowledgeSearch.search_events_sync` returns rows; assert `build_history_context(query, uid)` renders them into the `## 相关历史记忆` block (same format), and calls `search_events_sync` (not `search_user_memory_sync`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Change `build_history_context` to call `self.knowledge_search.search_events_sync(query, top_k=5, uid=uid)`; keep the render/degrade-to-empty logic.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(memory): <history> recall reads mem_events (P2 t3)`.

---

## Phase P3 — Conversation-summary sliding window

### Task P3.1: `mem_conv_summaries` collection + write/search (conv-scoped)

**Files:**
- Modify: `backend/nanoresearch/research/knowledge_search.py` (`mem_conv_summaries_store`; `write_conv_summary_sync`, `search_conv_summaries_sync`, `list_conv_summaries_sync`)
- Test: `backend/tests/unit/research/test_conv_summary_store_contract.py`

**Interfaces:**
- Produces: `write_conv_summary_sync(text, uid, conversation_id, turn_start, turn_end, topic) -> str`; `search_conv_summaries_sync(query, uid, conversation_id, top_k=5, exclude_ids=None) -> list[dict]` (semantic, filtered to conversation_id); `list_conv_summaries_sync(uid, conversation_id) -> list[dict]` (all for a conversation, metadata incl turn_start/turn_end — cheap listing for the recent window).

- [ ] **Step 1:** Contract test with fake store: write carries `conversation_id`+`turn_start/turn_end`+`topic`; `search` filters by both `uid` and `conversation_id`; `list` returns all for the conversation.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement mirroring `user_memory` methods, metadata `{type:"conv_summary", uid, conversation_id, turn_start, turn_end, topic, created_at, text}`; `search` adds `conversation_id` metadata filter + `exclude_ids`; `list` = query all with metadata filter (large top_k).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(memory): mem_conv_summaries collection (P3 t1)`.

### Task P3.2: sliding-window selection (pure) + consolidation routes summary here

**Files:**
- Modify: `backend/nanoresearch/agent/memory_facts.py` (add pure `select_recent_window`)
- Modify: `backend/nanoresearch/agent/memory.py` (`consolidate()` routes `summary`/`history_entry` → `write_conv_summary_sync`, not `user_memory`)
- Test: `backend/tests/unit/agent/test_conv_summary_window.py`, extend `test_consolidation_events.py`

**Interfaces:**
- Produces: `select_recent_window(summaries: list[dict], budget_tokens: int, est_tokens) -> tuple[list, list]` → `(near, far_remainder)`: greedily take newest by `turn_end` desc until `budget_tokens*recent_ratio` consumed; `near` returned in turn-ascending order, `far_remainder` = the rest (candidates for semantic recall).

- [ ] **Step 1:** Test `select_recent_window`: given 5 summaries with known token estimates + a budget, assert `near` = newest-fit set in ascending turn order and `far_remainder` = the rest; boundary cases (budget fits all / none).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `select_recent_window` (pure). In `consolidate()`, replace the `history_entry`→`user_memory` write with `write_conv_summary_sync(summary_text, uid, conversation_id, turn_start, turn_end, topic)` (turn_start/end from the chunk boundary).
- [ ] **Step 4:** Run → PASS; fix `test_consolidation_confidence.py` to assert conv_summary written via `write_conv_summary_sync`.
- [ ] **Step 5:** Commit `feat(memory): conv summary write + recent-window selection (P3 t2)`.

### Task P3.3: `<conversation_summary>` injection (near deterministic + far semantic)

**Files:**
- Modify: `backend/nanoresearch/agent/context.py` (`_build_dynamic_suffix`: add `<conversation_summary>` block; thread `conversation_id`)
- Test: `backend/tests/unit/agent/test_conv_summary_injection.py`

**Interfaces:**
- Consumes: `list_conv_summaries_sync`, `search_conv_summaries_sync` (P3.1), `select_recent_window` (P3.2).
- Produces: `<conversation_summary>` block = near window (deterministic, ~60% conv_summary budget, turn-ascending) + far semantic recall (`search_conv_summaries_sync` conv-filtered, `exclude_ids`=near ids, ~40%). Injection order: `<memory>` → `<conversation_summary>` → `<history>`.

- [ ] **Step 1:** Test with fake store: given this conv's summaries, assert near set injected deterministically (recent, turn order) + far via semantic search filtered to `conversation_id`, excluding near; assert block absent when no `conversation_id`/no summaries; assert size bounded by budget.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement in `_build_dynamic_suffix` (thread `conversation_id` from `build_system_prompt`/`build_messages` callers in `loop.py`). Budget split via `CONV_SUMMARY_RECENT_RATIO` (default 0.6).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(memory): <conversation_summary> sliding-window injection (P3 t3)`.

---

## Phase P4 — Rebuild from PG + retire old stores

### Task P4.1: rebuild orchestration (chunking) — unit-tested with mocked consolidator

**Files:**
- Create: `backend/nanoresearch/scripts/rebuild_memory_from_pg.py`
- Test: `backend/tests/unit/scripts/test_rebuild_memory_chunking.py`

**Interfaces:**
- Produces: `plan_rebuild_chunks(messages, tail_protect=0) -> list[(start,end)]` (pure, reuse `pick_consolidation_boundary`-style user-turn boundaries over the whole conversation); `async rebuild_uid(uid, repo, consolidator)` orchestration calling `consolidator.consolidate_messages(chunk, uid=uid, conversation_id=...)` per chunk.

- [ ] **Step 1:** Test `plan_rebuild_chunks` splits a synthetic conversation at user-turn boundaries covering all messages; test `rebuild_uid` with a **mocked consolidator** asserts one call per chunk with correct `conversation_id`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the script: iterate `ConversationRepository` per uid, `get_messages` per conversation, chunk, call the consolidator. CLI entry `python -m nanoresearch.scripts.rebuild_memory_from_pg [--uid ...]`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(memory): rebuild-from-PG orchestration + chunking (P4 t1)`.

### Task P4.2: retire `user_memory` + dead code

**Files:**
- Modify: `backend/nanoresearch/research/knowledge_search.py` (remove `user_memory_store` + `write_user_memory*`/`search_user_memory*`/`cleanup_old_user_memory` once no callers remain), `from_settings`
- Modify: any remaining callers (grep `user_memory`, `search_user_memory_sync`, `write_user_memory`)
- Test: reverse-grep guard test that no source imports the extractor or calls `*_user_memory*`

- [ ] **Step 1:** Test: `test_no_legacy_memory_refs` greps `nanoresearch/` for `user_memory`, `ConversationKnowledgeExtractor`, `search_user_memory` → asserts none (excluding migration/rename shims).
- [ ] **Step 2:** Run → FAIL (refs remain).
- [ ] **Step 3:** Remove the methods + store construction; delete the migrate_to_user_memory script if now dead; update `from_settings`.
- [ ] **Step 4:** Run → PASS; run full memory/agent/storage regression.
- [ ] **Step 5:** Commit `refactor(memory): retire user_memory + legacy extractor (P4 t2)`.

### Task P4.3 (manual, documented): bulk rebuild run

- [ ] Documented manual step: with a running env (LLM API + PG), run `python -m nanoresearch.scripts.rebuild_memory_from_pg`, then drop the physical `user_memory` Chroma collection. NOT run autonomously.

---

## Self-Review

- **Spec coverage:** events layer (P2.1–2.3) ✓; conv-summary sliding window near+far, conv_id-scoped (P3.1–3.3) ✓ (spec §4.2 + §6); Model B (no separate global digest — `<history>`=events recall) ✓; migration nuke+rebuild (P4) ✓; extractor deletion + user_memory retirement ✓.
- **Placeholder scan:** adapter methods reference the concrete `user_memory` analog + exact metadata; sliding-window + chunking are pure with explicit signatures. LLM-dependent bulk run is explicitly a manual step (not a hidden TODO).
- **Type consistency:** `write_events_sync`/`search_events_sync`, `write_conv_summary_sync`/`search_conv_summaries_sync`/`list_conv_summaries_sync`, `select_recent_window` names stable across tasks; `conversation_id` threaded consistently.
- **Risk:** consolidation keeps P1 fallback; injection block guarded by `conversation_id` presence; `user_memory` removal gated behind a reverse-grep test so no dangling callers.
