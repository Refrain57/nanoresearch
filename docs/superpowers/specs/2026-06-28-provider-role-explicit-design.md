# Provider Role Explicit — Design Spec

**Date:** 2026-06-28
**Status:** Approved for plan
**Scope:** Two related fixes shipped together:
1. Add missing `TunableObjectVersion` ORM class (unblocks server boot)
2. Replace heuristic provider/role detection with explicit provider pool + role assignment

## Why

After Phase 5 multitenant LLM config shipped, the user attempted
`uv run -m nanoresearch serve` and hit two issues:

**Issue 1 — server cannot start:**
```
ImportError: cannot import name 'TunableObjectVersion' from 'nanoresearch.storage.models'
```
The table `tunable_object_versions` exists in the DB with a verified schema
(`id uuid pk default gen_random_uuid()`, `kind varchar`, `target_id varchar`,
`content text`, `active boolean default false`, `created_at timestamptz`,
`created_by varchar nullable`). 17 references in `agent_eval_repo.py:18,
494-551`, consumed by `tunable.py` (lines 243, 253, 257, 339, 364, 368) and
`agent_eval_router.py:1638, 1671`. `server/main.py:123` imports
`agent_eval_router` inside `create_app()`, so the ImportError propagates and
the server cannot boot. The ORM class is simply missing from `models.py`.
This is a pre-existing bug Phase 5 worked around (it blocked Task 6's
`pytest.skip` plan and Task 9's manual e2e).

**Issue 2 — provider UX is too coarse:**
The current `AppLayout.vue` providers tab (lines 55-203) has these problems:
- `<a-input v-model:value="providerForm.name" placeholder="如 通义千问、OpenAI、DeepSeek" />` (line 174) is **free-text**, no preset dropdown — users can type "tongyi" vs "qwen" vs "dashscope" and the heuristic name match breaks.
- The "至少需要填两组 key" alert (line 73, copy at line 281-284) misleads users that exactly 2 keys are required, with role assignment implicit from provider name.
- `stores/settings.js:20-30` derives `coverage.hasEmbedding` by a heuristic — hard-coded provider name allow-list (`embProviderNames = ['dashscope', 'openai', 'azure_openai', 'siliconflow']`) plus a `/embed/i` regex on `p.models`. Brittle and silently wrong when user adds a new embedding provider not in the list, or names their provider "我的阿里" instead of "dashscope".
- No way to say "use provider A's `deepseek-chat` for chat, provider B's `text-embedding-v3` for embedding, provider C for vision" — `ModelFactory._resolve_chat` etc. fall back to `_match_user_provider_by_model` (first provider with matching model name) which means any model-name collision routes to whoever was added first.
- 6 backend roles exist (`chat`, `ingestion_llm`, `embedding`, `vision`, `eval_generator`, `eval_evaluator` per `model_factory.py:35-41`) but only the heuristic 2 are surfaced.

User feedback (verbatim):
> "当前前端，填apikey的位置，不太对，你只让填两组，然后没有默认的下拉框，且char和embedding那边应该是分开的也就是前端的粒度不够细"

Translated: "the API-key UI is wrong — you only let me fill two slots, no default dropdown, and chat vs. embedding should be separate. Granularity too coarse."

## Design

### Issue 1: TunableObjectVersion ORM class

Add a SQLAlchemy ORM class to `backend/nanoresearch/storage/models.py` matching the existing DB schema verbatim:

```python
class TunableObjectVersion(Base):
    __tablename__ = "tunable_object_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    kind: Mapped[str] = mapped_column(String, nullable=False)
    target_id: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"),
    )
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)
```

Adds `text` to the `sqlalchemy` import line at top of `models.py`.

**No migration needed** — DB table already exists; this is just the missing
Python binding. Verify by importing `TunableObjectVersion` and calling
`list_tunable_versions` on an empty DB.

### Issue 2: Provider role explicit

Two-section UX. Keep the existing providers list (the "pool"), add a new
roles section below it.

#### Section A — Provider pool (credentials)

Each provider is a credential, not a role. Fields:
- `id` (uuid, generated server-side)
- `provider` (NEW — **dropdown** with preset values; replaces freeform `name`)
- `name` (label — keep as freeform, but treat as display label only)
- `api_key`
- `api_base` (only required for `openai_compatible`)
- `models` (kept for compatibility — informational dropdown options downstream)

Preset values for the `provider` dropdown:
- `deepseek`
- `openai`
- `anthropic`
- `dashscope` (阿里云通义)
- `azure_openai`
- `siliconflow`
- `openai_compatible` (兜底 — 启用 `api_base` 必填)

The frontend shows these via an `a-select` with `:options` derived from a
constant `PROVIDER_PRESETS`. Each preset carries its model presets (already
present in `PROVIDER_MODEL_PRESETS` at `AppLayout.vue:314-323`). The free-text
`name` becomes a user-supplied label ("我的 DeepSeek"); `provider` is the
canonical preset identifier the backend matches against.

#### Section B — Role assignment (usage)

A new section in the providers tab, listed below the provider pool:

```
模型用途              Provider               Model
─────────────────────────────────────────────────────────────
聊天 (chat)           [我的 DeepSeek ▾]      [deepseek-chat]
RAG 摄取              [我的 DeepSeek ▾]      [deepseek-chat]
向量嵌入              [通义 ▾]               [text-embedding-v3]
视觉                  [未配置 ▾]             [—]            ← 留空可
评测生成              [未配置 ▾]             [—]            ← 留空 fallback chat
评测打分              [未配置 ▾]             [—]            ← 留空 fallback chat
```

- The Provider column is an `a-select` whose options are the providers in
  Section A (label = `name (provider)`, value = `provider.id`).
- The Model column is `a-auto-complete` with options drawn from the selected
  provider's `models[]` (since model naming varies wildly, hard dropdown
  isn't practical — auto-complete with the preset list is the sweet spot).
- "未配置" / null is allowed; the backend falls back to the existing
  config.json / settings.yaml chain (local mode) or raises `missing_provider`
  422 (server mode) per Phase 5.
- vision / eval_* default to null. Chat / ingestion / embedding get
  auto-assigned per the defaults below.

#### Default auto-assignment (first time user)

When the user adds their **first** provider:
- Auto-assign that provider to `chat` and `ingestion_llm` roles.
- Default model = first model in `provider.models[]`, or empty (auto-complete will hint).

When the user adds a **second** provider whose `provider` ∈ `{dashscope, openai, azure_openai, siliconflow}`:
- Auto-assign that provider to `embedding`.
- Default model = an embedding-flavored name from `provider.models[]` (heuristic: first model whose name contains `embed`), else empty.

No auto-assignment for vision / eval_* — these are explicit opt-in.

The user can always reassign or clear any role.

#### Data shape

`user_settings.extra` JSONB upgrades to:

```json
{
  "fast_model": "...",
  "providers": [
    {
      "id": "uuid1",
      "provider": "deepseek",
      "name": "我的 DeepSeek",
      "api_key": "sk-...",
      "api_base": null,
      "models": ["deepseek-chat", "deepseek-reasoner"]
    }
  ],
  "roles": {
    "chat":            {"provider_id": "uuid1", "model": "deepseek-chat"},
    "ingestion_llm":   {"provider_id": "uuid1", "model": "deepseek-chat"},
    "embedding":       {"provider_id": "uuid2", "model": "text-embedding-v3"},
    "vision":          null,
    "eval_generator":  null,
    "eval_evaluator":  null
  },
  "ragas_generator_model": "...",
  "ragas_evaluator_model": "...",
  "ragas_embedding_model": "..."
}
```

#### Legacy migration

Old format (pre-this-spec):
- `extra.providers[]` items have `name` but no `provider` field.
- `extra.roles` is absent.

On first read of a user_settings row that has `providers` but lacks `roles`,
the backend (`user_settings_repo.get`) runs a one-shot migration:
1. For each provider, infer `provider` field from `name` lowercase substring
   match against the preset list (`deepseek`/`openai`/...). If no match,
   set `provider = "openai_compatible"`.
2. Build `roles`:
   - `chat` and `ingestion_llm` → first provider with `api_key`.
   - `embedding` → first provider whose `provider` ∈ embedding-capable set
     (`dashscope`/`openai`/`azure_openai`/`siliconflow`) and has `api_key`;
     else null.
   - `vision`/`eval_*` → null.
3. Write the migrated `extra` back to DB; invalidate Redis cache.

Migration is idempotent (only runs when `roles` key absent).

#### Backend resolution

`ModelFactory._resolve_chat / _resolve_ingestion_llm / _resolve_embedding /
_resolve_vision / _resolve_eval_*` gain a new first lookup: consult
`user_settings.extra.roles[role.value]`. If present and `provider_id`
resolves to a provider in `user_providers`, use that provider's `api_key`,
`api_base`, and the role's `model` field (falling back to provider's first
model if role.model is empty).

If `roles[role]` is null:
- **server mode**: existing `_resolve_from_user_only` behavior (try
  `_match_user_provider_by_model`, then fall back to first provider with
  api_key; raise `missing_provider` 422 if none).
- **local mode**: existing fall-through chain (config.json / rag_settings /
  env var).

The existing `_match_user_provider_by_model` and `_match_user_provider_by_name`
helpers stay as fallback for `roles[role]` being null but a model being
passed in via `user_model` or `model_override`.

#### Coverage rewrite

`stores/settings.js:20-30` `coverage` computed:

```javascript
const coverage = computed(() => {
  const roles = rolesMap.value || {}
  return {
    hasChat:      !!roles.chat?.provider_id,
    hasEmbedding: !!roles.embedding?.provider_id,
  }
})
```

The heuristic `embProviderNames` array and `/embed/i` regex go away.

#### Misleading alert removal

The `<a-alert message="至少需要填两组 key" ... />` at `AppLayout.vue:69-75` and
the `providerGuideDesc` const at lines 281-284 are removed. Replace with a
short, accurate hint at the top of the new role assignment section:

> 添加 API key 后，指定每种模型用途使用哪个 provider。Chat 和 Embedding 是
> 最常用的两个；其它留空时按 fallback 链处理。

## Out of scope

- New providers beyond the 7 preset list (user can use `openai_compatible`
  with custom `api_base` as the catch-all).
- Per-role rate limits / quotas.
- DB schema migration (extra is JSONB; in-row migration is enough).
- The `.gitignore` encoding bug and untracked file disposition (separate
  cleanup).
- Phase 6 backlog items from the original cleanup spec (long-term).

## Acceptance criteria

1. `uv run -m nanoresearch serve` boots without `TunableObjectVersion` ImportError.
2. `pytest backend/tests/test_phase5_*` and the new `test_role_assignment.py`
   tests pass.
3. Frontend `pnpm build` clean.
4. Manual e2e: add 2 providers, assign chat to first + embedding to second,
   verify settings persist after browser refresh, verify a chat request and
   a knowledge-ingestion request both succeed using the assigned provider's
   key (no env fallback in server mode).
5. Existing users (with old `extra` schema) auto-migrate on first read —
   role assignment populated, no broken state.
6. Misleading "至少 2 组 key" alert is gone; no user is forced to add 2
   keys if they only need chat.

## File touch list

**Backend:**
- `backend/nanoresearch/storage/models.py` — add `TunableObjectVersion`, add `text` import
- `backend/nanoresearch/storage/repositories/user_settings_repo.py` — add `_migrate_legacy_extra(extra)`, call from `get()`
- `backend/nanoresearch/providers/model_factory.py` — add `_match_user_provider_by_id`, consult `roles` map at top of each resolver
- `backend/nanoresearch/server/routers/settings_router.py` — add `provider` field to `ProviderIn`, add `roles` field to `UserSettingsUpdate`, surface in `_to_dict` and merge logic
- `backend/tests/test_role_assignment.py` — NEW; 6 cases covering by-id resolution, fallback, migration, server-mode 422

**Frontend:**
- `web/src/stores/settings.js` — add `roles` ref, `saveRoles` action, rewrite `coverage`
- `web/src/layouts/AppLayout.vue` — provider modal: add `provider` preset dropdown; new role assignment section; remove misleading alert; remove `providerGuideDesc`; add default-on-first-add logic

## Open questions

None — design approved in conversation 2026-06-28.
