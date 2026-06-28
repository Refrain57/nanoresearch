# Provider Role Explicit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add missing `TunableObjectVersion` ORM class (unblocks server boot) and replace heuristic provider/role detection with explicit provider pool + role assignment UX.

**Architecture:** Backend gains explicit `roles` map in `user_settings.extra` JSONB, consulted first by `ModelFactory` resolvers. Frontend adds a provider-preset dropdown (replacing freeform name input) and a role-assignment table mapping each of the 6 backend `ModelRole` values to a `(provider_id, model)` pair. Legacy `extra` schemas auto-migrate on first read.

**Tech Stack:** SQLAlchemy 2.x (Mapped/mapped_column), FastAPI + pydantic v2, asyncpg, loguru, pytest, Vue 3 `<script setup>`, Pinia, ant-design-vue v4.

## Global Constraints

- Server mode (`NANORESEARCH_MODE=server`) must NEVER read api keys from env vars — preserve Phase 5 gating (`env_key_or_raise`, `env_key_fallback_allowed`).
- Server mode + unresolvable role MUST raise `ModelResolutionError` → 422 `missing_provider` with `role` field (per Phase 5 Task 6).
- Backend roles enumerated in `ModelRole`: `chat`, `ingestion_llm`, `embedding`, `vision`, `eval_generator`, `eval_evaluator`. Never invent new values.
- Provider preset list (canonical, copy verbatim): `deepseek`, `openai`, `anthropic`, `dashscope`, `azure_openai`, `siliconflow`, `openai_compatible`.
- `extra` JSONB stays one row; no new DB table.
- Migration is idempotent — runs only when `roles` key is absent.
- Frontend uses ant-design-vue v4 conventions already in `AppLayout.vue` (a-select, a-form-item, a-tag, etc.).
- TDD: each task starts with a failing test, then minimal code to pass.
- Commit per task. Commit messages follow existing repo style (`fix(...)`, `feat(...)`, lower-case scope).

---

### Task 1: Add TunableObjectVersion ORM class

**Files:**
- Modify: `backend/nanoresearch/storage/models.py:8` (import line) and append new class
- Test: `backend/tests/test_tunable_object_version_model.py` (NEW)

**Interfaces:**
- Consumes: nothing
- Produces: `TunableObjectVersion` ORM class with fields `id: UUID`, `kind: str`, `target_id: str`, `content: str`, `active: bool`, `created_at: datetime`, `created_by: str | None`. Importable as `from nanoresearch.storage.models import TunableObjectVersion`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_tunable_object_version_model.py`:

```python
"""Verify TunableObjectVersion ORM class exists with the expected schema."""

from __future__ import annotations


def test_tunable_object_version_importable():
    """Class must be importable from nanoresearch.storage.models."""
    from nanoresearch.storage.models import TunableObjectVersion

    assert TunableObjectVersion.__tablename__ == "tunable_object_versions"


def test_tunable_object_version_columns():
    """Class must declare id/kind/target_id/content/active/created_at/created_by columns."""
    from nanoresearch.storage.models import TunableObjectVersion

    columns = {c.name for c in TunableObjectVersion.__table__.columns}
    assert columns == {
        "id",
        "kind",
        "target_id",
        "content",
        "active",
        "created_at",
        "created_by",
    }


def test_tunable_object_version_id_is_uuid_pk():
    from nanoresearch.storage.models import TunableObjectVersion

    pk = [c for c in TunableObjectVersion.__table__.primary_key.columns]
    assert len(pk) == 1
    assert pk[0].name == "id"


def test_agent_eval_repo_imports_cleanly():
    """The whole reason this class exists — repo must import without ImportError."""
    from nanoresearch.storage.repositories import agent_eval_repo  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_tunable_object_version_model.py -v`

Expected: 4 FAILED — `ImportError: cannot import name 'TunableObjectVersion' from 'nanoresearch.storage.models'`

- [ ] **Step 3: Modify the sqlalchemy import in `backend/nanoresearch/storage/models.py:8`**

Change line 8 from:
```python
from sqlalchemy import ARRAY, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
```
to:
```python
from sqlalchemy import ARRAY, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
```

- [ ] **Step 4: Append the ORM class to `backend/nanoresearch/storage/models.py`**

Append at end of file (after the last existing class):

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

- [ ] **Step 5: Run test to verify it passes**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_tunable_object_version_model.py -v`

Expected: 4 PASSED

- [ ] **Step 6: Smoke-check server boot path**

Run: `./backend/.venv/Scripts/python -c "from nanoresearch.server.main import create_app; create_app(); print('OK')"`

Expected: `OK` printed. (If a different unrelated ImportError appears, document it but proceed — it is out of scope.)

- [ ] **Step 7: Commit**

```bash
git add backend/nanoresearch/storage/models.py backend/tests/test_tunable_object_version_model.py
git commit -m "fix(storage): add missing TunableObjectVersion ORM class"
```

---

### Task 2: Backend — extend settings_router pydantic + roles support

**Files:**
- Modify: `backend/nanoresearch/server/routers/settings_router.py:24-37` (ProviderIn + UserSettingsUpdate), `:53-86` (mask + merge helpers), `:89-99` (_to_dict)
- Test: `backend/tests/test_settings_roles_schema.py` (NEW)

**Interfaces:**
- Consumes: existing `UserSettingsRepository.get` / `.upsert`.
- Produces:
  - `ProviderIn` Pydantic model gains `provider: str | None = None` field (preset key like `"deepseek"`).
  - `UserSettingsUpdate` Pydantic model gains `roles: dict[str, dict | None] | None = None` field.
  - Stored `extra.providers[i]` dicts gain `"provider"` key (None allowed for legacy rows).
  - Stored `extra.roles` is a dict mapping role name → `{"provider_id": str, "model": str | None}` or None.
  - GET `/api/settings/me` response gains `"roles": {role_name: {provider_id, model} | None}` field; provider items in the response gain `"provider"` field (preset name, may be None for un-migrated legacy).
  - PUT `/api/settings/me` accepts `roles` field; missing → keep existing; null per role → clear that role.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_settings_roles_schema.py`:

```python
"""Pydantic schema + merge helpers for the new roles map and provider preset field."""

from __future__ import annotations

import pytest

from nanoresearch.server.routers.settings_router import (
    ProviderIn,
    UserSettingsUpdate,
    _merge_providers,
    _to_dict,
)


def test_provider_in_accepts_provider_preset_field():
    p = ProviderIn(name="我的 DeepSeek", provider="deepseek", api_key="sk-x")
    assert p.provider == "deepseek"


def test_provider_in_provider_field_defaults_to_none():
    p = ProviderIn(name="legacy", api_key="sk-x")
    assert p.provider is None


def test_user_settings_update_accepts_roles_field():
    u = UserSettingsUpdate(roles={
        "chat": {"provider_id": "uuid-1", "model": "deepseek-chat"},
        "embedding": None,
    })
    assert u.roles["chat"]["provider_id"] == "uuid-1"
    assert u.roles["embedding"] is None


def test_merge_providers_preserves_provider_preset_field():
    existing = [{"id": "u1", "name": "X", "provider": "deepseek", "api_key": "sk-old", "api_base": None, "models": []}]
    incoming = [ProviderIn(id="u1", name="X", provider="deepseek", api_key=None, models=[])]
    merged = _merge_providers(existing, incoming)
    assert merged[0]["provider"] == "deepseek"
    assert merged[0]["api_key"] == "sk-old"  # None means keep


def test_merge_providers_accepts_new_provider_field_on_new_row():
    incoming = [ProviderIn(name="New", provider="openai", api_key="sk-new", models=[])]
    merged = _merge_providers([], incoming)
    assert merged[0]["provider"] == "openai"
    assert merged[0]["api_key"] == "sk-new"


def test_to_dict_surfaces_roles_and_provider_field():
    class Row:
        model = None
        max_iterations = None
        extra = {
            "providers": [
                {"id": "u1", "name": "X", "provider": "deepseek", "api_key": "sk", "api_base": None, "models": []}
            ],
            "roles": {
                "chat": {"provider_id": "u1", "model": "deepseek-chat"},
                "embedding": None,
            },
        }

    out = _to_dict(Row(), defaults={})
    assert out["roles"]["chat"]["provider_id"] == "u1"
    assert out["roles"]["embedding"] is None
    assert out["providers"][0]["provider"] == "deepseek"
    assert out["providers"][0]["api_key_set"] is True
    assert "api_key" not in out["providers"][0]  # masked
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_settings_roles_schema.py -v`

Expected: Most tests FAIL with `pydantic.ValidationError` (extra field `provider` / `roles` not permitted) or `KeyError` (no `roles` in `_to_dict` output).

- [ ] **Step 3: Modify `ProviderIn` at `backend/nanoresearch/server/routers/settings_router.py:24-29`**

Replace the class with:

```python
class ProviderIn(BaseModel):
    id: str | None = None
    name: str
    provider: str | None = None   # preset key: deepseek/openai/anthropic/dashscope/azure_openai/siliconflow/openai_compatible
    api_key: str | None = None    # None = keep existing; "" = clear
    api_base: str | None = None
    models: list[str] = []
```

- [ ] **Step 4: Modify `UserSettingsUpdate` at `backend/nanoresearch/server/routers/settings_router.py:32-40`**

Replace the class with:

```python
class UserSettingsUpdate(BaseModel):
    # "" means "clear to null"; None means "don't touch"
    model: str | None = None
    fast_model: str | None = None
    max_iterations: int | None = None
    providers: list[ProviderIn] | None = None
    roles: dict[str, dict | None] | None = None
    ragas_generator_model: str | None = None
    ragas_evaluator_model: str | None = None
    ragas_embedding_model: str | None = None
```

- [ ] **Step 5: Modify `_merge_providers` at `backend/nanoresearch/server/routers/settings_router.py:67-86`**

Replace the function with:

```python
def _merge_providers(existing: list[dict], incoming: list[ProviderIn]) -> list[dict]:
    """Merge provider list, preserving stored api_keys when client sends None."""
    existing_map = {p["id"]: p for p in existing if p.get("id")}
    result = []
    for p in incoming:
        pid = p.id or str(_uuid.uuid4())
        base = existing_map.get(pid, {})
        # api_key=None → keep existing; api_key="" → clear; api_key="sk-xxx" → update
        if p.api_key is None:
            api_key = base.get("api_key", "")
        else:
            api_key = p.api_key  # "" clears it, non-empty sets it
        result.append({
            "id": pid,
            "name": p.name,
            "provider": p.provider if p.provider is not None else base.get("provider"),
            "api_base": p.api_base,
            "models": p.models,
            "api_key": api_key,
        })
    return result
```

- [ ] **Step 6: Modify `_mask_providers` at `backend/nanoresearch/server/routers/settings_router.py:53-64`**

Replace the function with:

```python
def _mask_providers(providers: list[dict]) -> list[dict]:
    return [
        {
            "id": p.get("id"),
            "name": p.get("name", ""),
            "provider": p.get("provider"),
            "api_base": p.get("api_base"),
            "api_key_set": bool(p.get("api_key")),
            "api_key_hint": _mask_key(p.get("api_key") or ""),
            "models": p.get("models", []),
        }
        for p in providers
    ]
```

- [ ] **Step 7: Modify `_to_dict` at `backend/nanoresearch/server/routers/settings_router.py:89-99`**

Replace the function with:

```python
def _to_dict(row, defaults: dict) -> dict:
    extra = (row.extra if row else None) or {}
    return {
        "model": (row.model if row else None) or defaults.get("model"),
        "fast_model": extra.get("fast_model"),
        "max_iterations": (row.max_iterations if row else None) or defaults.get("max_iterations"),
        "providers": _mask_providers(extra.get("providers", [])),
        "roles": extra.get("roles") or {},
        "ragas_generator_model": extra.get("ragas_generator_model") or defaults.get("ragas_generator_model"),
        "ragas_evaluator_model": extra.get("ragas_evaluator_model") or defaults.get("ragas_evaluator_model"),
        "ragas_embedding_model": extra.get("ragas_embedding_model") or defaults.get("ragas_embedding_model"),
    }
```

- [ ] **Step 8: Wire roles update through PUT `/api/settings/me`**

Modify `update_settings` at `backend/nanoresearch/server/routers/settings_router.py:125-169`. Insert after the existing `providers` handling block (after line 149, before the `for field, key in [...]` loop):

```python
    if "roles" in sent and body.roles is not None:
        extra["roles"] = body.roles
        extra_changed = True
```

- [ ] **Step 9: Run schema tests to verify they pass**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_settings_roles_schema.py -v`

Expected: 6 PASSED

- [ ] **Step 10: Commit**

```bash
git add backend/nanoresearch/server/routers/settings_router.py backend/tests/test_settings_roles_schema.py
git commit -m "feat(settings): add provider preset field and roles map to user settings schema"
```

---

### Task 3: Backend — legacy extra migration in user_settings_repo

**Files:**
- Modify: `backend/nanoresearch/storage/repositories/user_settings_repo.py:33-65` (UserSettingsRepository.get)
- Test: `backend/tests/test_user_settings_migration.py` (NEW)

**Interfaces:**
- Consumes: `UserSettings.extra` dict (may lack `roles` and/or per-provider `provider` field).
- Produces: A module-level helper `_migrate_legacy_extra(extra: dict) -> tuple[dict, bool]` returning `(possibly-migrated extra, changed: bool)`. `get()` calls it; if `changed`, writes the migrated extra back via `upsert(uid, extra=...)` before returning the row.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_user_settings_migration.py`:

```python
"""Idempotent legacy-extra migration for the new provider preset + roles fields."""

from __future__ import annotations

from nanoresearch.storage.repositories.user_settings_repo import _migrate_legacy_extra


def test_legacy_extra_no_roles_gets_roles_inferred():
    extra = {
        "providers": [
            {"id": "u1", "name": "我的 DeepSeek", "api_key": "sk-1", "api_base": None, "models": ["deepseek-chat"]},
            {"id": "u2", "name": "通义 dashscope", "api_key": "sk-2", "api_base": "https://x", "models": ["qwen-plus", "text-embedding-v3"]},
        ]
    }
    out, changed = _migrate_legacy_extra(extra)
    assert changed is True
    assert out["providers"][0]["provider"] == "deepseek"
    assert out["providers"][1]["provider"] == "dashscope"
    assert out["roles"]["chat"]["provider_id"] == "u1"
    assert out["roles"]["ingestion_llm"]["provider_id"] == "u1"
    assert out["roles"]["embedding"]["provider_id"] == "u2"
    assert out["roles"]["vision"] is None
    assert out["roles"]["eval_generator"] is None
    assert out["roles"]["eval_evaluator"] is None


def test_legacy_extra_unknown_name_falls_back_to_openai_compatible():
    extra = {"providers": [{"id": "u1", "name": "MyCustom", "api_key": "sk", "api_base": "https://x", "models": []}]}
    out, changed = _migrate_legacy_extra(extra)
    assert changed is True
    assert out["providers"][0]["provider"] == "openai_compatible"


def test_migration_is_idempotent_when_roles_present():
    extra = {
        "providers": [{"id": "u1", "name": "X", "provider": "deepseek", "api_key": "sk", "api_base": None, "models": []}],
        "roles": {"chat": {"provider_id": "u1", "model": "deepseek-chat"}, "embedding": None},
    }
    out, changed = _migrate_legacy_extra(extra)
    assert changed is False
    assert out is extra  # no copy when no work


def test_migration_skips_when_no_providers():
    extra = {"fast_model": "x"}
    out, changed = _migrate_legacy_extra(extra)
    assert changed is False


def test_embedding_role_left_null_when_no_embedding_capable_provider():
    extra = {
        "providers": [
            {"id": "u1", "name": "DeepSeek", "api_key": "sk", "api_base": None, "models": []},
        ]
    }
    out, changed = _migrate_legacy_extra(extra)
    assert changed is True
    assert out["roles"]["chat"]["provider_id"] == "u1"
    assert out["roles"]["embedding"] is None


def test_chat_role_left_null_when_no_provider_has_api_key():
    extra = {
        "providers": [
            {"id": "u1", "name": "DeepSeek", "api_key": "", "api_base": None, "models": []},
        ]
    }
    out, changed = _migrate_legacy_extra(extra)
    assert changed is True
    assert out["roles"]["chat"] is None
    assert out["providers"][0]["provider"] == "deepseek"  # preset field still inferred
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_user_settings_migration.py -v`

Expected: 6 FAILED — `ImportError: cannot import name '_migrate_legacy_extra'`

- [ ] **Step 3: Add `_migrate_legacy_extra` at the top of `backend/nanoresearch/storage/repositories/user_settings_repo.py`**

Insert after the existing `_us_from_hash` function (around line 30), before `class UserSettingsRepository`:

```python
_PROVIDER_PRESET_KEYS = (
    "deepseek",
    "openai",
    "anthropic",
    "dashscope",
    "azure_openai",
    "siliconflow",
)
_EMBEDDING_CAPABLE = {"dashscope", "openai", "azure_openai", "siliconflow"}
_ROLE_NAMES = (
    "chat",
    "ingestion_llm",
    "embedding",
    "vision",
    "eval_generator",
    "eval_evaluator",
)


def _infer_provider_preset(name: str) -> str:
    """Map a free-text provider name to a canonical preset key."""
    lname = (name or "").lower()
    for key in _PROVIDER_PRESET_KEYS:
        if key in lname:
            return key
    return "openai_compatible"


def _migrate_legacy_extra(extra: dict) -> tuple[dict, bool]:
    """Add `provider` field to each provider and build a default `roles` map.

    Idempotent: if `roles` is already present (even partially), returns extra
    unchanged. Migration only fires when providers exist AND roles key absent.
    """
    if "roles" in extra:
        return extra, False
    providers = extra.get("providers")
    if not providers:
        return extra, False

    migrated_providers = []
    for p in providers:
        if p.get("provider"):
            migrated_providers.append(p)
            continue
        new_p = dict(p)
        new_p["provider"] = _infer_provider_preset(p.get("name", ""))
        migrated_providers.append(new_p)

    chat_provider = next((p for p in migrated_providers if p.get("api_key")), None)
    embedding_provider = next(
        (
            p for p in migrated_providers
            if p.get("api_key") and (p.get("provider") or "") in _EMBEDDING_CAPABLE
        ),
        None,
    )

    def _role_entry(provider: dict | None, model_hint: str | None) -> dict | None:
        if not provider:
            return None
        models = provider.get("models") or []
        model = model_hint or (models[0] if models else "")
        return {"provider_id": provider["id"], "model": model}

    embedding_model = None
    if embedding_provider:
        embedding_model = next(
            (m for m in (embedding_provider.get("models") or []) if "embed" in m.lower()),
            None,
        )

    roles = {
        "chat": _role_entry(chat_provider, None),
        "ingestion_llm": _role_entry(chat_provider, None),
        "embedding": _role_entry(embedding_provider, embedding_model),
        "vision": None,
        "eval_generator": None,
        "eval_evaluator": None,
    }

    new_extra = dict(extra)
    new_extra["providers"] = migrated_providers
    new_extra["roles"] = roles
    return new_extra, True
```

- [ ] **Step 4: Wire the helper into `UserSettingsRepository.get` at `backend/nanoresearch/storage/repositories/user_settings_repo.py:37-65`**

Replace the `get` method body. The new flow: after fetching from DB, run `_migrate_legacy_extra`; if changed, call `self.upsert(uid, extra=new_extra)` to persist + invalidate cache:

```python
    async def get(self, uid: str) -> UserSettings | None:
        from nanoresearch.bus.redis_client import get_redis
        from nanoresearch.bus.redis_keys import RedisKeys
        cache_key = RedisKeys.user_settings(uid)
        try:
            cached = await get_redis().hgetall(cache_key)
            if cached:
                logger.bind(event="user_settings_cache_hit", cache_layer="user_settings_cache").debug(
                    "user_settings cache hit for {}", uid
                )
                return _us_from_hash(cached)
        except Exception:
            pass

        logger.bind(event="user_settings_cache_miss", cache_layer="user_settings_cache").debug(
            "user_settings cache miss for {}", uid
        )
        async with self._factory() as db:
            result = await db.execute(select(UserSettings).where(UserSettings.uid == uid))
            row = result.scalar_one_or_none()

        if row is not None:
            new_extra, changed = _migrate_legacy_extra(row.extra or {})
            if changed:
                row = await self.upsert(uid, extra=new_extra)
            else:
                try:
                    r = get_redis()
                    await r.hset(cache_key, mapping=_us_to_hash(row))
                    await r.expire(cache_key, RedisKeys.USER_SETTINGS_TTL)
                except Exception:
                    pass
        return row
```

Note: when migration fires, `upsert` already invalidates the cache, so we don't write the stale shape back. When migration doesn't fire, we cache normally as before.

- [ ] **Step 5: Run migration tests to verify they pass**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_user_settings_migration.py -v`

Expected: 6 PASSED

- [ ] **Step 6: Confirm existing settings router tests still pass**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_settings_roles_schema.py backend/tests/test_phase5_concurrent_leak.py -v`

Expected: all PASSED (no regression).

- [ ] **Step 7: Commit**

```bash
git add backend/nanoresearch/storage/repositories/user_settings_repo.py backend/tests/test_user_settings_migration.py
git commit -m "feat(settings): migrate legacy extra to provider preset + roles map on read"
```

---

### Task 4: Backend — ModelFactory consults roles map

**Files:**
- Modify: `backend/nanoresearch/providers/model_factory.py:170-193` (dispatch + new top-level lookup), `:467-489` (helpers)
- Test: `backend/tests/test_role_assignment.py` (NEW)

**Interfaces:**
- Consumes: `user_providers: list[dict]` (each dict may carry `provider` preset field), AND new `user_roles: dict[str, dict | None]` kwarg threaded through `ModelFactory.resolve`.
- Produces:
  - `ModelFactory.resolve` accepts a new optional `user_roles: dict | None = None` kwarg.
  - When `user_roles[role.value]` is `{"provider_id": "...", "model": "..."}`, the resolver looks up that provider by id and returns a `ModelSpec` from it. Model defaults: explicit override > roles[role].model > provider.models[0].
  - When `user_roles[role.value]` is None or absent, falls through to existing per-role resolver logic (unchanged).
  - New static method `_match_user_provider_by_id(provider_id: str, user_providers: list[dict]) -> dict | None`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_role_assignment.py`:

```python
"""ModelFactory consults user_roles map when present."""

from __future__ import annotations

import pytest

from nanoresearch.providers.model_factory import (
    ModelFactory,
    ModelResolutionError,
    ModelRole,
)


PROVIDERS = [
    {"id": "u1", "name": "我的 DeepSeek", "provider": "deepseek",
     "api_key": "sk-1", "api_base": None, "models": ["deepseek-chat"]},
    {"id": "u2", "name": "通义", "provider": "dashscope",
     "api_key": "sk-2", "api_base": "https://x", "models": ["qwen-plus", "text-embedding-v3"]},
]


def test_chat_role_by_id_resolves_to_explicit_provider(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    roles = {"chat": {"provider_id": "u2", "model": "qwen-plus"}}
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=PROVIDERS,
        user_roles=roles,
    )
    assert spec.api_key == "sk-2"
    assert spec.model == "qwen-plus"
    assert spec.base_url == "https://x"


def test_role_provider_id_unknown_falls_back_to_existing_logic(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    roles = {"chat": {"provider_id": "ghost", "model": "deepseek-chat"}}
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=PROVIDERS,
        user_roles=roles,
    )
    # Falls back to _match_user_provider_by_model with model="deepseek-chat" → u1
    assert spec.api_key == "sk-1"


def test_role_none_falls_through_to_existing_resolver(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    roles = {"chat": None}
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=PROVIDERS,
        user_roles=roles,
        user_model="qwen-plus",  # Drives model-match fallback to u2
    )
    assert spec.api_key == "sk-2"


def test_embedding_role_by_id(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    roles = {"embedding": {"provider_id": "u2", "model": "text-embedding-v3"}}

    class _Emb:
        provider = "dashscope"
        model = "fallback-model"
        api_key = None
        base_url = None

    class _Settings:
        embedding = _Emb()
        llm = None
        vision_llm = None

    spec = ModelFactory.resolve(
        ModelRole.EMBEDDING,
        user_providers=PROVIDERS,
        user_roles=roles,
        rag_settings=_Settings(),
    )
    assert spec.api_key == "sk-2"
    assert spec.model == "text-embedding-v3"


def test_server_mode_with_roles_uses_assigned_provider(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    roles = {"chat": {"provider_id": "u1", "model": "deepseek-chat"}}
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=PROVIDERS,
        user_roles=roles,
    )
    assert spec.api_key == "sk-1"


def test_server_mode_no_roles_no_match_raises_missing_provider(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    with pytest.raises(ModelResolutionError) as exc:
        ModelFactory.resolve(
            ModelRole.CHAT,
            user_providers=[],
            user_roles={"chat": None},
        )
    assert exc.value.missing_role == "chat"


def test_match_user_provider_by_id_helper():
    assert ModelFactory._match_user_provider_by_id("u1", PROVIDERS)["api_key"] == "sk-1"
    assert ModelFactory._match_user_provider_by_id("ghost", PROVIDERS) is None
    assert ModelFactory._match_user_provider_by_id("", PROVIDERS) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_role_assignment.py -v`

Expected: 7 FAILED — `TypeError: resolve() got an unexpected keyword argument 'user_roles'` and `AttributeError: type object 'ModelFactory' has no attribute '_match_user_provider_by_id'`.

- [ ] **Step 3: Add `_match_user_provider_by_id` helper**

In `backend/nanoresearch/providers/model_factory.py`, add the helper near the existing `_match_user_provider_by_model` (around line 467, just before it):

```python
    @staticmethod
    def _match_user_provider_by_id(provider_id: str, user_providers: list[dict]) -> dict | None:
        """Find user provider by exact id match."""
        if not provider_id or not user_providers:
            return None
        return next((p for p in user_providers if p.get("id") == provider_id), None)
```

- [ ] **Step 4: Thread `user_roles` through `ModelFactory.resolve`**

Find the `resolve` classmethod (around line 100). Add `user_roles: dict | None = None` to its signature. At the top of the method body (before the `_providers = user_providers or []` line), add:

```python
        _roles = user_roles or {}
        role_entry = _roles.get(role.value)
        if role_entry and role_entry.get("provider_id"):
            matched = cls._match_user_provider_by_id(role_entry["provider_id"], user_providers or [])
            if matched and matched.get("api_key"):
                model = (
                    overrides.get("model_override")
                    or role_entry.get("model")
                    or (matched.get("models") or [None])[0]
                    or ""
                )
                return ModelSpec(
                    model=model,
                    api_key=matched.get("api_key"),
                    base_url=matched.get("api_base") or None,
                    provider=matched.get("provider") or matched.get("name") or None,
                )
```

Place this block BEFORE the server-mode `_resolve_from_user_only` call so an explicit role assignment wins over server-mode fallback. If `role_entry` is set but `provider_id` doesn't match or matched provider has no api_key, fall through unchanged — this preserves the "falls back to existing logic" behavior tested in Step 1.

Find and update `ModelFactory.require_key` similarly (it's the wrapper that calls `resolve`). Add `user_roles: dict | None = None` to its signature and forward to `resolve(...)`.

Also thread `user_roles` through any other public entry points (e.g., `ModelFactory.resolve` may be called from `eval_router` / `knowledge_router` / `worker`; those call sites get updated in Step 6 below — only signature additions here).

- [ ] **Step 5: Run role assignment tests to verify they pass**

Run: `./backend/.venv/Scripts/python -m pytest backend/tests/test_role_assignment.py -v`

Expected: 7 PASSED

- [ ] **Step 6: Update existing call sites to pass `user_roles`**

Five call sites consume `user_providers` and now need to also pass `user_roles`. For each, extract `roles = (user_cfg.extra or {}).get("roles") or None` alongside the existing `user_providers` extraction, then add `user_roles=roles` to the `ModelFactory.resolve` / `require_key` call:

1. `backend/nanoresearch/server/routers/knowledge_router.py:42-58` (`_resolve_rag_settings`)
2. `backend/nanoresearch/server/routers/eval_router.py:45-70` (`_resolve_eval_spec`)
3. `backend/nanoresearch/server/routers/eval_router.py:1015-1030` (the standalone block)
4. `backend/nanoresearch/worker.py:95-115` (the chat-role resolution block)
5. `backend/nanoresearch/worker.py:482-510` (the ingestion-role resolution block)

Concrete pattern for each site (example for `knowledge_router.py`):

```python
        user_providers = (user_cfg.extra or {}).get("providers", []) if user_cfg else []
        user_roles = (user_cfg.extra or {}).get("roles") if user_cfg else None
        ...
        spec = ModelFactory.resolve(
            role,
            ...
            user_providers=user_providers,
            user_roles=user_roles,
            ...
        )
```

Apply this addition at each of the 5 sites. Do not change the other arguments.

- [ ] **Step 7: Run all phase-5 + new tests to verify no regression**

Run:
```bash
./backend/.venv/Scripts/python -m pytest \
  backend/tests/test_role_assignment.py \
  backend/tests/test_settings_roles_schema.py \
  backend/tests/test_user_settings_migration.py \
  backend/tests/test_phase5_concurrent_leak.py \
  backend/tests/test_env_fallback_gating.py \
  backend/tests/test_model_factory_mode.py \
  -v
```

Expected: all PASSED.

- [ ] **Step 8: Commit**

```bash
git add backend/nanoresearch/providers/model_factory.py \
        backend/nanoresearch/server/routers/knowledge_router.py \
        backend/nanoresearch/server/routers/eval_router.py \
        backend/nanoresearch/worker.py \
        backend/tests/test_role_assignment.py
git commit -m "feat(providers): ModelFactory consults user_roles map for explicit role assignment"
```

---

### Task 5: Frontend — settings store roles state + coverage rewrite

**Files:**
- Modify: `web/src/stores/settings.js` (whole file)
- Test: manual (Vue 3 stores typically don't carry unit tests in this repo; verify via Task 7's UI integration). Confirm by checking `getMySettings()` and `updateMySettings()` API contracts in `web/src/apis/settings.js`.

**Interfaces:**
- Consumes: GET `/api/settings/me` response now carries `roles` (dict) and each provider carries `provider` (preset string).
- Produces:
  - Store gains reactive `roles` ref (`Ref<Record<string, {provider_id: string, model: string} | null>>`).
  - Store gains `saveRoles(rolesMap)` action — calls `updateMySettings({ roles: rolesMap })`.
  - `coverage` computed rewritten to read from `roles.value` (no more heuristic).
  - `fetchAll` populates `roles.value` from response.

- [ ] **Step 1: Check the API helper to confirm contract**

Read: `web/src/apis/settings.js`

Confirm `updateMySettings` accepts a free-shape body (it should — it's a thin wrapper over `PUT /api/settings/me`). If it whitelists fields, expand the whitelist to include `roles`.

- [ ] **Step 2: Rewrite `web/src/stores/settings.js`**

Replace the whole file with:

```javascript
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getMySettings, updateMySettings } from '@/apis/settings'

const ROLE_NAMES = [
  'chat',
  'ingestion_llm',
  'embedding',
  'vision',
  'eval_generator',
  'eval_evaluator',
]

function emptyRoles() {
  return Object.fromEntries(ROLE_NAMES.map(r => [r, null]))
}

export const useSettingsStore = defineStore('settings', () => {
  const providers = ref([])
  const roles = ref(emptyRoles())
  const baseModel = ref(null)
  const ragasGeneratorModel = ref(null)
  const ragasEvaluatorModel = ref(null)
  const ragasEmbeddingModel = ref(null)
  const loading = ref(false)

  // All model names from user-configured providers only
  const allModelOptions = computed(() => {
    const fromProviders = providers.value.flatMap(p => p.models || [])
    return [...new Set(fromProviders)].map(m => ({ value: m, label: m }))
  })

  // Provider coverage: which roles have an explicit provider assignment.
  const coverage = computed(() => ({
    hasChat: !!roles.value?.chat?.provider_id,
    hasEmbedding: !!roles.value?.embedding?.provider_id,
  }))

  async function fetchAll() {
    loading.value = true
    try {
      const s = await getMySettings()
      providers.value = s.providers || []
      const fetchedRoles = s.roles || {}
      roles.value = { ...emptyRoles(), ...fetchedRoles }
      baseModel.value = s.model || null
      ragasGeneratorModel.value = s.ragas_generator_model
      ragasEvaluatorModel.value = s.ragas_evaluator_model
      ragasEmbeddingModel.value = s.ragas_embedding_model
    } finally {
      loading.value = false
    }
  }

  async function saveProviders(providerList) {
    const s = await updateMySettings({ providers: providerList })
    providers.value = s.providers || []
    roles.value = { ...emptyRoles(), ...(s.roles || {}) }
  }

  async function saveRoles(rolesMap) {
    const s = await updateMySettings({ roles: rolesMap })
    roles.value = { ...emptyRoles(), ...(s.roles || {}) }
  }

  async function saveBaseModel(model) {
    const s = await updateMySettings({ model: model || '' })
    baseModel.value = s.model || null
  }

  async function saveRagasSettings(data) {
    const s = await updateMySettings({
      ragas_generator_model: data.generatorModel || '',
      ragas_evaluator_model: data.evaluatorModel || '',
      ragas_embedding_model: data.embeddingModel || '',
    })
    ragasGeneratorModel.value = s.ragas_generator_model
    ragasEvaluatorModel.value = s.ragas_evaluator_model
    ragasEmbeddingModel.value = s.ragas_embedding_model
  }

  return {
    providers, roles, allModelOptions, coverage, baseModel,
    ragasGeneratorModel, ragasEvaluatorModel, ragasEmbeddingModel,
    loading,
    fetchAll, saveProviders, saveRoles, saveBaseModel, saveRagasSettings,
  }
})
```

- [ ] **Step 3: Verify the build is still clean**

Run: `cd web && pnpm build`

Expected: build succeeds. (If `coverage` shape changed and downstream consumers break, fix consumers — search via `Grep` for `settingsStore.coverage` and `coverage.hasChat` / `coverage.hasEmbedding` across `web/src`. The shape is unchanged so this should be clean.)

- [ ] **Step 4: Commit**

```bash
git add web/src/stores/settings.js
git commit -m "feat(web): add roles state and rewrite coverage on settings store"
```

---

### Task 6: Frontend — provider modal preset dropdown

**Files:**
- Modify: `web/src/layouts/AppLayout.vue:162-203` (provider modal template), `:308-336` (provider modal state + form handlers)

**Interfaces:**
- Consumes: `settingsStore.providers` items now carry `provider` (preset string).
- Produces:
  - Modal template adds an `a-select` for `providerForm.provider` ABOVE the `name` input.
  - `PROVIDER_PRESETS` constant: `[{value: 'deepseek', label: 'DeepSeek'}, {value: 'openai', label: 'OpenAI'}, {value: 'anthropic', label: 'Anthropic'}, {value: 'dashscope', label: '通义千问 (DashScope)'}, {value: 'azure_openai', label: 'Azure OpenAI'}, {value: 'siliconflow', label: 'SiliconFlow'}, {value: 'openai_compatible', label: 'OpenAI 兼容 (自定义)'}]`.
  - When `providerForm.provider === 'openai_compatible'`, the `api_base` form item shows a "*" required marker (visual only).
  - `providerForm` includes `provider: string` field.
  - `openProviderModal(p)` pre-fills `provider: p?.provider || ''`.
  - `saveProvider` writes `provider: providerForm.provider || null` into the payload.

- [ ] **Step 1: Add `PROVIDER_PRESETS` constant near `PROVIDER_MODEL_PRESETS`**

In `web/src/layouts/AppLayout.vue` `<script setup>`, immediately above the existing `const PROVIDER_MODEL_PRESETS = {...}` at line 314, add:

```javascript
const PROVIDER_PRESETS = [
  { value: 'deepseek',          label: 'DeepSeek' },
  { value: 'openai',            label: 'OpenAI' },
  { value: 'anthropic',         label: 'Anthropic' },
  { value: 'dashscope',         label: '通义千问 (DashScope)' },
  { value: 'azure_openai',      label: 'Azure OpenAI' },
  { value: 'siliconflow',       label: 'SiliconFlow' },
  { value: 'openai_compatible', label: 'OpenAI 兼容 (自定义)' },
]
```

- [ ] **Step 2: Extend `providerForm` reactive ref at line 312**

Change:
```javascript
const providerForm      = ref({ name: '', api_key: '', api_base: '', models: [] })
```
to:
```javascript
const providerForm      = ref({ provider: '', name: '', api_key: '', api_base: '', models: [] })
```

- [ ] **Step 3: Update `openProviderModal` at lines 331-337**

Replace with:

```javascript
function openProviderModal(p) {
  editingProvider.value = p
  providerForm.value = p
    ? { provider: p.provider || '', name: p.name, api_key: '', api_base: p.api_base || '', models: [...(p.models || [])] }
    : { provider: '', name: '', api_key: '', api_base: '', models: [] }
  providerModalOpen.value = true
}
```

- [ ] **Step 4: Update `saveProvider` at lines 339-373 to include `provider` in payload**

Find the two `next = ...` branches and add `provider: providerForm.value.provider || null` to each new-row dict, AND ensure the existing-row mapping preserves `provider` from `p`. Replace the function with:

```javascript
async function saveProvider() {
  if (!providerForm.value.name.trim()) {
    message.warning('请填写供应商名称')
    return
  }
  if (!providerForm.value.provider) {
    message.warning('请选择供应商类型')
    return
  }
  providerSaving.value = true
  try {
    const existing = settingsStore.providers.map(p => ({
      id: p.id, name: p.name, provider: p.provider, api_key: null, api_base: p.api_base, models: p.models,
    }))

    let next
    if (editingProvider.value) {
      next = existing.map(p =>
        p.id === editingProvider.value.id
          ? {
              id: p.id,
              name: providerForm.value.name,
              provider: providerForm.value.provider || null,
              api_key: providerForm.value.api_key || null,
              api_base: providerForm.value.api_base || null,
              models: providerForm.value.models,
            }
          : p
      )
    } else {
      next = [...existing, {
        name: providerForm.value.name,
        provider: providerForm.value.provider || null,
        api_key: providerForm.value.api_key || null,
        api_base: providerForm.value.api_base || null,
        models: providerForm.value.models,
      }]
    }

    await settingsStore.saveProviders(next)
    providerModalOpen.value = false
    message.success('供应商已保存')
  } catch (e) {
    message.error('保存失败：' + (e.message || '未知错误'))
  } finally {
    providerSaving.value = false
  }
}
```

- [ ] **Step 5: Update `providerModelOptions` computed at lines 325-329 to drive off `provider` preset, not free-text `name`**

Replace with:

```javascript
const providerModelOptions = computed(() => {
  const preset = providerForm.value.provider
  const presets = PROVIDER_MODEL_PRESETS[preset] || []
  return presets.map(m => ({ label: m, value: m }))
})
```

(The existing `PROVIDER_MODEL_PRESETS` keys already cover `deepseek`, `openai`, `anthropic`, `dashscope`. Add an `azure_openai`, `siliconflow`, `openai_compatible` entry with `[]` for now so the lookup never errors:)

Modify `PROVIDER_MODEL_PRESETS` (line 314-323) to add the missing keys at the end:

```javascript
  azure_openai:      [],
  siliconflow:       ['deepseek-v3', 'qwen-plus', 'BAAI/bge-large-zh-v1.5'],
  openai_compatible: [],
```

- [ ] **Step 6: Update modal template at lines 162-203 — add provider preset select above the name input**

Replace the `<a-form ...>` block (lines 172-202) with:

```vue
    <a-form layout="vertical" style="margin-top: 16px">
      <a-form-item label="供应商类型" required>
        <a-select
          v-model:value="providerForm.provider"
          placeholder="选择供应商"
          :options="PROVIDER_PRESETS"
        />
      </a-form-item>
      <a-form-item label="自定义名称（备注）">
        <a-input v-model:value="providerForm.name" placeholder="如 我的 DeepSeek、团队 OpenAI" />
      </a-form-item>
      <a-form-item label="API Key">
        <a-input-password
          v-model:value="providerForm.api_key"
          :placeholder="editingProvider?.api_key_set
            ? editingProvider.api_key_hint + '（留空保持不变）'
            : '请输入 API Key'"
          autocomplete="new-password"
        />
      </a-form-item>
      <a-form-item :label="providerForm.provider === 'openai_compatible' ? 'API Base URL（必填）' : 'API Base URL'">
        <a-input
          v-model:value="providerForm.api_base"
          placeholder="如 https://dashscope.aliyuncs.com/compatible-mode/v1"
        />
      </a-form-item>
      <a-form-item label="可用模型">
        <a-select
          v-model:value="providerForm.models"
          mode="tags"
          placeholder="输入模型名后按 Enter，如 qwen-plus"
          style="width: 100%"
          :token-separators="[',']"
          :options="providerModelOptions"
        />
        <div class="field-hint">这些模型将出现在 Agent 的模型选择下拉框中</div>
      </a-form-item>
    </a-form>
```

`PROVIDER_PRESETS` is already in `<script setup>` scope from Step 1, so the template can reference it directly.

- [ ] **Step 7: Show the `provider` preset on each provider card at lines 82-92**

Modify the provider card to surface the preset as a tag. Replace lines 82-92:

```vue
            <div v-for="p in settingsStore.providers" :key="p.id" class="provider-card">
              <div class="provider-card-body">
                <div class="provider-name">
                  {{ p.name }}
                  <a-tag v-if="p.provider" size="small" style="margin-left: 6px">
                    {{ p.provider }}
                  </a-tag>
                </div>
                <div class="provider-meta">
                  <a-tag v-if="p.api_key_set" color="green" size="small">Key 已配置</a-tag>
                  <span v-if="p.api_base" class="provider-base">{{ p.api_base }}</span>
                </div>
                <div v-if="p.models.length" class="provider-models">
                  {{ p.models.join(' · ') }}
                </div>
              </div>
              <div class="provider-card-actions">
                <a-button type="text" size="small" @click="openProviderModal(p)">
                  <edit-outlined />
                </a-button>
                <a-popconfirm title="确认删除？" ok-text="删除" cancel-text="取消" @confirm="deleteProvider(p.id)">
                  <a-button type="text" size="small" danger>
                    <delete-outlined />
                  </a-button>
                </a-popconfirm>
              </div>
            </div>
```

- [ ] **Step 8: Verify the build is clean**

Run: `cd web && pnpm build`

Expected: build succeeds, no template errors.

- [ ] **Step 9: Commit**

```bash
git add web/src/layouts/AppLayout.vue
git commit -m "feat(web): provider modal uses preset dropdown instead of freeform name"
```

---

### Task 7: Frontend — role assignment section + defaults + cleanup

**Files:**
- Modify: `web/src/layouts/AppLayout.vue:65-126` (replace alert + bonus role section), `:281-284` (delete `providerGuideDesc`), `:339-373` (`saveProvider` — add default auto-assignment after save)

**Interfaces:**
- Consumes: `settingsStore.roles` (Record<string, {provider_id, model} | null>), `settingsStore.providers`, `settingsStore.saveRoles(rolesMap)`.
- Produces:
  - Removed: `<a-alert message="至少需要填两组 key" .../>` (lines 69-75) and `providerGuideDesc` const (lines 281-284).
  - Added: new role assignment section below the provider list with 6 rows, one per `ROLE_LABELS` entry.
  - After `saveProviders` finishes adding the FIRST provider, auto-assign that provider to `chat` and `ingestion_llm` roles. After adding the second provider whose preset ∈ `{dashscope, openai, azure_openai, siliconflow}`, auto-assign to `embedding`.

- [ ] **Step 1: Add `ROLE_LABELS` constant in `<script setup>`**

Insert below `PROVIDER_PRESETS` (added in Task 6 Step 1):

```javascript
const ROLE_LABELS = [
  { key: 'chat',            label: '聊天 (chat)',           hint: '默认对话模型' },
  { key: 'ingestion_llm',   label: 'RAG 摄取',              hint: '处理知识库文档时使用' },
  { key: 'embedding',       label: '向量嵌入',              hint: '知识库检索需要' },
  { key: 'vision',          label: '视觉',                  hint: '图片理解；留空则关闭' },
  { key: 'eval_generator',  label: '评测 - 题目生成',       hint: '留空 fallback 到聊天模型' },
  { key: 'eval_evaluator',  label: '评测 - 打分',           hint: '留空 fallback 到聊天模型' },
]
const EMBEDDING_CAPABLE_PRESETS = new Set(['dashscope', 'openai', 'azure_openai', 'siliconflow'])
```

- [ ] **Step 2: Delete the misleading alert at `web/src/layouts/AppLayout.vue:69-75`**

Remove:
```vue
          <a-alert
            type="info"
            show-icon
            style="margin-bottom: 12px; font-size: 12px"
            message="至少需要填两组 key"
            :description="providerGuideDesc"
          />
```

Replace with:
```vue
          <a-alert
            type="info"
            show-icon
            style="margin-bottom: 12px; font-size: 12px"
            message="第一步：添加 API key"
            description="不同模型用途可以共用同一个 key，也可以分配不同 provider。下方「模型用途分配」决定每种调用走哪个 key。"
          />
```

- [ ] **Step 3: Delete `providerGuideDesc` const at lines 281-284**

Remove:
```javascript
const providerGuideDesc = `• 一组用于 Chat（如 deepseek / openai）
• 一组用于 Embedding（如 dashscope / openai；deepseek 不提供 embedding）

每个 provider 行的"models"字段标注此 key 能跑的模型；多 provider 共存时，按 model 精确匹配优先，匹配不到走第一个有 key 的 provider 兜底。`
```

(No replacement — unused once the alert is rewritten.)

- [ ] **Step 4: Add the role assignment section to the template**

In `AppLayout.vue`, insert a new section AFTER the provider list `</div>` closing tag (currently around line 105, end of the `<a-spin>` wrapper) and BEFORE the "Base 模型" section header (line 108). The full insert:

```vue
        <!-- 模型用途分配 -->
        <div class="section-header" style="margin-top: 24px">
          <span class="section-title">模型用途分配</span>
        </div>
        <div class="field-hint" style="margin-bottom: 10px">
          每种调用使用哪个 provider + 哪个模型。留空时按 fallback 规则处理。
        </div>
        <div v-if="settingsStore.providers.length === 0" class="empty-providers">
          先添加 API key，再分配用途
        </div>
        <div v-else class="role-assignment-list">
          <div v-for="role in ROLE_LABELS" :key="role.key" class="role-row">
            <div class="role-label">
              <div class="role-title">{{ role.label }}</div>
              <div class="role-hint">{{ role.hint }}</div>
            </div>
            <a-select
              :value="settingsStore.roles[role.key]?.provider_id || null"
              :options="providerSelectOptions"
              placeholder="未配置"
              allow-clear
              style="width: 180px"
              @change="(pid) => onRoleProviderChange(role.key, pid)"
            />
            <a-auto-complete
              :value="settingsStore.roles[role.key]?.model || ''"
              :options="modelOptionsForRole(role.key)"
              placeholder="模型名"
              allow-clear
              style="width: 200px"
              @change="(m) => onRoleModelChange(role.key, m)"
            />
          </div>
        </div>
```

- [ ] **Step 5: Add the supporting computed + handlers in `<script setup>`**

Insert near the other provider-form handlers (around line 386, just before `const activeKey = computed(...)`):

```javascript
const providerSelectOptions = computed(() =>
  settingsStore.providers.map(p => ({
    value: p.id,
    label: `${p.name}${p.provider ? ` (${p.provider})` : ''}`,
  }))
)

function modelOptionsForRole(roleKey) {
  const pid = settingsStore.roles[roleKey]?.provider_id
  if (!pid) return []
  const p = settingsStore.providers.find(x => x.id === pid)
  return (p?.models || []).map(m => ({ value: m, label: m }))
}

async function onRoleProviderChange(roleKey, providerId) {
  const next = { ...settingsStore.roles }
  if (!providerId) {
    next[roleKey] = null
  } else {
    const p = settingsStore.providers.find(x => x.id === providerId)
    const defaultModel = (p?.models || [])[0] || ''
    next[roleKey] = { provider_id: providerId, model: defaultModel }
  }
  try {
    await settingsStore.saveRoles(next)
  } catch (e) {
    message.error('保存失败：' + (e.message || ''))
  }
}

async function onRoleModelChange(roleKey, model) {
  const entry = settingsStore.roles[roleKey]
  if (!entry) return  // No provider chosen yet; the model field is disabled-ish
  const next = { ...settingsStore.roles, [roleKey]: { provider_id: entry.provider_id, model: model || '' } }
  try {
    await settingsStore.saveRoles(next)
  } catch (e) {
    message.error('保存失败：' + (e.message || ''))
  }
}
```

- [ ] **Step 6: Add default auto-assignment after `saveProvider` adds a new provider**

Modify `saveProvider` at lines 339-373 (already touched in Task 6). After the `await settingsStore.saveProviders(next)` line, before `providerModalOpen.value = false`, add:

```javascript
    // Default role auto-assignment on first add or first embedding-capable add
    if (!editingProvider.value) {
      const updatedProviders = settingsStore.providers
      const added = updatedProviders.find(p =>
        p.name === providerForm.value.name && p.provider === (providerForm.value.provider || null)
      )
      if (added) {
        const nextRoles = { ...settingsStore.roles }
        let rolesChanged = false
        if (!nextRoles.chat) {
          nextRoles.chat = { provider_id: added.id, model: (added.models || [])[0] || '' }
          rolesChanged = true
        }
        if (!nextRoles.ingestion_llm) {
          nextRoles.ingestion_llm = { provider_id: added.id, model: (added.models || [])[0] || '' }
          rolesChanged = true
        }
        if (!nextRoles.embedding && EMBEDDING_CAPABLE_PRESETS.has(added.provider || '')) {
          const embModel = (added.models || []).find(m => /embed/i.test(m)) || ''
          nextRoles.embedding = { provider_id: added.id, model: embModel }
          rolesChanged = true
        }
        if (rolesChanged) {
          await settingsStore.saveRoles(nextRoles)
        }
      }
    }
```

- [ ] **Step 7: Add style for the role assignment rows**

In the `<style scoped>` block at bottom of `AppLayout.vue`, add:

```css
.role-assignment-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.role-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  background: #fafafa;
  border-radius: 6px;
}
.role-label {
  flex: 1;
  min-width: 0;
}
.role-title {
  font-size: 13px;
  font-weight: 500;
}
.role-hint {
  font-size: 11px;
  color: #888;
}
```

- [ ] **Step 8: Verify the build is clean**

Run: `cd web && pnpm build`

Expected: build succeeds.

- [ ] **Step 9: Manual UI smoke (open dev server, open settings, add a provider)**

Run: `cd web && pnpm dev`

In browser:
1. Open settings → 供应商
2. 点「添加」 → 选择 "DeepSeek" → 填名称 "我的 DeepSeek" → 填 fake api_key → 保存
3. Verify the role section now shows `chat` and `ingestion_llm` both auto-assigned to "我的 DeepSeek (deepseek)".
4. 点「添加」 → 选择 "DashScope" → 填名称 "通义" → 填 fake key → 保存
5. Verify `embedding` is now auto-assigned to 通义.
6. Manually change `chat` provider from the dropdown → verify it persists across refresh.

If `pnpm dev` fails to start due to the pre-existing `TunableObjectVersion` ImportError affecting the API proxy, document the manual checklist and note "blocked by ImportError — fixed in Task 1; verify after server restart". Task 1 should have already unblocked this; if not, re-run Task 1.

- [ ] **Step 10: Commit**

```bash
git add web/src/layouts/AppLayout.vue
git commit -m "feat(web): explicit role assignment section + defaults + remove misleading alert"
```

---

## Self-review

After writing all tasks, verify:

**Spec coverage check** (each spec requirement → task that implements it):
- `TunableObjectVersion` ORM class → Task 1 ✓
- Provider preset `provider` field on `ProviderIn` and stored shape → Task 2 ✓
- `roles` map on `UserSettingsUpdate` and stored shape → Task 2 ✓
- API surface (GET emits `provider` + `roles`; PUT accepts both) → Task 2 ✓
- Legacy migration (`_migrate_legacy_extra`, idempotent, runs in `get()`) → Task 3 ✓
- ModelFactory consults `user_roles` first, falls back unchanged → Task 4 ✓
- 5 backend call sites threading `user_roles` → Task 4 ✓
- Frontend store `roles` state + `saveRoles` action + coverage rewrite → Task 5 ✓
- Provider modal preset dropdown → Task 6 ✓
- Provider card surfaces preset → Task 6 ✓
- Role assignment section UI → Task 7 ✓
- Default auto-assignment on first/embedding-capable add → Task 7 ✓
- Misleading "至少 2 组 key" alert removed → Task 7 ✓

**Placeholder scan:** no "TBD", "TODO", or "implement later" present. Every step has either a code block or an exact command.

**Type consistency:**
- `provider` (string preset key) — used identically in pydantic, repo migration, ModelFactory, frontend.
- `roles` keys (`chat`, `ingestion_llm`, `embedding`, `vision`, `eval_generator`, `eval_evaluator`) — match `ModelRole.value` exactly throughout; `ROLE_NAMES` JS array and `_ROLE_NAMES` Python tuple both list the same 6 values.
- `provider_id` (id of an entry in `providers`) — consistent across stored shape, ModelFactory `_match_user_provider_by_id`, frontend handlers.
- `model` (string, may be empty) — consistent.

**Independence check:** Tasks 1, 2, 3, 4 are backend; Tasks 5, 6, 7 are frontend. Tasks 2/3 are independent of each other (different files, different test surfaces); both must complete before Task 4 (uses migrated shape). Task 5 depends on Task 2's API contract. Tasks 6 and 7 both modify `AppLayout.vue` but at different ranges; Task 7 references `PROVIDER_PRESETS` added in Task 6, so Task 7 depends on Task 6. Final order: 1 → 2 → 3 → 4 → 5 → 6 → 7.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-provider-role-explicit.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
