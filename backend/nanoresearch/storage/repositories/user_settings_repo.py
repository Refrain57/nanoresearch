"""User settings repository."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from loguru import logger

from nanoresearch.storage.models import UserSettings


def _us_to_hash(us: UserSettings) -> dict:
    return {
        "uid": us.uid,
        "model": us.model or "",
        "max_iterations": str(us.max_iterations) if us.max_iterations is not None else "",
        "extra": json.dumps(us.extra or {}, ensure_ascii=False),
    }


def _us_from_hash(h: dict) -> UserSettings:
    us = UserSettings()
    us.uid = h["uid"]
    us.model = h.get("model") or None
    us.max_iterations = int(h["max_iterations"]) if h.get("max_iterations") else None
    us.extra = json.loads(h.get("extra") or "{}")
    return us


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


class UserSettingsRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

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

    async def upsert(self, uid: str, **fields) -> UserSettings:
        async with self._factory() as db:
            result = await db.execute(select(UserSettings).where(UserSettings.uid == uid))
            row = result.scalar_one_or_none()
            if row is None:
                row = UserSettings(uid=uid, extra={})
                db.add(row)
            for key, value in fields.items():
                setattr(row, key, value)
            await db.commit()
            await db.refresh(row)
        try:
            from nanoresearch.bus.redis_client import get_redis
            from nanoresearch.bus.redis_keys import RedisKeys
            await get_redis().delete(RedisKeys.user_settings(uid))
        except Exception:
            pass
        return row
