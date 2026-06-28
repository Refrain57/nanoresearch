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
