"""User settings repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanobot.storage.models import UserSettings


class UserSettingsRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def get(self, uid: str) -> UserSettings | None:
        async with self._factory() as db:
            result = await db.execute(select(UserSettings).where(UserSettings.uid == uid))
            return result.scalar_one_or_none()

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
        return row
