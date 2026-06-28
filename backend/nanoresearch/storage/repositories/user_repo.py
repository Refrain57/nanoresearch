"""User repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanoresearch.storage.models import User


class UserRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def get_by_uid(self, uid: str) -> User | None:
        async with self._factory() as db:
            result = await db.execute(select(User).where(User.uid == uid))
            return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        async with self._factory() as db:
            result = await db.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()

    async def create(self, uid: str, password_hash: str, email: str | None = None, role: str = "admin") -> User:
        user = User(uid=uid, password_hash=password_hash, email=email, role=role)
        async with self._factory() as db:
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user

    async def exists(self, uid: str) -> bool:
        async with self._factory() as db:
            result = await db.execute(select(User.uid).where(User.uid == uid))
            return result.scalar_one_or_none() is not None
