"""Conversation and message repository — core of session persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanobot.storage.models import Conversation, Message


class ConversationRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def get_by_session_key(self, key: str) -> Conversation | None:
        async with self._factory() as db:
            result = await db.execute(select(Conversation).where(Conversation.session_key == key))
            return result.scalar_one_or_none()

    async def get_by_id(self, conv_id: uuid.UUID) -> Conversation | None:
        async with self._factory() as db:
            result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
            return result.scalar_one_or_none()

    async def delete(self, conv_id: uuid.UUID) -> None:
        async with self._factory() as db:
            result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
            conv = result.scalar_one_or_none()
            if conv:
                await db.delete(conv)
                await db.commit()

    async def list_conversations(self, uid: str, limit: int = 20, offset: int = 0) -> list[Conversation]:
        async with self._factory() as db:
            result = await db.execute(
                select(Conversation)
                .where(Conversation.uid == uid)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def get_last_message(self, conv_id: uuid.UUID) -> Message | None:
        async with self._factory() as db:
            result = await db.execute(
                select(Message)
                .where(Message.conversation_id == conv_id)
                .order_by(Message.seq.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_messages_paged(self, conv_id: uuid.UUID, limit: int = 50, offset: int = 0) -> list[Message]:
        async with self._factory() as db:
            result = await db.execute(
                select(Message)
                .where(Message.conversation_id == conv_id)
                .order_by(Message.seq)
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def create(
        self,
        key: str,
        uid: str,
        agent_id: uuid.UUID | None = None,
        metadata: dict | None = None,
        created_at: datetime | None = None,
        title: str | None = None,
    ) -> Conversation:
        channel, _, chat_id = key.partition(":")
        conv = Conversation(
            uid=uid,
            agent_id=agent_id,
            session_key=key,
            channel=channel or "web",
            channel_chat_id=chat_id or key,
            title=title,
            conv_metadata=metadata or {},
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        async with self._factory() as db:
            db.add(conv)
            await db.commit()
            await db.refresh(conv)
        return conv

    async def update_meta(self, conv_id: uuid.UUID, last_consolidated: int, metadata: dict, updated_at: datetime) -> None:
        async with self._factory() as db:
            result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
            conv = result.scalar_one_or_none()
            if conv:
                conv.last_consolidated = last_consolidated
                conv.conv_metadata = metadata
                conv.updated_at = updated_at
                await db.commit()

    async def update_agent_override(self, conv_id: uuid.UUID, override: dict) -> Conversation | None:
        from sqlalchemy.orm.attributes import flag_modified
        async with self._factory() as db:
            result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
            conv = result.scalar_one_or_none()
            if conv is None:
                return None
            meta = dict(conv.conv_metadata or {})
            meta["agent_override"] = override
            conv.conv_metadata = meta
            flag_modified(conv, "conv_metadata")
            conv.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(conv)
        return conv

    async def get_messages(self, conv_id: uuid.UUID) -> list[Message]:
        async with self._factory() as db:
            result = await db.execute(
                select(Message).where(Message.conversation_id == conv_id).order_by(Message.seq)
            )
            return list(result.scalars().all())

    async def replace_messages(self, conv_id: uuid.UUID, messages: list[dict]) -> None:
        """Atomically replace all messages for a conversation."""
        async with self._factory() as db:
            await db.execute(delete(Message).where(Message.conversation_id == conv_id))
            for seq, msg in enumerate(messages):
                db.add(Message(
                    conversation_id=conv_id,
                    role=msg.get("role", "user"),
                    content=msg,
                    seq=seq,
                ))
            await db.commit()

    async def list_all(self, uid: str) -> list[dict]:
        async with self._factory() as db:
            result = await db.execute(
                select(Conversation)
                .where(Conversation.uid == uid)
                .order_by(Conversation.updated_at.desc())
            )
            convs = result.scalars().all()
            return [
                {
                    "key": c.session_key,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in convs
            ]
