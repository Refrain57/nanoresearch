"""Session management — DB-backed implementation with JSONL fallback."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanoresearch.config.paths import get_legacy_sessions_dir
from nanoresearch.utils.helpers import as_aware_utc, ensure_dir, safe_filename, utcnow_aware


@dataclass
class Session:
    """
    A conversation session.

    Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow_aware)
    updated_at: datetime = field(default_factory=utcnow_aware)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": utcnow_aware().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = utcnow_aware()

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start : i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        unconsolidated = self.messages[self.last_consolidated :]
        sliced = unconsolidated[-max_messages:]

        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = utcnow_aware()

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        start_idx = max(0, len(self.messages) - max_messages)
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = self.messages[start_idx:]
        start = self._find_legal_start(retained)
        if start:
            retained = retained[start:]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped)
        self.updated_at = utcnow_aware()


class SessionManager:
    """
    Async session manager backed by PostgreSQL.
    Falls back to JSONL when no session_factory is provided (e.g. tests).
    Redis is used as a write-through cache layer (2 h TTL) when available.
    """

    def __init__(
        self,
        workspace: Path,
        session_factory=None,
        default_uid: str = "admin",
    ) -> None:
        self.workspace = workspace
        self._factory = session_factory
        self._default_uid = default_uid
        self._cache: dict[str, Session] = {}
        self._sessions_dir = ensure_dir(workspace / "sessions")
        self._legacy_dir = get_legacy_sessions_dir()

    # ------------------------------------------------------------------
    # Redis key helpers
    # ------------------------------------------------------------------

    def _redis_keys(self, key: str) -> tuple[str, str]:
        """Return (msg_key, meta_key) for the given session key."""
        from nanoresearch.bus.redis_keys import RedisKeys
        ch, chat_id = key.split(":", 1)
        uid = self._default_uid
        return RedisKeys.session_msg(uid, ch, chat_id), RedisKeys.session_meta(uid, ch, chat_id)

    async def _redis_load(self, key: str) -> Session | None:
        """Try to load session from Redis. Returns None on miss or error."""
        try:
            from nanoresearch.bus.redis_client import get_redis
            from nanoresearch.bus.redis_keys import RedisKeys
            redis = get_redis()
            msg_key, meta_key = self._redis_keys(key)
            meta, raw_msgs = await redis.hgetall(meta_key), await redis.lrange(msg_key, 0, -1)
            if not meta or not raw_msgs:
                return None
            messages = [json.loads(m) for m in raw_msgs]
            created_at = as_aware_utc(datetime.fromisoformat(meta["created_at"])) if meta.get("created_at") else utcnow_aware()
            updated_at = as_aware_utc(datetime.fromisoformat(meta["updated_at"])) if meta.get("updated_at") else utcnow_aware()
            return Session(
                key=key,
                messages=messages,
                created_at=created_at,
                updated_at=updated_at,
                metadata=json.loads(meta.get("metadata") or "{}"),
                last_consolidated=int(meta.get("last_consolidated", 0)),
            )
        except Exception as e:
            logger.debug("Redis session load miss for {}: {}", key, e)
            return None

    async def _redis_save(self, session: Session) -> None:
        """Mirror the full Session into Redis via MULTI/EXEC. Fire-and-forget on error.

        We store the entire `messages` list — not the post-consolidation tail —
        because `last_consolidated` is an offset INTO the full list. Storing
        only the tail while preserving the original offset would make
        `get_history()` apply the offset a second time on load, silently
        dropping all history (production bug observed 2026-06-28). It would
        also cascade to `_db_save` via `replace_messages`, permanently
        truncating DB state.
        """
        try:
            from nanoresearch.bus.redis_client import get_redis
            from nanoresearch.bus.redis_keys import RedisKeys
            redis = get_redis()
            msg_key, meta_key = self._redis_keys(session.key)
            ts = utcnow_aware().isoformat()
            async with redis.pipeline(transaction=True) as pipe:
                pipe.delete(msg_key)
                if session.messages:
                    pipe.rpush(msg_key, *[json.dumps(m, ensure_ascii=False) for m in session.messages])
                pipe.hset(meta_key, mapping={
                    "updated_at": ts,
                    "created_at": session.created_at.isoformat(),
                    "metadata": json.dumps(session.metadata, ensure_ascii=False),
                    "last_consolidated": str(session.last_consolidated),
                })
                pipe.expire(msg_key, RedisKeys.SESSION_TTL)
                pipe.expire(meta_key, RedisKeys.SESSION_TTL)
                await pipe.execute()
        except Exception as e:
            logger.warning("Redis session save failed (non-fatal): {}", e)

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            logger.bind(event="session_l1_hit", cache_layer="session_cache").debug(
                "session L1 cache hit for {}", key
            )
            return self._cache[key]
        session = await self._redis_load(key)
        if session is not None:
            logger.bind(event="session_redis_hit", cache_layer="session_cache").debug(
                "session Redis cache hit for {}", key
            )
        else:
            logger.bind(event="session_cache_miss", cache_layer="session_cache").debug(
                "session cache miss for {}", key
            )
            session = await self._load(key)
            if session is None:
                session = Session(key=key)
            await self._redis_save(session)
        self._cache[key] = session
        return session

    async def save(self, session: Session) -> None:
        self._cache[session.key] = session
        await self._redis_save(session)
        if self._factory is not None:
            await self._db_save(session)
        else:
            self._file_save(session)

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    async def list_sessions(self) -> list[dict[str, Any]]:
        if self._factory is not None:
            return await self._db_list()
        return self._file_list()

    # ------------------------------------------------------------------
    # DB implementation
    # ------------------------------------------------------------------

    async def _load(self, key: str) -> Session | None:
        if self._factory is not None:
            return await self._db_load(key)
        return self._file_load(key)

    async def _db_load(self, key: str) -> Session | None:
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        repo = ConversationRepository(self._factory)
        conv = await repo.get_by_session_key(key)
        if conv is None:
            return None
        msgs = await repo.get_messages(conv.id)
        return Session(
            key=key,
            messages=[m.content for m in msgs],
            created_at=as_aware_utc(conv.created_at) if conv.created_at else utcnow_aware(),
            updated_at=as_aware_utc(conv.updated_at) if conv.updated_at else utcnow_aware(),
            metadata=conv.conv_metadata or {},
            last_consolidated=conv.last_consolidated or 0,
        )

    async def _db_save(self, session: Session) -> None:
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        repo = ConversationRepository(self._factory)
        conv = await repo.get_by_session_key(session.key)
        if conv is None:
            conv = await repo.create(
                key=session.key,
                uid=self._default_uid,
                metadata=session.metadata,
                created_at=session.created_at,
            )
        await repo.replace_messages(conv.id, session.messages)
        await repo.update_meta(conv.id, session.last_consolidated, session.metadata, session.updated_at)

    async def _db_list(self) -> list[dict[str, Any]]:
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        repo = ConversationRepository(self._factory)
        return await repo.list_all(self._default_uid)

    # ------------------------------------------------------------------
    # JSONL fallback (original logic, unchanged)
    # ------------------------------------------------------------------

    def _file_load(self, key: str) -> Session | None:
        path = self._get_file_path(key)
        if not path.exists():
            legacy = self._legacy_dir / f"{safe_filename(key.replace(':', '_'))}.jsonl"
            if legacy.exists():
                try:
                    shutil.move(str(legacy), str(path))
                except Exception:
                    logger.exception("Failed to migrate session {}", key)
        if not path.exists():
            return None
        try:
            messages, metadata, created_at, last_consolidated = [], {}, None, 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def _file_save(self, session: Session) -> None:
        path = self._get_file_path(session.key)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
            }, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def _file_list(self) -> list[dict[str, Any]]:
        results = []
        for path in self._sessions_dir.glob("*.jsonl"):
            try:
                with open(path, encoding="utf-8") as f:
                    first = f.readline().strip()
                if not first:
                    continue
                data = json.loads(first)
                if data.get("_type") == "metadata":
                    key = data.get("key") or path.stem.replace("_", ":", 1)
                    results.append({
                        "key": key,
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    })
            except Exception:
                continue
        return sorted(results, key=lambda x: x.get("updated_at") or "", reverse=True)

    def _get_file_path(self, key: str) -> Path:
        return self._sessions_dir / f"{safe_filename(key.replace(':', '_'))}.jsonl"
