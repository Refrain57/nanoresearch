"""Integration tests for storage repositories.

DB setup/teardown: psycopg2 (sync).
Repository calls: asyncio.run() with a fresh event loop each time.
"""

from __future__ import annotations

import asyncio
import pytest

from nanoresearch.auth.password import hash_password
from nanoresearch.storage.repositories.agent_repo import AgentRepository
from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
from nanoresearch.storage.repositories.user_repo import UserRepository

from tests.conftest import truncate_all, make_factory


def run(coro):
    """Run an async coroutine with SelectorEventLoop (required by asyncpg on Windows)."""
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.fixture(autouse=True)
def clean_tables():
    truncate_all()


# ── User ──────────────────────────────────────────────────────────────────────

def test_user_create_and_get():
    async def _():
        repo = UserRepository(make_factory())
        user = await repo.create("alice", hash_password("pw"), role="admin")
        assert user.uid == "alice"
        assert user.role == "admin"
        found = await repo.get_by_uid("alice")
        assert found is not None and found.uid == "alice"
    run(_())


def test_user_not_found():
    async def _():
        repo = UserRepository(make_factory())
        assert await repo.get_by_uid("nobody") is None
    run(_())


def test_user_exists():
    async def _():
        repo = UserRepository(make_factory())
        assert not await repo.exists("alice")
        await repo.create("alice", hash_password("pw"))
        assert await repo.exists("alice")
    run(_())


def test_user_get_by_email():
    async def _():
        repo = UserRepository(make_factory())
        await repo.create("bob", hash_password("pw"), email="bob@example.com")
        found = await repo.get_by_email("bob@example.com")
        assert found is not None and found.uid == "bob"
        assert await repo.get_by_email("nobody@example.com") is None
    run(_())


# ── Agent ─────────────────────────────────────────────────────────────────────

def test_agent_create_and_get_default():
    async def _():
        repo = AgentRepository(make_factory())
        assert not await repo.default_exists()
        agent = await repo.create({
            "name": "Test Agent",
            "is_default": True,
            "default_model": "anthropic/claude-opus-4-5",
            "provider": "anthropic",
        })
        assert agent.name == "Test Agent"
        assert await repo.default_exists()
        default = await repo.get_default()
        assert default is not None and default.id == agent.id
    run(_())


def test_agent_list_all():
    async def _():
        repo = AgentRepository(make_factory())
        await repo.create({"name": "Agent A", "is_default": False})
        await repo.create({"name": "Agent B", "is_default": False})
        agents = await repo.list_all()
        assert len(agents) == 2
    run(_())


# ── Conversation ──────────────────────────────────────────────────────────────

def test_conversation_create_and_get_by_key():
    async def _():
        user_repo = UserRepository(make_factory())
        await user_repo.create("bob", hash_password("pw"))
        conv_repo = ConversationRepository(make_factory())
        conv = await conv_repo.create(
            key="telegram:12345",
            uid="bob",
            metadata={"chat_name": "My Chat"},
        )
        assert conv.session_key == "telegram:12345"
        assert conv.channel == "telegram"
        assert conv.channel_chat_id == "12345"
        assert conv.conv_metadata == {"chat_name": "My Chat"}
        found = await conv_repo.get_by_session_key("telegram:12345")
        assert found is not None and found.id == conv.id
    run(_())


def test_conversation_not_found():
    async def _():
        repo = ConversationRepository(make_factory())
        assert await repo.get_by_session_key("nonexistent:key") is None
    run(_())


def test_conversation_list_all():
    async def _():
        user_repo = UserRepository(make_factory())
        await user_repo.create("dave", hash_password("pw"))
        conv_repo = ConversationRepository(make_factory())
        await conv_repo.create(key="web:conv1", uid="dave")
        await conv_repo.create(key="web:conv2", uid="dave")
        all_convs = await conv_repo.list_all("dave")
        assert len(all_convs) == 2
        keys = {c["key"] for c in all_convs}
        assert keys == {"web:conv1", "web:conv2"}
    run(_())


def test_replace_messages_and_get():
    async def _():
        user_repo = UserRepository(make_factory())
        await user_repo.create("carol", hash_password("pw"))
        conv_repo = ConversationRepository(make_factory())
        conv = await conv_repo.create(key="web:msgs", uid="carol")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        await conv_repo.replace_messages(conv.id, messages)
        saved = await conv_repo.get_messages(conv.id)
        assert len(saved) == 3
        assert saved[0].seq == 0 and saved[0].role == "user"
        assert saved[2].seq == 2
    run(_())


def test_replace_messages_is_idempotent():
    async def _():
        user_repo = UserRepository(make_factory())
        await user_repo.create("eve", hash_password("pw"))
        conv_repo = ConversationRepository(make_factory())
        conv = await conv_repo.create(key="web:idem", uid="eve")
        msgs = [{"role": "user", "content": "msg1"}, {"role": "assistant", "content": "resp1"}]
        await conv_repo.replace_messages(conv.id, msgs)
        await conv_repo.replace_messages(conv.id, msgs[:1])
        saved = await conv_repo.get_messages(conv.id)
        assert len(saved) == 1
    run(_())


def test_update_meta():
    from datetime import datetime, timezone

    async def _():
        user_repo = UserRepository(make_factory())
        await user_repo.create("frank", hash_password("pw"))
        conv_repo = ConversationRepository(make_factory())
        conv = await conv_repo.create(key="web:meta", uid="frank")
        assert conv.last_consolidated == 0
        await conv_repo.update_meta(
            conv.id,
            last_consolidated=5,
            metadata={"foo": "bar"},
            updated_at=datetime.now(timezone.utc),
        )
        updated = await conv_repo.get_by_session_key("web:meta")
        assert updated.last_consolidated == 5
        assert updated.conv_metadata == {"foo": "bar"}
    run(_())


def test_get_messages_paged_recent_first_and_cursor():
    async def _():
        user_repo = UserRepository(make_factory())
        await user_repo.create("grace", hash_password("pw"))
        conv_repo = ConversationRepository(make_factory())
        conv = await conv_repo.create(key="web:paged", uid="grace")
        await conv_repo.replace_messages(
            conv.id, [{"role": "user", "content": f"m{i}"} for i in range(5)]
        )
        # 最近 2 条，升序
        recent = await conv_repo.get_messages_paged(conv.id, limit=2)
        assert [m.seq for m in recent] == [3, 4]
        # seq < 3 的最近 2 条
        older = await conv_repo.get_messages_paged(conv.id, limit=2, before_seq=3)
        assert [m.seq for m in older] == [1, 2]
        # limit 超过总数 → 全部升序
        allm = await conv_repo.get_messages_paged(conv.id, limit=10)
        assert [m.seq for m in allm] == [0, 1, 2, 3, 4]
    run(_())
