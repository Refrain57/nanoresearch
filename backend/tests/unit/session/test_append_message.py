"""Phase 1: SessionManager.append_message — atomic Redis RPUSH + DB insert."""
import pytest

from nanoresearch.bus.redis_keys import RedisKeys
from nanoresearch.session.manager import SessionManager
from tests.conftest import make_factory, truncate_all


@pytest.fixture(autouse=True)
def _clean():
    truncate_all()


async def test_append_message_rpush_and_db(redis_client, monkeypatch, tmp_path):
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)

    factory = make_factory()
    from nanoresearch.auth.password import hash_password
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    await UserRepository(factory).create("u1", hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:append-c1", uid="u1")

    mgr = SessionManager(tmp_path, session_factory=factory, default_uid="u1")
    await mgr.append_message("web:append-c1", {"role": "user", "content": "sub-result-1"}, uid="u1")
    await mgr.append_message("web:append-c1", {"role": "user", "content": "sub-result-2"}, uid="u1")

    # Redis session list got both (atomic RPUSH)
    msg_key = RedisKeys.session_msg("u1", "web", "append-c1")
    raw = await redis_client.lrange(msg_key, 0, -1)
    assert len(raw) == 2

    # DB has both messages
    msgs = await ConversationRepository(factory).get_messages(conv.id)
    contents = [m.content.get("content") for m in msgs]
    assert "sub-result-1" in contents and "sub-result-2" in contents
