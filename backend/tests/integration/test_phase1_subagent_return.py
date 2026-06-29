"""Phase 1 integration: subagent async return to the main agent (real Redis + PG)."""
import pytest

from nanoresearch.bus.redis_keys import RedisKeys
from tests.conftest import make_factory, truncate_all


@pytest.fixture(autouse=True)
def _clean():
    truncate_all()


async def _seed_conv(uid="u1", key="web:bp-c1"):
    from nanoresearch.auth.password import hash_password
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    factory = make_factory()
    await UserRepository(factory).create(uid, hash_password("x"))
    conv = await ConversationRepository(factory).create(key=key, uid=uid)
    return factory, conv


async def test_build_run_payload_rebuilds_config_from_conversation(redis_client):
    from nanoresearch.server.routers.chat_router import _build_run_payload
    factory, conv = await _seed_conv()
    payload = await _build_run_payload(factory, str(conv.id), "u1",
                                       content="请汇总", run_id="orig-run-1")
    assert payload["run_id"] == "orig-run-1"
    assert payload["conversation_id"] == str(conv.id)
    assert payload["content"] == "请汇总"
    assert payload["uid"] == "u1"
    assert payload["session_key"] == conv.session_key  # "web:bp-c1" per seed
    assert "agent_id" in payload and "skill_names" in payload  # config keys present
