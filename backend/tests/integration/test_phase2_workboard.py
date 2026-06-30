"""Phase 2 serial-MVP workboard tests (real Redis + PG).

Covers Tasks 2-8: membership, card state machine, dependency promote, serial claim,
card-working run mode, collector single-writer, serial termination.
"""
import pytest

from tests.conftest import make_factory, truncate_all


@pytest.fixture(autouse=True)
def _clean():
    truncate_all()


async def _seed_user_agents(uid="u1", n=2):
    from nanoresearch.auth.password import hash_password
    from nanoresearch.storage.repositories.agent_repo import AgentRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    factory = make_factory()
    await UserRepository(factory).create(uid, hash_password("x"))
    agents = [await AgentRepository(factory).create({"name": f"A{i}", "created_by": uid})
              for i in range(n)]
    return factory, agents


async def _seed_conv(factory, uid="u1", agent_id=None):
    from nanoresearch.storage.models import Conversation
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    conv = await ConversationRepository(factory).create(key="web:tmp", uid=uid, agent_id=agent_id)
    async with factory() as db:
        c = await db.get(Conversation, conv.id)
        c.session_key = f"web:{conv.id}"
        await db.commit()
        await db.refresh(c)
        return c


# ---------------------------------------------------------------------------
# Task 2: conversation_agents membership
# ---------------------------------------------------------------------------

async def test_activate_and_list_member_agents():
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    factory, agents = await _seed_user_agents(n=2)
    conv = await _seed_conv(factory, agent_id=agents[0].id)
    repo = ConversationRepository(factory)

    await repo.activate_agents(conv.id, [agents[0].id, agents[1].id])
    members = await repo.list_member_agents(conv.id)

    assert {str(a.id) for a in members} == {str(agents[0].id), str(agents[1].id)}
    assert await repo.is_member(conv.id, agents[1].id) is True
    assert await repo.is_member(conv.id, conv.id) is False  # random uuid not a member


async def test_single_main_default_membership():
    """No explicit activation → membership defaults to {conv.agent_id} (single-main unchanged)."""
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    factory, agents = await _seed_user_agents(n=1)
    conv = await _seed_conv(factory, agent_id=agents[0].id)
    repo = ConversationRepository(factory)

    members = await repo.list_member_agents(conv.id)
    assert {str(a.id) for a in members} == {str(agents[0].id)}


async def test_membership_cascade_delete():
    from sqlalchemy import func, select
    from nanoresearch.storage.models import ConversationAgent
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    factory, agents = await _seed_user_agents(n=1)
    conv = await _seed_conv(factory, agent_id=agents[0].id)
    repo = ConversationRepository(factory)
    await repo.activate_agents(conv.id, [agents[0].id])

    await repo.delete(conv.id)

    async with factory() as db:
        count = (await db.execute(
            select(func.count()).select_from(ConversationAgent)
            .where(ConversationAgent.conversation_id == conv.id))).scalar()
    assert count == 0
