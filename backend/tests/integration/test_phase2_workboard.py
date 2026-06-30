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


# ---------------------------------------------------------------------------
# Task 3: workboard_cards state machine
# ---------------------------------------------------------------------------

async def _seed_conv_with_agent(uid="u1"):
    factory, agents = await _seed_user_agents(uid=uid, n=1)
    conv = await _seed_conv(factory, uid=uid, agent_id=agents[0].id)
    return factory, conv, agents[0]


async def test_create_card_defaults_backlog():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="do x")
    assert card.status == "backlog"
    got = await repo.get(card.id)
    assert got is not None and got.title == "t"


async def test_transition_status_cas():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="ready")
    assert await repo.transition(card.id, expect_status="ready", to_status="running",
                                 owner_agent_id=agent.id) is True
    assert await repo.transition(card.id, expect_status="ready", to_status="running") is False
    got = await repo.get(card.id)
    assert got.status == "running" and str(got.owner_agent_id) == str(agent.id)


async def test_illegal_transition_rejected():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="done")
    assert await repo.transition(card.id, expect_status="done", to_status="running") is False
    assert (await repo.get(card.id)).status == "done"


async def test_list_by_conversation_filtered():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="a", spec="x", status="ready")
    await repo.create_card(conversation_id=conv.id, title="b", spec="x", status="running")
    await repo.create_card(conversation_id=conv.id, title="c", spec="x", status="done")
    active = await repo.list_by_conversation(conv.id, statuses={"ready", "running"})
    assert {c.title for c in active} == {"a", "b"}
    assert len(await repo.list_by_conversation(conv.id)) == 3


# ---------------------------------------------------------------------------
# Task 4: dependency links + promote (parents-all-done → ready)
# ---------------------------------------------------------------------------

async def test_no_parents_ready_eligible():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="x", spec="x", status="todo")
    assert await repo.parents_all_done(card.id) is True


async def test_child_blocked_until_all_parents_done():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    p1 = await repo.create_card(conversation_id=conv.id, title="p1", spec="x", status="running")
    p2 = await repo.create_card(conversation_id=conv.id, title="p2", spec="x", status="running")
    child = await repo.create_card(conversation_id=conv.id, title="c", spec="x", status="todo")
    await repo.link(p1.id, child.id)
    await repo.link(p2.id, child.id)

    await repo.transition(p1.id, expect_status="running", to_status="done")
    assert await repo.promote_ready_children(p1.id) == []
    assert (await repo.get(child.id)).status == "todo"

    await repo.transition(p2.id, expect_status="running", to_status="done")
    promoted = await repo.promote_ready_children(p2.id)
    assert child.id in promoted
    assert (await repo.get(child.id)).status == "ready"


async def test_promote_idempotent():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    p = await repo.create_card(conversation_id=conv.id, title="p", spec="x", status="running")
    child = await repo.create_card(conversation_id=conv.id, title="c", spec="x", status="todo")
    await repo.link(p.id, child.id)
    await repo.transition(p.id, expect_status="running", to_status="done")
    assert child.id in await repo.promote_ready_children(p.id)
    assert await repo.promote_ready_children(p.id) == []  # already ready (todo CAS fails)
    assert (await repo.get(child.id)).status == "ready"
