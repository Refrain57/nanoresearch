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


# ---------------------------------------------------------------------------
# Task 5: serial claim (global WIP=1) + claim token
# ---------------------------------------------------------------------------

async def test_claim_moves_ready_to_running_with_token(redis_client):
    from nanoresearch.bus import workboard
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="ready")

    token = await workboard.claim_card(
        redis_client, repo, card_id=card.id, agent_id=agent.id, conv_id=conv.id)

    assert token is not None
    got = await repo.get(card.id)
    assert got.status == "running" and str(got.owner_agent_id) == str(agent.id)
    assert str(got.claim_token) == token
    assert await redis_client.get(RedisKeys.workboard_claim(str(card.id))) == token


async def test_claim_rejects_when_a_card_already_running(redis_client):
    from nanoresearch.bus import workboard
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="r", spec="x", status="running")
    card2 = await repo.create_card(conversation_id=conv.id, title="t2", spec="x", status="ready")

    token = await workboard.claim_card(
        redis_client, repo, card_id=card2.id, agent_id=agent.id, conv_id=conv.id)

    assert token is None  # global WIP=1
    assert (await repo.get(card2.id)).status == "ready"


async def test_heartbeat_then_release_returns_card(redis_client):
    from nanoresearch.bus import workboard
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="ready")
    token = await workboard.claim_card(
        redis_client, repo, card_id=card.id, agent_id=agent.id, conv_id=conv.id)

    assert await workboard.heartbeat_card(redis_client, repo, card_id=card.id, token=token) is True
    assert await workboard.release_card(redis_client, repo, card_id=card.id, token=token) is True
    assert (await repo.get(card.id)).status == "ready"
    assert await redis_client.get(RedisKeys.workboard_claim(str(card.id))) is None


async def test_claim_token_is_dist_lock_token(redis_client):
    """claim token is a real dist_lock token: a second acquire on the card lock fails."""
    from nanoresearch.bus import dist_lock, workboard
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="ready")
    token = await workboard.claim_card(
        redis_client, repo, card_id=card.id, agent_id=agent.id, conv_id=conv.id)

    assert token is not None
    assert await dist_lock.acquire(redis_client, RedisKeys.workboard_claim(str(card.id))) is None


# ---------------------------------------------------------------------------
# Task 6: card-working produce-to-card
# ---------------------------------------------------------------------------

async def test_attach_result_token_guarded(redis_client):
    from nanoresearch.bus import workboard
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="ready")
    token = await workboard.claim_card(
        redis_client, repo, card_id=card.id, agent_id=agent.id, conv_id=conv.id)

    assert await repo.attach_result(card.id, "RES", [{"f": "x"}], token=token) is True
    assert (await repo.get(card.id)).result == "RES"
    assert await repo.attach_result(card.id, "BAD", [], token="wrong") is False
    assert (await repo.get(card.id)).result == "RES"  # unchanged


async def test_create_card_rejects_oversized_spec():
    from nanoresearch.storage.repositories.workboard_repo import WORKBOARD_MAX_SPEC_CHARS, WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x" * 100_000)
    assert len(card.spec) <= WORKBOARD_MAX_SPEC_CHARS


async def test_finish_card_working_ok_marks_done_and_promotes(redis_client):
    import nanoresearch.worker as worker
    from nanoresearch.bus import workboard
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="ready")
    child = await repo.create_card(conversation_id=conv.id, title="c", spec="x", status="todo")
    await repo.link(card.id, child.id)
    token = await workboard.claim_card(
        redis_client, repo, card_id=card.id, agent_id=agent.id, conv_id=conv.id)

    await worker._finish_card_working(
        redis_client, repo, card_id=card.id, token=token, ok=True, result="R", artifacts=[])

    got = await repo.get(card.id)
    assert got.status == "done" and got.result == "R"
    assert (await repo.get(child.id)).status == "ready"
    assert await redis_client.get(RedisKeys.workboard_claim(str(card.id))) is None


async def test_finish_card_working_error_marks_blocked(redis_client):
    import nanoresearch.worker as worker
    from nanoresearch.bus import workboard
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="ready")
    token = await workboard.claim_card(
        redis_client, repo, card_id=card.id, agent_id=agent.id, conv_id=conv.id)

    await worker._finish_card_working(
        redis_client, repo, card_id=card.id, token=token, ok=False, result="boom")

    got = await repo.get(card.id)
    assert got.status == "blocked" and "boom" in (got.result or "")
    assert await redis_client.get(RedisKeys.workboard_claim(str(card.id))) is None


class _FakeArqPool:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, fn, **kw):
        self.jobs.append((fn, kw))


async def test_drive_board_claims_ready_card_and_enqueues(redis_client):
    import nanoresearch.worker as worker
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="do x",
                                  status="ready", target_agent_id=agent.id)
    pool = _FakeArqPool()

    result = await worker._drive_board(redis_client, repo, pool, str(conv.id), conv.uid)

    assert result == f"claimed:{card.id}"
    assert (await repo.get(card.id)).status == "running"
    fn, kw = pool.jobs[0]
    assert fn == "run_agent_job"
    assert kw["_card_id"] == str(card.id) and kw["_card_token"]
    assert kw["content"] == "do x" and kw["agent_id"] == str(agent.id)
    assert kw["session_key"] == f"web:{conv.id}"


async def test_drive_board_wip_busy_does_nothing(redis_client):
    import nanoresearch.worker as worker
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="run", spec="x", status="running")
    await repo.create_card(conversation_id=conv.id, title="rdy", spec="x", status="ready",
                           target_agent_id=agent.id)
    pool = _FakeArqPool()

    result = await worker._drive_board(redis_client, repo, pool, str(conv.id), conv.uid)

    assert result == "wip_busy"
    assert pool.jobs == []


async def test_process_direct_forwards_session_readonly():
    """process_direct forwards session_readonly to _process_message, where the inline guard skips
    the session save + consolidation (serial-MVP conclusion ①(b): card-working never persists the
    shared session). Built via __new__ to avoid AgentLoop's heavy native init in the shared suite."""
    from nanoresearch.agent.loop import AgentLoop
    loop = AgentLoop.__new__(AgentLoop)
    captured = {}

    async def _noop_mcp():
        return None

    async def _fake_pm(msg, **kw):
        captured.update(kw)
        return None

    loop._connect_mcp = _noop_mcp
    loop._process_message = _fake_pm

    await loop.process_direct("hello", session_key="web:c", channel="web", chat_id="c",
                              session_readonly=True)
    assert captured["session_readonly"] is True

    await loop.process_direct("hello", session_key="web:c", channel="web", chat_id="c")
    assert captured["session_readonly"] is False


# ---------------------------------------------------------------------------
# Task 7: collector single-writer
# ---------------------------------------------------------------------------

async def test_is_board_quiesced_true_when_done_and_no_active():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="d", spec="x", status="done")
    assert await repo.is_board_quiesced(conv.id) is True


async def test_is_board_quiesced_false_when_running():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="d", spec="x", status="done")
    await repo.create_card(conversation_id=conv.id, title="r", spec="x", status="running")
    assert await repo.is_board_quiesced(conv.id) is False


async def test_is_board_quiesced_false_when_promotable_todo():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    parent = await repo.create_card(conversation_id=conv.id, title="p", spec="x", status="done")
    child = await repo.create_card(conversation_id=conv.id, title="c", spec="x", status="todo")
    await repo.link(parent.id, child.id)  # child's parents all done → promotable → not quiesced
    assert await repo.is_board_quiesced(conv.id) is False


async def test_is_board_quiesced_false_when_all_collected():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="d", spec="x", status="done")
    await repo.mark_collected(conv.id)
    assert await repo.is_board_quiesced(conv.id) is False  # nothing uncollected to collect


async def test_try_claim_collector_fires_once(redis_client):
    from nanoresearch.bus import workboard
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="d", spec="x", status="done")
    assert await workboard.try_claim_collector(redis_client, repo, str(conv.id)) is True
    assert await workboard.try_claim_collector(redis_client, repo, str(conv.id)) is False


async def test_collect_cards_into_session_appends_and_marks(redis_client, monkeypatch, tmp_path):
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    import nanoresearch.worker as worker
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.session.manager import SessionManager
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    c1 = await repo.create_card(conversation_id=conv.id, title="L1", spec="x", status="done")
    c2 = await repo.create_card(conversation_id=conv.id, title="L2", spec="x", status="done")
    await repo.transition(c1.id, expect_status="done", to_status="done")  # no-op (illegal) ignore
    # set results directly
    async with factory() as db:
        from nanoresearch.storage.models import WorkboardCard
        for cid, res in ((c1.id, "r1"), (c2.id, "r2")):
            card = await db.get(WorkboardCard, cid)
            card.result = res
        await db.commit()
    sk = f"web:{conv.id}"
    sessions = SessionManager(tmp_path, session_factory=factory, default_uid=conv.uid)

    await worker._collect_cards_into_session(redis_client, sessions, repo, str(conv.id), sk, conv.uid)

    raw = await redis_client.lrange(RedisKeys.session_msg(conv.uid, "web", str(conv.id)), 0, -1)
    assert len(raw) == 2
    assert await repo.is_board_quiesced(conv.id) is False  # all collected now


async def test_drive_board_quiesced_enqueues_collector(redis_client):
    import nanoresearch.worker as worker
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="d", spec="x", status="done")
    pool = _FakeArqPool()

    result = await worker._drive_board(redis_client, repo, pool, str(conv.id), conv.uid)

    assert result == f"collect:{conv.id}"
    fn, kw = pool.jobs[0]
    assert fn == "run_agent_job" and kw["_collect"] is True
    assert kw["conversation_id"] == str(conv.id)
