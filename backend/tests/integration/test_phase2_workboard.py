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


async def test_offer_posts_board_offer_and_leaves_card_ready(redis_client):
    """Driver offers the card to the target's inbox; card stays ready with owner=None (self-claim)."""
    import json
    import nanoresearch.worker as worker
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="do x",
                                  status="ready", target_agent_id=agent.id)
    pool = _FakeArqPool()

    result = await worker._offer_next_or_collect(redis_client, repo, pool, str(conv.id), conv.uid)

    assert result == f"offered:{card.id}"
    # Card must remain ready with no owner — the agent hasn't claimed it yet
    got = await repo.get(card.id)
    assert got.status == "ready"
    assert got.owner_agent_id is None
    # A board_offer message must be in the target's inbox
    inbox_key = RedisKeys.agent_inbox(str(agent.id), str(conv.id))
    entries = await redis_client.xrange(inbox_key, "-", "+")
    assert len(entries) == 1
    _, fields = entries[0]
    payload = json.loads(fields["data"])
    assert payload["kind"] == "board_offer"
    assert payload["card_id"] == str(card.id)
    assert payload["conversation_id"] == str(conv.id)
    # No card-working job enqueued — the offer is the only action
    assert pool.jobs == []


async def test_offer_wip_busy_when_running(redis_client):
    import nanoresearch.worker as worker
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="run", spec="x", status="running")
    await repo.create_card(conversation_id=conv.id, title="rdy", spec="x", status="ready",
                           target_agent_id=agent.id)
    pool = _FakeArqPool()

    result = await worker._offer_next_or_collect(redis_client, repo, pool, str(conv.id), conv.uid)

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


async def test_offer_quiesced_enqueues_collector(redis_client):
    import nanoresearch.worker as worker
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="d", spec="x", status="done")
    pool = _FakeArqPool()

    result = await worker._offer_next_or_collect(redis_client, repo, pool, str(conv.id), conv.uid)

    assert result == f"collect:{conv.id}"
    fn, kw = pool.jobs[0]
    assert fn == "run_agent_job" and kw["_collect"] is True
    assert kw["conversation_id"] == str(conv.id)


# ---------------------------------------------------------------------------
# Task 8: serial termination (board_round + caps + late-drop + watchdog)
# ---------------------------------------------------------------------------

async def test_user_msg_deferred_during_round(redis_client):
    from nanoresearch.bus import mailbox, workboard
    from nanoresearch.bus.dispatcher import AgentDispatcher
    from nanoresearch.bus.redis_keys import RedisKeys
    factory, conv, agent = await _seed_conv_with_agent()
    cid = str(conv.id)
    await mailbox.post_message(redis_client, "none", cid, {"conversation_id": cid, "content": "hi"})
    fields = {"mailbox_key": RedisKeys.agent_inbox("none", cid),
              "cursor_key": RedisKeys.agent_inbox_cursor("none", cid),
              "lock_key": RedisKeys.agent_lock("none", cid)}
    disp = AgentDispatcher(redis_client, _FakeArqPool())

    await workboard.begin_round(redis_client, cid)
    assert await disp._handle_notify(fields) == "deferred_batch"

    await workboard.end_round(redis_client, cid)
    assert await disp._handle_notify(fields) == "enqueued"


async def test_can_create_successor_depth_cap():
    from nanoresearch.storage.repositories.workboard_repo import MAX_SUCCESSOR_DEPTH, WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    assert await repo.can_create_successor(conv.id, parent_depth=0) is True
    assert await repo.can_create_successor(conv.id, parent_depth=MAX_SUCCESSOR_DEPTH) is False


async def test_can_create_successor_count_cap(monkeypatch):
    import nanoresearch.storage.repositories.workboard_repo as wr
    monkeypatch.setattr(wr, "MAX_CARDS_PER_ROUND", 2)
    factory, conv, agent = await _seed_conv_with_agent()
    repo = wr.WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="a", spec="x")
    assert await repo.can_create_successor(conv.id, parent_depth=0) is True
    await repo.create_card(conversation_id=conv.id, title="b", spec="x")
    assert await repo.can_create_successor(conv.id, parent_depth=0) is False


async def test_late_fire_after_collection_is_dropped(redis_client):
    from nanoresearch.bus import workboard
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    await repo.create_card(conversation_id=conv.id, title="d", spec="x", status="done")
    assert await workboard.try_claim_collector(redis_client, repo, str(conv.id)) is True
    await repo.mark_collected(conv.id)
    await workboard.end_round(redis_client, str(conv.id))
    # late completion tries to fire again → board no longer quiesced (all collected) → dropped
    assert await workboard.try_claim_collector(redis_client, repo, str(conv.id)) is False


async def test_watchdog_reaps_stale_running_card(redis_client, monkeypatch):
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    from nanoresearch.heartbeat.stuck_run_watchdog import StuckRunWatchdog
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="r", spec="x", status="running")
    # running card with NO claim lock (lease lapsed) → watchdog reaps it to blocked
    wd = StuckRunWatchdog(redis_client, factory, _FakeArqPool())
    await wd._scan_stale_cards()
    assert (await repo.get(card.id)).status == "blocked"


# ---------------------------------------------------------------------------
# Task 1 (collab layer): workboard_cards.pass_count + record_pass
# ---------------------------------------------------------------------------

async def test_create_card_pass_count_zero():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x")
    assert card.pass_count == 0


async def test_record_pass_increments_and_logs():
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x")

    result1 = await repo.record_pass(card.id, "A")
    got = await repo.get(card.id)
    assert result1 == 1
    assert got.pass_count == 1
    assert {"passed": "A"} in got.artifacts

    result2 = await repo.record_pass(card.id, "B")
    got2 = await repo.get(card.id)
    assert result2 == 2
    assert got2.pass_count == 2
    assert {"passed": "B"} in got2.artifacts
    assert {"passed": "A"} in got2.artifacts


# ---------------------------------------------------------------------------
# Task 3: dispatcher board_offer kind distinction
# ---------------------------------------------------------------------------

def _notify_fields(aid, cid):
    from nanoresearch.bus.redis_keys import RedisKeys
    return {
        "mailbox_key": RedisKeys.agent_inbox(aid, cid),
        "cursor_key": RedisKeys.agent_inbox_cursor(aid, cid),
        "lock_key": RedisKeys.agent_lock(aid, cid),
    }


async def test_dispatcher_board_offer_bypasses_round_gate(redis_client):
    """board_round set + board_offer in inbox → enqueued_self_claim; gate not applied."""
    from nanoresearch.bus import mailbox, workboard
    from nanoresearch.bus.dispatcher import AgentDispatcher

    aid, cid = "agent-A", "disp-offer-1"
    card_id = "card-xyz"
    await mailbox.post_message(redis_client, aid, cid, {
        "kind": "board_offer",
        "card_id": card_id,
        "conversation_id": cid,
        "uid": "u1",
    })
    await workboard.begin_round(redis_client, cid)

    pool = _FakeArqPool()
    disp = AgentDispatcher(redis_client, pool)

    result = await disp._handle_notify(_notify_fields(aid, cid))

    assert result == "enqueued_self_claim"
    assert len(pool.jobs) == 1
    fn, kw = pool.jobs[0]
    assert fn == "run_agent_job"
    assert kw["_board_offer_card_id"] == card_id
    assert kw["agent_id"] == aid
    assert kw["session_key"] == f"web:{cid}"
    assert kw["content"] == ""
    assert kw["uid"] == "u1"
    assert "_lock_token" in kw and kw["_lock_token"]
    assert "_lock_key" in kw and "_entry_id" in kw


async def test_dispatcher_user_turn_still_deferred_in_round(redis_client):
    """board_round set + ordinary user turn (no kind) → deferred_batch (regression)."""
    from nanoresearch.bus import mailbox, workboard
    from nanoresearch.bus.dispatcher import AgentDispatcher

    aid, cid = "agent-B", "disp-offer-2"
    await mailbox.post_message(redis_client, aid, cid, {
        "conversation_id": cid,
        "content": "hello",
        "agent_id": aid,
    })
    await workboard.begin_round(redis_client, cid)

    pool = _FakeArqPool()
    disp = AgentDispatcher(redis_client, pool)

    result = await disp._handle_notify(_notify_fields(aid, cid))

    assert result == "deferred_batch"
    assert pool.jobs == []


async def test_dispatcher_user_turn_enqueues_when_idle(redis_client):
    """No gate conditions → ordinary turn is enqueued (regression)."""
    from nanoresearch.bus import mailbox
    from nanoresearch.bus.dispatcher import AgentDispatcher

    aid, cid = "agent-C", "disp-offer-3"
    await mailbox.post_message(redis_client, aid, cid, {
        "conversation_id": cid,
        "content": "hi",
        "agent_id": aid,
    })

    pool = _FakeArqPool()
    disp = AgentDispatcher(redis_client, pool)

    result = await disp._handle_notify(_notify_fields(aid, cid))

    assert result == "enqueued"
    assert len(pool.jobs) == 1
    assert pool.jobs[0][0] == "run_agent_job"


# ---------------------------------------------------------------------------
# Task 5 (collab): pass reroute + cap + fallback primary
# ---------------------------------------------------------------------------

async def test_reroute_offers_next_member(redis_client):
    """A passes card → rerouted to B; target_agent_id updated, board_offer in B's inbox."""
    import json
    import nanoresearch.worker as worker
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    # primary = agents[0], A = agents[1], B = agents[2]
    factory, agents = await _seed_user_agents(n=3)
    primary, A, B = agents[0], agents[1], agents[2]
    conv = await _seed_conv(factory, agent_id=primary.id)

    conv_repo = ConversationRepository(factory)
    await conv_repo.activate_agents(conv.id, [A.id, B.id])

    repo = WorkboardRepository(factory)
    card = await repo.create_card(
        conversation_id=conv.id, title="t", spec="x",
        status="ready", target_agent_id=A.id,
    )

    result = await worker._reroute_card(
        redis_client, repo, _FakeArqPool(), str(conv.id), conv.uid, card,
        passed_agent_id=str(A.id),
    )

    assert result == f"rerouted:{B.id}"
    got = await repo.get(card.id)
    assert got.target_agent_id == B.id
    assert got.pass_count == 1

    # board_offer must be in B's inbox
    inbox_key = RedisKeys.agent_inbox(str(B.id), str(conv.id))
    entries = await redis_client.xrange(inbox_key, "-", "+")
    assert len(entries) == 1
    _, fields = entries[0]
    payload = json.loads(fields["data"])
    assert payload["kind"] == "board_offer"
    assert payload["card_id"] == str(card.id)


async def test_reroute_fallback_primary_when_all_passed(redis_client):
    """All non-primary members pass → target falls back to primary, offer in primary's inbox."""
    import json
    import nanoresearch.worker as worker
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    factory, agents = await _seed_user_agents(n=3)
    primary, A, B = agents[0], agents[1], agents[2]
    conv = await _seed_conv(factory, agent_id=primary.id)

    conv_repo = ConversationRepository(factory)
    await conv_repo.activate_agents(conv.id, [A.id, B.id])

    repo = WorkboardRepository(factory)
    card = await repo.create_card(
        conversation_id=conv.id, title="t", spec="x",
        status="ready", target_agent_id=A.id,
    )

    # First pass: A → reroute to B
    await worker._reroute_card(
        redis_client, repo, _FakeArqPool(), str(conv.id), conv.uid, card,
        passed_agent_id=str(A.id),
    )

    # Re-fetch card so it reflects updated state
    card = await repo.get(card.id)

    # Second pass: B → fallback to primary
    result = await worker._reroute_card(
        redis_client, repo, _FakeArqPool(), str(conv.id), conv.uid, card,
        passed_agent_id=str(B.id),
    )

    assert result == "fallback_primary"
    got = await repo.get(card.id)
    assert got.target_agent_id == primary.id
    assert got.pass_count == 2

    # board_offer must be in primary's inbox
    inbox_key = RedisKeys.agent_inbox(str(primary.id), str(conv.id))
    entries = await redis_client.xrange(inbox_key, "-", "+")
    assert len(entries) == 1
    _, fields = entries[0]
    payload = json.loads(fields["data"])
    assert payload["kind"] == "board_offer"
    assert payload["card_id"] == str(card.id)


# ---------------------------------------------------------------------------
# Task 4 (collab): self-claim run branch — _decide_board_offer routing
# ---------------------------------------------------------------------------

async def _async_val(v):
    """Helper: return a coroutine that yields v (for monkeypatching async functions)."""
    return v


async def test_self_claim_idempotent_when_not_ready(redis_client):
    """card already done → _decide_board_offer returns 'not_ready', card untouched."""
    import nanoresearch.worker as worker
    from nanoresearch.agent.loop import AgentLoop
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(conversation_id=conv.id, title="t", spec="x", status="done")

    loop = AgentLoop.__new__(AgentLoop)
    pool = _FakeArqPool()

    decision, cid, tok = await worker._decide_board_offer(
        loop, redis_client, repo, pool, str(conv.id), conv.uid, str(agent.id), card)

    assert decision == "not_ready"
    assert cid is None and tok is None
    got = await repo.get(card.id)
    assert got.status == "done"       # unchanged
    assert got.owner_agent_id is None


async def test_self_claim_judge_pass_reroutes(redis_client, monkeypatch):
    """judge returns False → _reroute_card called with passed_agent_id, card not claimed."""
    import nanoresearch.worker as worker
    from nanoresearch.agent.loop import AgentLoop
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(
        conversation_id=conv.id, title="t", spec="x",
        status="ready", target_agent_id=agent.id,
    )

    monkeypatch.setattr(worker, "_judge_claim", lambda *a, **kw: _async_val(False))

    rerouted = []

    async def _fake_reroute(redis, wrepo, arq, conv_id, uid, card_, passed_agent_id):
        rerouted.append(passed_agent_id)
        return "rerouted:test"

    monkeypatch.setattr(worker, "_reroute_card", _fake_reroute)

    loop = AgentLoop.__new__(AgentLoop)
    pool = _FakeArqPool()

    decision, cid, tok = await worker._decide_board_offer(
        loop, redis_client, repo, pool, str(conv.id), conv.uid, str(agent.id), card)

    assert decision == "passed"
    assert cid is None and tok is None
    assert rerouted == [str(agent.id)], f"expected _reroute_card called once; got {rerouted}"
    got = await repo.get(card.id)
    assert got.status == "ready"
    assert got.owner_agent_id is None


async def test_self_claim_judge_claim_claims_card(redis_client, monkeypatch):
    """judge returns True → claim_card succeeds, card transitions to running with owner=agent."""
    import nanoresearch.worker as worker
    from nanoresearch.agent.loop import AgentLoop
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    factory, conv, agent = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(
        conversation_id=conv.id, title="t", spec="do x",
        status="ready", target_agent_id=agent.id,
    )

    monkeypatch.setattr(worker, "_judge_claim", lambda *a, **kw: _async_val(True))

    loop = AgentLoop.__new__(AgentLoop)
    pool = _FakeArqPool()

    decision, cid, tok = await worker._decide_board_offer(
        loop, redis_client, repo, pool, str(conv.id), conv.uid, str(agent.id), card)

    assert decision == "claimed"
    assert cid == str(card.id)
    assert tok is not None
    got = await repo.get(card.id)
    assert got.status == "running"
    assert str(got.owner_agent_id) == str(agent.id)
    assert await redis_client.get(RedisKeys.workboard_claim(str(card.id))) == tok


async def test_self_claim_judge_claim_runs_card_working(redis_client, monkeypatch):
    """run_agent_job with _board_offer_card_id: when judge claims, fall-through wiring fires the
    card-working branch and the card ends status='done' with owner=target.

    Uses a lightweight fake AgentLoop (no real provider/MCP) and monkeypatches _judge_claim →
    True. A real factory + redis_client are used for all DB/Redis state so the claim + finish
    path is exercised end-to-end without constructing a real AgentLoop."""
    import uuid

    import nanoresearch.bus.redis_client as rc
    import nanoresearch.worker as worker
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)

    factory, conv, target = await _seed_conv_with_agent()
    repo = WorkboardRepository(factory)
    card = await repo.create_card(
        conversation_id=conv.id, title="research task", spec="Analyze topic X",
        status="ready", target_agent_id=target.id,
    )

    class _FakeLoop:
        async def process_direct(self, content, **kw):
            return None

        async def close_mcp(self):
            pass

    async def _fake_build_loop(*a, **kw):
        return _FakeLoop()

    monkeypatch.setattr(worker, "_build_agent_loop", _fake_build_loop)
    monkeypatch.setattr(worker, "_judge_claim", lambda *a, **kw: _async_val(True))

    pool = _FakeArqPool()
    ctx = {
        "session_factory": factory,
        "arq_pool": pool,
        "loop_config": {},
        "rag_settings": None,
    }

    await worker.run_agent_job(
        ctx,
        run_id=uuid.uuid4().hex,
        session_key=f"web:{conv.id}",
        content="",
        uid=conv.uid,
        agent_id=str(target.id),
        conversation_id=str(conv.id),
        _board_offer_card_id=str(card.id),
    )

    got = await repo.get(card.id)
    assert got.status == "done", f"expected 'done', got '{got.status}'"
    assert str(got.owner_agent_id) == str(target.id)


# ---------------------------------------------------------------------------
# Task 6 (collab): DecomposeToBoardTool — primary decomposes task into cards
# ---------------------------------------------------------------------------

async def _seed_two_specialist_agents(uid="u1"):
    """Seed user + two specialist agents + a conversation owned by the first agent."""
    from nanoresearch.auth.password import hash_password
    from nanoresearch.storage.repositories.agent_repo import AgentRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    factory = make_factory()
    await UserRepository(factory).create(uid, hash_password("x"))
    research_agent = await AgentRepository(factory).create(
        {"name": "研究主", "created_by": uid, "description": "深度研究专家"})
    writing_agent = await AgentRepository(factory).create(
        {"name": "写作主", "created_by": uid, "description": "文章写作专家"})
    primary_agent = await AgentRepository(factory).create(
        {"name": "主协调", "created_by": uid, "description": "协调主"})
    conv = await _seed_conv(factory, uid=uid, agent_id=primary_agent.id)
    return factory, conv, primary_agent, research_agent, writing_agent


def _make_registry(*agents):
    return [{"id": str(a.id), "name": a.name, "description": a.description or ""}
            for a in agents]


async def test_decompose_creates_cards_and_links(redis_client, monkeypatch):
    """execute() creates cards with correct statuses and links; both agents activated."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)

    from nanoresearch.agent.tools.workboard_plan import DecomposeToBoardTool
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    factory, conv, primary, research, writing = await _seed_two_specialist_agents()
    registry = _make_registry(research, writing)

    tool = DecomposeToBoardTool(factory, _FakeArqPool())
    tool.set_context(
        conversation_id=str(conv.id),
        uid=conv.uid,
        primary_agent_id=str(primary.id),
        agents_registry=registry,
    )

    result = await tool.execute(cards=[
        {"title": "研究阶段", "spec": "深度研究量子计算现状", "target_agent": "研究主", "depends_on": []},
        {"title": "写作阶段", "spec": "根据研究结果撰写综述", "target_agent": "写作主", "depends_on": [0]},
    ])

    assert "2 张卡片" in result, f"receipt should mention 2 cards; got: {result!r}"
    assert "Error" not in result, f"unexpected error: {result!r}"

    repo = WorkboardRepository(factory)
    all_cards = await repo.list_by_conversation(conv.id)
    assert len(all_cards) == 2

    research_card = next(c for c in all_cards if c.title == "研究阶段")
    writing_card = next(c for c in all_cards if c.title == "写作阶段")

    assert research_card.status == "ready", f"root card should be ready; got {research_card.status}"
    assert str(research_card.target_agent_id) == str(research.id)
    assert writing_card.status == "todo", f"dependent card should be todo; got {writing_card.status}"
    assert str(writing_card.target_agent_id) == str(writing.id)

    # Dependency link: research → writing
    assert await repo.parents_all_done(writing_card.id) is False  # research not done yet

    # Both specialist agents (+ primary) should be activated
    members = await ConversationRepository(factory).list_member_agents(conv.id)
    member_ids = {str(m.id) for m in members}
    assert str(research.id) in member_ids, "research agent should be activated"
    assert str(writing.id) in member_ids, "writing agent should be activated"


async def test_decompose_activates_and_offers_first(redis_client, monkeypatch):
    """After execute: board_round key set, first ready card offered to research main's inbox."""
    import json
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)

    from nanoresearch.agent.tools.workboard_plan import DecomposeToBoardTool
    from nanoresearch.bus.redis_keys import RedisKeys

    factory, conv, primary, research, writing = await _seed_two_specialist_agents()
    registry = _make_registry(research, writing)

    tool = DecomposeToBoardTool(factory, _FakeArqPool())
    tool.set_context(
        conversation_id=str(conv.id),
        uid=conv.uid,
        primary_agent_id=str(primary.id),
        agents_registry=registry,
    )

    await tool.execute(cards=[
        {"title": "研究阶段", "spec": "深度研究量子计算", "target_agent": "研究主", "depends_on": []},
        {"title": "写作阶段", "spec": "撰写综述文章", "target_agent": "写作主", "depends_on": [0]},
    ])

    # board_round key must be set
    board_round_key = RedisKeys.board_round(str(conv.id))
    assert await redis_client.get(board_round_key) is not None, "board_round key must be set"

    # A board_offer must be in the research main's inbox (first ready card)
    inbox_key = RedisKeys.agent_inbox(str(research.id), str(conv.id))
    entries = await redis_client.xrange(inbox_key, "-", "+")
    assert len(entries) == 1, f"expected 1 inbox entry for research agent; got {len(entries)}"
    _, fields = entries[0]
    payload = json.loads(fields["data"])
    assert payload["kind"] == "board_offer", f"expected board_offer; got {payload!r}"
    assert "card_id" in payload

    # Writing agent's inbox should be empty (its card is still todo)
    writing_inbox = RedisKeys.agent_inbox(str(writing.id), str(conv.id))
    writing_entries = await redis_client.xrange(writing_inbox, "-", "+")
    assert len(writing_entries) == 0, "writing agent inbox should be empty; card is still todo"


async def test_decompose_unknown_target_returns_error(redis_client, monkeypatch):
    """Unknown target_agent → error string returned, no cards created."""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)

    from nanoresearch.agent.tools.workboard_plan import DecomposeToBoardTool
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    factory, conv, primary, research, writing = await _seed_two_specialist_agents()
    registry = _make_registry(research, writing)

    tool = DecomposeToBoardTool(factory, _FakeArqPool())
    tool.set_context(
        conversation_id=str(conv.id),
        uid=conv.uid,
        primary_agent_id=str(primary.id),
        agents_registry=registry,
    )

    result = await tool.execute(cards=[
        {"title": "任务", "spec": "do something", "target_agent": "不存在的Agent", "depends_on": []},
    ])

    assert "Error" in result, f"expected error string; got: {result!r}"
    assert "不存在的Agent" in result or "不存在" in result

    # No cards should have been created
    repo = WorkboardRepository(factory)
    all_cards = await repo.list_by_conversation(conv.id)
    assert len(all_cards) == 0, f"no cards should be created on error; got {len(all_cards)}"
