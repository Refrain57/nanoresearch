"""Phase 2 REAL-collaboration end-to-end — proves route B (self-claim) with REAL LLM.

Unlike phase2_e2e_smoke.py (which hand-seeded cards and had _drive_board claim FOR the agent),
this drives the real trigger + self-claim chain:
  primary's `decompose_to_board` tool → cards + offer to target's inbox →
  the REAL dispatcher (`AgentDispatcher._handle_notify`) wakes the target →
  the target main, IN ITS OWN run, judges (real LLM) and CLAIMS the card →
  card-working (real LLM) → relay offers the next card → ... → quiesce → collector synthesizes.

路B命门 (route A vs route B discriminator), asserted on the FIRST card:
  ① right after the offer, BEFORE the target's run: card is `ready` AND `owner_agent_id is None`
     (the system did NOT claim it for the agent).
  ② after the target's self-claim run: card is `running`/`done` AND `owner_agent_id == target`
     (the agent claimed it itself, in its own run).
Plus: the user receives ONE synthesized answer from the collector.

Run from backend/ with the venv:  .venv/Scripts/python.exe scripts/phase2_e2e_collab.py
Requires a configured LLM provider (.env), Redis, and PostgreSQL (the configured DATABASE_URL).
Seeds clearly-marked rows (uid `p2collab_*`) and deletes them at the end.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path


class _CapturingPool:
    """Captures collector jobs the relay enqueues (board_offers go to inboxes, not here)."""

    def __init__(self) -> None:
        self.jobs: list[tuple[str, dict]] = []

    async def enqueue_job(self, fn: str, **kw) -> None:
        self.jobs.append((fn, kw))


async def main():
    env = Path(__file__).resolve().parent.parent.parent / ".env"
    if env.exists():
        from dotenv import load_dotenv
        load_dotenv(env, override=False)
    from nanoresearch.utils.env_compat import apply_legacy_env_compat
    apply_legacy_env_compat()

    from nanoresearch.cli.commands import build_loop_config
    from nanoresearch.storage.database import init_db
    loop_config = await build_loop_config()
    await init_db()

    pool = _CapturingPool()
    ctx = {
        "loop_config": loop_config,
        "session_factory": loop_config["session_factory"],
        "rag_settings": loop_config.get("rag_settings"),
        "arq_pool": pool,
    }
    factory = ctx["session_factory"]

    from nanoresearch.bus import mailbox
    from nanoresearch.bus.dispatcher import AgentDispatcher
    from nanoresearch.bus.redis_client import get_redis
    from nanoresearch.bus.redis_keys import RedisKeys
    redis = get_redis()

    # ---- seed: user + primary + 2 specialists + conversation owned by primary ----
    from nanoresearch.auth.password import hash_password
    from nanoresearch.session.manager import SessionManager
    from nanoresearch.storage.models import Conversation
    from nanoresearch.storage.repositories.agent_repo import AgentRepository
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    uid = f"p2collab_{uuid.uuid4().hex[:6]}"
    await UserRepository(factory).create(uid, hash_password("x"))
    primary = await AgentRepository(factory).create({
        "name": "主协调", "created_by": uid, "description": "协调主：拆分复杂任务并综合各专长产出",
        "persona": "你是协调主。复杂的跨专长任务用 decompose_to_board 拆给专长主；最后综合各卡产出回复用户。"})
    research = await AgentRepository(factory).create({
        "name": "研究主", "created_by": uid, "description": "研究专家：事实研究与要点提炼",
        "persona": "你是研究型 agent，只做事实研究、输出要点式结论，简洁。"})
    writing = await AgentRepository(factory).create({
        "name": "写作主", "created_by": uid, "description": "写作专家：把研究要点组织成连贯文章",
        "persona": "你是写作型 agent，把给定的研究要点组织成通顺的中文段落。"})
    conv = await ConversationRepository(factory).create(key="web:tmp", uid=uid, agent_id=primary.id)
    async with factory() as db:
        c = await db.get(Conversation, conv.id)
        c.session_key = f"web:{conv.id}"
        await db.commit()
        await db.refresh(c)
        conv = c
    conv_id = str(conv.id)
    sk = f"web:{conv_id}"

    ws = loop_config["base_workspace"] / "users" / uid
    sessions = SessionManager(ws, session_factory=factory, default_uid=uid)
    await sessions.append_message(sk, {"role": "user", "content": "研究 Transformer 并写一段综述给我。"}, uid=uid)

    repo = WorkboardRepository(factory)

    # ---- trigger: invoke decompose_to_board (the primary's tool) deterministically ----
    from nanoresearch.agent.tools.workboard_plan import DecomposeToBoardTool
    registry = [{"id": str(research.id), "name": research.name, "description": research.description or ""},
                {"id": str(writing.id), "name": writing.name, "description": writing.description or ""}]
    tool = DecomposeToBoardTool(factory, pool)
    tool.set_context(conversation_id=conv_id, uid=uid, primary_agent_id=str(primary.id), agents_registry=registry)
    receipt = await tool.execute(cards=[
        {"title": "研究Transformer", "target_agent": "研究主", "depends_on": [],
         "spec": "用三条要点总结 Transformer 的核心思想（自注意力、并行、位置编码）。要点式，简洁。"},
        {"title": "写综述", "target_agent": "写作主", "depends_on": [0],
         "spec": "把研究卡的要点扩写成一段通顺的中文综述，3-4 句。"},
    ])
    print(f"[decompose] {receipt}")

    cards = {c.title: c for c in await repo.list_by_conversation(conv.id)}
    rc = cards["研究Transformer"]

    # 路B命门 ① — after the offer, BEFORE the target's run: card is ready + UNOWNED
    rc0 = await repo.get(rc.id)
    pre_ok = (rc0.status == "ready" and rc0.owner_agent_id is None)
    print(f"[命门①] after offer, before target run: status={rc0.status} owner={rc0.owner_agent_id} "
          f"=> {'PASS (system did NOT claim — route B)' if pre_ok else 'FAIL (route A: system claimed)'}")

    # ---- drive the chain via the REAL dispatcher logic + inline run_agent_job (real LLM) ----
    from nanoresearch.worker import run_agent_job
    disp = AgentDispatcher(redis, pool, lock_px_ms=600_000)  # long lock so real-LLM runs don't lose it
    _notify_cursor = {"id": "0-0"}  # persistent across calls so we never re-process a notify

    async def _drain_notifies() -> bool:
        """Consume NEW dispatch_notify entries via the real _handle_notify (enqueues self-claim
        runs into `pool`). Returns True if any were processed."""
        ran = False
        while True:
            ms, _, seq = _notify_cursor["id"].partition("-")
            nxt = f"{ms}-{int(seq or 0) + 1}"
            res = await redis.xrange(RedisKeys.DISPATCH_NOTIFY, min=nxt, max="+", count=50)
            if not res:
                break
            for entry_id, fields in res:
                _notify_cursor["id"] = entry_id
                out = await disp._handle_notify(fields)
                print(f"  dispatch._handle_notify -> {out}")
                ran = True
        return ran

    post_owner = None
    step = 0
    for _ in range(20):  # safety bound
        await _drain_notifies()
        if not pool.jobs:
            break
        fn, kw = pool.jobs.pop(0)
        step += 1
        kind = ("collector" if kw.get("_collect") else
                "self-claim" if kw.get("_board_offer_card_id") else "?")
        print(f"\n=== STEP {step}: {kind} run (agent_id={kw.get('agent_id')}) ===")
        await run_agent_job(ctx, **kw)
        if kind == "self-claim" and post_owner is None and kw.get("_board_offer_card_id") == str(rc.id):
            got = await repo.get(rc.id)
            post_owner = got.owner_agent_id
            post_ok = (str(post_owner) == str(research.id))
            print(f"[命门②] after target's self-claim run: status={got.status} owner={post_owner} "
                  f"=> {'PASS (target claimed in its OWN run — route B)' if post_ok else 'FAIL'}")

    # ---- final: user receives a synthesized answer ----
    final = await sessions.get_or_create(sk)
    print("\n=== FINAL CONVERSATION SESSION ===")
    for m in final.messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        print(f"  [{m.get('role')}] {str(content)[:300]}")
    allcards = await repo.list_by_conversation(conv.id)
    print("\nCARDS:", [(c.title, c.status, str(c.owner_agent_id)[:8] if c.owner_agent_id else None) for c in allcards])
    has_answer = any(m.get("role") == "assistant" and m.get("content") for m in final.messages)
    all_terminal = all(c.status in ("done", "blocked") for c in allcards)
    print(f"\n[命门①] system did not claim (offer keeps card unowned): {'PASS' if pre_ok else 'FAIL'}")
    print(f"[命门②] target self-claimed in its own run: {'PASS' if (post_owner and str(post_owner)==str(research.id)) else 'FAIL'}")
    print(f"[delivery] user got a synthesized answer: {'PASS' if has_answer else 'FAIL'}")
    print(f"[board] all cards terminal: {'PASS' if all_terminal else 'FAIL'}")
    return conv, (primary, research, writing), uid, conv_id


async def _cleanup(factory, conv, agents, uid, redis, conv_id):
    from sqlalchemy import text
    from nanoresearch.storage.models import Agent, User
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    try:
        await ConversationRepository(factory).delete(conv.id)
        async with factory() as db:
            await db.execute(text("DELETE FROM agent_run_snapshots WHERE uid = :u"), {"u": uid})
            for a in agents:
                row = await db.get(Agent, a.id)
                if row:
                    await db.delete(row)
            u = await db.get(User, uid)
            if u:
                await db.delete(u)
            await db.commit()
        await redis.delete(RedisKeys.board_round(conv_id), RedisKeys.collector_lock(conv_id),
                           RedisKeys.session_msg(uid, "web", conv_id), RedisKeys.session_meta(uid, "web", conv_id))
    except Exception as e:
        print(f"[cleanup] non-fatal: {e}")


if __name__ == "__main__":
    async def _run():
        from nanoresearch.bus.redis_client import get_redis
        from nanoresearch.storage.database import get_session_factory
        seeded = None
        try:
            seeded = await main()
        finally:
            if seeded is not None:
                conv, agents, uid, conv_id = seeded
                await _cleanup(get_session_factory(), conv, agents, uid, get_redis(), conv_id)
                print("[cleanup] done")
    asyncio.run(_run())
