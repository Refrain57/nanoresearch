"""Phase 2 serial-MVP END-TO-END smoke — REAL AgentLoop + REAL LLM.

Activates two main agents (研究主 + 写作主) on one conversation, creates a relay task
(research card → writing card that depends on it), and drives the FULL board chain INLINE
with the real `run_agent_job` (real loop, real LLM): claim → card-working → result-to-card →
promote → next card → quiesce → collector → user gets a synthesized final answer.

Verifies the two Phase 2 命门:
  ① the user receives ONE synthesized final answer in the conversation session
  ② a card-working run is truly session-read-only (no session:msg write, no consolidation)

Run from backend/ with the project venv:
    .venv/Scripts/python.exe scripts/phase2_e2e_smoke.py

Requires a configured LLM provider (repo-root .env), Redis, and PostgreSQL (the configured
DATABASE_URL). It seeds clearly-marked rows (uid `p2smoke_*`) and deletes them at the end.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path


class _CapturingPool:
    """Captures enqueued jobs so the script drives the relay inline (real loop, no ARQ worker)."""

    def __init__(self) -> None:
        self.jobs: list[tuple[str, dict]] = []

    async def enqueue_job(self, fn: str, **kw) -> None:
        self.jobs.append((fn, kw))


async def main() -> None:
    # 1) .env + ctx exactly like the ARQ worker startup -------------------------------------
    env = Path(__file__).resolve().parent.parent.parent / ".env"
    if env.exists():
        from dotenv import load_dotenv
        load_dotenv(env, override=False)
    from nanoresearch.utils.env_compat import apply_legacy_env_compat
    apply_legacy_env_compat()

    from nanoresearch.cli.commands import build_loop_config
    from nanoresearch.storage.database import init_db

    loop_config = await build_loop_config()
    await init_db()  # ensure workboard_* tables exist on the configured DB (create_all, idempotent)

    pool = _CapturingPool()
    ctx = {
        "loop_config": loop_config,
        "session_factory": loop_config["session_factory"],
        "rag_settings": loop_config.get("rag_settings"),
        "arq_pool": pool,
    }
    factory = ctx["session_factory"]

    from nanoresearch.bus.redis_client import get_redis
    from nanoresearch.bus.redis_keys import RedisKeys
    redis = get_redis()

    # 2) seed: user + 2 mains + conversation (primary/collector = 写作主) ---------------------
    from nanoresearch.auth.password import hash_password
    from nanoresearch.session.manager import SessionManager
    from nanoresearch.storage.models import Agent, Conversation, User
    from nanoresearch.storage.repositories.agent_repo import AgentRepository
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

    uid = f"p2smoke_{uuid.uuid4().hex[:6]}"
    await UserRepository(factory).create(uid, hash_password("x"))
    research = await AgentRepository(factory).create({
        "name": "研究主", "created_by": uid, "description": "事实研究、要点提炼",
        "persona": "你是研究型 agent。只做事实性研究，输出要点式结论，简洁，不要展开成段落。"})
    writing = await AgentRepository(factory).create({
        "name": "写作主", "created_by": uid, "description": "综合写作、润色",
        "persona": "你是写作型 agent。把给定的研究要点组织成连贯通顺的中文段落。"})
    conv = await ConversationRepository(factory).create(key="web:tmp", uid=uid, agent_id=writing.id)
    async with factory() as db:
        c = await db.get(Conversation, conv.id)
        c.session_key = f"web:{conv.id}"
        await db.commit()
        await db.refresh(c)
        conv = c
    await ConversationRepository(factory).activate_agents(conv.id, [research.id, writing.id])

    conv_id = str(conv.id)
    sk = f"web:{conv_id}"
    msg_key = RedisKeys.session_msg(uid, "web", conv_id)
    meta_key = RedisKeys.session_meta(uid, "web", conv_id)

    ws = loop_config["base_workspace"] / "users" / uid
    sessions = SessionManager(ws, session_factory=factory, default_uid=uid)
    # opening user request — so we can prove a card-working run does NOT touch the session
    await sessions.append_message(
        sk, {"role": "user", "content": "请研究 Transformer 并写一段综述给我。"}, uid=uid)

    # 3) cards: research (ready) → writing (todo, depends on research) ------------------------
    repo = WorkboardRepository(factory)
    rc = await repo.create_card(
        conversation_id=conv.id, title="研究 Transformer", status="ready",
        target_agent_id=research.id, depth=0,
        spec="用三条要点总结 Transformer 架构的核心思想（自注意力、并行计算、位置编码）。要点式，简洁。")
    wc = await repo.create_card(
        conversation_id=conv.id, title="写综述", status="todo",
        target_agent_id=writing.id, depth=1,
        spec="把看板上研究卡的要点扩写成一段通顺的中文综述，3-4 句。")
    await repo.link(rc.id, wc.id)

    async def _redis_snapshot() -> tuple[int, int]:
        n = await redis.llen(msg_key)
        lc = int((await redis.hget(meta_key, "last_consolidated")) or 0)
        return n, lc

    len_before, lc_before = await _redis_snapshot()
    print(f"[seed] session:msg len={len_before}, last_consolidated={lc_before}")

    from nanoresearch.worker import _drive_board, run_agent_job

    print("[drive] initial:", await _drive_board(redis, repo, pool, conv_id, uid))

    # 4) drive the relay inline — each run's own _drive_board enqueues the next job -----------
    step = 0
    readonly_pass = None
    while pool.jobs:
        fn, kw = pool.jobs.pop(0)
        step += 1
        kind = "collector" if kw.get("_collect") else ("card-working" if kw.get("_card_id") else "?")
        print(f"\n=== STEP {step}: {kind} run  (agent_id={kw.get('agent_id')})  ===")
        await run_agent_job(ctx, **kw)

        if kind == "card-working" and readonly_pass is None:
            # 命门 ②: a card-working run must not write the shared session
            n_after, lc_after = await _redis_snapshot()
            readonly_pass = (n_after == len_before) and (lc_after == lc_before)
            print(f"[命门②] card-working read-only: session:msg {len_before}->{n_after}, "
                  f"last_consolidated {lc_before}->{lc_after}  => "
                  f"{'PASS' if readonly_pass else 'FAIL'}")

    # 5) 命门 ①: the user received a synthesized final answer --------------------------------
    final = await sessions.get_or_create(sk)
    print("\n=== FINAL CONVERSATION SESSION ===")
    for m in final.messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        print(f"  [{m.get('role')}] {str(content)[:400]}")
    cards = await repo.list_by_conversation(conv.id)
    print("\nCARDS:", [(c.title, c.status, (c.result or '')[:50]) for c in cards])
    has_answer = any(m.get("role") == "assistant" and m.get("content") for m in final.messages)
    all_done = all(c.status in ("done", "blocked") for c in cards)
    print(f"\n[命门①] user received a synthesized assistant answer => {'PASS' if has_answer else 'FAIL'}")
    print(f"[board] all cards terminal => {'PASS' if all_done else 'FAIL'}")
    print(f"\nRESULT: {'ALL PASS' if (readonly_pass and has_answer and all_done) else 'CHECK ABOVE'}")

    return conv, research, writing, uid, msg_key, meta_key, conv_id


async def _cleanup(factory, conv, research, writing, uid, redis, keys) -> None:
    from sqlalchemy import text
    from nanoresearch.storage.models import Agent, User
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    try:
        await ConversationRepository(factory).delete(conv.id)
        async with factory() as db:
            # snapshots FK the uid (not SET NULL) — drop them before the user row
            await db.execute(text("DELETE FROM agent_run_snapshots WHERE uid = :u"), {"u": uid})
            for aid in (research.id, writing.id):
                a = await db.get(Agent, aid)
                if a:
                    await db.delete(a)
            u = await db.get(User, uid)
            if u:
                await db.delete(u)
            await db.commit()
        await redis.delete(*keys)
    except Exception as e:
        print(f"[cleanup] non-fatal: {e}")


if __name__ == "__main__":
    async def _run() -> None:
        from nanoresearch.bus.redis_client import get_redis
        from nanoresearch.bus.redis_keys import RedisKeys
        seeded = None
        try:
            seeded = await main()
        finally:
            if seeded is not None:
                conv, research, writing, uid, msg_key, meta_key, conv_id = seeded
                from nanoresearch.storage.database import get_session_factory  # engine already init
                factory = get_session_factory()
                await _cleanup(
                    factory, conv, research, writing, uid, get_redis(),
                    [msg_key, meta_key, RedisKeys.board_round(conv_id),
                     RedisKeys.collector_lock(conv_id)])
                print("[cleanup] done")

    asyncio.run(_run())
