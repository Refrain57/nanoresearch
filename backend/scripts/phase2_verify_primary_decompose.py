"""Verify the REAL primary-decompose trigger: does the primary main, given a real task, actually
call `decompose_to_board` (and only for multi-specialist tasks)?

Unlike phase2_e2e_collab.py (which invoked the tool directly), this runs the primary's OWN
run_agent_job (real AgentLoop + real LLM) with the decompose tool registered + the agent registry
in its prompt, then checks whether the LLM decomposed:
  Case A (multi-domain task): expect cards created (primary chose to decompose).
  Case B (trivial task): expect NO cards (primary answered directly — must not over-decompose).

Run from backend/:  .venv/Scripts/python.exe scripts/phase2_verify_primary_decompose.py
Requires .env LLM provider + Redis + PostgreSQL. Seeds `p2verify_*`, cleans up.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path


class _CapturingPool:
    def __init__(self) -> None:
        self.jobs: list = []

    async def enqueue_job(self, fn, **kw):
        self.jobs.append((fn, kw))


async def _seed(factory, uid):
    from nanoresearch.auth.password import hash_password
    from nanoresearch.storage.repositories.agent_repo import AgentRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    await UserRepository(factory).create(uid, hash_password("x"))
    primary = await AgentRepository(factory).create({
        "name": "主协调", "created_by": uid, "description": "协调主：把跨专长任务委派给专长主并综合",
        "persona": "你是协调主，负责统筹一支由专长主组成的团队。当任务跨越多个专长领域（例如既要"
                   "研究又要写作）且有对应专长主可用时，用 decompose_to_board 把各部分委派给对应"
                   "专长主（研究交给研究主、写作交给写作主），由他们完成、你负责综合，不要自己"
                   "独揽全部专长工作。简单或单领域问题、闲聊则直接回答，不要拆卡。"})
    research = await AgentRepository(factory).create({
        "name": "研究主", "created_by": uid, "description": "研究专家：事实研究与要点提炼"})
    writing = await AgentRepository(factory).create({
        "name": "写作主", "created_by": uid, "description": "写作专家：把要点组织成连贯文章"})
    return primary, research, writing


async def _make_conv(factory, uid, primary_id):
    from nanoresearch.storage.models import Conversation
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    conv = await ConversationRepository(factory).create(key="web:tmp", uid=uid, agent_id=primary_id)
    async with factory() as db:
        c = await db.get(Conversation, conv.id)
        c.session_key = f"web:{conv.id}"
        await db.commit()
        await db.refresh(c)
        return c


async def _run_primary(ctx, factory, conv, uid, content):
    """Run the primary's real run_agent_job with the decompose tool available."""
    from nanoresearch.server.routers.chat_router import _build_run_payload
    from nanoresearch.worker import run_agent_job
    payload = await _build_run_payload(factory, str(conv.id), uid, content=content,
                                       run_id=uuid.uuid4().hex, agent_id=str(conv.agent_id))
    await run_agent_job(ctx, **payload)


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
    ctx = {"loop_config": loop_config, "session_factory": loop_config["session_factory"],
           "rag_settings": loop_config.get("rag_settings"), "arq_pool": _CapturingPool()}
    factory = ctx["session_factory"]

    from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository
    repo = WorkboardRepository(factory)

    uid = f"p2verify_{uuid.uuid4().hex[:6]}"
    primary, research, writing = await _seed(factory, uid)

    # Case A — multi-domain task (should decompose)
    conv_a = await _make_conv(factory, uid, primary.id)
    print("\n=== Case A (multi-domain: research + write) — expect DECOMPOSE ===")
    await _run_primary(ctx, factory, conv_a, uid,
                       "请深入研究 Transformer 架构的核心机制，然后据此写一篇通顺的中文综述给我。")
    cards_a = await repo.list_by_conversation(conv_a.id)
    print(f"cards created: {len(cards_a)}")
    for c in cards_a:
        tgt = "研究主" if str(c.target_agent_id) == str(research.id) else \
              "写作主" if str(c.target_agent_id) == str(writing.id) else str(c.target_agent_id)
        print(f"  - [{c.status}] {c.title!r} -> {tgt}  spec={ (c.spec or '')[:50]!r}")
    a_ok = len(cards_a) >= 2
    print(f"[Case A] primary decomposed: {'PASS' if a_ok else 'FAIL (did not decompose a multi-domain task)'}")

    # Case B — trivial task (should NOT decompose)
    conv_b = await _make_conv(factory, uid, primary.id)
    print("\n=== Case B (trivial: single fact) — expect NO decompose ===")
    await _run_primary(ctx, factory, conv_b, uid, "1 加 1 等于几？")
    cards_b = await repo.list_by_conversation(conv_b.id)
    b_ok = len(cards_b) == 0
    print(f"cards created: {len(cards_b)}")
    print(f"[Case B] primary answered directly (no over-decompose): {'PASS' if b_ok else 'FAIL (over-decomposed a trivial task)'}")

    print(f"\nRESULT: {'BOTH PASS' if (a_ok and b_ok) else 'CHECK ABOVE — tune tool description / persona'}")
    return (primary, research, writing), (conv_a, conv_b), uid


async def _cleanup(factory, agents, convs, uid, redis):
    from sqlalchemy import text
    from nanoresearch.storage.models import Agent, User
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    try:
        for conv in convs:
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
        from nanoresearch.bus.redis_keys import RedisKeys
        for conv in convs:
            await redis.delete(RedisKeys.board_round(str(conv.id)), RedisKeys.collector_lock(str(conv.id)),
                               RedisKeys.session_msg(uid, "web", str(conv.id)),
                               RedisKeys.session_meta(uid, "web", str(conv.id)))
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
                agents, convs, uid = seeded
                await _cleanup(get_session_factory(), agents, convs, uid, get_redis())
                print("[cleanup] done")
    asyncio.run(_run())
