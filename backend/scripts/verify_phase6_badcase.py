"""Phase 6 acceptance: trigger a real conversation, collect context_trace,
classify the snapshot as a badcase, and report all fields needed for the
5-minute Phase 6 verification flow.

Usage:
    cd backend
    uv run scripts/verify_phase6_badcase.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:123456@localhost:5432/nanoresearch")
os.environ.setdefault("JWT_SECRET_KEY", "7d66efd73b80d64beb27189a354d3b797857b7fba280367de2caaf4691789e6d")

BASE_URL = "http://localhost:8000"
AGENT_ID = "dae84e2c-07e8-42bc-b467-c16893426d81"
USER_INPUT = "请从知识库里找出所有关于 Transformer 注意力机制的论文,并对比它们的核心创新点。"

POLL_INTERVAL = 3       # seconds between run-status polls
POLL_TIMEOUT  = 300     # max seconds to wait for run to finish


def _sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def _make_token(uid: str) -> str:
    """Generate a JWT token directly from the secret (no password needed)."""
    from nanoresearch.auth.jwt import create_token
    return create_token(uid)


async def _get_admin_uid(factory) -> str:
    from sqlalchemy import text
    async with factory() as session:
        rows = (await session.execute(
            text("SELECT uid FROM users WHERE uid = 'admin' OR role = 'admin' LIMIT 1")
        )).fetchall()
        if not rows:
            rows = (await session.execute(
                text("SELECT uid FROM users ORDER BY created_at ASC LIMIT 1")
            )).fetchall()
    if not rows:
        raise RuntimeError("找不到任何用户，请先运行 init_db.py")
    return rows[0][0]


async def _wait_for_run(client: httpx.AsyncClient, run_id: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        r = await client.get(f"{BASE_URL}/api/runs/{run_id}", headers=headers)
        r.raise_for_status()
        data = r.json()
        status = data.get("status", "")
        if status not in ("pending", "running", ""):
            return data
        print(f"    run status={status!r}, waiting {POLL_INTERVAL}s …")
        await asyncio.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Run {run_id} did not finish within {POLL_TIMEOUT}s")


async def _get_latest_snapshot_with_ct(factory, user_input: str):
    from sqlalchemy import text
    async with factory() as session:
        row = (await session.execute(
            text(
                "SELECT id, user_input, context_trace, tool_call_chain, final_response, "
                "run_status, scores, total_input_tokens, judge_metadata, badcase_category "
                "FROM agent_run_snapshots "
                "WHERE context_trace IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT 1"
            ),
        )).mappings().fetchone()
    return row


async def main() -> None:
    from nanoresearch.config.loader import load_config
    from nanoresearch.eval.badcase_classifier import BadcaseClassifier
    from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider
    from nanoresearch.storage.database import init_engine, get_session_factory
    from nanoresearch.storage.repositories.agent_eval_repo import AgentEvalRepository

    init_engine()
    factory = get_session_factory()
    eval_repo = AgentEvalRepository(factory)

    # ── Step 0: get admin uid + token ────────────────────────────────────────
    _sep("Step 0  获取 admin uid + 生成 JWT token")
    admin_uid = await _get_admin_uid(factory)
    token = _make_token(admin_uid)
    print(f"admin uid : {admin_uid}")
    print(f"token     : {token[:40]}…")

    # ── Step 1: 触发真实对话 ─────────────────────────────────────────────────
    _sep("Step 1  触发对话 → 发送消息给 agent")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        # 1a. 创建 conversation
        r = await client.post(
            f"{BASE_URL}/api/conversations",
            headers=headers,
            json={"agent_id": AGENT_ID, "title": "Phase6 验收测试"},
        )
        r.raise_for_status()
        conv = r.json()
        conv_id = conv["id"]
        print(f"conversation_id : {conv_id}")

        # 1b. 发送消息 → 创建 run
        r = await client.post(
            f"{BASE_URL}/api/runs",
            headers=headers,
            json={"conversation_id": conv_id, "content": USER_INPUT, "rag_mode": "agentic"},
        )
        r.raise_for_status()
        run_info = r.json()
        run_id = run_info["run_id"]
        print(f"run_id          : {run_id}")
        print(f"initial status  : {run_info.get('status')}")

        # 1c. 等待 run 完成
        print(f"\n等待 run 完成（最多 {POLL_TIMEOUT}s）…")
        final_run = await _wait_for_run(client, run_id, token)

    print(f"\nrun 完成: status={final_run.get('status')!r}  "
          f"duration={final_run.get('duration_ms')}ms")

    # ── Step 2: 确认 context_trace ──────────────────────────────────────────
    _sep("Step 2  查库确认 context_trace 不为 null")
    await asyncio.sleep(1)   # 给 worker 一点时间写 snapshot
    row = await _get_latest_snapshot_with_ct(factory, USER_INPUT)
    if row is None:
        print("[WARN] 未找到带 context_trace 的 snapshot。")
        print("  可能原因：Phase 0 上下文装配 trace 未生效，或 worker 尚未写入。")
        print("  请检查 worker 日志，或等几秒再重试。")
        return

    ct = row["context_trace"]
    snap_id = row["id"]
    print(f"snapshot id      : {snap_id}")
    print(f"context_trace 键 : {list(ct.keys())}")

    history_query = ct.get("history_query", "<missing>")
    fragment_ids  = ct.get("memory_fragment_ids") or ct.get("fragment_ids") or []
    budget        = ct.get("memory_budget_tokens", "<missing>")
    print(f"\nhistory_query        : {history_query!r}")
    print(f"memory_budget_tokens : {budget}")
    print(f"fragment_ids 数量    : {len(fragment_ids)}  → {fragment_ids[:3]}")

    assert ct is not None, "context_trace 为 null"
    print("✅ Step 2 通过：context_trace 有数据")

    # ── Step 3: 对 snapshot 跑分类器 ─────────────────────────────────────────
    _sep("Step 3  BadcaseClassifier 分类")
    cfg = load_config()
    ds = cfg.providers.deepseek
    provider = OpenAICompatProvider(
        api_key=ds.api_key,
        api_base=ds.api_base,
        default_model="deepseek-chat",
    )

    snap_obj = type("Snap", (), {
        "id": snap_id,
        "user_input": row["user_input"],
        "tool_call_chain": row["tool_call_chain"] or [],
        "final_response": row["final_response"],
        "run_status": row["run_status"],
        "scores": row["scores"],
        "total_input_tokens": row["total_input_tokens"],
        "judge_metadata": row["judge_metadata"],
        "badcase_category": row["badcase_category"],
        "context_trace": ct,
    })()

    classifier = BadcaseClassifier(provider=provider, model="deepseek-chat")
    result = await classifier.classify(snap_obj)
    print(f"semantic_category  : {result.semantic_category}")
    print(f"root_cause_auto    : {result.root_cause_auto}")
    print(f"confidence         : {result.confidence}")
    print(f"layer              : {result.layer}")
    print(f"target_kind        : {result.target_kind}")
    print(f"target_id          : {result.target_id}")
    print(f"reason             : {result.reason[:120]}")

    await eval_repo.update_snapshot_classification(snap_id, result)
    print("✅ Step 3 通过：分类结果已写库")

    # ── Step 4: 标记 is_badcase=True ────────────────────────────────────────
    _sep("Step 4  mark_badcase()")
    await eval_repo.mark_badcase(
        snapshot_id=snap_id,
        trigger="manual_phase6_acceptance",
        category=result.semantic_category,
    )
    print(f"✅ Step 4 通过：snapshot {snap_id} 已标记为 badcase")

    # ── Step 5: 汇总报告 ─────────────────────────────────────────────────────
    _sep("Step 5  验收汇总")
    fixable_layers = {"Context", "Tool"}
    fixable = result.layer in fixable_layers

    print(f"""
snapshot_id       : {snap_id}
history_query     : {history_query!r}
fragment_ids 数量  : {len(fragment_ids)}
classification:
  layer           : {result.layer}
  fixable         : {fixable}
  target_kind     : {result.target_kind}
  target_id       : {result.target_id}
  semantic_cat    : {result.semantic_category}
  confidence      : {result.confidence}

接下来：
  用这条 snapshot_id 进入 Phase 6 诊断面板 → 候选对比 → apply/rollback 验收流程。
""")


if __name__ == "__main__":
    asyncio.run(main())
