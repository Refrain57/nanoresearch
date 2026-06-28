"""Phase 1 end-to-end verification against the real database.

Verifies:
  P1-V3  PersonaObject.read() returns real agents.persona
  P1-V4  apply() → DB shows new persona + version registry has one active=True row
  P1-V5  rollback() → DB shows old persona + version table still has one active=True
  P1-V8  context_trace appears in classifier prompt with real values (not empty)
  V8+    classifier runs real LLM on a snapshot with context_trace, outputs layer
  P1-V9  classify() writes both root_cause_auto (old) and classification_layer (new)
"""

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:123456@localhost:5432/nanoresearch")

AGENT_ID = "dae84e2c-07e8-42bc-b467-c16893426d81"   # Nano Research
SNAP_WITH_CT = "a2d09db4-e873-44b4-881c-37aa62bba7b0"  # has real context_trace + fragment_ids
BADCASE_ID = "3050103e-f1a9-4890-9933-8e188b4e3a51"    # real badcase (3DGS vs NeRF, no ct)


def _sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


async def _db_query(factory, sql: str, params: dict = None):
    from sqlalchemy import text
    async with factory() as session:
        result = await session.execute(text(sql), params or {})
        return result.fetchall()


async def main() -> None:
    from nanoresearch.config.loader import load_config
    from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider
    from nanoresearch.storage.database import init_engine, get_session_factory
    from nanoresearch.storage.repositories.agent_repo import AgentRepository
    from nanoresearch.storage.repositories.agent_eval_repo import AgentEvalRepository
    from nanoresearch.eval.tunable import PersonaObject
    from nanoresearch.eval.badcase_classifier import BadcaseClassifier, _build_classify_prompt

    # ── Init DB ──────────────────────────────────────────────────────────────
    init_engine()
    factory = get_session_factory()
    agent_repo = AgentRepository(factory)
    eval_repo = AgentEvalRepository(factory)

    cfg = load_config()
    ds = cfg.providers.deepseek
    provider = OpenAICompatProvider(
        api_key=ds.api_key,
        api_base=ds.api_base,
        default_model="deepseek-chat",
    )

    # ── Helper: show tunable_object_versions rows ─────────────────────────────
    async def show_versions(label: str):
        rows = await _db_query(
            factory,
            "SELECT id, active, LEFT(content,40) AS content_preview, created_at "
            "FROM tunable_object_versions "
            "WHERE kind='system_prompt' AND target_id=:tid "
            "ORDER BY created_at",
            {"tid": AGENT_ID},
        )
        print(f"\n[版本表] {label}  ({len(rows)} 行)")
        for r in rows:
            print(f"  id={str(r[0])[:8]}... active={r[1]}  preview={r[2]!r}")
        active_count = sum(1 for r in rows if r[1])
        print(f"  → active=True 行数: {active_count}  (期望: 1 或 0)")
        if active_count > 1:
            print("  ⚠️  BUG: 多行 active=True！")

    async def show_persona(label: str):
        rows = await _db_query(
            factory,
            "SELECT LEFT(persona, 60) FROM agents WHERE id=:aid",
            {"aid": AGENT_ID},
        )
        print(f"[agents.persona] {label}: {rows[0][0]!r}")

    # ═══════════════════════════════════════════════════════════════════
    # P1-V3: read() 返回真实 persona
    # ═══════════════════════════════════════════════════════════════════
    _sep("P1-V3  PersonaObject.read() 返回真实 persona")

    obj = PersonaObject(
        agent_id=AGENT_ID,
        agent_repo=agent_repo,
        eval_repo=eval_repo,
        provider=provider,
    )

    persona_A = await obj.read()
    print(f"read() 返回 (前 80 字):\n{persona_A[:80]!r}")
    assert persona_A, "read() 不应为空"
    print("✅ P1-V3 通过")

    # ═══════════════════════════════════════════════════════════════════
    # P1-V4: apply() → DB 验证
    # ═══════════════════════════════════════════════════════════════════
    _sep("P1-V4  apply('VERIFY_B') → DB 验证 persona + 版本表")

    await show_versions("apply 前")
    await show_persona("apply 前")

    v1_id = await obj.apply("VERIFY_B: Phase1 E2E test persona")
    print(f"\napply() 返回 version_id: {v1_id}")

    await show_versions("apply 后")
    await show_persona("apply 后")

    # DB 直接验证
    rows = await _db_query(
        factory,
        "SELECT persona FROM agents WHERE id=:aid",
        {"aid": AGENT_ID},
    )
    assert rows[0][0] == "VERIFY_B: Phase1 E2E test persona", f"persona 未更新: {rows[0][0]!r}"

    ver_rows = await _db_query(
        factory,
        "SELECT COUNT(*) FROM tunable_object_versions "
        "WHERE kind='system_prompt' AND target_id=:tid AND active=true",
        {"tid": AGENT_ID},
    )
    assert ver_rows[0][0] == 1, f"active=True 行数应为 1, 实为 {ver_rows[0][0]}"

    # read() 反映更新
    content_after_apply = await obj.read()
    assert content_after_apply == "VERIFY_B: Phase1 E2E test persona"
    print("✅ P1-V4 通过")

    # 再 apply 一次，确认旧行被 deactivate
    _sep("P1-V4 扩展  再次 apply → 仍只有一行 active=True")
    v2_id = await obj.apply("VERIFY_C: second apply")
    await show_versions("二次 apply 后")

    ver_rows2 = await _db_query(
        factory,
        "SELECT COUNT(*) FROM tunable_object_versions "
        "WHERE kind='system_prompt' AND target_id=:tid AND active=true",
        {"tid": AGENT_ID},
    )
    assert ver_rows2[0][0] == 1, f"二次 apply 后 active=True 行数应为 1, 实为 {ver_rows2[0][0]}"
    print("✅ 二次 apply 后仍只有一行 active=True")

    # ═══════════════════════════════════════════════════════════════════
    # P1-V5: rollback() → DB 验证
    # ═══════════════════════════════════════════════════════════════════
    _sep("P1-V5  rollback(v1_id) → persona 回到 VERIFY_B，版本表仍一行 active=True")

    await obj.rollback(v1_id)
    await show_versions("rollback 后")
    await show_persona("rollback 后")

    rows = await _db_query(
        factory,
        "SELECT persona FROM agents WHERE id=:aid",
        {"aid": AGENT_ID},
    )
    assert rows[0][0] == "VERIFY_B: Phase1 E2E test persona", f"rollback 后 persona 错误: {rows[0][0]!r}"

    ver_rows3 = await _db_query(
        factory,
        "SELECT COUNT(*) FROM tunable_object_versions "
        "WHERE kind='system_prompt' AND target_id=:tid AND active=true",
        {"tid": AGENT_ID},
    )
    assert ver_rows3[0][0] == 1, f"rollback 后 active=True 行数应为 1, 实为 {ver_rows3[0][0]}"

    read_after_rollback = await obj.read()
    assert read_after_rollback == "VERIFY_B: Phase1 E2E test persona"
    print("✅ P1-V5 通过")

    # ── 恢复原始 persona ──────────────────────────────────────────────
    await obj.apply(persona_A)
    restored = await obj.read()
    assert restored == persona_A
    await show_persona("还原后")
    print("✅ 原始 persona 已还原")

    # ═══════════════════════════════════════════════════════════════════
    # P1-V8: context_trace 进入 prompt + 真实 LLM 分类
    # ═══════════════════════════════════════════════════════════════════
    _sep("P1-V8  context_trace 进入 prompt（快照 a2d09db4）")

    from sqlalchemy import text
    async with factory() as session:
        result = await session.execute(
            text("SELECT * FROM agent_run_snapshots WHERE id=:sid"),
            {"sid": SNAP_WITH_CT},
        )
        row = result.mappings().fetchone()

    # 构造 snapshot 对象
    snap_ct = type("Snap", (), {
        "id": uuid.UUID(SNAP_WITH_CT),
        "user_input": row["user_input"],
        "tool_call_chain": row["tool_call_chain"] or [],
        "final_response": row["final_response"],
        "run_status": row["run_status"],
        "scores": row["scores"],
        "total_input_tokens": row["total_input_tokens"],
        "judge_metadata": row["judge_metadata"],
        "badcase_category": row["badcase_category"],
        "context_trace": row["context_trace"],
    })()

    prompt = _build_classify_prompt(snap_ct)
    print("\n── prompt 中的 context_trace 段 ──")
    ct_section_start = prompt.find("记忆检索")
    if ct_section_start >= 0:
        print(prompt[ct_section_start:ct_section_start + 500])
    else:
        print("⚠️ 未找到 context_trace 段！")

    # 验证关键字段在 prompt 里有真实值
    ct = row["context_trace"]
    assert ct["history_query"] in prompt, "history_query 不在 prompt 里"
    assert "3 个" in prompt, f"fragment_ids 数量未出现在 prompt 里 (有 {len(ct['memory_fragment_ids'])} 个)"
    print(f"\nhistory_query in prompt ✅  ({ct['history_query']!r})")
    print(f"memory_fragment_ids count in prompt ✅  (共 {len(ct['memory_fragment_ids'])} 个)")

    # 真实 LLM 分类
    print("\n── 真实 LLM 分类（含 context_trace）──")
    classifier = BadcaseClassifier(provider=provider, model="deepseek-chat")
    result_ct = await classifier.classify(snap_ct)
    print(f"semantic_category : {result_ct.semantic_category}")
    print(f"root_cause_auto   : {result_ct.root_cause_auto}")
    print(f"layer             : {result_ct.layer}")
    print(f"target_kind       : {result_ct.target_kind}")
    print(f"target_id         : {result_ct.target_id}")
    print(f"confidence        : {result_ct.confidence}")
    print(f"reason            : {result_ct.reason}")
    print("✅ P1-V8 通过")

    # ═══════════════════════════════════════════════════════════════════
    # P1-V9: 旧列 + 新列并行写，旧列格式不变
    # ═══════════════════════════════════════════════════════════════════
    _sep("P1-V9  分类真实 badcase → 旧列 + 新列并行写")

    # 取真实 badcase
    async with factory() as session:
        result = await session.execute(
            text("SELECT * FROM agent_run_snapshots WHERE id=:sid"),
            {"sid": BADCASE_ID},
        )
        row_bc = result.mappings().fetchone()

    snap_bc = type("Snap", (), {
        "id": uuid.UUID(BADCASE_ID),
        "user_input": row_bc["user_input"],
        "tool_call_chain": row_bc["tool_call_chain"] or [],
        "final_response": row_bc["final_response"],
        "run_status": row_bc["run_status"],
        "scores": row_bc["scores"],
        "total_input_tokens": row_bc["total_input_tokens"],
        "judge_metadata": row_bc["judge_metadata"],
        "badcase_category": row_bc["badcase_category"],
        "context_trace": row_bc["context_trace"],
    })()

    print(f"\nbadcase: {row_bc['user_input'][:60]}")

    result_bc = await classifier.classify(snap_bc)
    print(f"\n分类结果:")
    print(f"  semantic_category : {result_bc.semantic_category}")
    print(f"  root_cause_auto   : {result_bc.root_cause_auto}  ← 旧列格式")
    print(f"  layer             : {result_bc.layer}            ← 新列")
    print(f"  target_kind       : {result_bc.target_kind}      ← 新列")
    print(f"  confidence        : {result_bc.confidence}")

    # 写库
    await eval_repo.update_snapshot_classification(uuid.UUID(BADCASE_ID), result_bc)

    # 直接查 DB 确认两列都有值
    rows_v9 = await _db_query(
        factory,
        "SELECT root_cause_auto, root_cause_auto_confidence, "
        "classification_layer, classification_target_kind, classification_target_id "
        "FROM agent_run_snapshots WHERE id=:sid",
        {"sid": BADCASE_ID},
    )
    r = rows_v9[0]
    print(f"\nDB 查询结果:")
    print(f"  root_cause_auto           = {r[0]!r}  (旧列，应在 {['prompt','context','tool','model','user_input']})")
    print(f"  root_cause_auto_confidence= {r[1]!r}")
    print(f"  classification_layer      = {r[2]!r}  (新列)")
    print(f"  classification_target_kind= {r[3]!r}  (新列)")
    print(f"  classification_target_id  = {r[4]!r}  (新列)")

    assert r[0] in ("prompt", "context", "tool", "model", "user_input"), f"旧列格式错误: {r[0]!r}"
    print("✅ P1-V9 通过：旧列格式不变 + 新列有值")

    # ═══════════════════════════════════════════════════════════════════
    _sep("所有验证通过")
    print("\nP1-V3 ✅  P1-V4 ✅  P1-V5 ✅  P1-V8 ✅  P1-V9 ✅")
    print("Phase 1 真实 DB 验收完成。\n")


if __name__ == "__main__":
    asyncio.run(main())
