"""A 类端到端 smoke test — §7.1 验证标准（合成多轮 + 指代消解）。

不依赖真实 LLM 和真实 DB。Mock 层：
- _get_subprocess_session_manager → 返回带合成 user/assistant 的 Session
- PlanQueryTool._call_llm → 返回固定字符串，断言 prompt 内容符合预期
"""

from unittest.mock import AsyncMock, MagicMock, patch

from nanoresearch.rag.mcp_server.tools.agentic import query_planning as qp
from nanoresearch.session.manager import Session


async def test_a_class_end_to_end_rewrite_resolves_pronoun(monkeypatch):
    """Synthetic idx=37：'你说 NeRF 很多，你具体指的是什么' 应被改写。"""
    # 1. mock subprocess SessionManager 返回合成历史
    fake_session = Session(key="test:1")
    fake_session.messages = [
        {"role": "user", "content": "介绍几个 3D 视觉先驱"},
        {"role": "assistant", "content": "Ben Mildenhall（NeRF 一作）, Jonathan Barron（Mip-NeRF）等"},
        {"role": "user", "content": "你说 NeRF 很多，你具体指的是什么"},
    ]
    fake_mgr = AsyncMock()
    fake_mgr.get_or_create = AsyncMock(return_value=fake_session)
    monkeypatch.setattr(qp, "_get_subprocess_session_manager", lambda: fake_mgr)

    # 2. mock LLM client：断言 prompt 含 assistant 内容 + 当前 query
    captured_prompts: list[str] = []

    def fake_call_llm(self, prompt):
        captured_prompts.append(prompt)
        # 模拟 LLM 改写：把"NeRF"改成"Ben Mildenhall / Jonathan Barron 的 NeRF 工作"
        return "Ben Mildenhall 和 Jonathan Barron 的 NeRF 相关工作具体是什么"

    monkeypatch.setattr(qp.PlanQueryTool, "_call_llm", fake_call_llm)

    # 3. mock _ensure_initialized 让 _llm_client 真的不为 None
    def fake_init(self):
        self._llm_client = MagicMock()
        self._llm_model = "test-model"
        self._initialized = True

    monkeypatch.setattr(qp.PlanQueryTool, "_ensure_initialized", fake_init)

    # 4. 跑 execute
    tool = qp.PlanQueryTool()
    response = await tool.execute(
        query="你说 NeRF 很多，你具体指的是什么",
        session_key="test:1",
    )

    # 5. 断言：rewritten_query 跟 original_query 不一样且语义更具体
    import json
    payload = json.loads(response.content)
    assert payload["original_query"] == "你说 NeRF 很多，你具体指的是什么"
    assert payload["rewritten_query"] != payload["original_query"]
    assert "Ben Mildenhall" in payload["rewritten_query"] or "Barron" in payload["rewritten_query"]

    # 6. 断言：prompt 实际含 assistant 内容（证明渲染对了）
    assert any("Ben Mildenhall" in p for p in captured_prompts), \
        "REWRITE_PROMPT 必须含 assistant 渲染内容"
    assert any("[助手]" in p for p in captured_prompts), \
        "渲染层必须给 assistant 加 [助手] 标签"


async def test_a_class_no_session_key_skips_rewrite(monkeypatch):
    """没有 session_key 时不调 SessionManager 也不 rewrite。"""

    fake_mgr = AsyncMock()
    monkeypatch.setattr(qp, "_get_subprocess_session_manager", lambda: fake_mgr)

    def fake_init(self):
        self._llm_client = MagicMock()
        self._llm_model = "test-model"
        self._initialized = True

    monkeypatch.setattr(qp.PlanQueryTool, "_ensure_initialized", fake_init)

    tool = qp.PlanQueryTool()
    response = await tool.execute(query="它是什么")

    import json
    payload = json.loads(response.content)
    assert payload["rewritten_query"] == payload["original_query"] == "它是什么"
    fake_mgr.get_or_create.assert_not_called()


async def test_a_class_subprocess_sm_failure_degrades(monkeypatch):
    """SessionManager 拿不到 → 不抛异常，rewritten = original。"""
    monkeypatch.setattr(qp, "_get_subprocess_session_manager", lambda: None)

    def fake_init(self):
        self._llm_client = MagicMock()
        self._llm_model = "test-model"
        self._initialized = True

    monkeypatch.setattr(qp.PlanQueryTool, "_ensure_initialized", fake_init)

    tool = qp.PlanQueryTool()
    response = await tool.execute(query="它是什么", session_key="test:dead")

    import json
    payload = json.loads(response.content)
    assert payload["rewritten_query"] == "它是什么"
