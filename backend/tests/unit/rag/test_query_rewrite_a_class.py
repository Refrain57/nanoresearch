"""A 类透传链 unit tests."""

from unittest.mock import AsyncMock, MagicMock
from nanoresearch.rag.internal_loop.state import SessionState, SessionStateManager


def test_session_state_has_caller_session_key_field():
    """SessionState 必须含 caller_session_key 字段，默认 None。"""
    s = SessionState(session_id="abc", original_query="q")
    assert s.caller_session_key is None


def test_session_state_accepts_caller_session_key():
    s = SessionState(
        session_id="abc",
        original_query="q",
        caller_session_key="telegram:123",
    )
    assert s.caller_session_key == "telegram:123"


def test_create_session_propagates_caller_session_key():
    SessionStateManager.reset_instance()
    mgr = SessionStateManager.get_instance()
    s = mgr.create_session(query="q", caller_session_key="telegram:456")
    assert s.caller_session_key == "telegram:456"


def test_create_session_default_caller_session_key_is_none():
    SessionStateManager.reset_instance()
    mgr = SessionStateManager.get_instance()
    s = mgr.create_session(query="q")
    assert s.caller_session_key is None


def test_rag_search_input_schema_has_session_key():
    from nanoresearch.rag.mcp_server.tools.rag_search import RAGSearchTool
    tool = RAGSearchTool()
    schema = tool.input_schema
    assert "session_key" in schema["properties"]
    assert "session_key" not in schema["required"]
    assert schema["properties"]["session_key"]["type"] == "string"


async def test_rag_search_handler_accepts_session_key(monkeypatch):
    from nanoresearch.rag.mcp_server.tools.rag_search import RAGSearchTool, rag_search_handler

    captured = {}

    async def fake_execute(self, query, collection="default", context=None,
                           max_iterations=5, session_key=None):
        captured["session_key"] = session_key
        from nanoresearch.rag.core.response.response_builder import MCPToolResponse
        return MCPToolResponse(content='{"success": true, "chunks": []}')

    monkeypatch.setattr(RAGSearchTool, "execute", fake_execute)
    await rag_search_handler(
        query="q",
        collection="c",
        session_key="telegram:789",
    )
    assert captured["session_key"] == "telegram:789"


import inspect
from nanoresearch.rag.internal_loop.runner import RAGLoopRunner, run_rag_loop


def test_runner_run_signature_has_session_key():
    sig = inspect.signature(RAGLoopRunner.run)
    assert "session_key" in sig.parameters
    assert sig.parameters["session_key"].default is None


def test_run_rag_loop_signature_has_session_key():
    sig = inspect.signature(run_rag_loop)
    assert "session_key" in sig.parameters


def test_plan_tool_schema_has_session_key():
    from nanoresearch.rag.internal_loop.tools import InternalTools

    tools = InternalTools()
    schema = tools.get_plan_tool_schema()
    params = schema["function"]["parameters"]
    assert "session_key" in params["properties"]
    assert "session_key" not in params.get("required", [])


def test_plan_query_signature_has_session_key():
    from nanoresearch.rag.internal_loop.tools import InternalTools

    sig = inspect.signature(InternalTools.plan_query)
    assert "session_key" in sig.parameters
    assert sig.parameters["session_key"].default is None


async def test_run_plan_phase_forwards_session_key():
    """_run_plan_phase 必须从 session.caller_session_key 读出并传给 plan_query。"""
    from nanoresearch.rag.internal_loop.runner import RAGLoopRunner

    runner = RAGLoopRunner()
    # avoid heavy init
    runner._initialized = True
    runner._tools = AsyncMock()
    runner._tools.plan_query = AsyncMock(return_value={"sub_queries": [], "complexity": "simple"})

    session = SessionState(
        session_id="abc",
        original_query="它是什么",
        caller_session_key="telegram:42",
    )

    await runner._run_plan_phase(session, messages=[])

    call_kwargs = runner.tools.plan_query.call_args.kwargs
    assert call_kwargs.get("session_key") == "telegram:42"
    assert call_kwargs.get("query") == "它是什么"


# MCPToolWrapper.execute() session_key injection tests (Task 6)


def _make_wrapper(original_name: str, session_key: str | None = None):
    """Helper: build a MCPToolWrapper bypassing __init__ heavy paths."""
    from nanoresearch.agent.tools.mcp import MCPToolWrapper

    w = MCPToolWrapper.__new__(MCPToolWrapper)
    w._session = AsyncMock()
    fake_result = MagicMock()
    fake_result.content = []
    w._session.call_tool = AsyncMock(return_value=fake_result)
    w._original_name = original_name
    w._name = f"mcp_test_{original_name}"
    w._description = ""
    w._parameters = {}
    w._tool_timeout = 5
    w._session_key = session_key
    w._kb_map = {"kb1": "col_user_kb1"}
    return w


async def test_mcp_wrapper_injects_session_key_for_kb_search():
    w = _make_wrapper("kb_search", session_key="telegram:99")
    await w.execute(query="x", kb_id="kb1")
    args, kwargs = w._session.call_tool.call_args
    forwarded = kwargs["arguments"]
    assert forwarded.get("session_key") == "telegram:99"


async def test_mcp_wrapper_injects_session_key_for_kb_retrieve():
    w = _make_wrapper("kb_retrieve", session_key="telegram:99")
    await w.execute(query="x", kb_id="kb1")
    forwarded = w._session.call_tool.call_args.kwargs["arguments"]
    assert forwarded.get("session_key") == "telegram:99"


async def test_mcp_wrapper_no_inject_when_session_key_none():
    w = _make_wrapper("kb_search", session_key=None)
    await w.execute(query="x", kb_id="kb1")
    forwarded = w._session.call_tool.call_args.kwargs["arguments"]
    assert "session_key" not in forwarded


async def test_mcp_wrapper_no_inject_for_other_tools():
    """session_key 不能注入到 memory_search / web_search 等——子进程会拒绝。"""
    w = _make_wrapper("memory_search", session_key="telegram:99")
    await w.execute(query="x")
    forwarded = w._session.call_tool.call_args.kwargs["arguments"]
    assert "session_key" not in forwarded


async def test_mcp_wrapper_user_provided_session_key_wins():
    """若 kwargs 已含 session_key（理论上不会发生），使用 setdefault 不覆盖。"""
    w = _make_wrapper("kb_search", session_key="auto")
    await w.execute(query="x", kb_id="kb1", session_key="explicit")
    forwarded = w._session.call_tool.call_args.kwargs["arguments"]
    assert forwarded["session_key"] == "explicit"
