"""A 类透传链 unit tests."""

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


def test_plan_query_signature_has_session_key():
    from nanoresearch.rag.internal_loop.tools import InternalTools

    sig = inspect.signature(InternalTools.plan_query)
    assert "session_key" in sig.parameters
    assert sig.parameters["session_key"].default is None
