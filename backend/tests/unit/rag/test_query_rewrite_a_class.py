"""Tests for RAG Query-Rewrite A-Class features."""

import inspect
from nanoresearch.rag.internal_loop.tools import InternalTools


def test_plan_tool_schema_has_session_key():
    """Test that plan_query schema includes session_key in properties (optional)."""
    tools = InternalTools()
    schema = tools.get_plan_tool_schema()
    params = schema["function"]["parameters"]
    assert "session_key" in params["properties"]
    assert "session_key" not in params.get("required", [])


def test_plan_query_signature_has_session_key():
    """Test that plan_query method signature has session_key parameter."""
    sig = inspect.signature(InternalTools.plan_query)
    assert "session_key" in sig.parameters
    assert sig.parameters["session_key"].default is None
