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
