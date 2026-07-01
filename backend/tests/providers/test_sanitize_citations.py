# backend/tests/providers/test_sanitize_citations.py
from nanoresearch.providers.base import LLMProvider  # holds _sanitize_empty_content


def test_sanitize_strips_top_level_citations():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "_citations": [{"index": 1, "source": "x.pdf"}]},
    ]
    out = LLMProvider._sanitize_empty_content(msgs)
    assert all("_citations" not in m for m in out)
    # do not damage normal fields
    assert out[1]["content"] == "a"
    assert out[1]["role"] == "assistant"


def test_sanitize_keeps_tool_calls():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]}]
    out = LLMProvider._sanitize_empty_content(msgs)
    assert out[0].get("tool_calls") == [{"id": "t1"}]
