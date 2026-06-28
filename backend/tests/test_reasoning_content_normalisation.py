"""Verify reasoning_content extraction tolerates alternate field names."""

from types import SimpleNamespace
from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider


def _msg(content="hi", reasoning_content=None, thinking=None, reasoning=None):
    d = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        d["reasoning_content"] = reasoning_content
    if thinking is not None:
        d["thinking"] = thinking
    if reasoning is not None:
        d["reasoning"] = reasoning
    return SimpleNamespace(
        message=SimpleNamespace(**d, tool_calls=None),
        finish_reason="stop",
    )


def _resp(choices):
    return SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


def test_reads_reasoning_content_canonical():
    resp = _resp([_msg(reasoning_content="thought-A")])
    parsed = OpenAICompatProvider._parse_response(resp)
    assert parsed.reasoning_content == "thought-A"


def test_falls_back_to_thinking_field():
    resp = _resp([_msg(thinking="thought-B")])
    parsed = OpenAICompatProvider._parse_response(resp)
    assert parsed.reasoning_content == "thought-B"


def test_falls_back_to_reasoning_field():
    resp = _resp([_msg(reasoning="thought-C")])
    parsed = OpenAICompatProvider._parse_response(resp)
    assert parsed.reasoning_content == "thought-C"


def test_reasoning_content_wins_over_thinking_when_both_present():
    resp = _resp([_msg(reasoning_content="canonical", thinking="legacy")])
    parsed = OpenAICompatProvider._parse_response(resp)
    assert parsed.reasoning_content == "canonical"


def test_no_reasoning_fields_returns_none():
    resp = _resp([_msg()])
    parsed = OpenAICompatProvider._parse_response(resp)
    assert parsed.reasoning_content is None


def test_dict_branch_falls_back_to_thinking_field():
    """Cover _parse's dict-branch path (HTTP responses parsed as raw dict)."""
    provider = OpenAICompatProvider(api_key="test-key")
    response_dict = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "thinking": "raw-dict-thought",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    parsed = provider._parse(response_dict)
    assert parsed.reasoning_content == "raw-dict-thought"
