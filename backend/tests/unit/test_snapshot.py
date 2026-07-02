"""Tests for RunSnapshotCollector — interleaved ordering, per-call LLM timing,
and input/output capture for the eval observability timeline.

Pure unit tests: the collector has no DB/IO dependencies.
"""

from __future__ import annotations

import nanoresearch.eval.snapshot as snapshot_mod
from nanoresearch.eval.snapshot import RunSnapshotCollector


# ---------------------------------------------------------------------------
# Minimal duck-typed stand-ins for the provider LLMResponse / ToolCallRequest
# ---------------------------------------------------------------------------


class _FakeTC:
    def __init__(self, name, arguments, id="tc1"):
        self.name = name
        self.arguments = arguments
        self.id = id


class _FakeResp:
    def __init__(self, content="", tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


# ---------------------------------------------------------------------------
# Interleaved ordering — LLM calls and tool calls share one monotonic order
# ---------------------------------------------------------------------------


class TestInterleavedOrder:
    def test_llm_and_tool_calls_share_one_increasing_order(self):
        c = RunSnapshotCollector()

        # iteration 1: model plans then calls a tool
        c.on_llm_start()
        c.on_llm_end({"prompt_tokens": 5, "completion_tokens": 3}, "m1")
        c.on_tool_start("t1", "search", {"q": "x"})
        c.on_tool_end("t1", "result-x")
        # iteration 2: model answers
        c.on_llm_start()
        c.on_llm_end({"prompt_tokens": 8, "completion_tokens": 4}, "m1")

        data = c.build(run_id="r1", user_input="hi", final_response="done", status="success")

        assert [x["order"] for x in data.llm_calls] == [1, 3]
        assert [x["order"] for x in data.tool_call_chain] == [2]

        # merged by order → strict LLM→tool→LLM interleave
        merged = sorted(
            [("llm", x["order"]) for x in data.llm_calls]
            + [("tool", x["order"]) for x in data.tool_call_chain],
            key=lambda p: p[1],
        )
        assert [kind for kind, _ in merged] == ["llm", "tool", "llm"]


# ---------------------------------------------------------------------------
# Per-call LLM duration (#10)
# ---------------------------------------------------------------------------


class TestLLMDuration:
    def test_records_per_call_duration_from_start_to_end(self, monkeypatch):
        c = RunSnapshotCollector()
        # patch AFTER construction so __init__'s start_time is untouched
        times = iter([100.0, 100.6])  # on_llm_start, on_llm_end
        monkeypatch.setattr(snapshot_mod.time, "monotonic", lambda: next(times))

        c.on_llm_start()
        c.on_llm_end({"prompt_tokens": 1, "completion_tokens": 1}, "m1")

        assert c._llm_calls[0]["duration_ms"] == 600.0

    def test_duration_none_when_start_not_called(self):
        # backward-compat: callers that never call on_llm_start still work
        c = RunSnapshotCollector()
        c.on_llm_end({"prompt_tokens": 1, "completion_tokens": 1}, "m1")
        assert c._llm_calls[0]["duration_ms"] is None


# ---------------------------------------------------------------------------
# Input (messages) + output capture, with truncation / multimodal stripping
# ---------------------------------------------------------------------------


class TestLLMInputOutputCapture:
    def test_captures_input_messages_and_output(self):
        c = RunSnapshotCollector()
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "search x"},
        ]
        resp = _FakeResp(
            content="the answer",
            tool_calls=[_FakeTC("search", {"q": "x"})],
            reasoning_content="let me think",
        )
        c.on_llm_start()
        c.on_llm_end({"prompt_tokens": 5, "completion_tokens": 3}, "m1",
                     messages=msgs, response=resp)

        call = c._llm_calls[0]
        assert call["input_messages"] == [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "search x"},
        ]
        assert call["output_text"] == "the answer"
        assert call["output_tool_calls"] == [{"name": "search", "arguments": {"q": "x"}}]
        assert call["output_reasoning"] == "let me think"

    def test_output_text_empty_when_only_tool_calls(self):
        # LLMResponse.content is often None on tool-call turns — must not crash
        c = RunSnapshotCollector()
        resp = _FakeResp(content=None, tool_calls=[_FakeTC("search", {"q": "x"})])
        c.on_llm_start()
        c.on_llm_end({}, "m1", messages=[{"role": "user", "content": "hi"}], response=resp)
        assert c._llm_calls[0]["output_text"] == ""

    def test_long_message_content_is_truncated(self):
        c = RunSnapshotCollector()
        big = "a" * 5000
        c.on_llm_start()
        c.on_llm_end({}, "m1", messages=[{"role": "user", "content": big}],
                     response=_FakeResp(content="ok"))
        snap = c._llm_calls[0]["input_messages"][0]["content"]
        assert len(snap) < 5000
        assert snap.endswith("…(truncated)")

    def test_multimodal_content_strips_base64_and_meta(self):
        c = RunSnapshotCollector()
        image_block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + ("Z" * 4000)},
            "_meta": {"path": "/secret/pic.png"},
        }
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                image_block,
            ],
        }]
        c.on_llm_start()
        c.on_llm_end({}, "m1", messages=msgs, response=_FakeResp(content="ok"))
        content = c._llm_calls[0]["input_messages"][0]["content"]
        assert "describe this" in content
        assert "[image]" in content
        assert "Z" * 100 not in content       # base64 payload stripped
        assert "_meta" not in content
        assert "/secret/pic.png" not in content

    def test_input_messages_array_is_capped(self):
        c = RunSnapshotCollector()
        msgs = [{"role": "system", "content": "SYS"}]
        msgs += [{"role": "user", "content": f"m{i}"} for i in range(100)]
        c.on_llm_start()
        c.on_llm_end({}, "m1", messages=msgs, response=_FakeResp(content="ok"))
        snap = c._llm_calls[0]["input_messages"]
        assert len(snap) < len(msgs)                       # capped
        assert snap[0]["content"] == "SYS"                 # first (system) preserved
        assert any("elided" in (m["content"] or "") for m in snap)  # elision marker

    def test_input_none_when_messages_not_passed(self):
        # backward-compat: existing callers that omit messages/response
        c = RunSnapshotCollector()
        c.on_llm_start()
        c.on_llm_end({"prompt_tokens": 1, "completion_tokens": 1}, "m1")
        call = c._llm_calls[0]
        assert call["input_messages"] is None
        assert call["output_text"] == ""
        assert call["output_tool_calls"] == []
