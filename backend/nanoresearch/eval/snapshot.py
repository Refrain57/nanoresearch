"""Run snapshot collection — zero-copy instrumentation of the agent loop."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Per-LLM-call input/output capture limits (audit view, not full transcript).
# Sampling upstream keeps snapshot volume bounded; these cap a single call.
_MAX_MSG_CHARS = 2000        # per-message content, truncated
_MAX_INPUT_MESSAGES = 40     # messages array cap (keep system + tail, elide middle)
_MAX_OUTPUT_CHARS = 4000     # response text / reasoning, truncated


def _truncate_text(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + "…(truncated)"
    return text


def _stringify_content(content: Any) -> str:
    """Flatten a message ``content`` (str or multimodal list) into a compact string.

    Multimodal blocks carry huge base64 image payloads and internal ``_meta``;
    those are replaced with a ``[image]`` placeholder so snapshots stay small and
    never leak raw base64 / local file paths.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" or ("text" in block and "image_url" not in block):
                    parts.append(str(block.get("text", "")))
                elif btype == "image_url" or "image_url" in block:
                    parts.append("[image]")
                else:
                    parts.append(f"[{btype or 'block'}]")
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _snapshot_messages(messages: Any) -> list[dict] | None:
    """Capture the messages the model saw as {role, content(str, truncated)}.

    Caps the array to _MAX_INPUT_MESSAGES by keeping the first (system) message
    plus the most recent tail, eliding the middle with a marker.
    """
    if not messages:
        return None

    def snap_one(m: Any) -> dict:
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        raw = m.get("content") if isinstance(m, dict) else m
        return {"role": role, "content": _truncate_text(_stringify_content(raw), _MAX_MSG_CHARS)}

    if len(messages) <= _MAX_INPUT_MESSAGES:
        return [snap_one(m) for m in messages]

    tail_n = _MAX_INPUT_MESSAGES - 2
    head = [snap_one(messages[0])]
    tail = [snap_one(m) for m in messages[-tail_n:]]
    elided = len(messages) - 1 - tail_n
    marker = {"role": "system", "content": f"…({elided} earlier messages elided)…"}
    return head + [marker] + tail


@dataclass
class RunSnapshotData:
    run_id: str
    user_input: str
    tool_call_chain: list[dict]
    llm_calls: list[dict]
    final_response: str | None
    run_status: str
    total_input_tokens: int
    total_output_tokens: int
    ttft_ms: float | None
    total_duration_ms: float
    tool_call_count: int
    llm_call_count: int
    retry_count: int
    # Phase 0: context assembly decisions captured at run start (ids + numbers + query text, no injected full text)
    context_trace: dict | None = None


class RunSnapshotCollector:
    """Collects run data during agent execution with no await points (thread-safe for asyncio)."""

    _MAX_RESULT_CHARS = 2000

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._first_token_time: float | None = None
        self._first_token_recorded = False
        self._tool_calls: list[dict] = []
        self._llm_calls: list[dict] = []
        self._retry_count = 0
        self._current_order = 0
        self._current_llm_start: float | None = None
        # keyed by tool_call_id to support concurrent tools
        self._pending: dict[str, tuple[dict, float]] = {}

    def on_first_token(self) -> None:
        if not self._first_token_recorded:
            self._first_token_time = time.monotonic()
            self._first_token_recorded = True

    def on_llm_start(self) -> None:
        """Mark the start of an LLM call so on_llm_end can measure its duration."""
        self._current_llm_start = time.monotonic()

    def on_llm_end(
        self,
        usage: dict,
        model: str,
        messages: Any = None,
        response: Any = None,
    ) -> None:
        # Share the tool-call order counter so LLM and tool events interleave on
        # one timeline (on_llm_end fires before the tools that call requested).
        self._current_order += 1
        duration_ms: float | None = None
        if self._current_llm_start is not None:
            duration_ms = round((time.monotonic() - self._current_llm_start) * 1000, 2)
            self._current_llm_start = None

        output_text = ""
        output_tool_calls: list[dict] = []
        output_reasoning: str | None = None
        if response is not None:
            try:
                output_text = _truncate_text(getattr(response, "content", None) or "", _MAX_OUTPUT_CHARS)
                for tc in (getattr(response, "tool_calls", None) or []):
                    output_tool_calls.append({
                        "name": getattr(tc, "name", None),
                        "arguments": getattr(tc, "arguments", None),
                    })
                reasoning = getattr(response, "reasoning_content", None)
                if reasoning:
                    output_reasoning = _truncate_text(str(reasoning), _MAX_OUTPUT_CHARS)
            except Exception:
                pass

        self._llm_calls.append({
            "order": self._current_order,
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "model": model,
            "duration_ms": duration_ms,
            "input_messages": _snapshot_messages(messages),
            "output_text": output_text,
            "output_tool_calls": output_tool_calls,
            "output_reasoning": output_reasoning,
        })

    def on_tool_start(self, tool_call_id: str, name: str, params: Any) -> None:
        self._current_order += 1
        entry = {
            "order": self._current_order,
            "name": name,
            "params": params if isinstance(params, dict) else {},
        }
        self._pending[tool_call_id] = (entry, time.monotonic())

    def on_tool_end(self, tool_call_id: str, result: Any) -> None:
        item = self._pending.pop(tool_call_id, None)
        if item is None:
            return
        entry, start = item
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        if isinstance(result, str):
            result_str = result
        else:
            try:
                result_str = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                result_str = str(result)
        is_error = result_str.startswith("Error:")
        if is_error:
            self._retry_count += 1
        self._tool_calls.append({
            **entry,
            "result": result_str[:self._MAX_RESULT_CHARS],
            "duration_ms": duration_ms,
            "error": is_error,
        })

    def build(
        self,
        run_id: str | None,
        user_input: str,
        final_response: str | None,
        status: str,
        context_trace: dict | None = None,
    ) -> RunSnapshotData:
        total_ms = round((time.monotonic() - self._start_time) * 1000, 2)
        ttft_ms = (
            round((self._first_token_time - self._start_time) * 1000, 2)
            if self._first_token_time is not None
            else None
        )
        return RunSnapshotData(
            run_id=run_id or str(uuid.uuid4()),
            user_input=user_input[:2000],
            tool_call_chain=self._tool_calls,
            llm_calls=self._llm_calls,
            final_response=final_response,
            run_status=status,
            total_input_tokens=sum(c["input_tokens"] for c in self._llm_calls),
            total_output_tokens=sum(c["output_tokens"] for c in self._llm_calls),
            ttft_ms=ttft_ms,
            total_duration_ms=total_ms,
            tool_call_count=len(self._tool_calls),
            llm_call_count=len(self._llm_calls),
            retry_count=self._retry_count,
            context_trace=context_trace,
        )
