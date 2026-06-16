"""Run snapshot collection — zero-copy instrumentation of the agent loop."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


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
        # keyed by tool_call_id to support concurrent tools
        self._pending: dict[str, tuple[dict, float]] = {}

    def on_first_token(self) -> None:
        if not self._first_token_recorded:
            self._first_token_time = time.monotonic()
            self._first_token_recorded = True

    def on_llm_end(self, usage: dict, model: str) -> None:
        self._llm_calls.append({
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "model": model,
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
        )
