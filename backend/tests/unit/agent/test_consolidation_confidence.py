"""C3: consolidation summaries must clear the 0.7 user_memory gate (not be dropped)."""
from __future__ import annotations

from pathlib import Path

import pytest

from nanoresearch.agent.memory import MemoryStore


class _CapturingKnowledge:
    """Captures memories passed to write_user_memory_sync."""
    def __init__(self):
        self.written: list[dict] = []

    def write_user_memory_sync(self, memories, uid=None, conversation_id=None):  # noqa: ARG002
        self.written.extend(memories)
        return (len(memories), 0)


class _FakeToolCall:
    def __init__(self, args):
        self.arguments = args


class _FakeResponse:
    def __init__(self):
        self.finish_reason = "tool_calls"
        self.content = ""
        self.has_tool_calls = True
        self.tool_calls = [_FakeToolCall(
            {"history_entry": "[2026-06-29 12:00] discussed CityGaussianV2 vs NeRF",
             "memory_update": "# User Memory\n## FACTS\n- prefers Python"}
        )]


class _FakeProvider:
    async def chat_with_retry(self, **kwargs):  # noqa: ARG002
        return _FakeResponse()


def _real_gate_passes(confidence: float) -> bool:
    """Mirror knowledge_search.py:153 — items below 0.7 are dropped."""
    return confidence >= 0.7


@pytest.mark.asyncio
async def test_consolidation_summary_clears_07_gate(tmp_path: Path):
    knowledge = _CapturingKnowledge()
    store = MemoryStore(workspace=tmp_path, knowledge_search=knowledge)

    ok = await store.consolidate(
        messages=[{"role": "user", "content": "tell me about CityGaussianV2"}],
        provider=_FakeProvider(), model="fake-model", uid="u1",
    )

    assert ok is True
    assert knowledge.written, "summary must be written, not silently dropped"
    summary = next(m for m in knowledge.written if m["type"] == "consolidation_summary")
    assert summary["confidence"] >= 0.7
    assert _real_gate_passes(summary["confidence"]), "must survive the real 0.7 filter"
