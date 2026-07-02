"""P3: the session summary is written to mem_conv_summaries (conv-scoped), not user_memory."""
from __future__ import annotations

from pathlib import Path

import pytest

from nanoresearch.agent.memory import MemoryStore


class _CapturingKnowledge:
    """Captures summary + event writes."""
    def __init__(self):
        self.summaries: list[dict] = []
        self.events: list = []

    def write_conv_summary_sync(self, text, uid=None, conversation_id=None,
                                turn_start=0, turn_end=0, topic=""):
        self.summaries.append({"text": text, "uid": uid, "conversation_id": conversation_id,
                               "turn_start": turn_start, "turn_end": turn_end})
        return "cs_1"

    def write_events_sync(self, events, uid=None):
        self.events.append((uid, events))
        return [f"ev_{i}" for i in range(len(events))]


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


@pytest.mark.asyncio
async def test_consolidation_summary_written_to_conv_summaries(tmp_path: Path):
    knowledge = _CapturingKnowledge()
    store = MemoryStore(workspace=tmp_path, knowledge_search=knowledge)

    ok = await store.consolidate(
        messages=[{"role": "user", "content": "tell me about CityGaussianV2"}],
        provider=_FakeProvider(), model="fake-model", uid="u1",
        conversation_id="c1", turn_start=0, turn_end=4,
    )

    assert ok is True
    assert knowledge.summaries, "summary must be written to conv_summaries, not dropped"
    s = knowledge.summaries[0]
    assert s["conversation_id"] == "c1" and s["turn_end"] == 4
    assert "CityGaussianV2" in s["text"]
