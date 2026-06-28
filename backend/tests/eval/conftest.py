"""Eval-scoped fixtures for optimizer integration tests.

Uses minimal in-memory fakes (NOT MagicMock) so tests exercise the real code paths.
"""

from __future__ import annotations

import uuid
import pytest


# ---------------------------------------------------------------------------
# Minimal in-memory implementations
# ---------------------------------------------------------------------------

class _FakeLLMResponse:
    """Minimal object matching the LLMResponse duck-type expected by AgentRunner."""

    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []
        self.finish_reason = "stop"
        self.usage = {}
        self.reasoning_content = None
        self.thinking_blocks = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class FakeLLMProvider:
    """Fake provider for integration tests.

    First call returns the canned candidate-generation JSON response.
    Subsequent calls (from AgentRunner during scoring) return a plain stop response
    so the agent loop terminates cleanly without tool calls.
    """

    def __init__(self, canned_responses: list[str]):
        self._queue = list(canned_responses)

    def get_default_model(self) -> str:
        return "test-model"

    async def chat_with_retry(self, messages, tools=None, model=None, **kwargs):
        if self._queue:
            content = self._queue.pop(0)
        else:
            # Default stop response for AgentRunner scoring iterations
            content = "I understand your question."
        return _FakeLLMResponse(content=content)


class InMemoryAgentEvalRepo:
    """Minimal in-memory fake for AgentEvalRepository.

    Implements only the methods called by OptimizationAgent.generate_proposals:
      - create_optimization_proposal
      - get_latest_snapshot_with_recordings
    """

    def __init__(self):
        self._proposals: list = []

    async def create_optimization_proposal(
        self,
        category: str,
        proposals: list,
        created_by: str,
        baseline_score=None,
        baseline_version_id=None,
        status: str = "pending",
    ):
        prop = type("Proposal", (), {
            "id": uuid.uuid4(),
            "category": category,
            "proposals": proposals,
            "status": status,
            "created_at": None,
            "approved_at": None,
            "created_by": created_by,
            "baseline_score": baseline_score,
            "baseline_version_id": baseline_version_id,
        })()
        self._proposals.append(prop)
        return prop

    async def get_latest_snapshot_with_recordings(self, user_input: str, uid=None):
        """Always return None — no pre-seeded recordings in integration tests."""
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_repo():
    """Fresh in-memory eval repository for each test."""
    return InMemoryAgentEvalRepo()


@pytest.fixture
def fake_llm_provider():
    """LLM provider that returns a valid candidate JSON on first call.

    The JSON is a list that _llm_generate_candidates can parse via re.search for a JSON array.
    Subsequent calls (AgentRunner scoring) get a plain text stop response.
    """
    canned = '[{"prompt": "Get real-time weather data for any city", "rationale": "more specific"}]'
    return FakeLLMProvider(canned_responses=[canned])


@pytest.fixture
def seeded_tool_registry():
    """ToolRegistry with a minimal weather_tool matching the target_id used in tests."""
    from nanoresearch.agent.tools.registry import ToolRegistry
    from nanoresearch.agent.tools.base import Tool

    class _WeatherTool(Tool):
        name = "weather_tool"
        description = "Get current weather for a location"
        parameters = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
        side_effect = False  # read-only query tool

        async def execute(self, **kwargs):
            return f"Weather in {kwargs.get('location', 'unknown')}: sunny, 22°C"

    reg = ToolRegistry()
    reg.register(_WeatherTool())
    return reg
