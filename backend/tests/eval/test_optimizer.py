"""Tests for OptimizationAgent — updated for Phase 1 TunableTextObject API."""

from __future__ import annotations

import uuid
import pytest

from nanoresearch.eval.tunable import OptimizationCandidate
from nanoresearch.eval.optimizer import OptimizationAgent


# ---------------------------------------------------------------------------
# Shared mocks
# ---------------------------------------------------------------------------

class _MockProvider:
    def __init__(self):
        self._responses = []

    def add_response(self, content: str):
        self._responses.append(type("R", (), {"content": content})())

    def get_default_model(self):
        return "test-model"

    async def chat_with_retry(self, messages, system=None, model=None, **kw):
        if self._responses:
            return self._responses.pop(0)
        return type("R", (), {"content": "[]"})()


class _MockRepo:
    def __init__(self):
        self.created_proposals = []

    async def create_optimization_proposal(self, category, proposals, created_by):
        p = type("P", (), {
            "id": uuid.uuid4(),
            "category": category,
            "proposals": proposals,
            "status": "pending",
            "created_at": None,
            "approved_at": None,
            "created_by": created_by,
        })()
        self.created_proposals.append(p)
        return p

    async def get_latest_snapshot_with_recordings(self, user_input):
        return None


class _MockRegistry:
    tool_names = ["dummy"]

    def get_definitions(self):
        return [{"name": "dummy", "description": "test"}]


# ---------------------------------------------------------------------------
# Mock TunableTextObject
# ---------------------------------------------------------------------------

class _MockPersonaTarget:
    kind = "system_prompt"
    target_id = "agent-123"
    _candidates = [
        OptimizationCandidate(prompt="be concise", rationale="shorter is better"),
        OptimizationCandidate(prompt="be detailed", rationale="more info"),
    ]

    async def generate_candidates(self, badcases):
        return self._candidates

    async def read(self):
        return "current persona"

    async def apply(self, content):
        return str(uuid.uuid4())

    async def get_current_version(self):
        return None

    async def rollback(self, version_id):
        pass


class _MockToolTarget:
    kind = "tool_description"
    target_id = "web_search"
    _candidates = [
        OptimizationCandidate(prompt="Search the web for real-time info", rationale="clearer"),
    ]

    async def generate_candidates(self, badcases):
        return self._candidates

    async def read(self):
        return "current description"

    async def apply(self, content):
        return str(uuid.uuid4())

    async def get_current_version(self):
        return None

    async def rollback(self, version_id):
        pass


# ---------------------------------------------------------------------------
# OptimizationAgent tests
# ---------------------------------------------------------------------------

class TestOptimizationAgent:
    def test_init_stores_registry(self):
        agent = OptimizationAgent(
            provider=_MockProvider(),
            repo=_MockRepo(),
            registry=_MockRegistry(),
        )
        assert agent._registry is not None

    def test_model_fallback(self):
        agent = OptimizationAgent(
            provider=_MockProvider(),
            repo=_MockRepo(),
            registry=_MockRegistry(),
        )
        assert agent._model is not None

    @pytest.mark.asyncio
    async def test_generate_proposals_empty_candidates(self):
        class _EmptyTarget(_MockPersonaTarget):
            async def generate_candidates(self, badcases):
                return []

        repo = _MockRepo()
        agent = OptimizationAgent(provider=_MockProvider(), repo=repo, registry=_MockRegistry())
        result = await agent.generate_proposals(
            target=_EmptyTarget(),
            representative_snapshots=[],
            golden_test_cases=[],
        )
        assert result.proposals == []

    @pytest.mark.asyncio
    async def test_generate_proposals_parses_candidates(self):
        repo = _MockRepo()
        agent = OptimizationAgent(provider=_MockProvider(), repo=repo, registry=_MockRegistry())
        result = await agent.generate_proposals(
            target=_MockPersonaTarget(),
            representative_snapshots=[],
            golden_test_cases=[],
        )
        assert len(result.proposals) == 2

    @pytest.mark.asyncio
    async def test_generate_proposals_assigns_rank(self):
        repo = _MockRepo()
        agent = OptimizationAgent(provider=_MockProvider(), repo=repo, registry=_MockRegistry())
        result = await agent.generate_proposals(
            target=_MockPersonaTarget(),
            representative_snapshots=[],
            golden_test_cases=[],
        )
        for p in result.proposals:
            assert "rank" in p

    @pytest.mark.asyncio
    async def test_generate_proposals_tool_description_no_crash(self):
        """ToolDescriptionObject: scoring raises NotImplementedError — proposals still persisted."""
        repo = _MockRepo()
        agent = OptimizationAgent(provider=_MockProvider(), repo=repo, registry=_MockRegistry())
        result = await agent.generate_proposals(
            target=_MockToolTarget(),
            representative_snapshots=[],
            golden_test_cases=[],
        )
        # Should NOT raise; should persist unscored candidates
        assert len(result.proposals) == 1
        assert result.proposals[0]["scores"] == {}

    @pytest.mark.asyncio
    async def test_score_candidate_raises_for_tool_description(self):
        """_score_candidate must raise NotImplementedError for tool_description."""
        agent = OptimizationAgent(provider=_MockProvider(), repo=_MockRepo(), registry=_MockRegistry())
        with pytest.raises(NotImplementedError, match="Phase 4"):
            await agent._score_candidate(
                target=_MockToolTarget(),
                candidate=OptimizationCandidate(prompt="x", rationale="y"),
                golden_test_cases=[],
                recordings_map={},
            )

    @pytest.mark.asyncio
    async def test_gather_recordings_returns_empty_when_none_found(self):
        agent = OptimizationAgent(provider=_MockProvider(), repo=_MockRepo(), registry=_MockRegistry())
        tc = type("TC", (), {"id": uuid.uuid4(), "user_input": "hi"})()
        result = await agent._gather_recordings([tc])
        assert result == {}
