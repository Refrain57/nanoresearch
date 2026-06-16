"""Tests for OptimizationAgent — candidate generation, scoring, proposal creation."""

from __future__ import annotations

import json
import uuid
import pytest

from nanobot.eval.optimizer import OptimizationAgent, OptimizationCandidate


# ---------------------------------------------------------------------------
# Mock dependencies
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
        return None  # no recordings by default


class _MockRegistry:
    tool_names = ["dummy"]

    def get_definitions(self):
        return [{"name": "dummy", "description": "test"}]

    def register(self, tool):
        pass

    async def execute(self, name, params):
        return f"result:{name}"


# ---------------------------------------------------------------------------
# Tests
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
    async def test_generate_proposals_with_empty_candidates(self):
        provider = _MockProvider()
        provider.add_response("[]")  # empty JSON array → no candidates
        repo = _MockRepo()
        agent = OptimizationAgent(provider=provider, repo=repo, registry=_MockRegistry())

        result = await agent.generate_proposals(
            category="test_cat",
            representative_snapshots=[],
            golden_test_cases=[],
        )
        assert result.category == "test_cat"
        assert result.proposals == []  # no proposals generated

    @pytest.mark.asyncio
    async def test_generate_proposals_handles_malformed_llm_output(self):
        provider = _MockProvider()
        provider.add_response("not json at all")
        repo = _MockRepo()
        agent = OptimizationAgent(provider=provider, repo=repo, registry=_MockRegistry())

        result = await agent.generate_proposals(
            category="test_cat",
            representative_snapshots=[],
            golden_test_cases=[],
        )
        assert result.proposals == []

    @pytest.mark.asyncio
    async def test_generate_proposals_parses_candidates(self):
        provider = _MockProvider()
        provider.add_response(
            '[{"prompt": "be concise", "rationale": "shorter is better"}, {"prompt": "be detailed", "rationale": "more info"}]'
        )
        repo = _MockRepo()
        agent = OptimizationAgent(provider=provider, repo=repo, registry=_MockRegistry())

        result = await agent.generate_proposals(
            category="verbose",
            representative_snapshots=[],
            golden_test_cases=[],
        )
        assert result.proposals != []
        assert len(result.proposals) == 2

    @pytest.mark.asyncio
    async def test_generate_proposals_ranks_by_mean_score(self):
        provider = _MockProvider()
        provider.add_response(
            '[{"prompt": "p1", "rationale": "r1"}, {"prompt": "p2", "rationale": "r2"}]'
        )
        repo = _MockRepo()
        agent = OptimizationAgent(provider=provider, repo=repo, registry=_MockRegistry())

        result = await agent.generate_proposals(
            category="test",
            representative_snapshots=[],
            golden_test_cases=[],
        )
        # All proposals should have rank assigned
        for p in result.proposals:
            assert "rank" in p

    @pytest.mark.asyncio
    async def test_generate_candidates_empty_on_no_match(self):
        provider = _MockProvider()
        provider.add_response("no json array here")
        agent = OptimizationAgent(provider=provider, repo=_MockRepo(), registry=_MockRegistry())

        candidates = await agent._generate_candidates("cat", [])
        assert candidates == []

    @pytest.mark.asyncio
    async def test_generate_candidates_extracts_json_array(self):
        provider = _MockProvider()
        provider.add_response(
            'some text\n[{"prompt": "p1", "rationale": "r1"}]\nmore text'
        )
        agent = OptimizationAgent(provider=provider, repo=_MockRepo(), registry=_MockRegistry())

        candidates = await agent._generate_candidates("cat", [])
        assert len(candidates) == 1
        assert candidates[0].prompt == "p1"
        assert candidates[0].rationale == "r1"

    @pytest.mark.asyncio
    async def test_gather_recordings_returns_empty_when_none_found(self):
        agent = OptimizationAgent(provider=_MockProvider(), repo=_MockRepo(), registry=_MockRegistry())
        tc = type("TC", (), {"id": uuid.uuid4(), "user_input": "hi"})()
        result = await agent._gather_recordings([tc])
        assert result == {}
