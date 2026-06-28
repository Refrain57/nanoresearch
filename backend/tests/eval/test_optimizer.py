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

    async def create_optimization_proposal(
        self, category, proposals, created_by,
        baseline_score=None, baseline_version_id=None, status="pending",
        score_sample=None,
    ):
        p = type("P", (), {
            "id": uuid.uuid4(),
            "category": category,
            "proposals": proposals,
            "status": status,
            "created_at": None,
            "approved_at": None,
            "created_by": created_by,
            "baseline_score": baseline_score,
            "baseline_version_id": baseline_version_id,
            "score_sample": score_sample,
        })()
        self.created_proposals.append(p)
        return p

    async def get_latest_snapshot_with_recordings(self, user_input, uid=None):
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
    async def test_gather_recordings_returns_empty_when_none_found(self):
        agent = OptimizationAgent(provider=_MockProvider(), repo=_MockRepo(), registry=_MockRegistry())
        # tc must have tool_recordings attribute (None = no recordings); id and user_input required
        tc = type("TC", (), {
            "id": uuid.uuid4(),
            "user_input": "hi",
            "tool_recordings": None,
        })()
        result = await agent._gather_recordings([tc])
        assert result == {}


# ---------------------------------------------------------------------------
# Integration tests — TestGenerateProposalsFullPath
# ---------------------------------------------------------------------------

class TestGenerateProposalsFullPath:
    """Integration tests for generate_proposals using real signatures, no method-mocking."""

    @pytest.mark.asyncio
    async def test_full_path_with_tool_description_target(
        self,
        in_memory_repo,
        fake_llm_provider,
        seeded_tool_registry,
    ):
        """End-to-end: generate proposals for a tool_description target, with real
        fix_test_cases and health_test_cases. Verifies the non-empty invariant at
        optimizer.py:98-107 is bypassed (sets are non-empty) and a proposal is produced."""
        from nanoresearch.eval.optimizer import OptimizationAgent

        agent = OptimizationAgent(
            provider=fake_llm_provider,
            repo=in_memory_repo,
            registry=seeded_tool_registry,
        )
        target = _MockToolTarget()
        fix_cases = [_make_case_with_metadata(dimension="tool_schema_correctness")]
        health_cases = [_make_case_with_metadata(dimension="general_health")]

        proposal = await agent.generate_proposals(
            target=target,
            representative_snapshots=[],
            fix_test_cases=fix_cases,
            health_test_cases=health_cases,
        )
        assert proposal is not None
        assert proposal.proposals  # non-empty list of candidate dicts

    @pytest.mark.asyncio
    async def test_full_path_raises_on_empty_fix_set(
        self, in_memory_repo, fake_llm_provider, seeded_tool_registry
    ):
        from nanoresearch.eval.optimizer import OptimizationAgent

        agent = OptimizationAgent(
            provider=fake_llm_provider, repo=in_memory_repo, registry=seeded_tool_registry
        )
        target = _MockToolTarget()
        with pytest.raises(ValueError, match="fix_test_cases"):
            await agent.generate_proposals(
                target=target,
                representative_snapshots=[],
                fix_test_cases=[],
                health_test_cases=[_make_case_with_metadata(dimension="general_health")],
            )

    @pytest.mark.asyncio
    async def test_full_path_persists_score_sample(
        self, in_memory_repo, fake_llm_provider, seeded_tool_registry
    ):
        """score_sample must be persisted on the proposal with the correct shape and n==3."""
        from nanoresearch.eval.optimizer import OptimizationAgent

        agent = OptimizationAgent(
            provider=fake_llm_provider,
            repo=in_memory_repo,
            registry=seeded_tool_registry,
        )
        target = _MockToolTarget()
        fix_cases = [_make_case_with_metadata(dimension="tool_schema_correctness")]
        health_cases = [_make_case_with_metadata(dimension="general_health")]

        proposal = await agent.generate_proposals(
            target=target,
            representative_snapshots=[],
            fix_test_cases=fix_cases,
            health_test_cases=health_cases,
        )

        assert proposal.score_sample is not None, "score_sample must be persisted"
        assert "fix" in proposal.score_sample
        assert "health" in proposal.score_sample
        # Drill into first candidate, first case — verify mean / std / n all present
        first_cand_fix = next(iter(proposal.score_sample["fix"].values()))
        first_case_sample = next(iter(first_cand_fix.values()))
        assert "mean" in first_case_sample and first_case_sample["mean"] is not None
        assert "std" in first_case_sample and first_case_sample["std"] is not None
        assert "n" in first_case_sample and first_case_sample["n"] == 3

    @pytest.mark.asyncio
    async def test_full_path_raises_on_empty_health_set(
        self, in_memory_repo, fake_llm_provider, seeded_tool_registry
    ):
        from nanoresearch.eval.optimizer import OptimizationAgent

        agent = OptimizationAgent(
            provider=fake_llm_provider, repo=in_memory_repo, registry=seeded_tool_registry
        )
        target = _MockToolTarget()
        with pytest.raises(ValueError, match="health_test_cases"):
            await agent.generate_proposals(
                target=target,
                representative_snapshots=[],
                fix_test_cases=[_make_case_with_metadata(dimension="tool_schema_correctness")],
                health_test_cases=[],
            )


def _make_case_with_metadata(dimension: str):
    """Helper: build a test-case-like namespace with all B4 metadata and duck-typing
    attributes needed by the optimizer.

    Uses SimpleNamespace instead of the SQLAlchemy AgentTestCase ORM class because
    AgentTestCase.tool_recordings is not yet added to the model (fixed in Task 6).
    The optimizer uses duck typing throughout, so this is safe and correct for unit
    and integration tests.
    """
    import types
    from datetime import datetime, timezone

    return types.SimpleNamespace(
        id=uuid.uuid4(),
        dataset_type="fix" if dimension == "tool_schema_correctness" else "health",
        name=f"case_for_{dimension}",
        user_input="test input",
        target_dimension=dimension,
        added_at=datetime.now(timezone.utc),
        added_by="test:integration",
        coverage_tags=[dimension],
        # Fields required by optimizer._gather_recordings and _score_candidate_set:
        tool_recordings=None,    # no pre-recorded tool calls
        expected_tools=None,     # no tool-presence assertion → no sandbox, no skip
        expected_keywords=None,
        expected_intent=None,
        token_budget=None,
        session_history=None,
        human_score=None,
    )
