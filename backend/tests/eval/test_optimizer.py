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

        # B2 gate audit-log keys must be present on every candidate (real-path coverage).
        for p in proposal.proposals:
            assert "gate_decision" in p
            assert "gate_reason" in p
            assert "sigma_combined" in p
            assert "delta_mean" in p
            assert "threshold" in p

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


# ---------------------------------------------------------------------------
# Task 5 (B2-Gate): σ-weighted gate tests
# ---------------------------------------------------------------------------

class _ScorePatternTarget:
    """Mock TunableTextObject that lets _score_candidate_set return controlled ScoreSamples.

    set_score_pattern / set_score_pattern_per_set configure what each call returns.
    """

    kind = "tool_description"
    target_id = "weather_tool"

    def __init__(self):
        self._pattern_global: dict | None = None
        self._pattern_per_set: dict[str, dict] | None = None
        self._call_count: dict[str, int] = {}  # key="fix"|"health", counter for repeats

    def set_score_pattern(self, baseline_mean, baseline_std, candidate_mean, candidate_std):
        """Same pattern for both fix and health sets."""
        self._pattern_global = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
        }
        self._pattern_per_set = None

    def set_score_pattern_per_set(self, fix: dict, health: dict):
        """Different pattern for fix vs health sets."""
        self._pattern_per_set = {"fix": fix, "health": health}
        self._pattern_global = None

    def _get_pattern(self, set_name: str) -> dict:
        if self._pattern_per_set:
            return self._pattern_per_set[set_name]
        if self._pattern_global:
            return self._pattern_global
        return {"baseline_mean": 0.5, "baseline_std": 0.0, "candidate_mean": 0.5, "candidate_std": 0.0}

    async def generate_candidates(self, badcases):
        from nanoresearch.eval.tunable import OptimizationCandidate
        return [OptimizationCandidate(prompt="improved weather tool desc", rationale="test candidate")]

    async def read(self):
        return "original weather tool desc"

    async def apply(self, content):
        import uuid as _uuid
        return str(_uuid.uuid4())

    async def get_current_version(self):
        return None

    async def rollback(self, version_id):
        pass


def _build_score_pattern_agent(fake_llm_provider, in_memory_repo, seeded_tool_registry):
    """Build OptimizationAgent with an evaluator that returns controlled scores.

    The agent's _score_one is monkey-patched post-construction to return scores
    derived from the target's score pattern, bypassing the real RuleEvaluator.
    This is approach (a): inject a fake evaluator at the OptimizationAgent level.

    Returns (agent, target) where target is a _ScorePatternTarget.
    """
    from nanoresearch.eval.optimizer import OptimizationAgent

    agent = OptimizationAgent(
        provider=fake_llm_provider,
        repo=in_memory_repo,
        registry=seeded_tool_registry,
    )

    target = _ScorePatternTarget()
    _patch_score_one_for_target(agent, target)
    return agent, target


def _patch_score_one_for_target(agent, target: _ScorePatternTarget):
    """Monkey-patch agent._score_one to return controlled per-call scores.

    The patch determines which set is being scored by inspecting the current
    _active_set attribute that we set on the target during _score_candidate_set.
    Since _score_one gets (target, candidate, tc, recordings_map), we track
    which set is active via a thread-local-style attribute on the target.

    Call sequence per (candidate, set):
      - baseline call for the set
      - candidate call for the same set
    We differentiate baseline vs candidate via the candidate.prompt comparison.
    """
    import nanoresearch.eval.optimizer as _opt_module

    # We override _score_candidate_set at agent level to return pre-built ScoreSamples.
    # This is the cleanest way: replace _score_candidate_set on the agent instance.
    import asyncio as _asyncio
    from nanoresearch.eval.score_sample import ScoreSample as _ScoreSample
    import math as _math

    baseline_text_holder = []  # will be set after agent reads the target

    async def _fake_score_candidate_set(self_inner, tgt, candidate, test_cases, recordings_map):
        # Fix vs health is disambiguated by the FIRST test case's dataset_type.
        # We assert rather than fall back: silent default would mask test-fixture drift.
        assert test_cases, "_fake_score_candidate_set requires at least one test case"
        dt = getattr(test_cases[0], "dataset_type", None)
        assert dt in ("fix", "health"), (
            f"test case {test_cases[0].id} must have dataset_type='fix' or 'health' "
            f"(got {dt!r}); update _make_case_with_metadata or the test fixture"
        )
        set_name = dt

        pattern = tgt._get_pattern(set_name)

        # Determine if this is baseline or candidate call.
        # After generate_proposals calls target.read(), it creates the baseline candidate
        # with prompt == baseline_text.  We capture the first candidate text as baseline.
        is_baseline = (candidate.prompt == tgt._baseline_text) if hasattr(tgt, "_baseline_text") else False

        if is_baseline:
            mean = pattern["baseline_mean"]
            std = pattern["baseline_std"]
        else:
            mean = pattern["candidate_mean"]
            std = pattern["candidate_std"]

        # Build one ScoreSample per test case with controlled mean/std/n.
        # n=3 to match _SCORE_REPEAT_N; std comes from pattern.
        result = {}
        for tc in test_cases:
            # With n=3 repeat, produce observations that give approx mean and std.
            # Use 3 observations: [mean-std, mean, mean+std] → sample std ≈ std.
            if std == 0.0:
                obs = [mean, mean, mean]
            else:
                obs = [mean - std, mean, mean + std]
            result[str(tc.id)] = _ScoreSample.from_observations(obs)
        # Return tuple (scores, fuzzy_ratio) to match the real _score_candidate_set signature.
        return result, 0.0

    # Store baseline text on target after read() is called.
    # We patch target.read() to record what it returns.
    _orig_read = target.read

    async def _patched_read():
        text = await _orig_read()
        target._baseline_text = text
        return text

    target.read = _patched_read

    import types as _types
    agent._score_candidate_set = _types.MethodType(_fake_score_candidate_set, agent)


class TestSigmaWeightedGate:
    """Tests for the σ-weighted optimization gate (B2)."""

    @pytest.mark.asyncio
    async def test_gate_rejects_candidate_within_noise_envelope(
        self, in_memory_repo, fake_llm_provider, seeded_tool_registry
    ):
        """A candidate whose delta < k·σ_combined is rejected even if delta > 0."""
        from nanoresearch.eval.optimizer import _GATE_SIGMA_K

        agent, target = _build_score_pattern_agent(
            fake_llm_provider, in_memory_repo, seeded_tool_registry
        )
        # delta = 0.52 - 0.50 = 0.02; σ_combined ≈ 0.08 → threshold ≈ 1.96 * 0.08 ≈ 0.157
        # delta < threshold → rejected
        target.set_score_pattern(
            baseline_mean=0.50, baseline_std=0.08,
            candidate_mean=0.52, candidate_std=0.08,
        )

        proposal = await agent.generate_proposals(
            target=target,
            representative_snapshots=[],
            fix_test_cases=[_make_case_with_metadata("tool_schema_correctness")],
            health_test_cases=[_make_case_with_metadata("general_health")],
        )
        assert all(p["gate_decision"] == "rejected" for p in proposal.proposals)
        assert all(p["gate_reason"] == "within_noise_envelope" for p in proposal.proposals)

    @pytest.mark.asyncio
    async def test_gate_approves_candidate_beyond_noise(
        self, in_memory_repo, fake_llm_provider, seeded_tool_registry
    ):
        """A candidate whose delta > k·σ_combined and health not regressing is approved."""
        agent, target = _build_score_pattern_agent(
            fake_llm_provider, in_memory_repo, seeded_tool_registry
        )
        # delta = 0.70 - 0.50 = 0.20; σ_combined ≈ 0.02 → threshold ≈ 1.96 * 0.02 ≈ 0.055
        # delta >> threshold → approved
        target.set_score_pattern(
            baseline_mean=0.50, baseline_std=0.02,
            candidate_mean=0.70, candidate_std=0.02,
        )

        proposal = await agent.generate_proposals(
            target=target,
            representative_snapshots=[],
            fix_test_cases=[_make_case_with_metadata("tool_schema_correctness")],
            health_test_cases=[_make_case_with_metadata("general_health")],
        )
        assert any(p["gate_decision"] == "approved" for p in proposal.proposals)

    @pytest.mark.asyncio
    async def test_gate_rejects_when_health_regresses_beyond_noise(
        self, in_memory_repo, fake_llm_provider, seeded_tool_registry
    ):
        """fix delta > threshold but health delta < -threshold → rejected (regression)."""
        agent, target = _build_score_pattern_agent(
            fake_llm_provider, in_memory_repo, seeded_tool_registry
        )
        target.set_score_pattern_per_set(
            fix={"baseline_mean": 0.50, "baseline_std": 0.02,
                 "candidate_mean": 0.70, "candidate_std": 0.02},
            health={"baseline_mean": 0.80, "baseline_std": 0.02,
                    "candidate_mean": 0.50, "candidate_std": 0.02},  # delta=-0.30
        )

        proposal = await agent.generate_proposals(
            target=target,
            representative_snapshots=[],
            fix_test_cases=[_make_case_with_metadata("tool_schema_correctness")],
            health_test_cases=[_make_case_with_metadata("general_health")],
        )
        assert all(p["gate_decision"] == "rejected" for p in proposal.proposals)
        assert all(p["gate_reason"] == "health_regression" for p in proposal.proposals)


def _make_case_with_metadata(dimension: str):
    """Helper: build a test-case-like namespace with all B4 metadata and duck-typing
    attributes needed by the optimizer.

    Uses SimpleNamespace instead of the SQLAlchemy AgentTestCase ORM class because
    it avoids a real DB connection. The optimizer uses duck typing throughout,
    so this is safe and correct for unit and integration tests.
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
        # Real criterion: fake_llm_provider's default stop response is
        # "I understand your question." — "question" is always present.
        expected_keywords=["question"],
        expected_intent=None,
        token_budget=None,
        session_history=None,
        human_score=None,
    )


# ---------------------------------------------------------------------------
# Task 6 (B2-Fuzzy): fuzzy-match-driven signal_unreliable tests
# ---------------------------------------------------------------------------

def _make_high_fuzzy_recordings() -> dict:
    """Build a recordings dict whose keys are in the JSON-object format with
    extra whitespace in values — exact match will fail; fuzzy match will succeed.

    Returns a dict: {case_id_str: recordings_json_str}
    used by the seeded_tool_registry_high_fuzzy fixture / _MockHighFuzzyRepo.
    """
    import json
    # Recorded with whitespace padding in value → exact match fails; fuzzy strips it.
    recorded_dict = {
        '{"tool":"weather_tool","params":{"location":"  Paris  "}}': "Weather: cloudy",
    }
    return json.dumps(recorded_dict)


class _MockHighFuzzyRepo:
    """In-memory repo that returns recordings whose keys only fuzzy-match, not exact-match."""

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


def _make_high_fuzzy_case(dimension: str):
    """Build a test case whose tool_recordings will only fuzzy-match in the sandbox.

    The recordings dict uses whitespace-padded param values so the exact key
    (built from the unpadded agent call) misses, triggering the fuzzy path.
    """
    import types, json
    from datetime import datetime, timezone

    # Recorded with padded location; agent will call with "Paris" (no padding)
    recorded_dict = {
        '{"tool":"weather_tool","params":{"location":"  Paris  "}}': "Weather: cloudy",
    }
    recordings_json = json.dumps(recorded_dict)

    return types.SimpleNamespace(
        id=uuid.uuid4(),
        dataset_type="fix" if dimension == "tool_schema_correctness" else "health",
        name=f"high_fuzzy_case_for_{dimension}",
        user_input="What is the weather in Paris?",
        target_dimension=dimension,
        added_at=datetime.now(timezone.utc),
        added_by="test:integration",
        coverage_tags=[dimension],
        tool_recordings=recorded_dict,   # dict stored directly (as in TC row)
        expected_tools=["weather_tool"],
        expected_keywords=["question"],
        expected_intent=None,
        token_budget=None,
        session_history=None,
        human_score=None,
    )


class _MockHighFuzzyLLMProvider:
    """LLM provider for high-fuzzy tests.

    First call: returns candidate JSON (for generate_candidates).
    Subsequent calls (AgentRunner): returns a response that triggers the weather_tool,
    so that the sandbox path (fuzzy hit) is exercised.
    """

    def __init__(self):
        # Queue of responses: first is candidate JSON, then tool-calling responses
        self._call_count = 0

    def get_default_model(self) -> str:
        return "test-model"

    async def chat_with_retry(self, messages, tools=None, model=None, **kwargs):
        from nanoresearch.eval.sandbox import SandboxedToolRegistry  # noqa
        self._call_count += 1
        if self._call_count == 1:
            # Candidate generation call
            content = '[{"prompt": "Retrieve weather data for any city", "rationale": "improved"}]'
        else:
            # AgentRunner scoring call — return tool call for weather_tool with "Paris"
            # to trigger the fuzzy match path (recorded has "  Paris  ", agent calls "Paris")
            import json as _json
            tool_call = {
                "id": f"call_{self._call_count}",
                "type": "function",
                "function": {"name": "weather_tool", "arguments": _json.dumps({"location": "Paris"})},
            }

            class _ToolResponse:
                content = ""
                finish_reason = "tool_calls"
                usage = {}
                reasoning_content = None
                thinking_blocks = None

                @property
                def has_tool_calls(self):
                    return True

                @property
                def tool_calls(self):
                    return [type("TC", (), {
                        "id": tool_call["id"],
                        "function": type("F", (), {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        })(),
                    })()]

            if self._call_count % 2 == 0:
                return _ToolResponse()
            else:
                # After tool call, return final stop response
                class _StopResponse:
                    content = "The weather in Paris: cloudy. Does this answer your question?"
                    finish_reason = "stop"
                    usage = {}
                    reasoning_content = None
                    thinking_blocks = None
                    tool_calls = []

                    @property
                    def has_tool_calls(self):
                        return False

                return _StopResponse()


class TestFuzzyMatchSignalUnreliable:
    """Tests for the fuzzy-match ratio gate (B2-Fuzzy): proposal gets status=signal_unreliable
    when sandbox fuzzy_match_ratio > _FUZZY_UNRELIABLE_THRESHOLD (0.30)."""

    @pytest.mark.asyncio
    async def test_high_fuzzy_ratio_marks_signal_unreliable(
        self, in_memory_repo, seeded_tool_registry
    ):
        """When sandbox fuzzy_match_ratio > 0.30, proposal status = signal_unreliable.

        Uses _ScorePatternTarget with monkey-patched _score_candidate_set that
        also sets a high fuzzy_match_ratio on the sandbox, verifying the status
        propagates to the proposal.
        """
        from nanoresearch.eval.optimizer import OptimizationAgent, _FUZZY_UNRELIABLE_THRESHOLD
        from nanoresearch.eval.score_sample import ScoreSample
        import types as _types

        provider = _MockHighFuzzyLLMProvider()
        agent = OptimizationAgent(
            provider=provider,
            repo=in_memory_repo,
            registry=seeded_tool_registry,
        )
        target = _ScorePatternTarget()
        target.set_score_pattern(
            baseline_mean=0.5, baseline_std=0.0,
            candidate_mean=0.5, candidate_std=0.0,
        )
        # Store baseline text early so patch can detect it
        target._baseline_text = "original weather tool desc"

        # Patch _score_candidate_set to return a high fuzzy ratio along with scores
        async def _fake_score_set_high_fuzzy(self_inner, tgt, candidate, test_cases, recordings_map):
            result = {}
            for tc in test_cases:
                result[str(tc.id)] = ScoreSample.from_observations([0.5, 0.5, 0.5])
            # Return tuple: (scores, fuzzy_ratio)
            fuzzy_ratio = 0.80  # well above 0.30 threshold
            return result, fuzzy_ratio

        agent._score_candidate_set = _types.MethodType(_fake_score_set_high_fuzzy, agent)

        fix_cases = [_make_case_with_metadata("tool_schema_correctness")]
        health_cases = [_make_case_with_metadata("general_health")]

        proposal = await agent.generate_proposals(
            target=target,
            representative_snapshots=[],
            fix_test_cases=fix_cases,
            health_test_cases=health_cases,
        )
        assert proposal.status == "signal_unreliable", (
            f"Expected signal_unreliable when fuzzy_ratio > {_FUZZY_UNRELIABLE_THRESHOLD}, "
            f"got {proposal.status!r}"
        )
        for c in proposal.proposals:
            assert "fuzzy_match_ratio" in c, f"fuzzy_match_ratio missing from candidate: {c.keys()}"
            assert c["fuzzy_match_ratio"] > 0.30, (
                f"Expected fuzzy_match_ratio > 0.30, got {c['fuzzy_match_ratio']}"
            )

    @pytest.mark.asyncio
    async def test_low_fuzzy_ratio_does_not_mark_unreliable(
        self, in_memory_repo, seeded_tool_registry
    ):
        """When fuzzy_match_ratio <= 0.30, proposal status must NOT be signal_unreliable."""
        from nanoresearch.eval.optimizer import OptimizationAgent
        from nanoresearch.eval.score_sample import ScoreSample
        import types as _types

        provider = _MockHighFuzzyLLMProvider()
        agent = OptimizationAgent(
            provider=provider,
            repo=in_memory_repo,
            registry=seeded_tool_registry,
        )
        target = _ScorePatternTarget()
        target.set_score_pattern(
            baseline_mean=0.5, baseline_std=0.0,
            candidate_mean=0.5, candidate_std=0.0,
        )
        target._baseline_text = "original weather tool desc"

        # Patch _score_candidate_set to return a LOW fuzzy ratio
        async def _fake_score_set_low_fuzzy(self_inner, tgt, candidate, test_cases, recordings_map):
            result = {}
            for tc in test_cases:
                result[str(tc.id)] = ScoreSample.from_observations([0.5, 0.5, 0.5])
            fuzzy_ratio = 0.10  # below 0.30 threshold
            return result, fuzzy_ratio

        agent._score_candidate_set = _types.MethodType(_fake_score_set_low_fuzzy, agent)

        fix_cases = [_make_case_with_metadata("tool_schema_correctness")]
        health_cases = [_make_case_with_metadata("general_health")]

        proposal = await agent.generate_proposals(
            target=target,
            representative_snapshots=[],
            fix_test_cases=fix_cases,
            health_test_cases=health_cases,
        )
        assert proposal.status != "signal_unreliable", (
            f"Expected NOT signal_unreliable when fuzzy_ratio <= 0.30, got {proposal.status!r}"
        )
        for c in proposal.proposals:
            assert c.get("fuzzy_match_ratio", 0.0) <= 0.30


# ---------------------------------------------------------------------------
# Pre-step: ORM ghost-field regression test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_test_case_tool_recordings_round_trips_through_orm():
    """Regression: tool_recordings was a ghost field on AgentTestCase, silently
    returning None and killing _gather_recordings's case-level priority path."""
    from sqlalchemy import inspect
    from nanoresearch.storage.models import AgentTestCase
    cols = {c.key for c in inspect(AgentTestCase).columns}
    assert "tool_recordings" in cols, "AgentTestCase.tool_recordings must be ORM-declared, not ghost"
