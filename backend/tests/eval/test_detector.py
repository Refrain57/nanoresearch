"""Tests for BadcaseDetector and RuleEvaluator."""

from __future__ import annotations

import pytest

from nanobot.eval.badcase_detector import BadcaseDetector
from nanobot.eval.evaluator import RuleEvaluator
from nanobot.eval.snapshot import RunSnapshotData


def _snap(**overrides) -> RunSnapshotData:
    d = dict(
        run_id="r1", user_input="test",
        tool_call_chain=[], llm_calls=[],
        final_response="ok", run_status="success",
        total_input_tokens=10, total_output_tokens=20,
        ttft_ms=None, total_duration_ms=100,
        tool_call_count=0, llm_call_count=1, retry_count=0,
    )
    d.update(overrides)
    return RunSnapshotData(**d)


# ---------------------------------------------------------------------------
# BadcaseDetector
# ---------------------------------------------------------------------------


class TestBadcaseDetector:
    def test_detects_run_failure(self):
        detector = BadcaseDetector()
        snap = _snap(run_status="failed")
        result = detector.detect(snap)
        assert result is not None
        trigger, cat = result
        assert cat == "run_failure"
        assert trigger.startswith("rule:")

    def test_detects_max_iterations(self):
        detector = BadcaseDetector()
        snap = _snap(run_status="max_iterations")
        result = detector.detect(snap)
        assert result is not None
        assert result[1] == "run_failure"

    def test_detects_timeout(self):
        detector = BadcaseDetector()
        snap = _snap(run_status="timeout")
        result = detector.detect(snap)
        assert result is not None
        assert result[1] == "run_failure"

    def test_skips_success(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap)
        assert result is None

    def test_token_spike_when_p95_set(self):
        detector = BadcaseDetector(p95_tokens=50)
        snap = _snap(run_status="success", total_input_tokens=100, total_output_tokens=100)
        result = detector.detect(snap)
        assert result is not None
        assert result[1] == "token_spike"

    def test_token_spike_skipped_when_p95_none(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success", total_input_tokens=100, total_output_tokens=100)
        result = detector.detect(snap)
        assert result is None

    def test_detects_excessive_retries(self):
        detector = BadcaseDetector(p95_tokens=None, max_retries=3)
        snap = _snap(run_status="success", retry_count=5)
        result = detector.detect(snap)
        assert result is not None
        assert result[1] == "excessive_retries"

    def test_skips_normal_retries(self):
        detector = BadcaseDetector(p95_tokens=None, max_retries=3)
        snap = _snap(run_status="success", retry_count=1)
        result = detector.detect(snap)
        assert result is None

    def test_low_score_triggers(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap, scores={"accuracy": 0.3})
        assert result is not None
        assert result[1] == "low_score"

    def test_low_score_skipped_when_no_scores(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap, scores=None)
        assert result is None

    def test_low_score_includes_dim_names(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap, scores={"accuracy": 0.3, "relevance": 0.5})
        assert result is not None
        assert "accuracy" in result[0]

    def test_run_failure_takes_priority_over_low_score(self):
        detector = BadcaseDetector()
        snap = _snap(run_status="failed")
        result = detector.detect(snap, scores={"accuracy": 0.3})
        assert result is not None
        assert result[1] == "run_failure"


# ---------------------------------------------------------------------------
# RuleEvaluator
# ---------------------------------------------------------------------------


def _tc(**overrides):
    """Minimal test case stand-in."""
    d = dict(
        token_budget=None, expected_tools=None, expected_keywords=None,
    )
    d.update(overrides)
    return type("TC", (), d)()


class TestRuleEvaluator:
    def test_token_budget_pass(self):
        snap = _snap(total_input_tokens=10, total_output_tokens=20)
        tc = _tc(token_budget=100)
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["token_budget"] == 1.0

    def test_token_budget_fail(self):
        snap = _snap(total_input_tokens=100, total_output_tokens=200)
        tc = _tc(token_budget=100)
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["token_budget"] == 0.0

    def test_tool_hit_rate(self):
        snap = _snap(tool_call_chain=[
            {"name": "search", "params": {}, "result": "", "order": 1, "duration_ms": 10, "error": False},
            {"name": "read", "params": {}, "result": "", "order": 2, "duration_ms": 10, "error": False},
        ])
        tc = _tc(expected_tools=["search", "summarize"])
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["tool_hit_rate"] == 0.5

    def test_tool_hit_rate_full_match(self):
        snap = _snap(tool_call_chain=[
            {"name": "search", "params": {}, "result": "", "order": 1, "duration_ms": 10, "error": False},
        ])
        tc = _tc(expected_tools=["search"])
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["tool_hit_rate"] == 1.0

    def test_extra_tool_calls(self):
        snap = _snap(tool_call_chain=[
            {"name": "search", "params": {}, "result": "", "order": 1, "duration_ms": 10, "error": False},
            {"name": "delete", "params": {}, "result": "", "order": 2, "duration_ms": 10, "error": False},
        ])
        tc = _tc(expected_tools=["search"])
        scores = RuleEvaluator().evaluate(snap, tc)
        # 1 extra tool → 1 - 0.2 = 0.8
        assert scores["extra_tool_calls"] == 0.8

    def test_keyword_coverage(self):
        snap = _snap(final_response="the cat sat on the mat")
        tc = _tc(expected_keywords=["cat", "dog"])
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["keyword_coverage"] == 0.5

    def test_keyword_coverage_case_insensitive(self):
        snap = _snap(final_response="The Cat")
        tc = _tc(expected_keywords=["cat"])
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["keyword_coverage"] == 1.0

    def test_keyword_empty_response(self):
        snap = _snap(final_response=None)
        tc = _tc(expected_keywords=["cat"])
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["keyword_coverage"] == 0.0

    def test_skips_missing_expectations(self):
        snap = _snap()
        tc = _tc(token_budget=None, expected_tools=None, expected_keywords=None)
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores == {}

    def test_is_passed_all_pass(self):
        evaluator = RuleEvaluator()
        passed, failed = evaluator.is_passed({"dim1": 0.8, "dim2": 0.9})
        assert passed is True
        assert failed == []

    def test_is_passed_some_fail(self):
        evaluator = RuleEvaluator()
        passed, failed = evaluator.is_passed({"dim1": 0.8, "dim2": 0.3})
        assert passed is False
        assert "dim2" in failed

    def test_is_passed_token_budget_hard_gate(self):
        evaluator = RuleEvaluator()
        # token_budget=0.0 should fail even if other dims are high
        passed, failed = evaluator.is_passed({"token_budget": 0.0, "dim2": 0.9})
        assert passed is False

    def test_is_passed_empty_scores(self):
        evaluator = RuleEvaluator()
        passed, failed = evaluator.is_passed({})
        assert passed is True
        assert failed == []
