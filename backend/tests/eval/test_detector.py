"""Tests for BadcaseDetector and RuleEvaluator."""

from __future__ import annotations

import pytest

from nanoresearch.eval.badcase_detector import BadcaseDetector
from nanoresearch.eval.evaluator import RuleEvaluator
from nanoresearch.eval.snapshot import RunSnapshotData


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


def _tc(**overrides):
    """Minimal test case stand-in."""
    d = dict(
        dataset_type="golden",
        token_budget=None, expected_tools=None, expected_keywords=None,
    )
    d.update(overrides)
    return type("TC", (), d)()


# ---------------------------------------------------------------------------
# BadcaseDetector — detect() returns list[tuple[str, str]]
# ---------------------------------------------------------------------------


class TestBadcaseDetector:
    def test_detects_run_failure(self):
        detector = BadcaseDetector()
        snap = _snap(run_status="failed")
        result = detector.detect(snap)
        assert len(result) == 1
        trigger, cat = result[0]
        assert cat == "run_failure"
        assert trigger.startswith("rule:")

    def test_detects_max_iterations(self):
        detector = BadcaseDetector()
        snap = _snap(run_status="max_iterations")
        result = detector.detect(snap)
        assert result[0][1] == "run_failure"

    def test_detects_timeout(self):
        detector = BadcaseDetector()
        snap = _snap(run_status="timeout")
        result = detector.detect(snap)
        assert result[0][1] == "run_failure"

    def test_skips_success(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap)
        assert result == []

    def test_token_spike_when_p95_set(self):
        detector = BadcaseDetector(p95_tokens=50)
        snap = _snap(run_status="success", total_input_tokens=100, total_output_tokens=100)
        result = detector.detect(snap)
        assert result[0][1] == "token_spike"

    def test_token_spike_skipped_when_p95_none(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success", total_input_tokens=100, total_output_tokens=100)
        result = detector.detect(snap)
        assert result == []

    def test_detects_excessive_retries(self):
        detector = BadcaseDetector(p95_tokens=None, max_retries=3)
        snap = _snap(run_status="success", retry_count=5)
        result = detector.detect(snap)
        assert result[0][1] == "excessive_retries"

    def test_skips_normal_retries(self):
        detector = BadcaseDetector(p95_tokens=None, max_retries=3)
        snap = _snap(run_status="success", retry_count=1)
        result = detector.detect(snap)
        assert result == []

    def test_low_quality_triggers_when_passed_false(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap, scores={"keyword_coverage": 0.3}, passed=False)
        assert len(result) >= 1
        assert result[0][1] == "low_quality"

    def test_low_quality_skipped_when_passed_true(self):
        # Quality badcase should not fire when the case passed.
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap, scores={"keyword_coverage": 0.3}, passed=True)
        assert all(cat != "low_quality" for _, cat in result)

    def test_low_quality_skipped_when_no_scores(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap, scores=None, passed=False)
        assert result == []

    def test_low_quality_includes_dim_names(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(snap, scores={"keyword_coverage": 0.3, "relevance": 0.5}, passed=False)
        assert len(result) >= 1
        assert "keyword_coverage" in result[0][0]

    def test_tool_skip_triggers_independently(self):
        # tool_skip fires even when the case passes quality check.
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        tc = _tc(expected_tools=["research"])
        result = detector.detect(snap, scores={"tool_skip": 0.0, "keyword_coverage": 0.9}, passed=True, tc=tc)
        cats = [cat for _, cat in result]
        assert "tool_skip" in cats

    def test_tool_skip_not_triggered_when_tool_called(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        tc = _tc(expected_tools=["research"])
        result = detector.detect(snap, scores={"tool_skip": 1.0, "keyword_coverage": 0.9}, passed=True, tc=tc)
        cats = [cat for _, cat in result]
        assert "tool_skip" not in cats

    def test_tool_skip_not_triggered_when_no_expected_tools(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        tc = _tc(expected_tools=None)
        result = detector.detect(snap, scores={"keyword_coverage": 0.9}, passed=True, tc=tc)
        assert result == []

    def test_both_quality_and_tool_skip_can_fire(self):
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        tc = _tc(expected_tools=["research"])
        result = detector.detect(
            snap,
            scores={"tool_skip": 0.0, "keyword_coverage": 0.3},
            passed=False,
            tc=tc,
        )
        cats = [cat for _, cat in result]
        assert "low_quality" in cats
        assert "tool_skip" in cats
        # Quality must come first.
        assert result[0][1] == "low_quality"

    def test_run_failure_takes_priority(self):
        detector = BadcaseDetector()
        snap = _snap(run_status="failed")
        result = detector.detect(snap, scores={"keyword_coverage": 0.3}, passed=False)
        assert len(result) == 1
        assert result[0][1] == "run_failure"

    def test_token_budget_and_tool_skip_excluded_from_quality(self):
        # token_budget=0 and tool_skip=0 must not trigger low_quality on their own.
        detector = BadcaseDetector(p95_tokens=None)
        snap = _snap(run_status="success")
        result = detector.detect(
            snap,
            scores={"token_budget": 0.0, "tool_skip": 0.0},
            passed=False,
        )
        cats = [cat for _, cat in result]
        assert "low_quality" not in cats


# ---------------------------------------------------------------------------
# RuleEvaluator
# ---------------------------------------------------------------------------


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

    def test_tool_skip_partial_match(self):
        snap = _snap(tool_call_chain=[
            {"name": "search", "params": {}, "result": "", "order": 1, "duration_ms": 10, "error": False},
        ])
        tc = _tc(expected_tools=["search", "summarize"])
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["tool_skip"] == 0.7

    def test_tool_skip_full_match(self):
        snap = _snap(tool_call_chain=[
            {"name": "search", "params": {}, "result": "", "order": 1, "duration_ms": 10, "error": False},
        ])
        tc = _tc(expected_tools=["search"])
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["tool_skip"] == 1.0

    def test_tool_skip_no_match(self):
        snap = _snap(tool_call_chain=[])
        tc = _tc(expected_tools=["search"])
        scores = RuleEvaluator().evaluate(snap, tc)
        assert scores["tool_skip"] == 0.0

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

    def test_keyword_bilingual_fallback(self):
        snap = _snap(final_response="这里用了三线性插值方法")
        tc = _tc(expected_keywords=["trilinear interpolation"])
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

    def test_is_passed_keyword_only_gate(self):
        evaluator = RuleEvaluator()
        # High keyword coverage passes even with tool_skip=0.
        tc = _tc(expected_keywords=["a", "b", "c", "d", "e"])  # >4, threshold=0.6
        passed, failed = evaluator.is_passed({"keyword_coverage": 0.8, "tool_skip": 0.0}, tc)
        assert passed is True

    def test_is_passed_fails_on_low_keywords(self):
        evaluator = RuleEvaluator()
        tc = _tc(expected_keywords=["a", "b", "c"])  # ≤4, threshold=0.8
        passed, failed = evaluator.is_passed({"keyword_coverage": 0.5}, tc)
        assert passed is False
        assert "keyword_coverage" in failed

    def test_is_passed_tool_skip_in_failed_dims_but_not_gate(self):
        evaluator = RuleEvaluator()
        tc = _tc(expected_keywords=["a", "b", "c", "d", "e"], expected_tools=["research"])
        passed, failed = evaluator.is_passed({"keyword_coverage": 0.8, "tool_skip": 0.0}, tc)
        assert passed is True          # high keyword coverage → pass
        assert "tool_skip" in failed   # but tool_skip still noted as behaviour signal

    def test_is_passed_no_keywords_defaults_pass(self):
        evaluator = RuleEvaluator()
        tc = _tc(expected_keywords=None, expected_tools=None)
        passed, failed = evaluator.is_passed({}, tc)
        assert passed is True
        assert failed == []

    def test_is_passed_token_budget_excluded(self):
        evaluator = RuleEvaluator()
        # token_budget=0 must NOT cause failure on its own.
        tc = _tc(expected_keywords=["a", "b", "c", "d", "e"])
        passed, failed = evaluator.is_passed({"token_budget": 0.0, "keyword_coverage": 0.8}, tc)
        assert passed is True
        assert "token_budget" not in failed
