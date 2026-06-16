"""Rule-based evaluator for agent run snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.eval.snapshot import RunSnapshotData
    from nanobot.storage.models import AgentTestCase


class RuleEvaluator:
    """Scores a run against a test case using deterministic rules.

    Only dimensions with expected values set are scored — missing expectations
    are skipped entirely, never recorded as 0.
    """

    PASS_THRESHOLD = 0.6

    def evaluate(
        self,
        snapshot: "RunSnapshotData",
        test_case: "AgentTestCase",
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        # Hard gate: token budget
        if test_case.token_budget is not None:
            total = snapshot.total_input_tokens + snapshot.total_output_tokens
            scores["token_budget"] = 1.0 if total <= test_case.token_budget else 0.0

        # Tool call hit rate + extra tool penalty
        if test_case.expected_tools:
            actual_names = {c["name"] for c in snapshot.tool_call_chain}
            expected = set(test_case.expected_tools)
            hit = len(actual_names & expected) / max(len(expected), 1)
            scores["tool_hit_rate"] = round(hit, 4)
            extra = len(actual_names - expected)
            scores["extra_tool_calls"] = max(0.0, round(1.0 - extra * 0.2, 4))

        # Keyword coverage (case-insensitive)
        if test_case.expected_keywords:
            resp_lower = (snapshot.final_response or "").lower()
            covered = sum(1 for kw in test_case.expected_keywords if kw.lower() in resp_lower)
            scores["keyword_coverage"] = round(covered / max(len(test_case.expected_keywords), 1), 4)

        return scores

    def is_passed(self, scores: dict[str, float]) -> tuple[bool, list[str]]:
        """Return (passed, list_of_failed_dimensions)."""
        # token_budget is a hard gate: any score of 0.0 on it = immediate fail
        if scores.get("token_budget", 1.0) == 0.0:
            failed = [d for d, s in scores.items() if s < self.PASS_THRESHOLD]
            return False, failed
        failed = [d for d, s in scores.items() if s < self.PASS_THRESHOLD]
        return len(failed) == 0, failed
