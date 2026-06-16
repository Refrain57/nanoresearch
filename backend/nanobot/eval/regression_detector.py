"""Compare two eval run score dicts and detect regressions per dimension."""

from __future__ import annotations

DEFAULT_THRESHOLDS: dict[str, float] = {
    "tool_hit_rate": -0.05,
    "extra_tool_calls": -0.05,
    "keyword_coverage": -0.08,
    "tool_rationality": -0.10,
    "task_completion": -0.10,
    "response_logic": -0.10,
    "multi_turn_coherence": -0.10,
}

# Hard-gate dimensions that are 0/1 values — not meaningful for delta comparison
SKIP_DIMS: frozenset[str] = frozenset({"token_budget"})


class RegressionDetector:
    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self._thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def compare(
        self,
        baseline_scores: dict[str, float],
        current_scores: dict[str, float],
    ) -> tuple[bool, dict]:
        """Compare scores against a baseline.

        Returns (has_regression, diffs) where diffs maps each dimension to
        {baseline, current, delta, threshold, regressed}.
        """
        dims = set(baseline_scores) | set(current_scores)
        diffs: dict[str, dict] = {}
        has_regression = False

        for dim in dims:
            if dim in SKIP_DIMS:
                continue
            baseline = baseline_scores.get(dim)
            current = current_scores.get(dim)
            if baseline is None or current is None:
                continue
            delta = current - baseline
            threshold = self._thresholds.get(dim, -0.05)
            regressed = delta < threshold
            if regressed:
                has_regression = True
            diffs[dim] = {
                "baseline": round(baseline, 4),
                "current": round(current, 4),
                "delta": round(delta, 4),
                "threshold": threshold,
                "regressed": regressed,
            }

        return has_regression, diffs
