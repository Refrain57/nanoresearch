"""Automatic badcase detection based on run snapshot heuristics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.eval.snapshot import RunSnapshotData

# Category labels written to agent_run_snapshots.badcase_category
CATEGORY_RUN_FAILURE = "run_failure"
CATEGORY_TOKEN_SPIKE = "token_spike"
CATEGORY_EXCESSIVE_RETRIES = "excessive_retries"
CATEGORY_LOW_QUALITY = "low_quality"
CATEGORY_TOOL_SKIP = "tool_skip"

# Dimensions excluded from quality checks — informational or behaviour signals only.
_NON_QUALITY_DIMS = frozenset({"token_budget", "tool_skip", "contextual_recall"})


CATEGORY_RETRIEVAL_FAILURE = "retrieval_failure"
CATEGORY_HALLUCINATION = "hallucination"
CATEGORY_REASONING_FAILURE = "reasoning_failure"


def _refine_failure_category(scores: dict[str, float]) -> str:
    """Decompose low_quality into a more specific failure mode when possible.

    Priority order: retrieval_failure > hallucination > reasoning_failure > low_quality.
    reasoning_failure requires both contextual_recall and task_completion (Judge) to be
    present; falls back to low_quality when Judge was not run.
    """
    contextual_recall = scores.get("contextual_recall")
    faithfulness = scores.get("faithfulness_score")
    task_completion = scores.get("task_completion")

    if contextual_recall is not None and contextual_recall < 0.3:
        return CATEGORY_RETRIEVAL_FAILURE

    if faithfulness is not None and faithfulness < 0.4:
        return CATEGORY_HALLUCINATION

    if (
        contextual_recall is not None
        and task_completion is not None
        and contextual_recall >= 0.5
        and task_completion < 0.5
    ):
        return CATEGORY_REASONING_FAILURE

    return CATEGORY_LOW_QUALITY


class BadcaseDetector:
    """Checks a completed run snapshot for known badcase patterns.

    p95_tokens=None disables the token_spike check (recommended for the first
    week while baseline data is being collected).

    detect() returns a list of (trigger_source, category) pairs — can be empty,
    or contain one or two items when quality failure and tool_skip both apply.
    Quality failures come first in the list.
    """

    def __init__(
        self,
        p95_tokens: int | None = None,
        max_retries: int = 3,
    ) -> None:
        self.p95_tokens = p95_tokens
        self.max_retries = max_retries

    def detect(
        self,
        snapshot: "RunSnapshotData",
        scores: dict[str, float] | None = None,
        passed: bool | None = None,
        tc: Any = None,
    ) -> list[tuple[str, str]]:
        """Return list of (trigger_source, category) badcase hits, or [] if clean."""
        results: list[tuple[str, str]] = []

        # Hard failures — short-circuit, no quality check needed.
        if snapshot.run_status in ("failed", "timeout", "max_iterations"):
            return [(f"rule:{CATEGORY_RUN_FAILURE}", CATEGORY_RUN_FAILURE)]

        if self.p95_tokens is not None:
            total = snapshot.total_input_tokens + snapshot.total_output_tokens
            if total > self.p95_tokens:
                return [(f"rule:{CATEGORY_TOKEN_SPIKE}", CATEGORY_TOKEN_SPIKE)]

        if snapshot.retry_count > self.max_retries:
            return [(f"rule:{CATEGORY_EXCESSIVE_RETRIES}", CATEGORY_EXCESSIVE_RETRIES)]

        # Line 1 — quality badcase: fires only when the case actually failed.
        if passed is False and scores:
            quality_failed = [
                d for d, s in scores.items()
                if d not in _NON_QUALITY_DIMS and s < 0.6
            ]
            if quality_failed:
                category = _refine_failure_category(scores)
                dims = ",".join(quality_failed[:3])
                results.append((f"rule:{category}:{dims}", category))

        # Line 2 — behaviour badcase: fires independently of pass/fail.
        # Persisted via failed_dimensions JSONB, mark_badcase only called when
        # quality did not already fire (to avoid clobbering the primary category).
        if (
            scores is not None
            and tc is not None
            and getattr(tc, "expected_tools", None)
            and scores.get("tool_skip", 1.0) == 0.0
        ):
            results.append((f"rule:{CATEGORY_TOOL_SKIP}", CATEGORY_TOOL_SKIP))

        return results
