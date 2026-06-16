"""Automatic badcase detection based on run snapshot heuristics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.eval.snapshot import RunSnapshotData

# Category labels written to agent_run_snapshots.badcase_category
CATEGORY_RUN_FAILURE = "run_failure"
CATEGORY_TOKEN_SPIKE = "token_spike"
CATEGORY_EXCESSIVE_RETRIES = "excessive_retries"
CATEGORY_LOW_SCORE = "low_score"


class BadcaseDetector:
    """Checks a completed run snapshot for known badcase patterns.

    p95_tokens=None disables the token_spike check (recommended for the first
    week while baseline data is being collected).
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
    ) -> tuple[str, str] | None:
        """Return (trigger_source, category) or None if not a badcase."""
        if snapshot.run_status in ("failed", "timeout", "max_iterations"):
            return f"rule:{CATEGORY_RUN_FAILURE}", CATEGORY_RUN_FAILURE

        if self.p95_tokens is not None:
            total = snapshot.total_input_tokens + snapshot.total_output_tokens
            if total > self.p95_tokens:
                return f"rule:{CATEGORY_TOKEN_SPIKE}", CATEGORY_TOKEN_SPIKE

        if snapshot.retry_count > self.max_retries:
            return f"rule:{CATEGORY_EXCESSIVE_RETRIES}", CATEGORY_EXCESSIVE_RETRIES

        if scores:
            failed = [d for d, s in scores.items() if s < 0.6]
            if failed:
                dims = ",".join(failed[:3])
                return f"rule:{CATEGORY_LOW_SCORE}:{dims}", CATEGORY_LOW_SCORE

        return None
