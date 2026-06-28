"""C2: startup consolidation planning — idle gate, turn counting, tail protect."""
from __future__ import annotations

from datetime import timedelta

from nanoresearch.agent.memory import plan_startup_consolidation
from nanoresearch.session.manager import Session
from nanoresearch.utils.helpers import utcnow_aware

IDLE = timedelta(minutes=30)


def _msgs(roles: list[str]) -> list[dict]:
    return [{"role": r, "content": f"{r}-{i}"} for i, r in enumerate(roles)]


def _fake_pick(end_idx: int | None):
    """Return a pick_boundary stub that yields (end_idx, 0) or None."""
    def _pick(session, tokens_to_remove, tail_protect):  # noqa: ARG001
        return None if end_idx is None else (end_idx, 0)
    return _pick


def test_skips_when_session_active_within_idle_window():
    """Back-to-back turn (5 min idle) must NOT consolidate — kills frequent compaction."""
    session = Session(key="web:1", messages=_msgs(["user", "assistant"] * 4),
                      last_consolidated=0, updated_at=utcnow_aware() - timedelta(minutes=5))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(2))
    assert result is None


def test_counts_turns_not_message_rows():
    """One tool-using turn = 8 rows but 1 user message → below min_turns=2 → skip.

    This reproduces problem 1: row-count fired on every turn; turn-count must not."""
    rows = _msgs(["user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"])
    session = Session(key="web:2", messages=rows, last_consolidated=0,
                      updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(4))
    assert result is None


def test_consolidates_when_idle_and_enough_turns():
    session = Session(key="web:3", messages=_msgs(["user", "assistant"] * 6),
                      last_consolidated=0, updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(4))
    assert result == (0, 4)


def test_returns_none_when_boundary_not_found():
    session = Session(key="web:4", messages=_msgs(["user", "assistant"] * 6),
                      last_consolidated=0, updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(None))
    assert result is None


def test_returns_none_when_boundary_at_or_before_start():
    session = Session(key="web:5", messages=_msgs(["user", "assistant"] * 6),
                      last_consolidated=4, updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(4))
    assert result is None


def test_real_boundary_picker_protects_tail():
    """End-to-end with the real pick_consolidation_boundary: tail is protected."""
    from nanoresearch.agent.memory import MemoryConsolidator
    consolidator = MemoryConsolidator.__new__(MemoryConsolidator)  # no heavy init needed
    rows = _msgs(["user", "assistant"] * 6)  # 12 rows
    session = Session(key="web:6", messages=rows, last_consolidated=0,
                      updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=consolidator.pick_consolidation_boundary)
    assert result is not None
    start, end_idx = result
    assert start == 0
    assert end_idx <= len(rows) - 8  # tail of 8 preserved
