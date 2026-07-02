"""P3.2 (C3): select_recent_window is a minimal recent slice with a cap ceiling, not greedy fill."""
from nanoresearch.agent.memory_facts import select_recent_window


def _seg(turn_end):
    return {"turn_end": turn_end, "text": "x" * 10}


def test_takes_only_newest_min_segment_not_greedy_fill():
    segs = [_seg(0), _seg(4), _seg(9), _seg(14), _seg(19)]
    near, far = select_recent_window(segs, cap_tokens=10_000, est_fn=lambda s: 100, min_segments=1)
    assert [s["turn_end"] for s in near] == [19]            # minimal — NOT filled toward cap
    assert [s["turn_end"] for s in far] == [0, 4, 9, 14]


def test_cap_is_ceiling_trims_oldest_of_near():
    segs = [_seg(0), _seg(9), _seg(19)]
    near, far = select_recent_window(segs, cap_tokens=150, est_fn=lambda s: 100, min_segments=3)
    assert [s["turn_end"] for s in near] == [19]            # 3*100 > 150 → trimmed to newest 1
    assert [s["turn_end"] for s in far] == [0, 9]


def test_min_segments_multiple_within_cap_ascending():
    segs = [_seg(0), _seg(9), _seg(19)]
    near, far = select_recent_window(segs, cap_tokens=1000, est_fn=lambda s: 100, min_segments=2)
    assert [s["turn_end"] for s in near] == [9, 19]         # turn-ascending
    assert [s["turn_end"] for s in far] == [0]


def test_empty():
    assert select_recent_window([], cap_tokens=100, est_fn=lambda s: 1) == ([], [])
