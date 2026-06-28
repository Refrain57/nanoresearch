"""Regression test for Session ↔ Redis marshalling round-trip.

Bug: `_redis_save` stored a sliced window `messages[last_consolidated:]` but
also stored the original `last_consolidated` offset in meta. `_redis_load`
then reconstructed a Session whose `messages` was already the post-offset
tail, but whose `last_consolidated` was the original offset. The next
`get_history()` call applies the offset a second time, silently dropping
all history when `len(window) <= last_consolidated`.

This file has two layers:
- contract simulations (no Redis needed) document the save/load semantics
- a real-function test uses an in-memory fake Redis to exercise
  `SessionManager._redis_save` and `_redis_load` directly so the fix
  must land in manager.py for it to pass.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from nanoresearch.session.manager import Session, SessionManager


def _simulate_redis_save(session: Session) -> tuple[list[dict], int]:
    """Mirror `_redis_save` marshalling exactly (manager.py:179-189)."""
    window = session.messages[session.last_consolidated :]
    meta_last_consolidated = session.last_consolidated
    return window, meta_last_consolidated


def _simulate_redis_load_buggy(
    key: str, window: list[dict], meta_last_consolidated: int
) -> Session:
    """Mirror the BUGGY `_redis_load` reconstruction (pre-fix)."""
    return Session(
        key=key,
        messages=window,
        last_consolidated=meta_last_consolidated,
    )


def _simulate_redis_load_fixed(
    key: str, window: list[dict], meta_last_consolidated: int  # noqa: ARG001
) -> Session:
    """Mirror the FIXED `_redis_load` reconstruction.

    The stored window is itself the post-consolidation tail, so its
    offset within the Session is 0 regardless of the meta value (which
    referenced the original, pre-slice messages array).
    """
    return Session(
        key=key,
        messages=window,
        last_consolidated=0,
    )


def _make_session(num_messages: int, last_consolidated: int) -> Session:
    return Session(
        key="test:roundtrip",
        messages=[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(num_messages)
        ],
        last_consolidated=last_consolidated,
    )


def test_baseline_get_history_returns_tail_after_consolidation():
    """Pre-Redis: get_history(0) returns messages past last_consolidated."""
    session = _make_session(num_messages=8, last_consolidated=4)
    history = session.get_history(max_messages=0)
    assert len(history) == 4
    assert [m["content"] for m in history] == ["m4", "m5", "m6", "m7"]


def test_redis_roundtrip_preserves_history_after_fix():
    """After save → load → get_history, the unconsolidated tail survives.

    This test demonstrates the F1 fix. With the buggy loader, the
    reconstructed Session.get_history(0) returns [] because
    `window[last_consolidated:]` slices a second time.
    """
    session = _make_session(num_messages=8, last_consolidated=4)
    baseline = session.get_history(max_messages=0)
    assert len(baseline) == 4, "baseline sanity check"

    window, meta_lc = _simulate_redis_save(session)

    loaded = _simulate_redis_load_fixed("test:roundtrip", window, meta_lc)
    roundtrip_history = loaded.get_history(max_messages=0)

    assert len(roundtrip_history) == len(baseline), (
        f"Redis round-trip dropped history: "
        f"baseline={len(baseline)}, roundtrip={len(roundtrip_history)}"
    )
    assert roundtrip_history == baseline


def test_buggy_loader_reproduces_history_loss():
    """Documents the pre-fix bug: buggy loader returns empty history."""
    session = _make_session(num_messages=8, last_consolidated=4)
    window, meta_lc = _simulate_redis_save(session)

    loaded_buggy = _simulate_redis_load_buggy("test:roundtrip", window, meta_lc)
    history = loaded_buggy.get_history(max_messages=0)

    assert history == [], (
        "If this fails, the bug semantics changed; either the fix landed "
        "or the round-trip contract was altered."
    )


def test_redis_roundtrip_after_subsequent_save():
    """Save → load → get_history → save again should still preserve history.

    This catches a regression where the load resets last_consolidated=0
    but the next save would slice an empty window because last_consolidated
    accidentally got re-inflated.
    """
    session = _make_session(num_messages=8, last_consolidated=4)
    baseline = session.get_history(max_messages=0)

    # First round-trip
    window1, meta1 = _simulate_redis_save(session)
    loaded1 = _simulate_redis_load_fixed("test:roundtrip", window1, meta1)
    assert loaded1.get_history(max_messages=0) == baseline

    # Append two new messages, then save again
    loaded1.messages.append({"role": "user", "content": "m8"})
    loaded1.messages.append({"role": "assistant", "content": "m9"})

    # Second round-trip
    window2, meta2 = _simulate_redis_save(loaded1)
    loaded2 = _simulate_redis_load_fixed("test:roundtrip", window2, meta2)
    history2 = loaded2.get_history(max_messages=0)

    expected = baseline + [
        {"role": "user", "content": "m8"},
        {"role": "assistant", "content": "m9"},
    ]
    assert history2 == expected, (
        f"Second round-trip lost history: got {history2}, expected {expected}"
    )


class _FakeRedisPipeline:
    def __init__(self, store: "_FakeRedis") -> None:
        self._store = store
        self._ops: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def delete(self, key: str) -> "_FakeRedisPipeline":
        self._ops.append(("delete", key))
        return self

    def rpush(self, key: str, *items: str) -> "_FakeRedisPipeline":
        self._ops.append(("rpush", key, items))
        return self

    def hset(self, key: str, mapping: dict) -> "_FakeRedisPipeline":
        self._ops.append(("hset", key, mapping))
        return self

    def expire(self, key: str, ttl: int) -> "_FakeRedisPipeline":
        self._ops.append(("expire", key, ttl))
        return self

    async def execute(self) -> list:
        for op in self._ops:
            if op[0] == "delete":
                self._store.lists.pop(op[1], None)
                self._store.hashes.pop(op[1], None)
            elif op[0] == "rpush":
                self._store.lists[op[1]].extend(op[2])
            elif op[0] == "hset":
                self._store.hashes[op[1]].update(op[2])
            elif op[0] == "expire":
                pass  # no-op for tests
        self._ops.clear()
        return []


class _FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list] = defaultdict(list)
        self.hashes: dict[str, dict] = defaultdict(dict)

    def pipeline(self, transaction: bool = True) -> _FakeRedisPipeline:  # noqa: ARG002
        return _FakeRedisPipeline(self)

    async def hgetall(self, key: str) -> dict:
        return dict(self.hashes.get(key, {}))

    async def lrange(self, key: str, start: int, end: int) -> list:
        data = self.lists.get(key, [])
        if end == -1:
            return list(data[start:])
        return list(data[start : end + 1])


async def test_real_redis_save_load_preserves_history(tmp_path: Path) -> None:
    """End-to-end: real _redis_save → real _redis_load preserves get_history(0).

    This test must pass after the F1 fix lands in manager.py. Before the
    fix it fails because the load reconstructs a Session whose
    `last_consolidated` no longer matches the truncated `messages`.
    """
    fake = _FakeRedis()
    manager = SessionManager(workspace=tmp_path)

    # Real-world Turn 4 setup: 8 msgs were consolidated (Turn 1),
    # then Turn 2 added 4 more
    session = Session(
        key="web:abcd-1234",
        messages=[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(12)
        ],
        last_consolidated=8,
    )
    baseline_history = session.get_history(max_messages=0)
    assert len(baseline_history) == 4, "baseline: Turn 2 produced 4 unconsolidated"

    with patch("nanoresearch.bus.redis_client.get_redis", return_value=fake):
        await manager._redis_save(session)
        # Force load from fake Redis (bypass any L1 caching)
        loaded = await manager._redis_load("web:abcd-1234")

    assert loaded is not None, "Redis must return a Session after save"
    loaded_history = loaded.get_history(max_messages=0)

    assert loaded_history == baseline_history, (
        f"Round-trip lost messages: "
        f"baseline={[m['content'] for m in baseline_history]}, "
        f"roundtrip={[m['content'] for m in loaded_history]}"
    )


def test_redis_roundtrip_with_full_consolidation_then_new_messages():
    """Real-world Turn 4 scenario: Turn 1 (8 msgs) consolidated, Turn 2/3 added."""
    # Turn 1 finished: 8 messages, none consolidated yet
    session = _make_session(num_messages=8, last_consolidated=0)

    # Turn 2 startup: all 8 consolidated → last_consolidated = 8
    session.last_consolidated = 8
    # Turn 2 added 4 messages (user, asst+tool_calls, tool, asst)
    for i in range(8, 12):
        session.messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})

    # End of Turn 2 baseline
    baseline = session.get_history(max_messages=0)
    assert len(baseline) == 4, "Turn 2 added 4 messages to unconsolidated tail"

    # Save to Redis end of Turn 2
    window, meta_lc = _simulate_redis_save(session)
    assert len(window) == 4
    assert meta_lc == 8

    # Turn 3 startup: load from Redis
    loaded = _simulate_redis_load_fixed("test:roundtrip", window, meta_lc)
    history_at_turn3_start = loaded.get_history(max_messages=0)

    # Without fix: history_at_turn3_start == [] (window[8:] = empty)
    # With fix: history_at_turn3_start == Turn 2's 4 messages
    assert history_at_turn3_start == baseline, (
        "Turn 3 startup must see Turn 2's history; this was the production bug"
    )
