"""Task 4: worker emits a citations SSE event via _make_on_citations.

Tests that _make_on_citations(redis, stream_key) returns an async callback
that writes {"type": "citations", "items": [...]} to the Redis stream via
xadd_event — same channel as tool_call events.

No live Redis required: a minimal _FakeRedis captures xadd calls (mirrors
the _FakeRedis approach in tests/unit/session/test_redis_roundtrip.py).

Test location rationale: backend/tests/ hosts all unit tests that exercise
backend-only modules (worker, storage, bus) without live infra.
"""
from __future__ import annotations

import json

import pytest


class _FakeRedis:
    """Minimal fake Redis that captures xadd calls for assertion."""

    def __init__(self):
        self._streams: dict[str, list[dict]] = {}

    async def xadd(self, key: str, fields: dict) -> None:
        event = json.loads(fields["data"])
        self._streams.setdefault(key, []).append(event)

    async def expire(self, key: str, ttl: int) -> None:
        pass  # no-op

    def xadds_for(self, key: str) -> list[dict]:
        return self._streams.get(key, [])


@pytest.mark.asyncio
async def test_on_citations_emits_citations_event():
    """_make_on_citations callback writes type=citations event to the run stream."""
    from nanoresearch.worker import _make_on_citations

    stream_key = "run_events:test-run-001"
    fake_redis = _FakeRedis()

    on_citations = _make_on_citations(fake_redis, stream_key)
    await on_citations([{"index": 1, "chunk_id": "c1", "source": "a.pdf"}])

    events = fake_redis.xadds_for(stream_key)
    assert any(e.get("type") == "citations" for e in events)
    cite_ev = next(e for e in events if e["type"] == "citations")
    assert cite_ev["items"][0]["source"] == "a.pdf"


@pytest.mark.asyncio
async def test_on_citations_serializes_items_immediately():
    """Items are JSON-serialized inside xadd_event at call time, so post-call
    mutation of the source list does not corrupt the stream record."""
    from nanoresearch.worker import _make_on_citations

    stream_key = "run_events:test-run-002"
    fake_redis = _FakeRedis()

    on_citations = _make_on_citations(fake_redis, stream_key)
    items = [{"index": 1, "chunk_id": "c1", "source": "original.pdf"}]
    await on_citations(items)
    # Mutate after the call — the stream record must be unaffected.
    items[0]["source"] = "mutated.pdf"

    events = fake_redis.xadds_for(stream_key)
    cite_ev = next(e for e in events if e["type"] == "citations")
    assert cite_ev["items"][0]["source"] == "original.pdf"


@pytest.mark.asyncio
async def test_on_citations_preserves_full_item_shape():
    """All citation fields survive the xadd_event JSON round-trip."""
    from nanoresearch.worker import _make_on_citations

    stream_key = "run_events:test-run-003"
    fake_redis = _FakeRedis()

    on_citations = _make_on_citations(fake_redis, stream_key)
    item = {
        "index": 2,
        "chunk_id": "c2",
        "source": "b.pdf",
        "score": 0.88,
        "snippet": "some text",
        "page": 5,
        "doc_id": "d2",
    }
    await on_citations([item])

    events = fake_redis.xadds_for(stream_key)
    cite_ev = next(e for e in events if e["type"] == "citations")
    assert cite_ev["items"][0] == item
