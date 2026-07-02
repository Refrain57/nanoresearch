"""worker emits agent_message SSE events with workspace attachment descriptors.

_make_web_message_sink(redis, stream_key, workspace_root) returns an async
send-callback that, for web-channel OutboundMessages, writes
{"type":"agent_message","content":..., "media":[descriptors]} to the run
stream via xadd_event. Media paths outside the user's workspace are dropped.

No live Redis: a minimal _FakeRedis captures xadd calls (mirrors
tests/test_worker_citations_event.py).
"""
from __future__ import annotations

import json

import pytest


class _FakeRedis:
    def __init__(self):
        self._streams: dict[str, list[dict]] = {}

    async def xadd(self, key, fields):
        self._streams.setdefault(key, []).append(json.loads(fields["data"]))

    async def expire(self, key, ttl):
        pass

    def xadds_for(self, key):
        return self._streams.get(key, [])


class _Msg:
    def __init__(self, channel, content, media):
        self.channel = channel
        self.content = content
        self.media = media


@pytest.mark.asyncio
async def test_web_sink_emits_media_descriptors(tmp_path):
    from nanoresearch.worker import _make_web_message_sink
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    (ws / "r.md").write_text("hi", encoding="utf-8")

    fake = _FakeRedis()
    sink = _make_web_message_sink(fake, "run_events:t1", ws)
    await sink(_Msg("web", "see file", [str(ws / "r.md")]))

    ev = next(e for e in fake.xadds_for("run_events:t1") if e["type"] == "agent_message")
    assert ev["content"] == "see file"
    assert ev["media"] == [{"path": "r.md", "name": "r.md", "size": 2}]


@pytest.mark.asyncio
async def test_web_sink_drops_out_of_workspace_media(tmp_path):
    from nanoresearch.worker import _make_web_message_sink
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x", encoding="utf-8")

    fake = _FakeRedis()
    sink = _make_web_message_sink(fake, "run_events:t2", ws)
    await sink(_Msg("web", "hi", [str(outside)]))

    ev = next(e for e in fake.xadds_for("run_events:t2") if e["type"] == "agent_message")
    assert ev["media"] == []


@pytest.mark.asyncio
async def test_web_sink_ignores_non_web_channel(tmp_path):
    from nanoresearch.worker import _make_web_message_sink
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    fake = _FakeRedis()
    sink = _make_web_message_sink(fake, "run_events:t3", ws)
    await sink(_Msg("telegram", "hi", []))
    assert fake.xadds_for("run_events:t3") == []
