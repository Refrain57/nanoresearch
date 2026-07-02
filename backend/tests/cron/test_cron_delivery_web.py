"""Tests for the web cron-delivery gate (worker → origin conversation + live SSE)."""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock

from nanoresearch.cron.delivery import deliver_cron_result_web


def _fake_redis():
    r = AsyncMock()
    r.xadd = AsyncMock()
    r.expire = AsyncMock()
    return r


def _fake_sessions():
    s = AsyncMock()
    s.append_message = AsyncMock()
    return s


@pytest.mark.asyncio
async def test_web_deliver_persists_and_pushes():
    redis, sessions = _fake_redis(), _fake_sessions()
    cron = {"deliver": True, "channel": "web", "to": "conv-123", "task_context": "查天气"}

    ok = await deliver_cron_result_web(
        redis, sessions, uid="u1", cron=cron, response_text="今天多云 26°C")

    assert ok is True

    # Persisted into the ORIGIN conversation as an assistant message.
    sessions.append_message.assert_awaited_once()
    args, kwargs = sessions.append_message.call_args
    assert args[0] == "web:conv-123"
    entry = args[1]
    assert entry["role"] == "assistant"
    assert entry["content"]["text"] == "今天多云 26°C"
    assert kwargs.get("uid") == "u1"

    # Pushed onto the conversation's live stream.
    redis.xadd.assert_awaited()
    stream_key = redis.xadd.call_args.args[0]
    assert stream_key == "conv_live:conv-123"
    payload = json.loads(redis.xadd.call_args.args[1]["data"])
    assert payload["type"] == "cron_message"
    assert payload["content"]["text"] == "今天多云 26°C"


@pytest.mark.asyncio
async def test_no_deliver_is_noop():
    redis, sessions = _fake_redis(), _fake_sessions()
    cron = {"deliver": False, "channel": "web", "to": "conv-1", "task_context": "x"}
    ok = await deliver_cron_result_web(redis, sessions, uid="u1", cron=cron, response_text="hi")
    assert ok is False
    sessions.append_message.assert_not_awaited()
    redis.xadd.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_web_channel_is_noop():
    """External channels go through the (deferred) outbound bridge, not this web path."""
    redis, sessions = _fake_redis(), _fake_sessions()
    cron = {"deliver": True, "channel": "feishu", "to": "chat-9", "task_context": "x"}
    ok = await deliver_cron_result_web(redis, sessions, uid="u1", cron=cron, response_text="hi")
    assert ok is False
    sessions.append_message.assert_not_awaited()
    redis.xadd.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_response_is_noop():
    redis, sessions = _fake_redis(), _fake_sessions()
    cron = {"deliver": True, "channel": "web", "to": "conv-1", "task_context": "x"}
    ok = await deliver_cron_result_web(redis, sessions, uid="u1", cron=cron, response_text="   ")
    assert ok is False
    sessions.append_message.assert_not_awaited()
    redis.xadd.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_cron_is_noop():
    redis, sessions = _fake_redis(), _fake_sessions()
    ok = await deliver_cron_result_web(redis, sessions, uid="u1", cron=None, response_text="hi")
    assert ok is False
