# backend/tests/test_message_tool.py
import pytest
from nanoresearch.agent.tools.message import MessageTool
from nanoresearch.bus.events import OutboundMessage


@pytest.mark.asyncio
async def test_sent_media_aligned_with_contents():
    sent: list[OutboundMessage] = []

    async def cb(m):
        sent.append(m)

    t = MessageTool(send_callback=cb, default_channel="web", default_chat_id="c1")
    t.start_turn()
    await t.execute(content="here", media=["/ws/users/alice/a.md"])
    await t.execute(content="no file")
    assert t.sent_contents() == ["here", "no file"]
    assert t.sent_media() == [["/ws/users/alice/a.md"], []]


@pytest.mark.asyncio
async def test_start_turn_resets_media():
    async def cb(m):
        return None

    t = MessageTool(send_callback=cb, default_channel="web", default_chat_id="c1")
    t.start_turn()
    await t.execute(content="x", media=["/ws/users/alice/a.md"])
    t.start_turn()
    assert t.sent_media() == []
