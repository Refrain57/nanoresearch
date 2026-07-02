"""Task 7: AgentLoop registers the DB-backed cron tool iff a DB + uid are present.

Replaces tests/agent/test_loop_cron_timezone.py (which drove the old
CronTool(cron_service) registration). Decision (B): without a DB, no cron tool
(gateway/CLI drop cron; production cron runs under serve + worker).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from nanoresearch.agent.loop import AgentLoop
from nanoresearch.agent.tools.cron import CronTool
from nanoresearch.bus.queue import MessageBus
from tests.conftest import make_factory


def test_agent_loop_registers_db_cron_tool_with_timezone(tmp_path):
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(
        bus=bus, provider=provider, workspace=tmp_path, model="test-model",
        session_factory=make_factory(), uid="u", timezone="Asia/Shanghai",
    )
    cron_tool = loop.tools.get("cron")
    assert isinstance(cron_tool, CronTool)
    assert cron_tool._default_timezone == "Asia/Shanghai"
    assert cron_tool._uid == "u"


def test_agent_loop_no_cron_tool_without_db(tmp_path):
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(
        bus=bus, provider=provider, workspace=tmp_path, model="test-model",
        timezone="Asia/Shanghai",
    )
    assert loop.tools.get("cron") is None
