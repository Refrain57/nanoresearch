"""Post-review fix I2: the worker marks cron-triggered runs so the self-schedule guard is live.

_apply_cron_run_context sets the guard when the run payload carries `cron`; the CronTool then
blocks `add` (covered by test_cron_tool_db.py::test_self_schedule_blocked).
"""
from __future__ import annotations

from types import SimpleNamespace

from nanoresearch.worker import _apply_cron_run_context


class _RecordingTool:
    def __init__(self):
        self.cron_context_set = False

    def set_cron_context(self, active):
        self.cron_context_set = active


def test_apply_cron_run_context_marks_cron_runs():
    tool = _RecordingTool()
    loop = SimpleNamespace(tools={"cron": tool})
    _apply_cron_run_context(loop, {"deliver": False})
    assert tool.cron_context_set is True


def test_apply_cron_run_context_noop_for_normal_runs():
    tool = _RecordingTool()
    loop = SimpleNamespace(tools={"cron": tool})
    _apply_cron_run_context(loop, None)
    assert tool.cron_context_set is False


def test_apply_cron_run_context_tolerates_missing_cron_tool():
    loop = SimpleNamespace(tools={})  # e.g. loop without a DB-backed cron tool
    _apply_cron_run_context(loop, {"deliver": True})  # must not raise
