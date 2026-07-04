import uuid
import pytest

from nanoresearch.eval.snapshot import RunSnapshotData
from nanoresearch.server.routers import agent_eval_router as R


class _FakeProvider:
    def get_default_model(self):
        return "m"


class _FakeTools:
    tool_names = []

    def get_definitions(self):
        return []

    async def execute(self, name, params):  # never called (lenient)
        raise AssertionError("no live calls")


class _FakeLoop:
    def __init__(self):
        self.tools = _FakeTools()
        self.provider = _FakeProvider()
        self.model = "m"


class _FakeState:
    channel_loop = _FakeLoop()


class _FakeApp:
    state = _FakeState()


class _FakeRequest:
    app = _FakeApp()


@pytest.mark.asyncio
async def test_replay_persists_and_compares(monkeypatch, eval_repo):
    import json
    # baseline live snapshot with one recorded tool call. tool_recordings is a JSON
    # STRING keyed "<tool>:<json-params>" exactly as the sandbox builds the key
    # (json.dumps → '{"q": "a"}' with a space after the colon). Build via json.dumps
    # so the embedded quotes are escaped correctly.
    rec = {f'search:{json.dumps({"q": "a"})}': "r"}
    base_id = await eval_repo.save_snapshot(
        RunSnapshotData(
            run_id="base", user_input="hi",
            tool_call_chain=[{"order": 1, "name": "search", "params": {"q": "a"}, "result": "r", "error": False}],
            llm_calls=[], final_response="answer", run_status="success",
            total_input_tokens=0, total_output_tokens=0, ttft_ms=None,
            total_duration_ms=1.0, tool_call_count=1, llm_call_count=0, retry_count=0,
        ),
        uid="u1", tool_recordings=json.dumps(rec),
    )

    # stub the runner so replay reproduces the same single tool call deterministically
    async def _fake_run(self, spec):
        await spec.tools.execute("search", {"q": "a"})
        if spec.snapshot_collector:
            spec.snapshot_collector.on_tool_start("t1", "search", {"q": "a"})
            spec.snapshot_collector.on_tool_end("t1", "r")
        from nanoresearch.agent.runner import AgentRunResult
        return AgentRunResult(final_content="answer", messages=[], stop_reason="completed")

    # replay_snapshot imports AgentRunner INSIDE the function, so patch the method
    # on the real class object by import path (works regardless of import location).
    monkeypatch.setattr("nanoresearch.agent.runner.AgentRunner.run", _fake_run)

    resp = await R.replay_snapshot(uuid.UUID(str(base_id)), _FakeRequest(), uid="u1", repo=eval_repo)

    assert resp["compare"]["verdicts"]["strict"] is True
    assert resp["compare"]["final_response_equal"] is True
    replays = await eval_repo.list_replays_for_root(uuid.UUID(str(base_id)))
    assert len(replays) == 1
    assert replays[0].origin == "replay"
    assert str(replays[0].id) == resp["replay_snapshot_id"]
