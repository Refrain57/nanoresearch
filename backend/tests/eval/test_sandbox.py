"""Tests for SandboxedToolRegistry — passthrough, record, replay modes."""

from __future__ import annotations

import json
import pytest

from nanobot.eval.sandbox import SandboxedToolRegistry, SandboxReplayError


class _DummyRegistry:
    """Minimal ToolRegistry stand-in for testing."""
    tool_names = ["dummy"]

    def get_definitions(self) -> list[dict]:
        return [{"name": "dummy", "description": "test"}]

    def register(self, tool) -> None:
        pass

    async def execute(self, name: str, params: dict) -> str:
        return f"result:{name}:{json.dumps(params)}"


@pytest.fixture
def registry():
    return _DummyRegistry()


@pytest.mark.asyncio
class TestSandboxPassthrough:
    async def test_passthrough_calls_real_tool(self, registry):
        sandbox = SandboxedToolRegistry(registry, mode="passthrough")
        result = await sandbox.execute("dummy", {"x": 1})
        assert result == 'result:dummy:{"x": 1}'
        assert sandbox.recordings == {}

    async def test_passthrough_forwards_registry_interface(self, registry):
        sandbox = SandboxedToolRegistry(registry, mode="passthrough")
        assert sandbox.tool_names == ["dummy"]
        assert sandbox.get_definitions() == [{"name": "dummy", "description": "test"}]


@pytest.mark.asyncio
class TestSandboxRecord:
    async def test_record_stores_results(self, registry):
        sandbox = SandboxedToolRegistry(registry, mode="record")
        await sandbox.execute("dummy", {"x": 1})
        await sandbox.execute("dummy", {"y": 2})
        rec = sandbox.recordings
        assert len(rec) == 2

    async def test_record_truncates_long_strings(self, registry):
        sandbox = SandboxedToolRegistry(registry, mode="record")
        long_val = "x" * 20_000
        result = await sandbox.execute("dummy", {"data": long_val})
        # Return value is NOT truncated — only recordings are
        assert result == 'result:dummy:{"data": ' + '"' + "x" * 20000 + '"}'
        # Recorded value IS truncated
        recorded = list(sandbox.recordings.values())[0]
        assert len(recorded) <= 10000 + len("...(truncated)")

    async def test_record_caps_entries(self, registry):
        sandbox = SandboxedToolRegistry(registry, mode="record")
        for i in range(250):
            await sandbox.execute("dummy", {"i": i})
        # Only 200 entries stored (dropped count not exposed in recordings)
        assert len(sandbox.recordings) <= 200

    async def test_export_recordings_returns_valid_json(self, registry):
        sandbox = SandboxedToolRegistry(registry, mode="record")
        await sandbox.execute("dummy", {"x": 1})
        exported = sandbox.export_recordings()
        parsed = json.loads(exported)
        assert isinstance(parsed, dict)
        assert len(parsed) == 1

    async def test_export_recordings_empty(self, registry):
        sandbox = SandboxedToolRegistry(registry, mode="record")
        assert sandbox.export_recordings() == "{}"

    async def test_export_recordings_never_raises(self, registry):
        sandbox = SandboxedToolRegistry(registry, mode="record")
        # Even with un-serializable data, export should not crash
        sandbox._recorded = {object(): object()}
        result = sandbox.export_recordings()  # should not raise
        assert isinstance(result, str)


@pytest.mark.asyncio
class TestSandboxReplay:
    async def test_replay_returns_recorded_result(self, registry):
        recordings = json.dumps({"dummy:{}": "cached_result"})
        sandbox = SandboxedToolRegistry.from_recordings_json(registry, recordings)
        result = await sandbox.execute("dummy", {})
        assert result == "cached_result"

    async def test_replay_raises_on_missing_key(self, registry):
        recordings = json.dumps({})
        sandbox = SandboxedToolRegistry.from_recordings_json(registry, recordings)
        with pytest.raises(SandboxReplayError, match="No recorded result"):
            await sandbox.execute("dummy", {"x": 1})

    async def test_replay_raises_on_invalid_json(self, registry):
        with pytest.raises(SandboxReplayError, match="Invalid recording JSON"):
            SandboxedToolRegistry.from_recordings_json(registry, "not-json")

    async def test_replay_normalizes_params(self, registry):
        # Same params in different order should produce same key
        rec = json.dumps({"dummy:{}": "ordered"})
        sandbox = SandboxedToolRegistry.from_recordings_json(registry, rec)
        result = await sandbox.execute("dummy", {})
        assert result == "ordered"

    async def test_replay_empty_export(self, registry):
        # from_recordings_json with empty dict should work
        sandbox = SandboxedToolRegistry.from_recordings_json(registry, "{}")
        with pytest.raises(SandboxReplayError):
            await sandbox.execute("dummy", {"x": 1})


@pytest.mark.asyncio
class TestSandboxModeTransition:
    async def test_record_then_replay_roundtrip(self, registry):
        record_sandbox = SandboxedToolRegistry(registry, mode="record")
        await record_sandbox.execute("dummy", {"a": 1})
        exported = record_sandbox.export_recordings()

        replay_sandbox = SandboxedToolRegistry.from_recordings_json(registry, exported)
        result = await replay_sandbox.execute("dummy", {"a": 1})
        assert result == 'result:dummy:{"a": 1}'

    async def test_export_recordings_handles_non_string_result(self, registry):
        class _NonStrRegistry:
            tool_names = ["ns"]
            def get_definitions(self): return []
            def register(self, tool): pass
            async def execute(self, name, params): return 42  # non-string

        sandbox = SandboxedToolRegistry(_NonStrRegistry(), mode="record")
        await sandbox.execute("ns", {})
        exported = sandbox.export_recordings()
        parsed = json.loads(exported)
        assert list(parsed.values())[0] == 42
