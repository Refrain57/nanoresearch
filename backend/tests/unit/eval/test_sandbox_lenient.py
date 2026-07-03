import json
import pytest

from nanoresearch.eval.sandbox import SandboxedToolRegistry


class _FakeRegistry:
    """Minimal ToolRegistry stand-in; execute() must never be called in lenient replay."""
    tool_names = ["search"]

    def get_definitions(self):
        return []

    async def execute(self, name, params):
        raise AssertionError("lenient replay must not make live calls")


@pytest.mark.asyncio
async def test_exact_hit_returns_recording():
    key = f"search:{json.dumps({'q': 'a'})}"
    sb = SandboxedToolRegistry.from_recordings_json(_FakeRegistry(), json.dumps({key: "RECORDED"}), lenient=True)
    assert await sb.execute("search", {"q": "a"}) == "RECORDED"
    assert sb.misses == []


@pytest.mark.asyncio
async def test_miss_returns_placeholder_and_records():
    sb = SandboxedToolRegistry.from_recordings_json(_FakeRegistry(), "{}", lenient=True)
    result = await sb.execute("web_search", {"q": "today 2025"})
    assert result == "[replay:no-recording]"
    assert sb.misses == [{"name": "web_search", "params": {"q": "today 2025"}}]
