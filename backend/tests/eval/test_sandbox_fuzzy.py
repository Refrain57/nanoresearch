"""Tests for fuzzy parameter matching in SandboxedToolRegistry side_effect_only mode."""
from __future__ import annotations

import json
import pytest

from nanoresearch.eval.sandbox import SandboxedToolRegistry


class _QueryToolRegistry:
    """Minimal ToolRegistry stand-in whose 'search' tool is a read-only query tool.

    Implements the .get(name) protocol so SandboxedToolRegistry can classify
    the tool as query vs side-effect.  Also provides live_response_for() for
    test assertions about passthrough results.
    """

    class _SearchTool:
        name = "search"
        side_effect = False  # query-only, safe to passthrough

    def __init__(self):
        self._tool = self._SearchTool()

    @property
    def tool_names(self) -> list[str]:
        return ["search"]

    def get_definitions(self) -> list[dict]:
        return [{"name": "search", "description": "search the web"}]

    def register(self, tool) -> None:
        pass

    def get(self, name: str):
        if name == "search":
            return self._tool
        return None

    async def execute(self, name: str, params: dict) -> str:
        """Live passthrough response — stable given the same params."""
        return f"live:{name}:{json.dumps(params, separators=(',', ':'), sort_keys=True)}"

    def live_response_for(self, params: dict) -> str:
        """Expected passthrough result for a given params dict (mirrors execute logic)."""
        return f"live:search:{json.dumps(params, separators=(',', ':'), sort_keys=True)}"


@pytest.fixture
def fake_query_tool_registry():
    return _QueryToolRegistry()


@pytest.mark.asyncio
async def test_exact_key_match_unchanged(fake_query_tool_registry):
    recorded = {'{"tool":"search","params":{"q":"weather"}}': "sunny"}
    sandbox = SandboxedToolRegistry(
        registry=fake_query_tool_registry, mode="side_effect_only", recorded=recorded
    )
    result = await sandbox.execute("search", {"q": "weather"})
    assert result == "sunny"
    assert sandbox.fuzzy_match_ratio == 0.0


@pytest.mark.asyncio
async def test_fuzzy_match_on_whitespace_difference(fake_query_tool_registry):
    """Param value with extra whitespace should match a recorded call with stripped whitespace."""
    recorded = {'{"tool":"search","params":{"q":"weather"}}': "sunny"}
    sandbox = SandboxedToolRegistry(
        registry=fake_query_tool_registry, mode="side_effect_only", recorded=recorded
    )
    result = await sandbox.execute("search", {"q": "  weather  "})  # extra whitespace
    assert result == "sunny"
    audit_entries = [e for e in sandbox._audit_log if e.get("match_type") == "fuzzy"]
    assert len(audit_entries) == 1
    assert sandbox.fuzzy_match_ratio == 1.0


@pytest.mark.asyncio
async def test_fuzzy_match_on_key_order_difference(fake_query_tool_registry):
    """Same params in different order should match."""
    recorded = {'{"tool":"search","params":{"a":"1","b":"2"}}': "ok"}
    sandbox = SandboxedToolRegistry(
        registry=fake_query_tool_registry, mode="side_effect_only", recorded=recorded
    )
    result = await sandbox.execute("search", {"b": "2", "a": "1"})
    assert result == "ok"
    assert sandbox.fuzzy_match_ratio == 1.0


@pytest.mark.asyncio
async def test_fuzzy_does_not_match_semantic_difference(fake_query_tool_registry):
    """Different param VALUES (not whitespace/order) must NOT fuzzy-match."""
    recorded = {'{"tool":"search","params":{"q":"weather"}}': "sunny"}
    sandbox = SandboxedToolRegistry(
        registry=fake_query_tool_registry, mode="side_effect_only", recorded=recorded
    )
    # Different query value — should miss, fall through to passthrough (query tool)
    result = await sandbox.execute("search", {"q": "stocks"})
    # passthrough returns whatever the live registry produces
    assert result == fake_query_tool_registry.live_response_for({"q": "stocks"})
    # Audit log should show miss → passthrough, NOT fuzzy match
    fuzzy_entries = [e for e in sandbox._audit_log if e.get("match_type") == "fuzzy"]
    assert len(fuzzy_entries) == 0
