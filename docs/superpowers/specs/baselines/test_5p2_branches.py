"""§5.2 (b) part 1 — isinstance branch dispatch unit test.

Verifies execute_tool routes each result type to the correct CallToolResult shape,
and the new MCPToolResponse branch is hit (not falling through to default str()).

Archived evidence under docs/superpowers/specs/baselines/. To re-run, invoke
`python docs/superpowers/specs/baselines/test_5p2_branches.py` from the
`backend/` root with `PYTHONPATH=.` so `nanobot.*` resolves.
"""

import asyncio

from mcp import types

from nanobot.rag.core.response.response_builder import MCPToolResponse
from nanobot.rag.mcp_server.protocol_handler import ProtocolHandler


async def main() -> None:
    h = ProtocolHandler(server_name="t", server_version="0")

    samples = {
        "callresult": types.CallToolResult(
            content=[types.TextContent(type="text", text="passthrough")],
            isError=False,
        ),
        "str": "plain string",
        "mcp_json": MCPToolResponse(content='{"k": 1}', metadata={"format": "json"}),
        "mcp_markdown": MCPToolResponse(content="# heading\n- bullet"),
        "list": [types.TextContent(type="text", text="block1")],
        "dict_fallback": {"any": "object"},
    }

    for tag, payload in samples.items():
        async def handler(_payload=payload):
            return _payload
        h.register_tool(
            name=f"t_{tag}",
            description="x",
            input_schema={"type": "object", "properties": {}},
            handler=handler,
        )

    results = {}
    for tag in samples:
        results[tag] = await h.execute_tool(f"t_{tag}", {})

    print("=== branch dispatch ===")
    for tag, r in results.items():
        assert isinstance(r, types.CallToolResult), f"{tag}: not CallToolResult"
        text = r.content[0].text if r.content else "<no content>"
        print(f"  {tag:14s} isError={r.isError} text[:60]={text[:60]!r}")

    # Hard checks
    assert results["callresult"].content[0].text == "passthrough"
    assert results["str"].content[0].text == "plain string"
    assert results["mcp_json"].content[0].text == '{"k": 1}', "MCP/JSON: bare content expected"
    assert "MCPToolResponse" not in results["mcp_json"].content[0].text, "must not be repr"
    assert results["mcp_markdown"].content[0].text == "# heading\n- bullet", "MCP/MD: bare content"
    assert "MCPToolResponse" not in results["mcp_markdown"].content[0].text
    assert results["list"].content[0].text == "block1", "list branch passthrough"
    # default branch still str()s unknown objects
    assert "any" in results["dict_fallback"].content[0].text, "dict falls to default str()"

    print("\nALL BRANCH ASSERTIONS PASSED")
    print(f"  - CallToolResult passthrough: OK")
    print(f"  - str branch:                 OK")
    print(f"  - MCPToolResponse (JSON):     OK  ← NEW branch hits, no repr")
    print(f"  - MCPToolResponse (Markdown): OK  ← NEW branch handles non-JSON content")
    print(f"  - list branch:                OK  (not stolen by new MCP branch)")
    print(f"  - default str() branch:       OK  (unknown obj still falls through)")


if __name__ == "__main__":
    asyncio.run(main())
