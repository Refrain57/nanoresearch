"""Path A baseline capture for §5.2 fix.

Spawns the RAG MCP subprocess directly (no agent loop), calls kb_search
3 times, dumps each TextContent.text — this is exactly what
MCPToolWrapper.execute() would receive and write into role:"tool" content.

Run TWICE: once before §5.2 fix, once after. Diff the dumps.

Archived evidence under docs/superpowers/specs/baselines/. To re-run, invoke
`python docs/superpowers/specs/baselines/baseline_5p2.py` from the `backend/`
root with `PYTHONPATH=.` so `nanobot.*` resolves.
"""

import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

QUERIES = [
    ("q1", "Agentic RAG 和普通 RAG 的核心区别是什么"),
    ("q2", "Skill 系统的设计是什么样的,Skill 和 MCP 协议有什么区别"),
    ("q3", "chunk 切分用了什么策略,为什么不用固定 chunk_size"),
]

OUT_DIR = Path("D:/Code/nanobot/docs/superpowers/specs/baselines")
PHASE = os.environ.get("BASELINE_PHASE", "pre-5p2")


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(
        command="python",
        args=["-m", "nanobot.rag.mcp_server.server"],
        env={**os.environ, "PYTHONPATH": "D:/Code/nanobot/backend"},
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"MCP_TOOLS: {[t.name for t in tools.tools]}", flush=True)
            for tag, q in QUERIES:
                print(f"\n=== {tag}: {q!r} ===", flush=True)
                result = await session.call_tool("kb_search", arguments={"query": q})
                texts = [
                    b.text for b in result.content if hasattr(b, "text")
                ]
                raw = "\n".join(texts)
                out = OUT_DIR / f"{PHASE}-{tag}-toolcontent.txt"
                with open(out, "w", encoding="utf-8") as f:
                    f.write(f"# Query: {q}\n")
                    f.write(f"# Tool: kb_search\n")
                    f.write(f"# isError: {result.isError}\n")
                    f.write(f"# Content blocks: {len(result.content)}\n")
                    f.write(f"# Total raw length: {len(raw)}\n")
                    f.write(f"# First 4 chars (hex): {raw[:4].encode('utf-8').hex() if raw else 'EMPTY'}\n")
                    f.write("---RAW TextContent.text START---\n")
                    f.write(raw)
                    f.write("\n---RAW TextContent.text END---\n")
                print(
                    f"  → {out}",
                    flush=True,
                )
                print(f"  raw[:120]={raw[:120]!r}", flush=True)
                # quick parse probe
                try:
                    json.loads(raw)
                    print("  PARSE: json.loads() OK on raw (would be bare JSON)", flush=True)
                except Exception as e:
                    print(f"  PARSE: json.loads() FAILED on raw → {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
