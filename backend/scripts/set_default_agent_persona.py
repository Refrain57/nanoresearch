#!/usr/bin/env python
"""Set persona for the default agent (Nano Research).

Usage:
    cd backend
    DATABASE_URL=... uv run scripts/set_default_agent_persona.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PERSONA = """# 角色
你是 Nano Research，一个专注于知识检索与研究的 AI 助理。
回答严谨、引用来源、结论明确。

---

# 工作流程

**第一步：始终先查本地知识库（kb_search）**

**第二步：根据结果选择路径**

- **路径 A — 直接回答**
  KB 结果已能完整支撑回答 → 直接基于 KB 内容作答，注明来源。

- **路径 B — 补充搜索后回答**
  KB 结果不足或无结果，且这是一个简单查询 → 调用 web_search 或 web_fetch 补充信息后作答。

- **路径 C — 启动深度调研**
  KB 结果不足或无结果，且这是一个调研任务 → 将 KB 已有内容作为背景资料，传入 Deep Research 启动调研。"""


async def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL 未设置", file=sys.stderr)
        sys.exit(1)

    from nanoresearch.storage.database import init_engine, get_session_factory
    from nanoresearch.storage.repositories.agent_repo import AgentRepository

    init_engine(database_url)
    factory = get_session_factory()

    repo = AgentRepository(factory)
    agent = await repo.get_default()
    if agent is None:
        print("ERROR: 未找到默认 Agent", file=sys.stderr)
        sys.exit(1)

    updated = await repo.update(agent.id, persona=PERSONA.strip())
    print(f"已更新 Agent '{updated.name}' (id={updated.id}) 的 persona，共 {len(PERSONA)} 字符。")


if __name__ == "__main__":
    asyncio.run(main())
