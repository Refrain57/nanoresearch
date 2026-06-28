# Consolidation Anchor Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 consolidation 之后外层 Agent 仍能从 history tail 和 prompt summary 两层看到最近对话锚点，修复"刚说完的话被压成长期记忆"的 bug。

**Architecture:** 三处独立修改：(1) `loop.py:_check_pending_consolidation` 加 idle gate + 复用 `pick_consolidation_boundary` 留 tail；(2) `memory.py:_CONSOLIDATION_SYSTEM_PROMPT` 追加 RECENT_TOPICS section 让 LLM 抽取关键实体写进 MEMORY.md；(3) `loop.py:_process_message` 把 RAG 召回 query 从单条用户输入扩成最近 3 轮 user message 拼接。

**Tech Stack:** Python 3.11+, pytest (asyncio_mode=auto), unittest.mock, loguru.

## Global Constraints

- 改动须可单独 revert（3 个独立 commit）
- 不动 `maybe_consolidate_by_tokens` 主路径
- 不动 MCP 子进程 uid 透传（独立 spec）
- 不动 user_memory Chroma collection 切分
- 不动 `SessionManager(default_uid="admin")` fallback
- `IDLE_THRESHOLD = 30 分钟`（1800 秒）默认；`tail_protect = 5`；`RECALL_TOPIC_USER_TURNS = 3`
- 不引入新的 schema 字段，复用 `Session.updated_at`
- MEMORY.md 增加 RECENT_TOPICS section 必须向后兼容（老 reader 按 markdown 解析无害）
- 所有新测试放在 `backend/tests/unit/agent/` 或 `backend/tests/integration/`，目录不存在则创建
- 测试用 `pytest-asyncio` 的 `asyncio_mode=auto`（已在 `pyproject.toml:144`），`async def test_*` 不加 decorator

## File Structure

| 文件 | 改动 | 责任 |
|---|---|---|
| `backend/nanoresearch/agent/loop.py` | Modify | `_check_pending_consolidation` 加 idle gate + tail protect；`_process_message` 加 `_build_recall_topic` 并替换 `topic=msg.content` |
| `backend/nanoresearch/agent/memory.py` | Modify | `_CONSOLIDATION_SYSTEM_PROMPT` 追加 RECENT_TOPICS section 说明和模板 |
| `backend/tests/unit/agent/__init__.py` | Create | 测试包初始化（空文件） |
| `backend/tests/unit/agent/test_check_pending_consolidation.py` | Create | 修法 1 单测：idle gate / tail protect / 边界情况 |
| `backend/tests/unit/agent/test_consolidation_prompt.py` | Create | 修法 2A 单测：prompt 模板包含 RECENT_TOPICS；mock LLM 返回 → MEMORY.md 写入正确 |
| `backend/tests/unit/agent/test_topic_expansion.py` | Create | 修法 2B 单测：`_build_recall_topic` 各种 history 形态 |
| `backend/tests/integration/test_consolidation_anchor_e2e.py` | Create | 端到端模拟：Turn 1 谈 X → idle 31min → Turn 2 触发 startup consolidation 留 tail → Turn 3 含代词 → 验证 history 仍含 X 原文 |

---

### Task 1: 修法 1 — `_check_pending_consolidation` idle gate + tail protect

**Files:**
- Modify: `backend/nanoresearch/agent/loop.py:530-560`
- Create: `backend/tests/unit/agent/__init__.py`
- Test: `backend/tests/unit/agent/test_check_pending_consolidation.py`

**Interfaces:**
- Consumes:
  - `Session(key: str, messages: list[dict], last_consolidated: int, updated_at: datetime, ...)` from `nanoresearch.session.manager`
  - `MemoryConsolidator.pick_consolidation_boundary(session, tokens_to_remove: int, tail_protect: int = 5) -> tuple[int, int] | None` from `memory.py:486` — 返回 `(end_idx, removed_tokens)` 或 None
  - `MemoryConsolidator.consolidate_messages(messages, agent_id=None, uid=None) -> bool`
  - `SessionManager.save(session) -> Awaitable` (async)
- Produces: `AgentLoop._check_pending_consolidation(self, session, agent_id=None) -> None` 行为变化（不变签名）

- [ ] **Step 1: 创建测试包初始化文件**

```bash
mkdir -p backend/tests/unit/agent
```

写入 `backend/tests/unit/agent/__init__.py`：（空文件）

```python
```

- [ ] **Step 2: 写完整失败测试**

写入 `backend/tests/unit/agent/test_check_pending_consolidation.py`：

```python
"""Unit tests for AgentLoop._check_pending_consolidation (idle gate + tail protect)."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanoresearch.agent.loop import AgentLoop
from nanoresearch.session.manager import Session


def _make_loop_stub(uid: str = "test_uid") -> SimpleNamespace:
    """Construct a minimal AgentLoop-like object for unbound method testing."""
    stub = SimpleNamespace()
    stub._startup_consolidated = set()
    stub._uid = uid
    stub.memory_consolidator = MagicMock()
    stub.memory_consolidator.consolidate_messages = AsyncMock(return_value=True)
    stub.memory_consolidator.pick_consolidation_boundary = MagicMock()
    stub.sessions = MagicMock()
    stub.sessions.save = AsyncMock()
    return stub


def _make_session(
    num_messages: int = 10,
    last_consolidated: int = 0,
    updated_at: datetime | None = None,
) -> Session:
    return Session(
        key="test:session",
        messages=[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(num_messages)
        ],
        last_consolidated=last_consolidated,
        updated_at=updated_at or datetime.now(),
    )


async def test_idle_gate_skips_when_session_recently_active():
    """If now - session.updated_at < 30min, skip consolidation entirely."""
    stub = _make_loop_stub()
    session = _make_session(
        num_messages=10,
        last_consolidated=0,
        updated_at=datetime.now() - timedelta(minutes=5),
    )

    await AgentLoop._check_pending_consolidation(stub, session)

    assert "test:session" in stub._startup_consolidated
    stub.memory_consolidator.consolidate_messages.assert_not_called()
    stub.memory_consolidator.pick_consolidation_boundary.assert_not_called()


async def test_idle_gate_triggers_when_session_stale():
    """If now - session.updated_at >= 30min and pending_count >= 5, trigger."""
    stub = _make_loop_stub()
    stub.memory_consolidator.pick_consolidation_boundary.return_value = (5, 100)
    session = _make_session(
        num_messages=10,
        last_consolidated=0,
        updated_at=datetime.now() - timedelta(minutes=45),
    )

    await AgentLoop._check_pending_consolidation(stub, session)

    stub.memory_consolidator.consolidate_messages.assert_awaited_once()
    stub.sessions.save.assert_awaited_once()
    assert "test:session" in stub._startup_consolidated


async def test_tail_protect_advances_to_boundary_not_to_end():
    """After consolidating, last_consolidated must == boundary, not len(messages)."""
    stub = _make_loop_stub()
    stub.memory_consolidator.pick_consolidation_boundary.return_value = (5, 100)
    session = _make_session(
        num_messages=10,
        last_consolidated=0,
        updated_at=datetime.now() - timedelta(minutes=45),
    )

    await AgentLoop._check_pending_consolidation(stub, session)

    assert session.last_consolidated == 5
    tail = session.messages[session.last_consolidated:]
    assert len(tail) == 5
    assert tail[0]["content"] == "m5"


async def test_no_boundary_marks_seen_and_returns():
    """If pick_consolidation_boundary returns None, mark seen and exit cleanly."""
    stub = _make_loop_stub()
    stub.memory_consolidator.pick_consolidation_boundary.return_value = None
    session = _make_session(
        num_messages=10,
        last_consolidated=0,
        updated_at=datetime.now() - timedelta(minutes=45),
    )

    await AgentLoop._check_pending_consolidation(stub, session)

    stub.memory_consolidator.consolidate_messages.assert_not_called()
    assert "test:session" in stub._startup_consolidated


async def test_pending_below_threshold_skips_regardless_of_idle():
    """If pending_count < 5, skip even when stale."""
    stub = _make_loop_stub()
    session = _make_session(
        num_messages=10,
        last_consolidated=6,
        updated_at=datetime.now() - timedelta(minutes=45),
    )

    await AgentLoop._check_pending_consolidation(stub, session)

    stub.memory_consolidator.consolidate_messages.assert_not_called()
    stub.memory_consolidator.pick_consolidation_boundary.assert_not_called()
    assert "test:session" in stub._startup_consolidated


async def test_already_seen_session_skips():
    """If session.key already marked, skip everything."""
    stub = _make_loop_stub()
    stub._startup_consolidated.add("test:session")
    session = _make_session(
        num_messages=10,
        last_consolidated=0,
        updated_at=datetime.now() - timedelta(minutes=45),
    )

    await AgentLoop._check_pending_consolidation(stub, session)

    stub.memory_consolidator.consolidate_messages.assert_not_called()
    stub.memory_consolidator.pick_consolidation_boundary.assert_not_called()


async def test_consolidation_failure_does_not_advance_pointer():
    """If consolidate_messages returns False, do NOT advance last_consolidated."""
    stub = _make_loop_stub()
    stub.memory_consolidator.consolidate_messages = AsyncMock(return_value=False)
    stub.memory_consolidator.pick_consolidation_boundary.return_value = (5, 100)
    session = _make_session(
        num_messages=10,
        last_consolidated=0,
        updated_at=datetime.now() - timedelta(minutes=45),
    )

    await AgentLoop._check_pending_consolidation(stub, session)

    assert session.last_consolidated == 0
    stub.sessions.save.assert_not_awaited()
```

- [ ] **Step 3: 跑测试确认失败（红）**

Run: `cd backend && python -m pytest tests/unit/agent/test_check_pending_consolidation.py -v`

Expected: 至少 `test_idle_gate_skips_when_session_recently_active` 和 `test_tail_protect_advances_to_boundary_not_to_end` 失败 —— 因为当前 `_check_pending_consolidation` 没有 idle gate（recently active 也会触发），且把 `last_consolidated` 推到 `len(messages)` 不是 boundary。

- [ ] **Step 4: 改 `loop.py` 实施修法 1**

打开 `backend/nanoresearch/agent/loop.py`，找到第 530-560 行的 `_check_pending_consolidation` 整段：

```python
    async def _check_pending_consolidation(self, session: Session, agent_id: str | None = None) -> None:
        """Check if there are unconsolidated messages from last session and consolidate them.

        This runs once per session when the first message arrives after startup.
        It ensures that important conversations from previous sessions are preserved
        in MEMORY.md even if the previous session ended normally without token pressure.
        """
        # Only check once per session to avoid repeated consolidation
        if session.key in self._startup_consolidated:
            return

        pending_count = len(session.messages) - session.last_consolidated
        if pending_count < 5:
            self._startup_consolidated.add(session.key)  # Mark as checked
            return  # Not enough messages to bother consolidating

        logger.info(
            "Found {} unconsolidated messages from previous session, consolidating...",
            pending_count
        )

        pending = session.messages[session.last_consolidated:]
        success = await self.memory_consolidator.consolidate_messages(pending, agent_id=agent_id, uid=self._uid)

        if success:
            session.last_consolidated = len(session.messages)
            await self.sessions.save(session)
            self._startup_consolidated.add(session.key)  # Mark as done
            logger.info("Startup consolidation complete for {} messages", pending_count)
        else:
            logger.warning("Startup consolidation failed, will retry on token pressure")
```

整体替换为：

```python
    async def _check_pending_consolidation(self, session: Session, agent_id: str | None = None) -> None:
        """Check if there are unconsolidated messages from last session and consolidate them.

        Triggers only when the session has been idle for >= STARTUP_CONSOLIDATION_IDLE_SECONDS
        (default 30 minutes). When triggered, consolidates only the head portion of pending
        messages and leaves the most recent tail unconsolidated so the next turn's
        `history = session.get_history(0)` still contains the recent dialogue anchor.
        """
        from datetime import datetime, timedelta

        # Only check once per session to avoid repeated consolidation
        if session.key in self._startup_consolidated:
            return

        # Idle gate: don't disturb sessions the user is actively engaged with
        idle_seconds = int(os.environ.get("STARTUP_CONSOLIDATION_IDLE_SECONDS", "1800"))
        if datetime.now() - session.updated_at < timedelta(seconds=idle_seconds):
            self._startup_consolidated.add(session.key)
            return

        pending_count = len(session.messages) - session.last_consolidated
        if pending_count < 5:
            self._startup_consolidated.add(session.key)
            return

        # Tail protect: pick a user-turn boundary that leaves the last 5 messages alone
        boundary = self.memory_consolidator.pick_consolidation_boundary(
            session, tokens_to_remove=1, tail_protect=5,
        )
        if boundary is None:
            self._startup_consolidated.add(session.key)
            return

        end_idx, _ = boundary
        chunk = session.messages[session.last_consolidated:end_idx]
        if not chunk:
            self._startup_consolidated.add(session.key)
            return

        logger.info(
            "Startup consolidation triggered (idle): pending={}, chunk={}, tail={}",
            pending_count, len(chunk), len(session.messages) - end_idx,
        )

        success = await self.memory_consolidator.consolidate_messages(
            chunk, agent_id=agent_id, uid=self._uid,
        )
        if success:
            session.last_consolidated = end_idx
            await self.sessions.save(session)
            self._startup_consolidated.add(session.key)
            logger.info("Startup consolidation complete for {} head messages", len(chunk))
        else:
            logger.warning("Startup consolidation failed, will retry on token pressure")
```

- [ ] **Step 5: 跑测试确认通过（绿）**

Run: `cd backend && python -m pytest tests/unit/agent/test_check_pending_consolidation.py -v`

Expected: 全部 7 条测试 PASS。

- [ ] **Step 6: 跑相邻回归避免破坏**

Run: `cd backend && python -m pytest tests/unit/session/ tests/unit/agent/ -v`

Expected: 全部 PASS。`tests/unit/session/` 有 6 个 Redis round-trip 测试是上一个 commit 加的，必须仍通过。

- [ ] **Step 7: 提交**

```bash
git add backend/tests/unit/agent/__init__.py backend/tests/unit/agent/test_check_pending_consolidation.py backend/nanoresearch/agent/loop.py
git commit -m "fix(agent): startup consolidation idle gate + tail protect

_check_pending_consolidation 之前在 pending_count >= 5 时把所有未消化
消息一锅端，token 远未到压力线也照压；session.updated_at 紧贴当前时间
也照样触发。结果 Turn N+1 起步时 history = messages[last_consolidated:]
变成空列表，外层 Agent 看不到最近对话的指代锚点。

新行为：
- idle gate 30 分钟（STARTUP_CONSOLIDATION_IDLE_SECONDS env var，默认 1800s）
- 触发后复用 pick_consolidation_boundary(tail_protect=5) 留尾，
  last_consolidated 推进到 boundary 而不是 len(messages)
- 找不到合法 boundary 时静默退出

修复 2026-06-28 真实 e2e 故障：Turn 3 起步时 history=[] 导致 Agent
无法解析'这篇'指代 CityGaussianV2。"
```

---

### Task 2: 修法 2A — `_CONSOLIDATION_SYSTEM_PROMPT` 加 RECENT_TOPICS section

**Files:**
- Modify: `backend/nanoresearch/agent/memory.py:39-121`
- Test: `backend/tests/unit/agent/test_consolidation_prompt.py`

**Interfaces:**
- Consumes:
  - `MemoryStore(workspace, knowledge_search=None, agent_id=None)` — already exists, `read_long_term() -> str`, `write_long_term(content: str) -> None`
  - `MemoryStore.consolidate(messages, provider, model, uid=None) -> bool`
- Produces: `_CONSOLIDATION_SYSTEM_PROMPT` 字符串包含 `RECENT_TOPICS` 模板说明。MEMORY.md 中 LLM 写入的 `## RECENT_TOPICS` section。

- [ ] **Step 1: 写失败测试**

写入 `backend/tests/unit/agent/test_consolidation_prompt.py`：

```python
"""Unit tests for consolidation prompt RECENT_TOPICS section."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanoresearch.agent.memory import _CONSOLIDATION_SYSTEM_PROMPT, MemoryStore


def test_prompt_template_documents_recent_topics_section():
    """The consolidation system prompt must instruct LLM to maintain a RECENT_TOPICS section."""
    assert "RECENT_TOPICS" in _CONSOLIDATION_SYSTEM_PROMPT
    assert "实体" in _CONSOLIDATION_SYSTEM_PROMPT  # entities
    assert "5" in _CONSOLIDATION_SYSTEM_PROMPT  # rolling cap mentioned somewhere


def test_prompt_template_documents_recent_topics_format():
    """The prompt must show the expected RECENT_TOPICS line format."""
    # Expect an example line like "- 实体名 — 一句话上下文" or similar
    assert "## RECENT_TOPICS" in _CONSOLIDATION_SYSTEM_PROMPT


async def test_consolidate_writes_recent_topics_to_memory_file(tmp_path: Path):
    """When LLM returns memory_update with RECENT_TOPICS, MemoryStore writes it verbatim."""
    store = MemoryStore(workspace=tmp_path)

    fake_memory_update = (
        "# User Memory\n\n"
        "## FACTS\n"
        "- 用户偏好 Python\n\n"
        "## USER_PROFILE\n"
        "资深工程师\n\n"
        "## FOCUS_AREAS\n"
        "- AI Agent 架构\n\n"
        "## RECENT_TOPICS\n"
        "- CityGaussianV2 — 大规模场景重建方法\n"
        "- PGSR — 早期对比的 GS 变体\n"
    )

    fake_response = SimpleNamespace(
        finish_reason="tool_calls",
        content="",
        has_tool_calls=True,
        tool_calls=[SimpleNamespace(arguments=json.dumps({
            "history_entry": "[2026-06-28 21:14] 用户讨论了 CityGaussianV2",
            "memory_update": fake_memory_update,
        }))],
    )

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=fake_response)

    ok = await store.consolidate(
        messages=[{"role": "user", "content": "讲讲 CityGaussianV2"}],
        provider=provider,
        model="test-model",
    )

    assert ok is True
    written = store.read_long_term()
    assert "## RECENT_TOPICS" in written
    assert "CityGaussianV2" in written
    assert "PGSR" in written


async def test_consolidate_preserves_existing_recent_topics_when_llm_keeps_them(tmp_path: Path):
    """If MEMORY.md already has RECENT_TOPICS and LLM returns same content, file unchanged."""
    store = MemoryStore(workspace=tmp_path)
    existing = (
        "# User Memory\n\n"
        "## FACTS\n- foo\n\n"
        "## RECENT_TOPICS\n- A — ctx1\n- B — ctx2\n"
    )
    store.write_long_term(existing)

    fake_response = SimpleNamespace(
        finish_reason="tool_calls",
        content="",
        has_tool_calls=True,
        tool_calls=[SimpleNamespace(arguments=json.dumps({
            "history_entry": "no-op",
            "memory_update": existing,
        }))],
    )
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=fake_response)

    ok = await store.consolidate(
        messages=[{"role": "user", "content": "x"}],
        provider=provider,
        model="test-model",
    )

    assert ok is True
    assert store.read_long_term() == existing
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/unit/agent/test_consolidation_prompt.py -v`

Expected: `test_prompt_template_documents_recent_topics_section` 和 `test_prompt_template_documents_recent_topics_format` 失败（当前 `_CONSOLIDATION_SYSTEM_PROMPT` 不含 RECENT_TOPICS）。

- [ ] **Step 3: 改 `memory.py` 加 RECENT_TOPICS section**

打开 `backend/nanoresearch/agent/memory.py`，找到第 39-121 行 `_CONSOLIDATION_SYSTEM_PROMPT`。

在 "## Output Format for save_memory" 段下的 `### memory_update（MEMORY.md）` 示例代码块里，加入 RECENT_TOPICS section。找到示例代码块：

```python
### memory_update（MEMORY.md）
只包含稳定事实，格式：

```markdown
# User Memory

## FACTS
- 用户偏好 Python
- 工作目录: D:\Code\nanoresearch
- 使用 Claude 模型

## USER_PROFILE
资深工程师，专注 AI Agent 开发。

## FOCUS_AREAS
- AI Agent 架构设计
- RAG 系统优化
```
```

替换为：

```python
### memory_update（MEMORY.md）
只包含稳定事实和近期对话锚点，格式：

```markdown
# User Memory

## FACTS
- 用户偏好 Python
- 工作目录: D:\Code\nanoresearch
- 使用 Claude 模型

## USER_PROFILE
资深工程师，专注 AI Agent 开发。

## FOCUS_AREAS
- AI Agent 架构设计
- RAG 系统优化

## RECENT_TOPICS
- CityGaussianV2 — 用户最近讨论的大规模场景重建方法
- PGSR — 同期对比的 GS 变体
```
```

然后在 "## Memory Update Rules" 段（约 98 行）后追加新 subsection（在 `### FOCUS_AREAS Section` 之后、`### history_entry` 之前）：

```python
### RECENT_TOPICS Section
- 列出本次对话涉及的具体实体（论文名、方法名、KB 标题、人物、专有名词）
- 每条 1 行，格式：- {实体名} — {一句话上下文}
- 滚动覆盖：保留最近 5 条，每次 consolidation 整段 rewrite
- 旧实体若在新对话中仍可能被代词指代（如 "刚才那个"），优先保留
- 与 FACTS 区别：RECENT_TOPICS 是临时锚点（几天到几周），FACTS 是 6 个月不变的稳定事实

```

完整改后的 prompt 文件局部如下（保留其余 section 不变）：

```python
_CONSOLIDATION_SYSTEM_PROMPT = r"""You are a memory consolidation agent. Analyze the conversation and update the memory following the exact format below.

## 内容分类规则（关键）

### 写入 MEMORY.md（稳定事实，6个月后仍成立）
- 用户偏好：语言偏好、工具偏好、工作习惯
- 环境约定：工作目录、API 配置、模型选择
- 长期决策：架构决策、技术选型
- 用户画像：角色、背景、专业领域

### 不写入 MEMORY.md（临时内容）
- 任务进度：当前任务状态、待办事项
- 讨论结论：本次讨论的结论、发现
- 临时焦点：当前调试目标、短期关注点
- 工具调用细节：具体的搜索结果、代码片段

判断标准：这条信息 6 个月后还成立吗？
→ 成立 → 写入 MEMORY.md
→ 不成立/不确定 → 只写入 history_entry，不进 MEMORY.md
但例外：本次对话涉及的具体实体（论文名、方法名等）作为 RECENT_TOPICS 写入 MEMORY.md，用于近期代词消解。

## Output Format for save_memory

### memory_update（MEMORY.md）
只包含稳定事实和近期对话锚点，格式：

```markdown
# User Memory

## FACTS
- 用户偏好 Python
- 工作目录: D:\Code\nanoresearch
- 使用 Claude 模型

## USER_PROFILE
资深工程师，专注 AI Agent 开发。

## FOCUS_AREAS
- AI Agent 架构设计
- RAG 系统优化

## RECENT_TOPICS
- CityGaussianV2 — 用户最近讨论的大规模场景重建方法
- PGSR — 同期对比的 GS 变体
```

### history_entry（HISTORY.md）
结构化摘要，格式：

```markdown
## Session Summary [YYYY-MM-DD HH:MM]
- Active Task: 当前正在进行的任务（如有）
- Completed Actions: 已完成的操作（简要）
- Key Decisions: 做出的关键决策
- Tools Used: 使用的工具列表
- Blocked/Issues: 遇到的阻碍或问题
- Stable Facts: 可进入 MEMORY.md 的事实（如有新发现）
```

注意：
- 临时任务结论不要写入 memory_update 的 FACTS 或 FOCUS_AREAS
- history_entry 使用结构化字段，便于后续解析
- 如果某字段无内容，写 "无" 或跳过该字段

## Memory Update Rules

### FACTS Section
- 只添加稳定事实（用户偏好、环境约定、长期决策）
- 移除被否定/过时的事实
- 每条一行，grep 可搜索
- 不重复已有事实

### USER_PROFILE Section
- 用户透露的新信息时更新
- 移除过时信息
- 最多 3 句

### FOCUS_AREAS Section
- 只保留长期关注点（非临时任务）
- 最多 5 个
- 如果本次对话没有新的长期焦点，保持原有不变

### RECENT_TOPICS Section
- 列出本次对话涉及的具体实体（论文名、方法名、KB 标题、人物、专有名词）
- 每条 1 行，格式：- {实体名} — {一句话上下文}
- 滚动覆盖：保留最近 5 条，每次 consolidation 整段 rewrite
- 旧实体若在新对话中仍可能被代词指代（如 "刚才那个"），优先保留
- 与 FACTS 区别：RECENT_TOPICS 是临时锚点（几天到几周），FACTS 是 6 个月不变的稳定事实

### history_entry
- 以 ## Session Summary [YYYY-MM-DD HH:MM] 开头
- 使用固定字段格式
- 关键词 grep 可搜索

Call the save_memory tool with your consolidation."""
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/unit/agent/test_consolidation_prompt.py -v`

Expected: 全部 4 条测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add backend/tests/unit/agent/test_consolidation_prompt.py backend/nanoresearch/agent/memory.py
git commit -m "feat(memory): RECENT_TOPICS section for consolidation

_CONSOLIDATION_SYSTEM_PROMPT 之前只让 LLM 维护 FACTS / USER_PROFILE /
FOCUS_AREAS 这三段稳定事实，临时实体（论文名、方法名）按规则不进
MEMORY.md。结果 consolidation 后下一轮代词消解只能靠 RAG 召回
user_memory，召回 query 是当前用户输入，命中率低。

新增 ## RECENT_TOPICS 段：
- 滚动保留 5 条近期对话涉及的具体实体
- 每次 consolidation 由 LLM 整段 rewrite
- 旧实体可保留（如仍可能被代词指代）
- 直接写入 MEMORY.md，通过 context.py 的 <memory> wrapper 自动注入 prompt

与修法 1 (idle gate + tail protect) 配套：tail 没保住的部分仍能
从 RECENT_TOPICS 找回锚点。"
```

---

### Task 3: 修法 2B — `_process_message` 加 `_build_recall_topic` 扩展召回 query

**Files:**
- Modify: `backend/nanoresearch/agent/loop.py:771-790`（在 `history = session.get_history(...)` 之后、`build_messages(..., topic=msg.content, ...)` 之前插入 topic 构造）
- Modify: `backend/nanoresearch/agent/loop.py` 类内加私有方法 `_build_recall_topic`
- Test: `backend/tests/unit/agent/test_topic_expansion.py`

**Interfaces:**
- Consumes: `history: list[dict]` —  历史消息列表，每条 `{"role": str, "content": str | list, ...}`
- Produces:
  - `AgentLoop._build_recall_topic(history: list[dict], current_msg: str, n: int = 3) -> str`：拼接最近 n 条 user role 消息 + 当前消息，用 `\n` 分隔
  - `_process_message` 调用 `build_messages` 时 `topic` 参数由 `self._build_recall_topic(history, msg.content)` 提供

- [ ] **Step 1: 写失败测试**

写入 `backend/tests/unit/agent/test_topic_expansion.py`：

```python
"""Unit tests for AgentLoop._build_recall_topic."""

from __future__ import annotations

from nanoresearch.agent.loop import AgentLoop


def test_recall_topic_concatenates_recent_user_messages():
    """Should concat last N user messages + current with newlines."""
    history = [
        {"role": "user", "content": "查查看3dgs"},
        {"role": "assistant", "content": "3DGS 是 ..."},
        {"role": "user", "content": "看看那个大规模场景的论文"},
        {"role": "assistant", "content": "CityGaussianV2 ..."},
    ]
    topic = AgentLoop._build_recall_topic(None, history, "这篇有和与nerf的对比吗？", n=3)

    assert "查查看3dgs" in topic
    assert "看看那个大规模场景的论文" in topic
    assert "这篇有和与nerf的对比吗？" in topic
    # current 必须是最后一行
    assert topic.endswith("这篇有和与nerf的对比吗？")


def test_recall_topic_skips_non_user_roles():
    """Only user role messages count; assistant/tool are skipped."""
    history = [
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "real user msg"},
    ]
    topic = AgentLoop._build_recall_topic(None, history, "current", n=3)

    lines = topic.split("\n")
    assert lines == ["real user msg", "current"]


def test_recall_topic_caps_at_n():
    """Should only take the last N user messages."""
    history = [
        {"role": "user", "content": f"msg{i}"}
        for i in range(10)
    ]
    topic = AgentLoop._build_recall_topic(None, history, "current", n=3)

    lines = topic.split("\n")
    assert lines == ["msg7", "msg8", "msg9", "current"]


def test_recall_topic_empty_history_returns_current_only():
    """Empty history returns just the current message."""
    topic = AgentLoop._build_recall_topic(None, [], "only msg", n=3)
    assert topic == "only msg"


def test_recall_topic_history_with_no_user_messages():
    """If history has no user messages, returns just current."""
    history = [
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
    ]
    topic = AgentLoop._build_recall_topic(None, history, "current", n=3)
    assert topic == "current"


def test_recall_topic_handles_list_content():
    """Should handle messages where content is a list (multimodal)."""
    history = [
        {"role": "user", "content": [{"type": "text", "text": "multimodal user"}]},
    ]
    topic = AgentLoop._build_recall_topic(None, history, "current", n=3)

    assert "multimodal user" in topic
    assert topic.endswith("current")


def test_recall_topic_truncates_long_messages():
    """Very long user messages are truncated to keep embedding focused (200 chars cap)."""
    long_msg = "x" * 500
    history = [{"role": "user", "content": long_msg}]
    topic = AgentLoop._build_recall_topic(None, history, "current", n=3)

    lines = topic.split("\n")
    assert len(lines[0]) <= 200
    assert lines[1] == "current"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/unit/agent/test_topic_expansion.py -v`

Expected: 全部 7 条测试失败，因 `_build_recall_topic` 不存在（`AttributeError: type object 'AgentLoop' has no attribute '_build_recall_topic'`）。

- [ ] **Step 3: 在 `loop.py` 加 `_build_recall_topic` 方法**

打开 `backend/nanoresearch/agent/loop.py`。在 `AgentLoop` 类内合适位置（例如 `_check_pending_consolidation` 上面或下面）加入：

```python
    @staticmethod
    def _build_recall_topic(history: list[dict], current_msg: str, n: int = 3) -> str:
        """Build RAG recall query by concatenating recent user messages + current.

        The single-message topic was too narrow: 'this paper' alone cannot retrieve
        the consolidated history_entry that mentions 'CityGaussianV2'. By including
        the last N user turns, the embedding combines into a topic-aware query.

        Args:
            history: chronological message list; only role=='user' entries are used.
            current_msg: the current user input, always appended last.
            n: max number of historical user messages to include (default 3).

        Returns:
            Newline-separated string. Each line capped at 200 chars to avoid
            embedding being dominated by one long message.
        """
        MAX_LINE_CHARS = 200
        user_contents: list[str] = []
        for m in history:
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not isinstance(content, str):
                content = str(content)
            user_contents.append(content[:MAX_LINE_CHARS])
        recent = user_contents[-n:] if n > 0 else []
        recent.append(current_msg[:MAX_LINE_CHARS])
        return "\n".join(recent)
```

- [ ] **Step 4: 跑单测确认 `_build_recall_topic` 通过**

Run: `cd backend && python -m pytest tests/unit/agent/test_topic_expansion.py -v`

Expected: 全部 7 条 PASS。

- [ ] **Step 5: 把 `_build_recall_topic` 接入 `_process_message`**

回到 `backend/nanoresearch/agent/loop.py`，找到第 771-790 行：

```python
        history = session.get_history(max_messages=0)
        # Capture context assembly decisions once at run start; not updated on subsequent turns.
        _ctx_trace: dict = {}
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            topic=msg.content,
            tool_names=self.tools.tool_names,
            use_cache_blocks=self._use_cache_blocks,
            skill_names=skill_names,
            agent_id=agent_id,
            custom_persona=custom_persona,
            kb_bindings=kb_bindings,
            total_token_budget=_total_token_budget,
            memory_budget_ratio=_memory_budget_ratio,
            agents_registry=agents_registry,
            _trace_out=_ctx_trace,
        )
```

把 `topic=msg.content,` 改为 `topic=self._build_recall_topic(history, msg.content, n=3),`：

```python
        history = session.get_history(max_messages=0)
        # Capture context assembly decisions once at run start; not updated on subsequent turns.
        _ctx_trace: dict = {}
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            topic=self._build_recall_topic(history, msg.content, n=3),
            tool_names=self.tools.tool_names,
            use_cache_blocks=self._use_cache_blocks,
            skill_names=skill_names,
            agent_id=agent_id,
            custom_persona=custom_persona,
            kb_bindings=kb_bindings,
            total_token_budget=_total_token_budget,
            memory_budget_ratio=_memory_budget_ratio,
            agents_registry=agents_registry,
            _trace_out=_ctx_trace,
        )
```

注意：system message 路径（loop.py:699 附近）的 `topic=msg.content` 保持不变 —— system 路径不需要多轮指代消解。

- [ ] **Step 6: 跑相邻回归确认接入没破坏**

Run: `cd backend && python -m pytest tests/unit/session/ tests/unit/agent/ -v`

Expected: 全部 PASS。

- [ ] **Step 7: 提交**

```bash
git add backend/tests/unit/agent/test_topic_expansion.py backend/nanoresearch/agent/loop.py
git commit -m "feat(agent): expand RAG recall topic to last N user turns

build_history_context 之前用 topic=msg.content 当 RAG 召回 query。
但用户当前的代词输入（如 '这篇'）根本不含被指代的实体名（如
'CityGaussianV2'），导致 user_memory 中已写入的相关 history_entry
召回不到。

加 _build_recall_topic：拼最近 N=3 条 user message + 当前 msg，
\n 分隔，每条截到 200 字符避免 embedding 被长输入主导。
_process_message 调 build_messages 时 topic 改用扩展版本。

system message 路径保持 topic=msg.content（不涉及多轮指代）。"
```

---

### Task 4: Integration e2e test — 端到端验证三个修法协同工作

**Files:**
- Create: `backend/tests/integration/test_consolidation_anchor_e2e.py`

**Interfaces:**
- Consumes:
  - `AgentLoop._check_pending_consolidation` (修法 1)
  - `_CONSOLIDATION_SYSTEM_PROMPT` 含 RECENT_TOPICS (修法 2A)
  - `AgentLoop._build_recall_topic` (修法 2B)
  - `Session(key, messages, last_consolidated, updated_at)` from `nanoresearch.session.manager`
- Produces: e2e 验证场景通过，不产 fixtures 给后续 task。

- [ ] **Step 1: 写 e2e 测试**

写入 `backend/tests/integration/test_consolidation_anchor_e2e.py`：

```python
"""End-to-end verification: three fixes together resolve coreference after consolidation.

Scenario (mirrors 2026-06-28 production bug):
- Turn 1: user 谈 "3DGS" → assistant 提到 CityGaussianV2
- 模拟 session idle 31 分钟
- Turn 2: user 含代词 "这篇"
- 验证：
  (a) startup consolidation 触发但留 tail (修法 1)
  (b) 期间 LLM 抽取 RECENT_TOPICS 含 CityGaussianV2 (修法 2A)
  (c) build_messages 的 topic 包含 Turn 1 user message (修法 2B)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanoresearch.agent.loop import AgentLoop
from nanoresearch.agent.memory import MemoryStore
from nanoresearch.session.manager import Session


def _make_loop_stub(uid="test_uid"):
    stub = SimpleNamespace()
    stub._startup_consolidated = set()
    stub._uid = uid
    stub.memory_consolidator = MagicMock()
    stub.memory_consolidator.consolidate_messages = AsyncMock(return_value=True)
    stub.memory_consolidator.pick_consolidation_boundary = MagicMock()
    stub.sessions = MagicMock()
    stub.sessions.save = AsyncMock()
    return stub


async def test_e2e_idle_session_keeps_tail_for_next_turn(tmp_path: Path):
    """After 31min idle, startup consolidation triggers but leaves tail.

    Models Turn 1 (10 msgs about CityGaussianV2) → idle 31min → Turn 2 startup.
    Verifies that messages[boundary:] survive so the next get_history(0)
    still surfaces the recent dialogue to the outer Agent.
    """
    stub = _make_loop_stub()
    # 10 messages from Turn 1 mentioning CityGaussianV2 in the last asst msg
    session = Session(
        key="web:test-session",
        messages=[
            {"role": "user", "content": "查查看3dgs"},
            {"role": "assistant", "content": "3DGS 是 ..."},
            {"role": "user", "content": "看看那个大规模场景的论文"},
            {"role": "assistant", "content": "CityGaussianV2 用了 ..."},
            {"role": "user", "content": "再深入讲讲"},
            {"role": "assistant", "content": "实验设置如下 ..."},
            {"role": "user", "content": "结果怎么样"},
            {"role": "assistant", "content": "PSNR 26.5, SSIM 0.82"},
            {"role": "user", "content": "和 PGSR 比呢"},
            {"role": "assistant", "content": "PGSR 在结构精度更好但速度慢"},
        ],
        last_consolidated=0,
        updated_at=datetime.now() - timedelta(minutes=31),
    )
    # boundary at index 5 (preserves last 5 messages as tail)
    stub.memory_consolidator.pick_consolidation_boundary.return_value = (5, 100)

    await AgentLoop._check_pending_consolidation(stub, session)

    assert session.last_consolidated == 5
    tail = session.messages[session.last_consolidated:]
    assert len(tail) == 5
    # The CityGaussianV2 mention is in messages[3], which IS now consolidated;
    # but the recent dialogue tail (m5..m9) is preserved, including "PGSR" reference
    assert any("PGSR" in m.get("content", "") for m in tail)


def test_e2e_build_recall_topic_includes_turn1_user_messages():
    """The expanded topic given to build_messages must contain Turn 1's user inputs."""
    history = [
        {"role": "user", "content": "查查看3dgs"},
        {"role": "assistant", "content": "3DGS 是 ..."},
        {"role": "user", "content": "看看那个大规模场景的论文"},
        {"role": "assistant", "content": "CityGaussianV2 ..."},
        {"role": "user", "content": "再深入讲讲"},
    ]
    topic = AgentLoop._build_recall_topic(history, "这篇有和与nerf的对比吗？", n=3)

    # All recent user turns + current packed into the recall query
    assert "查查看3dgs" in topic
    assert "看看那个大规模场景的论文" in topic
    assert "再深入讲讲" in topic
    assert topic.endswith("这篇有和与nerf的对比吗？")
    # vs old behavior where topic = "这篇有和与nerf的对比吗？" alone
    assert len(topic) > len("这篇有和与nerf的对比吗？")


async def test_e2e_recent_topics_persists_through_consolidation(tmp_path: Path):
    """Full path: consolidate() with LLM returning RECENT_TOPICS → MEMORY.md contains it."""
    store = MemoryStore(workspace=tmp_path)

    memory_update_with_topics = (
        "# User Memory\n\n"
        "## FACTS\n- 用户偏好 Python\n\n"
        "## USER_PROFILE\n资深工程师\n\n"
        "## FOCUS_AREAS\n- AI Agent 架构\n\n"
        "## RECENT_TOPICS\n"
        "- CityGaussianV2 — 用户讨论的大规模场景重建方法，对比了 GauU-Scene/MatrixCity\n"
        "- PGSR — 同期对比的 GS 变体，结构精度更好但速度慢\n"
    )

    fake_response = SimpleNamespace(
        finish_reason="tool_calls",
        content="",
        has_tool_calls=True,
        tool_calls=[SimpleNamespace(arguments=json.dumps({
            "history_entry": "[2026-06-28 21:14] 用户讨论 CityGaussianV2 vs PGSR",
            "memory_update": memory_update_with_topics,
        }))],
    )

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=fake_response)

    ok = await store.consolidate(
        messages=[
            {"role": "user", "content": "讲讲 CityGaussianV2"},
            {"role": "assistant", "content": "CityGaussianV2 ..."},
        ],
        provider=provider,
        model="test-model",
    )

    assert ok is True
    written = store.read_long_term()
    # All three sections survived alongside the new one
    assert "## FACTS" in written
    assert "## USER_PROFILE" in written
    assert "## FOCUS_AREAS" in written
    assert "## RECENT_TOPICS" in written
    assert "CityGaussianV2" in written
    assert "PGSR" in written
```

- [ ] **Step 2: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/integration/test_consolidation_anchor_e2e.py -v`

Expected: 3 条测试全部 PASS。

- [ ] **Step 3: 跑全套 unit + integration 回归**

Run: `cd backend && python -m pytest tests/unit/session/ tests/unit/agent/ tests/integration/test_consolidation_anchor_e2e.py tests/integration/test_query_rewrite_e2e.py -v`

Expected: 全部 PASS。`test_query_rewrite_e2e.py` 必须仍通过（不能因 topic 扩展破坏 A 类 e2e）。

- [ ] **Step 4: 提交**

```bash
git add backend/tests/integration/test_consolidation_anchor_e2e.py
git commit -m "test(integration): consolidation anchor retention e2e

端到端模拟 2026-06-28 真实故障：Turn 1 谈 CityGaussianV2 → 31min idle
→ Turn 2 startup consolidation 触发但留 tail（修法 1）→ topic 扩展
包含历史 user message（修法 2B）→ consolidation 写入 RECENT_TOPICS
进 MEMORY.md（修法 2A）。

三个修法独立测过的同时，这里验证它们的协同效果。"
```

---

## Self-Review

**1. Spec coverage**

| Spec §  | Plan Task |
|---|---|
| §3.1 修法 1 (idle gate + tail protect) | Task 1 ✅ |
| §3.2 修法 2A (RECENT_TOPICS section) | Task 2 ✅ |
| §3.3 修法 2B (topic expansion) | Task 3 ✅ |
| §4 数据流（三层协同） | Task 4 ✅ |
| §5 components & files | Task 1-4 全覆盖 ✅ |
| §6 error handling | Task 1 Step 2 含 `test_consolidation_failure_does_not_advance_pointer` ✅；Task 3 `_build_recall_topic` 处理 list content / empty history ✅ |
| §7 configuration | Task 1 用 `STARTUP_CONSOLIDATION_IDLE_SECONDS` env var ✅；`tail_protect=5`、`n=3` 是函数参数 ✅ |
| §8 risks | R1 (RECENT_TOPICS 丢实体) prompt 加 "优先保留旧实体" 说明（Task 2 Step 3）；R2 (topic 过长) Task 3 加 200 字符截断；R3 (idle 误差) 主路径 `maybe_consolidate_by_tokens` 不动是兜底（Global Constraints）；R4 (跨 agent 串扰) 范围外 |
| §10 migration / rollback | 三个独立 commit，env var 可热回滚 ✅ |
| §11 verification | Task 4 e2e 测试模拟故障场景 ✅ |

**2. Placeholder scan**

- 无 TBD / TODO / "implement later"
- 所有代码块完整可执行
- 所有 pytest 命令含具体路径
- commit message 完整

**3. Type consistency**

- `_build_recall_topic(history: list[dict], current_msg: str, n: int = 3) -> str` — Task 3 定义、Task 4 调用一致
- `pick_consolidation_boundary(session, tokens_to_remove, tail_protect=5) -> tuple[int, int] | None` — Task 1 调用方式与 `memory.py:486` 当前签名一致
- `MemoryStore(workspace, knowledge_search=None, agent_id=None)` — Task 2 测试调用与 `memory.py:184` 一致
- `consolidate_messages(messages, agent_id=None, uid=None) -> bool` — Task 1 调用与 `memory.py:445` 签名一致
- `Session(key, messages, last_consolidated, updated_at, ...)` — 所有测试构造一致
