# Consolidation 压缩修订 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修对多轮对话压缩（consolidation）的触发/留尾/时点，并修复压缩产物被 confidence 阈值静默丢弃的根因，使背靠背对话不再被频繁压缩、近期原文锚点不丢、归档摘要真正入库可召回。

**Architecture:** 四个独立可 revert 的 commit。C1 把 `updated_at` 统一成 aware-UTC（idle gate 的前置）；C2 重写 startup 计数触发器 T1（idle gate + 轮次计数 + 留尾 + end_idx 推进），其决策逻辑抽成纯函数 `plan_startup_consolidation` 便于 TDD；C3 把 consolidation 写入 user_memory 的 confidence 从 0.6/0.5 提到 0.7 让写入真正落地；C4 把 token 触发器 T2 的写死参数（tail_protect、target ratio）接到共享配置常量。

**Tech Stack:** Python 3.11+, pytest（`asyncio_mode=auto`），SQLAlchemy（PG，`DateTime(timezone=True)`），Redis（session 缓存），Chroma（user_memory 向量库）。

**Source spec:** `docs/superpowers/specs/2026-06-28-consolidation-compaction-design.md`

## Global Constraints

- 所有测试从 `backend/` 目录运行：`cd backend && python -m pytest <path> -v`。
- `backend/tests/conftest.py` 有 session-scoped autouse `setup_database`，**测试需要测试库 Postgres 在线**（默认 `host=localhost port=5432 dbname=nanoresearch_test user=postgres password=123456`，或设 `TEST_DATABASE_DSN`）。否则 collection 即报错。
- 不改 `knowledge_search.py:153` 的全局 0.7 阈值（extractor 路径依赖它当质量闸）。修在写入点。
- 不碰本轮"不做"项：`_CONSOLIDATION_SYSTEM_PROMPT`（memory.py:39-121）、召回 query 扩展（context.py:346 / loop.py:779）、RECENT_TOPICS、per-uid Chroma、MCP uid 透传。
- 推荐值（已拍板，verbatim）：`MIN_PENDING_TURNS=2`、`TAIL_PROTECT=8`、`IDLE=1800s`、`TOKEN_TARGET_RATIO=0.5`、consolidation/raw_archive confidence 均 `0.7`。**不**新增 `last_consolidation_at` 字段。
- 每个 commit 独立可 revert；commit 顺序 C1 → C2（C2 依赖 C1 的 helper），C3、C4 可任意序。
- 提交信息用祈使式英文前缀（`fix:`/`refactor:`/`feat:`），与现有 git log 一致。

---

## File Structure

- `backend/nanoresearch/utils/helpers.py` — **新增** 两个时区 helper：`utcnow_aware()`、`as_aware_utc(dt)`。被 manager / loop / memory 共用。
- `backend/nanoresearch/session/manager.py` — `updated_at`/`created_at` 写入与读取统一 aware-UTC。
- `backend/nanoresearch/agent/memory.py` — **新增** `plan_startup_consolidation` 纯函数 + 4 个配置常量；Lua meta 时间戳 aware-UTC；confidence 0.6/0.5→0.7；T2 接共享 tail_protect/ratio。
- `backend/nanoresearch/agent/loop.py` — 2 个 T1 配置常量；`_check_pending_consolidation` 改为薄包装调用纯函数；`_save_turn` 的 `updated_at` aware-UTC。
- `backend/tests/unit/session/test_updated_at_tz.py` — **新增**（C1）。
- `backend/tests/unit/agent/__init__.py` — **新增** 空包。
- `backend/tests/unit/agent/test_plan_startup_consolidation.py` — **新增**（C2）。
- `backend/tests/unit/agent/test_consolidation_confidence.py` — **新增**（C3）。
- `backend/tests/unit/agent/test_token_consolidation_params.py` — **新增**（C4）。

---

## Task 1 (C1): `updated_at` 统一为 aware-UTC + 时区 helper

**Files:**
- Modify: `backend/nanoresearch/utils/helpers.py`（追加两个函数）
- Modify: `backend/nanoresearch/session/manager.py:30-31,43,92,113,158-159,188,195,266-267`
- Modify: `backend/nanoresearch/agent/memory.py:631`（Lua meta 时间戳）
- Modify: `backend/nanoresearch/agent/loop.py:899`（`_save_turn` 的 updated_at）
- Test: `backend/tests/unit/session/test_updated_at_tz.py`

**Interfaces:**
- Produces:
  - `utcnow_aware() -> datetime` — 返回 `datetime.now(timezone.utc)`（tz-aware）。
  - `as_aware_utc(dt: datetime) -> datetime` — naive 输入按 UTC 兜底并告警；aware 输入转成 UTC。供 C2 的 idle gate 消费。

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/session/test_updated_at_tz.py`：

```python
"""C1: updated_at must round-trip as tz-aware UTC so the idle gate math is correct."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nanoresearch.utils.helpers import as_aware_utc, utcnow_aware


def test_utcnow_aware_is_timezone_aware():
    now = utcnow_aware()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_as_aware_utc_passes_through_aware():
    aware = datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)
    assert as_aware_utc(aware) == aware


def test_as_aware_utc_converts_offset_to_utc():
    plus8 = datetime(2026, 6, 29, 20, 0, tzinfo=timezone(timedelta(hours=8)))
    assert as_aware_utc(plus8) == datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)


def test_as_aware_utc_treats_naive_as_utc():
    naive = datetime(2026, 6, 29, 12, 0)
    assert as_aware_utc(naive) == datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)


def test_idle_delta_is_consistent_regardless_of_source_tz():
    """A 5-minute-old session must read as ~5 minutes idle whether the stored
    timestamp came back naive-UTC (Lua path) or aware-UTC (Redis path)."""
    now = utcnow_aware()
    five_min_ago_naive = (now - timedelta(minutes=5)).replace(tzinfo=None)
    five_min_ago_aware = now - timedelta(minutes=5)

    delta_naive = now - as_aware_utc(five_min_ago_naive)
    delta_aware = now - as_aware_utc(five_min_ago_aware)

    assert abs(delta_naive.total_seconds() - 300) < 1
    assert abs(delta_aware.total_seconds() - 300) < 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/unit/session/test_updated_at_tz.py -v`
Expected: FAIL（`ImportError: cannot import name 'as_aware_utc'`）。

- [ ] **Step 3: 加 helper**

在 `backend/nanoresearch/utils/helpers.py` 顶部确保有 `from datetime import datetime, timezone` 和 `from loguru import logger`（缺则补 import），并追加：

```python
def utcnow_aware() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def as_aware_utc(dt: datetime) -> datetime:
    """Normalize any datetime to aware-UTC.

    Naive values are assumed to be UTC (with a warning) — this is the
    fallback for legacy rows written before the timezone unification.
    """
    if dt.tzinfo is None:
        logger.warning("as_aware_utc: received naive datetime {!r}, assuming UTC", dt)
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/unit/session/test_updated_at_tz.py -v`
Expected: PASS（5 passed）。

- [ ] **Step 5: 把 session 持久层的时间戳写/读统一到 aware-UTC**

`backend/nanoresearch/session/manager.py` 顶部 import 改为 `from datetime import datetime, timezone`，并 `from nanoresearch.utils.helpers import ensure_dir, safe_filename, utcnow_aware, as_aware_utc`。逐处替换：

- 行 30-31 dataclass 字段默认：
  ```python
  created_at: datetime = field(default_factory=utcnow_aware)
  updated_at: datetime = field(default_factory=utcnow_aware)
  ```
- 行 43 `add_message`：`self.updated_at = utcnow_aware()`
- 行 92 `clear`：`self.updated_at = utcnow_aware()`
- 行 113 `retain_recent_legal_suffix`：`self.updated_at = utcnow_aware()`
- 行 158-159 `_redis_load`：
  ```python
  created_at = as_aware_utc(datetime.fromisoformat(meta["created_at"])) if meta.get("created_at") else utcnow_aware()
  updated_at = as_aware_utc(datetime.fromisoformat(meta["updated_at"])) if meta.get("updated_at") else utcnow_aware()
  ```
- 行 188 `_redis_save`：`ts = utcnow_aware().isoformat()`
- 行 195 `_redis_save` mapping 里 `"created_at": session.created_at.isoformat(),` 保持（值已 aware）。
- 行 266-267 `_db_load`：去掉 `.replace(tzinfo=None)`，改 `as_aware_utc`：
  ```python
  created_at=as_aware_utc(conv.created_at) if conv.created_at else utcnow_aware(),
  updated_at=as_aware_utc(conv.updated_at) if conv.updated_at else utcnow_aware(),
  ```

- [ ] **Step 6: 修 Lua meta 时间戳与 `_save_turn`**

- `backend/nanoresearch/agent/memory.py:631`：`datetime.utcnow().isoformat()` → `datetime.now(datetime_timezone_utc_placeholder).isoformat()`。具体做法：该文件已 `from datetime import datetime`，改成 `from datetime import datetime, timezone`，行 631 改为 `datetime.now(timezone.utc).isoformat()`。
- `backend/nanoresearch/agent/loop.py:899`：`_save_turn` 末尾 `session.updated_at = datetime.now()`。loop.py 顶部没有直接 import datetime（`_save_turn` 内 `from datetime import datetime`）。在 loop.py 顶部加 `from nanoresearch.utils.helpers import utcnow_aware`（与现有 helpers import 合并），把行 899 改为 `session.updated_at = utcnow_aware()`。

- [ ] **Step 7: 写持久层往返测试**

在 `backend/tests/unit/session/test_updated_at_tz.py` 末尾追加（复用现有 fake-Redis 模式）：

```python
from pathlib import Path
from unittest.mock import patch

from nanoresearch.session.manager import Session, SessionManager
# 直接复用已有 fake redis
from tests.unit.session.test_redis_roundtrip import _FakeRedis


async def test_redis_roundtrip_updated_at_is_aware_utc(tmp_path: Path):
    fake = _FakeRedis()
    manager = SessionManager(workspace=tmp_path)
    session = Session(key="web:tz-1", messages=[{"role": "user", "content": "hi"}])

    with patch("nanoresearch.bus.redis_client.get_redis", return_value=fake):
        await manager._redis_save(session)
        loaded = await manager._redis_load("web:tz-1")

    assert loaded is not None
    assert loaded.updated_at.tzinfo is not None
    delta = utcnow_aware() - loaded.updated_at
    assert -1 < delta.total_seconds() < 5
```

- [ ] **Step 8: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/unit/session/test_updated_at_tz.py tests/unit/session/test_redis_roundtrip.py -v`
Expected: PASS（新文件全过 + 旧 roundtrip 不回归）。

- [ ] **Step 9: 反向扫残留 naive 写入**

Run: `cd backend && grep -n "datetime.now()\|datetime.utcnow()" nanoresearch/session/manager.py nanoresearch/agent/memory.py`
Expected: manager.py 里与 `updated_at`/`created_at` 相关的 `datetime.now()` 已全部替换（剩余命中应仅为无关用途，如无则空）；memory.py:631 不再有 `utcnow()`。逐条确认无遗漏。

- [ ] **Step 10: Commit**

```bash
git add backend/nanoresearch/utils/helpers.py backend/nanoresearch/session/manager.py backend/nanoresearch/agent/memory.py backend/nanoresearch/agent/loop.py backend/tests/unit/session/test_updated_at_tz.py
git commit -m "fix(session): unify updated_at to aware-UTC for idle-gate correctness"
```

---

## Task 2 (C2): 重写 startup 触发器 T1（idle gate + 轮次计数 + 留尾 + end_idx）

**Files:**
- Modify: `backend/nanoresearch/agent/memory.py`（顶部加 `import os` + 常量；加纯函数 `plan_startup_consolidation`）
- Modify: `backend/nanoresearch/agent/loop.py`（顶部加 2 常量；重写 `_check_pending_consolidation` `loop.py:530-560`）
- Test: `backend/tests/unit/agent/__init__.py`、`backend/tests/unit/agent/test_plan_startup_consolidation.py`

**Interfaces:**
- Consumes: `as_aware_utc`（Task 1）；`MemoryConsolidator.pick_consolidation_boundary(session, tokens_to_remove, tail_protect) -> tuple[int,int] | None`（已存在，memory.py:486）。
- Produces:
  - `plan_startup_consolidation(session, *, now_utc, idle_threshold, min_turns, tail_protect, pick_boundary) -> tuple[int, int] | None` — 返回待压缩区间 `(start, end_idx)`（`start == session.last_consolidated`），不该压时返回 `None`。`pick_boundary` 是签名同 `pick_consolidation_boundary` 的可调用，便于注入。
  - 常量 `CONSOLIDATION_TAIL_PROTECT`（memory.py）、`STARTUP_CONSOLIDATION_IDLE_SECONDS`/`STARTUP_MIN_PENDING_TURNS`（loop.py）。

- [ ] **Step 1: 写纯函数失败测试**

创建 `backend/tests/unit/agent/__init__.py`（空文件）。创建 `backend/tests/unit/agent/test_plan_startup_consolidation.py`：

```python
"""C2: startup consolidation planning — idle gate, turn counting, tail protect."""
from __future__ import annotations

from datetime import timedelta

from nanoresearch.agent.memory import plan_startup_consolidation
from nanoresearch.session.manager import Session
from nanoresearch.utils.helpers import utcnow_aware

IDLE = timedelta(minutes=30)


def _msgs(roles: list[str]) -> list[dict]:
    return [{"role": r, "content": f"{r}-{i}"} for i, r in enumerate(roles)]


def _fake_pick(end_idx: int | None):
    """Return a pick_boundary stub that yields (end_idx, 0) or None."""
    def _pick(session, tokens_to_remove, tail_protect):  # noqa: ARG001
        return None if end_idx is None else (end_idx, 0)
    return _pick


def test_skips_when_session_active_within_idle_window():
    """Back-to-back turn (5 min idle) must NOT consolidate — kills frequent compaction."""
    session = Session(key="web:1", messages=_msgs(["user", "assistant"] * 4),
                      last_consolidated=0, updated_at=utcnow_aware() - timedelta(minutes=5))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(2))
    assert result is None


def test_counts_turns_not_message_rows():
    """One tool-using turn = 8 rows but 1 user message → below min_turns=2 → skip.

    This reproduces problem 1: row-count fired on every turn; turn-count must not."""
    rows = _msgs(["user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"])
    session = Session(key="web:2", messages=rows, last_consolidated=0,
                      updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(4))
    assert result is None


def test_consolidates_when_idle_and_enough_turns():
    session = Session(key="web:3", messages=_msgs(["user", "assistant"] * 6),
                      last_consolidated=0, updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(4))
    assert result == (0, 4)


def test_returns_none_when_boundary_not_found():
    session = Session(key="web:4", messages=_msgs(["user", "assistant"] * 6),
                      last_consolidated=0, updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(None))
    assert result is None


def test_returns_none_when_boundary_at_or_before_start():
    session = Session(key="web:5", messages=_msgs(["user", "assistant"] * 6),
                      last_consolidated=4, updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=_fake_pick(4))
    assert result is None


def test_real_boundary_picker_protects_tail():
    """End-to-end with the real pick_consolidation_boundary: tail is protected."""
    from nanoresearch.agent.memory import MemoryConsolidator
    consolidator = MemoryConsolidator.__new__(MemoryConsolidator)  # no heavy init needed
    rows = _msgs(["user", "assistant"] * 6)  # 12 rows
    session = Session(key="web:6", messages=rows, last_consolidated=0,
                      updated_at=utcnow_aware() - timedelta(minutes=45))
    result = plan_startup_consolidation(
        session, now_utc=utcnow_aware(), idle_threshold=IDLE,
        min_turns=2, tail_protect=8, pick_boundary=consolidator.pick_consolidation_boundary)
    assert result is not None
    start, end_idx = result
    assert start == 0
    assert end_idx <= len(rows) - 8  # tail of 8 preserved
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/unit/agent/test_plan_startup_consolidation.py -v`
Expected: FAIL（`ImportError: cannot import name 'plan_startup_consolidation'`）。

- [ ] **Step 3: 实现纯函数 + 常量（memory.py）**

`backend/nanoresearch/agent/memory.py`：顶部 import 区加 `import os`；在 `from loguru import logger` 之后、`MemoryConsolidator` 之前加常量：

```python
CONSOLIDATION_TAIL_PROTECT = int(os.environ.get("CONSOLIDATION_TAIL_PROTECT", "8"))
TOKEN_CONSOLIDATION_TARGET_RATIO = float(os.environ.get("TOKEN_CONSOLIDATION_TARGET_RATIO", "0.5"))
CONSOLIDATION_SUMMARY_CONFIDENCE = float(os.environ.get("CONSOLIDATION_SUMMARY_CONFIDENCE", "0.7"))
```

在 `MemoryConsolidator` 类定义之后（文件末尾）加纯函数：

```python
def plan_startup_consolidation(
    session,
    *,
    now_utc,
    idle_threshold,
    min_turns,
    tail_protect,
    pick_boundary,
):
    """Decide the startup-consolidation chunk for a session, or None.

    Returns (start, end_idx) where start == session.last_consolidated, or None
    when the session is too active (idle gate), has too few turns, or no safe
    tail-protected boundary exists.
    """
    from nanoresearch.utils.helpers import as_aware_utc

    if now_utc - as_aware_utc(session.updated_at) < idle_threshold:
        return None

    start = session.last_consolidated
    pending = session.messages[start:]
    pending_turns = sum(1 for m in pending if m.get("role") == "user")
    if pending_turns < min_turns:
        return None

    boundary = pick_boundary(session, tokens_to_remove=1, tail_protect=tail_protect)
    if boundary is None:
        return None
    end_idx, _ = boundary
    if end_idx <= start:
        return None
    return (start, end_idx)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/unit/agent/test_plan_startup_consolidation.py -v`
Expected: PASS（6 passed）。

- [ ] **Step 5: 接线 `_check_pending_consolidation`（loop.py）**

`backend/nanoresearch/agent/loop.py`：顶部（`_EVAL_*` 常量附近）加：

```python
from datetime import timedelta
STARTUP_CONSOLIDATION_IDLE_SECONDS = int(os.environ.get("STARTUP_CONSOLIDATION_IDLE_SECONDS", "1800"))
STARTUP_MIN_PENDING_TURNS = int(os.environ.get("STARTUP_MIN_PENDING_TURNS", "2"))
```

把 `loop.py:530-560` 整个 `_check_pending_consolidation` 替换为：

```python
    async def _check_pending_consolidation(self, session: Session, agent_id: str | None = None) -> None:
        """Consolidate a previous, now-idle session's tail-protected backlog.

        Only fires when the session has been idle past the threshold and has
        accumulated enough *turns* (not message rows). Always leaves the recent
        tail uncompacted so coreference anchors survive into the next turn.
        """
        if session.key in self._startup_consolidated:
            return

        from nanoresearch.agent.memory import (
            CONSOLIDATION_TAIL_PROTECT,
            plan_startup_consolidation,
        )
        from nanoresearch.utils.helpers import utcnow_aware

        plan = plan_startup_consolidation(
            session,
            now_utc=utcnow_aware(),
            idle_threshold=timedelta(seconds=STARTUP_CONSOLIDATION_IDLE_SECONDS),
            min_turns=STARTUP_MIN_PENDING_TURNS,
            tail_protect=CONSOLIDATION_TAIL_PROTECT,
            pick_boundary=self.memory_consolidator.pick_consolidation_boundary,
        )
        if plan is None:
            self._startup_consolidated.add(session.key)
            return

        start, end_idx = plan
        chunk = session.messages[start:end_idx]
        logger.info(
            "Startup consolidation for {}: {} msgs (tail protected, range {}:{})",
            session.key, len(chunk), start, end_idx,
        )
        success = await self.memory_consolidator.consolidate_messages(
            chunk, agent_id=agent_id, uid=self._uid
        )
        if success:
            session.last_consolidated = end_idx
            await self.sessions.save(session)
            self._startup_consolidated.add(session.key)
        else:
            logger.warning("Startup consolidation failed, will retry on token pressure")
```

- [ ] **Step 6: 静态自检**

Run: `cd backend && python -c "import nanoresearch.agent.loop"`
Expected: 无 ImportError（确认 `timedelta` import、常量、函数引用都解析）。

- [ ] **Step 7: 跑相关测试确认无回归**

Run: `cd backend && python -m pytest tests/unit/agent/ tests/unit/session/ -v`
Expected: PASS。

- [ ] **Step 8: Commit**

```bash
git add backend/nanoresearch/agent/memory.py backend/nanoresearch/agent/loop.py backend/tests/unit/agent/__init__.py backend/tests/unit/agent/test_plan_startup_consolidation.py
git commit -m "fix(consolidation): startup trigger counts turns, idle-gates, protects tail"
```

---

## Task 3 (C3): confidence gate —— 压缩产物从 0.6/0.5 提到 0.7 真正入库

**Files:**
- Modify: `backend/nanoresearch/agent/memory.py:355`（consolidation_summary 0.6→常量）、`:393`（raw_archive 0.5→常量）
- Test: `backend/tests/unit/agent/test_consolidation_confidence.py`

**Interfaces:**
- Consumes: `CONSOLIDATION_SUMMARY_CONFIDENCE`（Task 2 已定义于 memory.py，默认 0.7）；`KnowledgeSearch.write_user_memory_sync(memories, uid)` 内部过滤 `confidence >= 0.7`（knowledge_search.py:153，不改）。

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/agent/test_consolidation_confidence.py`：

```python
"""C3: consolidation summaries must clear the 0.7 user_memory gate (not be dropped)."""
from __future__ import annotations

from pathlib import Path

import pytest

from nanoresearch.agent.memory import MemoryStore


class _CapturingKnowledge:
    """Captures memories passed to write_user_memory_sync."""
    def __init__(self):
        self.written: list[dict] = []

    def write_user_memory_sync(self, memories, uid=None):  # noqa: ARG002
        self.written.extend(memories)
        return (len(memories), 0)


class _FakeToolCall:
    def __init__(self, args):
        self.arguments = args


class _FakeResponse:
    def __init__(self):
        self.finish_reason = "tool_calls"
        self.content = ""
        self.has_tool_calls = True
        self.tool_calls = [_FakeToolCall(
            {"history_entry": "[2026-06-29 12:00] discussed CityGaussianV2 vs NeRF",
             "memory_update": "# User Memory\n## FACTS\n- prefers Python"}
        )]


class _FakeProvider:
    async def chat_with_retry(self, **kwargs):  # noqa: ARG002
        return _FakeResponse()


def _real_gate_passes(confidence: float) -> bool:
    """Mirror knowledge_search.py:153 — items below 0.7 are dropped."""
    return confidence >= 0.7


async def test_consolidation_summary_clears_07_gate(tmp_path: Path):
    knowledge = _CapturingKnowledge()
    store = MemoryStore(workspace=tmp_path, knowledge_search=knowledge)

    ok = await store.consolidate(
        messages=[{"role": "user", "content": "tell me about CityGaussianV2"}],
        provider=_FakeProvider(), model="fake-model", uid="u1",
    )

    assert ok is True
    assert knowledge.written, "summary must be written, not silently dropped"
    summary = next(m for m in knowledge.written if m["type"] == "consolidation_summary")
    assert summary["confidence"] >= 0.7
    assert _real_gate_passes(summary["confidence"]), "must survive the real 0.7 filter"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/unit/agent/test_consolidation_confidence.py -v`
Expected: FAIL（`assert 0.6 >= 0.7`）。

- [ ] **Step 3: 改写入点 confidence**

`backend/nanoresearch/agent/memory.py`：
- 行 355 `"confidence": 0.6,` → `"confidence": CONSOLIDATION_SUMMARY_CONFIDENCE,`
- 行 393 `"confidence": 0.5,` → `"confidence": CONSOLIDATION_SUMMARY_CONFIDENCE,`

（常量已在 Task 2 于 memory.py 顶部定义。若单独 cherry-pick 本 commit，确保该常量存在。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/unit/agent/test_consolidation_confidence.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/agent/memory.py backend/tests/unit/agent/test_consolidation_confidence.py
git commit -m "fix(memory): raise consolidation/raw-archive confidence to 0.7 so summaries persist"
```

---

## Task 4 (C4): token 触发器 T2 接共享 tail_protect / target ratio

**Files:**
- Modify: `backend/nanoresearch/agent/memory.py:490`（`pick_consolidation_boundary` 默认 `tail_protect`）、`:562`（target 比例）、`:590`（T2 调用显式传 tail_protect）
- Test: `backend/tests/unit/agent/test_token_consolidation_params.py`

**Interfaces:**
- Consumes: `CONSOLIDATION_TAIL_PROTECT`、`TOKEN_CONSOLIDATION_TARGET_RATIO`（Task 2 定义）。

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/agent/test_token_consolidation_params.py`：

```python
"""C4: token trigger T2 uses the shared tail_protect / target-ratio config."""
from __future__ import annotations

import nanoresearch.agent.memory as memory_mod
from nanoresearch.agent.memory import MemoryConsolidator, CONSOLIDATION_TAIL_PROTECT
from nanoresearch.session.manager import Session


def test_pick_boundary_default_tail_protect_is_shared_constant():
    consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
    rows = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(40)]
    session = Session(key="web:t2", messages=rows, last_consolidated=0)

    # Tail of CONSOLIDATION_TAIL_PROTECT messages must never be selected.
    boundary = consolidator.pick_consolidation_boundary(session, tokens_to_remove=1)
    assert boundary is not None
    end_idx, _ = boundary
    assert end_idx <= len(rows) - CONSOLIDATION_TAIL_PROTECT


def test_target_ratio_constant_is_used(monkeypatch):
    """maybe_consolidate_by_tokens must derive `target` from the ratio constant."""
    import inspect
    src = inspect.getsource(MemoryConsolidator.maybe_consolidate_by_tokens)
    assert "TOKEN_CONSOLIDATION_TARGET_RATIO" in src
    assert "// 2" not in src
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/unit/agent/test_token_consolidation_params.py -v`
Expected: FAIL（默认仍是 5；`// 2` 仍在源码）。

- [ ] **Step 3: 接线 T2 参数**

`backend/nanoresearch/agent/memory.py`：
- 行 490 `pick_consolidation_boundary` 签名默认：`tail_protect: int = 5,` → `tail_protect: int = CONSOLIDATION_TAIL_PROTECT,`（删掉行内注释 `# 保护最近 5 条消息` 或更新为 8）。
- 行 562 `target = budget // 2` → `target = int(budget * TOKEN_CONSOLIDATION_TARGET_RATIO)`。
- 行 590 T2 内调用补显式 tail_protect：
  `boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))`
  → `boundary = self.pick_consolidation_boundary(session, max(1, estimated - target), tail_protect=CONSOLIDATION_TAIL_PROTECT)`

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/unit/agent/test_token_consolidation_params.py -v`
Expected: PASS。

- [ ] **Step 5: 全量回归**

Run: `cd backend && python -m pytest tests/unit/agent/ tests/unit/session/ -v`
Expected: PASS（C1-C4 全部测试）。

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/agent/memory.py backend/tests/unit/agent/test_token_consolidation_params.py
git commit -m "refactor(consolidation): wire token trigger to shared tail-protect/target-ratio config"
```

---

## Self-Review

**Spec coverage:**
- §3.1 修法 A（T1 idle/轮次/留尾/end_idx）→ Task 2 ✓
- §3.2 修法 B（confidence 0.6→0.7，raw_archive 0.5→0.7）→ Task 3 ✓（含 (a)(b) 结论：改写入点不改全局闸）
- §3.3 updated_at 时区前置（manager.py:188/159/267、memory.py:631、gate helper、负 delta 防护、三种输入单测）→ Task 1 ✓
- §6 推荐值（MIN_TURNS=2/TAIL=8/IDLE=1800/ratio=0.5/conf=0.7/不加字段）→ 常量默认值逐一对应 ✓
- §8 四 commit 独立可 revert + 配置点 → Task 1-4 各一 commit；常量全 env 可覆盖 ✓
- §7 "不做"项 → 未触及 prompt/召回/RECENT_TOPICS ✓
- §10 验证（idle 三时区一致、轮次计数复现问题1、留尾复现问题3/4、confidence 复现形态三）→ 各 Task 测试覆盖 ✓

**Placeholder scan:** 无 TBD/TODO/"add error handling"；每个 code step 给完整代码。Task 1 Step 6 的 `datetime_timezone_utc_placeholder` 已在同 step 文字明确解释为"import timezone 后写 `datetime.now(timezone.utc)`"，非占位符遗留。

**Type consistency:** `plan_startup_consolidation` 在 Task 2 定义、Task 2 Step 5 调用，参数名（`now_utc/idle_threshold/min_turns/tail_protect/pick_boundary`）与返回 `(start, end_idx) | None` 一致；`utcnow_aware`/`as_aware_utc` 在 Task 1 定义、Task 1/2 消费签名一致；`CONSOLIDATION_TAIL_PROTECT`/`CONSOLIDATION_SUMMARY_CONFIDENCE`/`TOKEN_CONSOLIDATION_TARGET_RATIO` 在 Task 2 定义，Task 3/4 消费名一致。

**Cross-commit 依赖说明:** 常量定义集中在 Task 2（C2）。C3、C4 消费这些常量，故若严格按 commit 顺序 revert C2 而保留 C3/C4 会缺常量——但 C2→C3→C4 的正常推进顺序无此问题；revert 时按逆序（C4→C3→C2）即安全。已在 Global Constraints 注明顺序。
