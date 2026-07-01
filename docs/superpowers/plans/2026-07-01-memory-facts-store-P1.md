# 画像 Provenance Store (P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the "consolidation 整篇覆写 MEMORY.md" pollution path with a provenance-tagged structured fact store updated by incremental diff, projecting to MEMORY.md one-way.

**Architecture:** New `memory_facts` PG table holds per-fact provenance (source/derived_from/edited_by/edited_at/active). A pure diff engine (`agent/memory_facts.py`) computes add/remove against the LLM's profile output — never wholesale overwrite, and never removes `source=manual` facts. `MemoryStore.consolidate` applies the diff via a repository, then renders MEMORY.md as a one-way projection of active facts. Injection (`context.py`) is untouched — it keeps reading the projected MEMORY.md file. Any failure in the new path falls back to the legacy overwrite so the live system never breaks.

**Tech Stack:** Python 3.12, SQLAlchemy 2.0 async (`Mapped`/`mapped_column`), PostgreSQL, pytest 9 (asyncio_mode=auto; DB tests use the `run()` SelectorEventLoop pattern).

## Global Constraints

- SQLAlchemy models use `Mapped[...]` + `mapped_column(...)`, `Base` from `nanoresearch.storage.database`.
- DB tests: import `truncate_all, make_factory` from `tests.conftest`; run coroutines via a local `run()` with `asyncio.SelectorEventLoop` (asyncpg/Windows). Add `memory_facts` cleanup in an autouse fixture.
- Pure-logic tests need no infra; DB tests need PG (`nanoresearch_test`) up.
- Frequent commits, one deliverable per task. No changes to `context.py` in P1.
- Scope: P1 only. Events (P2), conversation summaries (P3), migration (P4) are out of scope — do NOT touch `ConversationKnowledgeExtractor`, `user_memory`, or `build_history_context` here.

---

### Task 1: Pure diff engine + projection (`agent/memory_facts.py`)

**Files:**
- Create: `backend/nanoresearch/agent/memory_facts.py`
- Test: `backend/tests/unit/agent/test_memory_facts.py`

**Interfaces:**
- Produces: `Fact` dataclass (`text, section, source, id, uid, derived_from, confidence, edited_by, edited_at, active`); `ProfileDiff(add: list[tuple[str,str]], remove_ids: list[str])`; `normalize(text)->str`; `parse_memory_md(md)->list[tuple[str,str]]`; `render_memory_md(list[Fact])->str`; `compute_profile_diff(current: list[Fact], new_lines: list[tuple[str,str]])->ProfileDiff`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/agent/test_memory_facts.py
from nanoresearch.agent.memory_facts import (
    Fact, ProfileDiff, normalize, parse_memory_md, render_memory_md, compute_profile_diff,
)

def test_normalize_strips_bullet_and_casefolds():
    assert normalize("-  Prefers  Python ") == normalize("prefers python")

def test_parse_memory_md_sections_and_bullets():
    md = "# User Memory\n\n## FACTS\n- 偏好 Python\n- 用 Git\n\n## USER_PROFILE\n资深工程师。\n\n## FOCUS_AREAS\n- RAG\n"
    got = parse_memory_md(md)
    assert ("facts", "偏好 Python") in got
    assert ("facts", "用 Git") in got
    assert ("user_profile", "资深工程师。") in got
    assert ("focus_areas", "RAG") in got

def test_render_roundtrips_active_only():
    facts = [
        Fact(text="偏好 Python", section="facts"),
        Fact(text="过时", section="facts", active=False),
        Fact(text="RAG", section="focus_areas"),
    ]
    out = render_memory_md(facts)
    assert "偏好 Python" in out and "RAG" in out
    assert "过时" not in out
    assert "## FACTS" in out and "## FOCUS_AREAS" in out

def test_compute_diff_adds_new_lines():
    cur = [Fact(id="1", text="偏好 Python", section="facts", source="extracted")]
    new = [("facts", "偏好 Python"), ("facts", "喜欢 TDD")]
    diff = compute_profile_diff(cur, new)
    assert ("facts", "喜欢 TDD") in diff.add
    assert diff.remove_ids == []

def test_compute_diff_removes_absent_extracted():
    cur = [Fact(id="1", text="旧偏好", section="facts", source="extracted")]
    diff = compute_profile_diff(cur, [("facts", "新偏好")])
    assert diff.remove_ids == ["1"]
    assert ("facts", "新偏好") in diff.add

def test_compute_diff_never_removes_manual():
    cur = [Fact(id="m1", text="人工写的", section="facts", source="manual")]
    diff = compute_profile_diff(cur, [("facts", "别的")])  # manual absent from new
    assert "m1" not in diff.remove_ids

def test_compute_diff_dedups_new_and_existing():
    cur = [Fact(id="1", text="偏好 Python", section="facts", source="extracted")]
    new = [("facts", "偏好 Python"), ("facts", "偏好  python")]  # dup of existing + self-dup
    diff = compute_profile_diff(cur, new)
    assert diff.add == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/unit/agent/test_memory_facts.py -q`
Expected: FAIL (ModuleNotFoundError: nanoresearch.agent.memory_facts)

- [ ] **Step 3: Write minimal implementation**

```python
# backend/nanoresearch/agent/memory_facts.py
"""Structured 画像 (profile) facts + incremental-diff projection (P1 of memory-layering).

Profile = provenance-tagged facts (source: extracted|manual). Consolidation applies an
incremental diff (never wholesale overwrite); manual facts are never auto-removed.
MEMORY.md is a one-way rendered projection of the active facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SECTIONS = ("facts", "user_profile", "focus_areas")
_SECTION_TITLES = {"facts": "FACTS", "user_profile": "USER_PROFILE", "focus_areas": "FOCUS_AREAS"}
_TITLE_TO_SECTION = {
    "FACTS": "facts",
    "USER_PROFILE": "user_profile", "USER PROFILE": "user_profile",
    "FOCUS_AREAS": "focus_areas", "FOCUS AREAS": "focus_areas",
}


@dataclass
class Fact:
    text: str
    section: str
    source: str = "extracted"          # extracted | manual
    id: str | None = None
    uid: str | None = None
    derived_from: list[str] = field(default_factory=list)
    confidence: float | None = None
    edited_by: str | None = None
    edited_at: str | None = None
    active: bool = True


@dataclass
class ProfileDiff:
    add: list[tuple[str, str]] = field(default_factory=list)   # (section, text)
    remove_ids: list[str] = field(default_factory=list)        # Fact.id to deactivate


def normalize(text: str) -> str:
    t = text.strip().lstrip("-*").strip()
    t = re.sub(r"\s+", " ", t)
    return t.casefold()


def parse_memory_md(md_text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    section: str | None = None
    for raw in (md_text or "").splitlines():
        line = raw.rstrip()
        h = re.match(r"^#{1,6}\s*(.+?)\s*$", line)
        if h:
            section = _TITLE_TO_SECTION.get(h.group(1).strip().upper())
            continue
        if section is None:
            continue
        b = re.match(r"^\s*[-*]\s+(.+?)\s*$", line)
        text = b.group(1).strip() if b else line.strip()
        if text:
            out.append((section, text))
    return out


def render_memory_md(facts: list[Fact]) -> str:
    active = [f for f in facts if f.active]
    lines = ["# User Memory", ""]
    for sec in SECTIONS:
        sec_facts = [f for f in active if f.section == sec]
        if not sec_facts:
            continue
        lines.append(f"## {_SECTION_TITLES[sec]}")
        lines.extend(f"- {f.text}" for f in sec_facts)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compute_profile_diff(current: list[Fact], new_lines: list[tuple[str, str]]) -> ProfileDiff:
    active = [f for f in current if f.active]
    cur_keys = {(f.section, normalize(f.text)): f for f in active}
    new_keys = {(sec, normalize(txt)) for sec, txt in new_lines}
    diff = ProfileDiff()
    seen: set[tuple[str, str]] = set()
    for sec, txt in new_lines:
        key = (sec, normalize(txt))
        if key in seen:
            continue
        seen.add(key)
        if key not in cur_keys:
            diff.add.append((sec, txt))
    for key, fact in cur_keys.items():
        if key not in new_keys and fact.source == "extracted" and fact.id:
            diff.remove_ids.append(fact.id)
    return diff
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/unit/agent/test_memory_facts.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/agent/memory_facts.py backend/tests/unit/agent/test_memory_facts.py
git commit -m "feat(memory): pure profile-fact diff engine + MEMORY.md projection (P1 t1)"
```

---

### Task 2: `memory_facts` table + repository

**Files:**
- Modify: `backend/nanoresearch/storage/models.py` (add `MemoryFact`; ensure `Float` imported)
- Create: `backend/nanoresearch/storage/repositories/memory_facts_repo.py`
- Test: `backend/tests/storage/test_memory_facts_repo.py`

**Interfaces:**
- Consumes: `Fact` from Task 1.
- Produces: `MemoryFact` ORM model (`memory_facts`); `MemoryFactsRepository(session_factory)` with `list_active(uid)->list[Fact]`, `insert_extracted(uid, section, text, derived_from=None, confidence=None)->Fact`, `insert_manual(uid, section, text, edited_by)->Fact`, `deactivate(fact_id: str)->None`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/storage/test_memory_facts_repo.py
import asyncio
import pytest
from nanoresearch.storage.repositories.memory_facts_repo import MemoryFactsRepository
from tests.conftest import make_factory, pg_conn


def run(coro):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.fixture(autouse=True)
def clean_memory_facts():
    conn = pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE memory_facts RESTART IDENTITY CASCADE")
    finally:
        conn.close()


def test_insert_and_list_active():
    async def _():
        repo = MemoryFactsRepository(make_factory())
        f = await repo.insert_extracted("u1", "facts", "偏好 Python", confidence=0.9)
        assert f.id and f.source == "extracted" and f.active
        rows = await repo.list_active("u1")
        assert [r.text for r in rows] == ["偏好 Python"]
    run(_())


def test_deactivate_hides_from_active():
    async def _():
        repo = MemoryFactsRepository(make_factory())
        f = await repo.insert_extracted("u1", "facts", "旧")
        await repo.deactivate(f.id)
        assert await repo.list_active("u1") == []
    run(_())


def test_manual_carries_audit_and_uid_scoped():
    async def _():
        repo = MemoryFactsRepository(make_factory())
        m = await repo.insert_manual("u1", "facts", "人工", edited_by="u1")
        assert m.source == "manual" and m.edited_by == "u1" and m.edited_at
        assert await repo.list_active("u2") == []  # uid isolation
    run(_())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/storage/test_memory_facts_repo.py -q`
Expected: FAIL (ImportError / relation "memory_facts" does not exist)

- [ ] **Step 3: Write minimal implementation**

Add to `backend/nanoresearch/storage/models.py` (ensure `Float` is in the `from sqlalchemy import ...` line):

```python
class MemoryFact(Base):
    __tablename__ = "memory_facts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uid: Mapped[str] = mapped_column(String, index=True, nullable=False)
    section: Mapped[str] = mapped_column(String, nullable=False)      # facts|user_profile|focus_areas
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String, default="extracted")  # extracted|manual
    derived_from: Mapped[list] = mapped_column(JSONB, default=list)   # event ids (P2)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    edited_by: Mapped[str | None] = mapped_column(String, nullable=True)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
```

Create `backend/nanoresearch/storage/repositories/memory_facts_repo.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanoresearch.agent.memory_facts import Fact
from nanoresearch.storage.models import MemoryFact


def _to_fact(row: MemoryFact) -> Fact:
    return Fact(
        id=str(row.id), uid=row.uid, section=row.section, text=row.text,
        source=row.source, derived_from=list(row.derived_from or []),
        confidence=row.confidence, edited_by=row.edited_by,
        edited_at=row.edited_at.isoformat() if row.edited_at else None,
        active=row.active,
    )


class MemoryFactsRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def list_active(self, uid: str) -> list[Fact]:
        async with self._factory() as db:
            res = await db.execute(
                select(MemoryFact)
                .where(MemoryFact.uid == uid, MemoryFact.active.is_(True))
                .order_by(MemoryFact.created_at)
            )
            return [_to_fact(r) for r in res.scalars().all()]

    async def insert_extracted(self, uid, section, text, derived_from=None, confidence=None) -> Fact:
        async with self._factory() as db:
            row = MemoryFact(uid=uid, section=section, text=text, source="extracted",
                             derived_from=derived_from or [], confidence=confidence)
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return _to_fact(row)

    async def insert_manual(self, uid, section, text, edited_by) -> Fact:
        async with self._factory() as db:
            row = MemoryFact(uid=uid, section=section, text=text, source="manual",
                             edited_by=edited_by, edited_at=datetime.now(timezone.utc))
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return _to_fact(row)

    async def deactivate(self, fact_id: str) -> None:
        async with self._factory() as db:
            await db.execute(
                sa_update(MemoryFact).where(MemoryFact.id == uuid.UUID(fact_id)).values(active=False)
            )
            await db.commit()
```

Recreate test tables (the session fixture only runs once; new table needs creating):

Run: `cd backend && .venv/Scripts/python.exe -c "from tests.conftest import create_tables; create_tables()"`

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/storage/test_memory_facts_repo.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/storage/models.py backend/nanoresearch/storage/repositories/memory_facts_repo.py backend/tests/storage/test_memory_facts_repo.py
git commit -m "feat(memory): memory_facts table + repository (P1 t2)"
```

---

### Task 3: Wire consolidation to diff-apply + projection (with legacy fallback)

**Files:**
- Modify: `backend/nanoresearch/agent/memory.py` (`MemoryStore.__init__` add `session_factory`; replace wholesale `write_long_term(update)` with `_apply_profile_update`; `MemoryConsolidator._get_store` passes `session_factory`)
- Test: `backend/tests/unit/agent/test_profile_apply.py`

**Interfaces:**
- Consumes: `compute_profile_diff`, `parse_memory_md`, `render_memory_md`, `Fact` (Task 1); `MemoryFactsRepository` (Task 2).
- Produces: `MemoryStore._apply_profile_update(update_md: str, uid: str|None) -> None` (async) — applies diff to store + writes projection; on any error or missing `session_factory`/`uid`, falls back to legacy `write_long_term(update_md)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/agent/test_profile_apply.py
import asyncio
import pytest
from nanoresearch.agent.memory import MemoryStore
from nanoresearch.storage.repositories.memory_facts_repo import MemoryFactsRepository
from tests.conftest import make_factory, pg_conn


def run(coro):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.fixture(autouse=True)
def clean_memory_facts():
    conn = pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE memory_facts RESTART IDENTITY CASCADE")
    finally:
        conn.close()


def test_apply_populates_store_and_projects(tmp_path):
    async def _():
        factory = make_factory()
        store = MemoryStore(tmp_path, session_factory=factory)
        md = "# User Memory\n\n## FACTS\n- 偏好 Python\n- 用 Git\n"
        await store._apply_profile_update(md, uid="u1")
        rows = await MemoryFactsRepository(factory).list_active("u1")
        assert {r.text for r in rows} == {"偏好 Python", "用 Git"}
        projected = store.read_long_term()
        assert "偏好 Python" in projected and "用 Git" in projected
    run(_())


def test_apply_protects_manual_fact(tmp_path):
    async def _():
        factory = make_factory()
        repo = MemoryFactsRepository(factory)
        await repo.insert_manual("u1", "facts", "人工事实", edited_by="u1")
        store = MemoryStore(tmp_path, session_factory=factory)
        # LLM output omits the manual fact entirely
        await store._apply_profile_update("# User Memory\n\n## FACTS\n- 新事实\n", uid="u1")
        texts = {r.text for r in await repo.list_active("u1")}
        assert "人工事实" in texts   # protected, not removed
        assert "新事实" in texts     # new extracted added
    run(_())


def test_apply_falls_back_without_factory(tmp_path):
    async def _():
        store = MemoryStore(tmp_path, session_factory=None)
        await store._apply_profile_update("# User Memory\n\n## FACTS\n- x\n", uid="u1")
        assert "x" in store.read_long_term()  # legacy overwrite path
    run(_())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/unit/agent/test_profile_apply.py -q`
Expected: FAIL (MemoryStore.__init__ has no session_factory / no _apply_profile_update)

- [ ] **Step 3: Write minimal implementation**

In `memory.py`, update `MemoryStore.__init__` signature and add the method. Add imports at top:

```python
from nanoresearch.agent.memory_facts import (
    compute_profile_diff, parse_memory_md, render_memory_md,
)
```

Change `__init__`:

```python
    def __init__(self, workspace: Path, knowledge_search: Any = None, agent_id: str | None = None,
                 session_factory: Any = None):
        if agent_id:
            self.memory_dir = ensure_dir(workspace / "agents" / agent_id / "memory")
        else:
            self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self._consecutive_failures = 0
        self._cached_hash: str | None = None
        self._knowledge_search = knowledge_search
        self._session_factory = session_factory
```

Add method (near `write_long_term`):

```python
    async def _apply_profile_update(self, update_md: str, uid: str | None) -> None:
        """Apply the LLM profile output as an incremental diff to the memory_facts store,
        then render MEMORY.md as a one-way projection. Never wholesale-overwrites the store;
        never removes source=manual facts. On any error or missing store context, falls back
        to the legacy behaviour of writing update_md straight to MEMORY.md."""
        if not self._session_factory or not uid:
            self.write_long_term(update_md)
            return
        try:
            from nanoresearch.storage.repositories.memory_facts_repo import MemoryFactsRepository
            repo = MemoryFactsRepository(self._session_factory)
            current = await repo.list_active(uid)
            new_lines = parse_memory_md(update_md)
            diff = compute_profile_diff(current, new_lines)
            for section, text in diff.add:
                await repo.insert_extracted(uid, section, text)
            for fid in diff.remove_ids:
                await repo.deactivate(fid)
            active = await repo.list_active(uid)
            self.write_long_term(render_memory_md(active))
        except Exception:
            logger.exception("profile store apply failed; falling back to legacy overwrite")
            self.write_long_term(update_md)
```

Replace the wholesale-overwrite block inside `consolidate` — change:

```python
            update = _ensure_text(update)
            if update != current_memory:
                self.write_long_term(update)
```

to:

```python
            update = _ensure_text(update)
            await self._apply_profile_update(update, uid)
```

In `MemoryConsolidator._get_store`, thread the session factory:

```python
    def _get_store(self, agent_id: str | None = None) -> MemoryStore:
        factory = getattr(self.sessions, "_factory", None)
        return MemoryStore(self._workspace, knowledge_search=self._knowledge_search,
                           agent_id=agent_id, session_factory=factory)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/unit/agent/test_profile_apply.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/agent/memory.py backend/tests/unit/agent/test_profile_apply.py
git commit -m "feat(memory): consolidation writes profile via diff+projection, legacy fallback (P1 t3)"
```

---

### Task 4: Regression guard — existing memory tests still pass

**Files:**
- Test: run the existing agent/memory + repository suites.

- [ ] **Step 1: Run the memory-adjacent suites**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/unit/agent tests/storage tests/test_repositories.py -q`
Expected: PASS (no regressions; new tables created via session fixture)

- [ ] **Step 2: If a test references the removed wholesale-overwrite behaviour, update it to assert the projection instead** (show the exact edit in the commit).

- [ ] **Step 3: Commit any test fixups**

```bash
git add -A && git commit -m "test(memory): align existing suites with profile store (P1 t4)"
```

---

## Self-Review

- **Spec coverage (P1 slice):** memory_facts store schema w/ provenance (Task 2) ✓; incremental diff never overwrites (Task 1+3) ✓; manual protection (Task 1 `compute_profile_diff` + Task 3 test) ✓; MEMORY.md one-way projection (Task 1 render + Task 3 write) ✓; injection untouched reads projection ✓. Deferred (P2–P4): events, conv summaries, migration, manual-edit UI, risk-grading/human-review — correctly excluded.
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type consistency:** `Fact`/`ProfileDiff` names and fields identical across Tasks 1–3; repo returns `Fact`; `_apply_profile_update(update_md, uid)` signature stable.
- **Risk:** live path (`consolidate`) wrapped in try/except → legacy fallback; `context.py` untouched; new table additive.
