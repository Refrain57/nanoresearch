# Wiki LLM 词条（Phase 2 MVP）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 Phase-1 实体详情页顶部加一段按需生成、缓存持久化的 LLM 词条正文，正文带 `[^n]` 内联引用到该实体证据。

**Architecture:** 后端新增"取实体证据"查询 + `kg_entity_articles` 缓存表 + 同步生成服务（组装证据→LLM→回填引用）+ 2 个 GET/POST 端点；前端把聊天的 `[^n]` 渲染抽成共享件，实体页顶部复用它展示词条。无 faithfulness 打分。

**Tech Stack:** Python + SQLAlchemy async + FastAPI + AsyncOpenAI；Vue3 + Ant Design Vue + Pinia。

## Global Constraints

- **按需 + 缓存**：无缓存才生成；`evidence_hash` 变了 → stale。**不预生成/不 cron/不 async worker**（同步）。
- **grounding**：LLM 输入只喂该实体证据 + 事实；正文带 `[^n]`；**MVP 不做 faithfulness 打分/校验**（未来）。
- `[^n]` 渲染**复用**：抽共享件，聊天与词条共用（不复制逻辑）。
- 新表经 `Base.metadata.create_all`（database.py:49）自动建，**无 Alembic 迁移**。
- LLM 调用：`_resolve_rag_settings(uid, request)`（knowledge_router 已有）→ `AsyncOpenAI(base_url/api_key=settings.llm)` → `chat.completions.create`（镜像 worker.py:227-241）。
- 后端测试：真 PG（conftest `make_factory()` + `run()`），KG/kb 表自清（沿用 Phase-1 test_graph_repo 模式）。
- 前端 gate：`cd web && npm run build`（0 error）。
- 实体名 KG 里归一化小写；查询用 `_normalize`。citations 形状与聊天 `msg.citations` 对齐：`[{index:int, source:str, page, snippet:str}]`。

---

### Task 1: graph_repo `get_entity_evidence`

**Files:**
- Modify: `backend/nanoresearch/storage/repositories/graph_repo.py`（Query helpers 区追加）
- Test: `backend/tests/storage/test_graph_article.py`（新建）

**Interfaces:**
- Consumes: 模型 `KgEntity/KgEntityMention/KbChunk/KbDocument`；`_normalize`。
- Produces: `get_entity_evidence(kb_id: uuid.UUID, name: str, limit: int = 20) -> list[dict]` → `[{"chunk_id": str, "content": str, "source": str, "page": ...}]`（source=原文件名；按 chunk 出现去重；limit 上限）。

- [ ] **Step 1: 写失败测试** `backend/tests/storage/test_graph_article.py`

```python
"""GraphRepository entity-evidence + article cache tests (Wiki Phase 2). Real PG."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from nanoresearch.storage.repositories.graph_repo import GraphRepository
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
def clean_graph():
    conn = pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE TABLE kg_entity_articles, kg_triple_mentions, kg_entity_mentions, "
                "kg_triples, kg_entities, kb_chunks, kb_documents, knowledge_bases "
                "RESTART IDENTITY CASCADE"
            )
    finally:
        conn.close()


async def _seed(factory):
    from nanoresearch.storage.models import (
        KbChunk, KbDocument, KgEntity, KgEntityMention, KnowledgeBase,
    )
    kb_id = uuid.uuid4()
    d1 = uuid.uuid4()
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    e_gs = uuid.uuid4()
    async with factory() as db:
        db.add(KnowledgeBase(id=kb_id, uid="tester", name="KB", chroma_collection="c"))
        db.add_all([
            KbDocument(id=d1, kb_id=kb_id, filename="paperA.pdf", file_path="/tmp/a"),
            KbChunk(id=c1, kb_id=kb_id, document_id=d1, chunk_index=0, content="3dgs uses explicit points"),
            KbChunk(id=c2, kb_id=kb_id, document_id=d1, chunk_index=1, content="3dgs renders fast"),
            KgEntity(id=e_gs, kb_id=kb_id, name="3dgs", label="method"),
            KgEntityMention(entity_id=e_gs, chunk_id=c1, kb_id=kb_id),
            KgEntityMention(entity_id=e_gs, chunk_id=c2, kb_id=kb_id),
        ])
        await db.commit()
    return {"kb_id": kb_id}


def test_get_entity_evidence_returns_content_and_filename():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        ev = await repo.get_entity_evidence(s["kb_id"], "3DGS")
        assert len(ev) == 2
        contents = {e["content"] for e in ev}
        assert contents == {"3dgs uses explicit points", "3dgs renders fast"}
        assert all(e["source"] == "paperA.pdf" for e in ev)   # original filename, not path
        assert all("chunk_id" in e for e in ev)
    run(_())
```

- [ ] **Step 2: 跑测试确认失败** — `cd backend && python -m pytest tests/storage/test_graph_article.py -v` → FAIL（无 `get_entity_evidence`）。

- [ ] **Step 3: 实现**（graph_repo.py，追加到 Query helpers 区）

```python
    async def get_entity_evidence(self, kb_id: uuid.UUID, name: str, limit: int = 20) -> list[dict]:
        """Evidence chunks for an entity: content + original filename, for article generation."""
        from nanoresearch.storage.models import KbChunk, KbDocument
        norm = _normalize(name)
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk.id, KbChunk.content, KbChunk.chunk_metadata, KbDocument.filename)
                .join(KgEntityMention, KgEntityMention.chunk_id == KbChunk.id)
                .join(KgEntity, KgEntity.id == KgEntityMention.entity_id)
                .join(KbDocument, KbDocument.id == KbChunk.document_id)
                .where(KgEntity.kb_id == kb_id, KgEntity.name == norm)
                .distinct()
                .limit(limit)
            )
            out = []
            for r in result.all():
                out.append({
                    "chunk_id": str(r[0]),
                    "content": r[1] or "",
                    "page": (r[2] or {}).get("page"),
                    "source": r[3] or "",
                })
            return out
```

- [ ] **Step 4: 跑测试确认通过** — 同上命令 → PASS。

- [ ] **Step 5: Commit**
```bash
git add backend/nanoresearch/storage/repositories/graph_repo.py backend/tests/storage/test_graph_article.py
git commit -m "feat(graph): get_entity_evidence (content+filename) for wiki article generation"
```

---

### Task 2: `kg_entity_articles` 表 + 缓存 repo 方法

**Files:**
- Modify: `backend/nanoresearch/storage/models.py`（KG 表区追加模型）
- Modify: `backend/nanoresearch/storage/repositories/graph_repo.py`（追加 get/upsert）
- Test: `backend/tests/storage/test_graph_article.py`（追加）

**Interfaces:**
- Produces:
  - 模型 `KgEntityArticle`（表 `kg_entity_articles`）。
  - `get_article(kb_id, entity_name) -> KgEntityArticle | None`
  - `upsert_article(kb_id, entity_name, markdown, citations, evidence_hash, model) -> KgEntityArticle`

- [ ] **Step 1: 加模型**（models.py，紧接 `KgTripleMention` 之后）

```python
class KgEntityArticle(Base):
    __tablename__ = "kg_entity_articles"
    __table_args__ = (UniqueConstraint("kb_id", "entity_name", name="uq_kg_entity_article"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_name: Mapped[str] = mapped_column(String, nullable=False)
    markdown: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list] = mapped_column(JSONB, default=list)
    evidence_hash: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str | None] = mapped_column(String)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
```
（`_utcnow`、`JSONB`、`UniqueConstraint`、`Text` 等已在 models.py 顶部 import；照抄同文件其它模型的用法。）

- [ ] **Step 2: 写失败测试**（追加到 test_graph_article.py）

```python
def test_article_upsert_and_get():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        assert await repo.get_article(s["kb_id"], "3DGS") is None
        await repo.upsert_article(s["kb_id"], "3dgs", "正文[^1]", [{"index": 1, "source": "paperA.pdf", "snippet": "x"}], "hash1", "gpt-x")
        got = await repo.get_article(s["kb_id"], "3dgs")
        assert got is not None and got.markdown == "正文[^1]"
        assert got.evidence_hash == "hash1" and got.citations[0]["index"] == 1
        # upsert again → overwrite, single row
        await repo.upsert_article(s["kb_id"], "3dgs", "新正文", [], "hash2", "gpt-x")
        got2 = await repo.get_article(s["kb_id"], "3dgs")
        assert got2.markdown == "新正文" and got2.evidence_hash == "hash2"
    run(_())
```

- [ ] **Step 3: 跑确认失败** — `pytest tests/storage/test_graph_article.py::test_article_upsert_and_get -v` → FAIL。

- [ ] **Step 4: 实现 repo 方法**（graph_repo.py）

```python
    async def get_article(self, kb_id: uuid.UUID, entity_name: str):
        from nanoresearch.storage.models import KgEntityArticle
        norm = _normalize(entity_name)
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntityArticle).where(
                    KgEntityArticle.kb_id == kb_id, KgEntityArticle.entity_name == norm
                )
            )
            return result.scalar_one_or_none()

    async def upsert_article(self, kb_id: uuid.UUID, entity_name: str, markdown: str,
                             citations: list, evidence_hash: str, model: str | None):
        from nanoresearch.storage.models import KgEntityArticle
        norm = _normalize(entity_name)
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntityArticle).where(
                    KgEntityArticle.kb_id == kb_id, KgEntityArticle.entity_name == norm
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                row = KgEntityArticle(kb_id=kb_id, entity_name=norm)
                db.add(row)
            row.markdown = markdown
            row.citations = citations
            row.evidence_hash = evidence_hash
            row.model = model
            from datetime import datetime, timezone
            row.generated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(row)
            return row
```

- [ ] **Step 5: 跑确认通过** — `pytest tests/storage/test_graph_article.py -v` → PASS（3 项）。
  （注：新表由 conftest `create_tables()` 的 `create_all` 自动建；clean_graph 已含 `kg_entity_articles`。）

- [ ] **Step 6: Commit**
```bash
git add backend/nanoresearch/storage/models.py backend/nanoresearch/storage/repositories/graph_repo.py backend/tests/storage/test_graph_article.py
git commit -m "feat(graph): kg_entity_articles cache table + get/upsert_article"
```

---

### Task 3: 词条生成服务

**Files:**
- Create: `backend/nanoresearch/rag/wiki/article_generator.py`
- Create: `backend/nanoresearch/rag/wiki/__init__.py`（空）
- Test: `backend/tests/unit/rag/test_article_generator.py`（新建）

**Interfaces:**
- Consumes: `get_entity_evidence`（Task 1）、`get_entity_facts`（Phase 1）。
- Produces:
  - `build_article_prompt(name: str, facts: list[dict], evidence: list[dict]) -> tuple[str, str]` → `(system, user)`。纯函数，可单测。
  - `evidence_signature(evidence: list[dict]) -> str` → sha256(sorted chunk_ids)。纯函数。
  - `build_citations(evidence: list[dict]) -> list[dict]` → `[{index, source, page, snippet}]`（index 从 1，snippet=content 截断 300）。纯函数。
  - `async generate_article(llm_settings, name, facts, evidence) -> tuple[str, list[dict]]` → `(markdown, citations)`（调 LLM；`llm_settings` 即 `_resolve_rag_settings` 返回值）。

- [ ] **Step 1: 写失败测试**（纯函数部分，不打真 LLM）`backend/tests/unit/rag/test_article_generator.py`

```python
from nanoresearch.rag.wiki.article_generator import (
    build_article_prompt, evidence_signature, build_citations,
)


def test_evidence_signature_stable_and_order_independent():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "x"}]
    assert evidence_signature(a) == evidence_signature(b)
    assert evidence_signature(a) != evidence_signature([{"chunk_id": "z"}])


def test_build_citations_numbers_from_one_and_truncates():
    ev = [{"chunk_id": "x", "content": "c" * 400, "source": "p.pdf", "page": 2}]
    cites = build_citations(ev)
    assert cites[0]["index"] == 1 and cites[0]["source"] == "p.pdf" and cites[0]["page"] == 2
    assert len(cites[0]["snippet"]) <= 300


def test_build_article_prompt_includes_numbered_evidence_and_facts():
    system, user = build_article_prompt(
        "3dgs",
        [{"source": "3dgs", "label": "faster_than", "target": "nerf", "doc_count": 2}],
        [{"chunk_id": "x", "content": "explicit points", "source": "p.pdf"}],
    )
    assert "3dgs" in user
    assert "[1]" in user and "explicit points" in user   # numbered evidence
    assert "faster_than" in user                          # facts included
    assert "[^" in user                                   # instructs [^n] citation
```

- [ ] **Step 2: 跑确认失败** — `cd backend && python -m pytest tests/unit/rag/test_article_generator.py -v` → FAIL（模块不存在）。

- [ ] **Step 3: 实现** `backend/nanoresearch/rag/wiki/article_generator.py`

```python
"""Wiki entity article generation (Phase 2 MVP): grounded free synthesis with [^n]."""
from __future__ import annotations

import hashlib


def evidence_signature(evidence: list[dict]) -> str:
    ids = sorted(str(e.get("chunk_id", "")) for e in evidence)
    return hashlib.sha256(",".join(ids).encode()).hexdigest()


def build_citations(evidence: list[dict]) -> list[dict]:
    out = []
    for i, e in enumerate(evidence, start=1):
        out.append({
            "index": i,
            "source": e.get("source", ""),
            "page": e.get("page"),
            "snippet": (e.get("content", "") or "")[:300],
        })
    return out


def build_article_prompt(name: str, facts: list[dict], evidence: list[dict]) -> tuple[str, str]:
    system = "你是知识库词条编写助手。只依据给定证据编写，不使用外部知识，不编造。"
    fact_lines = "\n".join(
        f"- {f.get('source')} —{f.get('label')}→ {f.get('target')}" for f in facts
    ) or "（无结构化事实）"
    ev_lines = "\n".join(
        f"[{i}] {e.get('content','')}" for i, e in enumerate(evidence, start=1)
    ) or "（无证据）"
    user = (
        f"实体：{name}\n\n"
        f"已知事实：\n{fact_lines}\n\n"
        f"证据（编号）：\n{ev_lines}\n\n"
        "请为该实体写一段简洁的中文词条正文（markdown）。要求：\n"
        "- 只综合上述证据，不确定或无证据支撑的内容不要写；\n"
        "- 每处引用在句末标 [^n]，n 为对应证据编号；\n"
        "- 不要输出证据列表本身，只输出词条正文。"
    )
    return system, user


async def generate_article(llm_settings, name: str, facts: list[dict], evidence: list[dict]) -> tuple[str, list[dict]]:
    """Call the configured LLM once (non-streaming); return (markdown, citations)."""
    from openai import AsyncOpenAI
    from nanoresearch.config.loader import env_key_or_raise

    system, user = build_article_prompt(name, facts, evidence)
    llm_cfg = getattr(llm_settings, "llm", None)
    client = AsyncOpenAI(
        base_url=getattr(llm_cfg, "base_url", None) or "https://api.openai.com/v1",
        api_key=getattr(llm_cfg, "api_key", None) or env_key_or_raise("OPENAI_API_KEY", role="ingestion_llm"),
    )
    model = getattr(llm_cfg, "model", None) or "gpt-4o-mini"
    resp = await client.chat.completions.create(
        model=model, temperature=0.3,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    markdown = (resp.choices[0].message.content or "").strip()
    return markdown, build_citations(evidence)
```

- [ ] **Step 4: 跑确认通过** — 同 Step 1 命令 → PASS（3 项纯函数测试；`generate_article` 打真 LLM，不在单测覆盖）。

- [ ] **Step 5: Commit**
```bash
git add backend/nanoresearch/rag/wiki/ backend/tests/unit/rag/test_article_generator.py
git commit -m "feat(wiki): entity article generation service (prompt/citations/signature + LLM call)"
```

---

### Task 4: 2 个 API 端点

**Files:**
- Modify: `backend/nanoresearch/server/routers/knowledge_router.py`（Graph 区追加）

**Interfaces:**
- Consumes: `get_entity_evidence`/`get_entity_facts`/`get_article`/`upsert_article`（前面任务）、`generate_article`/`evidence_signature`（Task 3）、`_resolve_rag_settings`/`_get_kb_or_404`/`_graph_repo`（已有）。
- Produces:
  - `GET /api/knowledge/{kb_id}/graph/entities/{name}/article` → `{"article": {markdown, citations, model, generated_at, stale} | null}`
  - `POST /api/knowledge/{kb_id}/graph/entities/{name}/article` → `{"article": {...}}`（同步生成）

- [ ] **Step 1: 加 GET（读缓存 + stale 判定）**

```python
def _article_dict(row, stale: bool) -> dict:
    return {
        "markdown": row.markdown,
        "citations": row.citations or [],
        "model": row.model,
        "generated_at": row.generated_at.isoformat() if row.generated_at else None,
        "stale": stale,
    }


@router.get("/api/knowledge/{kb_id}/graph/entities/{name}/article")
async def get_entity_article(kb_id: str, name: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _graph_repo(request)
    kb_uuid = uuid.UUID(kb_id)
    row = await repo.get_article(kb_uuid, name)
    if row is None:
        return {"article": None}
    from nanoresearch.rag.wiki.article_generator import evidence_signature
    evidence = await repo.get_entity_evidence(kb_uuid, name)
    stale = evidence_signature(evidence) != row.evidence_hash
    return {"article": _article_dict(row, stale)}
```

- [ ] **Step 2: 加 POST（同步生成 + upsert）**

```python
@router.post("/api/knowledge/{kb_id}/graph/entities/{name}/article")
async def generate_entity_article(kb_id: str, name: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _graph_repo(request)
    kb_uuid = uuid.UUID(kb_id)
    if await repo.get_entity_summary(kb_uuid, name) is None:
        raise HTTPException(status_code=404, detail="entity not found")
    from nanoresearch.rag.wiki.article_generator import generate_article, evidence_signature
    evidence = await repo.get_entity_evidence(kb_uuid, name)
    facts = await repo.get_entity_facts(kb_uuid, name)
    settings = await _resolve_rag_settings(uid, request)
    markdown, citations = await generate_article(settings, name, facts, evidence)
    model = getattr(getattr(settings, "llm", None), "model", None)
    row = await repo.upsert_article(kb_uuid, name, markdown, citations, evidence_signature(evidence), model)
    return {"article": _article_dict(row, stale=False)}
```

- [ ] **Step 3: 语法检查** — `cd backend && python -c "import ast; ast.parse(open('nanoresearch/server/routers/knowledge_router.py',encoding='utf-8').read()); print('OK')"` → OK。（HTTP 冒烟需跑服务 + KG 已建，留 e2e。）

- [ ] **Step 4: Commit**
```bash
git add backend/nanoresearch/server/routers/knowledge_router.py
git commit -m "feat(api): wiki entity article endpoints (get cached+stale / generate)"
```

---

### Task 5: 抽共享 `[^n]` 渲染件

**Files:**
- Create: `web/src/components/CitationText.vue`
- Modify: `web/src/components/MessageList.vue`（改用共享件）
- Modify: `web/src/apis/knowledge.js`（加 article API）

**Interfaces:**
- Produces:
  - `<CitationText :text="markdown" :citations="citations" />` — 内部 `marked.parse` + `linkifyCitations`（按出现顺序编号、跳 code、有效索引过滤）+ 点击委托 popover（source/page/snippet，outside/Esc 关）。把 MessageList 现有的 `linkifyCitations`、`renderMd`、popover 状态与处理（`activeCite/citePos/onCiteClick/closeCite/onDocClick/onKey` + mounted/unmounted 监听）+ `.cite-ref`/`.cite-popover` CSS **移入**该组件。
  - `apis/knowledge.js`：`getEntityArticle(kbId, name)` = `apiGet('/api/knowledge/${kbId}/graph/entities/${encodeURIComponent(name)}/article')`；`generateEntityArticle(kbId, name)` = `apiPost(同URL, {})`。

- [ ] **Step 1: 建 CitationText.vue**——把 MessageList 里 `[^n]` 相关逻辑原样搬进来，props `text`(String)+`citations`(Array)，模板 `<div class="md-body" v-html="rendered" @click="onCite"></div>` + popover 元素；`rendered = computed(renderMd(text, citations))`。（读 MessageList 现有实现照搬，行为等价。）

- [ ] **Step 2: MessageList 改用**——assistant 答案处 `<CitationText :text="msgText(msg)" :citations="msg.citations" />`；删掉已移走的 `linkifyCitations/renderMd`/popover 状态/CSS（streaming 文本仍用原 `renderMd(streamingText, [])` 或直接 marked，无 citations）。import `CitationText`。

- [ ] **Step 3: 加 article API**（apis/knowledge.js，Knowledge Graph 区）
```javascript
export const getEntityArticle      = (kbId, name) => apiGet(`/api/knowledge/${kbId}/graph/entities/${encodeURIComponent(name)}/article`)
export const generateEntityArticle = (kbId, name) => apiPost(`/api/knowledge/${kbId}/graph/entities/${encodeURIComponent(name)}/article`, {})
```

- [ ] **Step 4: build** — `cd web && npm run build` → 0 error。手动确认聊天引用行为不变（e2e）。

- [ ] **Step 5: Commit**
```bash
git add web/src/components/CitationText.vue web/src/components/MessageList.vue web/src/apis/knowledge.js
git commit -m "refactor(web): extract shared CitationText ([^n]) from MessageList + article API"
```

---

### Task 6: 实体页顶部词条区

**Files:**
- Modify: `web/src/views/KnowledgeDetailView.vue`

**Interfaces:**
- Consumes: `getEntityArticle`/`generateEntityArticle`（Task 5）、`CitationText`（Task 5）、Phase-1 的 `entityDetail`/`selectEntity`。

- [ ] **Step 1: import + 状态 + 方法**（`<script setup>`）
```javascript
import CitationText from '@/components/CitationText.vue'
import { getEntityArticle, generateEntityArticle } from '@/apis/knowledge'  // 合并进现有 knowledge import

const article = ref(null)        // {markdown, citations, model, generated_at, stale}
const articleLoading = ref(false)

async function loadArticle(name) {
  article.value = null
  try { const r = await getEntityArticle(kbId, name); article.value = r.article } catch (e) {}
}
async function genArticle() {
  if (!entityDetail.value) return
  articleLoading.value = true
  try {
    const r = await generateEntityArticle(kbId, entityDetail.value.name)
    article.value = r.article
  } catch (e) { message.error('生成词条失败') }
  finally { articleLoading.value = false }
}
```

- [ ] **Step 2: 在 `selectEntity` 里联动加载**——`selectEntity(name)` 成功设 `entityDetail` 后，调 `loadArticle(name)`（找到 Phase-1 的 `selectEntity`，在其 try 成功分支末尾加 `await loadArticle(name)`）。

- [ ] **Step 3: 模板**——在实体详情 `<template v-if="entityDetail">` 内、**头部之后、「事实」之前**插入词条区：
```html
<div class="wiki-article">
  <a-spin :spinning="articleLoading">
    <template v-if="article">
      <CitationText :text="article.markdown" :citations="article.citations" />
      <div class="wiki-article-meta">
        <a-tag v-if="article.stale" color="orange">来源已更新</a-tag>
        <a-button size="small" type="link" @click="genArticle">
          {{ article.stale ? '重新生成' : '重新生成词条' }}
        </a-button>
      </div>
    </template>
    <a-button v-else type="dashed" block @click="genArticle">生成词条</a-button>
  </a-spin>
</div>
```

- [ ] **Step 4: CSS**（`<style scoped>` 末尾）
```css
.wiki-article { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #f0f0f0; }
.wiki-article-meta { margin-top: 8px; display: flex; align-items: center; gap: 8px; }
```

- [ ] **Step 5: build** — `cd web && npm run build` → 0 error。

- [ ] **Step 6: Commit**
```bash
git add web/src/views/KnowledgeDetailView.vue
git commit -m "feat(web): LLM entity article at top of wiki entity detail (generate/render/regenerate)"
```

---

## 手动 e2e（全部完成后；需服务在跑 + KG 已建 + 已配 LLM）
1. 进「知识图谱/Wiki」→ 点实体 → 顶部显「生成词条」按钮。
2. 点生成 → 转圈几秒 → 出现词条正文，句末带 `[^n]`，点开 popover 见来源(真文件名)/页/片段。
3. 重开该实体 → 直接读缓存(秒出)。
4. 该实体所属文档重新入库/建图后 → 词条区显「来源已更新」+ 重新生成可用。
5. 聊天里的引用 `[^n]` 行为不变（回归 CitationText 抽取）。

## Self-Review
- **Spec coverage**：按需+缓存(T2表+T4 GET/POST)、自由合成(T3)、`[^n]`(T3 prompt + T5 复用渲染)、grounding=只喂证据+facts(T3/T4)+下方 Phase-1 面板(既有)、stale=evidence_hash(T3 signature/T2/T4)、展示实体页顶(T6)、复用抽件(T5)、无打分(全程未做)。✅
- **Placeholder scan**：无 TODO；每代码步给完整代码；T5/T6 前端"照搬 MessageList 现有实现"是明确的抽取动作 + 给了组件 API/props/插入位，非空洞。✅
- **Type consistency**：citations `[{index,source,page,snippet}]` 贯穿 build_citations→upsert→API→CitationText；`get_entity_evidence` 返回含 chunk_id/content/source/page 与 evidence_signature/build_citations 输入一致；`article` 形状(markdown/citations/model/generated_at/stale)贯穿 API→前端。✅
- **无迁移**：新表靠 create_all（database.py:49 + conftest create_tables），clean_graph 含 kg_entity_articles。✅
- **LLM 调用**：镜像 worker.py:227-241（AsyncOpenAI + settings.llm），role 走 _resolve_rag_settings。✅
