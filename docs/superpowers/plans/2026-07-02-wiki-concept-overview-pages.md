# Wiki 概念页 + 总览页（Phase 3）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在知识图谱/Wiki tab 加"概念页"(用户输入主题→RAG 检索→grounded 词条)和"总览页"(全库一张,依 KG 结构),复用 Phase 2 的 article 内核。

**Architecture:** 扩展 `article_generator`(抽 `_complete` + concept/overview prompt);概念页复用 query-test 的 HybridSearch 检索取证据;总览页用 `get_stats` 骨架;都存进 Phase 2 的 `kg_entity_articles`(key 命名空间);前端左栏分 总览/概念/实体 三区,复用 `CitationText`。

**Tech Stack:** Python + FastAPI + SQLAlchemy async + AsyncOpenAI + 现有 RAG(HybridSearch);Vue3 + Ant Design Vue。

## Global Constraints

- 基于 **Phase 2 分支**(`feat/wiki-llm-entity-pages`);复用 `article_generator.py` / `CitationText.vue` / `kg_entity_articles` / `_article_dict` / `_resolve_rag_settings`。
- **存储零迁移**:复用 `kg_entity_articles`,`entity_name` 列当通用 key —— 实体=名;**概念=`concept::` + `_normalize(topic)`**;**总览=`__overview__`**。`UNIQUE(kb_id, entity_name)` 隔离。
- 概念 grounding=RAG 检索到的 chunk + `[^n]`;总览 grounding=KG top 实体+关系。**不打分**(沿用 Phase 2)。
- citations 形状 `[{index:int, source, page, snippet}]`(与 Phase 2/聊天一致,复用 `build_citations` / `CitationText`)。
- 按需 + 缓存 + stale(evidence_hash;概念=检索 chunk 签名,总览=骨架签名)。同步生成(复用 Phase 2 LLM 调用)。
- 后端测试:真 PG(conftest `make_factory()` + `run()`);KG/kb/`kg_entity_articles` 自清(沿用 `test_graph_article.py` 的 clean_graph)。纯函数测试无需 PG。
- 前端 gate:`cd web && npm run build`(0 error)。
- 不做:faithfulness 打分、对比页、`[[wiki-links]]`、cron/预生成、子领域总览、概念自动派生。

---

### Task 1: article_generator 扩展（_complete + concept/overview prompt + generate + signature）

**Files:**
- Modify: `backend/nanoresearch/rag/wiki/article_generator.py`
- Test: `backend/tests/unit/rag/test_article_generator.py`（追加）

**Interfaces:**
- Consumes: 现有 `build_citations` / `evidence_signature`。
- Produces:
  - `_complete(llm_settings, system: str, user: str) -> str`（抽出的 LLM 调用核心）
  - `build_concept_prompt(topic: str, evidence: list[dict]) -> tuple[str,str]`
  - `build_overview_prompt(top_entities: list[dict], facts: list[dict]) -> tuple[str,str]`
  - `overview_signature(top_entities: list[dict], facts: list[dict]) -> str`
  - `async generate_concept_article(llm_settings, topic, evidence) -> tuple[str, list[dict]]`
  - `async generate_overview_article(llm_settings, top_entities, facts) -> tuple[str, list[dict]]`

- [ ] **Step 1: 写失败测试**（追加到 `test_article_generator.py`）

```python
from nanoresearch.rag.wiki.article_generator import (
    build_concept_prompt, build_overview_prompt, overview_signature,
)


def test_build_concept_prompt_has_topic_numbered_evidence_and_citation_instr():
    system, user = build_concept_prompt(
        "体渲染",
        [{"chunk_id": "x", "content": "volume rendering integrates radiance", "source": "p.pdf"}],
    )
    assert "体渲染" in user
    assert "[1]" in user and "volume rendering" in user
    assert "[^" in user               # instructs [^n]
    assert "不" in system              # grounding guard (只依据/不编造)


def test_build_overview_prompt_lists_top_entities_and_relations():
    system, user = build_overview_prompt(
        [{"name": "3dgs", "mentions": 12}, {"name": "nerf", "mentions": 9}],
        [{"source": "3dgs", "label": "faster_than", "target": "nerf"}],
    )
    assert "3dgs" in user and "nerf" in user
    assert "faster_than" in user
    assert "导览" in user or "总览" in user


def test_overview_signature_stable_and_sensitive():
    a = overview_signature([{"name": "3dgs"}], [{"source": "3dgs", "label": "x", "target": "nerf"}])
    b = overview_signature([{"name": "3dgs"}], [{"source": "3dgs", "label": "x", "target": "nerf"}])
    c = overview_signature([{"name": "nerf"}], [])
    assert a == b and a != c
```

- [ ] **Step 2: 跑确认失败** — `cd backend && python -m pytest tests/unit/rag/test_article_generator.py -v` → FAIL（函数不存在）。

- [ ] **Step 3: 实现**（`article_generator.py`）—— 先把现有 `generate_article` 里的 LLM 调用抽成 `_complete`,`generate_article` 改用它;再加 concept/overview。追加/改写:

```python
async def _complete(llm_settings, system: str, user: str) -> str:
    """Single non-streaming LLM completion (shared by all wiki generators)."""
    from openai import AsyncOpenAI
    from nanoresearch.config.loader import env_key_or_raise
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
    return (resp.choices[0].message.content or "").strip()


def build_concept_prompt(topic: str, evidence: list[dict]) -> tuple[str, str]:
    system = "你是知识库词条编写助手。只依据给定的检索证据编写，不使用外部知识，不编造。"
    ev_lines = "\n".join(
        f"[{i}] {e.get('content','')}" for i, e in enumerate(evidence, start=1)
    ) or "（无证据）"
    user = (
        f"主题：{topic}\n\n"
        f"检索到的证据（编号）：\n{ev_lines}\n\n"
        "请围绕该主题写一段简洁的中文词条正文（markdown）。要求：\n"
        "- 只综合上述检索证据，不确定或无证据支撑的不要写；\n"
        "- 每处引用在句末标 [^n]，n 为对应证据编号；\n"
        "- 不要输出证据列表本身，只输出词条正文。"
    )
    return system, user


def build_overview_prompt(top_entities: list[dict], facts: list[dict]) -> tuple[str, str]:
    system = "你是知识库导览编写助手。只依据给定的实体与关系结构编写，不编造库中没有的内容。"
    ent_lines = "\n".join(
        f"- {e.get('name')}（被提及 {e.get('mentions', 0)} 次）" for e in top_entities
    ) or "（无实体）"
    rel_lines = "\n".join(
        f"- {f.get('source')} —{f.get('label')}→ {f.get('target')}" for f in facts
    ) or "（无关系）"
    user = (
        f"本知识库的主要实体：\n{ent_lines}\n\n"
        f"实体间关系：\n{rel_lines}\n\n"
        "请写一段中文总览/导览（markdown）：介绍本库有哪些主要主题、它们之间怎么关联。要求：\n"
        "- 只依据上面列出的实体与关系，不要编造未列出的内容；\n"
        "- 面向初次了解本库的读者，结构清晰。"
    )
    return system, user


def overview_signature(top_entities: list[dict], facts: list[dict]) -> str:
    ents = sorted(str(e.get("name", "")) for e in top_entities)
    rels = sorted(f"{f.get('source')}|{f.get('label')}|{f.get('target')}" for f in facts)
    return hashlib.sha256(("E:" + ",".join(ents) + ";R:" + ",".join(rels)).encode()).hexdigest()


async def generate_concept_article(llm_settings, topic: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    system, user = build_concept_prompt(topic, evidence)
    markdown = await _complete(llm_settings, system, user)
    return markdown, build_citations(evidence)


async def generate_overview_article(llm_settings, top_entities: list[dict], facts: list[dict]) -> tuple[str, list[dict]]:
    system, user = build_overview_prompt(top_entities, facts)
    markdown = await _complete(llm_settings, system, user)
    return markdown, []   # 总览引用实体/关系层，不逐句 [^n]；citations 留空
```

并把现有 `generate_article` 的末尾 LLM 调用替换为 `markdown = await _complete(llm_settings, system, user)`（保持其 `build_article_prompt` + `build_citations(evidence)` 返回不变）。

- [ ] **Step 4: 跑确认通过** — 同 Step 2 命令 → PASS（原 3 项 + 新 3 项;`generate_*` 打真 LLM 不在单测）。

- [ ] **Step 5: Commit**
```bash
git add backend/nanoresearch/rag/wiki/article_generator.py backend/tests/unit/rag/test_article_generator.py
git commit -m "feat(wiki): article_generator concept/overview prompts + shared _complete + overview_signature"
```

---

### Task 2: repo 按前缀列文章（listConcepts 用）

**Files:**
- Modify: `backend/nanoresearch/storage/repositories/graph_repo.py`
- Test: `backend/tests/storage/test_graph_article.py`（追加）

**Interfaces:**
- Produces: `list_articles_by_prefix(kb_id, prefix: str) -> list[dict]` → `[{"key": str, "generated_at": iso|None}]`（`key`=去掉前缀后的展示名;按 generated_at desc）。

- [ ] **Step 1: 写失败测试**（追加到 `test_graph_article.py`）

```python
def test_list_articles_by_prefix():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        await repo.upsert_article(s["kb_id"], "concept::体渲染", "a", [], "h", "m")
        await repo.upsert_article(s["kb_id"], "concept::实时渲染", "b", [], "h", "m")
        await repo.upsert_article(s["kb_id"], "3dgs", "c", [], "h", "m")   # entity, not concept
        rows = await repo.list_articles_by_prefix(s["kb_id"], "concept::")
        names = {r["key"] for r in rows}
        assert names == {"体渲染", "实时渲染"}     # entity excluded, prefix stripped
    run(_())
```
（`upsert_article` 对 `entity_name` 会 `_normalize`;测试里概念名用无需归一化改写的中文,`_normalize` 只小写+去空格+去括号,中文原样。故 key 存为 `concept::体渲染`。实现里列前缀时按存储值匹配。）

- [ ] **Step 2: 跑确认失败** — `cd backend && python -m pytest tests/storage/test_graph_article.py::test_list_articles_by_prefix -v` → FAIL。

- [ ] **Step 3: 实现**（`graph_repo.py`,追加）

```python
    async def list_articles_by_prefix(self, kb_id: uuid.UUID, prefix: str) -> list[dict]:
        from nanoresearch.storage.models import KgEntityArticle
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntityArticle.entity_name, KgEntityArticle.generated_at)
                .where(KgEntityArticle.kb_id == kb_id, KgEntityArticle.entity_name.like(f"{prefix}%"))
                .order_by(KgEntityArticle.generated_at.desc())
            )
            return [
                {"key": r[0][len(prefix):], "generated_at": r[1].isoformat() if r[1] else None}
                for r in result.all()
            ]
```

- [ ] **Step 4: 跑确认通过** — 同上命令 → PASS。

- [ ] **Step 5: Commit**
```bash
git add backend/nanoresearch/storage/repositories/graph_repo.py backend/tests/storage/test_graph_article.py
git commit -m "feat(graph): list_articles_by_prefix (for wiki concept list)"
```

---

### Task 3: 概念检索接线（抽 HybridSearch 助手 + 概念证据组装）

**Files:**
- Modify: `backend/nanoresearch/server/routers/knowledge_router.py`

**Interfaces:**
- Produces（供 Task 4 用):
  - `_build_hybrid_search(request, kb, settings, top_k: int)` → HybridSearch 实例（从现有 query-test 端点抽出的工厂装配，见下）。
  - `async _retrieve_concept_evidence(request, kb, settings, topic: str, top_k: int = 8) -> list[dict]` → `[{chunk_id, content, source(filename), page}]`（形状同 `get_entity_evidence`，供 `generate_concept_article`）。

- [ ] **Step 1: 抽 `_build_hybrid_search`** —— 阅读现有 query-test 端点(约 `knowledge_router.py:410-446`)里构造 `HybridSearch` 的那段(embedding/vector_store/bm25/dense/sparse/fusion/HybridSearch(...))，原样抽成模块级 helper（用传入的 `kb`/`settings`/`top_k`;`chroma_col = kb.chroma_collection or str(kb.id)`）。然后让 query-test 端点改调这个 helper（等价重构，不改行为）。

```python
def _build_hybrid_search(request: Request, kb, settings, top_k: int):
    from nanoresearch.rag.core.query_engine.dense_retriever import DenseRetriever
    from nanoresearch.rag.core.query_engine.sparse_retriever import SparseRetriever
    from nanoresearch.rag.core.query_engine.query_processor import QueryProcessor
    from nanoresearch.rag.core.query_engine.fusion import RRFFusion
    from nanoresearch.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
    from nanoresearch.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
    from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory
    from nanoresearch.rag.ingestion.storage.bm25_indexer import BM25Indexer
    from nanoresearch.rag.core.settings import resolve_path
    chroma_col = kb.chroma_collection or str(kb.id)
    embedding = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=chroma_col)
    bm25 = BM25Indexer(index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{chroma_col}")))
    dense = DenseRetriever(settings=settings, embedding_client=embedding, vector_store=vector_store)
    sparse = SparseRetriever(settings=settings, bm25_indexer=bm25, vector_store=vector_store, default_collection=chroma_col)
    return HybridSearch(
        settings=settings, query_processor=QueryProcessor(),
        dense_retriever=dense, sparse_retriever=sparse, fusion=RRFFusion(),
        config=HybridSearchConfig(fusion_top_k=top_k, enable_dense=True, enable_sparse=True,
                                  enable_graph_expansion=kb.enable_graph_expansion),
        session_factory=request.app.state.session_factory, kb_id=kb.id,
    )
```
> 抽取时把 query-test 端点里对应那段替换为 `hybrid = _build_hybrid_search(request, kb, settings, body.top_k)`（其余不变:`result = await hybrid.async_search(body.query, top_k=body.top_k, return_details=True)` 等）。exact import 路径以现有端点里出现的为准（`HybridSearch`/`HybridSearchConfig`/`DenseRetriever` 的 import 端点里已有，照搬）。

- [ ] **Step 2: 概念证据组装 helper**

```python
async def _retrieve_concept_evidence(request: Request, kb, settings, topic: str, top_k: int = 8) -> list[dict]:
    from nanoresearch.storage.models import KbDocument
    from sqlalchemy import select as _select
    hybrid = _build_hybrid_search(request, kb, settings, top_k)
    result = await hybrid.async_search(topic, top_k=top_k, return_details=True)
    chroma_ids = [r.chunk_id for r in result.results]
    pg_chunks = await _kb_repo(request).get_chunks_by_chroma_ids(chroma_ids)
    doc_ids = list({c.document_id for c in pg_chunks})
    name_map = {}
    if doc_ids:
        async with request.app.state.session_factory() as db:
            res = await db.execute(_select(KbDocument.id, KbDocument.filename).where(KbDocument.id.in_(doc_ids)))
            name_map = {row[0]: row[1] for row in res.all()}
    return [
        {"chunk_id": str(c.id), "content": c.content or "",
         "source": name_map.get(c.document_id, ""), "page": (c.chunk_metadata or {}).get("page")}
        for c in pg_chunks
    ]
```

- [ ] **Step 3: 语法检查** — `cd backend && python -c "import ast; ast.parse(open('nanoresearch/server/routers/knowledge_router.py',encoding='utf-8').read()); print('OK')"`（检索是运行时的,单测不覆盖,留 e2e）。

- [ ] **Step 4: Commit**
```bash
git add backend/nanoresearch/server/routers/knowledge_router.py
git commit -m "refactor(api): extract _build_hybrid_search; add _retrieve_concept_evidence for concept pages"
```

---

### Task 4: 概念 + 总览 API 端点

**Files:**
- Modify: `backend/nanoresearch/server/routers/knowledge_router.py`（Graph 区）

**Interfaces:**
- Consumes: `_retrieve_concept_evidence`/`_build_hybrid_search`(T3)、`generate_concept_article`/`generate_overview_article`/`evidence_signature`/`overview_signature`(T1)、`list_articles_by_prefix`(T2)、`get_stats`/`get_entity_facts`(Phase1)、`get_article`/`upsert_article`/`_article_dict`/`_resolve_rag_settings`/`_get_kb_or_404`(Phase2)。
- Produces JSON:
  - `GET /graph/concepts` → `{"concepts": [{key, generated_at}]}`
  - `GET/POST /graph/concept/article?topic=...` → `{"article": {...}|null}`
  - `GET/POST /graph/overview/article` → `{"article": {...}|null}`

- [ ] **Step 1: 概念列表 + 概念页 GET/POST**

```python
_CONCEPT_PREFIX = "concept::"
_OVERVIEW_KEY = "__overview__"


@router.get("/api/knowledge/{kb_id}/graph/concepts")
async def list_graph_concepts(kb_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    rows = await _graph_repo(request).list_articles_by_prefix(uuid.UUID(kb_id), _CONCEPT_PREFIX)
    return {"concepts": rows}


@router.get("/api/knowledge/{kb_id}/graph/concept/article")
async def get_concept_article(kb_id: str, topic: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    from nanoresearch.storage.repositories.graph_repo import _normalize
    key = _CONCEPT_PREFIX + _normalize(topic)
    row = await _graph_repo(request).get_article(uuid.UUID(kb_id), key)
    if row is None:
        return {"article": None}
    return {"article": _article_dict(row, stale=False)}   # 概念 stale 判定成本高(要重检索),MVP 不在 GET 判


@router.post("/api/knowledge/{kb_id}/graph/concept/article")
async def generate_concept_article_ep(kb_id: str, topic: str, request: Request, uid: str = Depends(get_current_user)):
    kb = await _get_kb_or_404(kb_id, uid, request)
    from nanoresearch.storage.repositories.graph_repo import _normalize
    from nanoresearch.rag.wiki.article_generator import generate_concept_article, evidence_signature
    settings = await _resolve_rag_settings(uid, request)
    evidence = await _retrieve_concept_evidence(request, kb, settings, topic)
    markdown, citations = await generate_concept_article(settings, topic, evidence)
    model = getattr(getattr(settings, "llm", None), "model", None)
    key = _CONCEPT_PREFIX + _normalize(topic)
    row = await _graph_repo(request).upsert_article(uuid.UUID(kb_id), key, markdown, citations, evidence_signature(evidence), model)
    return {"article": _article_dict(row, stale=False)}
```
> 注:`get_article`/`upsert_article` 内部会对传入 key 再 `_normalize`;而 `concept::体渲染` 过 `_normalize` 会小写(中文原样、`::` 保留)。为保证 GET/POST 用同一 key,两处都用 `_CONCEPT_PREFIX + _normalize(topic)` 传入(repo 内再 normalize 对该串幂等)。

- [ ] **Step 2: 总览页 GET/POST**

```python
@router.get("/api/knowledge/{kb_id}/graph/overview/article")
async def get_overview_article(kb_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    row = await _graph_repo(request).get_article(uuid.UUID(kb_id), _OVERVIEW_KEY)
    if row is None:
        return {"article": None}
    from nanoresearch.rag.wiki.article_generator import overview_signature
    stats = await _graph_repo(request).get_stats(uuid.UUID(kb_id))
    top = stats.get("top_entities", [])
    facts = []
    for e in top[:10]:
        facts.extend(await _graph_repo(request).get_entity_facts(uuid.UUID(kb_id), e["name"]))
    stale = overview_signature(top, facts) != row.evidence_hash
    return {"article": _article_dict(row, stale)}


@router.post("/api/knowledge/{kb_id}/graph/overview/article")
async def generate_overview_article_ep(kb_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    from nanoresearch.rag.wiki.article_generator import generate_overview_article, overview_signature
    repo = _graph_repo(request); kb_uuid = uuid.UUID(kb_id)
    stats = await repo.get_stats(kb_uuid)
    top = stats.get("top_entities", [])
    facts = []
    for e in top[:10]:
        facts.extend(await repo.get_entity_facts(kb_uuid, e["name"]))
    settings = await _resolve_rag_settings(uid, request)
    markdown, citations = await generate_overview_article(settings, top, facts)
    model = getattr(getattr(settings, "llm", None), "model", None)
    row = await repo.upsert_article(kb_uuid, _OVERVIEW_KEY, markdown, citations, overview_signature(top, facts), model)
    return {"article": _article_dict(row, stale=False)}
```

- [ ] **Step 3: 语法检查** — `cd backend && python -c "import ast; ast.parse(open('nanoresearch/server/routers/knowledge_router.py',encoding='utf-8').read()); print('OK')"`。

- [ ] **Step 4: Commit**
```bash
git add backend/nanoresearch/server/routers/knowledge_router.py
git commit -m "feat(api): wiki concept + overview article endpoints (+ concept list)"
```

---

### Task 5: 前端 apis + 抽 ArticleView 复用块

**Files:**
- Modify: `web/src/apis/knowledge.js`
- Create: `web/src/components/ArticleView.vue`
- Modify: `web/src/views/KnowledgeDetailView.vue`（实体页顶部改用 ArticleView）

**Interfaces:**
- Produces:
  - apis:`getConceptArticle(kbId, topic)`/`generateConceptArticle(kbId, topic)`/`getOverviewArticle(kbId)`/`generateOverviewArticle(kbId)`/`listConcepts(kbId)`。
  - `<ArticleView :article="article" :loading="loading" @generate="..." />` —— 封装 Phase 2 实体页顶部那套(有缓存→`CitationText`+stale 标+重生成;无→生成按钮;`a-spin`)。props `article`(obj|null)+`loading`(bool);emit `generate`。

- [ ] **Step 1: apis**（`apis/knowledge.js`,Knowledge Graph 区）
```javascript
export const listConcepts           = (kbId)        => apiGet(`/api/knowledge/${kbId}/graph/concepts`)
export const getConceptArticle      = (kbId, topic) => apiGet(`/api/knowledge/${kbId}/graph/concept/article?topic=${encodeURIComponent(topic)}`)
export const generateConceptArticle = (kbId, topic) => apiPost(`/api/knowledge/${kbId}/graph/concept/article?topic=${encodeURIComponent(topic)}`, {})
export const getOverviewArticle      = (kbId)        => apiGet(`/api/knowledge/${kbId}/graph/overview/article`)
export const generateOverviewArticle = (kbId)        => apiPost(`/api/knowledge/${kbId}/graph/overview/article`, {})
```

- [ ] **Step 2: 抽 `ArticleView.vue`** —— 阅读 `KnowledgeDetailView.vue` 里 Phase 2 实体页顶部的 `wiki-article` 块(`a-spin`+`CitationText`+stale 标+生成/重生成按钮),原样搬进新组件:props `article`(Object|null)、`loading`(Boolean);把"生成/重生成"按钮点击改成 `emit('generate')`;import `CitationText`。

- [ ] **Step 3: 实体页改用 ArticleView** —— `KnowledgeDetailView.vue` 实体详情顶部把原 `wiki-article` 块替换为 `<ArticleView :article="article" :loading="articleLoading" @generate="genArticle" />`(行为等价);import `ArticleView`。

- [ ] **Step 4: build** — `cd web && npm run build` → 0 error。（e2e:实体页词条行为不变。）

- [ ] **Step 5: Commit**
```bash
git add web/src/apis/knowledge.js web/src/components/ArticleView.vue web/src/views/KnowledgeDetailView.vue
git commit -m "refactor(web): extract ArticleView; concept/overview article APIs"
```

---

### Task 6: 前端左栏三区（总览 / 概念 / 实体）

**Files:**
- Modify: `web/src/views/KnowledgeDetailView.vue`

**Interfaces:**
- Consumes: `ArticleView`(T5)、concept/overview apis(T5)、现有实体列表/详情。

- [ ] **Step 1: 状态 + 方法**（`<script setup>`；import concept/overview apis + `listConcepts`）
```javascript
const wikiView = ref('entity')     // 'overview' | 'concept' | 'entity'
const concepts = ref([])           // [{key, generated_at}]
const conceptInput = ref('')
const activeConcept = ref('')      // 当前打开的概念 key
const conceptArticle = ref(null); const conceptLoading = ref(false)
const overviewArticle = ref(null); const overviewLoading = ref(false)

async function loadConcepts() { try { concepts.value = (await listConcepts(kbId)).concepts || [] } catch(e){} }
async function openConcept(topic) {
  wikiView.value = 'concept'; activeConcept.value = topic; conceptArticle.value = null
  try { conceptArticle.value = (await getConceptArticle(kbId, topic)).article } catch(e){}
}
async function genConcept() {
  const topic = activeConcept.value || conceptInput.value.trim(); if (!topic) return
  conceptLoading.value = true
  try { conceptArticle.value = (await generateConceptArticle(kbId, topic)).article; activeConcept.value = topic; await loadConcepts() }
  catch(e){ message.error('生成概念页失败') } finally { conceptLoading.value = false }
}
function newConcept() { const t = conceptInput.value.trim(); if(!t) return; activeConcept.value=t; conceptArticle.value=null; wikiView.value='concept'; genConcept(); conceptInput.value='' }
async function openOverview() {
  wikiView.value = 'overview'
  try { overviewArticle.value = (await getOverviewArticle(kbId)).article } catch(e){}
}
async function genOverview() {
  overviewLoading.value = true
  try { overviewArticle.value = (await generateOverviewArticle(kbId)).article } catch(e){ message.error('生成总览失败') } finally { overviewLoading.value = false }
}
```
挂到 tab 懒加载:现有 `watch(activeTab)` 里 `graph` 分支加 `loadConcepts()`。

- [ ] **Step 2: 左栏三区模板** —— 在知识图谱 tab 的 `wiki-list`(左栏)顶部,加"总览"入口 + "概念"区(输入框 + 列表),原有"实体"列表加个小标题。点击分别 `openOverview()` / `openConcept(c.key)` / 现有 `selectEntity()`;三者设 `wikiView`。
```html
<div class="wiki-nav">
  <div class="wiki-nav-item" :class="{active: wikiView==='overview'}" @click="openOverview">📄 总览</div>
  <div class="wiki-nav-sec">概念</div>
  <a-input-search v-model:value="conceptInput" placeholder="新建概念…" enter-button="生成" size="small" @search="newConcept" />
  <div v-for="c in concepts" :key="c.key" class="wiki-nav-item"
       :class="{active: wikiView==='concept' && activeConcept===c.key}" @click="openConcept(c.key)">{{ c.key }}</div>
  <div class="wiki-nav-sec">实体</div>
  <!-- 现有实体搜索框 + 实体列表(点击项里加 wikiView='entity') -->
</div>
```

- [ ] **Step 3: 右侧按 wikiView 渲染**
```html
<div class="wiki-detail">
  <ArticleView v-if="wikiView==='overview'" :article="overviewArticle" :loading="overviewLoading" @generate="genOverview" />
  <ArticleView v-else-if="wikiView==='concept'" :article="conceptArticle" :loading="conceptLoading" @generate="genConcept" />
  <template v-else><!-- 现有实体详情(含 Phase 2 ArticleView + 事实/邻居/图) --></template>
</div>
```
（选实体时 `selectEntity` 里设 `wikiView.value='entity'`。）

- [ ] **Step 4: CSS**（`<style scoped>`）
```css
.wiki-nav-sec { font-size: 12px; color: #999; margin: 10px 0 4px; }
.wiki-nav-item { padding: 6px 8px; border-radius: 6px; cursor: pointer; }
.wiki-nav-item:hover { background: #f5f5f5; }
.wiki-nav-item.active { background: #e6f0ff; }
```

- [ ] **Step 5: build** — `cd web && npm run build` → 0 error。

- [ ] **Step 6: Commit**
```bash
git add web/src/views/KnowledgeDetailView.vue
git commit -m "feat(web): wiki left-nav 3 sections (overview/concept/entity) + concept & overview pages"
```

---

## 手动 e2e（全部完成后;需 LLM+服务+KG 已建）
1. 知识图谱 tab 左栏出现 总览/概念/实体 三区。
2. 概念区输入"体渲染"→生成→出词条带 `[^n]`(点开 popover 见来源);已建概念进列表,重开秒读。
3. 点"总览"→生成→全库导览;源变(重建图)后 stale。
4. 实体页词条(Phase 2)行为不变(ArticleView 抽取回归)。
5. 概念/实体同名不冲突(key 命名空间)。

## Self-Review
- **Spec coverage**:概念(RAG:T3 检索+T1 prompt+T4 API+T6 UI)、总览(KG 骨架:T4+T1+T6)、统一内核复用(T1 _complete、T5 ArticleView、CitationText)、存储命名空间(T2 前缀列+T4 key)、左栏三区(T6)、按需+缓存+stale(T4)、不打分(全程)。✅
- **Placeholder scan**:无 TODO;后端给完整代码;T3 抽 HybridSearch / T5 抽 ArticleView 是"读现有块原样搬"的明确抽取,给了 helper 代码 + 组件 API。✅
- **Type consistency**:evidence `{chunk_id,content,source,page}` 贯穿 `_retrieve_concept_evidence`→`generate_concept_article`→`build_citations`;article 形状复用 `_article_dict`/`CitationText`;key 命名空间 `concept::`/`__overview__` 贯穿 T2/T4/T6。✅
- **无迁移**:复用 kg_entity_articles,key 命名空间,不加列。✅
- **LLM/检索**:_complete 复用 Phase2 pattern;检索复用 query-test 的 HybridSearch(抽 helper)。✅
