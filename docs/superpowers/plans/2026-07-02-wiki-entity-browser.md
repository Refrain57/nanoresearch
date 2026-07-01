# Wiki 实体浏览器（Phase 1）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 KnowledgeDetailView 增加「知识图谱/Wiki」tab —— 基于已有 KG 的 grounded 结构化浏览器：实体列表 + 实体详情（事实 + 每条事实的佐证文档数 + 邻居跳转 + 证据 chunk）+ 可折叠邻居小图。

**Architecture:** 后端在 `GraphRepository` 加只读查询、在 `knowledge_router` 加 3 个 GET 端点（复用现有 `_graph_repo`/`_get_kb_or_404`）；前端在 `KnowledgeDetailView.vue` 加一个 Ant Design tab，API 走 `apis/knowledge.js`。无 schema 变更、无新建图流程。

**Tech Stack:** Python + SQLAlchemy async (asyncpg) + FastAPI；Vue3 + Ant Design Vue + Pinia。

## Global Constraints

- 只读、per-KB；不改建图流程（复用现有手动「重建知识图谱」`buildKbGraph`）。
- **佐证（doc_count）= 该 triple 的 `KgTripleMention` join `KbChunk` 后 distinct `document_id` 数**。确定值，贯穿 repo→API→前端。
- 实体名在 KG 里是**归一化后（小写、去括号）**存储的 → 展示即归一名（MVP 限制，不额外还原大小写）。
- 明确不做：LLM 合成 / lint / 全 KG 力导向大图 / 自动建图 / 跨 KB。
- 后端 repo：`GraphRepository(session_factory)`，方法用 `async with self._factory() as db`。
- 后端测试：真实 PG 测试库（`tests/conftest.py` 的 `make_factory()` + `test_repositories.py` 的 `run()` 模式）；KG/kb 表需自行 TRUNCATE 清理。需本机 `nanoresearch_test` 库在跑；跑不了则报告环境缺失。
- 前端无单测框架 → 每个前端任务的验收 = `cd web && npm run build`（0 error；chunk-size 警告是既有的）+ 手动 e2e。
- 前端 API 统一走 `apis/knowledge.js` 的 `apiGet`/`apiPost`；tab 用 `a-tab-pane`，`activeTab` 懒加载。

---

### Task 1: GraphRepository 只读查询 + 单测

**Files:**
- Modify: `backend/nanoresearch/storage/repositories/graph_repo.py`（在 "Query helpers" 区追加方法）
- Create: `backend/tests/storage/test_graph_repo.py`

**Interfaces:**
- Consumes: 现有模型 `KgEntity/KgEntityMention/KgTriple/KgTripleMention/KbChunk/KbDocument/KnowledgeBase`；`_normalize`。
- Produces（后续 Task 2 依赖，签名固定）:
  - `list_entities(kb_id: uuid.UUID, search: str|None=None, limit: int=50, offset: int=0) -> list[dict]` → `[{"name": str, "label": str, "mentions": int}]`
  - `get_entity_summary(kb_id: uuid.UUID, name: str) -> dict | None` → `{"name": str, "label": str, "mention_count": int}`
  - `get_entity_facts(kb_id: uuid.UUID, name: str) -> list[dict]` → `[{"triple_id": str, "source": str, "label": str, "target": str, "doc_count": int}]`
  - `get_chunks_by_triple(triple_id: uuid.UUID) -> list[KbChunk]`

> **Note（对 spec 的落地细化）**：spec 列了 `get_entity_neighbors`，本计划**改为在 Task 2 的详情端点里从 facts 派生 neighbors**（spec 允许"可由 get_entity_facts 派生"），并新增 `get_entity_summary` 供详情页头部。净效果一致。

- [ ] **Step 1: 追加 import（graph_repo.py 顶部）**

把第 10 行的 sqlalchemy import 改为包含 `distinct` 和 `or_`，并引入 `aliased`：

```python
from sqlalchemy import distinct, func, or_, select, text
from sqlalchemy.orm import aliased
```

（`KbChunk` 已在个别方法内局部 import，保持一致，方法内 `from nanoresearch.storage.models import KbChunk, KbDocument`。）

- [ ] **Step 2: 写失败测试** `backend/tests/storage/test_graph_repo.py`

```python
"""GraphRepository read-layer tests (Wiki Phase 1). Real PG, sync psycopg2 cleanup."""
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
                "TRUNCATE TABLE kg_triple_mentions, kg_entity_mentions, kg_triples, "
                "kg_entities, kb_chunks, kb_documents, knowledge_bases RESTART IDENTITY CASCADE"
            )
    finally:
        conn.close()


async def _seed(factory):
    from nanoresearch.storage.models import (
        KbChunk, KbDocument, KgEntity, KgEntityMention, KgTriple, KgTripleMention, KnowledgeBase,
    )
    kb_id = uuid.uuid4()
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    c1, c2, c3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    e_gs, e_nerf = uuid.uuid4(), uuid.uuid4()
    tid = uuid.uuid4()
    async with factory() as db:
        db.add(KnowledgeBase(id=kb_id, uid="tester", name="KB", chroma_collection="c"))
        db.add_all([
            KbDocument(id=d1, kb_id=kb_id, filename="paperA.pdf", file_path="/tmp/a"),
            KbDocument(id=d2, kb_id=kb_id, filename="paperB.pdf", file_path="/tmp/b"),
            KbChunk(id=c1, kb_id=kb_id, document_id=d1, chunk_index=0, content="3dgs vs nerf"),
            KbChunk(id=c2, kb_id=kb_id, document_id=d2, chunk_index=0, content="3dgs faster"),
            KbChunk(id=c3, kb_id=kb_id, document_id=d1, chunk_index=1, content="nerf detail"),
            KgEntity(id=e_gs, kb_id=kb_id, name="3dgs", label="method"),
            KgEntity(id=e_nerf, kb_id=kb_id, name="nerf", label="method"),
            KgTriple(id=tid, kb_id=kb_id, source_id=e_gs, target_id=e_nerf, label="faster_than"),
            KgEntityMention(entity_id=e_gs, chunk_id=c1, kb_id=kb_id),
            KgEntityMention(entity_id=e_gs, chunk_id=c2, kb_id=kb_id),
            KgEntityMention(entity_id=e_nerf, chunk_id=c1, kb_id=kb_id),
            KgEntityMention(entity_id=e_nerf, chunk_id=c3, kb_id=kb_id),
            KgTripleMention(triple_id=tid, chunk_id=c1, kb_id=kb_id),  # doc d1
            KgTripleMention(triple_id=tid, chunk_id=c2, kb_id=kb_id),  # doc d2
        ])
        await db.commit()
    return {"kb_id": kb_id, "tid": tid}


def test_list_entities_counts_and_search():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        rows = await repo.list_entities(s["kb_id"])
        by_name = {r["name"]: r for r in rows}
        assert by_name["3dgs"]["mentions"] == 2
        assert by_name["nerf"]["mentions"] == 2
        assert by_name["3dgs"]["label"] == "method"
        only = await repo.list_entities(s["kb_id"], search="3d")
        assert [r["name"] for r in only] == ["3dgs"]
    run(_())


def test_get_entity_summary():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        summ = await repo.get_entity_summary(s["kb_id"], "3DGS")
        assert summ == {"name": "3dgs", "label": "method", "mention_count": 2}
        assert await repo.get_entity_summary(s["kb_id"], "nope") is None
    run(_())


def test_get_entity_facts_doc_count_is_distinct_documents():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        facts = await repo.get_entity_facts(s["kb_id"], "3DGS")
        assert len(facts) == 1
        fact = facts[0]
        assert fact["source"] == "3dgs"
        assert fact["label"] == "faster_than"
        assert fact["target"] == "nerf"
        assert fact["doc_count"] == 2  # triple mentioned in chunks from 2 distinct docs
        assert fact["triple_id"] == str(s["tid"])
    run(_())


def test_get_chunks_by_triple():
    async def _():
        f = make_factory()
        s = await _seed(f)
        repo = GraphRepository(f)
        chunks = await repo.get_chunks_by_triple(s["tid"])
        assert len(chunks) == 2
        assert {c.content for c in chunks} == {"3dgs vs nerf", "3dgs faster"}
    run(_())
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/storage/test_graph_repo.py -v`
Expected: FAIL（`AttributeError: 'GraphRepository' object has no attribute 'list_entities'`）。若报无法连接 `nanoresearch_test` 库 → 报告环境缺失（需先起测试 PG），不要跳过。

- [ ] **Step 4: 实现四个方法**（追加到 graph_repo.py 的 "Query helpers" 区，`get_stats` 之后）

```python
    async def list_entities(
        self, kb_id: uuid.UUID, search: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """Paginated/searchable entity list with mention counts (desc)."""
        conds = [KgEntity.kb_id == kb_id]
        if search:
            conds.append(KgEntity.name.ilike(f"%{_normalize(search)}%"))
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntity.name, KgEntity.label, func.count(KgEntityMention.id).label("mentions"))
                .join(KgEntityMention, KgEntityMention.entity_id == KgEntity.id)
                .where(*conds)
                .group_by(KgEntity.name, KgEntity.label)
                .order_by(text("mentions DESC"))
                .limit(limit)
                .offset(offset)
            )
            return [{"name": r[0], "label": r[1], "mentions": r[2]} for r in result.all()]

    async def get_entity_summary(self, kb_id: uuid.UUID, name: str) -> dict | None:
        """Header info for one entity (by normalized name), or None if absent."""
        norm = _normalize(name)
        async with self._factory() as db:
            result = await db.execute(
                select(KgEntity.name, KgEntity.label, func.count(KgEntityMention.id).label("mentions"))
                .join(KgEntityMention, KgEntityMention.entity_id == KgEntity.id, isouter=True)
                .where(KgEntity.kb_id == kb_id, KgEntity.name == norm)
                .group_by(KgEntity.name, KgEntity.label)
            )
            row = result.first()
            if not row:
                return None
            return {"name": row[0], "label": row[1], "mention_count": row[2]}

    async def get_entity_facts(self, kb_id: uuid.UUID, name: str) -> list[dict]:
        """Triples where the entity is source OR target, with distinct-document corroboration."""
        from nanoresearch.storage.models import KbChunk
        norm = _normalize(name)
        SrcE = aliased(KgEntity)
        TgtE = aliased(KgEntity)
        async with self._factory() as db:
            ids_res = await db.execute(
                select(KgEntity.id).where(KgEntity.kb_id == kb_id, KgEntity.name == norm)
            )
            entity_ids = [r[0] for r in ids_res.all()]
            if not entity_ids:
                return []
            doc_count_sq = (
                select(
                    KgTripleMention.triple_id.label("tid"),
                    func.count(distinct(KbChunk.document_id)).label("doc_count"),
                )
                .join(KbChunk, KbChunk.id == KgTripleMention.chunk_id)
                .group_by(KgTripleMention.triple_id)
                .subquery()
            )
            result = await db.execute(
                select(
                    KgTriple.id, SrcE.name, KgTriple.label, TgtE.name,
                    func.coalesce(doc_count_sq.c.doc_count, 0).label("doc_count"),
                )
                .join(SrcE, SrcE.id == KgTriple.source_id)
                .join(TgtE, TgtE.id == KgTriple.target_id)
                .outerjoin(doc_count_sq, doc_count_sq.c.tid == KgTriple.id)
                .where(
                    KgTriple.kb_id == kb_id,
                    or_(KgTriple.source_id.in_(entity_ids), KgTriple.target_id.in_(entity_ids)),
                )
                .order_by(text("doc_count DESC"))
            )
            return [
                {"triple_id": str(r[0]), "source": r[1], "label": r[2], "target": r[3], "doc_count": r[4]}
                for r in result.all()
            ]

    async def get_chunks_by_triple(self, triple_id: uuid.UUID) -> list:
        """Evidence chunks for a fact (triple), via triple mentions."""
        from nanoresearch.storage.models import KbChunk
        async with self._factory() as db:
            result = await db.execute(
                select(KbChunk)
                .join(KgTripleMention, KgTripleMention.chunk_id == KbChunk.id)
                .where(KgTripleMention.triple_id == triple_id)
                .distinct()
            )
            return list(result.scalars().all())
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/storage/test_graph_repo.py -v`
Expected: PASS（4 passed）。

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/storage/repositories/graph_repo.py backend/tests/storage/test_graph_repo.py
git commit -m "feat(graph): read-layer queries for wiki entity browser (list/summary/facts+corroboration/triple-chunks)"
```

---

### Task 2: 三个 GET API 端点

**Files:**
- Modify: `backend/nanoresearch/server/routers/knowledge_router.py`（在 "Graph endpoints" 区、`build_graph` 附近追加）
- Test: 手动 curl（见 Step 4）；无新单测（端点是薄封装，逻辑已在 Task 1 覆盖）。

**Interfaces:**
- Consumes: Task 1 的 `list_entities`/`get_entity_summary`/`get_entity_facts`/`get_chunks_by_triple`；现有 `_graph_repo(request)`、`_get_kb_or_404(kb_id, uid, request)`、`get_current_user`、`_kb_repo(request)`。
- Produces（Task 3 依赖的 JSON 形状）:
  - `GET /api/knowledge/{kb_id}/graph/entities?search=&limit=&offset=` → `{"entities": [{"name","label","mentions"}]}`
  - `GET /api/knowledge/{kb_id}/graph/entities/{name}` → `{"name","label","mention_count","facts":[{triple_id,source,label,target,doc_count}],"neighbors":[str]}`
  - `GET /api/knowledge/{kb_id}/graph/triples/{triple_id}/chunks` → `{"chunks":[{"content","source","page","document_id"}]}`

- [ ] **Step 1: 加实体列表端点**

```python
@router.get("/api/knowledge/{kb_id}/graph/entities")
async def list_graph_entities(
    kb_id: str,
    request: Request,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    entities = await _graph_repo(request).list_entities(
        uuid.UUID(kb_id), search=search, limit=limit, offset=offset
    )
    return {"entities": entities}
```

- [ ] **Step 2: 加实体详情端点（facts + 派生 neighbors）**

```python
@router.get("/api/knowledge/{kb_id}/graph/entities/{name}")
async def get_graph_entity(
    kb_id: str,
    name: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _graph_repo(request)
    kb_uuid = uuid.UUID(kb_id)
    summary = await repo.get_entity_summary(kb_uuid, name)
    if summary is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="entity not found")
    facts = await repo.get_entity_facts(kb_uuid, name)
    self_name = summary["name"]
    neighbors: list[str] = []
    seen = {self_name}
    for f in facts:
        other = f["target"] if f["source"] == self_name else f["source"]
        if other not in seen:
            seen.add(other)
            neighbors.append(other)
    return {**summary, "facts": facts, "neighbors": neighbors}
```

- [ ] **Step 3: 加事实证据端点（chunk + 原文件名）**

```python
@router.get("/api/knowledge/{kb_id}/graph/triples/{triple_id}/chunks")
async def get_triple_chunks(
    kb_id: str,
    triple_id: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    chunks = await _graph_repo(request).get_chunks_by_triple(uuid.UUID(triple_id))
    # Resolve document_id -> original filename (chunk carries only the ingest path context).
    from nanoresearch.storage.models import KbDocument
    from sqlalchemy import select as _select
    doc_ids = list({c.document_id for c in chunks})
    name_map: dict = {}
    if doc_ids:
        async with request.app.state.session_factory() as db:
            res = await db.execute(_select(KbDocument.id, KbDocument.filename).where(KbDocument.id.in_(doc_ids)))
            name_map = {r[0]: r[1] for r in res.all()}
    return {
        "chunks": [
            {
                "content": (c.content or "")[:500],
                "source": name_map.get(c.document_id, ""),
                "page": (c.chunk_metadata or {}).get("page"),
                "document_id": str(c.document_id),
            }
            for c in chunks
        ]
    }
```

- [ ] **Step 4: 手动冒烟（需服务在跑 + KG 已建）**

Run（示意，替换真实 token/kb/name）:
```bash
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/knowledge/$KB/graph/entities?limit=5"
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/knowledge/$KB/graph/entities/3dgs"
```
Expected: 200 + 上述 JSON 形状（entities 列表 / facts 带 doc_count + neighbors）。

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/server/routers/knowledge_router.py
git commit -m "feat(api): wiki graph endpoints (entities list / entity detail with corroboration / triple evidence)"
```

---

### Task 3: 前端 tab + 实体列表 + 详情

**Files:**
- Modify: `web/src/apis/knowledge.js`（加 3 个 API 函数）
- Modify: `web/src/views/KnowledgeDetailView.vue`（加 tab + 两栏 + 详情 + 状态/方法）

**Interfaces:**
- Consumes: Task 2 的三个端点。
- Produces（Task 4 依赖）: 组件内响应式 `entityDetail`（含 `neighbors`）、方法 `selectEntity(name)`。

- [ ] **Step 1: 加 API 函数**（`web/src/apis/knowledge.js`，接在 "Knowledge Graph" 区）

```javascript
export const listGraphEntities = (kbId, params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return apiGet(`/api/knowledge/${kbId}/graph/entities${qs ? '?' + qs : ''}`)
}
export const getGraphEntity  = (kbId, name)     => apiGet(`/api/knowledge/${kbId}/graph/entities/${encodeURIComponent(name)}`)
export const getTripleChunks = (kbId, tripleId) => apiGet(`/api/knowledge/${kbId}/graph/triples/${tripleId}/chunks`)
```

- [ ] **Step 2: import + 响应式状态**（`KnowledgeDetailView.vue` `<script setup>`）

在第 681 行的 `from '@/apis/knowledge'` import 列表里追加 `listGraphEntities, getGraphEntity, getTripleChunks`。在 `activeTab` 附近加：

```javascript
const wikiEntities   = ref([])        // [{name,label,mentions}]
const wikiSearch     = ref('')
const wikiLoading    = ref(false)
const entityDetail   = ref(null)      // {name,label,mention_count,facts,neighbors}
const detailLoading  = ref(false)
const expandedTriple = ref(null)      // triple_id whose evidence is open
const tripleChunks   = ref([])        // evidence chunks for expandedTriple

async function loadWikiEntities() {
  wikiLoading.value = true
  try {
    const r = await listGraphEntities(kbId, { search: wikiSearch.value, limit: 100 })
    wikiEntities.value = r.entities || []
  } finally { wikiLoading.value = false }
}

async function selectEntity(name) {
  detailLoading.value = true
  expandedTriple.value = null
  tripleChunks.value = []
  try {
    entityDetail.value = await getGraphEntity(kbId, name)
  } finally { detailLoading.value = false }
}

async function toggleTripleEvidence(tripleId) {
  if (expandedTriple.value === tripleId) { expandedTriple.value = null; return }
  expandedTriple.value = tripleId
  const r = await getTripleChunks(kbId, tripleId)
  tripleChunks.value = r.chunks || []
}
```

`kbId` 已是该组件用于其它 API 的 KB id（沿用现有 `testQuery(kbId,...)` 等所用的同一变量）。

- [ ] **Step 3: 懒加载挂到 tab 切换**

在现有 `watch(activeTab, ...)`（约 1034 行）里加分支：

```javascript
  if (tab === 'graph' && !wikiEntities.value.length) {
    await loadWikiEntities()
  }
```

- [ ] **Step 4: 加 tab 模板**（在 `RAG 评估` 的 `</a-tab-pane>` 之后、`</a-tabs>` 之前插入）

```html
<a-tab-pane key="graph" tab="知识图谱/Wiki">
  <div v-if="!wikiEntities.length && !wikiLoading" class="wiki-empty">
    <a-empty description="暂无知识图谱数据，请先在「文档」页重建知识图谱" />
  </div>
  <div v-else class="wiki-wrap">
    <div class="wiki-list">
      <a-input-search v-model:value="wikiSearch" placeholder="搜索实体…"
        allow-clear @search="loadWikiEntities" style="margin-bottom:8px" />
      <a-spin :spinning="wikiLoading">
        <div v-for="e in wikiEntities" :key="e.name"
          class="wiki-ent" :class="{ active: entityDetail?.name === e.name }"
          @click="selectEntity(e.name)">
          <span class="wiki-ent-name">{{ e.name }}</span>
          <span class="wiki-ent-count">{{ e.mentions }}</span>
        </div>
      </a-spin>
    </div>
    <div class="wiki-detail">
      <a-spin :spinning="detailLoading">
        <template v-if="entityDetail">
          <h2 class="wiki-title">{{ entityDetail.name }}
            <a-tag>{{ entityDetail.label }}</a-tag>
            <span class="wiki-sub">被 {{ entityDetail.mention_count }} 处提及</span>
          </h2>

          <h3>事实</h3>
          <a-empty v-if="!entityDetail.facts.length" description="无事实" />
          <div v-for="f in entityDetail.facts" :key="f.triple_id" class="wiki-fact">
            <div class="wiki-fact-row" @click="toggleTripleEvidence(f.triple_id)">
              <span>{{ f.source }} <em>—{{ f.label }}→</em> {{ f.target }}</span>
              <a-tag color="blue">{{ f.doc_count }} 篇文档</a-tag>
            </div>
            <div v-if="expandedTriple === f.triple_id" class="wiki-evidence">
              <div v-for="(c, i) in tripleChunks" :key="i" class="wiki-chunk">
                <div class="wiki-chunk-src">{{ c.source }}<span v-if="c.page != null"> · p{{ c.page }}</span></div>
                <div class="wiki-chunk-text">{{ c.content }}</div>
              </div>
            </div>
          </div>

          <h3 style="margin-top:16px">邻居实体</h3>
          <a-tag v-for="n in entityDetail.neighbors" :key="n"
            class="wiki-neighbor" @click="selectEntity(n)">{{ n }}</a-tag>
        </template>
        <a-empty v-else description="选择左侧实体查看详情" />
      </a-spin>
    </div>
  </div>
</a-tab-pane>
```

- [ ] **Step 5: 加样式**（`<style scoped>` 末尾）

```css
.wiki-wrap { display: flex; gap: 16px; }
.wiki-list { width: 260px; flex-shrink: 0; max-height: 70vh; overflow-y: auto; border-right: 1px solid #eee; padding-right: 8px; }
.wiki-ent { display: flex; justify-content: space-between; padding: 6px 8px; border-radius: 6px; cursor: pointer; }
.wiki-ent:hover { background: #f5f5f5; }
.wiki-ent.active { background: #e6f0ff; }
.wiki-ent-count { color: #999; font-size: 12px; }
.wiki-detail { flex: 1; min-width: 0; }
.wiki-title { display: flex; align-items: center; gap: 8px; }
.wiki-sub { color: #999; font-size: 13px; font-weight: normal; }
.wiki-fact { border: 1px solid #eee; border-radius: 6px; margin-bottom: 6px; }
.wiki-fact-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; cursor: pointer; }
.wiki-fact-row em { color: #C15F3C; font-style: normal; }
.wiki-evidence { border-top: 1px dashed #eee; padding: 8px 10px; background: #fafafa; }
.wiki-chunk { margin-bottom: 8px; }
.wiki-chunk-src { font-size: 12px; color: #888; }
.wiki-chunk-text { font-size: 13px; white-space: pre-wrap; }
.wiki-neighbor { cursor: pointer; margin-bottom: 6px; }
.wiki-empty { padding: 40px 0; }
```

- [ ] **Step 6: build**

Run: `cd web && npm run build`
Expected: 0 error（chunk-size 警告既有）。

- [ ] **Step 7: Commit**

```bash
git add web/src/apis/knowledge.js web/src/views/KnowledgeDetailView.vue
git commit -m "feat(web): wiki entity browser tab (list + detail with facts/corroboration/evidence/neighbors)"
```

---

### Task 4: 可折叠邻居关系图

**Files:**
- Create: `web/src/components/EntityNeighborGraph.vue`（轻量 SVG，无新依赖）
- Modify: `web/src/views/KnowledgeDetailView.vue`（详情页嵌入可折叠图）

**Interfaces:**
- Consumes: Task 3 的 `entityDetail.name` + `entityDetail.neighbors`（`string[]`）、`selectEntity(name)`。
- Produces: 无（叶子）。

- [ ] **Step 1: 建 SVG 组件** `web/src/components/EntityNeighborGraph.vue`

中心实体 + 邻居围一圈、连线；点邻居 emit `select`。

```vue
<template>
  <svg :width="size" :height="size" class="eng">
    <line v-for="(p, i) in points" :key="'l'+i" :x1="c" :y1="c" :x2="p.x" :y2="p.y" class="eng-edge" />
    <g>
      <circle :cx="c" :cy="c" r="26" class="eng-center" />
      <text :x="c" :y="c" class="eng-label eng-center-label">{{ trunc(center) }}</text>
    </g>
    <g v-for="(p, i) in points" :key="'n'+i" class="eng-node" @click="$emit('select', p.name)">
      <circle :cx="p.x" :cy="p.y" r="20" />
      <text :x="p.x" :y="p.y" class="eng-label">{{ trunc(p.name) }}</text>
    </g>
  </svg>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({
  center: { type: String, default: '' },
  neighbors: { type: Array, default: () => [] },
})
defineEmits(['select'])
const size = 360
const c = size / 2
const R = 130
const points = computed(() => {
  const ns = props.neighbors.slice(0, 10)
  return ns.map((name, i) => {
    const a = (2 * Math.PI * i) / ns.length - Math.PI / 2
    return { name, x: c + R * Math.cos(a), y: c + R * Math.sin(a) }
  })
})
const trunc = (s) => (s && s.length > 8 ? s.slice(0, 7) + '…' : s)
</script>

<style scoped>
.eng { max-width: 100%; }
.eng-edge { stroke: #ddd; stroke-width: 1; }
.eng-center { fill: #C15F3C; }
.eng-node circle { fill: #5E7355; cursor: pointer; }
.eng-node:hover circle { fill: #6f875f; }
.eng-label { fill: #fff; font-size: 11px; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }
.eng-center-label { font-weight: 600; }
</style>
```

- [ ] **Step 2: 在详情页嵌入可折叠图**（Task 3 模板的「邻居实体」块之后）

```html
<a-collapse ghost style="margin-top:8px">
  <a-collapse-panel key="graph" header="知识图谱">
    <EntityNeighborGraph
      :center="entityDetail.name"
      :neighbors="entityDetail.neighbors"
      @select="selectEntity" />
  </a-collapse-panel>
</a-collapse>
```

- [ ] **Step 3: import 组件**（`KnowledgeDetailView.vue` script，AppLayout import 附近）

```javascript
import EntityNeighborGraph from '@/components/EntityNeighborGraph.vue'
```

- [ ] **Step 4: build**

Run: `cd web && npm run build`
Expected: 0 error。

- [ ] **Step 5: Commit**

```bash
git add web/src/components/EntityNeighborGraph.vue web/src/views/KnowledgeDetailView.vue
git commit -m "feat(web): collapsible neighbor graph in wiki entity detail"
```

---

## 手动 e2e（全部完成后，需服务在跑 + 目标 KB 已建图）
1. 进 KnowledgeDetailView → 「知识图谱/Wiki」tab → 左侧出现实体列表（按提及数排序）；搜索可过滤。
2. 点实体 → 右侧详情：头部（名/label/提及数）、事实列表每条带「N 篇文档」。
3. 点一条事实 → 展开证据 chunk，来源显示**原文件名**（非 temp 路径）。
4. 点邻居 chip / 图里邻居节点 → 切换到该实体详情。
5. 展开「知识图谱」→ 中心 + 邻居小图；默认收起。
6. KG 未建的 KB → 显示空态引导。

## Self-Review
- **Spec coverage**：tab（T3）、实体列表（T1 list_entities/T2/T3）、事实级+佐证 distinct-doc（T1 get_entity_facts/测试断言 doc_count=2）、证据+原文件名（T1 get_chunks_by_triple/T2 filename 解析/T3 展开）、邻居跳转（T2 派生/T3）、可折叠图（T4）、空态+依赖现有建图（T3 Step4）、非目标均未触及。✅
- **Placeholder scan**：无 TODO/占位；每个代码步给了完整代码。✅
- **Type consistency**：`doc_count`(int)、`triple_id`(str)、`neighbors`(list[str])、`entityDetail.{name,label,mention_count,facts,neighbors}`、`selectEntity(name)`、`EntityNeighborGraph` props `center/neighbors` + emit `select` —— repo/API/前端三层一致。✅
- **偏差记录**：spec 的 `get_entity_neighbors` 改为 API 层从 facts 派生 + 新增 `get_entity_summary`（spec 允许派生）；neighbors 为 `list[str]`（名字，MVP 足够）。
