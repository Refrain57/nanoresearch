# NanoResearch 知识库闭环改造方案

> 基于实际源码分析，非伪代码

---

## 一、核心设计

### 1.1 目标

让研究系统"越用越好用"：每次研究的结论沉淀到知识库，下次研究可复用。

### 1.2 与 Autogenesis 的区别

| 维度 | Autogenesis | NanoResearch (本方案) |
|------|-------------|-----------------|
| **优化对象** | Agent 自身 (prompt/tools) | 知识库内容 |
| **核心问题** | "我这次做得不好，下次怎么做得更好" | "我这次学到了什么，下次能复用" |
| **持久化** | Memory 事件系统 | RAG 向量存储 |
| **复用方式** | 读取历史 Insight 指导优化 | 检索已有知识指导研究方向 |

两者不互斥，可以结合使用。

---

## 二、三层知识结构

```
┌─────────────────────────────────────────────────────────────┐
│                    三层知识结构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 1: Event (原始事件)                                   │
│    - 每次研究的完整结果                                       │
│    - 包含: topic, report, findings, contradictions          │
│    - 不直接复用，作为提炼原料                                 │
│                                                             │
│  Layer 2: Claim (原子化事实)                                 │
│    - 从 Event 提取的原子陈述                                  │
│    - 例: "3DGS 在薄表面渲染上存在 artifact"                  │
│    - 绑定到具体 topic，迁移性弱                               │
│    - 存储: collection = research_claims                     │
│                                                             │
│  Layer 3: Insight (可迁移洞察)                               │
│    - 从多个 Claim 提炼的深层规律                              │
│    - 例: "基于显式点表示的方法对薄结构处理较差"               │
│    - 跨 topic 可迁移，复用价值高                              │
│    - 存储: collection = research_insights                   │
│    - 有 maturity 字段: candidate / confirmed                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、存储设计

### 3.1 分开存储

| Collection | 内容 | 粒度 | 用途 |
|------------|------|------|------|
| `research_claims` | 具体事实 | 细 | 补充细节 |
| `research_insights` | 可迁移规律 | 粗 | 指导研究方向 |

**理由**：混在一起会影响检索质量和 Planner prompt 质量。

### 3.2 数据结构

```python
@dataclass
class Claim:
    claim: str
    type: str  # factual | interpretation | method | insight
    is_evergreen: bool
    source_urls: list[str]
    confidence: float
    created_at: datetime

@dataclass
class Insight:
    insight: str
    supporting_claims: list[str]  # 来源 claims
    applicable_domains: list[str]  # 适用领域
    confidence: float
    is_evergreen: bool
    maturity: str  # candidate | confirmed
    created_at: datetime
```

### 3.3 `is_evergreen` 的用途：检索权重衰减

```python
async def search_with_decay(self, query: str, top_k: int, decay_factor: float = 0.95):
    """检索时对非 evergreen 内容做时间衰减"""
    results = await self.search(query, top_k * 2)
    now = datetime.now()

    for r in results:
        is_evergreen = r.metadata.get("is_evergreen", False)
        created_at = r.metadata.get("created_at")

        if not is_evergreen and created_at:
            days = (now - datetime.fromisoformat(created_at)).days
            r.score *= decay_factor ** (days / 30)

    return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
```

---

## 四、核心流程

### 4.1 即时阶段 (每次研究后)

```
ResearchResult → Claims → Candidate Insights → 写入知识库
```

```python
async def process(self, result: ResearchResult) -> KnowledgeProcessResult:
    # Step 1: 提取 Claims
    claims = await self._extract_claims(result)

    # Step 2: 提炼 Candidate Insights
    insights = await self._extract_insights(claims, result.topic)

    # Step 3: 置信度评估
    claims = self._assess_confidence(claims, result.synthesis)
    insights = self._assess_insight_confidence(insights, claims)

    # Step 4: 去重检查
    new_claims, dup_claims, conflict_claims = await self._check_claim_duplicates(claims)
    new_insights = await self._check_insight_duplicates(insights)

    # Step 5: 写入知识库
    await self._write_claims(new_claims)
    await self._write_insights(new_insights, maturity="candidate")

    # Step 6: 记录到 Tracker
    for insight in new_insights:
        self.tracker.add_candidate(insight.id, insight)

    return KnowledgeProcessResult(...)
```

### 4.2 批量阶段 (每 N 次研究后或手动触发)

```
Candidate Insights → 跨会话提炼 → Confirmed Insights → 删除旧 Candidates
```

```python
async def batch_refine_insights(self) -> BatchRefineResult:
    # 1. 读取所有未处理的 candidates
    candidates = self.tracker.list_candidates()
    if not candidates:
        return BatchRefineResult(processed=0, confirmed=0)

    # 2. 跨会话提炼（返回 insights + 映射关系）
    result = await self._llm_refine_with_mapping(candidates)
    confirmed_insights = result["insights"]

    # 3. 写入 confirmed insights（先写，确保成功）
    confirmed_ids = await self._write_insights(confirmed_insights, maturity="confirmed")

    # 4. 删除旧 candidates（ChromaDB + Tracker 同步）
    old_ids = [c["id"] for c in candidates]
    await self.knowledge_search.delete_insights(ids=old_ids)
    for old_id in old_ids:
        self.tracker.mark_processed(old_id)

    logger.info(
        "Batch refine: {} candidates → {} confirmed insights",
        len(candidates), len(confirmed_ids)
    )

    return BatchRefineResult(processed=len(candidates), confirmed=len(confirmed_ids))
```

---

## 五、关键组件

### 5.1 KnowledgeSearch

```python
# nanobot/research/knowledge_search.py

class KnowledgeSearch:
    """知识库检索 + 写入（封装所有存储操作）"""

    def __init__(
        self,
        claim_store: ChromaStore,
        insight_store: ChromaStore,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        bm25_indexer: BM25Indexer,
    ):
        self.claim_store = claim_store
        self.insight_store = insight_store
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.bm25_indexer = bm25_indexer

    # === 检索方法 ===

    async def search_claims(self, query: str, top_k: int = 5, apply_decay: bool = True):
        """检索 claims，可选时间衰减"""
        results = await self.claim_store.query(
            vector=self._embed(query),
            top_k=top_k * 2,
        )
        if apply_decay:
            results = self._apply_decay(results)
        return results[:top_k]

    async def search_insights(self, query: str, top_k: int = 3, maturity: str = None):
        """检索 insights，可按 maturity 过滤"""
        filters = {"maturity": maturity} if maturity else None
        results = await self.insight_store.query(
            vector=self._embed(query),
            top_k=top_k,
            filters=filters,
        )
        return results

    async def search_all(self, query: str, top_k_claims: int = 5, top_k_insights: int = 3):
        """分别检索，分区返回"""
        claims = await self.search_claims(query, top_k_claims)
        insights = await self.search_insights(query, top_k_insights)
        return claims, insights

    def _apply_decay(self, results: list, decay_factor: float = 0.95) -> list:
        """对非 evergreen 内容做时间衰减"""
        now = datetime.now()
        for r in results:
            is_evergreen = r.get("metadata", {}).get("is_evergreen", False)
            created_at = r.get("metadata", {}).get("created_at")
            if not is_evergreen and created_at:
                days = (now - datetime.fromisoformat(created_at)).days
                r["score"] *= decay_factor ** (days / 30)
        return sorted(results, key=lambda x: x["score"], reverse=True)

    # === 写入方法 ===

    async def write_claims(self, claims: list[Claim]) -> list[str]:
        """写入 claims collection"""
        if not claims:
            return []

        # 过滤低置信度
        claims = [c for c in claims if c.confidence >= 0.7]

        chunks = [
            Chunk(
                id=f"claim_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}",
                text=c.claim,
                metadata={
                    "source_path": f"research_claim://{c.created_at.isoformat()}",
                    "type": "research_claim",
                    "claim_type": c.type,
                    "confidence": c.confidence,
                    "is_evergreen": c.is_evergreen,
                    "source_urls": c.source_urls,
                    "created_at": c.created_at.isoformat(),
                },
            )
            for c in claims
        ]

        dense_vectors = self.dense_encoder.encode(chunks)
        sparse_vectors = self.sparse_encoder.encode(chunks)

        self.claim_store.upsert(chunks, dense_vectors)
        self.bm25_indexer.add_documents(sparse_vectors)

        return [c.id for c in chunks]

    async def write_insights(self, insights: list[Insight], maturity: str = "candidate") -> list[str]:
        """写入 insights collection"""
        if not insights:
            return []

        # 过滤低置信度
        insights = [i for i in insights if i.confidence >= 0.6]

        chunks = [
            Chunk(
                id=f"insight_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}",
                text=i.insight,
                metadata={
                    "source_path": f"research_insight://{i.created_at.isoformat()}",
                    "type": "research_insight",
                    "maturity": maturity,
                    "supporting_claims": i.supporting_claims,
                    "applicable_domains": i.applicable_domains,
                    "confidence": i.confidence,
                    "is_evergreen": i.is_evergreen,
                    "created_at": i.created_at.isoformat(),
                },
            )
            for i in insights
        ]

        dense_vectors = self.dense_encoder.encode(chunks)
        sparse_vectors = self.sparse_encoder.encode(chunks)

        self.insight_store.upsert(chunks, dense_vectors)
        self.bm25_indexer.add_documents(sparse_vectors)

        return [c.id for c in chunks]

    async def delete_insights(self, ids: list[str]):
        """删除 insights"""
        await self.insight_store.delete(ids=ids)
```

### 5.2 KnowledgeProcessor

```python
# nanobot/research/knowledge_processor.py

class KnowledgeProcessor:
    """研究结论写入知识库（职责单一，只做流程编排）"""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        knowledge_search: KnowledgeSearch,  # 封装了所有存储操作
        tracker: InsightTracker,
        similarity_threshold: float = 0.85,
    ):
        self.provider = provider
        self.model = model
        self.knowledge_search = knowledge_search
        self.tracker = tracker
        self.similarity_threshold = similarity_threshold

    async def _write_claims(self, claims: list[Claim]):
        """委托给 knowledge_search"""
        return await self.knowledge_search.write_claims(claims)

    async def _write_insights(self, insights: list[Insight], maturity: str):
        """委托给 knowledge_search"""
        return await self.knowledge_search.write_insights(insights, maturity)
```

### 5.3 InsightTracker

```python
# nanobot/research/insight_tracker.py

class InsightTracker:
    """跟踪未处理的 candidate insights"""

    def __init__(self, storage_path: str = "~/.nanoresearch/insight_tracker.json"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def add_candidate(self, insight_id: str, insight: dict):
        """新增 candidate"""
        data = self._load()
        data[insight_id] = {**insight, "added_at": datetime.now().isoformat()}
        self._save(data)

    def list_candidates(self) -> list[dict]:
        """列出所有未处理的 candidates"""
        data = self._load()
        return [v | {"id": k} for k, v in data.items()]

    def mark_processed(self, insight_id: str):
        """标记为已处理（从 tracker 移除）"""
        data = self._load()
        data.pop(insight_id, None)
        self._save(data)

    def _load(self) -> dict:
        if self.storage_path.exists():
            return json.loads(self.storage_path.read_text())
        return {}

    def _save(self, data: dict):
        self.storage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
```

---

## 六、Planner 预查询

```python
# nanobot/research/runner.py

async def run(self, topic: str, ...) -> ResearchResult:
    # Phase 0: 预查询已有知识
    existing_knowledge = await self._get_existing_knowledge(topic)

    # Phase 1: Planning (传入已有知识)
    plan = await self.planner.plan(topic, depth, existing_context=existing_knowledge)

    # ... 原有流程 ...

async def _get_existing_knowledge(self, topic: str) -> str:
    """预查询已有知识，分区呈现"""
    claims, insights = await self.knowledge_search.search_all(topic)

    context = ""

    if insights:
        context += "## 已有相关规律 (Insights)\n"
        context += "以下是从过往研究中提炼的跨域规律：\n"
        for i in insights:
            maturity_tag = "✓" if i.metadata.get("maturity") == "confirmed" else "?"
            context += f"- [{maturity_tag}] {i.text}\n"
        context += "\n"

    if claims:
        context += "## 已知相关事实 (Claims)\n"
        context += "以下是过往研究中的具体发现：\n"
        for c in claims:
            context += f"- {c.text}\n"

    return context
```

**Planner prompt 使用**：

```python
_USER_TEMPLATE = """## 研究方向
{topic}

## 研究深度
{depth}

## 已有知识
{existing_context}

## 任务
请生成增量子问题：
- 不要重复已有事实已经回答的问题
- 利用已有规律指导研究方向
- 深入未覆盖的领域

请调用 research_plan 工具返回规划结果。
"""
```

---

## 七、去重逻辑

### 7.1 中文支持

```python
def _is_same_claim(self, claim1: str, claim2: str) -> bool:
    """字符级 bigram，对中英文都有效"""

    def bigrams(s):
        s = s.strip()
        return set(zip(s[:-1], s[1:])) if len(s) > 1 else set()

    b1, b2 = bigrams(claim1), bigrams(claim2)
    if not b1 or not b2:
        return False

    jaccard = len(b1 & b2) / len(b1 | b2)
    return jaccard >= 0.6
```

### 7.2 向量相似度

```python
async def _check_claim_duplicates(self, claims: list[Claim]) -> tuple:
    """去重检查，使用向量相似度"""

    new_claims = []
    duplicate_count = 0
    conflict_count = 0

    for claim in claims:
        # 用 dense 检索，score 是余弦相似度 (0-1)
        results = await self.knowledge_search.search_claims(
            claim.claim, top_k=3, apply_decay=False
        )

        if not results:
            new_claims.append(claim)
            continue

        top_match = results[0]
        similarity = top_match["score"]

        if similarity >= self.similarity_threshold:  # 0.85
            if self._is_same_claim(claim.claim, top_match["text"]):
                duplicate_count += 1
            else:
                conflict_count += 1
        else:
            new_claims.append(claim)

    return new_claims, duplicate_count, conflict_count
```

---

## 八、写入知识库

写入逻辑已封装到 `KnowledgeSearch.write_claims()` 和 `KnowledgeSearch.write_insights()`，参见第五节。

`KnowledgeProcessor` 通过委托方式调用：

```python
async def _write_claims(self, claims: list[Claim]):
    return await self.knowledge_search.write_claims(claims)

async def _write_insights(self, insights: list[Insight], maturity: str):
    return await self.knowledge_search.write_insights(insights, maturity)
```

---

## 九、批量提炼

### 9.1 Prompt

```python
_BATCH_REFINE_PROMPT = """从以下候选 Insights 中提炼跨域规律。

## 候选 Insights (来自多次研究)
{candidates}

## 任务
1. 识别重复或相似的候选，合并为一条
2. 识别跨域模式，提炼为更通用的规律
3. 标注适用领域和边界条件

## 示例
候选1: "3DGS 对薄结构处理较差" (来自 3DGS 研究)
候选2: "NeRF 对透明物体有伪影" (来自 NeRF 研究)

→ 提炼: "基于点/高斯表示的渲染方法对薄结构和透明物体普遍处理较差，原因是缺乏体积约束"
   适用领域: ["3D重建", "渲染"]

## 输出格式
```json
{{
  "insights": [
    {{
      "insight": "提炼后的规律",
      "supporting_candidate_ids": ["id1", "id2"],
      "applicable_domains": ["domain1", "domain2"],
      "confidence": 0.9,
      "is_evergreen": true
    }}
  ]
}}
```
"""
```

### 9.2 完整流程

```python
async def batch_refine_insights(self) -> BatchRefineResult:
    """批量提炼 candidate → confirmed"""

    # 1. 读取所有未处理的 candidates
    candidates = self.tracker.list_candidates()
    if not candidates:
        return BatchRefineResult(processed=0, confirmed=0)

    # 2. LLM 提炼
    candidates_text = "\n".join(
        f"- [{c['id']}] {c['insight']}" for c in candidates
    )
    response = await self.provider.chat_with_retry(
        messages=[{"role": "user", "content": _BATCH_REFINE_PROMPT.format(
            candidates=candidates_text
        )}],
        model=self.model,
        ...
    )
    confirmed_insights = self._parse_refine_response(response)

    # 3. 写入 confirmed insights（先写，确保成功）
    confirmed_ids = await self._write_insights(confirmed_insights, maturity="confirmed")

    # 4. 删除旧 candidates（ChromaDB + Tracker 同步）
    old_ids = [c["id"] for c in candidates]
    await self.knowledge_search.delete_insights(ids=old_ids)
    for old_id in old_ids:
        self.tracker.mark_processed(old_id)

    logger.info(
        "Batch refine: {} candidates → {} confirmed insights",
        len(candidates), len(confirmed_ids)
    )

    return BatchRefineResult(processed=len(candidates), confirmed=len(confirmed_ids))
```

---

## 十、工作量估计

| 步骤 | 工作量 | 说明 |
|------|--------|------|
| `KnowledgeProcessor` 主类 | 4小时 | 流程编排 |
| `_extract_claims()` | 2小时 | LLM 提取 + Prompt |
| `_extract_insights()` | 2小时 | LLM 提炼 + Prompt |
| `KnowledgeSearch` 类 | 3小时 | 检索 + 时间衰减 + 分区 |
| `InsightTracker` 类 | 2小时 | JSON 索引管理 |
| `_write_claims/insights()` | 3小时 | dense + sparse 写入 |
| `_check_duplicates()` | 2小时 | 向量相似度 + bigram |
| `batch_refine_insights()` | 3小时 | 批量提炼 + 删除同步 |
| Planner 预查询 | 4小时 | 修改接口 + Prompt |
| 测试 | 4小时 | 单元测试 + 端到端 |

**总工作量**：约 **3.5 天**

---

## 十一、文件结构

```
nanobot/research/
├── knowledge_processor.py   # 新增：主处理器
├── knowledge_search.py      # 新增：知识检索
├── insight_tracker.py       # 新增：候选索引
├── runner.py                # 修改：加预查询 + 调用 processor
├── planner.py               # 修改：加 existing_context 参数
├── types.py                 # 修改：加 Claim, Insight 类型
└── ...
```

---

## 十二、验证方案

```python
# 测试即时阶段
async def test_immediate_processing():
    runner = ResearchRunner(..., knowledge_processor=processor)
    result = await runner.run("3DGS 的局限性")

    # 检查 claims 和 insights 写入
    claims = await knowledge_search.search_claims("3DGS 薄表面", top_k=5)
    assert len(claims) > 0

    insights = await knowledge_search.search_insights("点表示 渲染", top_k=3)
    assert len(insights) > 0
    assert insights[0].metadata["maturity"] == "candidate"

# 测试批量阶段
async def test_batch_refine():
    # 触发批量提炼
    result = await processor.batch_refine_insights()

    # 检查 candidate 被删除
    candidates = tracker.list_candidates()
    assert len(candidates) == 0

    # 检查 confirmed 写入
    confirmed = await knowledge_search.search_insights(
        "渲染 薄结构", top_k=3, maturity="confirmed"
    )
    assert len(confirmed) > 0

# 测试 Planner 预查询
async def test_planner_prequery():
    runner = ResearchRunner(..., knowledge_processor=processor)

    # 第一次研究
    await runner.run("3DGS 的局限性")

    # 第二次研究（应该能复用）
    result = await runner.run("NeRF vs 3DGS 对比")

    # 检查 plan 里没有重复已有知识的问题
    # ...
```
