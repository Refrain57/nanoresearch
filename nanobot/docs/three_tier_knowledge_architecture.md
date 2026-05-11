# 三层知识架构改造计划

## 目标架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Insight（跨域规律）                          │
│                  supporting_claim_ids: [id1, id2, ...]          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Claim（原子索引）                            │
│                  evidence_ids: [chunk_id1, chunk_id2, ...]    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RAG Chunk（原始证据）                          │
│                  metadata: {source_type: "research" | "document"}│
└─────────────────────────────────────────────────────────────────┘
```

## 两个写入入口

### 入口 1: research 完成
```
report.md
├── 切分 chunks → 写入 RAG（metadata.source_type = "research"）
└── 提取 Claims → 写入 research_claims（带 evidence_ids）
    └── 提取 Insights → 写入 research_insights（带 supporting_claim_ids）
```

### 入口 2: PDF 手动导入
```
PDF 文档
├── 切分 chunks → 写入 RAG（metadata.source_type = "document"）
└── 提取 Claims → 写入 research_claims（带 evidence_ids）
```

---

## 改动清单

### Step 1: types.py - 改造类型定义

**文件**: `nanobot/research/types.py`

**Claim 类** - 新增 evidence_ids 字段：
```python
@dataclass
class Claim:
    claim: str
    type: str
    is_evergreen: bool
    source_urls: list[str]
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)  # 新增
    created_at: datetime = field(default_factory=datetime.now)
```

**Insight 类** - supporting_claims 改为 ID：
```python
@dataclass
class Insight:
    insight: str
    supporting_claim_ids: list[str] = field(default_factory=list)  # 改名
    applicable_domains: list[str]
    confidence: float
    is_evergreen: bool
    maturity: str = "candidate"
    created_at: datetime = field(default_factory=datetime.now)
```

---

### Step 2: knowledge_search.py - 写入时保存 evidence_ids 和 supporting_claim_ids

**文件**: `nanobot/research/knowledge_search.py`

**write_claims()** - 写入 metadata 时加上 evidence_ids：
```python
records.append({
    "id": record_id,
    "vector": self._embed(c.claim),
    "metadata": {
        "type": "research_claim",
        "claim_type": c.type,
        "confidence": c.confidence,
        "is_evergreen": c.is_evergreen,
        "source_urls": c.source_urls,
        "evidence_ids": c.evidence_ids,  # 新增
        "created_at": c.created_at.isoformat(),
        "text": c.claim,
    },
})
```

**write_insights()** - 写入 metadata 时加上 supporting_claim_ids：
```python
records.append({
    "id": record_id,
    "vector": self._embed(i.insight),
    "metadata": {
        "type": "research_insight",
        "maturity": maturity,
        "supporting_claim_ids": i.supporting_claim_ids,  # 改名
        ...
    },
})
```

---

### Step 3: knowledge_processor.py - 新增 RAG 写入和关联逻辑

**文件**: `nanobot/research/knowledge_processor.py`

**3.1 新增 `__init__` 参数**：
```python
def __init__(
    self,
    provider: LLMProvider,
    model: str,
    knowledge_search: KnowledgeSearch,
    tracker: InsightTracker,
    correction_tracker: "CorrectionTracker | None" = None,
    similarity_threshold: float = 0.85,
    rag_store: Any = None,  # 新增：RAG store
):
    self.rag_store = rag_store
```

**3.2 新增 `_write_report_to_rag()` 方法**：
```python
async def _write_report_to_rag(self, report: str, topic: str) -> list[str]:
    """将报告切分为 chunks 并写入 RAG"""
    if not report or not self.rag_store:
        return []

    # 按段落切分
    chunks = self._split_into_chunks(report)

    records = []
    for chunk in chunks:
        chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
        records.append({
            "id": chunk_id,
            "vector": self.knowledge_search._embed(chunk),
            "metadata": {
                "source_type": "research",
                "topic": topic,
                "text": chunk,
                "created_at": datetime.now().isoformat(),
            },
        })

    if records:
        self.rag_store.upsert(records)
        logger.info(f"KnowledgeProcessor: wrote {len(records)} chunks to RAG")

    return [r["id"] for r in records]
```

**3.3 新增 `_split_into_chunks()` 方法**：
```python
def _split_into_chunks(self, text: str, max_chars: int = 1000) -> list[str]:
    """按段落切分文本"""
    # 按换行分割段落
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks
```

**3.4 新增 `_match_chunks_to_claims()` 方法（向量匹配）**：
```python
async def _match_chunks_to_claims(
    self, claims: list[Claim], chunk_ids: list[str]
) -> dict[str, list[str]]:
    """用向量相似度匹配 Claims 和 Chunks"""
    if not claims or not chunk_ids or not self.rag_store:
        return {c.claim: [] for c in claims}

    # 获取所有 chunk 的向量
    chunk_vectors = {}
    for cid in chunk_ids:
        result = self.rag_store.get_by_id(cid)
        if result:
            chunk_vectors[cid] = result.get("vector", [])

    # 为每个 claim 找最相似的 chunk
    claim_chunk_map = {}
    for claim in claims:
        claim_vec = self.knowledge_search._embed(claim.claim)
        best_chunk_id = None
        best_score = -1

        for cid, chunk_vec in chunk_vectors.items():
            score = self._cosine_sim(claim_vec, chunk_vec)
            if score > best_score:
                best_score = score
                best_chunk_id = cid

        claim_chunk_map[claim.claim] = [best_chunk_id] if best_chunk_id else []

    return claim_chunk_map
```

**3.5 改造 `process()` 方法顺序**：
```python
async def process(self, result: ResearchResult) -> KnowledgeProcessResult:
    logger.info(f"KnowledgeProcessor: processing result for topic '{result.topic}'")

    # Step 1: 切分 report → 写入 RAG → 拿到 chunk_ids
    chunk_ids = await self._write_report_to_rag(result.report, result.topic)

    # Step 2: Extract Claims
    claims = await self._extract_claims(result)

    # Step 3: 用向量匹配关联 evidence_ids
    if chunk_ids:
        claim_chunk_map = await self._match_chunks_to_claims(claims, chunk_ids)
        for claim in claims:
            claim.evidence_ids = claim_chunk_map.get(claim.claim, [])

    if not claims:
        logger.info("KnowledgeProcessor: no claims extracted")
        return KnowledgeProcessResult(...)

    # Step 4: Write Claims → 拿到 claim_ids
    claim_ids = await self.knowledge_search.write_claims(claims)

    # Step 5: Extract Insights（用 claim_ids）
    insights = await self._extract_insights(claims, claim_ids, result.topic)

    # Step 6: Write Insights
    if insights:
        await self.knowledge_search.write_insights(insights)

    return KnowledgeProcessResult(...)
```

**3.6 改造 `_extract_insights()` 方法签名**：
```python
async def _extract_insights(
    self, claims: list[Claim], claim_ids: list[str], topic: str
) -> list[Insight]:
    """提取 Insights，claims 用 ID 而不是文本"""
    # 在 prompt 中提供 claim ID 和内容的映射
```

**3.7 改造 `_assess_insight_confidence()` - 用 claim_ids 匹配**：
```python
# 原来：用文本匹配
for claim_text in insight.supporting_claims:
    for claim in claims:
        if claim_text in claim.claim:

# 改成：用 ID 匹配（需要先建立 ID 到文本的映射）
claim_id_to_text = {f"claim_{i}": c.claim for i, c in enumerate(claims)}
claim_id_to_conf = {f"claim_{i}": c.confidence for i, c in enumerate(claims)}

for claim_id in insight.supporting_claim_ids:
    if claim_id in claim_id_to_conf:
        supporting_scores.append(claim_id_to_conf[claim_id])
```

---

### Step 4: collections.py (IngestDocumentTool) - PDF 导入后提取 Claims

**文件**: `nanobot/rag/mcp_server/tools/agentic/collections.py`

**IngestDocumentTool.execute()** - 导入完成后提取 Claims：
```python
# 在 _ingest() 完成后，调用 KnowledgeProcessor
def _ingest():
    pipeline = IngestionPipeline(...)
    result = pipeline.run(str(path))
    return result

# 改造：在后台任务中调用 KnowledgeProcessor
def _ingest_and_process():
    pipeline = IngestionPipeline(...)
    result = pipeline.run(str(path))

    # 提取 Claims（如果配置了 KnowledgeProcessor）
    if hasattr(self, 'knowledge_processor') and self.knowledge_processor:
        # 模拟一个 ResearchResult
        from nanobot.research.types import ResearchResult, ResearchStatus
        mock_result = ResearchResult(
            topic=f"Imported: {path.name}",
            status=ResearchStatus.COMPLETED,
            report=result.doc_id,  # 存储 doc_id
        )
        # 这里需要传递 doc_id 和 chunk_ids
        await self.knowledge_processor.process_document_chunks(
            doc_id=result.doc_id,
            chunk_ids=[],  # 从 pipeline 结果获取
            topic=f"Imported: {path.name}",
        )

    return result
```

---

## 验证方案

### 1. 研究完成后检查
```
nanobot
> 研究 3DGS 的局限性
```
检查日志：
- RAG 是否写入了 chunks（source_type = "research"）
- Claims 是否有 evidence_ids
- Insights 是否有 supporting_claim_ids

### 2. ChromaDB 查询验证
```python
# 查询 claims 是否有 evidence_ids
claims = chromadb_client.get_collection("research_claims")
claim = claims.get(limit=1)
print(claim["metadata"]["evidence_ids"])

# 查询 insights 是否有 supporting_claim_ids
insights = chromadb_client.get_collection("research_insights")
insight = insights.get(limit=1)
print(insight["metadata"]["supporting_claim_ids"])
```

### 3. PDF 导入验证
```
nanobot
> 把 xxx.pdf 导入知识库
```
检查：
- RAG 是否写入了 PDF chunks（source_type = "document"）
- Claims 是否有 evidence_ids

---

## 文件改动汇总

| 文件 | 改动 |
|------|------|
| `nanobot/research/types.py` | Claim 新增 evidence_ids；Insight 改名 supporting_claim_ids |
| `nanobot/research/knowledge_search.py` | 写入时保存 evidence_ids 和 supporting_claim_ids |
| `nanobot/research/knowledge_processor.py` | 新增 RAG 写入、向量匹配、调整 process() 顺序 |
| `nanobot/rag/mcp_server/tools/agentic/collections.py` | PDF 导入后调用 KnowledgeProcessor |

---

## 依赖关系

```
knowledge_processor.py
├── knowledge_search.py（写入 claims/insights）
├── rag_store（写入 chunks，需要注入）
└── embedding（向量匹配）
```

需要确保 KnowledgeProcessor 能访问 RAG store。
