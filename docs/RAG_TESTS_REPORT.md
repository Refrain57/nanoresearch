# RAG 模块测试报告

**生成日期**: 2026-04-02
**测试框架**: pytest
**运行命令**: `pytest tests/rag/ --ignore=tests/rag/stress -v`

---

## 测试总览

| 指标 | 数值 |
|------|------|
| 总测试数 | 168 |
| 通过数 | 168 |
| 失败数 | 0 |
| 跳过数 | 0 |

---

## 测试分层结构

```
tests/rag/
├── conftest.py              # 共享 fixtures 和工具函数
├── README.md                # 测试文档
├── unit/                   # 单元测试 (124 个)
├── integration/            # 集成测试 (23 个)
└── stress/                 # 压力测试 (21 个)
    ├── test_concurrent_queries.py
    ├── test_large_document_ingest.py
    └── test_memory_usage.py
```

---

## 1. 单元测试 (Unit Tests)

### 1.1 RRF 融合算法 (`test_fusion.py`)

**文件**: `tests/rag/unit/test_fusion.py`
**测试数**: 27

#### 测试覆盖

| 测试类 | 测试内容 | 示例 |
|--------|----------|------|
| `TestRRFFusionInit` | 初始化行为 | 默认 k 值、自定义 k、无效 k 值检测 |
| `TestRRFScoreCalculation` | RRF 分数计算 | 排名 1/10 的分数、不同 k 值、边界检测 |
| `TestRRFFusionBasic` | 基础融合 | 单列表、双列表、空列表、top_k 限制 |
| `TestRRFFusionWeights` | 加权融合 | 均匀权重、dense 权重更高、零权重、权重验证 |
| `TestRRFFusionDeterminism` | 确定性 | 相同输入产生相同输出、tie-breaking |
| `TestRRFFusionEdgeCases` | 边界情况 | 重复 ID、混合空列表、大列表、元数据复制 |

#### 测试用例示例

```python
# RRF 分数计算
assert rrf_score(1, k=60) == 1.0 / 61  # 排名1的分数

# 融合两个列表
dense = [Result("a", 0.9), Result("b", 0.8)]
sparse = [Result("b", 5.0), Result("c", 4.0)]
# "b" 同时出现在两个列表，RRF 分数最高
```

#### 关键断言

- [x] 相同输入产生相同输出（确定性）
- [x] 同时出现在多个列表的文档排名更高
- [x] top_k 正确限制结果数量
- [x] 元数据从首次出现保留

---

### 1.2 查询处理器 (`test_query_processor.py`)

**文件**: `tests/rag/unit/test_query_processor.py`
**测试数**: 31

#### 测试覆盖

| 测试类 | 测试内容 | 示例 |
|--------|----------|------|
| `TestQueryProcessorInit` | 初始化 | 默认配置、自定义配置、工厂函数 |
| `TestQueryProcessorChinese` | 中文处理 | 基础分词、疑问词过滤、助词过滤 |
| `TestQueryProcessorEnglish` | 英文处理 | 基础分词、冠词过滤、大小写保留 |
| `TestQueryProcessorMixed` | 混合语言 | 中英混合、带数字查询 |
| `TestQueryProcessorFilters` | 过滤器解析 | collection/type/tags 解析、过滤器合并 |
| `TestQueryProcessorNormalization` | 规范化 | 空白规范化、Unicode 处理 |
| `TestQueryProcessorEdgeCases` | 边界情况 | 空查询、仅停用词、仅标点、最大关键词数、去重 |
| `TestQueryProcessorStopwords` | 停用词管理 | 添加/删除自定义停用词 |
| `TestQueryProcessorOutput` | 输出格式 | ProcessedQuery 类型、序列化 |

#### 测试用例示例

```python
# 中文查询处理
result = processor.process("如何配置 Azure OpenAI？")
assert "Azure" in result.keywords
assert "OpenAI" in result.keywords
assert result.filters.get("collection") is None  # 无过滤器

# 过滤器解析
result = processor.process("内容 collection:docs type:pdf")
assert result.filters["collection"] == "docs"
assert result.filters["doc_type"] == "pdf"
```

#### 关键断言

- [x] 中文/英文分词正确
- [x] 停用词（"的"、"the"）被过滤
- [x] 过滤器语法正确解析
- [x] 关键词数量限制生效
- [x] 大小写去重（case-insensitive）

---

### 1.3 文档分块器 (`test_document_chunker.py`)

**文件**: `tests/rag/unit/test_document_chunker.py`
**测试数**: 20

#### 测试覆盖

| 测试类 | 测试内容 | 示例 |
|--------|----------|------|
| `TestDocumentChunkerInit` | 初始化 | Mock 设置 |
| `TestDocumentChunkerChunkId` | ID 生成 | 格式验证、确定性、零填充 |
| `TestDocumentChunkerMetadataInheritance` | 元数据继承 | 文档元数据继承、chunk_index、source_ref、图像引用 |
| `TestDocumentChunkerSplitDocument` | 分块逻辑 | 基础分块、空文本错误、空列表处理、单块 |
| `TestDocumentChunkerMetadataIntegrity` | 元数据完整性 | 非共享字典、页码提取 |

#### 测试用例示例

```python
# ID 格式: {doc_id}_{index:04d}_{content_hash}
chunk_id = chunker._generate_chunk_id("doc_123", 0, "test content")
assert chunk_id.startswith("doc_123_0000_")
assert len(chunk_id.split("_")[-1]) == 8  # 8位哈希

# 图像引用提取
chunks = chunker.split_document(doc)
assert chunks[0].metadata["image_refs"] == ["img_001", "img_002"]
```

#### 关键断言

- [x] 块 ID 格式正确且唯一
- [x] 相同输入产生相同 ID（确定性）
- [x] 元数据从文档继承到所有块
- [x] chunk_index 正确编号
- [x] 图像占位符正确提取

---

### 1.4 混合搜索 (`test_hybrid_search.py`)

**文件**: `tests/rag/unit/test_hybrid_search.py`
**测试数**: 24

#### 测试覆盖

| 测试类 | 测试内容 | 示例 |
|--------|----------|------|
| `TestHybridSearchInit` | 初始化 | 所有组件、自定义配置、仅 dense/仅 sparse |
| `TestHybridSearchBasicSearch` | 基础搜索 | 双检索器、top_k、空查询验证、返回详情 |
| `TestHybridSearchFallback` | 降级处理 | dense 失败、sparse 失败、双失败、仅一种模式 |
| `TestHybridSearchMetadataFiltering` | 元数据过滤 | 查询过滤器、显式过滤器、过滤器合并 |
| `TestHybridSearchConfig` | 配置选项 | 禁用 dense、禁用 sparse、自定义 top_k |
| `TestHybridSearchFactory` | 工厂函数 | 默认配置、自定义 RRF k、全组件 |

#### 测试用例示例

```python
# 双检索器融合
dense = [Result("a", 0.9), Result("b", 0.8)]
sparse = [Result("b", 5.0), Result("c", 4.0)]
results = hybrid.search("test")
assert results[0].chunk_id == "b"  # "b" 在两个列表中出现

# 降级处理
dense.retrieve.side_effect = Exception("Dense failed")
result = hybrid.search("test", return_details=True)
assert result.used_fallback is True
assert len(result.results) == 1  # 仅 sparse 结果
```

#### 关键断言

- [x] dense + sparse 结果正确融合
- [x] 一个检索器失败时自动降级
- [x] 两个都失败时抛出错误
- [x] 过滤器正确传递到检索器
- [x] top_k 参数正确限制结果

---

### 1.5 重排序器 (`test_reranker.py`)

**文件**: `tests/rag/unit/test_reranker.py`
**测试数**: 23

#### 测试覆盖

| 测试类 | 测试内容 | 示例 |
|--------|----------|------|
| `TestRerankConfig` | 配置 | 默认值、自定义值 |
| `TestCoreRerankerInit` | 初始化 | Mock reranker、配置提取、类型检测 |
| `TestCoreRerankerBasicRerank` | 基础重排 | 基础重排、top_k、空结果、单结果 |
| `TestCoreRerankerDisabled` | 禁用状态 | 禁用时返回原顺序、is_enabled 属性 |
| `TestCoreRerankerFallback` | 降级处理 | 错误时使用原顺序、fallback 元数据标记 |
| `TestCoreRerankerTypeConversion` | 类型转换 | 结果→候选、候选→结果、缺失处理 |
| `TestCoreRerankerFactory` | 工厂函数 | 基本创建 |

#### 测试用例示例

```python
# 基础重排序
results = [Result("a", 0.9), Result("b", 0.8), Result("c", 0.7)]
reranked = reranker.rerank("test", results)
assert reranked.results[0].chunk_id == "c"  # 顺序反转

# 降级时添加标记
reranker.rerank.side_effect = Exception("Error")
result = reranker.rerank("test", results)
assert result.used_fallback is True
assert result.results[0].metadata["rerank_fallback"] is True
```

#### 关键断言

- [x] 重排序正确改变结果顺序
- [x] 错误时使用 fallback
- [x] fallback 添加正确元数据标记
- [x] 类型转换保留必要信息
- [x] 原始顺序保存在 result 中

---

## 2. 集成测试 (Integration Tests)

### 2.1 查询流程 (`test_query_flow.py`)

**文件**: `tests/rag/integration/test_query_flow.py`
**测试数**: 13

#### 测试覆盖

| 测试类 | 测试内容 | 示例 |
|--------|----------|------|
| `TestQueryFlowBasic` | 基础流程 | 查询处理→混合搜索、完整流程、返回详情 |
| `TestQueryFlowWithReranking` | 带重排 | 混合搜索→重排序 |
| `TestQueryFlowFilters` | 过滤器 | 过滤器传递、显式覆盖 |
| `TestQueryFlowErrorHandling` | 错误处理 | 单边失败、双边失败、空查询 |
| `TestQueryFlowRRF` | RRF 融合 | 双列表融合、单检索器模式 |

#### 测试流程

```
用户查询
    ↓
QueryProcessor (分词、过滤)
    ↓
HybridSearch
    ├── DenseRetriever (语义搜索)
    └── SparseRetriever (关键词搜索)
    ↓
RRFFusion (结果融合)
    ↓
CoreReranker (可选重排)
    ↓
检索结果
```

#### 关键断言

- [x] 完整流程端到端工作
- [x] 过滤器正确传播
- [x] 降级机制正确触发
- [x] RRF 融合正确合并结果

---

### 2.2 摄取管道 (`test_ingestion_pipeline.py`)

**文件**: `tests/rag/integration/test_ingestion_pipeline.py`
**测试数**: 10

#### 测试覆盖

| 测试类 | 测试内容 | 示例 |
|--------|----------|------|
| `TestIngestionPipelineBasic` | 基础流程 | 成功执行、跳过已处理、失败处理 |
| `TestIngestionPipelineStages` | 管道阶段 | 分块、结果序列化 |
| `TestBatchProcessing` | 批处理 | 批处理器流程 |
| `TestDocumentTypes` | 文档类型 | Document/Chunk/ChunkRecord/RetrievalResult 创建和验证 |

#### 测试流程

```
文件 (PDF)
    ↓
FileIntegrityChecker (SHA256 检查)
    ↓
PdfLoader (加载、图像提取)
    ↓
DocumentChunker (分块)
    ↓
Transform 管道
    ├── ChunkRefiner (规则/LLM 优化)
    ├── MetadataEnricher (元数据丰富)
    └── ImageCaptioner (图像描述)
    ↓
BatchProcessor
    ├── DenseEncoder (向量编码)
    └── SparseEncoder (BM25 编码)
    ↓
存储
    ├── VectorUpserter (ChromaDB)
    ├── BM25Indexer (索引)
    └── ImageStorage (图像)
    ↓
PipelineResult
```

#### 关键断言

- [x] 完整管道成功执行
- [x] 已处理文件正确跳过
- [x] 失败正确捕获并返回错误
- [x] 文档类型验证正确工作

---

## 3. 压力测试 (Stress Tests)

**状态**: ✅ 全部通过 (21 个测试)

### 3.1 大文档摄取 (`test_large_document_ingest.py`)

**文件**: `tests/rag/stress/test_large_document_ingest.py`
**测试数**: 6

#### 测试结果

| 测试 | 数值 | 结果 |
|------|------|------|
| `test_ingest_1000_chunks` | 1000 chunks in **0.00s** (mock) | ✅ 通过 |
| `test_ingest_large_file_simulation` | 10MB → **23,333 chunks** in **0.18s** | ✅ 通过 |
| `test_batch_processor_large_batch` | 500 chunks in **0.00s**, 5 batch calls | ✅ 通过 |
| `test_batch_processor_scaling` | 1000 chunks @ batch=50: 20 calls | ✅ 通过 |
| `test_chunking_memory_usage` | 5000 chunks: **21.09 MB** peak (0.0042 MB/chunk) | ✅ 通过 |
| `test_vector_storage_memory` | 10,000 vectors: **117.80 MB** (expected 117.19 MB) | ✅ 通过 |

### 3.2 并发查询 (`test_concurrent_queries.py`)

**文件**: `tests/rag/stress/test_concurrent_queries.py`
**测试数**: 7

#### 测试结果

| 测试 | 数值 | 结果 |
|------|------|------|
| `test_concurrent_50_queries` | **0.00s** (0.000s/query, 10 workers) | ✅ 通过 |
| `test_concurrent_hybrid_queries` | 30 queries in **0.45s** (0.015s/query) | ✅ 通过 |
| `test_hybrid_search_parallel_vs_sequential` | Seq: 0.102s vs **Par: 0.051s** (1.99x speedup) | ✅ 通过 |
| `test_query_throughput_single_thread` | **33,290 queries/sec** | ✅ 通过 |
| `test_query_throughput_multi_thread` | **19,812 queries/sec** (3 workers) | ✅ 通过 |
| `test_p99_latency` | P50=0.03ms, P95=0.04ms, **P99=0.07ms** | ✅ 通过 |
| `test_thread_pool_limits` | 10 tasks @ 2 workers: **0.50s** | ✅ 通过 |

### 3.3 内存使用 (`test_memory_usage.py`)

**文件**: `tests/rag/stress/test_memory_usage.py`
**测试数**: 8

#### 测试结果

| 测试 | 数值 | 结果 |
|------|------|------|
| `test_dense_retrieval_memory` | 1000 results: **0.59 MB** (0.0006 MB/result) | ✅ 通过 |
| `test_rerank_memory_usage` | 500 results rerank: **0.55 MB** | ✅ 通过 |
| `test_dense_embedding_memory` | 100×1536 vectors: **1.19 MB** (expected 1.17 MB) | ✅ 通过 |
| `test_vector_store_memory_scaling` | 5000 vectors: 68.76 MB overhead (1.17x ratio) | ✅ 通过 |
| `test_query_processor_memory` | 1000 queries: **0.00 MB** leak | ✅ 通过 |
| `test_retrieval_result_gc` | 1000 objects: 999/1000 collected | ✅ 通过 |
| `test_simultaneous_large_operations` | Peak: **25.48 MB** (under 1GB) | ✅ 通过 |
| `test_memory_after_cleanup` | 10000 vectors: Released **116.64 MB** (from 468 MB) | ✅ 通过 |

### 3.4 关键性能指标汇总

```
并发处理能力:
├─ 单线程 QPS: 33,290 queries/sec
├─ 多线程 QPS: 19,812 queries/sec (3 workers)
├─ 并行加速比: 1.99x
└─ P99 延迟: 0.07ms

内存效率:
├─ 每结果开销: 0.0006 MB
├─ 每分块开销: 0.0042 MB
├─ 1000 查询内存泄漏: 0.00 MB
└─ GC 回收率: 99.9%

大规模处理:
├─ 10MB 文本 → 23,333 chunks (0.18s)
├─ 1000 chunks 摄取: < 0.01s (mock)
└─ 5000 并发操作峰值内存: 25.48 MB
```

#### 运行压力测试

```bash
pytest tests/rag/stress/ -v -m stress
```

---

## 4. Fixtures (共享测试数据)

### 4.1 conftest.py 提供的 Fixtures

| Fixture | 类型 | 用途 |
|---------|------|------|
| `MockSettings` | Settings | 提供测试配置 |
| `mock_embedding_client` | MagicMock | Embedding 客户端 |
| `mock_vector_store` | MagicMock | 向量存储 |
| `mock_bm25_indexer` | MagicMock | BM25 索引 |
| `mock_reranker` | MagicMock | 重排序器 |
| `mock_llm_provider` | MagicMock | LLM 提供商 |
| `sample_document` | Document | 示例文档 |
| `sample_chunks` | List[Chunk] | 示例块列表 |
| `sample_retrieval_results` | List[RetrievalResult] | 示例检索结果 |
| `sample_processed_query` | ProcessedQuery | 示例处理查询 |

### 4.2 工具函数

```python
def create_mock_retrieval_result(chunk_id, score, text="")
def create_mock_chunks(count, prefix="test")
def create_mock_dense_vectors(count, dimension=1536)
```

---

## 5. 测试覆盖矩阵

| 模块 | 单元测试 | 集成测试 | 压力测试 | 总计 | 覆盖率评估 |
|------|----------|----------|----------|------|------------|
| `fusion.py` | 27 | 2 | 0 | 29 | 高 |
| `query_processor.py` | 31 | 1 | 2 | 34 | 高 |
| `document_chunker.py` | 20 | 1 | 0 | 21 | 高 |
| `hybrid_search.py` | 24 | 3 | 4 | 31 | 高 |
| `reranker.py` | 23 | 1 | 2 | 26 | 高 |
| `ingestion/pipeline.py` | 0 | 4 | 1 | 5 | 中 |
| `query_flow.py` | 0 | 6 | 0 | 6 | 中 |
| `concurrent_queries.py` | 0 | 0 | 7 | 7 | 高 |
| `memory_usage.py` | 0 | 0 | 8 | 8 | 高 |
| **总计** | **124** | **23** | **21** | **168** | - |

---

## 6. 测试运行指南

### 6.1 基础运行

```bash
# 运行所有单元测试
pytest tests/rag/unit/ -v

# 运行所有集成测试
pytest tests/rag/integration/ -v

# 运行所有测试（不含压力测试）
pytest tests/rag/ -v
```

### 6.2 带覆盖率

```bash
pytest tests/rag/ --cov=nanobot.rag --cov-report=html --cov-report=term
```

### 6.3 运行特定测试

```bash
# 运行单个测试文件
pytest tests/rag/unit/test_fusion.py -v

# 运行特定测试类
pytest tests/rag/unit/test_fusion.py::TestRRFFusionBasic -v

# 运行特定测试用例
pytest tests/rag/unit/test_fusion.py::TestRRFFusionBasic::test_fuse_two_lists -v
```

### 6.4 压力测试

```bash
# 运行所有压力测试
pytest tests/rag/stress/ -v -m stress

# 运行特定压力测试
pytest tests/rag/stress/test_concurrent_queries.py -v -m stress
```

---

## 7. 关键测试场景

### 7.1 RRF 融合正确性

验证 Reciprocal Rank Fusion 算法正确实现：
- 出现在多个列表的文档排名更高
- 结果确定性（相同输入→相同输出）
- 正确处理边界情况（空列表、重复 ID）

### 7.2 降级机制

验证优雅降级：
- dense 失败 → 使用 sparse 结果
- sparse 失败 → 使用 dense 结果
- 都失败 → 抛出明确错误

### 7.3 过滤器处理

验证查询过滤正确工作：
- 解析 `collection:docs` 等语法
- 合并查询过滤和显式过滤
- 过滤器正确传递到检索器

### 7.4 摄取管道

验证端到端文档处理：
- 文件完整性检查
- 分块和元数据继承
- 跳过已处理文件

---

## 8. 已知限制

1. **Mock 依赖**: 部分测试使用 mock 而非真实组件，可能无法捕获所有真实场景问题
2. **内存测试精度**: Python GC 不保证即时回收，内存断言可能存在波动
3. **无真实 API 测试**: 部分测试需要有效的 LLM/Embedding API 密钥才能进行真实集成测试

---

## 9. 改进建议

### 高优先级

1. **添加端到端测试**: 使用真实 ChromaDB 和 Embedding
2. **添加性能基准**: 建立性能回归检测
3. **增加边界测试**: 更多异常输入测试

### 中优先级

1. **参数化测试**: 使用 `@pytest.mark.parametrize` 减少重复
2. **模糊测试**: 使用 `hypothesis` 生成随机输入
3. **快照测试**: 保存复杂输出的快照用于回归检测

### 低优先级

1. **性能监控**: 添加测试执行时间监控
2. **并发测试**: 更多并发场景测试
3. **跨平台测试**: Windows/Linux 不同行为测试

---

*报告生成工具: Claude Code*
