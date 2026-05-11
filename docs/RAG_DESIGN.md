# NanoResearch RAG 模块设计文档

> 本文档详细记录 NanoResearch 的 RAG（Retrieval-Augmented Generation）模块架构设计，包括数据流、核心组件、工具列表和设计决策。

---

## 目录

- [总体架构](#总体架构)
- [核心数据类型](#核心数据类型)
- [Ingestion Pipeline（入库流水线）](#ingestion-pipeline入库流水线)
- [Retrieval Pipeline（检索流水线）](#retrieval-pipeline检索流水线)
- [MCP Server Tools（14 个 Agentic 工具）](#mcp-server-tools14-个-agentic-工具)
- [Session 管理](#session-管理)
- [配置系统](#配置系统)
- [设计模式与原则](#设计模式与原则)

---

## 总体架构

### 分层架构

```
nanobot/rag/
├── core/              # 核心业务逻辑层
│   ├── types.py       # 核心数据契约（Document, Chunk, ChunkRecord, RetrievalResult）
│   ├── types_agentic.py  # Agentic 专用类型（SearchSession, FusionResult 等）
│   ├── settings.py    # 配置管理与验证
│   ├── query_engine/  # 检索编排
│   ├── session/       # Session 存储抽象
│   └── response/      # 响应构建与引用
│
├── ingestion/         # 文档处理流水线
│   ├── document_manager.py  # 跨存储生命周期管理
│   ├── chunking/      # 文档分块
│   ├── embedding/     # 稠密/稀疏编码
│   ├── storage/       # 向量写入 & BM25 索引
│   └── transform/     # 元数据增强、图像描述
│
├── libs/              # 外部集成（可插拔 Provider）
│   ├── embedding/     # Azure, OpenAI, Ollama embeddings
│   ├── llm/           # LLM providers（用于 reranking/verification）
│   ├── vector_store/  # ChromaDB（可扩展）
│   ├── reranker/      # Cross-encoder & LLM rerankers
│   └── loader/        # PDF, Markdown 加载器
│
├── mcp_server/        # MCP 协议实现
│   ├── tools/         # Agentic 工具
│   └── protocol_handler.py
│
└── internal_loop/     # 内部推理循环
```

### 数据流概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Ingestion Pipeline                                 │
│  Document (Loader) → Chunking → Transform → Embedding → Storage            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Retrieval Pipeline                                 │
│  Query → QueryProcessor → Dense/Sparse (并行) → RRF Fusion → Expansion      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Agentic Tools (MCP)                                │
│  plan_query → retrieve_* → fuse → rerank → verify → build_citations        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心数据类型

### 基础类型（`core/types.py`）

| 类型 | 用途 | 关键字段 |
|------|------|----------|
| `Document` | 原始加载的文档 | `id`, `text`, `metadata`（必须包含 `source_path`） |
| `Chunk` | 分块后的文本段 | `id`, `text`, `metadata`, `start_offset`, `end_offset`, `source_ref` |
| `ChunkRecord` | 完全处理后待存储 | 增加 `dense_vector`, `sparse_vector` |
| `ProcessedQuery` | 预处理后的查询 | `original_query`, `keywords`, `filters`, `expanded_terms` |
| `RetrievalResult` | 统一检索输出 | `chunk_id`, `score`, `text`, `metadata` |

### Agentic 类型（`core/types_agentic.py`）

| 类型 | 用途 |
|------|------|
| `FusionResult` | 融合操作输出，包含 method、weights、RRF 参数 |
| `VerificationResult` | LLM 验证结果，包含 confidence 评分 |
| `QueryPlan` | 查询分析，包含分解和策略建议 |
| `SearchSession` | 多轮 Session 状态，包含查询历史和累积结果 |

### 类型设计原则

- 所有类型都是 `@dataclass`
- 支持 `to_dict()` / `from_dict()` 序列化
- `__post_init__` 验证必填字段
- 元数据可扩展（最小必填 + 灵活扩展）

---

## Ingestion Pipeline（入库流水线）

### 流水线流程

```
Document (Loader)
    ↓
Chunking (DocumentChunker)
    ↓
Transform (MetadataEnricher, ImageCaptioner, ChunkRefiner)
    ↓
Embedding (DenseEncoder + SparseEncoder)
    ↓
Storage (VectorUpserter + BM25Indexer + ImageStorage)
```

### DocumentChunker

**两种策略**：
1. `fixed`：传统递归分块
2. `document_based`：结构感知分块

**自动检测逻辑**（`detect_chunk_strategy`）：
1. PDF bookmarks/TOC（最可靠）
2. Markdown 标题层级
3. 编号章节模式（1.1, 1.2, etc.）

**Chunk ID 生成**：确定性 ID，格式 `{doc_id}_{index:04d}_{content_hash}`

**元数据继承**：自动传播 `chunk_index`、`image_refs`、`prev_chunk_id`、`next_chunk_id`

### DenseEncoder

- 批处理，可配置 batch size
- 依赖注入 `BaseEmbedding` provider
- 验证向量维度一致性

### SparseEncoder

- 使用 `jieba` 进行中英文分词
- 输出词频用于 BM25 索引
- 计算语料级统计（平均文档长度、文档频率）

### DocumentManager

协调 4 个存储的生命周期：
- ChromaDB（向量存储）
- BM25（稀疏索引）
- ImageStorage（图像存储）
- FileIntegrity（文件完整性）

`delete_document()` 跨存储级联删除，支持部分失败容错。

---

## Retrieval Pipeline（检索流水线）

### Hybrid Search 架构

```
Query
    ↓
QueryProcessor（关键词提取、过滤器解析）
    ↓
┌─────────────────┬─────────────────┐
│  DenseRetriever │  SparseRetriever │  （并行执行）
└─────────────────┴─────────────────┘
    ↓
RRFFusion（Reciprocal Rank Fusion）
    ↓
Structure Expansion（邻居、兄弟节点）
    ↓
RetrievalResult[]
```

### DenseRetriever

- 使用配置的 embedding provider 对查询编码
- 查询向量存储获取相似向量
- 返回标准化的 `RetrievalResult`

### SparseRetriever

- 使用 BM25 indexer 进行关键词匹配
- 按 chunk ID 从向量存储获取文本/元数据
- 始终从磁盘重新加载 BM25 索引以确保一致性

### HybridSearch

**核心特性**：
1. 使用 `ThreadPoolExecutor` 并行检索
2. 优雅降级：一条路径失败时回退到另一条
3. 结构感知扩展：添加邻居 chunk 提供上下文
4. 后融合元数据过滤

**RRF 公式**：
```
RRF_score(d) = Σ 1/(k + rank(d))
```
默认 k=60（来自原始论文）

### 结构感知扩展

融合后自动扩展：
- **邻居 chunk**：`prev_chunk_id`、`next_chunk_id`
- **父级章节**：背景上下文

扩展结果标记 `is_neighbor: True`，评分较低，排在主结果之后。

---

## MCP Server Tools（14 个 Agentic 工具）

### 工具分类

#### 检索工具（5 个）

| 工具 | 用途 |
|------|------|
| `retrieve_dense` | 纯语义搜索（embedding） |
| `retrieve_sparse` | 纯 BM25 关键词搜索 |
| `retrieve_hybrid` | 组合 dense + sparse + RRF 融合（推荐默认） |
| `fetch_section` | 按章节路径获取 chunk |
| `fetch_neighbors` | 获取 prev/next chunk 提供上下文 |

#### 批量并发工具（3 个）

| 工具 | 用途 |
|------|------|
| `execute_retrieval_batch` | 并发执行多个检索任务 |
| `get_round_status` | 查询 round 状态 |
| `fuse_and_fetch_round` | 融合并返回完整结果 |

#### 融合与重排工具（2 个）

| 工具 | 用途 |
|------|------|
| `fuse_results` | 手动融合，可配置方法（rrf/weighted/interleave） |
| `rerank_results` | 应用 cross-encoder 重排 |

#### 查询规划工具（2 个）

| 工具 | 用途 |
|------|------|
| `plan_query` | 分析查询复杂度、建议策略、分解多部分查询 |
| `process_query` | 提取关键词和过滤器 |

#### 验证工具（1 个）

| 工具 | 用途 |
|------|------|
| `verify_results` | LLM 评估：answered?、confidence、missing_aspects、next_actions |

#### 引用工具（1 个）

| 工具 | 用途 |
|------|------|
| `build_citations` | 生成结构化引用（structured/markdown/numbered 格式） |

#### 集合管理工具（5 个）

| 工具 | 用途 |
|------|------|
| `list_collections` | 列出可用的知识库集合 |
| `list_documents` | 列出集合中的文档 |
| `ingest_document` | 异步文档入库（返回 task_id） |
| `delete_document` | 从所有存储中删除文档 |
| `get_task_status` | 检查异步任务进度 |

#### Session 管理工具（5 个）

| 工具 | 用途 |
|------|------|
| `create_session` | 创建多轮搜索 session |
| `get_session` | 获取 session 状态 |
| `update_session` | 添加查询/结果到 session |
| `close_session` | 结束 session |
| `list_sessions` | 列出活跃 session |

### 批量检索工作流

```
1. execute_retrieval_batch(tasks) → 获得 round_id
   - 自动创建 round（如果未提供）
   - 并发执行所有任务
   - 返回摘要（不含完整文本，保护上下文）

2. get_round_status(round_id) → 检查进度
   - 任务数、chunk 数
   - 每个 task 的详情
   - 是否已融合

3. fuse_and_fetch_round(round_id) → 获取完整结果
   - 应用融合策略
   - 返回完整文本
   - 标记 round 为已融合
```

### ConcurrentRetrievalEngine

**设计要点**：
- 使用 `asyncio.gather` 并发执行
- 组件缓存（避免为同一 collection 创建多个 ChromaDB 客户端）
- 线程安全缓存访问（`threading.Lock`）
- 失败时存储空结果（不阻塞其他任务）

---

## Session 管理

### 架构

```
SessionStore (ABC)
    ├── MemorySessionStore  （内存，用于测试）
    └── FileSessionStore    （JSON 持久化）
```

### SessionStore 接口

```python
class SessionStore(ABC):
    def create(session) -> None
    def get(session_id) -> SearchSession
    def update(session) -> None
    def delete(session_id) -> None
    def list_active() -> List[SearchSession]
    def cleanup_expired(max_age_seconds) -> int
```

### SearchSession 状态

```python
@dataclass
class SearchSession:
    session_id: str
    created_at: datetime
    initial_query: str
    collection: str
    
    # 查询历史
    current_query: str
    query_history: List[str]
    refined_queries: List[str]
    
    # 结果累积
    retrieval_results: Dict[str, List[Dict]]  # 按 dense/sparse 分类
    all_results: List[Dict]  # 跨查询累积
    fusion_results: Optional[List[Dict]]
    reranked_results: Optional[List[Dict]]
    
    # 验证与引用
    verified: bool
    verification_result: Optional[Dict]
    citations: List[Dict]
```

---

## 配置系统

### 配置结构

```yaml
llm:           # provider, model, temperature, max_tokens, api_key
embedding:     # provider, model, dimensions
vector_store:  # provider, persist_directory, collection_name
retrieval:     # dense_top_k, sparse_top_k, fusion_top_k, rrf_k
rerank:        # enabled, provider, model, top_k
evaluation:    # enabled, provider, metrics
observability: # log_level, trace_enabled, trace_file
ingestion:     # chunk_size, chunk_overlap, splitter, batch_size, chunk_strategy
vision_llm:    # enabled, provider, model, max_image_size
```

### 设计原则

- **验证**：严格类型检查，抛出 `SettingsError`
- **路径解析**：`resolve_path()` 处理相对/绝对/~ 路径
- **单例缓存**：`get_settings()` 加载一次并缓存
- **环境覆盖**：`RAG_SETTINGS_PATH` 指定配置文件位置

---

## 设计模式与原则

### 可插拔 Provider 模式

所有外部依赖使用抽象基类：

| 抽象基类 | 用途 |
|----------|------|
| `BaseEmbedding` | embedding providers |
| `BaseVectorStore` | 向量数据库 |
| `BaseLLM` / `BaseVisionLLM` | LLM providers |
| `BaseReranker` | 重排模型 |
| `BaseSplitter` | 文本分割器 |

### 工厂模式

| 工厂 | 创建对象 |
|------|----------|
| `EmbeddingFactory` | embedding provider |
| `VectorStoreFactory` | vector store |
| `SplitterFactory` | text splitter |
| `LLMFactory` | LLM provider |
| `RerankerFactory` | reranker |

### 数据流模式

- 所有核心类型都是 `@dataclass`
- 支持 `to_dict()` / `from_dict()` 序列化
- `__post_init__` 验证必填字段

### 优雅降级模式

- Hybrid search：一条路径失败时回退到另一条
- Verification tool：LLM 不可用时使用启发式
- Query planning：LLM 不可用时使用启发式

### Async-First 模式

- 所有 MCP tool handlers 都是 `async`
- 重 I/O 操作使用 `asyncio.to_thread()`
- 文档入库在后台运行，带任务追踪

### 组件缓存模式

```python
# 避免为同一 collection 创建多个 ChromaDB 客户端
self._retriever_cache: Dict[str, Dict[str, Any]] = {}
self._cache_lock = threading.Lock()
```

---

## 附录：关键文件路径

| 模块 | 关键文件 |
|------|----------|
| 核心类型 | `nanobot/rag/core/types.py` |
| Agentic 类型 | `nanobot/rag/core/types_agentic.py` |
| 配置 | `nanobot/rag/core/settings.py` |
| Hybrid Search | `nanobot/rag/core/query_engine/hybrid_search.py` |
| RRF Fusion | `nanobot/rag/core/query_engine/fusion.py` |
| Document Chunker | `nanobot/rag/ingestion/chunking/document_chunker.py` |
| Batch Retrieval | `nanobot/rag/mcp_server/tools/agentic/batch_retrieval.py` |
| Verification | `nanobot/rag/mcp_server/tools/agentic/verification.py` |
| Reranking | `nanobot/rag/mcp_server/tools/agentic/reranking.py` |
| Session | `nanobot/rag/core/session/base.py` |

---

## 附录：与主 Agent 的关系

### RAG 与 Agent 的 Provider 问题

当前存在两套独立的 Provider 体系：

| 体系 | 位置 | 用途 |
|------|------|------|
| Agent Provider | `nanobot/providers/` | 主 Agent 的 LLM |
| RAG LLM | `nanobot/rag/libs/llm/` | RAG 模块的 LLM |

**潜在问题**：
- 配置可能需要同步维护两份
- 无法直接复用 Agent 的 provider 配置

**建议改进方向**：
- 统一 Provider 抽象
- RAG 模块复用主 Agent 的 Provider Registry

### RAG 工具与 Agent 工具的关系

RAG MCP Server 的 14 个工具通过 MCP 协议暴露给 Agent。

Agent 可以：
1. 通过 `plan_query` 分析查询
2. 通过 `execute_retrieval_batch` 并发检索
3. 通过 `fuse_and_fetch_round` 获取结果
4. 通过 `verify_results` 验证充分性
5. 通过 `build_citations` 构建引用

---

*本文档整理自 NanoResearch RAG 模块源码分析，涵盖架构、数据流、核心组件和设计决策。*
