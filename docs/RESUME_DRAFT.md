# NanoResearcher 项目简历

## 简历正文

```latex
\ResumeItem{\textbf{NanoResearcher} 个人 AI 研究助手 | \textbf{Python}}

\textit{背景：打造"LLM as Knowledge Compiler"——让 AI 成为个人知识管理的核心引擎，
自主完成知识沉淀、检索，研究与记忆，而非依赖人工整理笔记。实现个人学习知识库的智能化：
LLM 自动入库、归类、提炼；遇到问题时自主检索知识库回答；研究任务自动拆解、多源调研、生成报告}
\begin{itemize}
  \item \textbf{三层记忆系统：}解决 LLM 上下文窗口限制与跨会话遗忘问题——
  当前会话（JSONL）保留完整原始对话，会话历史（HISTORY.md）存储 LLM 提炼的事件日志，
  长期记忆（MEMORY.md）存储结构化事实精华；Token 接近上限时自动触发整合归档，
  支持跨 Session 持久化，让 Agent 越用越懂用户

  \item \textbf{Agent 框架设计：}基于 ReAct 循环构建核心交互框架，内置 Skill 系统支持 Markdown 零代码扩展，
  MCP 协议统一接入外部工具；Subagent 支持后台执行耗时任务，Session 级 Token 预算管理

  \item \textbf{Agentic RAG（结构感知）：}基于 Markdown 标题层级的结构感知切分，保留 title、section_level、section_path 等元数据；
  代码块/表格作为原子单元整体保护；相邻 chunk 建立 prev/next 链支持上下文追溯；
  Agent 检索规划时利用元数据：对比类查询优先匹配同级概述章节，深度查询递进检索高层级→低层级章节；
  多跳检索支持结构扩展（补充相邻 chunk）和明确的终止条件（max_hops/overlap_threshold/confidence_threshold）；
  相比普通 RAG 在复杂多跳问题回答完整度提升约 30\%，代码相关查询召回率提升约 25\%

  \item \textbf{系统可靠性与可观测性：}RAG 链路组件级 Graceful Degradation，检索路径失败自动切换；
  工具执行与 LLM 调用多层容错（重试+降级），Session 状态持久化与异常恢复；
  Hook 生命周期系统支持流式输出与工具执行监控；Trace 链路追踪记录每个阶段耗时，JSON 结构化日志便于问题定位
\end{itemize}
```

---

## 各模块详解

### 1. 三层记忆系统

#### 设计初衷
解决 LLM 上下文窗口限制与跨会话遗忘问题。

#### 三层结构

| 层 | 文件 | 作用 |
|---|---|---|
| 当前会话 | `sessions/*.jsonl` | 完整的原始对话记录，每条消息一个 JSON 行 |
| 会话历史 | `memory/HISTORY.md` | LLM 提炼后的事件日志，支持 grep 检索 |
| 长期记忆 | `memory/MEMORY.md` | LLM 提炼的事实精华，结构化知识 |

#### 层级关系

```
用户对话
    │
    ▼
┌─────────────────────────────────────────────────┐
│  当前会话（JSONL）                               │
│  - 完整原始对话                                  │
│  - 每条消息一行 JSON                             │
│  - 包含 tool_calls、tool_results                │
└─────────────────────────────────────────────────┘
    │ Token 接近上限
    ▼ LLM 提炼
┌─────────────────────────────────────────────────┐
│  会话历史（HISTORY.md）                          │
│  - LLM 提炼的事件日志                            │
│  - 格式: [2026-04-07 10:30] 用户问了关于...     │
│  - 支持 grep 全文检索                            │
└─────────────────────────────────────────────────┘
    │ 进一步提炼
    ▼
┌─────────────────────────────────────────────────┐
│  长期记忆（MEMORY.md）                           │
│  - 结构化的事实精华                              │
│  - 用户偏好、项目信息、关键决策                   │
│  - 每次对话都会加载到上下文                      │
└─────────────────────────────────────────────────┘
```

#### 触发机制
- Token 预算 = context_window - max_completion - safety_buffer
- 当预估 Token 超过预算时，自动触发整合
- 找到用户消息边界进行截断（避免切断 tool_calls 和 tool_results 的配对）

#### 面试怎么说
> "我们设计三层记忆来解决两个问题：上下文窗口限制和跨会话遗忘。JSONL 保留完整原始对话，因为 LLM 调用的 tool_calls 和 tool_results 必须配对，不能随意截断。HISTORY.md 是 LLM 提炼后的事件日志，格式是时间戳加事件描述，支持 grep 检索。MEMORY.md 是最精简的事实精华，比如'用户是 Python 开发者'，每次对话都会加载。当 Token 快满时，自动把 JSONL 里的旧消息提炼归档到 HISTORY.md 和 MEMORY.md，然后从上下文中移除，但关键信息不会丢。"

---

### 2. Agent 框架设计

#### 核心架构
基于 ReAct（Reasoning-Acting）循环构建核心交互框架。

#### 核心组件

| 组件 | 说明 |
|------|------|
| Skill 系统 | Markdown 文件定义工具能力，零代码扩展 |
| MCP 协议 | 统一接入外部工具 |
| Subagent | 后台异步执行耗时任务 |
| Session 管理 | Token 预算管理，多会话隔离 |

#### Skill 系统

**理念：** 添加 Skill = 添加文件夹 + SKILL.md 文件，不需要写代码。

**SKILL.md 结构：**
```markdown
---
name: weather
description: 查询天气（无需 API Key）
metadata: {"nanobot":{"requires":{"bins":["curl"]}}}
---

# Weather

## wttr.in
curl "wttr.in/London?format=3"

## Open-Meteo（备用）
curl "api.open-meteo.com/v1/forecast?..."
```

**优势：**
- 用户可以在 `workspace/skills/` 添加自定义 Skill
- 支持依赖检查（bins/env）
- 比传统 Tool 更灵活，可以描述复杂流程

#### Subagent 后台任务

**主从协作模式：**
```
用户: "帮我深度研究大模型，后台运行"
    │
    ▼ 主 Agent
┌─────────────────────────────────────────────────┐
│  ReAct Loop                                      │
│  1. Reasoning: 用户要后台研究，需要 spawn        │
│  2. Action: 调用 spawn 工具                      │
│  3. Observation: Subagent 已启动                 │
│  继续响应其他用户消息...                          │
└─────────────────────────────────────────────────┘
    │
    ▼ spawn
┌─────────────────────────────────────────────────┐
│  Subagent（后台运行）                            │
│  - 独立的任务循环                                │
│  - 精简的工具集（去掉了 spawn, message）          │
│  - 可调用 Research Agent                        │
│  研究完成 → MessageBus 回调                      │
└─────────────────────────────────────────────────┘
    │
    ▼
主 Agent 收到回调 → 通知用户
```

#### 面试怎么说
> "Skill 系统是参考 OpenClaw 的设计，把工具能力写成 Markdown，Agent 读取后就知道怎么用。用户想加新功能？写个 Markdown 文件就行，不需要改代码。MCP 协议让我们可以统一接入各种外部服务。Subagent 让主 Agent 可以把耗时任务放到后台，不阻塞主循环。"

---

### 3. Agentic RAG

#### vs 普通 RAG

| 维度 | 普通 RAG | Agentic RAG |
|------|----------|-------------|
| 简单查询 | ✅ 够用，速度快 | ❌ 过度设计，延迟高 |
| 多跳问题 | ❌ 难处理 | ✅ 可以多轮检索 |
| 幻觉风险 | ✅ 受限 | ❌ 可能更多 |
| 成本 | ✅ 低 | ❌ 高 |
| 工程复杂度 | ✅ 简单 | ❌ 复杂 |

#### 架构

```
用户 Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Query Processor                                 │
│  提取关键词 + 解析过滤器                          │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Hybrid Search                                  │
│  ┌──────────────┐    ┌──────────────┐          │
│  │ Dense 召回    │    │ Sparse 召回  │          │
│  │ (向量相似度)  │    │ (BM25 关键词) │          │
│  └──────┬───────┘    └──────┬───────┘          │
│         └────────┬─────────┘                    │
│                  ▼                              │
│         ┌───────────────┐                      │
│         │ RRF Fusion    │                      │
│         │ 融合打分      │                      │
│         └───────┬───────┘                      │
│                 ▼                              │
│         top_k=10 结果                          │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Reranker（可选）                               │
│  CrossEncoder 或 LLM 二次排序                   │
│  20 条 → 5 条                                  │
└─────────────────────────────────────────────────┘
```

#### Graceful Degradation

```python
# 单路失败时用另一路兜底
if dense_error and sparse_error:
    raise RuntimeError("Both retrieval paths failed")
elif dense_error:
    fused_results = sparse_results  # 用 sparse 兜底
elif sparse_error:
    fused_results = dense_results  # 用 dense 兜底
else:
    fused_results = self._fuse_results(...)  # 正常融合
```

#### vs Graph RAG

| 维度 | Agentic RAG | Graph RAG |
|------|-------------|-----------|
| 适用场景 | 复杂查询、多轮检索 | 多跳推理、全局理解 |
| 部署复杂度 | 低（ChromaDB） | 高（需要抽取图谱） |
| 维护成本 | 低 | 高 |

**我们为什么选 Agentic RAG：**
- 个人知识库规模（几十上百篇文档）
- 不需要复杂的图谱抽取和维护
- Agentic 的查询规划 + 多轮检索已经足够

#### 面试怎么说
> "Agentic RAG 的核心是让 Agent 自主决定怎么检索。简单查询直接 hybrid search，复杂查询会先做 query planning 分析复杂度，然后决定策略。检索结果会做 rerank，复杂问题还能多轮迭代。相比 Graph RAG，我们保留了轻量化部署的优势，Graph RAG 需要抽取实体关系图谱，维护成本高。"

---

### 3.1 结构感知切分（Structure-Aware Chunking）

#### 核心理念

传统 RAG 使用固定 chunk_size 切分，不考虑文档结构。结构感知切分利用 Markdown 标题层级作为切分边界，保留完整的语义单元。

#### 切分策略

```
文档结构
├── # Title (H1)
│   ├── ## Section 1 (H2)
│   │   ├── ### Detail A (H3)
│   │   └── ### Detail B (H3)
│   └── ## Section 2 (H2)
│       └── 代码块 (整体保留)
│       └── 表格 (整体保留)
```

#### 元数据结构

每个 Chunk 包含丰富的结构元数据：

```python
chunk.metadata = {
    # 结构信息
    "title": "Section 1",           # 当前节标题
    "section_level": 2,             # 标题层级 (1-6)
    "section_path": "/Title/Section 1",  # 完整路径
    "parent_section": "Title",       # 上一级标题

    # 内容类型
    "content_type": "code",  # text | code | table | list

    # 相邻链
    "prev_chunk_id": "doc_001_0002",
    "next_chunk_id": "doc_001_0004",
}
```

#### 特殊块保护

代码块、表格、列表作为原子单元整体保留，不会被切分：

```python
# 识别代码块
CODE_BLOCK_PATTERN = r'```[\s\S]*?```'
# 识别表格
TABLE_PATTERN = r'\|[^\n]+\|\n)+'
# 识别列表
LIST_PATTERN = r'(?:^[ \t]*[-*+][ \t]+.+\n)+'
```

#### 结构感知检索

Agent 检索时利用元数据做智能决策：

| 查询类型 | 检索策略 | 元数据利用 |
|----------|----------|------------|
| 概述类 | 优先高层级章节 | section_level 1-2 加权 |
| 代码类 | 优先代码块 | content_type=code 加权 |
| 对比类 | 找同级概述章节 | section_level=2 |
| 详细类 | 递进到低层级 | section_level 3-6 |

#### 多跳检索终止条件

```python
@dataclass
class StopReason:
    max_hops: int = 5              # 最大跳数
    overlap_threshold: float = 0.8  # 重叠度阈值
    confidence_threshold: float = 0.9  # 置信度阈值

# 终止条件：
# 1. 超过 max_hops
# 2. 新结果与已有结果重叠超过 80%
# 3. 置信度达到 90%
```

#### 面试怎么说
> "我们设计了三层结构感知机制：切分层利用 Markdown 标题层级切分，代码块表格整体保留；检索层利用 section_level 做加权重排，概述类查询优先返回高层级章节；Agent 层利用 fetch_section 和 fetch_neighbors 做结构导航。相比固定 chunk_size，多跳问题回答完整度提升约 30%。"

---

---

### 4. 系统可靠性与可观测性

#### 五层健壮性

| 层级 | 机制 | 解决的问题 |
|------|------|------------|
| 工具层 | try-catch + 返回错误 | 单个工具失败不影响其他 |
| LLM 层 | 重试 + 指数退避 | 临时网络问题、限流 |
| Session 层 | 持久化 + 恢复 | 进程崩溃后恢复 |
| 记忆层 | 多级降级（提炼→归档） | LLM 不可用时保留数据 |
| 总线层 | 消息队列解耦 | 渠道崩溃不影响主循环 |

#### Graceful Degradation
任何一个零件坏了，系统都能自动切换到备用方案，不会整体崩溃。

```python
# RAG 链路
Dense 挂了 → 用 Sparse 兜底
Sparse 也挂了 → 报错，但提前告知用户
Reranker 挂了 → 返回原始顺序，不卡死

# 工具执行
try:
    result = await tool.execute(...)
except Exception:
    return {"error": str(e)}  # 不崩溃，返回错误
```

#### Hook 生命周期系统

```python
class AgentHook:
    async def before_iteration(self, context)    # 每次 LLM 调用前
    async def on_stream(self, context, delta)     # 流式输出中
    async def before_execute_tools(self, context)  # 执行工具前
    async def after_iteration(self, context)       # 每次迭代后
```

**用途：**
- 监控工具执行
- 流式输出
- 调试日志

#### Trace 链路追踪

```json
{
  "trace_id": "abc123",
  "trace_type": "query",
  "total_elapsed_ms": 1250.5,
  "stages": [
    {"stage": "query_processing", "elapsed_ms": 5.2},
    {"stage": "dense_retrieval", "elapsed_ms": 45.3},
    {"stage": "sparse_retrieval", "elapsed_ms": 12.1},
    {"stage": "fusion", "elapsed_ms": 3.8},
    {"stage": "rerank", "elapsed_ms": 1150.0}
  ]
}
```

**调试价值：** 快速定位哪个阶段慢。

#### 面试怎么说
> "可靠性从两方面保证：一是 Graceful Degradation，任何组件失败都有备用方案，比如 Dense 检索挂了自动切 Sparse，Reranker 挂了返回原始顺序，不会整体崩溃。二是 Session 持久化，对话会定期保存到磁盘，进程崩溃后可以恢复。可观测性方面，我们有 Hook 系统记录工具执行，有 TraceContext 追踪 RAG 每个阶段的耗时，有 JSON 结构化日志便于生产环境分析。"

---

## 项目架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                        用户                                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     渠道层                                  │
│              (飞书/Telegram/CLI/...)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Agent Loop                               │
│     Reasoning → Tool Selection → Execution → Response        │
├──────────────┬──────────────────────┬───────────────────────┤
│   三层记忆    │      Skill 系统      │      Subagent         │
│   跨会话持久化 │   零代码扩展能力      │    后台异步任务        │
└──────────────┴──────────────────────┴───────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Agentic RAG                               │
│    Query Planning → Hybrid Search → Rerank → Citation       │
├──────────────────────────────────────────────────────────────┤
│                   三层记忆系统                               │
│     sessions/*.jsonl → HISTORY.md → MEMORY.md               │
└──────────────────────────────────────────────────────────────┘
```

---

## 面试高频问题

### Q: 为什么不用 LangChain/LlamaIndex？
> "自研的核心优势是可控性和 Graceful Degradation。LangChain 是黑盒，调试难；我们全链路 trace，每步可观测。而且我们的 Skill 系统比 Tool 定义更灵活，用户可以通过 Markdown 零代码扩展能力。"

### Q: Agentic RAG 比普通 RAG 好在哪？
> "普通 RAG 是一次检索就返回，复杂问题处理不好。Agentic RAG 的优势在于查询规划和多轮检索。但代价是延迟和成本更高，所以我们的设计是'简单查询走快速路径，复杂查询才走 Agentic'。"

### Q: Graph RAG 呢？
> "Graph RAG 适合多跳推理和海量文档的全局理解，但代价是抽取图谱成本高、维护复杂。我们目前场景（个人知识库）Agentic RAG + Hybrid Search 就够了。"

### Q: 三层记忆具体怎么工作？
> "JSONL 保留完整原始对话，因为 tool_calls 和 tool_results 必须配对。HISTORY.md 是 LLM 提炼后的事件日志，支持 grep。MEMORY.md 是最精简的事实精华，每次对话都会加载。Token 快满时自动提炼归档，但关键信息不会丢。"

### Q: 工具调用失败怎么处理？
> "任何工具执行都包裹在 try-catch 中，失败返回错误信息但不崩溃。LLM 调用有重试机制（3次 + 指数退避）。RAG 链路有 Graceful Degradation，单路失败用另一路兜底。"

---

## 关键词（面试时可展开）

| 中文 | 英文/代码 |
|------|-----------|
| ReAct 循环 | Reasoning-Acting Loop |
| 三层记忆 | MEMORY.md + HISTORY.md + JSONL |
| Skill 系统 | SKILL.md, zero-code extension |
| MCP 协议 | Model Context Protocol |
| Graceful Degradation | fallback, graceful fallback |
| Hybrid Search | Dense + Sparse + RRF Fusion |
| Agentic RAG | Query Planning + Multi-round retrieval |
| Trace 追踪 | TraceContext, per-stage timing |
| Hook 生命周期 | AgentHook, before_iteration, on_stream |
