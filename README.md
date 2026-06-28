<div align="center">
  <!-- TODO: 替换为项目 logo -->
  <img src="nanoresearch_logo.png" alt="NanoResearch" width="500">
  <h1>NanoResearch: 个人 AI 研究助手</h1>
  <p>
    <a href="https://pypi.org/project/nanoresearch-ai/"><img src="https://img.shields.io/pypi/v/nanoresearch-ai" alt="PyPI"></a>
    <a href="https://pepy.tech/project/nanoresearch-ai"><img src="https://static.pepy.tech/badge/nanoresearch-ai" alt="Downloads"></a>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</div>

🔬 **NanoResearch** 是一个轻量级 AI 知识库与研究助手，基于自研 **ReAct Agent** 框架构建，实现任务分解、工具调用与跨会话记忆能力，支持复杂课题 **Deep Research** 与个人知识持续沉淀。

## 📢 News

> [!IMPORTANT]
> **2026-05** — 完成 Deep Research 与 Agentic RAG 核心功能

- **2026-05-10** 🧠 优化 RAG Skill 决策流程，Agent 先查知识库再决定是否深度研究
- **2026-05-08** 🔬 Deep Research 支持 Claims/Insights 知识沉淀，跨会话复用研究结论
- **2026-05-05** 📚 Agentic RAG 多路召回 + Cross-Encoder 精排，Top-10 召回率提升 ~25%
- **2026-05-01** 💾 Token 感知记忆压缩，整合后上下文降至触发阈值 50%
- **2026-04-28** ⚡ Subagent 异步执行耗时研究任务，MessageBus 回注主会话
- **2026-04-25" 🏗️ 完成研究流程编排：Planner → Searcher → Synthesizer → Refiner → Reporter

<details>
<summary>Earlier news</summary>

- **2026-04-20** 🚀 实现覆盖度阈值驱动的迭代收敛，平均 2.3 轮收敛
- **2026-04-15** 📊 Cross-Encoder 精排 + 去重，去重率约 38%
- **2026-04-10** 📝 带引用溯源的报告生成器
- **2026-04-05** 🔍 并行搜索 + RRF 融合的多路召回
- **2026-04-01" 🎯 项目启动，基于 NanoResearch 框架扩展研究能力

</details>

## ✨ 核心特性

🔬 **Deep Research** — 复杂课题的自动化深度研究
- 主题分解为 3-6 个子问题，并行搜索
- 多轮迭代 + 覆盖度阈值驱动收敛
- Cross-Encoder 精排 + 去重
- 结构化提取核心发现、矛盾观点与知识空白
- 带引用溯源的报告生成

🧠 **Agentic RAG** — Agent 驱动的检索增强
- MCP 协议发布 RAG 检索服务
- 多路召回（Dense 向量 + Sparse BM25）+ RRF 融合
- Cross-Encoder 精排重排
- Agent 自主决策检索策略，多轮迭代优化查询

💾 **混合记忆系统** — 长短期记忆融合
- Token 感知自动压缩，整合后降至触发阈值 50%
- 长期记忆通过 MCP 复用 RAG 检索服务
- 每轮自动提取事实入库
- 跨会话语义召回替代全量注入

⚡ **Subagent 异步执行** — 耗时任务后台处理
- 主流程即时响应，后台任务完成自动通知
- 复用 RAG 检索服务作为长期记忆

🪶 **轻量级 Agent 框架**
- 基于 ReAct 循环自研，代码简洁易扩展
- Skill 渐进式披露，按需加载工具
- 支持 20+ LLM Provider

## 🏗️ 系统架构

<!-- TODO: 绘制系统架构图，建议包含以下模块的关系图 -->
<p align="center">
  <img src="nanoresearch_arch.png" alt="NanoResearch architecture" width="800">
</p>

### 架构分层

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Channels (多平台接入)                           │
│   Telegram | Discord | Feishu | WhatsApp | WeChat | QQ | Slack | ...   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Message Bus                                  │
│                   消息路由与事件分发 (Pub/Sub 模式)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Agent Loop                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   Context    │◄───►│     LLM      │◄───►│    Tools     │            │
│  │   Builder    │     │   Provider   │     │   Registry   │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│         │                                           │                   │
│         ▼                                           ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Tool Layers                               │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │   │
│  │  │    RAG    │ │  Research │ │   Spawn   │ │   Memory  │       │   │
│  │  │   (MCP)   │ │   Runner  │ │ (Subagent)│ │   Store   │       │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │ RAG Server  │ │   Research  │ │  Knowledge  │
            │    (MCP)    │ │    Runner   │ │    Loop     │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                            ┌─────────────┐
                            │   ChromaDB  │
                            │(Vector Store)│
                            └─────────────┘
```

### 核心模块

| 模块 | 路径 | 功能 |
|------|------|------|
| **Agent Loop** | `agent/loop.py` | ReAct 执行循环，LLM ↔ 工具调用 |
| **Context Builder** | `agent/context.py` | 上下文构建，Skill 加载，记忆注入 |
| **Memory** | `agent/memory.py` | Token 感知压缩，长期记忆管理 |
| **Subagent** | `agent/subagent.py` | 后台异步任务执行，MessageBus 通知 |
| **RAG Server** | `rag/mcp_server/` | MCP 协议检索服务 |
| **Hybrid Search** | `rag/core/query_engine/hybrid_search.py` | 多路召回 + RRF 融合 |
| **Reranker** | `rag/libs/reranker/` | Cross-Encoder 精排 |
| **Research Runner** | `research/runner.py` | 研究流程编排 |
| **Knowledge Loop** | `research/knowledge_*.py` | Claims/Insights 沉淀与检索 |

## Table of Contents

- [News](#-news)
- [核心特性](#-核心特性)
- [系统架构](#️-系统架构)
- [Install](#-install)
- [Quick Start](#-quick-start)
- [Deep Research](#-deep-research)
- [Agentic RAG](#-agentic-rag)
- [混合记忆系统](#-混合记忆系统)
- [Chat Apps](#-chat-apps)
- [Configuration](#️-configuration)
- [Multiple Instances](#-multiple-instances)
- [CLI Reference](#-cli-reference)
- [Docker](#-docker)
- [Linux Service](#-linux-service)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Contribute](#-contribute)

## 📦 Install

**Install from source** (latest features, recommended for development)

```bash
git clone https://github.com/Refrain57/nanobot.git
cd nanobot
pip install -e .
```

**Install with [uv](https://github.com/astral-sh/uv)** (stable, fast)

```bash
uv tool install nanoresearch-ai
```

**Install from PyPI** (stable)

```bash
pip install nanoresearch-ai
```

### Update to latest version

```bash
pip install -U nanoresearch-ai
nr --version
```

## 🚀 Quick Start

> [!TIP]
> Set your API key in `~/.nanoresearch/config.json`.
> Get API keys: [OpenRouter](https://openrouter.ai/keys) (Global)

**1. Initialize**

```bash
nr onboard
```

Use `nr onboard --wizard` for interactive setup.

**2. Configure** (`~/.nanoresearch/config.json`)

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-sonnet-4-6",
      "provider": "openrouter"
    }
  }
}
```

**3. Chat**

```bash
nr agent
```

That's it! You have a working AI assistant in 2 minutes.

## 🔬 Deep Research

Deep Research 是 NanoResearch 的核心能力，用于复杂课题的自动化深度研究。

### 研究流程

<!-- TODO: 绘制 Deep Research 流程图 -->
```
用户输入 Topic
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 0: 知识预查询                                            │
│  - 从 research_claims/insights 检索历史知识                      │
│  - 从用户上传文档检索相关上下文                                   │
└─────────────────────────────────────────────────────────────────┘
    │ combined_context
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Planner (研究规划)                                    │
│  - LLM 将 topic 拆解为 3-6 个子问题                              │
│  - 每个子问题生成中英文关键词                                     │
└─────────────────────────────────────────────────────────────────┘
    │ plan
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Searcher (搜索+抓取)  [可迭代 max_iterations 次]      │
│  - 并行搜索子问题的关键词                                        │
│  - WebFetch 抓取 URL 内容                                        │
│  - 评分、去重、Cross-Encoder Rerank                              │
└─────────────────────────────────────────────────────────────────┘
    │ search_results
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Synthesizer (信息综合)                                │
│  - LLM 分析搜索结果                                              │
│  - 提取高层发现、来源映射、矛盾点、知识空白                        │
│  - 计算覆盖度评分 (coverage_score)                               │
└─────────────────────────────────────────────────────────────────┘
    │ synthesis
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Refiner (迭代判断)                                    │
│  - coverage < threshold 且 iteration < max → 继续               │
│  - 决定新增子问题/补充关键词                                      │
└─────────────────────────────────────────────────────────────────┘
    │ 循环回到 Phase 2 (如需继续)
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: Reporter (报告生成)                                   │
│  - 按子问题分节撰写                                              │
│  - 整合为完整 Markdown 报告                                      │
│  - 自评质量 (如低于阈值则重试一次)                                │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 6: Knowledge Processor (知识入库)                        │
│  - 报告写入 RAG → chunk_ids                                      │
│  - 提取 Claims (原子事实) → claim_store                          │
│  - 提取 Insights (跨域规律) → insight_store                      │
└─────────────────────────────────────────────────────────────────┘
```

### 使用示例

**基础研究**

```
用户：帮我研究一下 3D Gaussian Splatting 和 NeRF 的对比

Agent：
1. 先查知识库（research_claims）→ 无相关结论
2. 告知用户："正在深度研究「3DGS vs NeRF」，预计 10-30 分钟..."
3. spawn 后台执行 research
4. 主 Agent 返回

→ 后台 subagent 执行完整研究流程
→ 完成后自动通知用户
```

**命令行调用**

```python
# 启动研究
research(action="start", topic="3DGS vs NeRF 对比", depth="normal")

# 查看进度
research(action="status", research_id="abc123")

# 列出历史
research(action="list")
```

### 研究深度

| depth | 迭代轮次 | 每轮来源数 | 适用场景 |
|-------|---------|-----------|---------|
| `quick` | 1 轮 | 5 篇 | 快速了解，时间敏感 |
| `normal` | 3 轮 | 10 篇 | 常规研究（默认） |
| `deep` | 5 轮 | 20 篇 | 深度研究，学术报告 |

### 知识沉淀

研究完成后，系统自动提取并存储：

| 存储类型 | Collection | 说明 |
|----------|------------|------|
| **Claims** | `research_claims` | 原子级事实陈述，如"3DGS 训练速度比 NeRF 快 10 倍" |
| **Insights** | `research_insights` | 跨域可复用规律，如"显式表示方法在效率上优于隐式表示" |
| **报告 Chunks** | `default` | 原始报告片段，用于证据追溯 |

## 🧠 Agentic RAG

NanoResearch 的 RAG 系统通过 MCP 协议发布，支持 Agent 自主决策检索策略。

### MCP 协议架构

LLM 看到的是一堆本地工具，每个工具背后可能是子进程、也可能是远端 HTTP 服务。
中间通过 `MCPToolWrapper`（翻译转运层）和 `ProtocolHandler`（服务端接线员）对接。

```
                        Agent 主进程
   ┌───────────────────────────────────────────────────────────┐
   │                                                           │
   │   LLM 调用工具                                             │
   │        │                                                  │
   │        ▼                                                  │
   │   ToolRegistry（工具注册表，就是个 dict）                    │
   │   ┌─────────────────────────────┐                        │
   │   │ 本地工具    │ MCP 工具        │                       │
   │   │ file_read   │ mcp_rag_kb_search                     │
   │   │ exec        │ mcp_rag_list_collections              │
   │   │ web_search  │ mcp_rag_ingest_document               │
   │   └─────────────┴──────────────────┘                    │
   │                    │                                      │
   │                    ▼                                      │
   │   MCPToolWrapper  ← 翻译转运层                            │
   │   ┌────────────────────────────────────┐                 │
   │   │ 1. 参数翻译（AI 的说法→底层说法）    │                 │
   │   │ 2. Schema 适配（不同 AI 格式兼容）   │                 │
   │   │ 3. 转发：session.call_tool()       │                 │
   │   │ 4. 兜底：超时/报错不崩，返回文字给 AI │                 │
   │   └────────────────┬───────────────────┘                 │
   │                    │                                      │
   └────────────────────┼──────────────────────────────────────┘
                        │
               JSON-RPC 2.0（一条条 JSON 字符串，带 id 对账）
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐   ┌──────────┐   ┌───────────┐
   │  stdio  │   │   SSE    │   │   HTTP    │
   │子进程管道│   │长连接+POST│   │ 请求-响应  │
   └────┬────┘   └────┬─────┘   └─────┬─────┘
        │             │               │
        └─────────────┼───────────────┘
                      │
                      ▼
   ┌───────────────────────────────────────────────────────────┐
   │                   MCP Server 进程                          │
   │                                                           │
   │   ProtocolHandler ← 服务端接线员                           │
   │   ┌──────────────────────────────────────┐               │
   │   │ tools = {                            │               │
   │   │   "kb_search"     → search_handler   │  收到请求→    │
   │   │   "kb_retrieve"   → retrieve_handler │  字典查名字→  │
   │   │   "ingest_doc"    → ingest_handler   │  调对应函数    │
   │   │   ...                                │               │
   │   │ }                                    │               │
   │   └────────────────┬─────────────────────┘               │
   │                    │                                      │
   │                    ▼                                      │
   │   ┌──────────────────────────────────────┐               │
   │   │        RAG 检索引擎                    │               │
   │   │  ┌──────────┐ ┌────────┐ ┌────────┐  │               │
   │   │  │  Dense   │ │ Sparse │ │  RRF   │  │               │
   │   │  │ (向量)   │ │ (BM25) │ │ 融合   │  │               │
   │   │  └──────────┘ └────────┘ └────────┘  │               │
   │   │              ↓                        │               │
   │   │  ┌──────────────────────────┐        │               │
   │   │  │ Cross-Encoder Reranker   │        │               │
   │   │  └──────────────────────────┘        │               │
   │   └──────────────────────────────────────┘               │
   │                    │                                      │
   └────────────────────┼──────────────────────────────────────┘
                        │
                        ▼
                ┌─────────────┐
                │  ChromaDB   │
                │ (向量数据库) │
                └─────────────┘
```

**核心组件说明：**

| 组件 | 在哪 | 一句话 |
|------|------|--------|
| **ToolRegistry** | Agent 主进程 | 工具注册表，就是个 dict，不管工具从哪来 |
| **MCPToolWrapper** | Agent 主进程 | 翻译转运层——对 AI 暴露统一接口，对后端转发 JSON-RPC 调用 |
| **ProtocolHandler** | MCP Server 进程 | 服务端接线员——收到请求按名字查字典，调对应函数，结果包好返回 |
| **ClientSession** | Agent 主进程 | MCP SDK 封装，负责 JSON-RPC 的编解码和请求-响应匹配 |
| **Server** | MCP Server 进程 | MCP SDK 封装，负责监听 stdio/HTTP，路由到 ProtocolHandler |

**JSON-RPC 请求示例（底层就是这样的字符串）：**
```json
→ {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"kb_search","arguments":{"query":"什么是RAG"}}}
← {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"检索结果..."}]}}
```

### Collections

| Collection | 内容 | 使用场景 |
|------------|------|----------|
| `default` | 用户上传的文档 | 查询用户文档中的信息 |
| `research_claims` | 研究结论 (Claims) | 查询已有的研究事实 |
| `research_insights` | 研究洞察 (Insights) | 查询跨域规律 |
| `user_memory` | 对话产生的记忆 | 跨会话用户偏好 |

### RAG Skill 决策流程

```
知识类问题
    │
    ▼
先查知识库（~1s）
    │
    ├─► 有答案（claims >= 3 且 confidence >= 0.7）→ 直接回答
    │
    └─► 无答案 → 
            message("正在研究...")
            spawn(task="用 research 工具研究...")
            "已启动后台研究，完成后通知您。"
            主 Agent 返回
                │
                └─► subagent 执行 research（10-30min）
                    完成后自动通知用户
```

### 查询规划与改写

检索前先对查询做两件事情：指代消解和策略规划。

**指代消解（Query Rewriting）：**

多轮对话中用户经常会说"它的缺点是什么""和刚才那个对比一下"——这些指代词单拿出来没法检索。系统会获取最近 5 轮对话历史，调 LLM 把查询改写为独立完整的检索句。如果查询本身已经完整，就原样返回。

**策略规划（Query Planning）：**

改写后的查询送入 Planner，LLM 判断复杂度并拆解子查询，同时为每个子查询标注检索策略：

| 查询类型 | 推荐策略 | 例子 |
|---------|---------|------|
| 专有名词、方法名、指标 | sparse | "PGSR", "PSNR" |
| 概念描述、通用术语 | dense | "核心思想", "渲染质量" |
| 复杂对比、多问题 | hybrid | "A 和 B 对比" |

### Internal Loop（多轮检索）

复杂查询（对比类、多子问题、指代消解等）会进入内部循环，用 4 阶段状态机迭代检索，直到验证通过或达到最大轮次。

```
用户查询
    │
    ▼
┌──────────────────────────────────────────┐
│  Phase 1: Plan（查询规划）                │
│  - LLM 将查询拆解为多个子查询              │
│  - 每个子查询标注检索策略（dense/sparse/hybrid）│
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  Phase 2: Search（批量检索）              │
│  - 并发执行所有子查询                      │
│  - 对比类查询额外拉取关联段落              │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  Phase 3: Fuse + Verify（融合验证）       │
│  - RRF 融合多路结果                       │
│  - LLM 验证：confidence >= 0.7？         │
│    ├─ 通过 → Phase 4                     │
│    └─ 不通过 → 扩展邻居 chunk             │
│               生成 next_actions          │
│               回到 Phase 2（最多 5 轮）    │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  Phase 4: Finalize（构建引用）            │
│  - 整理最终结果                           │
│  - 构建带溯源引用的回复                    │
└──────────────────────────────────────────┘
```

**触发多轮的条件：**
- 查询复杂度判定为 complex（规则判断：指代词、对比关键词、多问号等）
- 单轮检索后 verify 认为 confidence 不足或存在缺失方面

**防偏控制：**
- 最大 5 轮硬上限，不会无限循环
- 每轮保留原始查询上下文，避免越搜越偏
- 验证不通过时先扩展邻居 chunk 补充上下文，再决定下一步方向

### 查询归一化与 Redis 缓存

查询改写只解决了指代消解（把"它的缺点"补成"3DGS 的缺点"），但同一个意思可以有很多种问法——"3DGS 的缺点""3DGS 有什么局限性""3DGS 不足"——改写过后的文本不同，如果直接用文本做缓存 key，语义等价的查询永远对不上。

所以在改写之后加了一层**查询归一化**：把语义等价的查询映射到同一个规范形式上。规范形式同时用作 Redis 缓存的 key。

```
查询进来
    │
    ▼
查询改写（指代消解）
    │
    ▼
查询归一化（映射到规范形式）
    │
    ▼
Redis 查缓存 ── 命中 → 直接返回
    │
    └── 未命中 → 走完整检索链路 → 结果写回 Redis
```

- 常见问题提前写入 Redis，TTL 设得较长
- 未命中的查询检索完成后自动回种缓存
- Redis 不可用时降级为直接调 embedding API

### 使用示例

```python
# 查询用户文档
mcp_rag_rag_search(query="RAG 的实现原理", collection="default")

# 查询已有的研究结论
mcp_rag_rag_search(query="3DGS 的局限性", collection="research_claims")

# 查询跨域洞察
mcp_rag_rag_search(query="性能优化的一般方法", collection="research_insights")
```

### 配置

```json
{
  "rag": {
    "retrieval": {
      "dense_top_k": 20,
      "sparse_top_k": 20,
      "fusion_top_k": 10,
      "rrf_k": 60
    },
    "rerank": {
      "enabled": true,
      "provider": "cross_encoder",
      "model": "BAAI/bge-reranker-v2-m3",
      "top_k": 10
    }
  }
}
```

## 💾 混合记忆系统

NanoResearch 采用分层记忆架构，结合短期上下文与长期知识库。

### 架构

<!-- TODO: 绘制记忆系统架构图 -->
```
┌─────────────────────────────────────────────────────────────────┐
│                        Conversation                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Short-term Memory (Session)                               │ │
│  │  - 最近 N 轮对话                                            │ │
│  │  - Token 预算限制                                           │ │
│  │  - 超出阈值 → 自动压缩整合                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼ 触发压缩                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Memory Consolidator                                       │ │
│  │  - LLM 提取稳定事实 → MEMORY.md                             │ │
│  │  - 对话知识 → user_memory collection (RAG)                  │ │
│  │  - 整合后上下文降至触发阈值 50%                              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Long-term Memory                            │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │   MEMORY.md      │  │  user_memory     │  │ research_*    │ │
│  │  (稳定事实)       │  │  (对话知识)       │  │ (研究沉淀)     │ │
│  │                  │  │                  │  │               │ │
│  │  - 用户偏好      │  │  - 语义可检索    │  │  - claims     │ │
│  │  - 项目上下文    │  │  - 跨会话召回    │  │  - insights   │ │
│  │  - 长期决策      │  │                  │  │               │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│                            │                                     │
│                            ▼ 每轮自动检索                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Context Builder                                           │ │
│  │  - 从 RAG 检索相关记忆                                      │ │
│  │  - 语义召回替代全量注入                                      │ │
│  │  - Token 预算控制                                           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 特性

| 特性 | 说明 |
|------|------|
| **Token 感知压缩** | 上下文超出预算时自动整合，降至触发阈值 50% |
| **分层存储** | 稳定事实 → MEMORY.md，对话知识 → RAG collection |
| **语义召回** | 每轮根据当前话题从 RAG 检索相关记忆，替代全量注入 |
| **跨会话复用** | research claims/insights 自动沉淀，后续研究可复用 |

### 配置

```json
{
  "memory": {
    "consolidation_threshold": 0.8,
    "max_context_tokens": 65536
  }
}
```

## 💬 Chat Apps

Connect NanoResearch to your favorite chat platform.

| Channel | What you need |
|---------|---------------|
| **Telegram** | Bot token from @BotFather |
| **Discord** | Bot token + Message Content intent |
| **WhatsApp** | QR code scan (`nr channels login whatsapp`) |
| **WeChat (Weixin)** | QR code scan (`nr channels login weixin`) |
| **Feishu** | App ID + App Secret |
| **DingTalk** | App Key + App Secret |
| **Slack** | Bot token + App-Level token |
| **Matrix** | Homeserver URL + Access token |
| **Email** | IMAP/SMTP credentials |
| **QQ** | App ID + App Secret |
| **Wecom** | Bot ID + Bot Secret |

<details>
<summary><b>Telegram</b> (Recommended)</summary>

**1. Create a bot**
- Open Telegram, search `@BotFather`
- Send `/newbot`, follow prompts
- Copy the token

**2. Configure**

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

**3. Run**

```bash
nr gateway
```

</details>

<details>
<summary><b>Discord</b></summary>

**1. Create a bot**
- Go to https://discord.com/developers/applications
- Create an application → Bot → Add Bot
- Copy the bot token

**2. Enable intents**
- In the Bot settings, enable **MESSAGE CONTENT INTENT**

**3. Configure**

```json
{
  "channels": {
    "discord": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"],
      "groupPolicy": "mention"
    }
  }
}
```

**4. Run**

```bash
nr gateway
```

</details>

<details>
<summary><b>Feishu</b></summary>

Uses **WebSocket** long connection — no public IP required.

**1. Create a Feishu bot**
- Visit [Feishu Open Platform](https://open.feishu.cn/app)
- Create a new app → Enable **Bot** capability
- Get **App ID** and **App Secret**

**2. Configure**

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "cli_xxx",
      "appSecret": "xxx",
      "allowFrom": ["ou_YOUR_OPEN_ID"],
      "groupPolicy": "mention"
    }
  }
}
```

**3. Run**

```bash
nr gateway
```

</details>

<details>
<summary><b>WhatsApp</b></summary>

Requires **Node.js ≥18**.

**1. Link device**

```bash
nr channels login whatsapp
# Scan QR with WhatsApp → Settings → Linked Devices
```

**2. Configure**

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

**3. Run**

```bash
nr gateway
```

</details>

<details>
<summary><b>Slack</b></summary>

Uses **Socket Mode** — no public URL required.

**1. Create a Slack app**
- Go to [Slack API](https://api.slack.com/apps) → **Create New App**
- **Socket Mode**: Toggle ON → Generate App-Level Token
- **OAuth & Permissions**: Add bot scopes: `chat:write`, `app_mentions:read`
- **Install App** → copy Bot Token

**2. Configure**

```json
{
  "channels": {
    "slack": {
      "enabled": true,
      "botToken": "xoxb-...",
      "appToken": "xapp-...",
      "allowFrom": ["YOUR_SLACK_USER_ID"]
    }
  }
}
```

**3. Run**

```bash
nr gateway
```

</details>

<details>
<summary><b>WeChat (微信)</b></summary>

```bash
pip install "nanoresearch-ai[weixin]"
nr channels login weixin
```

```json
{
  "channels": {
    "weixin": {
      "enabled": true,
      "allowFrom": ["YOUR_WECHAT_USER_ID"]
    }
  }
}
```

</details>

## ⚙️ Configuration

Config file: `~/.nanoresearch/config.json`

### Custom base directory (multi-tenant / containerized deployments)

By default, NanoResearch stores all runtime state under `~/.nanoresearch`.
Set the `NANORESEARCH_HOME` environment variable to relocate this base
directory — for example to support multiple tenants on a single host or
to mount a non-home volume inside a container:

```bash
export NANORESEARCH_HOME=/data/tenant_alice
nr serve
```

Tilde expansion (`~/custom-root`) and absolute paths are both supported.
The directory is created automatically on first write. The legacy
`NANOBOT_HOME` environment variable is also accepted for backward
compatibility and will be removed in v0.3.0.

### Providers

| Provider | Purpose | Get API Key |
|----------|---------|-------------|
| `openrouter` | LLM (recommended) | [openrouter.ai](https://openrouter.ai) |
| `anthropic` | LLM (Claude direct) | [console.anthropic.com](https://console.anthropic.com) |
| `openai` | LLM (GPT direct) | [platform.openai.com](https://platform.openai.com) |
| `deepseek` | LLM (DeepSeek) | [platform.deepseek.com](https://platform.deepseek.com) |
| `dashscope` | LLM (Qwen) | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) |
| `ollama` | LLM (local) | — |

<details>
<summary><b>Ollama (local)</b></summary>

**1. Start Ollama:**
```bash
ollama run llama3.2
```

**2. Configure:**
```json
{
  "providers": {
    "ollama": {
      "apiBase": "http://localhost:11434"
    }
  },
  "agents": {
    "defaults": {
      "provider": "ollama",
      "model": "llama3.2"
    }
  }
}
```

</details>

### Web Search

```json
{
  "tools": {
    "web": {
      "search": {
        "provider": "brave",
        "apiKey": "YOUR_API_KEY"
      }
    }
  }
}
```

| Provider | Config fields | Env var fallback | Free |
|----------|--------------|------------------|------|
| `brave` (default) | `apiKey` | `BRAVE_API_KEY` | No |
| `tavily` | `apiKey` | `TAVILY_API_KEY` | No |
| `jina` | `apiKey` | `JINA_API_KEY` | Free tier |
| `duckduckgo` | — | — | Yes |

### MCP (Model Context Protocol)

```json
{
  "tools": {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
      }
    }
  }
}
```

### Research 配置

```json
{
  "research": {
    "enabled": true,
    "max_iterations": 3,
    "max_sources_per_question": 10,
    "min_coverage_threshold": 0.7,
    "rerank_enabled": true,
    "rerank_provider": "cross_encoder",
    "rerank_model": "BAAI/bge-reranker-v2-m3"
  }
}
```

### Security

| Option | Default | Description |
|--------|---------|-------------|
| `tools.restrictToWorkspace` | `false` | Restrict all tools to workspace directory |
| `tools.exec.enable` | `true` | Enable shell execution |
| `channels.*.allowFrom` | `[]` | Whitelist of user IDs |

## 🧩 Multiple Instances

Run multiple instances with separate configs:

```bash
# Initialize
nr onboard --config ~/.nanoresearch-telegram/config.json --workspace ~/.nanoresearch-telegram/workspace
nr onboard --config ~/.nanoresearch-discord/config.json --workspace ~/.nanoresearch-discord/workspace

# Run
nr gateway --config ~/.nanoresearch-telegram/config.json
nr gateway --config ~/.nanoresearch-discord/config.json
```

## 💻 CLI Reference

| Command | Description |
|---------|-------------|
| `nr onboard` | Initialize config & workspace |
| `nr agent -m "..."` | Chat with the agent |
| `nr agent` | Interactive chat mode |
| `nr gateway` | Start the gateway |
| `nr status` | Show status |
| `nr channels login <channel>` | Authenticate a channel |

<details>
<summary><b>Heartbeat (Periodic Tasks)</b></summary>

Edit `~/.nanoresearch/workspace/HEARTBEAT.md`:

```markdown
## Periodic Tasks

- [ ] Check weather forecast
- [ ] Scan inbox for urgent emails
```

The agent executes tasks every 30 minutes and delivers results to your active channel.

</details>

## 🐳 Docker

```bash
# Build
docker build -t nanoresearch .

# Initialize
docker run -v ~/.nanoresearch:/root/.nanoresearch --rm nanoresearch onboard

# Run gateway
docker run -v ~/.nanoresearch:/root/.nanoresearch -p 18790:18790 nanoresearch gateway

# Or use docker-compose
docker compose up -d nanoresearch-gateway
```

**Custom base directory in docker-compose:**

Set `NANORESEARCH_HOME` and bind-mount it instead of `~/.nanoresearch`. See
the commented `Multi-tenant / custom-base example` block under each service in
`docker-compose.yml`.

## 🐧 Linux Service

```bash
# Create service file
cat > ~/.config/systemd/user/nanoresearch-gateway.service << 'EOF'
[Unit]
Description=NanoResearch Gateway
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/nr gateway
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user enable --now nanoresearch-gateway
```

## 📁 Project Structure

```
nanobot/
├── agent/                    # 🧠 Core Agent
│   ├── loop.py               #    ReAct 执行循环
│   ├── context.py            #    上下文构建
│   ├── memory.py             #    Token 压缩 + 长期记忆
│   ├── subagent.py           #    后台任务执行
│   ├── skills.py             #    Skill 渐进式加载
│   └── tools/                #    内置工具
│       ├── research.py       #    Deep Research 工具
│       ├── spawn.py          #    Subagent 启动工具
│       └── ...
│
├── research/                 # 🔬 Deep Research
│   ├── runner.py             #    研究流程编排
│   ├── planner.py            #    主题分解
│   ├── searcher.py           #    并行搜索 + 评分
│   ├── synthesizer.py        #    信息综合
│   ├── refiner.py            #    迭代判断
│   ├── reporter.py           #    报告生成
│   ├── knowledge_processor.py #   知识沉淀
│   └── knowledge_search.py   #    知识检索
│
├── rag/                      # 📚 RAG 系统
│   ├── core/
│   │   ├── query_engine/     #    检索引擎
│   │   │   ├── hybrid_search.py   # 多路召回
│   │   │   ├── dense_retriever.py # 向量检索
│   │   │   ├── sparse_retriever.py# BM25 检索
│   │   │   └── fusion.py     #    RRF 融合
│   │   └── response/         #    响应构建
│   ├── libs/
│   │   ├── reranker/         #    精排模块
│   │   ├── embedding/        #    向量编码
│   │   └── vector_store/     #    向量存储
│   └── mcp_server/           #    MCP 协议服务
│       └── tools/agentic/    #    Agentic RAG 工具
│
├── skills/                   # 🎯 Skills (deep-research, rag, memory...)
├── channels/                 # 📱 聊天平台接入
├── bus/                      # 🚌 消息总线
├── providers/                # 🤖 LLM 提供商
├── session/                  # 💬 会话管理
├── config/                   # ⚙️ 配置
└── cli/                      # 🖥️ 命令行
```

## 📊 Performance

| 指标 | 数值 | 说明 |
|------|------|------|
| Deep Research 平均收敛轮次 | ~2.3 轮 | 覆盖度阈值驱动迭代终止 |
| 去重率 | ~38% | Cross-Encoder 精排后去重 |
| Token 整合压缩率 | 降至 50% | 触发阈值后自动压缩 |
| RAG Top-10 召回率提升 | +25% | 多路召回+重排 vs 单路 |
| 多轮 RAG 准确率提升 | +15% | 多轮迭代 vs 单轮 |

> 注：以上为内部测试数据，实际表现因场景而异

## 🤝 Contribute

PRs welcome! The codebase is intentionally small and readable.

### Roadmap

- [ ] **Multi-modal** — 图像、语音、视频理解
- [ ] **Better reasoning** — 多步规划与反思
- [ ] **More integrations** — 日历、更多平台
- [ ] **Self-improvement** — 从反馈中学习

---

<p align="center">
  <em>让 AI 成为你的个人研究助手 🔬</em>
</p>
