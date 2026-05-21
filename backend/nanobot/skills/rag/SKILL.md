---
name: rag
description: RAG knowledge base search with unified entry point.
always: false
---

# RAG Search

## 概述

RAG 工具用于检索知识库内容。支持多个 collection：

| Collection | 内容 | 使用场景 |
|------------|------|----------|
| `default` | 用户上传的文档 | 查询用户文档中的信息 |
| `research_claims` | Deep Search 研究结论 | 查询已有的研究事实 |
| `research_insights` | Deep Search 研究洞察 | 查询跨域规律 |

## 工具列表

| 工具 | 用途 |
|------|------|
| `mcp_rag_rag_search` | 统一检索入口（推荐） |
| `mcp_rag_retrieve_hybrid` | 混合检索（dense + sparse） |
| `mcp_rag_ingest_document` | 添加文档到知识库 |

## 使用示例

### 查询用户上传的文档
```
mcp_rag_rag_search(query="RAG 的实现原理", collection="default")
```

### 查询已有的研究结论
```
mcp_rag_rag_search(query="3DGS 的局限性", collection="research_claims")
```

### 查询跨域洞察
```
mcp_rag_rag_search(query="性能优化的一般方法", collection="research_insights")
```

### 直接使用 retrieve_hybrid
```
mcp_rag_retrieve_hybrid(
    query="项目技术选型",
    collection="default",
    top_k=10
)
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `query` | 用户查询（必填） | - |
| `collection` | 检索集合名称 | "default" |
| `top_k` | 返回结果数量 | 10 |

## 工作原理

`rag_search` / `retrieve_hybrid` 内部实现：
1. **Dense 检索**：向量相似度搜索
2. **Sparse 检索**：BM25 关键词匹配
3. **RRF 融合**：合并两种检索结果，按相关性排序
4. **可选重排序**：使用 reranker 进一步优化

## 何时使用

| 场景 | 推荐工具 |
|------|----------|
| 用户问"我的文档里说..." | `retrieve_hybrid(collection="default")` |
| 用户问"我们之前研究过..." | `retrieve_hybrid(collection="research_claims")` |
| 用户问"有什么跨领域规律..." | `retrieve_hybrid(collection="research_insights")` |
| 需要深入了解某个话题 | **先查知识库，再决定是否 deep_search** |

---

## 决策流程：遇到知识类问题怎么办？

**重要原则**：遇到需要查询知识的问题时，**先轻量查一次，再决定是否深度研究**。

### 决策树

```
用户提问
    │
    ▼
这是知识类问题吗？
（需要事实、原理、案例、对比等）
    │
    ├─► 否 → 直接回答 或 使用 web_search/web_fetch
    │
    └─► 是 → 先查知识库
              │
              ▼
         查询 research_claims + research_insights
              │
              ▼
         现有知识足够吗？
         （有相关结论，且置信度 >= 70%）
              │
              ├─► 是 → 直接基于现有知识回答
              │         （可引用来源，说明来自之前的研究）
              │
              └─► 否 → 调用 deep_search 进行完整研究
```

### 为什么先查知识库？

| 方式 | 耗时 | 成本 | 适用场景 |
|------|------|------|----------|
| 先查知识库 | ~1-2s | 低 | 已有结论，直接回答 |
| 直接 deep_search | 10-30min | 高 | 需要新研究 |

**示例**：
- 用户问："3DGS 相比 NeRF 有什么优势？"
  - 先查 `research_claims`：已有 "3DGS 训练速度比 NeRF 快 10 倍" → 直接回答
- 用户问："最新的大模型架构发展趋势？"
  - 先查 `research_claims`：无相关信息 → 调用 deep_search

### 操作步骤

**步骤 1：先查知识库**

```python
# 并行查询 claims 和 insights
claims = retrieve_hybrid(query="用户问题关键词", collection="research_claims", top_k=5)
insights = retrieve_hybrid(query="用户问题关键词", collection="research_insights", top_k=3)
```

**步骤 2：判断是否足够**

判断标准：
- claims 数量 >= 3 条，且平均置信度 >= 0.7
- 或 insights 有相关内容
- 内容与用户问题直接相关

**步骤 3：决定下一步**

| 判断结果 | 行动 |
|----------|------|
| 知识足够 | 直接回答，引用相关 claim/insight |
| 知识不足 | 调用 deep_search（它会自动补充新知识） |

### 示例

**场景 1：知识库已有答案**

```
用户："3DGS 在实时渲染方面有什么优势？"

Agent 执行：
1. 查询 research_claims("3DGS 实时渲染")
   → 找到 3 条相关 claims，置信度 0.8+
2. 现有知识足够 → 直接回答：
   "根据之前的研究，3DGS 在实时渲染方面的主要优势包括：
   - 训练速度比 NeRF 快 10 倍
   - 可以达到 60fps 的实时渲染速度
   - ...（引用来源）"
```

**场景 2：知识库没有答案**

```
用户："2025 年大模型 Agent 有什么最新进展？"

Agent 执行：
1. 查询 research_claims("大模型 Agent 2025")
   → 找到 0 条相关 claims
2. 查询 research_insights("Agent 架构")
   → 只有 2024 年的过时信息
3. 知识不足 → 调用 deep_search(topic="大模型 Agent 2025 最新进展", depth="normal")
```

---

## 与 Deep Search 的协作

### 重要：Deep Search 是耗时任务，必须后台执行

`research` 工具执行时间 10-30min，**必须通过 `spawn` 后台执行**，不能前台阻塞用户。

### 知识不足时的正确操作流程

当知识库查询结果不足时，按以下步骤操作：

**步骤 1：告知用户**

```
message("正在深度研究「{topic}」，预计 10-30 分钟，完成后通知您...")
```

**步骤 2：后台启动研究**

```
spawn(
    task="用 research 工具研究以下问题，完成后把完整报告发给用户：{用户的问题}",
    label="{简短标签}"
)
```

**步骤 3：主 Agent 直接返回**

告诉用户已启动后台研究，然后结束当前回复，不要等待。

### 完整示例

```
用户："2025 年大模型 Agent 有什么最新进展？"

Agent 执行：
1. 查询 research_claims("大模型 Agent 2025")
   → 找到 0 条相关 claims
2. 知识不足 → 准备启动后台研究
3. 调用 message 告知用户："正在深度研究「大模型 Agent 最新进展」，预计 10-30 分钟..."
4. 调用 spawn 后台执行：
   spawn(
       task="用 research 工具研究「2025 年大模型 Agent 最新进展」，完成后把完整报告发给用户",
       label="Agent 研究"
   )
5. 回复用户："已启动后台研究，完成后会通知您。"
6. 主 Agent 结束本轮对话，不等待研究完成

--- 后台 subagent 执行 ---
subagent 收到任务后：
1. 调用 research(action="start", topic="...", depth="normal")
2. 等待研究完成（10-30min）
3. 把完整报告发给用户（自动通知）
```

### 错误做法

❌ **前台直接调用 research**：
```
research(action="start", topic="...")
# 这会阻塞用户 10-30min，体验很差
```

✅ **通过 spawn 后台执行**：
```
message("正在研究...")
spawn(task="用 research 工具研究...")
"已启动后台研究，完成后通知您。"
```

### 工具职责分工

| 工具 | 职责 |
|------|------|
| `mcp_rag_rag_search` | 轻量查询（~1s），先查这个 |
| `message` | 告知用户状态 |
| `spawn` | 后台执行耗时任务 |
| `research` | 深度研究（由 subagent 调用） |

### 流程总结

```
知识类问题
    │
    ▼
RAG 查询（~1s）
    │
    ├─► 有答案 → 直接回答
    │
    └─► 无答案 → message 告知 + spawn 后台研究
                   │
                   └─► 主 Agent 返回，subagent 执行 research
```