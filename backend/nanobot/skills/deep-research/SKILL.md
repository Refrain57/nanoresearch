---
name: deep-research
description: 调用 research Tool 进行系统性网络研究。当用户需要深入了解某个话题、对比多个观点、生成研究报告时使用。此工具会自动完成搜索、抓取、综合、报告生成全流程。
---

# Deep Research Skill

## 概述

本 Skill 指导如何正确调用 `research` Tool 进行系统性网络研究。

`research` Tool 是一个自动化研究引擎，内部封装了：
- 规划：将主题分解为多个子问题
- 搜索：并行搜索 + 去重 + Rerank
- 综合：提取结构化发现
- 迭代：根据覆盖度自动决定是否继续搜索
- 报告：生成带引用的研究报告
- 知识写入：提取 claims/insights 到知识库

---

## 何时调用 research Tool

| 场景 | 推荐 | 原因 |
|-----|------|------|
| 需要 5+ 来源支撑的深度研究 | `research` Tool | 自动化流程，效率高 |
| 对比多个观点/方案 | `research` Tool | 自动提取矛盾点 |
| 生成研究报告/文档 | `research` Tool | 自动生成带引用的报告 |
| 快速查询一个事实 | `web_search` | 单次搜索足够 |
| 验证某个具体链接内容 | `web_fetch` | 已知 URL 直接抓取 |
| 特殊领域（学术论文、法律） | 自己组合 `web_search` + `web_fetch` | 需要特定搜索策略 |

---

## 调用参数

```json
{
  "action": "start",
  "topic": "研究主题",
  "depth": "normal",
  "background": true
}
```

| 参数 | 类型 | 说明 |
|-----|------|------|
| `action` | string | `"start"` 启动研究 / `"status"` 查看进度 / `"list"` 列出历史 |
| `topic` | string | 研究主题（仅 start 时需要） |
| `depth` | string | `"quick"` / `"normal"` / `"deep"` |
| `background` | bool | 是否后台运行（默认 true） |

### depth 参数说明

| depth | 迭代轮次 | 每轮来源数 | 适用场景 |
|-------|---------|-----------|---------|
| `quick` | 1 轮 | 5 篇 | 快速了解，时间敏感 |
| `normal` | 3 轮 | 10 篇 | 常规研究（默认） |
| `deep` | 5 轮 | 20 篇 | 深度研究，学术报告 |

---

## 返回结构

调用成功后返回：

```json
{
  "id": "abc123",
  "topic": "AI 医疗",
  "status": "completed",
  "iterations": 2,
  "total_sources": 15,
  "quality_score": 7.5,
  "report": "# AI 医疗研究报告\n\n...",
  "execution_log": {
    "research_id": "abc123",
    "depth": "normal",
    "stop_reason": "coverage_threshold",
    "final_coverage_score": 0.72,
    "iterations": [
      {
        "iteration": 0,
        "sub_questions_searched": ["AI healthcare", "medical AI"],
        "search_results_count": 12,
        "coverage_score": 0.65
      }
    ],
    "knowledge_write": {
      "claims": 5,
      "insights": 2
    }
  }
}
```

---

## 质量标准

### 停止条件

研究引擎会在以下情况停止迭代：

| 条件 | 说明 |
|-----|------|
| `coverage_score >= 0.7` | 覆盖度达标，停止 |
| `iteration >= max_iterations` | 达到最大轮次 |
| 无知识缺口 | Refiner 判断无需继续 |

### 质量分数

`quality_score` 是自评估分数（0-10）：

| 分数 | 含义 |
|-----|------|
| >= 8.0 | 高质量，可直接使用 |
| 6.0 - 8.0 | 可接受，建议人工复核 |
| < 6.0 | 质量不足，自动重试报告 |

---

## 执行日志解读

`execution_log` 提供完整的执行过程（白盒化）：

### stop_reason 含义

| 值 | 说明 |
|-----|------|
| `coverage_threshold` | 覆盖度达标，正常停止 |
| `max_iterations` | 达到最大轮次，可能覆盖不足 |
| `no_gaps` | 无知识缺口，提前停止 |
| `failed` | 执行失败 |

### 每轮迭代日志

```json
{
  "iteration": 0,
  "sub_questions_searched": ["AI healthcare", "medical imaging"],
  "search_results_count": 12,
  "rerank_enabled": true,
  "coverage_score": 0.65,
  "stopped": false
}
```

- `sub_questions_searched`: 本轮搜索的关键词
- `search_results_count`: 本轮搜索结果数
- `coverage_score`: 本轮覆盖度

---

## 失败处理

| 情况 | 处理方式 |
|-----|---------|
| 网络失败 | 自动重试 3 次 |
| 知识库冲突 | 记录但不阻塞报告 |
| `quality_score < 6.0` | 自动重试报告一次 |
| `stop_reason = failed` | 检查日志，人工介入 |

---

## 最佳实践

### 1. 选择合适的 depth

```
用户问："简单介绍一下 AI 医疗"
→ depth="quick"（1轮/5源）

用户问："写一份 AI 医疗研究报告"
→ depth="normal"（3轮/10源）

用户问："AI 医疗的学术综述"
→ depth="deep"（5轮/20源）
```

### 2. 检查 execution_log

```
stop_reason = "max_iterations" 且 final_coverage_score < 0.7
→ 覆盖不足，建议用 depth="deep" 重跑
```

### 3. 利用 knowledge_write

```
knowledge_write.claims = 5
→ 研究结果已写入知识库，后续可复用
```

---

## 与原子工具的协作

`research` Tool 内部会调用 `web_search` 和 `web_fetch`，因此：

- 调用 `research` 后，无需再手动调用 `web_search`
- 如需特定链接详情，可在 `research` 后补充 `web_fetch`
- `research` 返回的 `report` 已包含综合结果，可直接使用

---

## 示例

### 快速研究

```
用户："AI 医疗最近有什么进展？"

调用：
research(action="start", topic="AI 医疗最新进展", depth="quick")

预期返回：
- iterations: 1
- total_sources: ~5
- report: 简要报告
```

### 深度研究

```
用户："帮我写一份 AI 医疗诊断技术的调研报告"

调用：
research(action="start", topic="AI 医疗诊断技术", depth="deep")

预期返回：
- iterations: 3-5
- total_sources: 30-50
- report: 详细报告 + 引用
- knowledge_write: 多条 claims/insights
```
