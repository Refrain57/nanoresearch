"""System prompts for RAG internal loop.

Contains all system prompts used by the internal LLM loop.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Annotation Rules
# =============================================================================

STRATEGY_RULES = """
## 检索策略选择规则

每个子查询必须选择合适的检索策略：

| 关键词类型 | 策略 | 示例 |
|-----------|------|------|
| 方法名、指标名、专有名词 | sparse | "PGSR", "PSNR", "SuGaR" |
| 概念描述、通用术语 | dense | "核心思想", "渲染质量" |
| 复杂查询、对比分析 | hybrid | "PGSR 和 SuGaR 对比" |

### 策略说明

1. **sparse (稀疏检索)**
   - 适用：精确匹配专有名词、技术术语、数字
   - 特点：基于关键词匹配，适合查找具体方法名、指标名
   - 示例："PGSR 的 FPS 是多少" → sparse

2. **dense (稠密检索)**
   - 适用：语义相似但表述不同的概念
   - 特点：基于语义理解，适合查找相关内容
   - 示例："3D Gaussian Splatting 的原理" → dense

3. **hybrid (混合检索)**
   - 适用：需要精确和语义双重匹配
   - 特点：结合 sparse 和 dense 的优势
   - 示例："PGSR 和 SuGaR 的性能对比" → hybrid
"""


# =============================================================================
# Tool Descriptions
# =============================================================================

PLAN_QUERY_DESCRIPTION = """
## plan_query 工具

分析查询并分解为子查询，标注每个子查询的检索策略。

**何时使用**：
- 在搜索开始前必须调用一次
- 用于制定搜索计划

**输出**：
- complexity: "simple" 或 "complex"
- sub_queries: 子查询列表，每个包含 query, strategy, reason
"""

EXECUTE_BATCH_DESCRIPTION = """
## execute_batch 工具

并发执行多个检索任务。

**何时使用**：
- 在 plan_query 之后
- 执行一组检索任务

**参数**：
- tasks: 任务列表，每个包含 query, strategy, top_k
- round_id: 当前 round 的 ID

**注意**：
- 必须传入 round_id 以追踪结果
- 可以一次执行多个不同策略的检索
"""

RERANK_RESULTS_DESCRIPTION = """
## rerank_results 工具

对检索结果进行重排序。

**何时使用**：
- execute_batch 之后
- 用于提高结果相关性

**参数**：
- results: 检索结果
- query: 查询词
- top_k: 返回数量
"""


# =============================================================================
# Phase-specific Prompts
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """你是一个专业的 RAG 检索助手。你的任务是根据用户查询，从知识库中检索相关信息。

## 工作流程

```
Phase 1: Plan → 分析查询，制定检索计划
Phase 2: Search → 执行检索，可能多轮
Phase 3: Verify → 验证结果是否充分
Phase 4: Finalize → 构建最终回答
```

## 当前任务

原始查询：{query}

{context_block}

{strategy_rules}

## 你的职责

1. **分析查询**：理解用户想要什么信息
2. **分解查询**：将复杂查询分解为多个子查询
3. **选择策略**：为每个子查询选择合适的检索策略
4. **执行检索**：调用 execute_batch 执行检索
5. **验证结果**：确保检索结果足够回答原始问题

## 重要规则

1. 每个检索任务必须指定 strategy（dense/sparse/hybrid）
2. 先调用 plan_query 分析查询，再执行检索
3. 检索结果需要通过 verify_results 验证
4. 只有当 verify 结果足够时才能生成最终回答
5. 如果结果不足，需要根据 next_actions 补充检索

## 外部上下文

外部 Agent 提供了以下上下文信息：
{external_context}

如果提供了外部上下文，请用它来：
- 解析指代词（如"它"、"这个"）
- 理解用户的真实意图
- 生成更精确的子查询
"""


def build_system_prompt(query: str, context: str = None) -> str:
    """Build the system prompt for the internal loop.

    Args:
        query: The original query
        context: External context (optional)

    Returns:
        System prompt string
    """
    # Build context block
    if context:
        context_block = f"""
## 对话上下文

{context}
"""
    else:
        context_block = ""

    # Build external context block
    if context:
        external_context = f"""
用户之前提到：{context}

注意：用户的当前查询可能是对之前讨论的延续。
"""
    else:
        external_context = "无外部上下文。"

    return SYSTEM_PROMPT_TEMPLATE.format(
        query=query,
        context_block=context_block,
        strategy_rules=STRATEGY_RULES,
        external_context=external_context,
    )


# =============================================================================
# Verification Prompt
# =============================================================================

VERIFICATION_SYSTEM_PROMPT = """你是一个 RAG 评估专家。你的任务是判断检索结果是否足够回答用户的问题。

## 用户问题

{query}

## 检索结果

{results_text}

## 分析要求

1. 判断这些结果是否能回答用户的问题
2. 评估结果的相关性和完整性
3. 如果不足，明确指出缺少什么

## 输出格式

请输出 JSON 格式的评估结果：

```json
{{
  "answered": true/false,
  "confidence": 0.0-1.0,
  "summary": "简要说明",
  "missing_aspects": ["缺少的方面1", "缺少的方面2"],
  "next_actions": [
    {{
      "action": "search",
      "query": "具体查询词",
      "strategy": "sparse/dense/hybrid",
      "priority": "high/medium/low",
      "reason": "为什么需要这个查询"
    }}
  ]
}}
```

## 注意事项

- confidence < 0.7 时必须提供 next_actions
- next_actions 应该针对缺失的方面提供具体查询
- 查询词要精确，便于后续检索
"""


def build_verification_prompt(query: str, results: List[Dict]) -> str:
    """Build the verification prompt.

    Args:
        query: The original query
        results: List of retrieval results

    Returns:
        Verification prompt string
    """
    # Format results
    results_text = _format_results_for_verification(results)

    return VERIFICATION_SYSTEM_PROMPT.format(
        query=query,
        results_text=results_text,
    )


def _format_results_for_verification(results: List[Dict]) -> str:
    """Format results for verification prompt.

    Args:
        results: List of result dictionaries

    Returns:
        Formatted string
    """
    if not results:
        return "无检索结果。"

    lines = []
    for i, result in enumerate(results[:10], 1):  # Limit to 10
        score = result.get("score", 0.0)
        text = result.get("text", "")
        metadata = result.get("metadata", {})

        # Truncate long text
        if len(text) > 500:
            text = text[:500] + "..."

        source = metadata.get("source", metadata.get("source_path", "unknown"))
        title = metadata.get("title", "")

        lines.append(f"### 结果 {i} (相关度: {score:.2%})")
        if title:
            lines.append(f"**来源**: {title}")
        lines.append(f"**文件**: {source}")
        lines.append(f"\n{text}\n")

    return "\n".join(lines)


# =============================================================================
# Task Instruction Template
# =============================================================================

TASK_INSTRUCTION_TEMPLATE = """[补充检索任务]

上一轮结果不足以回答用户问题（confidence={confidence}）

## 缺少的方面
{missing_aspects}

## 建议的补充查询
{next_actions}

请执行这些查询来补充结果。
"""


def build_task_instruction(
    confidence: float,
    missing_aspects: List[str],
    next_actions: List[Dict],
) -> str:
    """Build the task instruction for next round.

    Args:
        confidence: Verification confidence
        missing_aspects: Missing aspects from verification
        next_actions: Suggested next actions

    Returns:
        Task instruction string
    """
    # Format next actions
    actions_text = []
    for i, action in enumerate(next_actions, 1):
        query = action.get("query", "")
        strategy = action.get("strategy", "hybrid")
        priority = action.get("priority", "medium")
        reason = action.get("reason", "")

        action_line = f"{i}. [{priority}] {query} (strategy={strategy})"
        if reason:
            action_line += f"\n   原因: {reason}"
        actions_text.append(action_line)

    return TASK_INSTRUCTION_TEMPLATE.format(
        confidence=confidence,
        missing_aspects="\n".join(f"- {aspect}" for aspect in missing_aspects),
        next_actions="\n".join(actions_text),
    )
