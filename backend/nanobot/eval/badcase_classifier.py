"""LLM-based semantic classifier for badcase root-cause categorization."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.storage.models import AgentRunSnapshot

SEMANTIC_TAXONOMY = [
    "retrieval_failure",
    "hallucination",
    "tool_error",
    "reasoning_error",
    "context_loss",
    "instruction_following",
    "output_format",
]

TAXONOMY_LABELS_ZH = {
    "retrieval_failure": "检索失败",
    "hallucination": "幻觉捏造",
    "tool_error": "工具调用错误",
    "reasoning_error": "推理错误",
    "context_loss": "上下文丢失",
    "instruction_following": "指令遵循问题",
    "output_format": "输出格式问题",
}

ROOT_CAUSE_VALUES = ["prompt", "context", "tool", "model", "user_input"]

_FALLBACK_CATEGORY = "reasoning_error"
_FALLBACK_ROOT_CAUSE = "prompt"
_MODEL_TOKEN_LIMIT = 128_000
_HIGH_TOKEN_RATIO = 0.85
_MAX_TOOL_CHARS = 500
_MAX_RESP_CHARS = 800


@dataclass
class ClassifyResult:
    semantic_category: str
    root_cause_auto: str
    confidence: str  # high | medium | low
    reason: str


_SYSTEM_PROMPT = (
    "你是一位 Agent 质量分析专家。给定一条 badcase 的运行快照，"
    "请同时完成两个分类任务：\n"
    "1. 语义分类（semantic_category），从以下选项中选一个：\n"
    "   " + " | ".join(SEMANTIC_TAXONOMY) + "\n"
    "2. 根因分类（root_cause_auto），从以下选项中选一个：\n"
    "   prompt | context | tool | model | user_input\n\n"
    "根因判断锚点：\n"
    "- prompt：工具确实返回了内容，但 Agent 忽略了或错误解读了这些内容；Agent 推理方式或行为模式有问题\n"
    "- context：工具返回内容本身就不足以回答问题（检索结果为空或完全无关）\n"
    "- tool：工具返回错误数据、空结果、解析失败\n"
    "- model：复杂推理或长上下文丢失信息，模型能力边界\n"
    "- user_input：用户表达歧义，根因在输入侧\n\n"
    "请先用 2-3 句话进行 chain-of-thought 分析，然后输出 JSON：\n"
    '{"semantic_category": "...", "root_cause_auto": "...", "confidence": "high|medium|low", "reason": "一句话理由"}\n\n'
    "confidence 规则：context 和 prompt 难以区分时输出 low，此时归 context（保守策略）。"
)


class BadcaseClassifier:
    def __init__(self, provider: "LLMProvider", model: str | None = None) -> None:
        self._provider = provider
        self._model = model or os.environ.get("EVAL_JUDGE_MODEL") or provider.get_default_model()

    async def classify(self, snapshot: "AgentRunSnapshot") -> ClassifyResult:
        """Return ClassifyResult with semantic_category + root_cause_auto. Falls back gracefully."""
        rule_result = _rule_based_root_cause(snapshot)
        if rule_result is not None:
            return rule_result

        prompt = _build_classify_prompt(snapshot)
        try:
            response = await self._provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM_PROMPT,
                model=self._model,
                max_tokens=384,
                temperature=0.0,
            )
            return _parse_llm_response(response.content or "")
        except Exception:
            pass
        return ClassifyResult(
            semantic_category=_FALLBACK_CATEGORY,
            root_cause_auto=_FALLBACK_ROOT_CAUSE,
            confidence="low",
            reason="分类失败，降级处理",
        )


def _rule_based_root_cause(snapshot: "AgentRunSnapshot") -> ClassifyResult | None:
    chain = snapshot.tool_call_chain or []

    if any(entry.get("error") is True for entry in chain):
        return ClassifyResult(
            semantic_category="tool_error",
            root_cause_auto="tool",
            confidence="high",
            reason="工具调用链中存在 error=True 的条目",
        )

    if chain and all(_is_empty_output(entry.get("output")) for entry in chain):
        return ClassifyResult(
            semantic_category="retrieval_failure",
            root_cause_auto="context",
            confidence="high",
            reason="所有工具调用均返回空结果",
        )

    if snapshot.run_status == "failed" and not chain:
        return ClassifyResult(
            semantic_category="reasoning_error",
            root_cause_auto="prompt",
            confidence="medium",
            reason="运行失败且无工具调用，疑似 prompt 推理问题",
        )

    if snapshot.total_input_tokens >= _MODEL_TOKEN_LIMIT * _HIGH_TOKEN_RATIO:
        return ClassifyResult(
            semantic_category="context_loss",
            root_cause_auto="model",
            confidence="high",
            reason=f"输入 token 超过模型上限 {int(_HIGH_TOKEN_RATIO * 100)}%",
        )

    return None


def _is_empty_output(output: object) -> bool:
    if output is None:
        return True
    if isinstance(output, str) and output.strip() == "":
        return True
    if isinstance(output, list) and len(output) == 0:
        return True
    return False


def _parse_llm_response(text: str) -> ClassifyResult:
    # LLM outputs CoT first then JSON — try matches from last to first so CoT
    # fragments like "{工具}" don't shadow the real JSON block at the end.
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        try:
            data = json.loads(match.group())
            semantic = data.get("semantic_category", "").strip().lower()
            root_cause = data.get("root_cause_auto", "").strip().lower()
            confidence = data.get("confidence", "low").strip().lower()
            reason = str(data.get("reason", "")).strip()

            if semantic not in SEMANTIC_TAXONOMY:
                semantic = _FALLBACK_CATEGORY
            if root_cause not in ROOT_CAUSE_VALUES:
                root_cause = _FALLBACK_ROOT_CAUSE
            if confidence not in ("high", "medium", "low"):
                confidence = "low"

            # Only accept if the JSON had at least one of the expected keys
            if data.get("semantic_category") or data.get("root_cause_auto"):
                return ClassifyResult(
                    semantic_category=semantic,
                    root_cause_auto=root_cause,
                    confidence=confidence,
                    reason=reason,
                )
        except (json.JSONDecodeError, KeyError):
            continue
    return ClassifyResult(
        semantic_category=_FALLBACK_CATEGORY,
        root_cause_auto=_FALLBACK_ROOT_CAUSE,
        confidence="low",
        reason="LLM 输出解析失败",
    )


def _build_classify_prompt(snapshot: "AgentRunSnapshot") -> str:
    parts: list[str] = [f"用户输入：{(snapshot.user_input or '')[:500]}"]

    chain = snapshot.tool_call_chain or []
    if chain:
        summaries = []
        for entry in chain:
            s = json.dumps(entry, ensure_ascii=False, default=str)
            summaries.append(s[:_MAX_TOOL_CHARS] + ("..." if len(s) > _MAX_TOOL_CHARS else ""))
        parts.append("工具调用链：\n" + "\n".join(summaries))

    resp = (snapshot.final_response or "(无回复)")[:_MAX_RESP_CHARS]
    parts.append(f"最终回复：{resp}")

    meta = snapshot.judge_metadata or {}
    if meta.get("failed_dimensions"):
        parts.append(f"Judge 失分维度：{', '.join(meta['failed_dimensions'])}")
    if meta.get("comment"):
        parts.append(f"Judge 原始评语：{str(meta['comment'])[:300]}")

    if snapshot.badcase_category:
        parts.append(f"规则触发类别（供参考）：{snapshot.badcase_category}")

    return "\n\n".join(parts)
