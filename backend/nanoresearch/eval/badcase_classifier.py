"""LLM-based semantic classifier for badcase root-cause categorization.

Phase 1 upgrade: output now includes a structured pointer
  (layer, target_kind, target_id) in addition to the legacy root_cause_auto.

Layer semantics (SDD §3.2):
  fixable_layers      = {Context, Tool}   → have TunableTextObject implementations
  diagnosis_only      = {Memory, Recovery} → pointer only, no auto-fix chain
  layer=None          = user_input cases or ambiguous cases where the classifier
                        cannot determine a system-side root cause with confidence

Rule-layer new-user safety note:
  The "all tools returned empty" pattern is NOT handled in the rule layer because
  it cannot distinguish "new user with no memories" from "retrieval strategy failure".
  Both produce fragment_ids=[], history_actual_chars=0.  This case is intentionally
  delegated to the LLM, which receives the full context_trace and can reason about it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoresearch.providers.base import LLMProvider
    from nanoresearch.storage.models import AgentRunSnapshot

SEMANTIC_TAXONOMY = [
    "retrieval_failure",
    "hallucination",
    "tool_error",
    "reasoning_failure",
    "context_loss",
    "instruction_following",
    "output_format",
]

TAXONOMY_LABELS_ZH = {
    "retrieval_failure": "检索失败",
    "hallucination": "幻觉捏造",
    "tool_error": "工具调用错误",
    "reasoning_failure": "推理失败",
    "context_loss": "上下文丢失",
    "instruction_following": "指令遵循问题",
    "output_format": "输出格式问题",
}

# Legacy root_cause_auto values — preserved for backward compatibility
ROOT_CAUSE_VALUES = ["prompt", "context", "tool", "model", "user_input"]

# Phase 1: layer enumeration with explicit fixable / diagnosis_only split
FIXABLE_LAYERS = {"Context", "Tool"}
DIAGNOSIS_ONLY_LAYERS = {"Memory", "Recovery"}
ALL_LAYERS = FIXABLE_LAYERS | DIAGNOSIS_ONLY_LAYERS

_FALLBACK_CATEGORY = "reasoning_failure"
_FALLBACK_ROOT_CAUSE = "prompt"
_MODEL_TOKEN_LIMIT = 128_000
_HIGH_TOKEN_RATIO = 0.85
_MAX_TOOL_CHARS = 500
_MAX_RESP_CHARS = 800


@dataclass
class ClassifyResult:
    semantic_category: str
    root_cause_auto: str        # legacy field — preserved, parallel-written
    confidence: str             # high | medium | low
    reason: str
    # Phase 1: structured pointer
    layer: str | None = field(default=None)            # "Context"|"Tool"|"Memory"|"Recovery"|None
    target_kind: str | None = field(default=None)      # e.g. "system_prompt", "tool_description"
    target_id: str | None = field(default=None)        # agent_id | tool_name | None


_SYSTEM_PROMPT = (
    "你是一位 Agent 质量分析专家。给定一条 badcase 的运行快照，"
    "请同时完成两个分类任务：\n"
    "1. 语义分类（semantic_category），从以下选项中选一个：\n"
    "   " + " | ".join(SEMANTIC_TAXONOMY) + "\n"
    "   reasoning_failure：推理没有成功——无论是推理链路断裂，还是有检索结果但未能正确利用\n"
    "2. 结构化根因指针，输出以下三个字段：\n"
    "   layer：从 Context | Tool | Memory | Recovery 中选一个，或输出 null（用户侧问题、无系统根因时）\n"
    "   target_kind：layer 内的具体对象类型，从下表选择：\n"
    "     Context → system_prompt（prompt 指令有问题）| retrieval_strategy（检索 query/预算/top-k 有问题）\n"
    "     Tool    → tool_description（工具描述歧义）| tool_impl（工具本身返回错误）\n"
    "     Memory  → memory_write_rule（记忆写入策略）| memory_decay_weight（衰减参数，数值类）\n"
    "     Recovery → circuit_breaker_threshold（熔断阈值，数值类）| timeout_config（超时配置，数值类）\n"
    "     null → user_input（用户表达歧义，不在平台修复范围）\n"
    "   target_id：具体对象标识（工具名、agent_id 等），如无法确定则输出 null\n"
    "3. 同时输出 root_cause_auto（向后兼容）：prompt | context | tool | model | user_input\n\n"
    "根因判断锚点：\n"
    "- Context/system_prompt：有检索结果但 Agent 忽略了或错误解读；或 Agent 推理行为有系统性问题\n"
    "- Context/retrieval_strategy：检索查询为空、结果与问题无关、检索预算明显不足\n"
    "- Tool/tool_description：工具被错误调用或未被调用（描述歧义导致）\n"
    "- Tool/tool_impl：工具返回 error、空结果、解析失败（工具本身有 bug）\n"
    "- Memory/memory_write_rule：记忆固化策略导致关键信息丢失\n"
    "- Recovery/circuit_breaker_threshold：escape hatch 触发时机不对\n"
    "- null/user_input：用户表达歧义，根因在输入侧，系统无法自动修复\n\n"
    "注意：\n"
    "- fragment_ids 为空不一定是 retrieval_strategy 问题——新用户无历史记忆时这是正常情况\n"
    "- fixable（Context/Tool）层有自动修复链路；Memory/Recovery 层只有诊断指针，没有自动修复\n\n"
    "请先用 2-3 句话进行 chain-of-thought 分析，然后输出 JSON：\n"
    '{"semantic_category": "...", "root_cause_auto": "...", '
    '"layer": "...", "target_kind": "...", "target_id": "...", '
    '"confidence": "high|medium|low", "reason": "一句话理由"}\n\n'
    "confidence 规则：layer 或 target_kind 难以确定时输出 low；"
    "context/prompt 难以区分时归 Context（保守策略）。"
)


class BadcaseClassifier:
    def __init__(self, provider: "LLMProvider", model: str | None = None) -> None:
        self._provider = provider
        self._model = model or os.environ.get("EVAL_JUDGE_MODEL") or provider.get_default_model()

    async def classify(self, snapshot: "AgentRunSnapshot") -> ClassifyResult:
        """Return ClassifyResult with semantic_category + structured pointer. Falls back gracefully."""
        rule_result = _rule_based_root_cause(snapshot)
        if rule_result is not None:
            return rule_result

        prompt = _build_classify_prompt(snapshot)
        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self._model,
                max_tokens=512,
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
            layer=None,
            target_kind=None,
            target_id=None,
        )


def _rule_based_root_cause(snapshot: "AgentRunSnapshot") -> ClassifyResult | None:
    """High-confidence rule-based shortcuts.

    Only fires when a single observable fact unambiguously determines the outcome.
    Cases that require reasoning about context_trace (e.g. "did the user have
    any memories?") are intentionally NOT handled here — they go to the LLM
    which receives the full context_trace as structured input.
    """
    chain = snapshot.tool_call_chain or []

    # Rule 1: explicit tool error flag — unambiguous, always Tool/tool_impl
    if any(entry.get("error") is True for entry in chain):
        tool_name = next(
            (e.get("tool") or e.get("name") for e in chain if e.get("error")), None
        )
        return ClassifyResult(
            semantic_category="tool_error",
            root_cause_auto="tool",
            confidence="high",
            reason="工具调用链中存在 error=True 的条目",
            layer="Tool",
            target_kind="tool_impl",
            target_id=tool_name,
        )

    # Rule 2: run failed with no tool calls — unambiguous prompt/system_prompt issue
    if snapshot.run_status == "failed" and not chain:
        return ClassifyResult(
            semantic_category="reasoning_failure",
            root_cause_auto="prompt",
            confidence="medium",
            reason="运行失败且无工具调用，疑似 system prompt 推理问题",
            layer="Context",
            target_kind="system_prompt",
            target_id=None,
        )

    # Rule 3: retrieved results but agent failed to use them
    scores = snapshot.scores or {}
    contextual_recall = scores.get("contextual_recall")
    task_completion = scores.get("task_completion")
    if (
        contextual_recall is not None
        and task_completion is not None
        and contextual_recall >= 0.5
        and task_completion < 0.5
    ):
        return ClassifyResult(
            semantic_category="reasoning_failure",
            root_cause_auto="prompt",
            confidence="high",
            reason=(
                f"contextual_recall={contextual_recall:.2f} 合格但 "
                f"task_completion={task_completion:.2f} 偏低，有检索结果但未正确利用"
            ),
            layer="Context",
            target_kind="system_prompt",
            target_id=None,
        )

    # Rule 4: token limit — cannot determine layer without context_trace analysis;
    # attach budget evidence for human review, leave layer=None.
    # (Splitting to Context/Memory via memory_chars heuristic would manufacture a
    # false pointer when the root cause might be task complexity.)
    if snapshot.total_input_tokens >= _MODEL_TOKEN_LIMIT * _HIGH_TOKEN_RATIO:
        ct = snapshot.context_trace or {}
        reason = (
            f"输入 token 超过模型上限 {int(_HIGH_TOKEN_RATIO * 100)}%；"
            f"预算分配：memory={ct.get('memory_budget_tokens', 0)} tokens, "
            f"knowledge={ct.get('knowledge_budget_tokens', 0)} tokens；"
            f"实际注入：memory={ct.get('memory_actual_chars', 0)} chars, "
            f"history={ct.get('history_actual_chars', 0)} chars"
        )
        return ClassifyResult(
            semantic_category="context_loss",
            root_cause_auto="model",
            confidence="medium",
            reason=reason,
            layer=None,
            target_kind=None,
            target_id=None,
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
    # LLM outputs CoT first then JSON — match from last to first so CoT
    # fragments like "{工具}" don't shadow the real JSON block.
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        try:
            data = json.loads(match.group())
            semantic = data.get("semantic_category", "").strip().lower()
            root_cause = data.get("root_cause_auto", "").strip().lower()
            confidence = data.get("confidence", "low").strip().lower()
            reason = str(data.get("reason", "")).strip()
            layer_raw = data.get("layer")
            layer = layer_raw if layer_raw in ALL_LAYERS else None
            target_kind = data.get("target_kind") or None
            target_id = data.get("target_id") or None

            if semantic not in SEMANTIC_TAXONOMY:
                semantic = _FALLBACK_CATEGORY
            if root_cause not in ROOT_CAUSE_VALUES:
                root_cause = _FALLBACK_ROOT_CAUSE
            if confidence not in ("high", "medium", "low"):
                confidence = "low"

            if data.get("semantic_category") or data.get("root_cause_auto") or data.get("layer"):
                return ClassifyResult(
                    semantic_category=semantic,
                    root_cause_auto=root_cause,
                    confidence=confidence,
                    reason=reason,
                    layer=layer,
                    target_kind=target_kind,
                    target_id=target_id,
                )
        except (json.JSONDecodeError, KeyError):
            continue
    return ClassifyResult(
        semantic_category=_FALLBACK_CATEGORY,
        root_cause_auto=_FALLBACK_ROOT_CAUSE,
        confidence="low",
        reason="LLM 输出解析失败",
        layer=None,
        target_kind=None,
        target_id=None,
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

    # Phase 1: include context_trace as structured evidence
    ct = snapshot.context_trace or {}
    if ct:
        history_query = ct.get("history_query")
        fids = ct.get("memory_fragment_ids", [])
        lines = [
            f"- history_query: {'「' + history_query + '」' if history_query else '未发起检索'}",
            f"- memory_fragment_ids: {len(fids)} 个{'（空——可能是新用户无记忆，或检索未命中，需结合上下文判断）' if not fids else ''}",
            f"- memory_actual_chars: {ct.get('memory_actual_chars', 0)}",
            f"- history_actual_chars: {ct.get('history_actual_chars', 0)}",
            f"- memory_budget_tokens: {ct.get('memory_budget_tokens', 0)}",
            f"- knowledge_budget_tokens: {ct.get('knowledge_budget_tokens', 0)}",
            f"- persona_active: {ct.get('persona_active', False)}",
            f"- skills: {ct.get('skill_names') or ct.get('always_skill_names') or '无'}",
        ]
        parts.append("记忆检索与上下文装配决策（Phase 0 trace）：\n" + "\n".join(lines))

    return "\n\n".join(parts)
