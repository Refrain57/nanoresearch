"""LLM-based semantic classifier for badcase root-cause categorization."""

from __future__ import annotations

import json
import os
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

_FALLBACK_CATEGORY = "reasoning_error"

_SYSTEM_PROMPT = (
    "你是一位 Agent 质量分析专家。给定一条 badcase 的运行快照，"
    "请从以下语义分类中选择最匹配的一个：\n"
    + " | ".join(SEMANTIC_TAXONOMY)
    + "\n\n只输出分类标签，不要任何解释，不要引号。"
)

_MAX_TOOL_CHARS = 800
_MAX_RESP_CHARS = 800


class BadcaseClassifier:
    def __init__(self, provider: "LLMProvider", model: str | None = None) -> None:
        self._provider = provider
        self._model = model or os.environ.get("EVAL_JUDGE_MODEL") or provider.get_default_model()

    async def classify(self, snapshot: "AgentRunSnapshot") -> str:
        """Return a SEMANTIC_TAXONOMY label.  Falls back to 'reasoning_error' on failure."""
        prompt = _build_classify_prompt(snapshot)
        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "user", "content": prompt},
                ],
                system=_SYSTEM_PROMPT,
                model=self._model,
                max_tokens=32,
                temperature=0.0,
            )
            label = (response.content or "").strip().lower()
            if label in SEMANTIC_TAXONOMY:
                return label
        except Exception:
            pass
        return _FALLBACK_CATEGORY


def _build_classify_prompt(snapshot: "AgentRunSnapshot") -> str:
    parts: list[str] = [f"用户输入：{(snapshot.user_input or '')[:500]}"]

    chain = snapshot.tool_call_chain or []
    recent = chain[-3:] if len(chain) > 3 else chain
    if recent:
        chain_str = json.dumps(recent, ensure_ascii=False, default=str)
        if len(chain_str) > _MAX_TOOL_CHARS:
            chain_str = chain_str[:_MAX_TOOL_CHARS] + "...(truncated)"
        parts.append(f"最近工具调用：{chain_str}")

    resp = (snapshot.final_response or "(无回复)")[:_MAX_RESP_CHARS]
    parts.append(f"最终回复：{resp}")

    if snapshot.badcase_category:
        parts.append(f"规则触发类别（供参考）：{snapshot.badcase_category}")

    return "\n\n".join(parts)
