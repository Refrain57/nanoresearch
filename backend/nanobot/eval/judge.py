"""LLM-based judge for scoring agent run snapshots."""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.eval.snapshot import RunSnapshotData
    from nanobot.providers.base import LLMProvider
    from nanobot.storage.models import AgentTestCase

@dataclass
class CalibrationResult:
    passed: bool
    mad: float
    sample_count: int
    judge_model: str


_SYSTEM_PROMPT = """\
你是一位专业的 AI Agent 质量评审员。你的任务是评估 Agent 对用户请求的处理质量。

请根据以下维度对 Agent 的表现打分（1-5 整数）：
- tool_rationality：工具调用的合理性（选择了正确的工具、顺序合理、没有多余调用）
- task_completion：任务完成度（用户问题是否真正得到了解答）
- response_logic：回复逻辑清晰度（结构清晰、内容准确、表述流畅）
- hallucination：幻觉检测——对比 Agent 最终回复与工具实际返回值，检查有无捏造数据、虚假引用或与工具返回不符的陈述。以工具返回值为 ground truth，1 = 严重幻觉，5 = 完全准确

仅在有多轮对话历史时额外评估：
- multi_turn_coherence：多轮对话连贯性（前后一致、上下文理解准确）

评分标准：
1 = 很差，2 = 较差，3 = 一般，4 = 良好，5 = 优秀

**必须以如下 JSON 格式输出，不要有其他内容：**
{"scores": {"tool_rationality": <int>, "task_completion": <int>, "response_logic": <int>, "hallucination": <int>}, "reasoning": "<一句话说明>"}
"""

_MAX_TOOL_CHAIN_CHARS = 3000
_MAX_RESPONSE_CHARS = 2000


class LLMJudge:
    """Score an agent run snapshot using an LLM as evaluator.

    Failures are swallowed — returns {} on any parse error so the calling
    code can treat judge scores as optional enrichment.
    """

    def __init__(self, provider: "LLMProvider", model: str | None = None) -> None:
        self._provider = provider
        self._model = model or os.environ.get("EVAL_JUDGE_MODEL") or provider.get_default_model()

    async def score(
        self,
        snapshot: "RunSnapshotData",
        test_case: "AgentTestCase | None" = None,
        session_history: list[dict] | None = None,
    ) -> tuple[dict[str, float], str]:
        """Return (scores, raw_output).  scores is {dimension: 0.0-1.0}.  Empty dict on failure."""
        prompt = _build_prompt(snapshot, test_case, session_history)
        try:
            response = await self._provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                max_tokens=512,
                temperature=0.0,
            )
            raw = response.content or ""
            return _parse_scores(raw), raw
        except Exception:
            return {}, ""

    async def calibrate(
        self,
        samples: list[tuple["RunSnapshotData", "AgentTestCase"]],
    ) -> "CalibrationResult":
        """Run judge on calibration samples and check MAD vs human_score.

        Returns CalibrationResult.  passed=True if MAD <= 0.15.
        Returns passed=True, mad=0.0 if no samples have human_score set.
        """
        deviations: list[float] = []
        for snapshot, tc in samples:
            if tc.human_score is None:
                continue
            scores, _ = await self.score(snapshot, tc)
            if not scores:
                continue
            judge_avg = sum(scores.values()) / len(scores)
            deviations.append(abs(judge_avg - tc.human_score))

        if not deviations:
            return CalibrationResult(passed=True, mad=0.0, sample_count=0, judge_model=self._model)
        mad = statistics.mean(deviations)
        return CalibrationResult(
            passed=mad <= 0.15,
            mad=round(mad, 4),
            sample_count=len(deviations),
            judge_model=self._model,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(
    snapshot: "RunSnapshotData",
    test_case: "AgentTestCase | None",
    session_history: list[dict] | None,
) -> str:
    parts: list[str] = []

    if session_history:
        history_str = "\n".join(
            f"[{m.get('role', '?')}]: {str(m.get('content', ''))[:300]}"
            for m in session_history[-6:]
        )
        parts.append(f"## 历史对话\n{history_str}")

    parts.append(f"## 用户输入\n{snapshot.user_input[:1000]}")

    if snapshot.tool_call_chain:
        chain_str = json.dumps(snapshot.tool_call_chain, ensure_ascii=False, default=str)
        if len(chain_str) > _MAX_TOOL_CHAIN_CHARS:
            chain_str = chain_str[:_MAX_TOOL_CHAIN_CHARS] + "...(truncated)"
        parts.append(f"## 工具调用链\n{chain_str}")
    else:
        parts.append("## 工具调用链\n（无工具调用）")

    resp = (snapshot.final_response or "(无回复)")[:_MAX_RESPONSE_CHARS]
    parts.append(f"## Agent 最终回复\n{resp}")

    if test_case and test_case.expected_keywords:
        kws = ", ".join(test_case.expected_keywords[:10])
        parts.append(f"## 期望关键词（参考）\n{kws}")

    has_history = bool(session_history)
    parts.append(
        "## 评分要求\n"
        + ("请评估以上维度（含 multi_turn_coherence）并返回 JSON。"
           if has_history
           else "请评估 tool_rationality / task_completion / response_logic 并返回 JSON。")
    )

    return "\n\n".join(parts)


def _parse_scores(raw: str) -> dict[str, float]:
    """Extract scores from LLM JSON output and normalize 1-5 → 0.0-1.0."""
    # Extract JSON block (may be wrapped in markdown)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group())
        raw_scores: dict[str, Any] = data.get("scores", {})
        result: dict[str, float] = {}
        for dim, val in raw_scores.items():
            try:
                v = float(val)
                if 1.0 <= v <= 5.0:
                    result[dim] = round((v - 1) / 4, 4)  # normalize to 0-1
            except (TypeError, ValueError):
                pass
        return result
    except (json.JSONDecodeError, AttributeError):
        return {}
