"""LLM-based judge for scoring agent run snapshots."""

from __future__ import annotations

import asyncio
import json
from loguru import logger
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
你是 AI Agent 质量评审员。按两步评分：
1. 为每个维度写出 1-2 条针对本题的评分标准（criteria）
2. 按标准逐维度打 1-5 分并给理由

维度：tool_rationality（工具调用合理性）、task_completion（任务完成度）、response_logic（回复逻辑清晰度）、faithfulness_score（忠实度：对比工具返回 ground truth 检查捏造）
有历史时加 multi_turn_coherence（检查与历史回复的事实矛盾）

仅输出如下 JSON：
{"dimensions":{"tool_rationality":{"criteria":["..."],"score":1,"reason":"..."},"task_completion":{"criteria":["..."],"score":1,"reason":"..."},"response_logic":{"criteria":["..."],"score":1,"reason":"..."},"faithfulness_score":{"criteria":["..."],"score":1,"reason":"..."}}}
"""

_MAX_TOOL_CHAIN_CHARS = 1500
_MAX_RESPONSE_CHARS = 1000


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
                messages=[{"role": "user", "content": _SYSTEM_PROMPT + "\n\n---\n\n" + prompt}],
                model=self._model,
                max_tokens=4096,
                temperature=0.0,
            )
            raw = response.content or ""
            logger.info(
                "LLMJudge.score(): finish_reason={}, content_len={}, model={}",
                response.finish_reason, len(raw), self._model,
            )
            return _parse_scores(raw), raw
        except Exception:
            import traceback
            logger.warning("LLMJudge.score(): provider call failed:\n{}", traceback.format_exc())
            return {}, ""

    async def score_with_consistency(
        self,
        snapshot: "RunSnapshotData",
        test_case: "AgentTestCase | None" = None,
        session_history: list[dict] | None = None,
        runs: int = 3,
    ) -> tuple[dict[str, float], str, bool]:
        """Run score() `runs` times concurrently, return (consensus_scores, raw_output, low_confidence).

        Consensus is the median per dimension (rounded to nearest 0.25 step after
        normalisation). low_confidence=True when any dimension's max–min spread
        across runs exceeds 1 raw score point (0.25 normalised).
        """
        tasks = [self.score(snapshot, test_case, session_history) for _ in range(runs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid: list[dict[str, float]] = [
            r[0] for r in results if isinstance(r, tuple) and r[0]
        ]
        if not valid:
            return {}, "", False

        all_dims = set(d for s in valid for d in s)
        consensus: dict[str, float] = {}
        low_confidence = False

        for dim in all_dims:
            vals = sorted(s[dim] for s in valid if dim in s)
            if not vals:
                continue
            mid = vals[len(vals) // 2]
            consensus[dim] = mid
            if max(vals) - min(vals) > 0.25:
                low_confidence = True

        raw_out = results[0][1] if isinstance(results[0], tuple) else ""
        return consensus, raw_out, low_confidence

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
        parts.append(
            f"## 历史对话\n{history_str}\n\n"
            "（评估 multi_turn_coherence 时，请重点检查 assistant 历史回复中的事实性陈述"
            "——如时间、数字、实体名称——是否与当前回复存在矛盾）"
        )

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
    if has_history:
        parts.append("（请评估含 multi_turn_coherence 在内的所有维度）")

    return "\n\n".join(parts)


def _parse_scores(raw: str) -> dict[str, float]:
    """Extract scores from LLM JSON output and normalize 1-5 → 0.0-1.0.

    Handles two formats:
    - G-Eval format: {"dimensions": {"dim": {"criteria": [...], "score": int, "reason": "..."}}}
    - Legacy format: {"scores": {"dim": int, ...}, "reasoning": "..."}
    """
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        return {}

    result: dict[str, float] = {}

    # G-Eval format
    if "dimensions" in data and isinstance(data["dimensions"], dict):
        for dim, dim_data in data["dimensions"].items():
            if not isinstance(dim_data, dict):
                continue
            try:
                v = float(dim_data.get("score", 0))
                if 1.0 <= v <= 5.0:
                    result[dim] = round((v - 1) / 4, 4)
            except (TypeError, ValueError):
                pass
        return result

    # Legacy format fallback
    raw_scores: dict[str, Any] = data.get("scores", {})
    for dim, val in raw_scores.items():
        try:
            v = float(val)
            if 1.0 <= v <= 5.0:
                result[dim] = round((v - 1) / 4, 4)
        except (TypeError, ValueError):
            pass
    return result
