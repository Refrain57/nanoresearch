"""Optimization Agent: generate and score system-prompt improvement candidates
for a given badcase category.

Flow:
  1. LLM generates 3-5 candidate prompt improvements given category + representative badcases.
  2. For each golden test case, find the most recent snapshot that has tool_recordings.
  3. Run each candidate prompt through SandboxedToolRegistry(replay) on those snapshots.
  4. Score via RuleEvaluator, rank candidates by mean score.
  5. Persist as OptimizationProposal.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMProvider
    from nanobot.storage.models import AgentRunSnapshot, AgentTestCase, OptimizationProposal
    from nanobot.storage.repositories.agent_eval_repo import AgentEvalRepository


@dataclass
class OptimizationCandidate:
    prompt: str
    rationale: str


_GENERATE_SYSTEM = """\
你是一位 AI Agent 系统 prompt 优化专家。
给定一类 badcase 的失败模式和代表性样本，生成 3-5 条候选系统 prompt 改进方案。
每条方案只需包含改进后的系统 prompt 文本及简要说明（rationale）。
以 JSON 数组输出，格式：[{"prompt": "...", "rationale": "..."}]
不要输出任何其他内容。
"""


class OptimizationAgent:
    def __init__(
        self,
        provider: "LLMProvider",
        repo: "AgentEvalRepository",
        registry: "ToolRegistry",
        model: str | None = None,
    ) -> None:
        self._provider = provider
        self._repo = repo
        self._registry = registry
        self._model = model or os.environ.get("EVAL_JUDGE_MODEL") or provider.get_default_model()

    async def generate_proposals(
        self,
        category: str,
        representative_snapshots: "list[AgentRunSnapshot]",
        golden_test_cases: "list[AgentTestCase]",
        created_by: str = "system",
    ) -> "OptimizationProposal":
        """Main entry: generate, score, rank, and persist optimization proposals."""
        # Step 1: generate candidate prompts from LLM
        candidates = await self._generate_candidates(category, representative_snapshots)
        if not candidates:
            logger.warning("OptimizationAgent: no candidates generated for category={}", category)
            return await self._repo.create_optimization_proposal(
                category=category, proposals=[], created_by=created_by
            )

        # Step 2: gather recordings for golden test cases
        recordings_map = await self._gather_recordings(golden_test_cases)
        if not recordings_map:
            logger.warning(
                "OptimizationAgent: no tool recordings found for any golden test case, "
                "cannot score candidates reliably"
            )
            proposals = [
                {"prompt": c.prompt, "rationale": c.rationale, "scores": {}, "rank": i + 1}
                for i, c in enumerate(candidates)
            ]
            return await self._repo.create_optimization_proposal(
                category=category, proposals=proposals, created_by=created_by
            )

        # Step 3: score each candidate
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            scores = await self._score_candidate(candidate, golden_test_cases, recordings_map)
            mean_score = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
            scored.append({
                "prompt": candidate.prompt,
                "rationale": candidate.rationale,
                "scores": scores,
                "mean_score": mean_score,
            })

        # Step 4: rank by mean score descending
        scored.sort(key=lambda x: x["mean_score"], reverse=True)
        for i, item in enumerate(scored):
            item["rank"] = i + 1

        return await self._repo.create_optimization_proposal(
            category=category, proposals=scored, created_by=created_by
        )

    async def _generate_candidates(
        self,
        category: str,
        snapshots: "list[AgentRunSnapshot]",
    ) -> list[OptimizationCandidate]:
        # Build a summary of badcase examples
        examples: list[str] = []
        for snap in snapshots[:5]:
            chain = snap.tool_call_chain or []
            recent = chain[-2:] if len(chain) > 2 else chain
            chain_str = json.dumps(recent, ensure_ascii=False, default=str)[:500]
            resp = (snap.final_response or "(无回复)")[:300]
            examples.append(
                f"- 用户输入: {(snap.user_input or '')[:200]}\n"
                f"  工具调用(最近): {chain_str}\n"
                f"  最终回复: {resp}"
            )

        prompt = (
            f"Badcase 类别：{category}\n\n"
            f"代表性 badcase 样本（共 {len(examples)} 条）：\n"
            + "\n\n".join(examples)
            + "\n\n请生成 3-5 条候选系统 prompt 改进方案（JSON 数组）。"
        )

        try:
            response = await self._provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                system=_GENERATE_SYSTEM,
                model=self._model,
                max_tokens=2048,
                temperature=0.3,
            )
            raw = (response.content or "").strip()
            # Extract JSON array
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not match:
                return []
            items: list[dict] = json.loads(match.group())
            return [
                OptimizationCandidate(
                    prompt=str(item.get("prompt", "")),
                    rationale=str(item.get("rationale", "")),
                )
                for item in items
                if item.get("prompt")
            ]
        except Exception as exc:
            logger.warning("OptimizationAgent._generate_candidates failed: {}", exc)
            return []

    async def _gather_recordings(
        self,
        golden_test_cases: "list[AgentTestCase]",
    ) -> dict[uuid.UUID, str]:
        """Return {test_case_id: recordings_json} for cases that have recordings."""
        result: dict[uuid.UUID, str] = {}
        for tc in golden_test_cases:
            snap = await self._repo.get_latest_snapshot_with_recordings(
                user_input=tc.user_input  # uid=None matches any user
            )
            if snap and snap.tool_recordings:
                result[tc.id] = json.dumps(snap.tool_recordings)
        return result

    async def _score_candidate(
        self,
        candidate: OptimizationCandidate,
        golden_test_cases: "list[AgentTestCase]",
        recordings_map: dict[uuid.UUID, str],
    ) -> dict[str, float]:
        """Score a candidate prompt against golden test cases using replay mode."""
        from nanobot.agent.runner import AgentRunner, AgentRunSpec
        from nanobot.eval.evaluator import RuleEvaluator
        from nanobot.eval.sandbox import SandboxedToolRegistry
        from nanobot.eval.snapshot import RunSnapshotCollector

        runner = AgentRunner(self._provider)
        evaluator = RuleEvaluator()
        all_scores: list[dict[str, float]] = []

        for tc in golden_test_cases:
            recordings_json = recordings_map.get(tc.id)
            if recordings_json is None:
                logger.warning(
                    "OptimizationAgent: no recordings for test case {}, skipping", tc.id
                )
                continue

            try:
                sandboxed = SandboxedToolRegistry.from_recordings_json(
                    self._registry,
                    recordings_json,
                )
                collector = RunSnapshotCollector()
                initial_messages = [
                    {"role": "system", "content": candidate.prompt},
                ]
                if tc.session_history:
                    initial_messages.extend(tc.session_history)
                initial_messages.append({"role": "user", "content": tc.user_input})

                spec = AgentRunSpec(
                    initial_messages=initial_messages,
                    tools=sandboxed,
                    model=self._model,
                    max_iterations=10,
                    concurrent_tools=False,
                    snapshot_collector=collector,
                )
                result = await runner.run(spec)
                status = (
                    "failed"
                    if result.stop_reason in ("error", "tool_error", "consecutive_failures")
                    else "success"
                )
                snapshot_data = collector.build(
                    run_id=str(uuid.uuid4()),
                    user_input=tc.user_input,
                    final_response=result.final_content,
                    status=status,
                )
                scores = evaluator.evaluate(snapshot_data, tc)
                all_scores.append(scores)
            except Exception as exc:
                logger.warning(
                    "OptimizationAgent._score_candidate: test case {} failed: {}", tc.id, exc
                )

        if not all_scores:
            return {}

        # Average across all successfully scored test cases
        all_dims = set(d for s in all_scores for d in s)
        avg: dict[str, float] = {}
        for dim in all_dims:
            vals = [s[dim] for s in all_scores if dim in s]
            if vals:
                avg[dim] = round(sum(vals) / len(vals), 4)
        return avg
