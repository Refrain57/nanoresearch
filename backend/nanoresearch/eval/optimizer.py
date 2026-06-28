"""Optimization Agent: generate and score TunableTextObject improvement candidates.

Flow:
  1. target.generate_candidates(badcases) — LLM proposes 3-5 candidate texts.
  2. Read & score BASELINE (current deployed version) on both sets (Phase 5):
     CRITICAL: baseline uses the EXACT SAME fix_test_cases and health_test_cases
     Python objects as candidate scoring — same list instances, same recordings.
     The only variable is the text content (baseline vs candidate).
  3. Score each candidate on TWO independent sets (Phase 2):
       fix_set  — the badcases that triggered this optimization run (runtime, dynamic).
       health_set — explicitly constructed cases (set_kind="health" in DB, ≥50 cases).
  4. Run each candidate through SandboxedToolRegistry(replay) on those test cases.
  5. Score via RuleEvaluator, rank by mean fix_set score.
  6. Phase 5 gate: candidate.fix_set_delta ≥ GATE_IMPROVE AND
     candidate.health_set_delta ≥ -GATE_TOLERATE → gate_status = "pending_approval",
     else "rejected_by_gate".  deltas are pre-computed and stored in JSONB for
     direct SQL query: proposals->'proposals'->0->>'fix_set_delta'.
  7. Persist as OptimizationProposal with baseline_score, baseline_version_id,
     and per-candidate gate_status / deltas.

Phase constraints:
  - Phase 1: Only PersonaObject (kind="system_prompt") can complete the full flow.
    ToolDescriptionObject scoring requires Phase 4 sandbox layering.
  - Phase 2: generate_proposals requires BOTH fix_cases and health_cases to be non-empty.
    Missing either set raises ValueError — the dual-set invariant must hold.
  - Phase 5: baseline is scored fresh every generate_proposals call using the same
    test case objects as candidates.  Historical baseline scores are never reused.
    Gate thresholds are hardcoded — no dynamic threshold (insufficient calibration data).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanoresearch.agent.tools.registry import ToolRegistry
    from nanoresearch.eval.tunable import OptimizationCandidate, TunableTextObject
    from nanoresearch.providers.base import LLMProvider
    from nanoresearch.storage.models import AgentRunSnapshot, AgentTestCase, OptimizationProposal
    from nanoresearch.storage.repositories.agent_eval_repo import AgentEvalRepository

# OptimizationCandidate is defined in tunable.py — import here for callers that
# previously imported it from optimizer.py.
from nanoresearch.eval.tunable import OptimizationCandidate  # noqa: F401  (re-export)

# ---------------------------------------------------------------------------
# Phase 5: deployment gate thresholds (hardcoded — no dynamic threshold yet;
# insufficient calibration data to tune.  Revisit after 2-3 months of real
# proposal data per SDD §4.4 Rationale.)
# ---------------------------------------------------------------------------
_GATE_IMPROVE = 0.05   # fix_set delta must be ≥ this (candidate beats baseline on fix set)
_GATE_TOLERATE = 0.02  # health_set delta must be ≥ -this (candidate doesn't wreck health set)


def _dict_mean(d: dict) -> float:
    """Mean of dict values. Returns 0.0 for empty dict (scores absent = neutral)."""
    return round(sum(d.values()) / len(d), 4) if d else 0.0


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
        import os
        self._model = model or os.environ.get("EVAL_JUDGE_MODEL") or provider.get_default_model()

    async def generate_proposals(
        self,
        target: "TunableTextObject",
        representative_snapshots: "list[AgentRunSnapshot]",
        fix_test_cases: "list[AgentTestCase]",
        health_test_cases: "list[AgentTestCase]",
        created_by: str = "system",
    ) -> "OptimizationProposal":
        """Generate, score on both sets, gate, rank, and persist.

        fix_test_cases  — test cases derived from the badcases triggering this run.
        health_test_cases — independently constructed health set (set_kind="health").

        Both sets must be non-empty; missing either raises ValueError (Phase 2 invariant).

        CRITICAL (Phase 5): baseline scoring and candidate scoring share the EXACT SAME
        fix_test_cases and health_test_cases Python objects.  The only variable is the
        text content (baseline vs candidate).  This ensures delta computations are
        meaningful — if the test sets differed, the gate would be comparing apples
        to oranges.
        """
        if not fix_test_cases:
            raise ValueError(
                "generate_proposals requires fix_test_cases — "
                "provide test cases derived from the triggering badcases"
            )
        if not health_test_cases:
            raise ValueError(
                "generate_proposals requires health_test_cases — "
                "construct the health set first (set_kind='health', ≥50 cases, see Phase 2 SDD)"
            )

        # ---- Phase 5: baseline anchor (before any candidate work) ----
        baseline_text = await target.read()
        baseline_version_id_raw = await target.get_current_version()
        baseline_version_id: uuid.UUID | None = (
            uuid.UUID(baseline_version_id_raw) if baseline_version_id_raw else None
        )

        # ---- generate candidates ----
        candidates = await target.generate_candidates(representative_snapshots)
        if not candidates:
            logger.warning(
                "OptimizationAgent: no candidates generated for kind={} target_id={}",
                target.kind, target.target_id,
            )
            return await self._repo.create_optimization_proposal(
                category=f"{target.kind}:{target.target_id}",
                proposals=[],
                created_by=created_by,
                baseline_score=None,
                baseline_version_id=baseline_version_id,
            )

        # ---- gather recordings ONCE for both baseline and candidates ----
        fix_recordings = await self._gather_recordings(fix_test_cases)
        health_recordings = await self._gather_recordings(health_test_cases)

        # ---- score baseline on EXACT SAME test case objects (Phase 5 invariant) ----
        baseline_candidate = OptimizationCandidate(
            prompt=baseline_text,
            rationale="baseline (current deployed version)",
        )
        baseline_fix = await self._score_candidate_set(
            target, baseline_candidate, fix_test_cases, fix_recordings
        )
        baseline_health = await self._score_candidate_set(
            target, baseline_candidate, health_test_cases, health_recordings
        )
        baseline_score = {"fix_set": baseline_fix, "health_set": baseline_health}

        baseline_fix_mean = _dict_mean(baseline_fix)
        baseline_health_mean = _dict_mean(baseline_health)

        # ---- score candidates ----
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                fix_scores = await self._score_candidate_set(
                    target, candidate, fix_test_cases, fix_recordings
                )
                health_scores = await self._score_candidate_set(
                    target, candidate, health_test_cases, health_recordings
                )
            except NotImplementedError as exc:
                logger.warning(
                    "OptimizationAgent: scoring not available for kind={}: {} — "
                    "persisting candidate without scores",
                    target.kind, exc,
                )
                fix_scores = {}
                health_scores = {}

            dual_scores = {"fix_set": fix_scores, "health_set": health_scores}
            fix_mean = (
                round(sum(fix_scores.values()) / len(fix_scores), 4)
                if fix_scores else 0.0
            )

            # ---- Phase 5: compute deltas & gate decision ----
            health_mean = _dict_mean(health_scores)

            fix_set_delta = round(fix_mean - baseline_fix_mean, 4)
            health_set_delta = round(health_mean - baseline_health_mean, 4)

            gate_passed = (
                fix_set_delta >= _GATE_IMPROVE
                and health_set_delta >= -_GATE_TOLERATE
            )
            gate_status = "pending_approval" if gate_passed else "rejected_by_gate"

            scored.append({
                "prompt": candidate.prompt,
                "rationale": candidate.rationale,
                "scores": dual_scores,
                "fix_mean_score": fix_mean,
                "fix_set_delta": fix_set_delta,
                "health_set_delta": health_set_delta,
                "gate_status": gate_status,
            })

        scored.sort(key=lambda x: x["fix_mean_score"], reverse=True)
        for i, item in enumerate(scored):
            item["rank"] = i + 1

        # ---- proposal-level status ----
        all_rejected = all(
            item["gate_status"] == "rejected_by_gate" for item in scored
        )
        proposal_status = "gate_all_rejected" if all_rejected else "pending"

        return await self._repo.create_optimization_proposal(
            category=f"{target.kind}:{target.target_id}",
            proposals=scored,
            created_by=created_by,
            baseline_score=baseline_score,
            baseline_version_id=baseline_version_id,
            status=proposal_status,
        )

    async def _gather_recordings(
        self,
        test_cases: "list[AgentTestCase]",
    ) -> dict[uuid.UUID, str]:
        """Return {test_case_id: recordings_json} for cases that have tool recordings.

        Priority:
          1. tc.tool_recordings — used by health-set cases (recordings stored directly
             on the test case row at construction time).
          2. Latest snapshot matching user_input — used by fix-set cases (recordings
             come from the original badcase run).
        """
        import json
        result: dict[uuid.UUID, str] = {}
        for tc in test_cases:
            if tc.tool_recordings:
                result[tc.id] = json.dumps(tc.tool_recordings)
                continue
            snap = await self._repo.get_latest_snapshot_with_recordings(
                user_input=tc.user_input
            )
            if snap and snap.tool_recordings:
                result[tc.id] = json.dumps(snap.tool_recordings)
        return result

    async def _score_candidate_set(
        self,
        target: "TunableTextObject",
        candidate: "OptimizationCandidate",
        test_cases: "list[AgentTestCase]",
        recordings_map: dict[uuid.UUID, str],
    ) -> dict[str, float]:
        """Score a candidate against one test set (fix or health).

        system_prompt assembly:
          - system_prompt (kind="tool_description"): built via ContextBuilder with agent
            persona/skills/kb_bindings from DB; workspace = tmp dir; knowledge_search = None
            (no user history injection — evaluation must be reproducible).
            Candidate text goes into description_overrides on the sandbox, not the system msg.
          - system_prompt (kind="system_prompt"): candidate.prompt used directly as the full
            system message (Phase 1 simplification — no ContextBuilder; PersonaObject
            evaluation environment does not match production context assembly).

        sandbox mode:
          - tool_description: side_effect_only — query tools passthrough on cache miss,
            side-effect tools intercepted and logged.
          - system_prompt: strict replay — SandboxReplayError on any cache miss.

        Cases without tool_recordings run the agent live (no sandbox) — suitable for
        health cases that are purely text-based (no expected_tools). Fix cases without
        recordings are skipped with a warning.
        """
        import json

        from nanoresearch.agent.runner import AgentRunner, AgentRunSpec
        from nanoresearch.eval.evaluator import RuleEvaluator
        from nanoresearch.eval.sandbox import SandboxedToolRegistry
        from nanoresearch.eval.snapshot import RunSnapshotCollector

        runner = AgentRunner(self._provider)
        evaluator = RuleEvaluator()
        all_scores: list[dict[str, float]] = []

        # --- tool_description: build system prompt once via ContextBuilder ---
        tool_desc_system_prompt: str | None = None
        if target.kind == "tool_description":
            tool_desc_system_prompt = await _build_tool_desc_system_prompt(
                target, self._registry.tool_names
            )

        for tc in test_cases:
            recordings_json = recordings_map.get(tc.id)
            use_sandbox = recordings_json is not None
            if not use_sandbox and tc.expected_tools:
                logger.warning(
                    "OptimizationAgent: test case {} has expected_tools but no recordings — skipping",
                    tc.id,
                )
                continue

            try:
                recorded = json.loads(recordings_json) if recordings_json else {}

                if target.kind == "tool_description":
                    tools: Any = SandboxedToolRegistry(
                        registry=self._registry,
                        mode="side_effect_only",
                        recorded=recorded,
                        description_overrides={target.target_id: candidate.prompt},
                    )
                    system_content: Any = tool_desc_system_prompt or ""
                else:
                    tools = (
                        SandboxedToolRegistry.from_recordings_json(self._registry, recordings_json)
                        if use_sandbox
                        else self._registry
                    )
                    system_content = candidate.prompt

                collector = RunSnapshotCollector()
                initial_messages = [{"role": "system", "content": system_content}]
                if tc.session_history:
                    initial_messages.extend(tc.session_history)
                initial_messages.append({"role": "user", "content": tc.user_input})

                spec = AgentRunSpec(
                    initial_messages=initial_messages,
                    tools=tools,
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
                scores = await evaluator.evaluate(snapshot_data, tc)
                all_scores.append(scores)
            except Exception as exc:
                logger.warning(
                    "OptimizationAgent._score_candidate_set: test case {} failed: {}", tc.id, exc
                )

        if not all_scores:
            return {}

        all_dims = set(d for s in all_scores for d in s)
        avg: dict[str, float] = {}
        for dim in all_dims:
            vals = [s[dim] for s in all_scores if dim in s]
            if vals:
                avg[dim] = round(sum(vals) / len(vals), 4)
        return avg


async def _build_tool_desc_system_prompt(target: Any, tool_names: list[str]) -> str:
    """Build system prompt for ToolDescriptionObject evaluation via ContextBuilder.

    Uses a tmp workspace (RAG tools don't touch user files) and knowledge_search=None
    (no user history injection — evaluation must be stable and reproducible).
    Agent persona, skills summary, and KB bindings are fetched from the agent config.

    Accesses target._agent_id and target._agent_repo via duck typing — these are
    ToolDescriptionObject-specific attributes not in the TunableTextObject interface.
    Patching in the caller per SDD §4.1: interface is frozen until Phase 6.
    """
    import tempfile
    from pathlib import Path

    from nanoresearch.agent.context import ContextBuilder

    agent_id: str | None = getattr(target, "_agent_id", None)
    agent_repo = getattr(target, "_agent_repo", None)

    persona: str | None = None
    kb_bindings: list[dict] = []

    if agent_id and agent_repo:
        try:
            import uuid as _uuid
            agent = await agent_repo.get_by_id(_uuid.UUID(agent_id))
            if agent:
                persona = agent.persona or None
            kbs = await agent_repo.list_bound_kbs(_uuid.UUID(agent_id))
            kb_bindings = [
                {"id": str(kb.id), "name": kb.name or "", "description": kb.description or ""}
                for kb in kbs
            ]
        except Exception as exc:
            logger.warning(
                "_build_tool_desc_system_prompt: failed to fetch agent config for {}: {}",
                agent_id, exc,
            )

    tmp_workspace = Path(tempfile.mkdtemp(prefix="nanoresearch_eval_"))
    ctx = ContextBuilder(workspace=tmp_workspace, knowledge_search=None)
    return ctx.build_system_prompt(
        custom_persona=persona,
        skill_names=None,   # None = include all available skills in summary
        tool_names=tool_names,
        agent_id=agent_id,
        kb_bindings=kb_bindings or None,
        topic=None,         # no history recall during evaluation
    )
