"""Batch test runner: executes test cases through the agent and scores results."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanoresearch.eval.judge import LLMJudge
    from nanoresearch.providers.base import LLMProvider
    from nanoresearch.storage.repositories.agent_eval_repo import AgentEvalRepository


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class EvalRunConfig:
    dataset_type: str | None = None
    test_case_ids: list[uuid.UUID] | None = None
    use_judge: bool = False
    judge_model: str = "claude-sonnet-4-6"
    judge_consistency_runs: int = 3    # set to 1 to skip consistency voting
    sandbox_mode: str = "record"  # passthrough / record / replay
    max_iterations: int = 20           # lower than production to save tokens
    baseline_run_id: uuid.UUID | None = None
    concurrency: int = 5               # concurrent test cases
    enable_flywheel: bool = False
    flywheel_thresholds: dict = field(default_factory=lambda: {
        "retrieval_failure": 0.20,
        "hallucination": 0.15,
        "reasoning_failure": 0.25,
        "tool_skip": 0.30,
    })
    flywheel_adversarial_per_run: int = 0  # 0 = disabled; N = generate N adversarial cases per run


@dataclass
class EvalRunSummary:
    eval_run_id: uuid.UUID
    total: int
    passed: int
    failed: int
    avg_scores: dict[str, float]
    snapshot_ids: list[uuid.UUID] = field(default_factory=list)


class TestRunner:
    """Run test cases through the agent runner and score the results.

    Uses the same provider and tools as the live agent (injected from
    channel_loop.provider and channel_loop.tools) so results reflect
    production behaviour.
    """

    def __init__(
        self,
        provider: "LLMProvider",
        tools: Any,          # ToolRegistry (or SandboxedToolRegistry)
        repo: "AgentEvalRepository",
        model: str | None = None,
        judge: "LLMJudge | None" = None,
        embedding_fn: Any | None = None,  # Callable[[list[str]], list[list[float]]]
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._repo = repo
        self._model = model or provider.get_default_model()
        self._judge = judge
        self._embedding_fn = embedding_fn

    async def run_all(
        self,
        config: EvalRunConfig,
        eval_run_id: uuid.UUID,
        uid: str,
    ) -> EvalRunSummary:
        """Execute all test cases in config and return aggregated summary."""
        from nanoresearch.agent.runner import AgentRunner, AgentRunSpec
        from nanoresearch.eval.badcase_detector import BadcaseDetector
        from nanoresearch.eval.evaluator import RuleEvaluator
        from nanoresearch.eval.sandbox import SandboxedToolRegistry
        from nanoresearch.eval.snapshot import RunSnapshotCollector

        # Mark run as started
        await self._repo.update_eval_run(
            eval_run_id,
            status="running",
        )

        # Fetch test cases
        if config.test_case_ids:
            all_rows = await self._repo.list_test_cases()
            id_set = set(config.test_case_ids)
            test_cases = [r for r in all_rows if r.id in id_set]
        else:
            test_cases = await self._repo.list_test_cases(dataset_type=config.dataset_type)

        if not test_cases:
            await self._repo.update_eval_run(
                eval_run_id,
                status="completed",
                passed=0,
                failed=0,
                summary_scores={},
                completed_at=_utcnow(),
            )
            return EvalRunSummary(eval_run_id=eval_run_id, total=0, passed=0, failed=0, avg_scores={})

        evaluator = RuleEvaluator(embedding_fn=self._embedding_fn)
        detector = BadcaseDetector(p95_tokens=None)

        # Auto-calibrate judge before scoring if there are calibration samples
        if config.use_judge and self._judge is not None:
            calibration_cases = [tc for tc in test_cases if tc.human_score is not None]
            if calibration_cases:
                # Build stub snapshots for calibration — use latest saved snapshot for each
                cal_samples = []
                for tc in calibration_cases:
                    existing = await self._repo.get_latest_snapshot_with_recordings(
                        user_input=tc.user_input
                    )
                    if existing:
                        from nanoresearch.eval.snapshot import RunSnapshotData
                        stub = RunSnapshotData(
                            run_id=str(existing.id),
                            user_input=existing.user_input,
                            tool_call_chain=existing.tool_call_chain or [],
                            llm_calls=existing.llm_calls or [],
                            final_response=existing.final_response,
                            run_status=existing.run_status,
                            total_input_tokens=existing.total_input_tokens,
                            total_output_tokens=existing.total_output_tokens,
                            ttft_ms=existing.ttft_ms,
                            total_duration_ms=existing.total_duration_ms or 0.0,
                            tool_call_count=existing.tool_call_count,
                            llm_call_count=existing.llm_call_count,
                            retry_count=existing.retry_count,
                        )
                        cal_samples.append((stub, tc))
                if cal_samples:
                    cal_result = await self._judge.calibrate(cal_samples)
                    try:
                        await self._repo.create_calibration_log(
                            judge_model=cal_result.judge_model,
                            mad_value=cal_result.mad,
                            passed=cal_result.passed,
                            sample_count=cal_result.sample_count,
                            eval_run_id=eval_run_id,
                        )
                    except Exception as _exc:
                        logger.warning("TestRunner: calibration log save failed: {}", _exc)
                    if not cal_result.passed:
                        logger.warning(
                            "TestRunner: Judge calibration failed (MAD={:.3f} > 0.15) — "
                            "judge scores may be unreliable for run {}",
                            cal_result.mad, eval_run_id,
                        )

        semaphore = asyncio.Semaphore(config.concurrency)
        lock = asyncio.Lock()
        snapshot_ids: list[uuid.UUID] = []
        all_scores: list[dict[str, float]] = []
        passed_count = 0
        failed_count = 0
        completed_count = 0
        total = len(test_cases)

        async def run_one(idx: int, tc: Any) -> None:
            nonlocal passed_count, failed_count, completed_count
            from nanoresearch.agent.runner import AgentRunner, AgentRunSpec
            from nanoresearch.agent.hook import AgentHook, AgentHookContext

            async with semaphore:
                logger.info("TestRunner: [{}/{}] running test case {} — {!r}", idx, total, tc.id, (tc.user_input or "")[:60])
                runner = AgentRunner(self._provider)
                try:
                    if config.sandbox_mode == "passthrough":
                        tools = self._tools
                    else:
                        tools = SandboxedToolRegistry(self._tools, mode=config.sandbox_mode)

                    collector = RunSnapshotCollector()
                    initial_messages = []
                    if tc.session_history:
                        initial_messages.extend(tc.session_history)
                    initial_messages.append({"role": "user", "content": tc.user_input})

                    class _TtftHook(AgentHook):
                        def wants_streaming(self) -> bool:
                            return True

                        async def on_stream(self, context: AgentHookContext, delta: str) -> None:
                            collector.on_first_token()

                    spec = AgentRunSpec(
                        initial_messages=initial_messages,
                        tools=tools,
                        model=self._model,
                        max_iterations=config.max_iterations,
                        concurrent_tools=True,
                        snapshot_collector=collector,
                        hook=_TtftHook(),
                    )

                    result = await runner.run(spec)

                    if result.stop_reason in ("error", "tool_error", "consecutive_failures"):
                        status = "failed"
                    elif result.stop_reason == "max_iterations":
                        status = "max_iterations"
                    else:
                        status = "success"

                    snapshot_data = collector.build(
                        run_id=str(uuid.uuid4()),
                        user_input=tc.user_input,
                        final_response=result.final_content,
                        status=status,
                    )

                    if tc.dataset_type == "calibration":
                        # Calibration cases are for judge training only — skip rule eval.
                        # passed=None excludes them from pass/fail statistics.
                        rule_scores: dict[str, float] = {}
                        passed: bool | None = None
                        failed_dims: list[str] = []
                    else:
                        rule_scores = await evaluator.evaluate(snapshot_data, tc)
                        passed, failed_dims = evaluator.is_passed(rule_scores, tc)

                    judge_scores: dict[str, float] = {}
                    judge_raw_output: str | None = None
                    judge_low_confidence: bool = False
                    if config.use_judge and self._judge is not None:
                        if config.judge_consistency_runs > 1:
                            judge_scores, judge_raw_output, judge_low_confidence = (
                                await self._judge.score_with_consistency(
                                    snapshot_data, tc,
                                    session_history=tc.session_history or None,
                                    runs=config.judge_consistency_runs,
                                )
                            )
                        else:
                            judge_scores, judge_raw_output = await self._judge.score(
                                snapshot_data, tc,
                                session_history=tc.session_history or None,
                            )

                    combined_scores = {**rule_scores, **judge_scores}

                    detections = detector.detect(snapshot_data, combined_scores, passed=passed, tc=tc)

                    tool_recordings_json: str | None = None
                    if config.sandbox_mode == "record" and hasattr(tools, "export_recordings"):
                        tool_recordings_json = tools.export_recordings()

                    snap_id = await self._repo.save_snapshot(
                        snapshot_data, uid=uid, eval_run_id=eval_run_id,
                        system_prompt_version="production",
                        tool_recordings=tool_recordings_json,
                    )
                    judge_metadata: dict[str, Any] | None = None
                    if judge_raw_output is not None:
                        judge_metadata = {"judge_model": config.judge_model, "raw_output": judge_raw_output}
                        if judge_low_confidence:
                            judge_metadata["low_confidence"] = True
                    await self._repo.write_scores(
                        snap_id, combined_scores, passed, failed_dims,
                        judge_metadata=judge_metadata,
                    )
                    # mark_badcase uses the primary detection only (quality > tool_skip).
                    # tool_skip is persisted separately via failed_dimensions JSONB.
                    if detections:
                        trigger, category = detections[0]
                        await self._repo.mark_badcase(snap_id, trigger, category)

                    await self._repo.touch_test_case(tc.id)

                    # ---- [TEMPORARY] Write to production conversation tables ----
                    try:
                        from nanoresearch.storage.database import get_session_factory
                        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
                        from nanoresearch.storage.repositories.run_repo import RunRepository

                        _sf = get_session_factory()
                        conv_repo = ConversationRepository(_sf)
                        run_repo = RunRepository(_sf)
                        session_key = f"eval:{tc.id}"
                        conv = await conv_repo.create(
                            key=session_key, uid=uid,
                            title=f"[Eval] {tc.name}",
                        )
                        msgs = [{"role": "user", "content": tc.user_input}]
                        if result.final_content:
                            msgs.append({"role": "assistant", "content": result.final_content})
                        await conv_repo.replace_messages(conv.id, msgs)
                        ar = await run_repo.create(conversation_id=conv.id, uid=uid)
                        await run_repo.update(
                            ar.id, status=status,
                            tool_calls=snapshot_data.tool_call_chain or [],
                            tokens_used={
                                "input": snapshot_data.total_input_tokens,
                                "output": snapshot_data.total_output_tokens,
                            },
                            duration_ms=int(snapshot_data.total_duration_ms) if snapshot_data.total_duration_ms else None,
                            finished_at=_utcnow(),
                        )
                    except Exception:
                        pass
                    # ---- END TEMPORARY ----

                    async with lock:
                        snapshot_ids.append(snap_id)
                        all_scores.append(combined_scores)
                        if passed is True:
                            passed_count += 1
                        elif passed is False:
                            failed_count += 1
                        # passed=None (calibration) is excluded from both counts.
                        completed_count += 1
                        if completed_count % 5 == 0 or completed_count == total:
                            await self._repo.update_eval_run(
                                eval_run_id,
                                passed=passed_count,
                                failed=failed_count,
                            )

                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    logger.warning(
                        "TestRunner: test case {} failed with exception:\n{}",
                        tc.id, tb,
                    )
                    async with lock:
                        failed_count += 1
                        completed_count += 1
                    try:
                        error_response = f"[EXCEPTION] {type(exc).__name__}: {exc}"
                        snapshot_data = collector.build(
                            run_id=str(uuid.uuid4()),
                            user_input=tc.user_input,
                            final_response=error_response,
                            status="failed",
                        )
                        snap_id = await self._repo.save_snapshot(
                            snapshot_data, uid=uid, eval_run_id=eval_run_id,
                            system_prompt_version="production",
                        )
                        await self._repo.write_scores(snap_id, {}, False, ["exception"])
                        async with lock:
                            snapshot_ids.append(snap_id)
                    except Exception:
                        pass
                    try:
                        error_msg = f"[{tc.name or str(tc.id)}] {type(exc).__name__}: {exc}\n{tb}"
                        await self._repo.update_eval_run(eval_run_id, error=error_msg[:2000])
                    except Exception:
                        pass

        tasks = [run_one(i, tc) for i, tc in enumerate(test_cases, 1)]
        await asyncio.gather(*tasks)

        # Aggregate average scores per dimension
        avg_scores: dict[str, float] = {}
        if all_scores:
            all_dims = set(d for s in all_scores for d in s)
            for dim in all_dims:
                vals = [s[dim] for s in all_scores if dim in s]
                if vals:
                    avg_scores[dim] = round(sum(vals) / len(vals), 4)

        await self._repo.update_eval_run(
            eval_run_id,
            status="completed",
            passed=passed_count,
            failed=failed_count,
            summary_scores=avg_scores,
            completed_at=_utcnow(),
        )

        # Regression detection against baseline
        if config.baseline_run_id is not None and avg_scores:
            try:
                from nanoresearch.eval.regression_detector import RegressionDetector
                baseline = await self._repo.get_eval_run(config.baseline_run_id)
                if baseline and baseline.summary_scores:
                    detector_reg = RegressionDetector()
                    has_regression, diffs = detector_reg.compare(baseline.summary_scores, avg_scores)
                    await self._repo.update_eval_run(
                        eval_run_id,
                        has_regression=has_regression,
                        regression_diffs=diffs,
                        baseline_eval_run_id=config.baseline_run_id,
                    )
                    if has_regression:
                        regressed_dims = [d for d, v in diffs.items() if v["regressed"]]
                        logger.warning(
                            "TestRunner: regression detected in run {} vs baseline {} — "
                            "dimensions: {}",
                            eval_run_id, config.baseline_run_id, regressed_dims,
                        )
            except Exception as _exc:
                logger.warning("TestRunner: regression comparison failed: {}", _exc)

        # Data flywheel: generate pending cases from badcase patterns
        if config.enable_flywheel:
            try:
                from nanoresearch.eval.data_flywheel import DataFlywheel, run_flywheel
                flywheel = DataFlywheel(self._provider, model=config.judge_model)
                await run_flywheel(flywheel, self._repo, eval_run_id, uid, config)
            except Exception as _exc:
                logger.warning("TestRunner: flywheel post-processing failed: {}", _exc)

        return EvalRunSummary(
            eval_run_id=eval_run_id,
            total=len(test_cases),
            passed=passed_count,
            failed=failed_count,
            avg_scores=avg_scores,
            snapshot_ids=snapshot_ids,
        )
