"""Agent evaluation REST API — snapshots, badcases, test cases, eval runs, trends."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from nanobot.server.middleware.auth import get_current_user
from nanobot.storage.database import get_db
from nanobot.storage.repositories.agent_eval_repo import AgentEvalRepository

router = APIRouter(prefix="/api/eval/agent", tags=["Agent Eval"])


def _repo(db: AsyncSession = Depends(get_db)) -> AgentEvalRepository:
    from nanobot.storage.database import get_session_factory
    return AgentEvalRepository(get_session_factory())


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

@router.get("/snapshots")
async def list_snapshots(
    is_badcase: bool | None = Query(None),
    run_status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    rows, total = await repo.list_snapshots(
        uid=uid,
        is_badcase=is_badcase,
        run_status=run_status,
        page=page,
        page_size=page_size,
    )
    return {
        "items": [_snap_summary(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/snapshots/{snapshot_id}")
async def get_snapshot(
    snapshot_id: uuid.UUID,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    snap = await repo.get_snapshot(snapshot_id)
    if snap is None or snap.uid != uid:
        raise HTTPException(404, "Snapshot not found")
    return _snap_detail(snap)


# ---------------------------------------------------------------------------
# Badcases
# ---------------------------------------------------------------------------

@router.get("/badcases")
async def list_badcases(
    badcase_status: str | None = Query(None),
    badcase_category: str | None = Query(None),
    semantic_category: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    rows, total = await repo.list_badcases(
        badcase_status=badcase_status,
        badcase_category=badcase_category,
        semantic_category=semantic_category,
        page=page,
        page_size=page_size,
    )
    return {
        "items": [_snap_summary(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


class AnnotateRequest(BaseModel):
    root_cause: str | None = None
    fix_notes: str | None = None
    badcase_status: str | None = None


class FlagBadcaseRequest(BaseModel):
    trigger: str = "user:dislike"
    category: str = "user_feedback"


@router.post("/badcases/{snapshot_id}/flag")
async def flag_snapshot_as_badcase(
    snapshot_id: uuid.UUID,
    body: FlagBadcaseRequest,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Manually flag a snapshot as badcase (e.g. user dislike feedback).

    This is a lightweight endpoint — it marks the snapshot and returns
    immediately, no test-case creation or other side effects.
    """
    snap = await repo.get_snapshot(snapshot_id)
    if snap is None or snap.uid != uid:
        raise HTTPException(404, "Snapshot not found")
    await repo.mark_badcase(snapshot_id, trigger=body.trigger, category=body.category)
    return {"ok": True}


@router.patch("/badcases/{snapshot_id}")
async def annotate_badcase(
    snapshot_id: uuid.UUID,
    body: AnnotateRequest,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    snap = await repo.get_snapshot(snapshot_id)
    if snap is None or snap.uid != uid:
        raise HTTPException(404, "Snapshot not found")
    await repo.annotate_badcase(
        snapshot_id=snapshot_id,
        root_cause=body.root_cause,
        fix_notes=body.fix_notes,
        badcase_status=body.badcase_status,
        annotated_by=uid,
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

@router.get("/testcases")
async def list_testcases(
    dataset_type: str | None = Query(None),
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    rows = await repo.list_test_cases(dataset_type=dataset_type)
    return [_tc_dict(r) for r in rows]


class TestCaseCreate(BaseModel):
    dataset_type: str
    name: str
    user_input: str
    description: str | None = None
    expected_tools: list[str] | None = None
    expected_keywords: list[str] | None = None
    token_budget: int | None = None
    human_score: float | None = None


@router.post("/testcases")
async def create_testcase(
    body: TestCaseCreate,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    tc = await repo.create_test_case(
        dataset_type=body.dataset_type,
        name=body.name,
        user_input=body.user_input,
        description=body.description,
        expected_tools=body.expected_tools,
        expected_keywords=body.expected_keywords,
        token_budget=body.token_budget,
        human_score=body.human_score,
    )
    return _tc_dict(tc)


@router.delete("/testcases/{tc_id}")
async def delete_testcase(
    tc_id: uuid.UUID,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    await repo.delete_test_case(tc_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@router.get("/stats")
async def get_stats(
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Quick overview numbers for the dashboard header."""
    all_rows, total = await repo.list_snapshots(uid=uid, page=1, page_size=1)
    badcase_rows, badcase_total = await repo.list_badcases(page=1, page_size=1)
    pending_rows, pending_total = await repo.list_badcases(badcase_status="pending", page=1, page_size=1)
    return {
        "total_snapshots": total,
        "total_badcases": badcase_total,
        "pending_review": pending_total,
    }


# ---------------------------------------------------------------------------
# Eval Runs (batch evaluation)
# ---------------------------------------------------------------------------

class EvalRunRequest(BaseModel):
    name: str
    dataset_type: str | None = None
    test_case_ids: list[uuid.UUID] | None = None
    use_judge: bool = False
    judge_model: str | None = None
    sandbox_mode: str = "record"
    baseline_run_id: uuid.UUID | None = None


@router.post("/eval-runs")
async def trigger_eval_run(
    body: EvalRunRequest,
    request: Request,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Trigger a batch evaluation run against test cases (async background task)."""
    from nanobot.eval.test_runner import EvalRunConfig, TestRunner  # noqa: PLC0415

    state = request.app.state
    channel_loop = getattr(state, "channel_loop", None)
    if channel_loop is None:
        raise HTTPException(503, "Agent loop not available")

    # Count test cases
    test_cases = await repo.list_test_cases(dataset_type=body.dataset_type)
    if body.test_case_ids:
        total = len(body.test_case_ids)
    else:
        total = len(test_cases)

    if total == 0:
        raise HTTPException(400, "No test cases found for the given criteria")

    eval_run = await repo.create_eval_run(
        name=body.name,
        created_by=uid,
        dataset_type=body.dataset_type,
        total_cases=total,
        baseline_eval_run_id=body.baseline_run_id,
    )

    judge = None
    judge_model_resolved: str | None = None
    if body.use_judge:
        from nanobot.eval.judge import LLMJudge
        judge = LLMJudge(provider=channel_loop.provider, model=body.judge_model)
        judge_model_resolved = judge._model

    runner = TestRunner(
        provider=channel_loop.provider,
        tools=channel_loop.tools,
        repo=repo,
        model=getattr(channel_loop, "model", None),
        judge=judge,
    )

    config = EvalRunConfig(
        dataset_type=body.dataset_type,
        test_case_ids=body.test_case_ids,
        use_judge=body.use_judge,
        judge_model=judge_model_resolved,
        sandbox_mode=body.sandbox_mode,
        baseline_run_id=body.baseline_run_id,
    )

    asyncio.create_task(runner.run_all(config, eval_run.id, uid))

    return {"id": str(eval_run.id), "status": "pending", "total_cases": total}


@router.get("/eval-runs")
async def list_eval_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    rows, total = await repo.list_eval_runs(page=page, page_size=page_size)
    return {
        "items": [_eval_run_dict(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/eval-runs/comparison")
async def compare_eval_runs_inline(
    run_a: uuid.UUID = Query(...),
    run_b: uuid.UUID = Query(...),
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Side-by-side dimension comparison between two eval runs.
    Registered before /{run_id} so 'comparison' doesn't get parsed as a UUID.
    """
    from nanobot.eval.regression_detector import RegressionDetector
    a = await repo.get_eval_run(run_a)
    b = await repo.get_eval_run(run_b)
    if a is None or b is None:
        raise HTTPException(404, "One or both eval runs not found")
    detector = RegressionDetector()
    _, diffs = detector.compare(a.summary_scores or {}, b.summary_scores or {})
    return {
        "run_a": _eval_run_dict(a),
        "run_b": _eval_run_dict(b),
        "dimension_diffs": diffs,
    }


@router.get("/eval-runs/{run_id}")
async def get_eval_run(
    run_id: uuid.UUID,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    run = await repo.get_eval_run(run_id)
    if run is None:
        raise HTTPException(404, "Eval run not found")
    snapshots = await repo.get_snapshots_for_eval_run(run_id)
    data = _eval_run_dict(run)
    data["snapshots"] = [_snap_summary(s) for s in snapshots]
    return data


@router.delete("/eval-runs/{run_id}")
async def delete_eval_run(
    run_id: uuid.UUID,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    run = await repo.get_eval_run(run_id)
    if run is None:
        raise HTTPException(404, "Eval run not found")
    await repo.delete_eval_run(run_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------

@router.get("/trends")
async def get_trends(
    days: int = Query(30, ge=1, le=365),
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Aggregate per-dimension scores grouped by day (last N days)."""
    from sqlalchemy import func, cast, Date, text
    from nanobot.storage.models import AgentRunSnapshot
    from nanobot.storage.database import get_session_factory

    factory = get_session_factory()
    from sqlalchemy import select
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with factory() as session:
        # Pull scored snapshots for this user within the window
        q = (
            select(AgentRunSnapshot)
            .where(AgentRunSnapshot.uid == uid)
            .where(AgentRunSnapshot.scores.isnot(None))
            .where(AgentRunSnapshot.timestamp >= cutoff)
            .order_by(AgentRunSnapshot.timestamp)
        )
        rows = list((await session.execute(q)).scalars().all())

    # Group by date in Python (avoids complex JSONB SQL aggregation)
    from collections import defaultdict
    daily: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for snap in rows:
        day = snap.timestamp.strftime("%Y-%m-%d")
        for dim, val in (snap.scores or {}).items():
            try:
                daily[day][dim].append(float(val))
            except (TypeError, ValueError):
                pass

    result = []
    for day in sorted(daily.keys()):
        day_scores = {
            dim: round(sum(vals) / len(vals), 4)
            for dim, vals in daily[day].items()
        }
        day_count = max(len(v) for v in daily[day].values()) if daily[day] else 0
        result.append({"date": day, "scores": day_scores, "count": day_count})

    # Overall averages
    all_dims: dict[str, list[float]] = defaultdict(list)
    for snap in rows:
        for dim, val in (snap.scores or {}).items():
            try:
                all_dims[dim].append(float(val))
            except (TypeError, ValueError):
                pass
    overall = {
        dim: round(sum(vals) / len(vals), 4)
        for dim, vals in all_dims.items()
    }

    return {"daily": result, "overall": overall, "total_scored": len(rows)}


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def _snap_summary(r: Any) -> dict:
    return {
        "id": str(r.id),
        "run_id": r.run_id,
        "uid": r.uid,
        "user_input": (r.user_input or "")[:120],
        "run_status": r.run_status,
        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        "total_input_tokens": r.total_input_tokens,
        "total_output_tokens": r.total_output_tokens,
        "total_duration_ms": r.total_duration_ms,
        "tool_call_count": r.tool_call_count,
        "llm_call_count": r.llm_call_count,
        "retry_count": r.retry_count,
        "ttft_ms": r.ttft_ms,
        "is_badcase": r.is_badcase,
        "badcase_trigger": r.badcase_trigger,
        "badcase_category": r.badcase_category,
        "badcase_status": r.badcase_status,
        "semantic_category": r.semantic_category,
        "passed": r.passed,
        "scores": r.scores,
        "final_response": r.final_response,
        "failed_dimensions": r.failed_dimensions,
    }


def _snap_detail(r: Any) -> dict:
    base = _snap_summary(r)
    base.update({
        "user_input": r.user_input,
        "final_response": r.final_response,
        "tool_call_chain": r.tool_call_chain or [],
        "llm_calls": r.llm_calls or [],
        "failed_dimensions": r.failed_dimensions,
        "root_cause": r.root_cause,
        "fix_notes": r.fix_notes,
        "annotated_by": r.annotated_by,
        "annotated_at": r.annotated_at.isoformat() if r.annotated_at else None,
        "has_recordings": bool(r.tool_recordings),
        "judge_metadata": r.judge_metadata,
    })
    return base


def _eval_run_dict(r: Any) -> dict:
    pass_rate = None
    if r.total_cases > 0:
        pass_rate = round(r.passed_cases / r.total_cases, 4)
    return {
        "id": str(r.id),
        "name": r.name,
        "dataset_type": r.dataset_type,
        "status": r.status,
        "created_by": r.created_by,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "total_cases": r.total_cases,
        "passed_cases": r.passed_cases,
        "failed_cases": r.failed_cases,
        "pass_rate": pass_rate,
        "summary_scores": r.summary_scores,
        "error": r.error,
        "has_regression": r.has_regression,
        "regression_diffs": r.regression_diffs,
        "baseline_eval_run_id": str(r.baseline_eval_run_id) if r.baseline_eval_run_id else None,
    }


def _tc_dict(r: Any) -> dict:
    return {
        "id": str(r.id),
        "dataset_type": r.dataset_type,
        "name": r.name,
        "description": r.description,
        "user_input": r.user_input,
        "expected_tools": r.expected_tools,
        "expected_keywords": r.expected_keywords,
        "token_budget": r.token_budget,
        "human_score": r.human_score,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "last_reviewed_at": r.last_reviewed_at.isoformat() if r.last_reviewed_at else None,
        "is_active": r.is_active,
    }


def _calibration_dict(r: Any) -> dict:
    return {
        "id": str(r.id),
        "judge_model": r.judge_model,
        "mad_value": r.mad_value,
        "passed": r.passed,
        "sample_count": r.sample_count,
        "checked_at": r.checked_at.isoformat() if r.checked_at else None,
        "eval_run_id": str(r.eval_run_id) if r.eval_run_id else None,
    }


def _proposal_dict(r: Any) -> dict:
    return {
        "id": str(r.id),
        "category": r.category,
        "proposals": r.proposals or [],
        "status": r.status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "approved_at": r.approved_at.isoformat() if r.approved_at else None,
        "created_by": r.created_by,
    }


# ---------------------------------------------------------------------------
# Snapshot Replay
# ---------------------------------------------------------------------------

@router.post("/snapshots/{snapshot_id}/replay")
async def replay_snapshot(
    snapshot_id: uuid.UUID,
    request: Request,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Re-run a snapshot in replay mode using its stored tool recordings."""
    snap = await repo.get_snapshot(snapshot_id)
    if snap is None or snap.uid != uid:
        raise HTTPException(404, "Snapshot not found")
    if not snap.tool_recordings:
        raise HTTPException(400, "No tool recordings available for this snapshot")

    import json as _json
    from nanobot.agent.runner import AgentRunner, AgentRunSpec
    from nanobot.eval.sandbox import SandboxedToolRegistry
    from nanobot.eval.snapshot import RunSnapshotCollector

    state = request.app.state
    channel_loop = getattr(state, "channel_loop", None)
    if channel_loop is None:
        raise HTTPException(503, "Agent loop not available")

    sandboxed = SandboxedToolRegistry.from_recordings_json(
        channel_loop.tools,
        _json.dumps(snap.tool_recordings),
    )
    collector = RunSnapshotCollector()
    initial_messages: list[dict] = [{"role": "user", "content": snap.user_input}]

    from nanobot.agent.runner import AgentRunSpec
    spec = AgentRunSpec(
        initial_messages=initial_messages,
        tools=sandboxed,
        model=getattr(channel_loop, "model", None) or channel_loop.provider.get_default_model(),
        max_iterations=20,
        concurrent_tools=False,
        snapshot_collector=collector,
    )
    runner = AgentRunner(channel_loop.provider)
    result = await runner.run(spec)
    status = (
        "failed"
        if result.stop_reason in ("error", "tool_error", "consecutive_failures")
        else "success"
    )
    replay_data = collector.build(
        run_id=None,
        user_input=snap.user_input,
        final_response=result.final_content,
        status=status,
    )
    return {
        "original_snapshot_id": str(snapshot_id),
        "replay_run_id": replay_data.run_id,
        "final_response": replay_data.final_response,
        "tool_call_chain": replay_data.tool_call_chain,
        "total_input_tokens": replay_data.total_input_tokens,
        "total_output_tokens": replay_data.total_output_tokens,
        "run_status": replay_data.run_status,
    }


# ---------------------------------------------------------------------------
# Badcase: promote to test case + batch semantic classification
# ---------------------------------------------------------------------------

class PromoteRequest(BaseModel):
    dataset_type: str  # golden / regression / calibration / custom
    name: str
    expected_tools: list[str] | None = None
    expected_keywords: list[str] | None = None
    token_budget: int | None = None


@router.post("/badcases/{snapshot_id}/promote")
async def promote_badcase(
    snapshot_id: uuid.UUID,
    body: PromoteRequest,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Promote a badcase snapshot into a test case (golden or regression set)."""
    snap = await repo.get_snapshot(snapshot_id)
    if snap is None or snap.uid != uid:
        raise HTTPException(404, "Snapshot not found")
    if not snap.is_badcase:
        raise HTTPException(400, "This snapshot is not marked as a badcase")

    tc = await repo.create_test_case(
        dataset_type=body.dataset_type,
        name=body.name,
        user_input=snap.user_input,
        expected_tools=body.expected_tools,
        expected_keywords=body.expected_keywords,
        token_budget=body.token_budget,
    )
    return _tc_dict(tc)


# Classify-batch task state (in-process; single-worker deployment)
_classify_tasks: dict[str, dict] = {}


@router.post("/badcases/classify-batch")
async def classify_badcases_batch(
    request: Request,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Trigger async LLM semantic classification for unclassified badcases."""
    from nanobot.eval.badcase_classifier import BadcaseClassifier
    import time

    state = request.app.state
    channel_loop = getattr(state, "channel_loop", None)
    if channel_loop is None:
        raise HTTPException(503, "Agent loop not available")

    unclassified = await repo.list_unclassified_badcases(limit=50)
    if not unclassified:
        return {"queued": 0, "task_id": None, "message": "No unclassified badcases"}

    task_id = str(uuid.uuid4())
    _classify_tasks[task_id] = {"status": "running", "total": len(unclassified), "processed": 0}

    async def _run() -> None:
        classifier = BadcaseClassifier(provider=channel_loop.provider)
        processed = 0
        for snap in unclassified:
            try:
                category = await classifier.classify(snap)
                await repo.update_snapshot_semantic_category(snap.id, category)
                processed += 1
                _classify_tasks[task_id]["processed"] = processed
            except Exception as exc:
                from loguru import logger
                logger.warning("classify_batch: snap {} failed: {}", snap.id, exc)
        _classify_tasks[task_id]["status"] = "done"

    asyncio.create_task(_run())
    return {"queued": len(unclassified), "task_id": task_id}


@router.get("/badcases/classify-batch/status")
async def classify_batch_status(
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Return current unclassified badcase count for frontend polling."""
    unclassified, total = await repo.count_unclassified_badcases()
    return {"unclassified": unclassified, "total_badcases": total}


@router.get("/eval-runs/{run_id}/regression")
async def get_eval_run_regression(
    run_id: uuid.UUID,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Return regression analysis for a specific eval run."""
    run = await repo.get_eval_run(run_id)
    if run is None:
        raise HTTPException(404, "Eval run not found")
    if not run.baseline_eval_run_id:
        raise HTTPException(400, "No baseline set for this eval run")
    return {
        "run_id": str(run_id),
        "baseline_eval_run_id": str(run.baseline_eval_run_id),
        "has_regression": run.has_regression,
        "diffs": run.regression_diffs or {},
    }


# ---------------------------------------------------------------------------
# Judge Calibration
# ---------------------------------------------------------------------------

class CalibrationRequest(BaseModel):
    judge_model: str | None = None
    eval_run_id: uuid.UUID | None = None


@router.post("/judge-calibration")
async def trigger_calibration(
    body: CalibrationRequest,
    request: Request,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Run judge calibration against test cases with human_score."""
    from nanobot.eval.judge import LLMJudge
    from nanobot.eval.snapshot import RunSnapshotData

    state = request.app.state
    channel_loop = getattr(state, "channel_loop", None)
    if channel_loop is None:
        raise HTTPException(503, "Agent loop not available")

    all_cases = await repo.list_test_cases()
    calibration_cases = [tc for tc in all_cases if tc.human_score is not None]
    if not calibration_cases:
        raise HTTPException(400, "No test cases with human_score available for calibration")

    # Build calibration samples from stored snapshots
    samples = []
    for tc in calibration_cases:
        existing = await repo.get_latest_snapshot_with_recordings(user_input=tc.user_input)
        if existing:
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
            samples.append((stub, tc))

    if not samples:
        raise HTTPException(400, "No stored snapshots found for calibration test cases")

    judge = LLMJudge(provider=channel_loop.provider, model=body.judge_model)
    cal_result = await judge.calibrate(samples)
    log = await repo.create_calibration_log(
        judge_model=cal_result.judge_model,
        mad_value=cal_result.mad,
        passed=cal_result.passed,
        sample_count=cal_result.sample_count,
        eval_run_id=body.eval_run_id,
    )
    return _calibration_dict(log)


@router.get("/judge-calibration")
async def list_calibrations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    rows, total = await repo.list_calibration_logs(page=page, page_size=page_size)
    return {
        "items": [_calibration_dict(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# Optimization Proposals
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    category: str
    snapshot_ids: list[uuid.UUID]


@router.post("/optimize")
async def trigger_optimization(
    body: OptimizeRequest,
    request: Request,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    """Trigger the optimization agent for a badcase category (async)."""
    from nanobot.eval.optimizer import OptimizationAgent

    state = request.app.state
    channel_loop = getattr(state, "channel_loop", None)
    if channel_loop is None:
        raise HTTPException(503, "Agent loop not available")

    snapshots = [await repo.get_snapshot(sid) for sid in body.snapshot_ids]
    snapshots = [s for s in snapshots if s is not None and s.uid == uid]
    if not snapshots:
        raise HTTPException(400, "No valid snapshots provided")

    golden_cases = await repo.list_test_cases(dataset_type="golden")

    async def _run() -> None:
        optimizer = OptimizationAgent(
            provider=channel_loop.provider,
            repo=repo,
            registry=channel_loop.tools,
        )
        await optimizer.generate_proposals(
            category=body.category,
            representative_snapshots=snapshots,
            golden_test_cases=golden_cases,
            created_by=uid,
        )

    asyncio.create_task(_run())
    return {"status": "optimization_started", "category": body.category}


@router.get("/optimize")
async def list_optimizations(
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    rows, total = await repo.list_optimization_proposals(status=status, page=page, page_size=page_size)
    return {
        "items": [_proposal_dict(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


class ProposalStatusUpdate(BaseModel):
    status: str  # approved | rejected


@router.patch("/optimize/{proposal_id}")
async def update_optimization_status(
    proposal_id: uuid.UUID,
    body: ProposalStatusUpdate,
    uid: str = Depends(get_current_user),
    repo: AgentEvalRepository = Depends(_repo),
):
    if body.status not in ("approved", "rejected"):
        raise HTTPException(400, "status must be 'approved' or 'rejected'")
    await repo.update_optimization_proposal_status(proposal_id, body.status)
    return {"ok": True}
