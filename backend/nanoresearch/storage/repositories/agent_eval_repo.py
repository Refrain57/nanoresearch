"""Repository for agent evaluation snapshots, test cases, and badcase management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanoresearch.eval.badcase_classifier import ClassifyResult

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

import json

from nanoresearch.eval.snapshot import RunSnapshotData
from nanoresearch.storage.models import (
    AgentEvalRun,
    AgentRunSnapshot,
    AgentTestCase,
    JudgeCalibrationLog,
    OptimizationProposal,
    TunableObjectVersion,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _migrate_snapshot_scores(snap: AgentRunSnapshot) -> AgentRunSnapshot:
    """Rename legacy 'hallucination' key to 'faithfulness_score' in scores JSONB.

    Historical snapshots stored the Judge's hallucination dimension under the key
    'hallucination'. New snapshots use 'faithfulness_score'. This mapping is applied
    at read time so callers always see 'faithfulness_score'.
    """
    if snap.scores and isinstance(snap.scores, dict):
        if "hallucination" in snap.scores and "faithfulness_score" not in snap.scores:
            snap.scores = {
                **{k: v for k, v in snap.scores.items() if k != "hallucination"},
                "faithfulness_score": snap.scores["hallucination"],
            }
    return snap


class AgentEvalRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    async def save_snapshot(
        self,
        data: RunSnapshotData,
        uid: str,
        conversation_id: uuid.UUID | None = None,
        agent_id: uuid.UUID | None = None,
        system_prompt_version: str | None = None,
        eval_run_id: uuid.UUID | None = None,
        tool_recordings: str | None = None,
    ) -> uuid.UUID:
        snap = AgentRunSnapshot(
            run_id=data.run_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
            uid=uid,
            user_input=data.user_input,
            system_prompt_version=system_prompt_version,
            timestamp=_utcnow(),
            eval_run_id=eval_run_id,
            total_input_tokens=data.total_input_tokens,
            total_output_tokens=data.total_output_tokens,
            ttft_ms=data.ttft_ms,
            total_duration_ms=data.total_duration_ms,
            tool_call_count=data.tool_call_count,
            llm_call_count=data.llm_call_count,
            retry_count=data.retry_count,
            run_status=data.run_status,
            tool_call_chain=data.tool_call_chain,
            llm_calls=data.llm_calls,
            final_response=data.final_response,
            tool_recordings=json.loads(tool_recordings) if tool_recordings else None,
            context_trace=data.context_trace,
        )
        async with self._factory() as session:
            session.add(snap)
            await session.commit()
            return snap.id

    async def get_snapshot(self, snapshot_id: uuid.UUID) -> AgentRunSnapshot | None:
        async with self._factory() as session:
            snap = await session.get(AgentRunSnapshot, snapshot_id)
            return _migrate_snapshot_scores(snap) if snap else None

    async def list_snapshots(
        self,
        uid: str | None = None,
        is_badcase: bool | None = None,
        run_status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[AgentRunSnapshot], int]:
        from sqlalchemy import func
        async with self._factory() as session:
            base = select(AgentRunSnapshot)
            if uid is not None:
                base = base.where(AgentRunSnapshot.uid == uid)
            if is_badcase is not None:
                base = base.where(AgentRunSnapshot.is_badcase == is_badcase)
            if run_status is not None:
                base = base.where(AgentRunSnapshot.run_status == run_status)

            count_q = select(func.count()).select_from(base.subquery())
            total = (await session.execute(count_q)).scalar_one()

            rows_q = base.order_by(desc(AgentRunSnapshot.timestamp)).offset((page - 1) * page_size).limit(page_size)
            rows = (await session.execute(rows_q)).scalars().all()
            return [_migrate_snapshot_scores(r) for r in rows], total

    async def list_badcases(
        self,
        badcase_status: str | None = None,
        badcase_category: str | None = None,
        semantic_category: str | None = None,
        root_cause_auto: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[AgentRunSnapshot], int]:
        from sqlalchemy import func
        async with self._factory() as session:
            base = select(AgentRunSnapshot).where(AgentRunSnapshot.is_badcase == True)  # noqa: E712
            if badcase_status is not None:
                base = base.where(AgentRunSnapshot.badcase_status == badcase_status)
            if badcase_category is not None:
                base = base.where(AgentRunSnapshot.badcase_category == badcase_category)
            if semantic_category is not None:
                base = base.where(AgentRunSnapshot.semantic_category == semantic_category)
            if root_cause_auto is not None:
                base = base.where(AgentRunSnapshot.root_cause_auto == root_cause_auto)

            count_q = select(func.count()).select_from(base.subquery())
            total = (await session.execute(count_q)).scalar_one()

            rows_q = base.order_by(desc(AgentRunSnapshot.timestamp)).offset((page - 1) * page_size).limit(page_size)
            rows = (await session.execute(rows_q)).scalars().all()
            return [_migrate_snapshot_scores(r) for r in rows], total

    async def mark_badcase(
        self,
        snapshot_id: uuid.UUID,
        trigger: str,
        category: str,
    ) -> None:
        async with self._factory() as session:
            await session.execute(
                update(AgentRunSnapshot)
                .where(AgentRunSnapshot.id == snapshot_id)
                .values(
                    is_badcase=True,
                    badcase_trigger=trigger,
                    badcase_category=category,
                    badcase_status="pending",
                )
            )
            await session.commit()

    async def write_scores(
        self,
        snapshot_id: uuid.UUID,
        scores: dict[str, float],
        passed: bool | None,
        failed_dims: list[str],
        judge_metadata: dict | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "scores": scores,
            "passed": passed,
            "failed_dimensions": failed_dims,
        }
        if judge_metadata is not None:
            values["judge_metadata"] = judge_metadata
        async with self._factory() as session:
            await session.execute(
                update(AgentRunSnapshot)
                .where(AgentRunSnapshot.id == snapshot_id)
                .values(**values)
            )
            await session.commit()

    async def annotate_badcase(
        self,
        snapshot_id: uuid.UUID,
        root_cause: str | None = None,
        fix_notes: str | None = None,
        badcase_status: str | None = None,
        annotated_by: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"annotated_at": _utcnow()}
        if root_cause is not None:
            values["root_cause"] = root_cause
        if fix_notes is not None:
            values["fix_notes"] = fix_notes
        if badcase_status is not None:
            values["badcase_status"] = badcase_status
        if annotated_by is not None:
            values["annotated_by"] = annotated_by
        async with self._factory() as session:
            await session.execute(
                update(AgentRunSnapshot)
                .where(AgentRunSnapshot.id == snapshot_id)
                .values(**values)
            )
            await session.commit()

    # ------------------------------------------------------------------
    # Test Cases
    # ------------------------------------------------------------------

    async def create_test_case(
        self,
        dataset_type: str,
        name: str,
        user_input: str,
        description: str | None = None,
        session_history: list | None = None,
        expected_tools: list[str] | None = None,
        expected_intent: str | None = None,
        expected_keywords: list[str] | None = None,
        token_budget: int | None = None,
        human_score: float | None = None,
        status: str = "active",
        generated_from_snapshot_id: uuid.UUID | None = None,
        generation_reason: str | None = None,
        set_kind: str | None = None,
        tool_recordings: dict | None = None,
    ) -> AgentTestCase:
        tc = AgentTestCase(
            dataset_type=dataset_type,
            name=name,
            description=description,
            user_input=user_input,
            session_history=session_history or [],
            expected_tools=expected_tools,
            expected_intent=expected_intent,
            expected_keywords=expected_keywords,
            token_budget=token_budget,
            human_score=human_score,
            status=status,
            generated_from_snapshot_id=generated_from_snapshot_id,
            generation_reason=generation_reason,
            set_kind=set_kind,
            tool_recordings=tool_recordings,
        )
        async with self._factory() as session:
            session.add(tc)
            await session.commit()
            await session.refresh(tc)
            return tc

    async def list_test_cases(
        self,
        dataset_type: str | None = None,
        set_kind: str | None = None,
        active_only: bool = True,
    ) -> list[AgentTestCase]:
        async with self._factory() as session:
            q = select(AgentTestCase)
            if dataset_type is not None:
                q = q.where(AgentTestCase.dataset_type == dataset_type)
            if set_kind is not None:
                q = q.where(AgentTestCase.set_kind == set_kind)
            if active_only:
                q = q.where(AgentTestCase.is_active == True)  # noqa: E712
                q = q.where(AgentTestCase.status == "active")
            q = q.order_by(desc(AgentTestCase.created_at))
            return list((await session.execute(q)).scalars().all())

    async def delete_test_case(self, test_case_id: uuid.UUID) -> None:
        async with self._factory() as session:
            tc = await session.get(AgentTestCase, test_case_id)
            if tc:
                tc.is_active = False
                await session.commit()

    async def touch_test_case(self, test_case_id: uuid.UUID) -> None:
        async with self._factory() as session:
            await session.execute(
                update(AgentTestCase)
                .where(AgentTestCase.id == test_case_id)
                .values(last_reviewed_at=_utcnow())
            )
            await session.commit()

    async def list_pending_cases(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[AgentTestCase], int]:
        from sqlalchemy import func
        async with self._factory() as session:
            base = select(AgentTestCase).where(AgentTestCase.status == "pending_review")
            count_q = select(func.count()).select_from(base.subquery())
            total = (await session.execute(count_q)).scalar_one()
            rows_q = base.order_by(desc(AgentTestCase.created_at)).offset((page - 1) * page_size).limit(page_size)
            rows = (await session.execute(rows_q)).scalars().all()
            return list(rows), total

    async def approve_pending_case(self, test_case_id: uuid.UUID) -> None:
        async with self._factory() as session:
            await session.execute(
                update(AgentTestCase)
                .where(AgentTestCase.id == test_case_id)
                .values(status="active", is_active=True)
            )
            await session.commit()

    async def reject_pending_case(self, test_case_id: uuid.UUID) -> None:
        async with self._factory() as session:
            await session.execute(
                update(AgentTestCase)
                .where(AgentTestCase.id == test_case_id)
                .values(status="rejected", is_active=False)
            )
            await session.commit()

    # ------------------------------------------------------------------
    # Eval Runs (batch evaluation sessions)
    # ------------------------------------------------------------------

    async def create_eval_run(
        self,
        name: str,
        created_by: str,
        dataset_type: str | None = None,
        total_cases: int = 0,
        baseline_eval_run_id: uuid.UUID | None = None,
    ) -> AgentEvalRun:
        run = AgentEvalRun(
            name=name,
            dataset_type=dataset_type,
            status="pending",
            created_by=created_by,
            created_at=_utcnow(),
            total_cases=total_cases,
            baseline_eval_run_id=baseline_eval_run_id,
        )
        async with self._factory() as session:
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run

    async def update_eval_run(
        self,
        run_id: uuid.UUID,
        status: str | None = None,
        passed: int | None = None,
        failed: int | None = None,
        summary_scores: dict | None = None,
        error: str | None = None,
        completed_at: datetime | None = None,
        has_regression: bool | None = None,
        regression_diffs: dict | None = None,
        baseline_eval_run_id: uuid.UUID | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if status is not None:
            values["status"] = status
        if passed is not None:
            values["passed_cases"] = passed
        if failed is not None:
            values["failed_cases"] = failed
        if summary_scores is not None:
            values["summary_scores"] = summary_scores
        if error is not None:
            values["error"] = error
        if completed_at is not None:
            values["completed_at"] = completed_at
        if has_regression is not None:
            values["has_regression"] = has_regression
        if regression_diffs is not None:
            values["regression_diffs"] = regression_diffs
        if baseline_eval_run_id is not None:
            values["baseline_eval_run_id"] = baseline_eval_run_id
        if not values:
            return
        async with self._factory() as session:
            await session.execute(
                update(AgentEvalRun).where(AgentEvalRun.id == run_id).values(**values)
            )
            await session.commit()

    async def get_eval_run(self, run_id: uuid.UUID) -> AgentEvalRun | None:
        async with self._factory() as session:
            return await session.get(AgentEvalRun, run_id)

    async def list_eval_runs(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[AgentEvalRun], int]:
        from sqlalchemy import func
        async with self._factory() as session:
            base = select(AgentEvalRun)
            count_q = select(func.count()).select_from(base.subquery())
            total = (await session.execute(count_q)).scalar_one()
            rows_q = base.order_by(desc(AgentEvalRun.created_at)).offset((page - 1) * page_size).limit(page_size)
            rows = (await session.execute(rows_q)).scalars().all()
            return list(rows), total

    async def delete_eval_run(self, run_id: uuid.UUID) -> None:
        from sqlalchemy import delete as sql_delete
        async with self._factory() as session:
            await session.execute(
                sql_delete(AgentRunSnapshot).where(AgentRunSnapshot.eval_run_id == run_id)
            )
            run = await session.get(AgentEvalRun, run_id)
            if run:
                await session.delete(run)
            await session.commit()

    async def get_snapshots_for_eval_run(
        self,
        eval_run_id: uuid.UUID,
    ) -> list[AgentRunSnapshot]:
        async with self._factory() as session:
            q = (
                select(AgentRunSnapshot)
                .where(AgentRunSnapshot.eval_run_id == eval_run_id)
                .order_by(AgentRunSnapshot.timestamp)
            )
            rows = (await session.execute(q)).scalars().all()
            return [_migrate_snapshot_scores(r) for r in rows]

    # ------------------------------------------------------------------
    # Badcase semantic classification
    # ------------------------------------------------------------------

    async def update_snapshot_semantic_category(
        self,
        snapshot_id: uuid.UUID,
        category: str,
    ) -> None:
        async with self._factory() as session:
            await session.execute(
                update(AgentRunSnapshot)
                .where(AgentRunSnapshot.id == snapshot_id)
                .values(semantic_category=category)
            )
            await session.commit()

    async def update_snapshot_classification(
        self,
        snapshot_id: uuid.UUID,
        result: "ClassifyResult",
    ) -> None:
        async with self._factory() as session:
            await session.execute(
                update(AgentRunSnapshot)
                .where(AgentRunSnapshot.id == snapshot_id)
                .values(
                    # legacy columns — preserved for backward compat
                    semantic_category=result.semantic_category,
                    root_cause_auto=result.root_cause_auto,
                    root_cause_auto_confidence=result.confidence,
                    root_cause_auto_reason=result.reason,
                    # Phase 1: structured pointer (parallel write, old columns untouched)
                    classification_layer=result.layer,
                    classification_target_kind=result.target_kind,
                    classification_target_id=result.target_id,
                )
            )
            await session.commit()

    # ------------------------------------------------------------------
    # Phase 1: tunable object version registry
    # ------------------------------------------------------------------

    async def create_tunable_version(
        self,
        kind: str,
        target_id: str,
        content: str,
        created_by: str = "system",
    ) -> uuid.UUID:
        """Insert a new version (active=True) and deactivate all prior versions."""
        async with self._factory() as session:
            # deactivate existing active versions for this kind+target_id
            await session.execute(
                update(TunableObjectVersion)
                .where(
                    TunableObjectVersion.kind == kind,
                    TunableObjectVersion.target_id == target_id,
                    TunableObjectVersion.active == True,  # noqa: E712
                )
                .values(active=False)
            )
            new_version = TunableObjectVersion(
                kind=kind,
                target_id=target_id,
                content=content,
                active=True,
                created_by=created_by,
            )
            session.add(new_version)
            await session.commit()
            await session.refresh(new_version)
            return new_version.id

    async def get_current_tunable_version(
        self, kind: str, target_id: str
    ) -> TunableObjectVersion | None:
        """Return the currently active version, or None."""
        async with self._factory() as session:
            result = await session.execute(
                select(TunableObjectVersion)
                .where(
                    TunableObjectVersion.kind == kind,
                    TunableObjectVersion.target_id == target_id,
                    TunableObjectVersion.active == True,  # noqa: E712
                )
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_tunable_version_by_id(
        self, version_id: uuid.UUID
    ) -> TunableObjectVersion | None:
        """Return a specific version by ID (used for rollback)."""
        async with self._factory() as session:
            result = await session.execute(
                select(TunableObjectVersion).where(TunableObjectVersion.id == version_id)
            )
            return result.scalar_one_or_none()

    async def list_tunable_versions(
        self, kind: str, target_id: str, limit: int = 20
    ) -> list[TunableObjectVersion]:
        """Return version history for a kind+target_id, newest first."""
        async with self._factory() as session:
            result = await session.execute(
                select(TunableObjectVersion)
                .where(
                    TunableObjectVersion.kind == kind,
                    TunableObjectVersion.target_id == target_id,
                )
                .order_by(desc(TunableObjectVersion.created_at))
                .limit(limit)
            )
            return list(result.scalars().all())

    async def list_unclassified_badcases(self, limit: int = 50) -> list[AgentRunSnapshot]:
        from sqlalchemy import func
        async with self._factory() as session:
            q = (
                select(AgentRunSnapshot)
                .where(AgentRunSnapshot.is_badcase == True)  # noqa: E712
                .where(AgentRunSnapshot.semantic_category.is_(None))
                .order_by(desc(AgentRunSnapshot.timestamp))
                .limit(limit)
            )
            return list((await session.execute(q)).scalars().all())

    async def count_unclassified_badcases(self) -> tuple[int, int]:
        """Returns (unclassified_count, total_badcase_count)."""
        from sqlalchemy import func
        async with self._factory() as session:
            total_q = select(func.count()).where(AgentRunSnapshot.is_badcase == True)  # noqa: E712
            unclassified_q = (
                select(func.count())
                .where(AgentRunSnapshot.is_badcase == True)  # noqa: E712
                .where(AgentRunSnapshot.semantic_category.is_(None))
            )
            total = (await session.execute(select(func.count()).select_from(
                select(AgentRunSnapshot).where(AgentRunSnapshot.is_badcase == True).subquery()  # noqa: E712
            ))).scalar_one()
            unclassified = (await session.execute(select(func.count()).select_from(
                select(AgentRunSnapshot)
                .where(AgentRunSnapshot.is_badcase == True)  # noqa: E712
                .where(AgentRunSnapshot.semantic_category.is_(None))
                .subquery()
            ))).scalar_one()
            return unclassified, total

    async def get_latest_snapshot_with_recordings(
        self,
        user_input: str,
        uid: str | None = None,
    ) -> AgentRunSnapshot | None:
        async with self._factory() as session:
            q = (
                select(AgentRunSnapshot)
                .where(AgentRunSnapshot.user_input == user_input)
                .where(AgentRunSnapshot.tool_recordings.is_not(None))
            )
            if uid:
                q = q.where(AgentRunSnapshot.uid == uid)
            q = q.order_by(desc(AgentRunSnapshot.timestamp)).limit(1)
            return (await session.execute(q)).scalars().first()

    # ------------------------------------------------------------------
    # Judge Calibration Logs
    # ------------------------------------------------------------------

    async def create_calibration_log(
        self,
        judge_model: str,
        mad_value: float,
        passed: bool,
        sample_count: int,
        eval_run_id: uuid.UUID | None = None,
    ) -> JudgeCalibrationLog:
        log = JudgeCalibrationLog(
            judge_model=judge_model,
            mad_value=mad_value,
            passed=passed,
            sample_count=sample_count,
            checked_at=_utcnow(),
            eval_run_id=eval_run_id,
        )
        async with self._factory() as session:
            session.add(log)
            await session.commit()
            await session.refresh(log)
            return log

    async def list_calibration_logs(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[JudgeCalibrationLog], int]:
        from sqlalchemy import func
        async with self._factory() as session:
            base = select(JudgeCalibrationLog)
            count_q = select(func.count()).select_from(base.subquery())
            total = (await session.execute(count_q)).scalar_one()
            rows_q = (
                base.order_by(desc(JudgeCalibrationLog.checked_at))
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = (await session.execute(rows_q)).scalars().all()
            return list(rows), total

    # ------------------------------------------------------------------
    # Optimization Proposals
    # ------------------------------------------------------------------

    async def create_optimization_proposal(
        self,
        category: str,
        proposals: list[dict],
        created_by: str,
        baseline_score: dict | None = None,
        baseline_version_id: uuid.UUID | None = None,
        status: str = "pending",
    ) -> OptimizationProposal:
        prop = OptimizationProposal(
            category=category,
            proposals=proposals,
            status=status,
            created_at=_utcnow(),
            created_by=created_by,
            baseline_score=baseline_score,
            baseline_version_id=baseline_version_id,
        )
        async with self._factory() as session:
            session.add(prop)
            await session.commit()
            await session.refresh(prop)
            return prop

    async def list_optimization_proposals(
        self,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[OptimizationProposal], int]:
        from sqlalchemy import func
        async with self._factory() as session:
            base = select(OptimizationProposal)
            if status is not None:
                base = base.where(OptimizationProposal.status == status)
            count_q = select(func.count()).select_from(base.subquery())
            total = (await session.execute(count_q)).scalar_one()
            rows_q = (
                base.order_by(desc(OptimizationProposal.created_at))
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = (await session.execute(rows_q)).scalars().all()
            return list(rows), total

    async def update_optimization_proposal_status(
        self,
        proposal_id: uuid.UUID,
        status: str,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if status == "approved":
            values["approved_at"] = _utcnow()
        async with self._factory() as session:
            await session.execute(
                update(OptimizationProposal)
                .where(OptimizationProposal.id == proposal_id)
                .values(**values)
            )
            await session.commit()
