"""Evaluation dataset and run repository."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nanobot.storage.models import EvalDataset, EvalDatasetItem, EvalRun, EvalRunItem


class EvalRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    # ------------------------------------------------------------------
    # EvalDataset
    # ------------------------------------------------------------------

    async def list_datasets(self, kb_id: uuid.UUID) -> list[EvalDataset]:
        async with self._factory() as db:
            result = await db.execute(
                select(EvalDataset)
                .where(EvalDataset.kb_id == kb_id)
                .order_by(EvalDataset.created_at.desc())
            )
            return list(result.scalars().all())

    async def get_dataset(self, dataset_id: uuid.UUID) -> EvalDataset | None:
        async with self._factory() as db:
            result = await db.execute(select(EvalDataset).where(EvalDataset.id == dataset_id))
            return result.scalar_one_or_none()

    async def create_dataset(self, kb_id: uuid.UUID, name: str) -> EvalDataset:
        ds = EvalDataset(kb_id=kb_id, name=name)
        async with self._factory() as db:
            db.add(ds)
            await db.commit()
            await db.refresh(ds)
        return ds

    async def add_items(self, dataset_id: uuid.UUID, items: list[dict]) -> None:
        rows = [
            EvalDatasetItem(
                dataset_id=dataset_id,
                item_index=i,
                query=item.get("query", ""),
                gold_chunk_ids=item.get("gold_chunk_ids"),
                gold_answer=item.get("gold_answer"),
            )
            for i, item in enumerate(items)
        ]
        async with self._factory() as db:
            ds_result = await db.execute(select(EvalDataset).where(EvalDataset.id == dataset_id))
            ds = ds_result.scalar_one_or_none()
            if ds:
                ds.item_count = len(rows)
            db.add_all(rows)
            await db.commit()

    async def list_items(self, dataset_id: uuid.UUID) -> list[EvalDatasetItem]:
        async with self._factory() as db:
            result = await db.execute(
                select(EvalDatasetItem)
                .where(EvalDatasetItem.dataset_id == dataset_id)
                .order_by(EvalDatasetItem.item_index)
            )
            return list(result.scalars().all())

    async def delete_dataset(self, dataset_id: uuid.UUID) -> None:
        async with self._factory() as db:
            result = await db.execute(select(EvalDataset).where(EvalDataset.id == dataset_id))
            ds = result.scalar_one_or_none()
            if ds:
                await db.delete(ds)
                await db.commit()

    # ------------------------------------------------------------------
    # EvalRun
    # ------------------------------------------------------------------

    async def list_runs(self, kb_id: uuid.UUID) -> list[EvalRun]:
        async with self._factory() as db:
            result = await db.execute(
                select(EvalRun)
                .where(EvalRun.kb_id == kb_id)
                .order_by(EvalRun.created_at.desc())
            )
            return list(result.scalars().all())

    async def get_run(self, run_id: uuid.UUID) -> EvalRun | None:
        async with self._factory() as db:
            result = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
            return result.scalar_one_or_none()

    async def create_run(
        self,
        kb_id: uuid.UUID,
        dataset_id: uuid.UUID,
        name: str | None = None,
        retrieval_config: dict | None = None,
    ) -> EvalRun:
        run = EvalRun(
            kb_id=kb_id,
            dataset_id=dataset_id,
            name=name,
            retrieval_config=retrieval_config or {},
        )
        async with self._factory() as db:
            db.add(run)
            await db.commit()
            await db.refresh(run)
        return run

    async def update_run(self, run_id: uuid.UUID, **fields) -> None:
        async with self._factory() as db:
            result = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
            run = result.scalar_one_or_none()
            if run:
                for key, value in fields.items():
                    setattr(run, key, value)
                await db.commit()

    async def add_run_items(self, run_id: uuid.UUID, items: list[dict]) -> None:
        rows = [
            EvalRunItem(
                run_id=run_id,
                query=item.get("query"),
                gold_answer=item.get("gold_answer"),
                generated_answer=item.get("generated_answer"),
                retrieved_chunk_ids=item.get("retrieved_chunk_ids"),
                item_metrics=item.get("metrics", {}),
            )
            for item in items
        ]
        async with self._factory() as db:
            db.add_all(rows)
            await db.commit()

    async def list_run_items(self, run_id: uuid.UUID) -> list[EvalRunItem]:
        async with self._factory() as db:
            result = await db.execute(
                select(EvalRunItem).where(EvalRunItem.run_id == run_id)
            )
            return list(result.scalars().all())

    async def delete_run(self, run_id: uuid.UUID) -> None:
        async with self._factory() as db:
            result = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
            run = result.scalar_one_or_none()
            if run:
                await db.delete(run)
                await db.commit()
