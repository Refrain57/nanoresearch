"""Evaluation dataset and run endpoints."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

from nanobot.server.middleware.auth import get_current_user
from nanobot.storage.repositories.eval_repo import EvalRepository
from nanobot.storage.repositories.knowledge_repo import KnowledgeRepository

router = APIRouter()


def _eval_repo(request: Request) -> EvalRepository:
    return EvalRepository(request.app.state.session_factory)


def _kb_repo(request: Request) -> KnowledgeRepository:
    return KnowledgeRepository(request.app.state.session_factory)


def _rag_settings(request: Request):
    settings = getattr(request.app.state, "rag_settings", None)
    if settings is None:
        from nanobot.rag.core.settings import load_settings
        settings = load_settings()
        request.app.state.rag_settings = settings
    return settings


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EvalRunCreate(BaseModel):
    dataset_id: str
    name: str | None = None
    top_k: int = 5


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@router.get("/api/eval/{kb_id}/datasets")
async def list_datasets(kb_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    datasets = await _eval_repo(request).list_datasets(uuid.UUID(kb_id))
    return [_dataset_to_dict(d) for d in datasets]


@router.post("/api/eval/{kb_id}/datasets/upload", status_code=201)
async def upload_dataset(
    kb_id: str,
    request: Request,
    name: str = "dataset",
    file: UploadFile = File(...),
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    content = await file.read()
    try:
        lines = [l.strip() for l in content.decode("utf-8").splitlines() if l.strip()]
        items = [json.loads(l) for l in lines]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSONL 解析失败: {e}")

    repo = _eval_repo(request)
    ds = await repo.create_dataset(uuid.UUID(kb_id), name)
    await repo.add_items(ds.id, items)
    ds = await repo.get_dataset(ds.id)
    return _dataset_to_dict(ds)


@router.delete("/api/eval/datasets/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str, request: Request, uid: str = Depends(get_current_user)):
    repo = _eval_repo(request)
    ds = await repo.get_dataset(uuid.UUID(dataset_id))
    if ds is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    # Verify ownership via KB
    kb = await _kb_repo(request).get(ds.kb_id)
    if kb is None or kb.uid != uid:
        raise HTTPException(status_code=404, detail="数据集不存在")
    await repo.delete_dataset(uuid.UUID(dataset_id))


# ---------------------------------------------------------------------------
# Eval Runs
# ---------------------------------------------------------------------------

@router.get("/api/eval/{kb_id}/runs")
async def list_runs(kb_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    runs = await _eval_repo(request).list_runs(uuid.UUID(kb_id))
    return [_run_to_dict(r) for r in runs]


@router.post("/api/eval/{kb_id}/runs", status_code=202)
async def create_run(
    kb_id: str,
    request: Request,
    body: EvalRunCreate,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _eval_repo(request)

    ds = await repo.get_dataset(uuid.UUID(body.dataset_id))
    if ds is None or str(ds.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="数据集不存在")

    run = await repo.create_run(
        kb_id=uuid.UUID(kb_id),
        dataset_id=uuid.UUID(body.dataset_id),
        name=body.name or f"Run {datetime.now(timezone.utc).strftime('%m-%d %H:%M')}",
        retrieval_config={"top_k": body.top_k},
    )

    asyncio.create_task(_run_evaluation(request, run.id, uuid.UUID(kb_id), body.top_k))
    return _run_to_dict(run)


@router.get("/api/eval/{kb_id}/runs/{run_id}")
async def get_run(kb_id: str, run_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _eval_repo(request)
    run = await repo.get_run(uuid.UUID(run_id))
    if run is None or str(run.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="运行不存在")
    items = await repo.list_run_items(uuid.UUID(run_id))
    result = _run_to_dict(run)
    result["items"] = [_run_item_to_dict(i) for i in items]
    return result


@router.delete("/api/eval/{kb_id}/runs/{run_id}", status_code=204)
async def delete_run(kb_id: str, run_id: str, request: Request, uid: str = Depends(get_current_user)):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _eval_repo(request)
    run = await repo.get_run(uuid.UUID(run_id))
    if run is None or str(run.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="运行不存在")
    await repo.delete_run(uuid.UUID(run_id))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_kb_or_404(kb_id: str, uid: str, request: Request):
    try:
        kid = uuid.UUID(kb_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="知识库不存在")
    kb = await _kb_repo(request).get(kid)
    if kb is None or kb.uid != uid:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


def _dataset_to_dict(ds) -> dict:
    return {
        "id": str(ds.id),
        "kb_id": str(ds.kb_id),
        "name": ds.name,
        "item_count": ds.item_count,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
    }


def _run_to_dict(run) -> dict:
    return {
        "id": str(run.id),
        "kb_id": str(run.kb_id),
        "dataset_id": str(run.dataset_id),
        "name": run.name,
        "status": run.status,
        "metrics": run.metrics,
        "overall_score": run.overall_score,
        "total_items": run.total_items,
        "completed_items": run.completed_items,
        "retrieval_config": run.retrieval_config,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


def _run_item_to_dict(item) -> dict:
    return {
        "id": str(item.id),
        "query": item.query,
        "gold_answer": item.gold_answer,
        "generated_answer": item.generated_answer,
        "retrieved_chunk_ids": item.retrieved_chunk_ids,
        "metrics": item.item_metrics,
    }


async def _run_evaluation(request: Request, run_id: uuid.UUID, kb_id: uuid.UUID, top_k: int) -> None:
    """Background task: run evaluation against all dataset items."""
    repo = _eval_repo(request)
    kb_repo = _kb_repo(request)
    settings = _rag_settings(request)

    await repo.update_run(run_id, status="running", started_at=datetime.now(timezone.utc))
    try:
        run = await repo.get_run(run_id)
        if run is None:
            return

        kb = await kb_repo.get(kb_id)
        chroma_col = (kb.chroma_collection if kb else None) or str(kb_id)

        items = await repo.list_items(run.dataset_id)
        await repo.update_run(run_id, total_items=len(items))

        # Build hybrid search once for this KB
        from nanobot.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
        from nanobot.rag.core.query_engine.dense_retriever import DenseRetriever
        from nanobot.rag.core.query_engine.sparse_retriever import SparseRetriever
        from nanobot.rag.core.query_engine.query_processor import QueryProcessor
        from nanobot.rag.core.query_engine.fusion import RRFFusion
        from nanobot.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
        from nanobot.rag.libs.embedding.embedding_factory import EmbeddingFactory
        from nanobot.rag.ingestion.storage.bm25_indexer import BM25Indexer
        from nanobot.rag.libs.evaluator.evaluator_factory import EvaluatorFactory
        from nanobot.rag.core.settings import resolve_path

        embedding = EmbeddingFactory.create(settings)
        vector_store = VectorStoreFactory.create(settings, collection_name=chroma_col)
        bm25 = BM25Indexer(index_dir=str(resolve_path("~/.nanoresearch/rag/bm25_index")))

        dense = DenseRetriever(settings=settings, embedding_client=embedding, vector_store=vector_store)
        sparse = SparseRetriever(
            settings=settings, bm25_indexer=bm25, vector_store=vector_store, default_collection=chroma_col
        )
        hybrid = HybridSearch(
            settings=settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(),
            config=HybridSearchConfig(fusion_top_k=top_k),
        )
        evaluator = EvaluatorFactory.create(settings)

        run_items: list[dict] = []
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        for idx, item in enumerate(items):
            try:
                search_result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda q=item.query: hybrid.search(q, top_k=top_k)
                )
                retrieved = search_result.results
                retrieved_ids = [r.chunk_id for r in retrieved]

                item_metrics = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda q=item.query, r=retrieved, g=item.gold_chunk_ids: evaluator.evaluate(
                        query=q,
                        retrieved_chunks=r,
                        ground_truth=g or [],
                    ),
                )
            except Exception:
                item_metrics = {}
                retrieved_ids = []

            run_items.append({
                "query": item.query,
                "gold_answer": item.gold_answer,
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": item_metrics,
            })

            for k, v in item_metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
                metric_counts[k] = metric_counts.get(k, 0) + 1

            await repo.update_run(run_id, completed_items=idx + 1)

        # Aggregate metrics
        agg_metrics = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        overall = sum(agg_metrics.values()) / len(agg_metrics) if agg_metrics else None

        await repo.add_run_items(run_id, run_items)
        await repo.update_run(
            run_id,
            status="completed",
            metrics=agg_metrics,
            overall_score=overall,
            finished_at=datetime.now(timezone.utc),
        )

    except Exception as exc:
        await repo.update_run(run_id, status="failed", finished_at=datetime.now(timezone.utc))
