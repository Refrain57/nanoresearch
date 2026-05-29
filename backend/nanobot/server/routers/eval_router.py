"""Evaluation dataset and run endpoints — Quick (Recall@K/F1@K) + RAGAS."""

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


class RagasRunCreate(BaseModel):
    dataset_id: str
    name: str | None = None
    top_k: int = 5
    generator_model: str | None = None  # falls back to settings.eval.generator_model
    evaluator_model: str | None = None  # falls back to settings.eval.evaluator_model


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
        name=body.name or f"Quick {datetime.now(timezone.utc).strftime('%m-%d %H:%M')}",
        retrieval_config={"top_k": body.top_k},
        eval_type="quick",
    )

    asyncio.create_task(_run_quick_evaluation(request, run.id, uuid.UUID(kb_id), body.top_k))
    return _run_to_dict(run)


@router.post("/api/eval/{kb_id}/runs/ragas", status_code=202)
async def create_ragas_run(
    kb_id: str,
    request: Request,
    body: RagasRunCreate,
    uid: str = Depends(get_current_user),
):
    await _get_kb_or_404(kb_id, uid, request)
    repo = _eval_repo(request)

    ds = await repo.get_dataset(uuid.UUID(body.dataset_id))
    if ds is None or str(ds.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="数据集不存在")

    from nanobot.storage.repositories.user_settings_repo import UserSettingsRepository
    user_cfg = await UserSettingsRepository(request.app.state.session_factory).get(uid)
    user_extra = (user_cfg.extra if user_cfg else None) or {}

    settings = _rag_settings(request)
    eval_cfg = getattr(settings, "eval", None) or {}
    generator_model = (
        body.generator_model
        or user_extra.get("ragas_generator_model")
        or getattr(eval_cfg, "generator_model", None)
        or "qwen-plus"
    )
    evaluator_model = (
        body.evaluator_model
        or user_extra.get("ragas_evaluator_model")
        or getattr(eval_cfg, "evaluator_model", None)
        or "qwen-max"
    )
    embedding_model = (
        user_extra.get("ragas_embedding_model")
        or getattr(getattr(settings, "embedding", None), "model", None)
        or "text-embedding-v3"
    )

    run = await repo.create_run(
        kb_id=uuid.UUID(kb_id),
        dataset_id=uuid.UUID(body.dataset_id),
        name=body.name or f"RAGAS {datetime.now(timezone.utc).strftime('%m-%d %H:%M')}",
        retrieval_config={
            "top_k": body.top_k,
            "generator_model": generator_model,
            "evaluator_model": evaluator_model,
            "embedding_model": embedding_model,
        },
        eval_type="ragas",
    )

    asyncio.create_task(
        _run_ragas_evaluation(
            request, run.id, uuid.UUID(kb_id), body.top_k,
            generator_model, evaluator_model, embedding_model,
        )
    )
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
        "eval_type": run.eval_type,
        "status": run.status,
        "metrics": run.metrics,
        "overall_score": run.overall_score,
        "total_items": run.total_items,
        "completed_items": run.completed_items,
        "error_message": run.error_message,
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
        "retrieved_contexts": item.retrieved_contexts,
        "metrics": item.item_metrics,
    }


def _build_hybrid_search(settings, chroma_col: str, top_k: int):
    from nanobot.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
    from nanobot.rag.core.query_engine.dense_retriever import DenseRetriever
    from nanobot.rag.core.query_engine.sparse_retriever import SparseRetriever
    from nanobot.rag.core.query_engine.query_processor import QueryProcessor
    from nanobot.rag.core.query_engine.fusion import RRFFusion
    from nanobot.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
    from nanobot.rag.libs.embedding.embedding_factory import EmbeddingFactory
    from nanobot.rag.ingestion.storage.bm25_indexer import BM25Indexer
    from nanobot.rag.core.settings import resolve_path

    embedding = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=chroma_col)
    bm25 = BM25Indexer(index_dir=str(resolve_path("~/.nanoresearch/rag/bm25_index")))

    dense = DenseRetriever(settings=settings, embedding_client=embedding, vector_store=vector_store)
    sparse = SparseRetriever(
        settings=settings, bm25_indexer=bm25, vector_store=vector_store, default_collection=chroma_col
    )
    return HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=RRFFusion(),
        config=HybridSearchConfig(fusion_top_k=top_k),
    )


# ---------------------------------------------------------------------------
# Quick eval background task (Recall@K + F1@K)
# ---------------------------------------------------------------------------

async def _run_quick_evaluation(request: Request, run_id: uuid.UUID, kb_id: uuid.UUID, top_k: int) -> None:
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

        hybrid = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _build_hybrid_search(settings, chroma_col, top_k)
        )

        from nanobot.rag.libs.evaluator.evaluator_factory import EvaluatorFactory
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
                "retrieved_contexts": retrieved_ids,
                "metrics": item_metrics,
            })

            for k, v in item_metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
                metric_counts[k] = metric_counts.get(k, 0) + 1

            await repo.update_run(run_id, completed_items=idx + 1)

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
        await repo.update_run(
            run_id, status="failed",
            error_message=str(exc),
            finished_at=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# RAGAS eval background task (Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall)
# ---------------------------------------------------------------------------

async def _run_ragas_evaluation(
    request: Request,
    run_id: uuid.UUID,
    kb_id: uuid.UUID,
    top_k: int,
    generator_model: str,
    evaluator_model: str,
    embedding_model: str = "text-embedding-v3",
) -> None:
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

        dataset_items = await repo.list_items(run.dataset_id)
        await repo.update_run(run_id, total_items=len(dataset_items))

        hybrid = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _build_hybrid_search(settings, chroma_col, top_k)
        )

        # Generator LLM (answers) + Evaluator LLM (judges) kept separate to avoid self-eval bias
        import os
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

        base_url = settings.llm.base_url or "https://api.openai.com/v1"
        api_key = settings.llm.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
        emb_model = embedding_model

        shared_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        evaluator_llm = llm_factory(evaluator_model, client=shared_client, max_tokens=8192)
        eval_embedding = embedding_factory("openai", model=emb_model, client=shared_client, interface="modern")

        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=eval_embedding),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
        ]

        run_items: list[dict] = []
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        for idx, item in enumerate(dataset_items):
            # Retrieve context texts
            try:
                search_result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda q=item.query: hybrid.search(q, top_k=top_k)
                )
                contexts = [r.text for r in search_result.results]
            except Exception:
                contexts = []

            # Generate answer
            generated_answer = ""
            try:
                context_text = "\n\n---\n\n".join(contexts)
                resp = await shared_client.chat.completions.create(
                    model=generator_model,
                    temperature=0,
                    max_tokens=2048,
                    messages=[
                        {"role": "system", "content": "基于提供的上下文信息准确回答问题。"},
                        {"role": "user", "content": f"上下文:\n{context_text}\n\n问题: {item.query}"},
                    ],
                )
                generated_answer = resp.choices[0].message.content or ""
            except Exception:
                pass

            # Score each metric individually (skip failures, don't corrupt aggregate)
            from ragas.dataset_schema import SingleTurnSample
            sample = SingleTurnSample(
                user_input=item.query,
                response=generated_answer,
                retrieved_contexts=contexts,
                reference=item.gold_answer or "",
            )
            item_metrics: dict[str, float] = {}
            for metric in metrics:
                try:
                    mn = metric.name
                    if mn == "faithfulness":
                        raw = await metric.ascore(
                            user_input=sample.user_input,
                            response=sample.response,
                            retrieved_contexts=sample.retrieved_contexts,
                        )
                    elif mn == "answer_relevancy":
                        raw = await metric.ascore(
                            user_input=sample.user_input,
                            response=sample.response,
                        )
                    elif mn == "context_precision":
                        raw = await metric.ascore(
                            user_input=sample.user_input,
                            reference=sample.reference,
                            retrieved_contexts=sample.retrieved_contexts,
                        )
                    elif mn == "context_recall":
                        raw = await metric.ascore(
                            user_input=sample.user_input,
                            retrieved_contexts=sample.retrieved_contexts,
                            reference=sample.reference,
                        )
                    else:
                        continue
                    score_val = float(raw.score) if hasattr(raw, "score") else float(raw)
                    item_metrics[mn] = score_val
                    metric_sums[mn] = metric_sums.get(mn, 0.0) + score_val
                    metric_counts[mn] = metric_counts.get(mn, 0) + 1
                except Exception:
                    pass

            run_items.append({
                "query": item.query,
                "gold_answer": item.gold_answer,
                "generated_answer": generated_answer,
                "retrieved_contexts": contexts,
                "metrics": item_metrics,
            })
            await repo.update_run(run_id, completed_items=idx + 1)

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
        await repo.update_run(
            run_id, status="failed",
            error_message=str(exc),
            finished_at=datetime.now(timezone.utc),
        )
