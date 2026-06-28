"""Evaluation dataset and run endpoints — Quick (Recall@K/F1@K) + RAGAS."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.storage.repositories.eval_repo import EvalRepository
from nanoresearch.storage.repositories.knowledge_repo import KnowledgeRepository

router = APIRouter()


def _eval_repo(request: Request) -> EvalRepository:
    return EvalRepository(request.app.state.session_factory)


def _kb_repo(request: Request) -> KnowledgeRepository:
    return KnowledgeRepository(request.app.state.session_factory)


def _rag_settings(request: Request):
    settings = getattr(request.app.state, "rag_settings", None)
    if settings is None:
        from nanoresearch.rag.core.settings import load_settings
        settings = load_settings()
        request.app.state.rag_settings = settings
    return settings


async def _resolve_eval_spec(uid: str, request: Request, role, model_override=None):
    """Resolve a ModelSpec for an eval role using ModelFactory."""
    from nanoresearch.providers.model_factory import ModelFactory
    from nanoresearch.storage.repositories.user_settings_repo import UserSettingsRepository

    config = getattr(request.app.state, "config", None)
    rag_settings = _rag_settings(request)
    try:
        user_cfg = await UserSettingsRepository(request.app.state.session_factory).get(uid)
        user_model = user_cfg.model if user_cfg else None
        user_providers = (user_cfg.extra or {}).get("providers", []) if user_cfg else []
        user_extra = (user_cfg.extra or {}) if user_cfg else {}
    except Exception:
        user_model, user_providers, user_extra = None, [], {}

    return ModelFactory.resolve(
        role,
        config=config,
        rag_settings=rag_settings,
        user_model=user_model,
        user_providers=user_providers,
        model_override=model_override,
        user_extra=user_extra,
    )


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


class AgentRunCreate(BaseModel):
    dataset_id: str
    name: str | None = None
    top_k: int = 5
    evaluator_model: str | None = None
    embedding_model: str | None = None


class DatasetGenerateRequest(BaseModel):
    name: str | None = None
    n_questions: int = 20
    simple_ratio: float = 0.3
    multi_context_ratio: float = 0.5
    reasoning_ratio: float = 0.2
    generator_model: str | None = None
    embeddings_model: str | None = None


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


@router.post("/api/eval/{kb_id}/datasets/generate", status_code=202)
async def generate_dataset(
    kb_id: str,
    body: DatasetGenerateRequest,
    request: Request,
    uid: str = Depends(get_current_user),
):
    """Generate a RAGAS testset (multi-hop + single-hop) in the background."""
    from nanoresearch.providers.model_factory import ModelRole
    await _get_kb_or_404(kb_id, uid, request)
    repo = _eval_repo(request)
    settings = _rag_settings(request)

    gen_spec = await _resolve_eval_spec(uid, request, ModelRole.EVAL_GENERATOR, body.generator_model)
    emb_spec = await _resolve_eval_spec(uid, request, ModelRole.EMBEDDING)

    embeddings_model = (
        body.embeddings_model
        or emb_spec.model
        or "text-embedding-v3"
    )

    ds_name = body.name or f"自动生成 {datetime.now(timezone.utc).strftime('%m-%d %H:%M')}"
    ds = await repo.create_dataset(uuid.UUID(kb_id), ds_name)

    import os
    gen_api_key = gen_spec.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    gen_base_url = gen_spec.base_url or "https://api.openai.com/v1"
    emb_api_key = emb_spec.api_key or gen_api_key
    emb_base_url = emb_spec.base_url or gen_base_url

    asyncio.create_task(
        _run_dataset_generation(
            request, ds.id, uuid.UUID(kb_id), body,
            gen_spec.model, embeddings_model, gen_api_key, gen_base_url,
            emb_api_key, emb_base_url,
        )
    )
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

@router.post("/api/eval/{kb_id}/runs/agent", status_code=202)
async def create_agent_run(
    kb_id: str,
    request: Request,
    body: AgentRunCreate,
    uid: str = Depends(get_current_user),
):
    """Start an Agentic RAG eval run — agent answers each question; eval_contexts
    are collected from retrieve_* tool calls via on_tool_call callback."""
    await _get_kb_or_404(kb_id, uid, request)
    repo = _eval_repo(request)

    ds = await repo.get_dataset(uuid.UUID(body.dataset_id))
    if ds is None or str(ds.kb_id) != kb_id:
        raise HTTPException(status_code=404, detail="数据集不存在")

    from nanoresearch.providers.model_factory import ModelRole
    eval_spec = await _resolve_eval_spec(uid, request, ModelRole.EVAL_EVALUATOR, body.evaluator_model)
    emb_spec = await _resolve_eval_spec(uid, request, ModelRole.EMBEDDING)

    settings = _rag_settings(request)
    embedding_model = (
        body.embedding_model
        or emb_spec.model
        or getattr(getattr(settings, "embedding", None), "model", None)
        or "text-embedding-v3"
    )

    import os
    _default_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    _default_url = "https://api.openai.com/v1"

    run = await repo.create_run(
        kb_id=uuid.UUID(kb_id),
        dataset_id=uuid.UUID(body.dataset_id),
        name=body.name or f"Agent {datetime.now(timezone.utc).strftime('%m-%d %H:%M')}",
        retrieval_config={
            "top_k": body.top_k,
            "evaluator_model": eval_spec.model,
            "embedding_model": embedding_model,
        },
        eval_type="agent",
    )

    asyncio.create_task(
        _run_agent_eval(
            request, run.id, uuid.UUID(kb_id), uid, body.top_k,
            eval_spec.model, embedding_model,
            eval_api_key=eval_spec.api_key or _default_key,
            eval_base_url=eval_spec.base_url or _default_url,
            emb_api_key=emb_spec.api_key or _default_key,
            emb_base_url=emb_spec.base_url or _default_url,
        )
    )
    return _run_to_dict(run)


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

    from nanoresearch.providers.model_factory import ModelRole
    gen_spec = await _resolve_eval_spec(uid, request, ModelRole.EVAL_GENERATOR, body.generator_model)
    eval_spec = await _resolve_eval_spec(uid, request, ModelRole.EVAL_EVALUATOR, body.evaluator_model)
    emb_spec = await _resolve_eval_spec(uid, request, ModelRole.EMBEDDING)

    settings = _rag_settings(request)
    embedding_model = (
        emb_spec.model
        or getattr(getattr(settings, "embedding", None), "model", None)
        or "text-embedding-v3"
    )

    import os
    _default_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    _default_url = "https://api.openai.com/v1"

    run = await repo.create_run(
        kb_id=uuid.UUID(kb_id),
        dataset_id=uuid.UUID(body.dataset_id),
        name=body.name or f"RAGAS {datetime.now(timezone.utc).strftime('%m-%d %H:%M')}",
        retrieval_config={
            "top_k": body.top_k,
            "generator_model": gen_spec.model,
            "evaluator_model": eval_spec.model,
            "embedding_model": embedding_model,
        },
        eval_type="ragas",
    )

    asyncio.create_task(
        _run_ragas_evaluation(
            request, run.id, uuid.UUID(kb_id), body.top_k,
            gen_spec.model, eval_spec.model, embedding_model,
            gen_api_key=gen_spec.api_key or _default_key,
            gen_base_url=gen_spec.base_url or _default_url,
            eval_api_key=eval_spec.api_key or _default_key,
            eval_base_url=eval_spec.base_url or _default_url,
            emb_api_key=emb_spec.api_key or _default_key,
            emb_base_url=emb_spec.base_url or _default_url,
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
        "question_type": item.question_type,
    }


async def _run_dataset_generation(
    request: Request,
    dataset_id: uuid.UUID,
    kb_id: uuid.UUID,
    body: DatasetGenerateRequest,
    generator_model: str,
    embeddings_model: str,
    api_key: str = "sk-placeholder",
    base_url: str = "https://api.openai.com/v1",
    emb_api_key: str | None = None,
    emb_base_url: str | None = None,
) -> None:
    """Background task: generate RAGAS testset from KB chunks."""
    repo = _eval_repo(request)
    kb_repo = _kb_repo(request)
    try:
        # Load all KB chunks as LangChain Documents (chunk_id in metadata for gold lookup)
        chunks = await kb_repo.list_chunks_by_kb(kb_id, limit=2000)
        if not chunks:
            await repo.delete_dataset(dataset_id)
            return
        if len(chunks) == 2000:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[DatasetGenerate] chunk count hit limit=2000 for KB %s; "
                "chunks beyond the limit are silently excluded",
                kb_id,
            )

        from langchain_core.documents import Document as LCDocument
        lc_docs = [
            LCDocument(
                page_content=c.content,
                metadata={"chunk_id": str(c.id), "source": (c.chunk_metadata or {}).get("source_path", str(c.id))}
            )
            for c in chunks if c.content and c.content.strip()
        ]

        # Normalize ratios before entering the thread
        s, m, r = body.simple_ratio, body.multi_context_ratio, body.reasoning_ratio
        total = s + m + r
        if total <= 0:
            s, m, r = 0.4, 0.4, 0.2
            total = 1.0
        simple_w = s / total
        multi_abs_w = (m / 2) / total
        multi_spec_w = (m / 2) / total

        _emb_api_key = emb_api_key or api_key
        _emb_base_url = emb_base_url or base_url
        _n_questions = body.n_questions

        def _run_generation_in_thread() -> object:
            # All ragas/openai objects must be created inside the thread because
            # ragas uses asyncio.run() internally, which creates a new event loop.
            # AsyncOpenAI clients bound to the caller's loop would fail here.
            import logging as _logging
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            from ragas.embeddings.base import embedding_factory
            from ragas.testset import TestsetGenerator
            from ragas.testset.synthesizers import (
                SingleHopSpecificQuerySynthesizer,
                MultiHopAbstractQuerySynthesizer,
                MultiHopSpecificQuerySynthesizer,
            )
            from ragas.testset.transforms import default_transforms_for_prechunked
            from ragas.testset.transforms.extractors.llm_based import (
                ThemesExtractor,
                SummaryExtractor,
                NERExtractor,
            )
            from ragas.testset.transforms.extractors.embeddings import EmbeddingExtractor
            from ragas.testset.transforms.relationship_builders.cosine import CosineSimilarityBuilder
            from ragas.testset.graph import Relationship as _Relationship
            import numpy as _np
            from ragas.run_config import RunConfig

            gen_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            emb_client = AsyncOpenAI(base_url=_emb_base_url, api_key=_emb_api_key)
            gen_llm = llm_factory(generator_model, client=gen_client, max_tokens=8192)
            gen_emb = embedding_factory("openai", model=embeddings_model, client=emb_client, interface="modern")

            _transforms = default_transforms_for_prechunked(llm=gen_llm, embedding_model=gen_emb)
            # Monkey-patch extract() on LLM-based extractor instances (Summary, Themes)
            # so a single-node token-limit failure doesn't abort the entire pipeline.
            import types as _types
            _summary_skip_count = [0]
            _summary_total_count = [0]
            _themes_skip_count = [0]
            _themes_total_count = [0]
            _ner_skip_count = [0]
            _ner_total_count = [0]

            def _patch_transforms(obj):
                # obj may be a plain list (from default_transforms_for_prechunked)
                # or a Parallel/Transforms object with a .transformations attribute.
                if isinstance(obj, list):
                    items = obj
                elif hasattr(obj, "transformations"):
                    items = obj.transformations
                else:
                    return
                for t in items:
                    if isinstance(t, SummaryExtractor):
                        _orig = t.extract.__func__
                        async def _safe_summary(self, node, _orig=_orig):
                            _summary_total_count[0] += 1
                            try:
                                return await _orig(self, node)
                            except Exception as _e:
                                _summary_skip_count[0] += 1
                                _logging.getLogger(__name__).warning(
                                    "SummaryExtractor skipped node %s (%s: %s)",
                                    getattr(node, "id", "?"), type(_e).__name__, _e,
                                )
                                return self.property_name, "[extraction skipped]"
                        t.extract = _types.MethodType(_safe_summary, t)
                    elif isinstance(t, ThemesExtractor):
                        _orig = t.extract.__func__
                        async def _safe_themes(self, node, _orig=_orig):
                            _themes_total_count[0] += 1
                            try:
                                return await _orig(self, node)
                            except Exception as _e:
                                _themes_skip_count[0] += 1
                                _logging.getLogger(__name__).warning(
                                    "ThemesExtractor skipped node %s (%s: %s)",
                                    getattr(node, "id", "?"), type(_e).__name__, _e,
                                )
                                return self.property_name, []
                        t.extract = _types.MethodType(_safe_themes, t)
                    elif isinstance(t, EmbeddingExtractor):
                        _orig_emb = t.extract.__func__
                        async def _safe_embedding(self, node, _orig_emb=_orig_emb):
                            # Guard: read from embed_property_name (source text), not property_name (output destination)
                            try:
                                text = node.get_property(self.embed_property_name)
                            except Exception:
                                text = node.properties.get(self.embed_property_name, "")
                            text = text or ""
                            if not text.strip():
                                _logging.getLogger(__name__).warning(
                                    "EmbeddingExtractor skipped node %s — empty '%s'",
                                    getattr(node, "id", "?"), self.embed_property_name,
                                )
                                return self.property_name, []
                            return await _orig_emb(self, node)
                        t.extract = _types.MethodType(_safe_embedding, t)
                    elif isinstance(t, NERExtractor):
                        _orig_ner = t.extract.__func__
                        async def _safe_ner(self, node, _orig_ner=_orig_ner):
                            _ner_total_count[0] += 1
                            try:
                                return await _orig_ner(self, node)
                            except Exception as _e:
                                _ner_skip_count[0] += 1
                                _logging.getLogger(__name__).warning(
                                    "NERExtractor skipped node %s (%s: %s)",
                                    getattr(node, "id", "?"), type(_e).__name__, _e,
                                )
                                return self.property_name, []
                        t.extract = _types.MethodType(_safe_ner, t)
                    elif isinstance(t, CosineSimilarityBuilder):
                        # CosineSimilarityBuilder.generate_execution_plan raises ValueError
                        # for None embeddings and crashes numpy for [] embeddings.
                        # Patch it to filter out nodes whose embedding is None or [].
                        _orig_csp = t.generate_execution_plan.__func__
                        def _safe_cosine_plan(self, kg, _orig_csp=_orig_csp):
                            filtered_kg = self.filter(kg)
                            pairs = [
                                (n, n.get_property(self.property_name))
                                for n in filtered_kg.nodes
                            ]
                            pairs = [(n, e) for n, e in pairs if e]  # drop None / []
                            if len(pairs) < 2:
                                return []
                            valid_nodes, embs = zip(*pairs)
                            valid_nodes, embs = list(valid_nodes), list(embs)
                            first_len = len(embs[0])
                            if not all(len(e) == first_len for e in embs):
                                _logging.getLogger(__name__).warning(
                                    "CosineSimilarityBuilder: inconsistent embedding sizes, skipping"
                                )
                                return []
                            async def _add_rels(valid_nodes=valid_nodes, embs=embs):
                                similar_pairs = self._find_similar_embedding_pairs(
                                    _np.array(embs), self.threshold
                                )
                                for i, j, sim in similar_pairs:
                                    kg.relationships.append(_Relationship(
                                        source=valid_nodes[i],
                                        target=valid_nodes[j],
                                        type=self.new_property_name,
                                        properties={self.new_property_name: sim},
                                        bidirectional=True,
                                    ))
                            return [_add_rels()]
                        t.generate_execution_plan = _types.MethodType(_safe_cosine_plan, t)
                    _patch_transforms(t)

            _patch_transforms(_transforms)

            class _ChunkIdsMixin:
                async def _generate_sample(self, scenario, callbacks):
                    sample = await super()._generate_sample(scenario, callbacks)
                    chunk_ids = [
                        n.properties.get("document_metadata", {}).get("chunk_id")
                        for n in scenario.nodes
                        if n.properties.get("document_metadata", {}).get("chunk_id")
                    ]
                    sample.reference_context_ids = chunk_ids
                    return sample

            class _SingleHopSpecificWithIds(_ChunkIdsMixin, SingleHopSpecificQuerySynthesizer):
                pass

            class _MultiHopAbstractWithIds(_ChunkIdsMixin, MultiHopAbstractQuerySynthesizer):
                pass

            class _MultiHopSpecificWithIds(_ChunkIdsMixin, MultiHopSpecificQuerySynthesizer):
                pass

            generator = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb)
            distributions = [
                (_SingleHopSpecificWithIds(llm=gen_llm), simple_w),
                (_MultiHopAbstractWithIds(llm=gen_llm), multi_abs_w),
                (_MultiHopSpecificWithIds(llm=gen_llm), multi_spec_w),
            ]
            # deepseek ~60 RPM → max_workers=8 safe; retries handle transient 429s
            run_cfg = RunConfig(max_workers=10, max_retries=5, max_wait=60, timeout=120)
            result = generator.generate_with_chunks(
                lc_docs,
                testset_size=_n_questions,
                transforms=_transforms,
                query_distribution=distributions,
                run_config=run_cfg,
            )
            s_skip, s_total = _summary_skip_count[0], _summary_total_count[0]
            t_skip, t_total = _themes_skip_count[0], _themes_total_count[0]
            n_skip, n_total = _ner_skip_count[0], _ner_total_count[0]
            actual_n = len(result.samples)
            if not s_total and not t_total and not n_total:
                _logging.getLogger(__name__).warning(
                    "[Extractors] monkey-patch did not intercept any extractor calls — "
                    "the installed ragas version may have changed the transforms pipeline"
                )
            elif s_skip or t_skip or n_skip:
                parts = []
                if s_skip:
                    parts.append(f"SummaryExtractor {s_skip}/{s_total} skipped")
                if t_skip:
                    parts.append(f"ThemesExtractor {t_skip}/{t_total} skipped")
                if n_skip:
                    parts.append(f"NERExtractor {n_skip}/{n_total} skipped")
                _logging.getLogger(__name__).warning(
                    "[Extractors] %s. Generated %d samples (target=%d).",
                    "; ".join(parts), actual_n, _n_questions,
                )
            elif actual_n < _n_questions:
                _logging.getLogger(__name__).warning(
                    "[DatasetGenerate] only %d/%d samples produced — likely not enough "
                    "connected nodes in the knowledge graph after filtering",
                    actual_n, _n_questions,
                )
            return result

        dataset = await asyncio.get_running_loop().run_in_executor(None, _run_generation_in_thread)

        # Convert RAGAS dataset to eval items
        # dataset.samples is List[TestsetSample]; actual fields live on .eval_sample
        items = []
        for sample in dataset.samples:
            es = sample.eval_sample
            query = getattr(es, "user_input", None) or ""
            reference = getattr(es, "reference", None) or ""

            # reference_context_ids populated by _ChunkIdsMixin._generate_sample
            gold_chunk_ids = list(
                filter(None, getattr(es, "reference_context_ids", None) or [])
            )

            if "Multi" in (getattr(sample, "synthesizer_name", "") or ""):
                q_type = "multi_context"
            else:
                q_type = "single_hop"

            items.append({
                "query": query,
                "gold_answer": reference,
                "gold_chunk_ids": gold_chunk_ids,
                "question_type": q_type,
            })

        await repo.add_items(dataset_id, items)

    except Exception as exc:
        import sys, traceback
        print(f"[DatasetGenerate] failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            await repo.delete_dataset(dataset_id)
        except Exception:
            pass


def _build_hybrid_search(settings, chroma_col: str, top_k: int):
    from nanoresearch.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
    from nanoresearch.rag.core.query_engine.dense_retriever import DenseRetriever
    from nanoresearch.rag.core.query_engine.sparse_retriever import SparseRetriever
    from nanoresearch.rag.core.query_engine.query_processor import QueryProcessor
    from nanoresearch.rag.core.query_engine.fusion import RRFFusion
    from nanoresearch.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
    from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory
    from nanoresearch.rag.ingestion.storage.bm25_indexer import BM25Indexer
    from nanoresearch.rag.core.settings import resolve_path

    embedding = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=chroma_col)
    bm25 = BM25Indexer(index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{chroma_col}")))

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

        from nanoresearch.rag.libs.evaluator.evaluator_factory import EvaluatorFactory
        evaluator = EvaluatorFactory.create(settings)

        run_items: list[dict] = []
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        for idx, item in enumerate(items):
            try:
                search_result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda q=item.query: hybrid.search(q, top_k=top_k, return_details=True)
                )
                retrieved = search_result.results
                retrieved_ids = [r.chunk_id for r in retrieved]

                # Convert chroma_ids → kb_chunks.id UUIDs so they match gold_chunk_ids
                retrieved_db_chunks = await kb_repo.get_chunks_by_chroma_ids(retrieved_ids)
                retrieved_uuids = [str(c.id) for c in retrieved_db_chunks]

                item_metrics = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda q=item.query, ids=retrieved_uuids, g=item.gold_chunk_ids: evaluator.evaluate(
                        query=q,
                        retrieved_chunks=ids,
                        ground_truth=g or [],
                    ),
                )
            except Exception as _eval_exc:
                logger.warning("Quick eval failed for item %d: %s", idx, _eval_exc, exc_info=True)
                item_metrics = {}
                retrieved_ids = []

            run_items.append({
                "query": item.query,
                "gold_answer": item.gold_answer,
                "retrieved_contexts": retrieved_ids,
                "metrics": item_metrics,
                "question_type": item.question_type,
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
    gen_api_key: str = "sk-placeholder",
    gen_base_url: str = "https://api.openai.com/v1",
    eval_api_key: str = "sk-placeholder",
    eval_base_url: str = "https://api.openai.com/v1",
    emb_api_key: str = "sk-placeholder",
    emb_base_url: str = "https://api.openai.com/v1",
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

        from nanoresearch.rag.core.query_engine.reranker import create_core_reranker
        reranker = await asyncio.get_running_loop().run_in_executor(
            None, lambda: create_core_reranker(settings)
        )

        # Generator / Evaluator / Embedding use separate clients since they may live
        # on different providers (e.g. deepseek gen + qwen eval + dashscope emb).
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

        gen_client = AsyncOpenAI(base_url=gen_base_url, api_key=gen_api_key)
        eval_client = AsyncOpenAI(base_url=eval_base_url, api_key=eval_api_key)
        emb_client = AsyncOpenAI(base_url=emb_base_url, api_key=emb_api_key)

        evaluator_llm = llm_factory(evaluator_model, client=eval_client, max_tokens=8192)
        eval_embedding = embedding_factory("openai", model=embedding_model, client=emb_client, interface="modern")

        faithfulness = Faithfulness(llm=evaluator_llm)
        answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=eval_embedding)
        context_precision = ContextPrecision(llm=evaluator_llm)
        context_recall = ContextRecall(llm=evaluator_llm)

        run_items: list[dict] = []
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        for idx, item in enumerate(dataset_items):
            # Retrieve → Rerank → extract context texts
            try:
                search_result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda q=item.query: hybrid.search(q, top_k=top_k)
                )
                rerank_result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda q=item.query, r=search_result: reranker.rerank(q, r, top_k=top_k)
                )
                contexts = [r.text for r in rerank_result.results]
            except Exception as _retrieval_exc:
                logger.warning("RAGAS retrieval failed for item %d: %s", idx, _retrieval_exc, exc_info=True)
                contexts = []

            # Generate answer
            generated_answer = ""
            try:
                context_text = "\n\n---\n\n".join(contexts)
                resp = await gen_client.chat.completions.create(
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

            # Score each metric individually — RAGAS 0.4.3 uses ascore() with per-metric params
            gold = item.gold_answer or ""
            query = item.query
            item_metrics: dict[str, float] = {}
            metric_calls = [
                ("faithfulness",      faithfulness.ascore(user_input=query, response=generated_answer, retrieved_contexts=contexts)),
                ("answer_relevancy",  answer_relevancy.ascore(user_input=query, response=generated_answer)),
                ("context_precision", context_precision.ascore(user_input=query, reference=gold, retrieved_contexts=contexts)),
                ("context_recall",    context_recall.ascore(user_input=query, retrieved_contexts=contexts, reference=gold)),
            ]
            for mn, coro in metric_calls:
                try:
                    result = await coro
                    score_val = float(result)
                    item_metrics[mn] = score_val
                    metric_sums[mn] = metric_sums.get(mn, 0.0) + score_val
                    metric_counts[mn] = metric_counts.get(mn, 0) + 1
                except Exception as _metric_exc:
                    logger.warning("RAGAS metric %s failed for item %d: %s", mn, idx, _metric_exc, exc_info=True)

            run_items.append({
                "query": item.query,
                "gold_answer": item.gold_answer,
                "generated_answer": generated_answer,
                "retrieved_contexts": contexts,
                "metrics": item_metrics,
                "question_type": item.question_type,
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


# ---------------------------------------------------------------------------
# Agent eval background task — uses on_tool_call to harvest retrieve_* contexts
# ---------------------------------------------------------------------------

async def _build_eval_agent_loop(uid: str, app_state):
    """Create a per-user AgentLoop for eval runs using app.state.loop_config."""
    from pathlib import Path
    from nanoresearch.agent.loop import AgentLoop
    from nanoresearch.providers.model_factory import ModelFactory, ModelRole
    from nanoresearch.session.manager import SessionManager
    from nanoresearch.storage.repositories.user_settings_repo import UserSettingsRepository

    cfg = getattr(app_state, "loop_config", {})
    factory = app_state.session_factory
    user_cfg = await UserSettingsRepository(factory).get(uid)

    base: Path = cfg.get("base_workspace") or Path.home() / ".nanoresearch" / "workspace"
    ws = base / "users" / uid
    ws.mkdir(parents=True, exist_ok=True)

    session_manager = SessionManager(ws, session_factory=factory, default_uid=uid)

    providers = ((user_cfg.extra or {}).get("providers") or []) if user_cfg else []
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        config=cfg.get("config"),
        rag_settings=None,
        user_model=user_cfg.model if user_cfg else None,
        user_providers=providers,
        model_override=None,
    )

    def _build_provider(spec, fallback):
        if not spec or not spec.api_key:
            return fallback
        from nanoresearch.providers.registry import find_by_name
        p_spec = find_by_name(spec.provider) if spec.provider else None
        backend = p_spec.backend if p_spec else "openai_compat"
        if backend == "anthropic":
            from nanoresearch.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(
                api_key=spec.api_key,
                api_base=spec.base_url or (p_spec.default_api_base if p_spec else None),
                default_model=spec.model or "claude-sonnet-4-20250514",
                extra_headers=spec.extra_headers,
            )
        if backend == "azure_openai":
            from nanoresearch.providers.azure_openai_provider import AzureOpenAIProvider
            return AzureOpenAIProvider(
                api_key=spec.api_key, api_base=spec.base_url,
                default_model=spec.model or "gpt-4o",
            )
        from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=spec.api_key, api_base=spec.base_url,
            default_model=spec.model or "gpt-4o",
            extra_headers=spec.extra_headers, spec=p_spec,
        )

    provider = _build_provider(spec, cfg.get("provider"))

    return AgentLoop(
        bus=cfg.get("bus"),
        provider=provider,
        workspace=ws,
        model=user_cfg.model if user_cfg else None,
        max_iterations=cfg.get("max_iterations", 40),
        context_window_tokens=cfg.get("context_window_tokens", 65536),
        web_search_config=cfg.get("web_search_config"),
        web_proxy=cfg.get("web_proxy"),
        exec_config=cfg.get("exec_config"),
        cron_service=cfg.get("cron_service"),
        restrict_to_workspace=True,
        session_manager=session_manager,
        mcp_servers=cfg.get("mcp_servers"),
        channels_config=cfg.get("channels_config"),
        timezone=cfg.get("timezone"),
        research_config=cfg.get("research_config"),
        knowledge_search=cfg.get("knowledge_search"),
        rag_store=cfg.get("rag_store"),
        uid=uid,
        session_factory=factory,
        rag_settings=cfg.get("rag_settings"),
    )


async def _run_agent_eval(
    request: Request,
    run_id: uuid.UUID,
    kb_id: uuid.UUID,
    uid: str,
    top_k: int,
    evaluator_model: str,
    embedding_model: str,
    eval_api_key: str = "sk-placeholder",
    eval_base_url: str = "https://api.openai.com/v1",
    emb_api_key: str = "sk-placeholder",
    emb_base_url: str = "https://api.openai.com/v1",
) -> None:
    """Agent eval: uses RAG internal loop (plan → multi-hop search → fuse → verify)
    to collect eval_contexts, then scores with RAGAS."""
    repo = _eval_repo(request)
    kb_repo = _kb_repo(request)

    await repo.update_run(run_id, status="running", started_at=datetime.now(timezone.utc))
    try:
        run = await repo.get_run(run_id)
        if run is None:
            return

        kb = await kb_repo.get(kb_id)
        chroma_col = (kb.chroma_collection if kb else None) or str(kb_id)

        dataset_items = await repo.list_items(run.dataset_id)
        await repo.update_run(run_id, total_items=len(dataset_items))

        # RAG internal loop runner — pure KB retrieval, no web/file tools
        from nanoresearch.rag.internal_loop.runner import RAGLoopRunner
        rag_runner = RAGLoopRunner()

        # RAGAS evaluator setup — LLM for eval, LLM for answer generation (reuse eval_client)
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

        gen_client = AsyncOpenAI(base_url=eval_base_url, api_key=eval_api_key)
        eval_client = AsyncOpenAI(base_url=eval_base_url, api_key=eval_api_key)
        emb_client = AsyncOpenAI(base_url=emb_base_url, api_key=emb_api_key)
        evaluator_llm = llm_factory(evaluator_model, client=eval_client, max_tokens=8192)
        eval_embedding = embedding_factory("openai", model=embedding_model, client=emb_client, interface="modern")
        faithfulness = Faithfulness(llm=evaluator_llm)
        answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=eval_embedding)
        context_precision = ContextPrecision(llm=evaluator_llm)
        context_recall = ContextRecall(llm=evaluator_llm)

        run_items: list[dict] = []
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        for idx, item in enumerate(dataset_items):
            # Run agentic RAG internal loop: query planning + multi-hop retrieval + fusion + verify
            rag_result = await rag_runner.run(
                query=item.query,
                collection=chroma_col,
                max_iterations=3,
            )

            # Extract context texts from fused chunks (dicts with 'text' or 'content' key)
            eval_contexts: list[str] = []
            for chunk in rag_result.chunks:
                text = ""
                if isinstance(chunk, dict):
                    text = chunk.get("text") or chunk.get("content") or ""
                elif hasattr(chunk, "text"):
                    text = chunk.text or ""
                elif hasattr(chunk, "content"):
                    text = chunk.content or ""
                if text:
                    eval_contexts.append(str(text)[:2000])

            logger.info(
                "Agent eval item %d: %d chunks, %d iterations, success=%s",
                idx, len(rag_result.chunks), rag_result.iterations, rag_result.success,
            )

            # Generate answer from retrieved contexts
            generated_answer = ""
            try:
                context_text = "\n\n---\n\n".join(eval_contexts)
                resp = await gen_client.chat.completions.create(
                    model=evaluator_model,
                    temperature=0,
                    max_tokens=2048,
                    messages=[
                        {"role": "system", "content": "基于提供的上下文信息准确回答问题。"},
                        {"role": "user", "content": f"上下文:\n{context_text}\n\n问题: {item.query}"},
                    ],
                )
                generated_answer = resp.choices[0].message.content or ""
            except Exception as _gen_exc:
                logger.warning("Agent eval answer generation failed for item %d: %s", idx, _gen_exc)
                generated_answer = rag_result.summary or ""

            # RAGAS scoring — 0.4.3 API: per-metric ascore() with explicit params
            gold = item.gold_answer or ""
            query = item.query
            item_metrics: dict = {
                "_hops": rag_result.iterations,
                "_success": rag_result.success,
            }
            metric_calls = [
                ("faithfulness",      faithfulness.ascore(user_input=query, response=generated_answer, retrieved_contexts=eval_contexts)),
                ("answer_relevancy",  answer_relevancy.ascore(user_input=query, response=generated_answer)),
                ("context_precision", context_precision.ascore(user_input=query, reference=gold, retrieved_contexts=eval_contexts)),
                ("context_recall",    context_recall.ascore(user_input=query, retrieved_contexts=eval_contexts, reference=gold)),
            ]
            for mn, coro in metric_calls:
                try:
                    result = await coro
                    score_val = float(result)
                    item_metrics[mn] = score_val
                    metric_sums[mn] = metric_sums.get(mn, 0.0) + score_val
                    metric_counts[mn] = metric_counts.get(mn, 0) + 1
                except Exception as _metric_exc:
                    logger.warning("Agent eval RAGAS metric %s failed for item %d: %s", mn, idx, _metric_exc, exc_info=True)

            run_items.append({
                "query": item.query,
                "gold_answer": item.gold_answer,
                "generated_answer": generated_answer,
                "retrieved_contexts": eval_contexts,
                "metrics": item_metrics,
                "question_type": item.question_type,
            })
            await repo.update_run(run_id, completed_items=idx + 1)

        agg_metrics = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        if run_items:
            all_hops = [it["metrics"].get("_hops", 0) for it in run_items]
            agg_metrics["_avg_hops"] = sum(all_hops) / len(all_hops)
        non_private = {k: v for k, v in agg_metrics.items() if not k.startswith("_")}
        overall = sum(non_private.values()) / len(non_private) if non_private else None

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
