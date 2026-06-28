"""Context root-cause diagnoser: distinguishes kb_gap vs transient retrieval failures."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from nanoresearch.storage.models import AgentRunSnapshot, KnowledgeBase

_QUERY_KEYS = ["query", "q", "search_query", "text", "keyword"]


class ContextDiagnoser:
    async def diagnose(
        self,
        snapshot: "AgentRunSnapshot",
        kb_list: "list[KnowledgeBase]",
        session_factory: "async_sessionmaker",
        top_k: int = 5,
    ) -> dict:
        from nanoresearch.eval.badcase_classifier import _is_empty_output
        from nanoresearch.rag.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
        from nanoresearch.rag.core.query_engine.dense_retriever import DenseRetriever
        from nanoresearch.rag.core.query_engine.sparse_retriever import SparseRetriever
        from nanoresearch.rag.core.query_engine.query_processor import QueryProcessor
        from nanoresearch.rag.core.query_engine.fusion import RRFFusion
        from nanoresearch.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
        from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory
        from nanoresearch.rag.ingestion.storage.bm25_indexer import BM25Indexer
        from nanoresearch.rag.core.settings import load_settings, resolve_path

        settings = load_settings()
        chain = snapshot.tool_call_chain or []
        empty_entries = [e for e in chain if _is_empty_output(e.get("result"))]

        queries: list[dict] = []
        for entry in empty_entries:
            params = entry.get("params") or {}
            query = next((params[k] for k in _QUERY_KEYS if k in params), None)
            tool_name = entry.get("name")

            if query is None or not kb_list:
                queries.append({
                    "tool_name": tool_name,
                    "query": query,
                    "kb_name": None,
                    "now_count": None,
                    "verdict": "unknown",
                })
                continue

            verdict = "kb_gap"
            result_kb_name: str | None = kb_list[0].name
            result_count = 0
            any_searched = False

            for kb in kb_list:
                chroma_col = kb.chroma_collection or str(kb.id)
                try:
                    embedding = EmbeddingFactory.create(settings)
                    vector_store = VectorStoreFactory.create(settings, collection_name=chroma_col)
                    bm25 = BM25Indexer(
                        index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{chroma_col}"))
                    )
                    dense = DenseRetriever(
                        settings=settings, embedding_client=embedding, vector_store=vector_store
                    )
                    sparse = SparseRetriever(
                        settings=settings,
                        bm25_indexer=bm25,
                        vector_store=vector_store,
                        default_collection=chroma_col,
                    )
                    hybrid = HybridSearch(
                        settings=settings,
                        query_processor=QueryProcessor(),
                        dense_retriever=dense,
                        sparse_retriever=sparse,
                        fusion=RRFFusion(),
                        config=HybridSearchConfig(fusion_top_k=top_k),
                        session_factory=session_factory,
                        kb_id=kb.id,
                    )
                    result = await hybrid.async_search(query, top_k=top_k)
                    now_count = len(result.results) if result else 0
                    any_searched = True
                    if now_count > 0:
                        verdict = "transient"
                        result_kb_name = kb.name
                        result_count = now_count
                        break
                except Exception:
                    pass

            if not any_searched:
                verdict = "unknown"

            queries.append({
                "tool_name": tool_name,
                "query": query,
                "kb_name": result_kb_name if any_searched else None,
                "now_count": result_count if any_searched else None,
                "verdict": verdict,
            })

        return {
            "snapshot_id": str(snapshot.id),
            "user_input": (snapshot.user_input or "")[:120],
            "queries": queries,
        }
