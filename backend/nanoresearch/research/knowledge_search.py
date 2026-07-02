"""Knowledge Search — user memory storage and retrieval.

Provides storage and retrieval for user-specific knowledge derived from
conversations: preferences, habits, and decisions.

Research claims/insights have been deprecated. Research output is now
saved as MD files and ingested into the KB via the standard pipeline.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from nanoresearch.rag.libs.embedding.base_embedding import BaseEmbedding
from nanoresearch.rag.libs.vector_store.chroma_store import ChromaStore

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nanoresearch.rag.core.settings import Settings
    from nanoresearch.rag.libs.reranker.base_reranker import BaseReranker


class KnowledgeSearch:
    """Storage and retrieval for user memory (conversation-derived knowledge).

    Search pipeline: BM25 (in-memory) + vector → RRF fusion → optional Rerank → time decay.

    Attributes:
        mem_events_store: ChromaDB collection for atomic events.
        mem_conv_summaries_store: ChromaDB collection for conversation summaries.
        dense_encoder: Embedding encoder for vector search.
    """

    def __init__(
        self,
        dense_encoder: BaseEmbedding,
        settings: "Settings | None" = None,
        mem_events_store: ChromaStore | None = None,
        mem_conv_summaries_store: ChromaStore | None = None,
    ):
        self.dense_encoder = dense_encoder
        self.mem_events_store = mem_events_store
        self.mem_conv_summaries_store = mem_conv_summaries_store
        self._settings = settings
        self._reranker: "BaseReranker | None | bool" = None  # None=uninit, False=unavailable

    @classmethod
    def from_settings(cls, settings: "Settings", collection_suffix: str = "") -> "KnowledgeSearch":
        from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory

        mem_events_store = ChromaStore(
            settings=settings,
            collection_name=f"mem_events{collection_suffix}",
        )
        mem_conv_summaries_store = ChromaStore(
            settings=settings,
            collection_name=f"mem_conv_summaries{collection_suffix}",
        )
        dense_encoder = EmbeddingFactory.create(settings)

        return cls(
            dense_encoder=dense_encoder,
            settings=settings,
            mem_events_store=mem_events_store,
            mem_conv_summaries_store=mem_conv_summaries_store,
        )

    def _get_reranker(self) -> "BaseReranker | None":
        """Lazy-init cross-encoder reranker from settings."""
        if self._reranker is False:
            return None
        if self._reranker is not None:
            return self._reranker  # type: ignore[return-value]
        if self._settings is None:
            self._reranker = False
            return None
        try:
            from nanoresearch.rag.libs.reranker.reranker_factory import RerankerFactory
            from nanoresearch.rag.libs.reranker.base_reranker import NoneReranker
            reranker = RerankerFactory.create(self._settings)
            if isinstance(reranker, NoneReranker):
                self._reranker = False
                return None
            self._reranker = reranker
            return self._reranker  # type: ignore[return-value]
        except Exception as e:
            logger.debug(f"KnowledgeSearch: reranker unavailable ({e})")
            self._reranker = False
            return None

    # ============== User Memory Methods ==============

    def _hybrid_search(
        self,
        store: "ChromaStore | None",
        query: str,
        top_k: int = 5,
        apply_decay: bool = True,
        uid: str | None = None,
        extra_filters: dict[str, Any] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Shared hybrid recall chain (over-retrieve → uid/metadata filter → BM25 → RRF →
        optional rerank → time decay). Reused verbatim by every memory collection so ranking
        behaviour never diverges per store (plan C6)."""
        if not store:
            return []

        # Step 1: Vector search — over-retrieve for downstream filtering
        vector = self._embed(query)
        candidates = store.query(vector=vector, top_k=top_k * 4)

        if uid:
            candidates = [r for r in candidates if r.get("metadata", {}).get("uid") == uid]
        if extra_filters:
            for fk, fv in extra_filters.items():
                candidates = [r for r in candidates if r.get("metadata", {}).get(fk) == fv]
        if exclude_ids:
            candidates = [r for r in candidates if r.get("id") not in exclude_ids]

        if not candidates:
            return []

        # Step 2: In-memory BM25 scoring
        bm25_ranked = self._bm25_rank(query, candidates)

        # Step 3: RRF fusion of vector rank and BM25 rank
        fused = self._rrf_fuse(candidates, bm25_ranked, top_k=top_k * 2)

        # Step 4: Rerank (optional, lazy cross-encoder)
        reranker = self._get_reranker()
        if reranker is not None:
            rerank_inputs = [
                {**r, "text": r.get("metadata", {}).get("text", "")}
                for r in fused
            ]
            try:
                reranked = reranker.rerank(query, rerank_inputs, top_k=top_k * 2)
                fused = [
                    {**r, "score": r.get("rerank_score", r.get("score", 0.0))}
                    for r in reranked
                ]
            except Exception as e:
                logger.debug(f"KnowledgeSearch: rerank failed, using fused results ({e})")

        # Step 5: Time decay and final ranking
        if apply_decay and fused:
            fused = self._apply_decay(fused)

        return fused[:top_k]

    # ============== Events (P2) — append-only, no TTL/cleanup (plan C5) ==============

    def write_events_sync(self, events: list[dict[str, Any]], uid: str | None = None) -> list[str]:
        """Append atomic events to mem_events; returns the new record ids (used to backfill
        memory_facts.derived_from — plan C4). Append-only: no dedup gate, no TTL/cleanup."""
        if not events or not self.mem_events_store:
            return []
        texts = [
            f"{e.get('topic', '')} | {e.get('action', '')} | {e.get('result', '')}"
            for e in events
        ]
        vectors = self.dense_encoder.embed(texts)
        to_insert, ids = [], []
        for e, vec, text in zip(events, vectors, texts):
            record_id = f"ev_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            to_insert.append({
                "id": record_id,
                "vector": vec,
                "metadata": {
                    "type": "event",
                    "uid": uid or "",
                    "conversation_id": e.get("conversation_id", ""),
                    "created_at": e.get("time", datetime.now().isoformat()),
                    "topic": e.get("topic", ""),
                    "action": e.get("action", ""),
                    "result": e.get("result", ""),
                    "text": text,
                },
            })
            ids.append(record_id)
        self.mem_events_store.insert_batch(to_insert)
        logger.info(f"KnowledgeSearch: wrote {len(ids)} events")
        return ids

    def search_events_sync(self, query: str, top_k: int = 5, apply_decay: bool = True,
                           uid: str | None = None) -> list[dict[str, Any]]:
        """Semantic recall over mem_events via the shared hybrid chain. Decay is a ranking
        weight only — events are never cleaned up (plan C5)."""
        return self._hybrid_search(
            self.mem_events_store, query, top_k=top_k, apply_decay=apply_decay, uid=uid,
        )

    # ============== Conversation summaries (P3) — conv-scoped, append-only ==============

    def write_conv_summary_sync(self, text: str, uid: str | None, conversation_id: str,
                                turn_start: int, turn_end: int, topic: str = "") -> str:
        """Append one conversation-segment summary; returns its record id. Append-only, no TTL."""
        if not text or not self.mem_conv_summaries_store:
            return ""
        vec = self.dense_encoder.embed([text])[0]
        record_id = f"cs_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.mem_conv_summaries_store.insert_batch([{
            "id": record_id,
            "vector": vec,
            "metadata": {
                "type": "conv_summary",
                "uid": uid or "",
                "conversation_id": conversation_id or "",
                "turn_start": turn_start,
                "turn_end": turn_end,
                "topic": topic,
                "created_at": datetime.now().isoformat(),
                "text": text,
            },
        }])
        return record_id

    def search_conv_summaries_sync(self, query: str, uid: str | None = None,
                                   conversation_id: str | None = None, top_k: int = 5,
                                   exclude_ids: list[str] | None = None,
                                   apply_decay: bool = True) -> list[dict[str, Any]]:
        """Semantic recall over mem_conv_summaries, filtered to one conversation (plan §4.2)."""
        extra = {"conversation_id": conversation_id} if conversation_id else None
        return self._hybrid_search(
            self.mem_conv_summaries_store, query, top_k=top_k, apply_decay=apply_decay,
            uid=uid, extra_filters=extra, exclude_ids=set(exclude_ids) if exclude_ids else None,
        )

    def list_conv_summaries_sync(self, uid: str | None, conversation_id: str) -> list[dict[str, Any]]:
        """All summaries for one conversation (cheap listing for the recent window)."""
        if not self.mem_conv_summaries_store:
            return []
        rows = self.mem_conv_summaries_store.query(vector=self._embed(""), top_k=1000)
        return [
            r for r in rows
            if r.get("metadata", {}).get("uid") == uid
            and r.get("metadata", {}).get("conversation_id") == conversation_id
        ]

    # ============== Statistics ==============

    def get_stats(self) -> dict[str, int]:
        return {
            "mem_events": self.mem_events_store.get_collection_stats()["count"]
            if self.mem_events_store else 0,
            "mem_conv_summaries": self.mem_conv_summaries_store.get_collection_stats()["count"]
            if self.mem_conv_summaries_store else 0,
        }

    # ============== Helper Methods ==============

    def _bm25_rank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[dict[str, Any]]:
        """In-memory BM25 scoring over candidate texts; returns candidates sorted by BM25."""
        try:
            import jieba
            tokenize = lambda t: list(jieba.cut_for_search(t))
        except ImportError:
            tokenize = str.split

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return list(candidates)

        texts = [r.get("metadata", {}).get("text", "") for r in candidates]
        tokenized = [tokenize(t) for t in texts]
        tf_lists = [Counter(tokens) for tokens in tokenized]

        N = len(tokenized)
        avg_dl = sum(len(t) for t in tokenized) / max(N, 1)

        scored: list[tuple[dict, float]] = []
        for candidate, tf, tokens in zip(candidates, tf_lists, tokenized):
            dl = len(tokens)
            score = 0.0
            for term in query_tokens:
                df = sum(1 for tfl in tf_lists if term in tfl)
                if df == 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                tf_val = tf.get(term, 0)
                score += idf * (tf_val * (k1 + 1)) / (tf_val + k1 * (1 - b + b * dl / avg_dl))
            scored.append((candidate, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    def _rrf_fuse(
        self,
        vector_ranked: list[dict[str, Any]],
        bm25_ranked: list[dict[str, Any]],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of vector-ranked and BM25-ranked lists."""
        vector_rank = {r["id"]: i for i, r in enumerate(vector_ranked)}
        bm25_rank = {r["id"]: i for i, r in enumerate(bm25_ranked)}

        id_to_result = {r["id"]: r for r in vector_ranked + bm25_ranked}
        all_ids = set(id_to_result)

        rrf: dict[str, float] = {
            rid: 1.0 / (k + vector_rank.get(rid, len(vector_ranked)) + 1)
                 + 1.0 / (k + bm25_rank.get(rid, len(bm25_ranked)) + 1)
            for rid in all_ids
        }

        sorted_ids = sorted(all_ids, key=lambda x: rrf[x], reverse=True)[:top_k]
        results = []
        for rid in sorted_ids:
            r = dict(id_to_result[rid])
            r["score"] = rrf[rid]
            results.append(r)
        return results

    def _embed(self, text: str) -> list[float]:
        vectors = self.dense_encoder.embed([text])
        return vectors[0]

    def _apply_decay(
        self,
        results: list[dict[str, Any]],
        decay_factor: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Apply time decay to non-evergreen results."""
        now = datetime.now()
        for r in results:
            metadata = r.get("metadata", {})
            is_evergreen = metadata.get("is_evergreen", False)
            created_at_str = metadata.get("created_at")
            if not is_evergreen and created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    days = (now - created_at).days
                    r["score"] *= decay_factor ** (days / 30)
                except (ValueError, TypeError):
                    pass
        return sorted(results, key=lambda x: x["score"], reverse=True)
