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
        user_memory_store: ChromaDB collection for user memory.
        dense_encoder: Embedding encoder for vector search.
    """

    def __init__(
        self,
        dense_encoder: BaseEmbedding,
        user_memory_store: ChromaStore | None = None,
        settings: "Settings | None" = None,
    ):
        self.dense_encoder = dense_encoder
        self.user_memory_store = user_memory_store
        self._settings = settings
        self._reranker: "BaseReranker | None | bool" = None  # None=uninit, False=unavailable

    @classmethod
    def from_settings(cls, settings: "Settings", collection_suffix: str = "") -> "KnowledgeSearch":
        from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory

        user_memory_collection = f"user_memory{collection_suffix}"
        user_memory_store = ChromaStore(
            settings=settings,
            collection_name=user_memory_collection,
        )
        dense_encoder = EmbeddingFactory.create(settings)

        return cls(
            dense_encoder=dense_encoder,
            user_memory_store=user_memory_store,
            settings=settings,
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

    def search_user_memory_sync(
        self,
        query: str,
        top_k: int = 5,
        apply_decay: bool = True,
        uid: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search user memory using BM25 + vector → RRF → optional Rerank → time decay."""
        if not self.user_memory_store:
            return []

        # Step 1: Vector search — over-retrieve for downstream filtering
        vector = self._embed(query)
        candidates = self.user_memory_store.query(
            vector=vector, top_k=top_k * 4,
        )

        if uid:
            candidates = [r for r in candidates if r.get("metadata", {}).get("uid") == uid]

        # Phase 2 scope adjustment: narrow L2/L3 recall to the conversation when requested.
        if conversation_id:
            candidates = [r for r in candidates
                          if r.get("metadata", {}).get("conversation_id") == conversation_id]

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

    def write_user_memory_sync(
        self,
        memories: list[dict[str, Any]],
        similarity_threshold: float = 0.85,
        uid: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[int, int]:
        """Write user memories with deduplication (sync version)."""
        if not memories or not self.user_memory_store:
            return (0, 0)

        memories = [m for m in memories if m.get("confidence", 0) >= 0.7]
        if not memories:
            return (0, 0)

        texts = [m["text"] for m in memories]
        vectors = self.dense_encoder.embed(texts)

        existing_results = self.user_memory_store.query_batch(
            vectors, top_k=1, threshold=similarity_threshold
        )

        written_ids, skipped_ids = [], []
        to_insert = []

        for mem, vec, existing_list in zip(memories, vectors, existing_results):
            if existing_list:
                skipped_ids.append(existing_list[0]["id"])
            else:
                record_id = f"um_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
                to_insert.append({
                    "id": record_id,
                    "vector": vec,
                    "metadata": {
                        "type": mem.get("type", "factual"),
                        "confidence": mem.get("confidence", 0.7),
                        "is_evergreen": mem.get("is_evergreen", False),
                        "created_at": mem.get("created_at", datetime.now().isoformat()),
                        "text": mem["text"],
                        "uid": uid or "",
                        "conversation_id": conversation_id or "",
                    }
                })
                written_ids.append(record_id)

        if to_insert:
            self.user_memory_store.insert_batch(to_insert)

        logger.info(
            f"KnowledgeSearch: wrote {len(written_ids)} user memories, "
            f"skipped {len(skipped_ids)} duplicates"
        )
        return (len(written_ids), len(skipped_ids))

    async def write_user_memory(
        self,
        memories: list[dict[str, Any]],
        similarity_threshold: float = 0.85,
        uid: str | None = None,
    ) -> tuple[int, int]:
        """Write user memories with deduplication (async wrapper)."""
        return self.write_user_memory_sync(memories, similarity_threshold, uid=uid)

    async def cleanup_old_user_memory(self, max_age_days: int = 90) -> int:
        """Delete non-evergreen user memories older than max_age_days."""
        if not self.user_memory_store or max_age_days <= 0:
            return 0

        cutoff = datetime.now() - timedelta(days=max_age_days)
        all_memories = self.user_memory_store.query(
            vector=self._embed(""), top_k=10000,
            filters={"is_evergreen": False}
        )

        ids_to_delete = []
        for mem in all_memories:
            created_at_str = mem.get("metadata", {}).get("created_at", "")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at < cutoff:
                        ids_to_delete.append(mem["id"])
                except (ValueError, TypeError):
                    pass

        if ids_to_delete:
            self.user_memory_store.delete(ids=ids_to_delete)
            logger.info(f"KnowledgeSearch: cleaned up {len(ids_to_delete)} old user memories")

        return len(ids_to_delete)

    # ============== Statistics ==============

    def get_stats(self) -> dict[str, int]:
        return {
            "user_memory": self.user_memory_store.get_collection_stats()["count"]
            if self.user_memory_store else 0,
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
