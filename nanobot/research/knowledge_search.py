"""Knowledge Search — RAG for research knowledge loop.

This module provides knowledge retrieval and storage for the research knowledge loop:
- Claims: atomic facts extracted from research
- Insights: transferable patterns across domains
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from nanobot.rag.libs.embedding.base_embedding import BaseEmbedding
from nanobot.rag.libs.vector_store.chroma_store import ChromaStore

if TYPE_CHECKING:
    from nanobot.research.types import Claim, Insight
    from nanobot.rag.core.settings import Settings


@dataclass
class WriteClaimsResult:
    """Result of writing claims with deduplication."""
    claims_written: int
    claims_updated: int
    duplicates_skipped: int
    claim_ids: list[str]


class KnowledgeSearch:
    """Knowledge retrieval and storage for research insights.

    Provides a unified interface for:
    - Searching claims and insights (with time decay)
    - Writing claims and insights to separate ChromaDB collections
    - Deleting insights (for batch refinement)
    - User memory: conversation-derived knowledge stored separately

    Attributes:
        claim_store: ChromaDB collection for research claims.
        insight_store: ChromaDB collection for research insights.
        user_memory_store: ChromaDB collection for user memory (conversation-derived).
        dense_encoder: Embedding encoder for dense retrieval.
    """

    def __init__(
        self,
        claim_store: ChromaStore,
        insight_store: ChromaStore,
        dense_encoder: BaseEmbedding,
        user_memory_store: ChromaStore | None = None,
    ):
        self.claim_store = claim_store
        self.insight_store = insight_store
        self.dense_encoder = dense_encoder
        self.user_memory_store = user_memory_store

    @classmethod
    def from_settings(cls, settings: Settings, collection_suffix: str = "") -> "KnowledgeSearch":
        """Create KnowledgeSearch from settings.

        Args:
            settings: Application settings containing vector_store and embedding configuration.
            collection_suffix: Optional suffix for collection names (e.g., "_test").
        """
        from nanobot.rag.libs.embedding.embedding_factory import EmbeddingFactory

        # Create stores with distinct collection names
        claim_collection = f"research_claims{collection_suffix}"
        insight_collection = f"research_insights{collection_suffix}"
        user_memory_collection = f"user_memory{collection_suffix}"

        claim_store = ChromaStore(
            settings=settings,
            collection_name=claim_collection,
        )
        insight_store = ChromaStore(
            settings=settings,
            collection_name=insight_collection,
        )
        user_memory_store = ChromaStore(
            settings=settings,
            collection_name=user_memory_collection,
        )

        # Create embedding encoder
        dense_encoder = EmbeddingFactory.create(settings)

        return cls(
            claim_store=claim_store,
            insight_store=insight_store,
            dense_encoder=dense_encoder,
            user_memory_store=user_memory_store,
        )

    # ============== Retrieval Methods ==============

    def search_claims_sync(
        self,
        query: str,
        top_k: int = 5,
        apply_decay: bool = True,
        include_draft: bool = False,
    ) -> list[dict[str, Any]]:
        """Synchronous search for claims with optional time decay.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            apply_decay: Whether to apply time decay to non-evergreen results.
            include_draft: If False (default), only return research-sourced claims.
                           If True, include conversation-sourced claims (weighted lower in RRF).

        Returns:
            List of matching claims with scores and metadata.
        """
        vector = self._embed(query)
        filters = {}
        if not include_draft:
            filters["source"] = "research"
        results = self.claim_store.query(
            vector=vector, top_k=top_k * 2,
            filters=filters if filters else None,
        )

        if apply_decay and results:
            results = self._apply_decay(results)

        return results[:top_k]

    async def search_claims(
        self,
        query: str,
        top_k: int = 5,
        apply_decay: bool = True,
        include_draft: bool = False,
    ) -> list[dict[str, Any]]:
        """Search claims with optional time decay.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            apply_decay: Whether to apply time decay to non-evergreen results.
            include_draft: If False (default), only return research-sourced claims.
                           If True, include conversation-sourced claims.

        Returns:
            List of matching claims with scores and metadata.
        """
        return self.search_claims_sync(query, top_k, apply_decay, include_draft)

    def search_insights_sync(
        self,
        query: str,
        top_k: int = 3,
        maturity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous search for insights with optional maturity filter.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            maturity: Optional maturity filter ("candidate" or "confirmed").

        Returns:
            List of matching insights with scores and metadata.
        """
        vector = self._embed(query)
        filters = {"maturity": maturity} if maturity else None

        return self.insight_store.query(
            vector=vector,
            top_k=top_k,
            filters=filters,
        )

    async def search_insights(
        self,
        query: str,
        top_k: int = 3,
        maturity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search insights with optional maturity filter.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            maturity: Optional maturity filter ("candidate" or "confirmed").

        Returns:
            List of matching insights with scores and metadata.
        """
        return self.search_insights_sync(query, top_k, maturity)

    def search_all_sync(
        self,
        query: str,
        top_k_claims: int = 5,
        top_k_insights: int = 3,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Synchronous search for both claims and insights.

        Args:
            query: Search query string.
            top_k_claims: Number of claim results to return.
            top_k_insights: Number of insight results to return.

        Returns:
            Tuple of (claims, insights) lists.
        """
        claims = self.search_claims_sync(query, top_k=top_k_claims)
        insights = self.search_insights_sync(query, top_k=top_k_insights)
        return claims, insights

    async def search_all(
        self,
        query: str,
        top_k_claims: int = 5,
        top_k_insights: int = 3,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Search both claims and insights separately.

        Args:
            query: Search query string.
            top_k_claims: Number of claim results to return.
            top_k_insights: Number of insight results to return.

        Returns:
            Tuple of (claims, insights) lists.
        """
        return self.search_all_sync(query, top_k_claims, top_k_insights)

    # ============== Storage Methods ==============

    async def write_claims(
        self,
        claims: list["Claim"],
        source: str = "research",
        verified: bool = True,
        similarity_threshold: float = 0.85,
    ) -> WriteClaimsResult:
        """Write claims with automatic deduplication (batch processing).

        This method implements fixed 3 I/O operations regardless of claim count:
        1. Batch embed all claims
        2. Batch query for similar existing claims
        3. Batch write (update + insert in parallel)

        Args:
            claims: List of Claim objects to write.
            source: Source type ("research" or "conversation").
            verified: Whether the claims have been verified.
            similarity_threshold: Threshold for considering claims as duplicates.

        Returns:
            WriteClaimsResult with counts and IDs.
        """
        if not claims:
            return WriteClaimsResult(0, 0, 0, [])

        # Filter low confidence claims
        claims = [c for c in claims if c.confidence >= 0.7]
        if not claims:
            return WriteClaimsResult(0, 0, 0, [])

        written_ids, updated_ids, skipped_ids = [], [], []

        # I/O 1: Batch embed all claims
        claim_texts = [c.claim for c in claims]
        vectors = self.dense_encoder.embed(claim_texts)

        # I/O 2: Batch query for similar existing claims
        existing_results = self.claim_store.query_batch(
            vectors, top_k=1, threshold=similarity_threshold
        )

        # Classify claims into update/insert/skip
        to_update, to_insert = [], []

        for i, (claim, vec, existing_list) in enumerate(zip(claims, vectors, existing_results)):
            if existing_list:
                existing = existing_list[0]
                existing_conf = existing.get("metadata", {}).get("confidence", 0)

                if claim.confidence > existing_conf:
                    # Update: new claim has higher confidence
                    to_update.append((
                        existing["id"],
                        {
                            "type": "research_claim",
                            "claim_type": claim.type,
                            "confidence": claim.confidence,
                            "is_evergreen": claim.is_evergreen,
                            "source_urls": claim.source_urls,
                            "evidence_ids": claim.evidence_ids,
                            "source": source,
                            "verified": verified,
                            "created_at": claim.created_at.isoformat(),
                            "text": claim.claim,
                        }
                    ))
                    updated_ids.append(existing["id"])
                else:
                    # Skip: duplicate with equal or higher confidence
                    skipped_ids.append(existing["id"])
            else:
                # Insert: new claim
                record_id = f"claim_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
                to_insert.append({
                    "id": record_id,
                    "vector": vec,
                    "metadata": {
                        "type": "research_claim",
                        "claim_type": claim.type,
                        "confidence": claim.confidence,
                        "is_evergreen": claim.is_evergreen,
                        "source_urls": claim.source_urls,
                        "evidence_ids": claim.evidence_ids,
                        "source": source,
                        "verified": verified,
                        "created_at": claim.created_at.isoformat(),
                        "text": claim.claim,
                    }
                })

        # I/O 3: Batch write (update + insert in parallel)
        tasks = []
        if to_update:
            tasks.append(self._update_batch_async(to_update))
        if to_insert:
            tasks.append(self._insert_batch_async(to_insert))

        if tasks:
            results = await asyncio.gather(*tasks)
            # Extract inserted IDs from results
            if to_insert:
                insert_result = results[-1] if to_update else results[0]
                if isinstance(insert_result, list):
                    written_ids = insert_result

        logger.info(
            f"KnowledgeSearch: wrote {len(written_ids)} claims, "
            f"updated {len(updated_ids)}, skipped {len(skipped_ids)} (source={source})"
        )

        return WriteClaimsResult(
            claims_written=len(written_ids),
            claims_updated=len(updated_ids),
            duplicates_skipped=len(skipped_ids),
            claim_ids=written_ids + updated_ids,
        )

    async def _update_batch_async(self, items: list[tuple]) -> None:
        """Async wrapper for update_batch."""
        self.claim_store.update_batch(items)

    async def _insert_batch_async(self, records: list[dict]) -> list[str]:
        """Async wrapper for insert_batch."""
        return self.claim_store.insert_batch(records)

    async def write_insights(
        self,
        insights: list["Insight"],
        maturity: str = "candidate",
    ) -> list[str]:
        """Write insights to the insights collection.

        Args:
            insights: List of Insight objects to write.
            maturity: Maturity level ("candidate" or "confirmed").

        Returns:
            List of written insight IDs.
        """
        if not insights:
            return []

        # Filter low confidence insights
        insights = [i for i in insights if i.confidence >= 0.6]

        records = []
        for i in insights:
            record_id = f"insight_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            records.append({
                "id": record_id,
                "vector": self._embed(i.insight),
                "metadata": {
                    "type": "research_insight",
                    "maturity": maturity,
                    "supporting_claim_ids": i.supporting_claim_ids,  # Claim IDs
                    "applicable_domains": i.applicable_domains,
                    "confidence": i.confidence,
                    "is_evergreen": i.is_evergreen,
                    "created_at": i.created_at.isoformat(),
                    "text": i.insight,
                },
            })

        if records:
            self.insight_store.upsert(records)
            logger.info(f"KnowledgeSearch: wrote {len(records)} insights (maturity={maturity})")

        return [r["id"] for r in records]

    async def write_insights_with_backlink(
        self,
        insights: list["Insight"],
        maturity: str = "candidate",
    ) -> list[str]:
        """Write insights and update claims with backlinks (atomic-safe order).

        Order: Update claim backlinks FIRST, then write insight.
        This ensures that if crash happens, claim has extra backlink (detectable)
        rather than insight exists without backlink (silent).

        Args:
            insights: List of Insight objects to write.
            maturity: Maturity level ("candidate" or "confirmed").

        Returns:
            List of written insight IDs.
        """
        if not insights:
            return []

        # Filter low confidence insights
        insights = [i for i in insights if i.confidence >= 0.6]
        if not insights:
            return []

        insight_ids = []

        for insight in insights:
            insight_id = f"insight_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            claim_ids = self._parse_claim_ids(insight.supporting_claim_ids)

            # Step 1: Update claim backlinks FIRST (batch read + parallel update)
            if claim_ids:
                # Batch read existing claim data
                claims_data = await asyncio.gather(*[
                    self._get_claim_async(cid) for cid in claim_ids
                ])

                # Prepare backlink updates
                to_update = []
                for cid, data in zip(claim_ids, claims_data):
                    if data:
                        existing = data.get("metadata", {}).get("used_in_insight_ids", "")
                        if existing:
                            updated = f"{existing},{insight_id}"
                        else:
                            updated = insight_id
                        to_update.append((cid, {"used_in_insight_ids": updated}))

                # Parallel batch update
                if to_update:
                    await self._update_batch_async(to_update)

            # Step 2: Write insight AFTER backlink update
            records = [{
                "id": insight_id,
                "vector": self._embed(insight.insight),
                "metadata": {
                    "type": "research_insight",
                    "maturity": maturity,
                    "supporting_claim_ids": insight.supporting_claim_ids,
                    "applicable_domains": insight.applicable_domains,
                    "confidence": insight.confidence,
                    "is_evergreen": insight.is_evergreen,
                    "created_at": insight.created_at.isoformat(),
                    "text": insight.insight,
                },
            }]
            self.insight_store.upsert(records)
            insight_ids.append(insight_id)

        logger.info(f"KnowledgeSearch: wrote {len(insight_ids)} insights with backlinks")

        return insight_ids

    async def cleanup_draft_claims(self, max_age_days: int = 90) -> int:
        """Delete conversation claims older than max_age_days that haven't been verified.

        Args:
            max_age_days: Delete claims older than this many days (default 90).

        Returns:
            Number of claims deleted.
        """
        if max_age_days <= 0:
            return 0

        cutoff = datetime.now() - timedelta(days=max_age_days)

        # Query all draft claims
        draft_claims = self.claim_store.query_by_metadata(
            filters={"source": "conversation", "verified": False}
        )

        # Collect IDs to delete
        ids_to_delete = []
        for claim in draft_claims:
            created_at_str = claim.get("metadata", {}).get("created_at", "")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at < cutoff:
                        ids_to_delete.append(claim["id"])
                except (ValueError, TypeError):
                    pass

        # Batch delete
        if ids_to_delete:
            self.claim_store.delete(ids=ids_to_delete)
            logger.info(f"KnowledgeSearch: cleaned up {len(ids_to_delete)} draft claims")

        return len(ids_to_delete)

    def _parse_claim_ids(self, raw: str | list) -> list[str]:
        """Parse claim IDs from comma-separated string or list."""
        if isinstance(raw, list):
            return [str(cid).strip() for cid in raw if cid]
        if not raw:
            return []
        return [cid.strip() for cid in str(raw).split(",") if cid.strip()]

    async def _get_claim_async(self, claim_id: str) -> dict | None:
        """Async wrapper for getting a single claim."""
        results = self.claim_store.get_by_ids([claim_id])
        return results[0] if results else None

    async def delete_insights(self, ids: list[str]) -> None:
        """Delete insights by IDs.

        Args:
            ids: List of insight IDs to delete.
        """
        if not ids:
            return
        self.insight_store.delete(ids=ids)
        logger.info(f"KnowledgeSearch: deleted {len(ids)} insights")

    # ============== Helper Methods ==============

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        vectors = self.dense_encoder.embed([text])
        return vectors[0]

    def _apply_decay(
        self,
        results: list[dict[str, Any]],
        decay_factor: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Apply time decay to non-evergreen results.

        Results older than 30 days will have their scores multiplied by decay_factor.
        """
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

    # ============== Statistics ==============

    def get_stats(self) -> dict[str, int]:
        """Get statistics about stored knowledge."""
        return {
            "claims": self.claim_store.get_collection_stats()["count"],
            "insights": self.insight_store.get_collection_stats()["count"],
            "user_memory": self.user_memory_store.get_collection_stats()["count"] if self.user_memory_store else 0,
        }

    # ============== User Memory Methods ==============

    def search_user_memory_sync(
        self,
        query: str,
        top_k: int = 5,
        apply_decay: bool = True,
    ) -> list[dict[str, Any]]:
        """Search user memory for conversation-derived knowledge.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            apply_decay: Whether to apply time decay to non-evergreen results.

        Returns:
            List of matching user memories with scores and metadata.
        """
        if not self.user_memory_store:
            return []

        vector = self._embed(query)
        results = self.user_memory_store.query(
            vector=vector, top_k=top_k * 2,
        )

        if apply_decay and results:
            results = self._apply_decay(results)

        return results[:top_k]

    def write_user_memory_sync(
        self,
        memories: list[dict[str, Any]],
        similarity_threshold: float = 0.85,
    ) -> tuple[int, int]:
        """Write user memories with deduplication (sync version).

        This sync version is used by _raw_archive() fallback path.

        Args:
            memories: List of memory dicts with keys: text, type, confidence, is_evergreen, created_at.
            similarity_threshold: Threshold for considering memories as duplicates.

        Returns:
            Tuple of (written_count, skipped_count).
        """
        if not memories or not self.user_memory_store:
            return (0, 0)

        # Filter low confidence
        memories = [m for m in memories if m.get("confidence", 0) >= 0.7]
        if not memories:
            return (0, 0)

        written_ids, skipped_ids = [], []

        # Batch embed
        texts = [m["text"] for m in memories]
        vectors = self.dense_encoder.embed(texts)

        # Batch query for similar existing
        existing_results = self.user_memory_store.query_batch(
            vectors, top_k=1, threshold=similarity_threshold
        )

        to_insert = []
        for i, (mem, vec, existing_list) in enumerate(zip(memories, vectors, existing_results)):
            if existing_list:
                # Skip duplicate
                skipped_ids.append(existing_list[0]["id"])
            else:
                # Insert new
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
                    }
                })
                written_ids.append(record_id)

        if to_insert:
            self.user_memory_store.insert_batch(to_insert)

        logger.info(
            f"KnowledgeSearch: wrote {len(written_ids)} user memories, skipped {len(skipped_ids)} duplicates"
        )

        return (len(written_ids), len(skipped_ids))

    async def write_user_memory(
        self,
        memories: list[dict[str, Any]],
        similarity_threshold: float = 0.85,
    ) -> tuple[int, int]:
        """Write user memories with deduplication (async wrapper).

        Args:
            memories: List of memory dicts with keys: text, type, confidence, is_evergreen, created_at.
            similarity_threshold: Threshold for considering memories as duplicates.

        Returns:
            Tuple of (written_count, skipped_count).
        """
        return self.write_user_memory_sync(memories, similarity_threshold)

    async def cleanup_old_user_memory(self, max_age_days: int = 90) -> int:
        """Delete user memories older than max_age_days that aren't evergreen.

        Args:
            max_age_days: Delete memories older than this many days (default 90).

        Returns:
            Number of memories deleted.
        """
        if not self.user_memory_store or max_age_days <= 0:
            return 0

        cutoff = datetime.now() - timedelta(days=max_age_days)

        # Query all non-evergreen memories
        all_memories = self.user_memory_store.query(
            vector=self._embed(""), top_k=10000,  # Get all
            filters={"is_evergreen": False}
        )

        # Collect IDs to delete
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

        # Batch delete
        if ids_to_delete:
            self.user_memory_store.delete(ids=ids_to_delete)
            logger.info(f"KnowledgeSearch: cleaned up {len(ids_to_delete)} old user memories")

        return len(ids_to_delete)
