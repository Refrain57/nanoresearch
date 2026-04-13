"""Core layer Reranker orchestrating libs.reranker backends with fallback support.

This module implements the CoreReranker class that:
1. Integrates with libs.reranker (LLM, CrossEncoder, None) via RerankerFactory
2. Provides graceful fallback when backend fails or times out
3. Converts RetrievalResult to/from reranker input/output format
4. Supports TraceContext for observability

Design Principles:
- Pluggable: Uses RerankerFactory to instantiate configured backend
- Config-Driven: Reads rerank settings from settings.yaml
- Graceful Fallback: Returns original order on backend failure
- Observable: TraceContext integration for debugging
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from nanobot.rag.core.types import RetrievalResult
from nanobot.rag.libs.reranker.base_reranker import BaseReranker, NoneReranker
from nanobot.rag.libs.reranker.reranker_factory import RerankerFactory

if TYPE_CHECKING:
    from nanobot.rag.core.settings import Settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """Raised when reranking fails."""


@dataclass
class RerankConfig:
    """Configuration for CoreReranker.
    
    Attributes:
        enabled: Whether reranking is enabled
        top_k: Number of results to return after reranking
        timeout: Timeout for reranker backend (seconds)
        fallback_on_error: Whether to return original order on error
    """
    enabled: bool = True
    top_k: int = 5
    timeout: float = 30.0
    fallback_on_error: bool = True


@dataclass
class RerankResult:
    """Result of a rerank operation.
    
    Attributes:
        results: Reranked list of RetrievalResults
        used_fallback: Whether fallback was used due to backend failure
        fallback_reason: Reason for fallback (if applicable)
        reranker_type: Type of reranker used ('llm', 'cross_encoder', 'none')
        original_order: Original results before reranking (for debugging)
    """
    results: List[RetrievalResult] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    reranker_type: str = "none"
    original_order: Optional[List[RetrievalResult]] = None


class CoreReranker:
    """Core layer Reranker with fallback support.
    
    This class wraps libs.reranker implementations and provides:
    1. Type conversion between RetrievalResult and reranker dict format
    2. Graceful fallback when backend fails
    3. Configuration-driven backend selection
    4. TraceContext integration
    
    Design Principles Applied:
    - Pluggable: Backend via RerankerFactory
    - Config-Driven: All parameters from settings
    - Fallback: Returns original order on failure
    - Observable: TraceContext support
    
    Example:
        >>> from nanobot.rag.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> reranker = CoreReranker(settings)
        >>> results = [RetrievalResult(chunk_id="1", score=0.8, text="...", metadata={})]
        >>> reranked = reranker.rerank("query", results)
        >>> print(reranked.results)
    """
    
    def __init__(
        self,
        settings: Settings,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankConfig] = None,
    ) -> None:
        """Initialize CoreReranker.
        
        Args:
            settings: Application settings containing rerank configuration.
            reranker: Optional reranker backend. If None, creates via RerankerFactory.
            config: Optional RerankConfig. If None, extracts from settings.
        """
        self.settings = settings
        
        # Extract config from settings or use provided
        if config is not None:
            self.config = config
        else:
            self.config = self._extract_config(settings)
        
        # Initialize reranker backend
        if reranker is not None:
            self._reranker = reranker
        elif not self.config.enabled:
            self._reranker = NoneReranker(settings=settings)
        else:
            try:
                self._reranker = RerankerFactory.create(settings)
            except Exception as e:
                logger.warning(f"Failed to create reranker, using NoneReranker: {e}")
                self._reranker = NoneReranker(settings=settings)
        
        # Determine reranker type for result reporting
        self._reranker_type = self._get_reranker_type()
    
    def _extract_config(self, settings: Settings) -> RerankConfig:
        """Extract RerankConfig from settings.
        
        Args:
            settings: Application settings.
            
        Returns:
            RerankConfig with values from settings.
        """
        try:
            rerank_settings = settings.rerank
            return RerankConfig(
                enabled=bool(rerank_settings.enabled) if rerank_settings else False,
                top_k=int(rerank_settings.top_k) if rerank_settings and hasattr(rerank_settings, 'top_k') else 5,
                timeout=float(getattr(rerank_settings, 'timeout', 30.0)) if rerank_settings else 30.0,
                fallback_on_error=True,
            )
        except AttributeError:
            logger.warning("Missing rerank configuration, using defaults (disabled)")
            return RerankConfig(enabled=False)
    
    def _get_reranker_type(self) -> str:
        """Get the type name of the current reranker backend.
        
        Returns:
            String identifier for the reranker type.
        """
        class_name = self._reranker.__class__.__name__
        if "LLM" in class_name:
            return "llm"
        elif "CrossEncoder" in class_name:
            return "cross_encoder"
        elif "None" in class_name:
            return "none"
        else:
            return class_name.lower()
    
    def _results_to_candidates(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Convert RetrievalResults to reranker candidate format.
        
        Args:
            results: List of RetrievalResult objects.
            
        Returns:
            List of dicts suitable for reranker input.
        """
        candidates = []
        for result in results:
            candidates.append({
                "id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata.copy(),
            })
        return candidates
    
    def _candidates_to_results(
        self,
        candidates: List[Dict[str, Any]],
        original_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Convert reranked candidates back to RetrievalResults.
        
        Args:
            candidates: Reranked candidates from reranker.
            original_results: Original results for reference.
            
        Returns:
            List of RetrievalResult in reranked order.
        """
        # Build lookup from original results
        id_to_original = {r.chunk_id: r for r in original_results}
        
        results = []
        for candidate in candidates:
            chunk_id = candidate["id"]
            
            # Get original result or build new one
            if chunk_id in id_to_original:
                original = id_to_original[chunk_id]
                # Create new result with updated score
                rerank_score = candidate.get("rerank_score", candidate.get("score", 0.0))
                results.append(RetrievalResult(
                    chunk_id=original.chunk_id,
                    score=rerank_score,
                    text=original.text,
                    metadata={
                        **original.metadata,
                        "original_score": original.score,
                        "rerank_score": rerank_score,
                        "reranked": True,
                    },
                ))
            else:
                # Candidate not in original - build from candidate data
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    score=candidate.get("rerank_score", candidate.get("score", 0.0)),
                    text=candidate.get("text", ""),
                    metadata=candidate.get("metadata", {}),
                ))
        
        return results
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> RerankResult:
        """Rerank retrieval results using configured backend.
        
        Args:
            query: The user query string.
            results: List of RetrievalResult objects to rerank.
            top_k: Number of results to return. If None, uses config.top_k.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters passed to reranker backend.
            
        Returns:
            RerankResult containing reranked results and metadata.
        """
        effective_top_k = top_k if top_k is not None else self.config.top_k
        
        # Early return for empty or single results
        if not results:
            return RerankResult(
                results=[],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
        
        if len(results) == 1:
            return RerankResult(
                results=results[:],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
        
        # If reranking disabled, return top_k results in original order
        if not self.config.enabled or isinstance(self._reranker, NoneReranker):
            return RerankResult(
                results=results[:effective_top_k],
                used_fallback=False,
                reranker_type="none",
                original_order=results[:],
            )
        
        # Convert to reranker input format
        candidates = self._results_to_candidates(results)
        
        # Attempt reranking
        try:
            logger.debug(f"Reranking {len(candidates)} candidates with {self._reranker_type}")
            _t0 = time.monotonic()
            reranked_candidates = self._reranker.rerank(
                query=query,
                candidates=candidates,
                trace=trace,
                **kwargs,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            # Convert back to RetrievalResult
            reranked_results = self._candidates_to_results(reranked_candidates, results)
            
            # Apply top_k limit
            final_results = reranked_results[:effective_top_k]
            
            logger.info(f"Reranking complete: {len(final_results)} results returned")
            
            if trace is not None:
                trace.record_stage("rerank", {
                    "method": self._reranker_type,
                    "provider": self._reranker_type,
                    "input_count": len(candidates),
                    "output_count": len(final_results),
                    "chunks": [
                        {
                            "chunk_id": r.chunk_id,
                            "score": round(r.score, 4),
                            "text": r.text or "",
                            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                        }
                        for r in final_results
                    ],
                }, elapsed_ms=_elapsed)
            
            return RerankResult(
                results=final_results,
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=results[:],
            )
            
        except Exception as e:
            logger.warning(f"Reranking failed, using fallback: {e}")
            
            if self.config.fallback_on_error:
                # Return original order as fallback
                fallback_results = []
                for result in results[:effective_top_k]:
                    fallback_results.append(RetrievalResult(
                        chunk_id=result.chunk_id,
                        score=result.score,
                        text=result.text,
                        metadata={
                            **result.metadata,
                            "reranked": False,
                            "rerank_fallback": True,
                        },
                    ))
                
                return RerankResult(
                    results=fallback_results,
                    used_fallback=True,
                    fallback_reason=str(e),
                    reranker_type=self._reranker_type,
                    original_order=results[:],
                )
            else:
                raise RerankError(f"Reranking failed and fallback disabled: {e}") from e
    
    @property
    def reranker_type(self) -> str:
        """Get the type of the current reranker backend."""
        return self._reranker_type
    
    @property
    def is_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return self.config.enabled and not isinstance(self._reranker, NoneReranker)


def create_core_reranker(
    settings: Settings,
    reranker: Optional[BaseReranker] = None,
) -> CoreReranker:
    """Factory function to create a CoreReranker instance.

    Args:
        settings: Application settings.
        reranker: Optional reranker backend override.

    Returns:
        Configured CoreReranker instance.
    """
    return CoreReranker(settings=settings, reranker=reranker)


# ============================================================================
# Structure-Aware Reranker
# ============================================================================


class StructureAwareReranker:
    """Reranker that considers document structure metadata.

    This reranker enhances the base reranker by weighting results based on:
    1. Section level: Higher-level sections (H1, H2) get higher weight for
       overview/summary queries
    2. Content type: Code blocks get higher weight for code-related queries

    This enables structure-aware retrieval that prioritizes results based on
    their position in the document hierarchy.

    Example:
        >>> base_reranker = CoreReranker(settings)
        >>> structure_reranker = StructureAwareReranker(base_reranker, settings)
        >>> result = structure_reranker.rerank("How does RAG work?", candidates)
    """

    # Default weights for section levels (1-6, lower number = higher level)
    DEFAULT_SECTION_WEIGHTS = {
        1: 1.5,   # H1 - Main title/overview
        2: 1.3,   # H2 - Major sections
        3: 1.0,   # H3 - Subsections (baseline)
        4: 0.8,   # H4 - Details
        5: 0.7,   # H5 - Fine details
        6: 0.6,   # H6 - Very fine details
        0: 1.0,   # No heading - baseline
    }

    # Default weights for content types
    DEFAULT_TYPE_WEIGHTS = {
        "text": 1.0,
        "code": 1.5,    # Code gets boost for technical queries
        "table": 1.3,   # Tables often contain structured info
        "list": 1.1,    # Lists are often summaries
    }

    # Keywords that indicate code-related queries
    CODE_KEYWORDS = [
        "代码", "code", "函数", "function", "实现", "implement",
        "示例", "example", "sample", "用法", "usage", "api",
        "python", "javascript", "java", "golang", "rust", "sql",
    ]

    # Keywords that indicate overview/summary queries
    OVERVIEW_KEYWORDS = [
        "概述", "overview", "简介", "introduction", "总结", "summary",
        "什么是", "what is", "介绍", "intro", "概览", "guide",
        "原理", "原理是什么", "how does it work", "架构", "architecture",
    ]

    def __init__(
        self,
        base_reranker: CoreReranker,
        settings: Optional[Settings] = None,
        section_weights: Optional[Dict[int, float]] = None,
        type_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize StructureAwareReranker.

        Args:
            base_reranker: The underlying CoreReranker instance.
            settings: Application settings (optional).
            section_weights: Custom section level weights (optional).
            type_weights: Custom content type weights (optional).
        """
        self.base_reranker = base_reranker
        self.settings = settings
        self.section_weights = section_weights or self.DEFAULT_SECTION_WEIGHTS
        self.type_weights = type_weights or self.DEFAULT_TYPE_WEIGHTS

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> RerankResult:
        """Rerank results with structure-aware weighting.

        Args:
            query: The user query string.
            results: List of RetrievalResult objects to rerank.
            top_k: Number of results to return.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            RerankResult with structure-weighted scores.
        """
        # Step 1: Get base reranking results
        base_result = self.base_reranker.rerank(
            query=query,
            results=results,
            top_k=len(results),  # Don't limit yet
            trace=trace,
            **kwargs,
        )

        # Step 2: Apply structure-aware weighting
        weighted_results = self._apply_structure_weights(
            query=query,
            results=base_result.results,
        )

        # Step 3: Sort by weighted score and limit
        sorted_results = sorted(
            weighted_results,
            key=lambda r: r.score,
            reverse=True,
        )

        effective_top_k = top_k if top_k is not None else self.base_reranker.config.top_k
        final_results = sorted_results[:effective_top_k]

        # Add metadata about structure weighting
        for result in final_results:
            result.metadata["structure_weighted"] = True

        return RerankResult(
            results=final_results,
            used_fallback=base_result.used_fallback,
            fallback_reason=base_result.fallback_reason,
            reranker_type=f"structure_aware_{base_result.reranker_type}",
            original_order=base_result.original_order,
        )

    def _apply_structure_weights(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Apply structure-aware weighting to results.

        Args:
            query: The user query.
            results: Results from base reranker.

        Returns:
            Results with adjusted scores.
        """
        # Detect query intent
        is_code_query = self._is_code_query(query)
        is_overview_query = self._is_overview_query(query)

        weighted_results = []
        for result in results:
            # Get structure metadata
            section_level = result.metadata.get("section_level", 3)
            content_type = result.metadata.get("content_type", "text")

            # Calculate weight multiplier
            weight = 1.0

            # Section level weighting
            if is_overview_query:
                # Overview queries prefer higher-level sections
                weight *= self.section_weights.get(section_level, 1.0)

            # Content type weighting
            if is_code_query:
                # Code queries prefer code blocks
                weight *= self.type_weights.get(content_type, 1.0)
                if content_type == "code":
                    weight *= 1.3  # Extra boost for code in code queries

            # Apply weight to score
            new_result = RetrievalResult(
                chunk_id=result.chunk_id,
                score=result.score * weight,
                text=result.text,
                metadata={
                    **result.metadata,
                    "structure_weight": weight,
                    "section_level": section_level,
                    "content_type": content_type,
                },
            )
            weighted_results.append(new_result)

        return weighted_results

    def _is_code_query(self, query: str) -> bool:
        """Detect if query is about code/implementation.

        Args:
            query: User query string.

        Returns:
            True if query appears to be code-related.
        """
        query_lower = query.lower()
        matches = sum(1 for kw in self.CODE_KEYWORDS if kw in query_lower)
        return matches >= 1

    def _is_overview_query(self, query: str) -> bool:
        """Detect if query is asking for overview/summary.

        Args:
            query: User query string.

        Returns:
            True if query appears to be asking for overview.
        """
        query_lower = query.lower()
        matches = sum(1 for kw in self.OVERVIEW_KEYWORDS if kw in query_lower)
        return matches >= 1

    @property
    def reranker_type(self) -> str:
        """Get the type of the reranker."""
        return f"structure_aware_{self.base_reranker.reranker_type}"

    @property
    def is_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return self.base_reranker.is_enabled


def create_structure_aware_reranker(
    settings: Settings,
    reranker: Optional[BaseReranker] = None,
) -> StructureAwareReranker:
    """Factory function to create a StructureAwareReranker instance.

    Args:
        settings: Application settings.
        reranker: Optional reranker backend override.

    Returns:
        Configured StructureAwareReranker instance.
    """
    base_reranker = CoreReranker(settings=settings, reranker=reranker)
    return StructureAwareReranker(base_reranker=base_reranker, settings=settings)
