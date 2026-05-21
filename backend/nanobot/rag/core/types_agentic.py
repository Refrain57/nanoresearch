"""Agentic RAG data types.

This module defines data structures for Agentic RAG capabilities:
- FusionResult: Results from fusion operations
- VerificationResult: Results from answer verification
- SearchSession: Multi-turn search session state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FusionResult:
    """Result of fusion operation.

    Attributes:
        results: Fused and ranked retrieval results
        method: Fusion method used ("rrf", "weighted_rrf", "interleave")
        weights: Optional weights applied to each source
        rrf_k: RRF k parameter (for RRF method)
        input_counts: Count of results from each input source
        unique_chunks: Number of unique chunks across all sources
    """
    results: List[Dict[str, Any]]
    method: str = "rrf"
    weights: Optional[Dict[str, float]] = None
    rrf_k: Optional[int] = None
    input_counts: Dict[str, int] = field(default_factory=dict)
    unique_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fusion_method": self.method,
            "weights": self.weights,
            "rrf_k": self.rrf_k,
            "input_counts": self.input_counts,
            "unique_chunks": self.unique_chunks,
            "results_count": len(self.results),
            "results": self.results,
        }


@dataclass
class VerificationResult:
    """Result of answer verification.

    Attributes:
        query: The original query
        total_results: Number of results checked
        answered: Whether the results answer the query
        confidence: Confidence score (0.0 - 1.0)
        summary: Brief explanation of the verdict
        per_result: Per-result relevance analysis
        suggestions: Suggestions for improvement (refined queries, etc.)
    """
    query: str
    total_results: int
    answered: bool
    confidence: float
    summary: str
    per_result: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "total_results": self.total_results,
            "verification": {
                "answered": self.answered,
                "confidence": round(self.confidence, 2),
                "summary": self.summary,
            },
            "per_result": self.per_result,
            "suggestions": self.suggestions,
        }


@dataclass
class QueryPlan:
    """Result of query planning.

    Attributes:
        original_query: The original user query
        suggested_queries: Alternative/refined query suggestions
        suggested_strategy: "dense", "sparse", or "hybrid"
        strategy_reasoning: Why this strategy is recommended
        suggested_filters: Metadata filters to apply
        estimated_difficulty: "easy", "medium", or "hard"
        decomposition: Query decomposition details
    """
    original_query: str
    suggested_queries: List[str] = field(default_factory=list)
    suggested_strategy: str = "hybrid"
    strategy_reasoning: str = ""
    suggested_filters: Dict[str, Any] = field(default_factory=dict)
    estimated_difficulty: str = "medium"
    decomposition: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_query": self.original_query,
            "suggested_queries": self.suggested_queries,
            "suggested_strategy": self.suggested_strategy,
            "strategy_reasoning": self.strategy_reasoning,
            "suggested_filters": self.suggested_filters,
            "estimated_difficulty": self.estimated_difficulty,
            "decomposition": self.decomposition,
        }


@dataclass
class SearchSession:
    """Multi-turn search session state.

    Attributes:
        session_id: Unique session identifier
        created_at: Session creation timestamp
        initial_query: The first query in the session
        collection: Target collection name
        current_query: Current query being processed
        query_history: History of all queries
        refined_queries: Queries generated through refinement
        retrieval_results: Raw retrieval results by type
        fusion_results: Fused results
        reranked_results: Reranked results
        verified: Whether results have been verified
        citations: Generated citations
        metadata: Additional session metadata
    """
    session_id: str
    created_at: datetime
    initial_query: str
    collection: str = "default"
    current_query: str = ""
    query_history: List[str] = field(default_factory=list)
    refined_queries: List[str] = field(default_factory=list)
    retrieval_results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    all_results: List[Dict[str, Any]] = field(default_factory=list)  # Accumulated results across queries
    fusion_results: Optional[List[Dict[str, Any]]] = None
    reranked_results: Optional[List[Dict[str, Any]]] = None
    verified: bool = False
    verification_result: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived fields."""
        if not self.current_query:
            self.current_query = self.initial_query
        if self.initial_query not in self.query_history:
            self.query_history.append(self.initial_query)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "initial_query": self.initial_query,
            "collection": self.collection,
            "current_query": self.current_query,
            "query_history": self.query_history,
            "refined_queries": self.refined_queries,
            "retrieval_results": self.retrieval_results,
            "all_results": self.all_results,
            "fusion_results": self.fusion_results,
            "reranked_results": self.reranked_results,
            "verified": self.verified,
            "verification_result": self.verification_result,
            "citations": self.citations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSession":
        """Create SearchSession from dictionary.

        Args:
            data: Dictionary with session data

        Returns:
            SearchSession instance
        """
        # Handle datetime parsing
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()

        return cls(
            session_id=data["session_id"],
            created_at=created_at,
            initial_query=data["initial_query"],
            collection=data.get("collection", "default"),
            current_query=data.get("current_query", data["initial_query"]),
            query_history=data.get("query_history", []),
            refined_queries=data.get("refined_queries", []),
            retrieval_results=data.get("retrieval_results", {}),
            all_results=data.get("all_results", []),
            fusion_results=data.get("fusion_results"),
            reranked_results=data.get("reranked_results"),
            verified=data.get("verified", False),
            verification_result=data.get("verification_result"),
            citations=data.get("citations", []),
            metadata=data.get("metadata", {}),
        )


def retrieval_result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert RetrievalResult to dictionary.

    Args:
        result: RetrievalResult instance or dict

    Returns:
        Dictionary representation
    """
    if isinstance(result, dict):
        return result

    # Assuming RetrievalResult from types.py
    return {
        "chunk_id": getattr(result, "chunk_id", ""),
        "score": getattr(result, "score", 0.0),
        "text": getattr(result, "text", ""),
        "metadata": getattr(result, "metadata", {}),
    }


def dict_to_retrieval_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure dictionary has required retrieval result fields.

    Args:
        data: Input dictionary

    Returns:
        Dictionary with required fields
    """
    return {
        "chunk_id": data.get("chunk_id", ""),
        "score": data.get("score", 0.0),
        "text": data.get("text", ""),
        "metadata": data.get("metadata", {}),
    }