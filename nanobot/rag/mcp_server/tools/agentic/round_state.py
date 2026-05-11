"""Round state management for concurrent multi-round retrieval.

This module provides in-memory state management for retrieval rounds,
enabling:
- Concurrent execution of multiple retrieval tasks
- Cross-round result accumulation
- TTL-based automatic cleanup (30 minutes)

Design Principles:
- In-memory storage: Fast access, no persistence needed
- TTL cleanup: 30 minutes idle timeout
- Single instance: Singleton pattern for global state
- Task tracking: Each task has unique ID for result mapping
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nanobot.rag.core.types import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTask:
    """A single retrieval task within a round.

    Attributes:
        task_id: Unique identifier for this task
        query: The search query
        strategy: Retrieval strategy ("dense", "sparse", or "hybrid")
        created_at: When this task was created
    """
    task_id: str
    query: str
    strategy: str  # "dense" | "sparse" | "hybrid"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RoundState:
    """State for a retrieval round.

    A round tracks all tasks and results for a single retrieval session.
    Results are accumulated across multiple batch calls until fused.

    Attributes:
        round_id: Unique identifier for this round
        tasks: List of tasks in this round
        results: Mapping from task_id to list of results
        created_at: When this round was created
        last_access: Last time this round was accessed (for TTL)
        fused: Whether this round has been fused (final results returned)
    """
    round_id: str
    tasks: List[RetrievalTask] = field(default_factory=list)
    results: Dict[str, List[RetrievalResult]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_access: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fused: bool = False

    def add_task(self, task: RetrievalTask) -> None:
        """Add a task to this round."""
        self.tasks.append(task)
        self.last_access = datetime.now(timezone.utc)

    def add_results(self, task_id: str, results: List[RetrievalResult]) -> None:
        """Add results for a task."""
        self.results[task_id] = results
        self.last_access = datetime.now(timezone.utc)

    def get_total_chunks(self) -> int:
        """Get total number of chunks across all tasks."""
        return sum(len(r) for r in self.results.values())

    def get_all_results(self) -> List[RetrievalResult]:
        """Get all results flattened."""
        all_results = []
        for task_results in self.results.values():
            all_results.extend(task_results)
        return all_results


@dataclass
class RoundStatus:
    """Status information for a round.

    Attributes:
        round_id: The round identifier
        tasks: List of task summaries
        total_chunks: Total chunks across all tasks
        fused: Whether round has been fused
        age_minutes: Minutes since creation
    """
    round_id: str
    tasks: List[Dict[str, Any]]
    total_chunks: int
    fused: bool
    age_minutes: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_id": self.round_id,
            "tasks": self.tasks,
            "total_chunks": self.total_chunks,
            "fused": self.fused,
            "age_minutes": round(self.age_minutes, 1),
        }


class RoundStateManager:
    """Manager for all retrieval rounds.

    This is a singleton class that manages:
    - Round creation and storage
    - Task and result tracking
    - TTL-based cleanup (30 minutes)
    - Round fusion

    Thread-safe implementation using a lock for concurrent access.

    Example:
        >>> manager = RoundStateManager.get_instance()
        >>> round_id = manager.create_round()
        >>> manager.add_task(round_id, RetrievalTask(...))
        >>> status = manager.get_status(round_id)
    """

    _instance: Optional["RoundStateManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RoundStateManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the manager."""
        if self._initialized:
            return

        self._rounds: Dict[str, RoundState] = {}
        self._ttl_seconds = 30 * 60  # 30 minutes
        self._cleanup_interval = 60  # Cleanup every minute
        self._last_cleanup = time.time()
        self._initialized = True

        logger.info("RoundStateManager initialized with 30-minute TTL")

    @classmethod
    def get_instance(cls) -> "RoundStateManager":
        """Get the singleton instance.

        Returns:
            RoundStateManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing).

        Warning: This should only be used in tests.
        """
        with cls._lock:
            cls._instance = None

    def create_round(self) -> str:
        """Create a new round.

        Returns:
            The new round_id
        """
        self._maybe_cleanup()

        round_id = str(uuid.uuid4())[:8]
        round_state = RoundState(round_id=round_id)
        self._rounds[round_id] = round_state

        logger.debug(f"Created round: {round_id}")
        return round_id

    def get_or_create_round(self, round_id: Optional[str]) -> str:
        """Get existing round or create new one.

        Args:
            round_id: Existing round_id, or None to create new

        Returns:
            The round_id (existing or new)
        """
        if round_id is not None:
            if round_id in self._rounds:
                self._rounds[round_id].last_access = datetime.now(timezone.utc)
                return round_id
            # Create round with specified ID
            self._rounds[round_id] = RoundState(round_id=round_id)
            logger.info(f"Created round with specified ID: {round_id}")
            return round_id

        return self.create_round()

    def get_round(self, round_id: str) -> Optional[RoundState]:
        """Get a round by ID.

        Args:
            round_id: The round identifier

        Returns:
            RoundState if found, None otherwise
        """
        self._maybe_cleanup()

        if round_id not in self._rounds:
            return None

        self._rounds[round_id].last_access = datetime.now(timezone.utc)
        return self._rounds[round_id]

    def add_task(self, round_id: str, task: RetrievalTask) -> None:
        """Add a task to a round.

        Args:
            round_id: The round identifier
            task: The task to add

        Raises:
            ValueError: If round not found
        """
        round_state = self.get_round(round_id)
        if round_state is None:
            raise ValueError(f"Round not found: {round_id}")

        round_state.add_task(task)
        logger.debug(f"Added task {task.task_id} to round {round_id}")

    def add_results(self, round_id: str, task_id: str, results: List[Any]) -> None:
        """Add results for a task.

        Args:
            round_id: The round identifier
            task_id: The task identifier
            results: List of retrieval results

        Raises:
            ValueError: If round not found
        """
        # Convert to RetrievalResult if needed
        converted_results = []
        for r in results:
            if isinstance(r, dict):
                converted_results.append(
                    RetrievalResult(
                        chunk_id=r.get("chunk_id", ""),
                        score=r.get("score", 0.0),
                        text=r.get("text", ""),
                        metadata=r.get("metadata", {}),
                    )
                )
            elif isinstance(r, RetrievalResult):
                converted_results.append(r)
            else:
                # Skip invalid results
                continue

        round_state = self.get_round(round_id)
        if round_state is None:
            raise ValueError(f"Round not found: {round_id}")

        round_state.add_results(task_id, converted_results)
        logger.debug(f"Added {len(converted_results)} results for task {task_id} in round {round_id}")

    def get_status(self, round_id: str) -> Optional[RoundStatus]:
        """Get status for a round.

        Args:
            round_id: The round identifier

        Returns:
            RoundStatus if found, None otherwise
        """
        round_state = self.get_round(round_id)
        if round_state is None:
            return None

        # Calculate age
        now = datetime.now(timezone.utc)
        age = now - round_state.created_at
        age_minutes = age.total_seconds() / 60

        # Build task summaries
        tasks = []
        for task in round_state.tasks:
            task_results = round_state.results.get(task.task_id, [])
            tasks.append({
                "task_id": task.task_id,
                "query": task.query,
                "strategy": task.strategy,
                "chunks": len(task_results),
            })

        return RoundStatus(
            round_id=round_id,
            tasks=tasks,
            total_chunks=round_state.get_total_chunks(),
            fused=round_state.fused,
            age_minutes=age_minutes,
        )

    def fuse(
        self,
        round_id: str,
        strategy: str = "rrf",
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fuse all results in a round and return.

        Args:
            round_id: The round identifier
            strategy: Fusion strategy ("rrf", "weighted", "interleave")
            top_k: Maximum results to return

        Returns:
            List of fused results as dictionaries

        Raises:
            ValueError: If round not found
        """
        round_state = self.get_round(round_id)
        if round_state is None:
            raise ValueError(f"Round not found: {round_id}")

        if round_state.fused:
            logger.warning(f"Round {round_id} already fused")
            # Return existing fused results
            return [r.to_dict() if hasattr(r, 'to_dict') else r
                    for r in round_state.get_all_results()[:top_k]]

        all_results = round_state.get_all_results()
        if not all_results:
            logger.debug(f"No results to fuse in round {round_id}")
            return []

        # Apply fusion
        if strategy == "rrf":
            fused = self._rrf_fusion(all_results, k=60, top_k=top_k)
        elif strategy == "weighted":
            fused = self._weighted_fusion(all_results, top_k=top_k)
        elif strategy == "interleave":
            fused = self._interleave_fusion(all_results, top_k=top_k)
        else:
            # Default to RRF
            fused = self._rrf_fusion(all_results, k=60, top_k=top_k)

        # Mark as fused
        round_state.fused = True

        logger.info(f"Fused {len(all_results)} results into {len(fused)} in round {round_id}")
        return fused

    def delete_round(self, round_id: str) -> bool:
        """Delete a round.

        Args:
            round_id: The round identifier

        Returns:
            True if deleted, False if not found
        """
        if round_id in self._rounds:
            del self._rounds[round_id]
            logger.debug(f"Deleted round: {round_id}")
            return True
        return False

    def list_rounds(self) -> List[str]:
        """List all active round IDs.

        Returns:
            List of round_id strings
        """
        self._maybe_cleanup()
        return list(self._rounds.keys())

    def _maybe_cleanup(self) -> None:
        """Cleanup expired rounds if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._cleanup()
        self._last_cleanup = now

    def _cleanup(self) -> None:
        """Remove expired rounds based on TTL."""
        now = datetime.now(timezone.utc)
        expired = []

        for round_id, round_state in self._rounds.items():
            age = now - round_state.last_access
            if age.total_seconds() > self._ttl_seconds:
                expired.append(round_id)

        for round_id in expired:
            del self._rounds[round_id]
            logger.debug(f"Cleaned up expired round: {round_id}")

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired rounds")

    def _rrf_fusion(
        self,
        results: List[RetrievalResult],
        k: int = 60,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each list the doc appears in.
        Since we have all results in one list, we rank by score.

        Args:
            results: All retrieval results
            k: RRF parameter (default 60)
            top_k: Maximum results to return

        Returns:
            Fused results sorted by score
        """
        # Rank by score (descending)
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        # Calculate RRF scores
        scored_results = []
        for rank, result in enumerate(sorted_results, 1):
            rrf_score = 1.0 / (k + rank)

            # Combine original score with RRF score
            combined_score = result.score + rrf_score

            # Convert to dict
            if hasattr(result, 'to_dict'):
                item = result.to_dict()
            else:
                item = {
                    "chunk_id": getattr(result, "chunk_id", ""),
                    "score": result.score,
                    "text": getattr(result, "text", ""),
                    "metadata": getattr(result, "metadata", {}),
                }

            item["fusion_score"] = combined_score
            item["fusion_method"] = "rrf"
            item["original_rank"] = rank
            scored_results.append(item)

        # Sort by fusion score
        scored_results.sort(key=lambda r: r["fusion_score"], reverse=True)

        return scored_results[:top_k]

    def _weighted_fusion(
        self,
        results: List[RetrievalResult],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Weighted score fusion (normalized).

        Args:
            results: All retrieval results
            top_k: Maximum results to return

        Returns:
            Fused results
        """
        if not results:
            return []

        # Normalize scores to [0, 1]
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        fused = []
        for result in results:
            normalized_score = (result.score - min_score) / score_range

            if hasattr(result, 'to_dict'):
                item = result.to_dict()
            else:
                item = {
                    "chunk_id": getattr(result, "chunk_id", ""),
                    "score": result.score,
                    "text": getattr(result, "text", ""),
                    "metadata": getattr(result, "metadata", {}),
                }

            item["normalized_score"] = normalized_score
            item["fusion_method"] = "weighted"
            fused.append(item)

        fused.sort(key=lambda r: r["normalized_score"], reverse=True)
        return fused[:top_k]

    def _interleave_fusion(
        self,
        results: List[RetrievalResult],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Interleave results (alternate between sources).

        Since we don't have multiple sources here, just sort by score.

        Args:
            results: All retrieval results
            top_k: Maximum results to return

        Returns:
            Fused results
        """
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        fused = []
        for rank, result in enumerate(sorted_results[:top_k]):
            if hasattr(result, 'to_dict'):
                item = result.to_dict()
            else:
                item = {
                    "chunk_id": getattr(result, "chunk_id", ""),
                    "score": result.score,
                    "text": getattr(result, "text", ""),
                    "metadata": getattr(result, "metadata", {}),
                }

            item["fusion_method"] = "interleave"
            item["fusion_rank"] = rank
            fused.append(item)

        return fused


# Module-level convenience functions
def get_round_manager() -> RoundStateManager:
    """Get the singleton RoundStateManager instance.

    Returns:
        RoundStateManager instance
    """
    return RoundStateManager.get_instance()
