"""Internal RAG Loop - State management for internal agent loop.

This module provides state management for the internal RAG loop,
including:
- Session state for the entire loop
- Round-level state for each retrieval round
- Sub-query tracking with strategy annotations
- Fusion and verification results

Design Principles:
- In-memory storage for fast access
- Phase-based state transitions
- Thread-safe operations
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class SubQuery:
    """A single sub-query with strategy annotation.

    Attributes:
        query: The sub-query text
        strategy: Retrieval strategy ("dense", "sparse", "hybrid")
        reason: Why this strategy was chosen
        completed: Whether this sub-query has been executed
    """
    query: str
    strategy: str  # "dense" | "sparse" | "hybrid"
    reason: str = ""
    completed: bool = False


@dataclass
class PlanResult:
    """Result from plan_query tool.

    Attributes:
        complexity: Overall complexity ("simple", "complex")
        sub_queries: List of sub-queries with strategy annotations
        context_aware: Whether context was used for rewriting
    """
    complexity: str
    sub_queries: List[SubQuery]
    context_aware: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity,
            "sub_queries": [
                {
                    "query": sq.query,
                    "strategy": sq.strategy,
                    "reason": sq.reason,
                    "completed": sq.completed,
                }
                for sq in self.sub_queries
            ],
            "context_aware": self.context_aware,
        }


@dataclass
class SessionState:
    """State for an entire RAG search session.

    This holds all state for a single kb_search call,
    including the original query, context, and all rounds.

    Attributes:
        session_id: Unique identifier for this session
        original_query: The user's original query
        context: External context from main agent (optional)
        plan: Result from plan_query
        rounds: List of round states
        current_phase: Current phase in the loop
        max_iterations: Maximum iterations allowed
    """
    session_id: str
    original_query: str
    context: Optional[str] = None
    plan: Optional[PlanResult] = None
    rounds: List["RoundState"] = field(default_factory=list)
    current_phase: str = "plan"  # "plan" | "search" | "verify" | "finalize"
    max_iterations: int = 5
    iteration: int = 0
    fused_chunks: List[Dict[str, Any]] = field(default_factory=list)
    verification_results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_round(self) -> Optional["RoundState"]:
        """Get the current (last) round."""
        return self.rounds[-1] if self.rounds else None

    def add_round(self) -> "RoundState":
        """Add a new round and return it."""
        round_state = RoundState(round_id=str(uuid.uuid4())[:8])
        self.rounds.append(round_state)
        return round_state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "context": self.context,
            "plan": self.plan.to_dict() if self.plan else None,
            "current_phase": self.current_phase,
            "iteration": self.iteration,
            "rounds": [r.to_dict() for r in self.rounds],
        }


@dataclass
class RoundState:
    """State for a single retrieval round.

    Attributes:
        round_id: Unique identifier for this round
        sub_queries: Sub-queries planned for this round
        results: Mapping from query to list of results
        fused: Whether this round has been fused
        fused_chunks: Fuse后的结果
    """
    round_id: str
    sub_queries: List[SubQuery] = field(default_factory=list)
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    fused: bool = False
    fused_chunks: List[Dict[str, Any]] = field(default_factory=list)

    def add_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Add results for a sub-query."""
        self.results[query] = results

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all results flattened."""
        all_results = []
        for query_results in self.results.values():
            all_results.extend(query_results)
        return all_results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "sub_queries": [
                {"query": sq.query, "strategy": sq.strategy, "completed": sq.completed}
                for sq in self.sub_queries
            ],
            "result_count": sum(len(r) for r in self.results.values()),
            "fused": self.fused,
            "fused_chunks_count": len(self.fused_chunks),
        }


class SessionStateManager:
    """Manager for internal loop sessions.

    Thread-safe singleton that manages:
    - Session creation and storage
    - Session lookup by ID
    - TTL-based cleanup

    Example:
        >>> manager = SessionStateManager.get_instance()
        >>> session = manager.create_session(query="...", context="...")
        >>> session.add_round()
    """

    _instance: Optional["SessionStateManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SessionStateManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._sessions: Dict[str, SessionState] = {}
        self._ttl_seconds = 30 * 60  # 30 minutes
        self._cleanup_interval = 60
        self._initialized = True
        self._last_cleanup = datetime.now()

        logger.info("SessionStateManager initialized")

    @classmethod
    def get_instance(cls) -> "SessionStateManager":
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def create_session(
        self,
        query: str,
        context: Optional[str] = None,
        max_iterations: int = 5,
    ) -> SessionState:
        """Create a new session.

        Args:
            query: The original query
            context: External context (optional)
            max_iterations: Maximum iterations

        Returns:
            New SessionState
        """
        session_id = str(uuid.uuid4())[:12]
        session = SessionState(
            session_id=session_id,
            original_query=query,
            context=context,
            max_iterations=max_iterations,
        )
        self._sessions[session_id] = session

        logger.debug(f"Created session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def _maybe_cleanup(self) -> None:
        """Cleanup expired sessions."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        if (now - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return

        expired = []
        for session_id, session in self._sessions.items():
            # Check if session is too old or has completed
            if session.current_phase == "finalize":
                # Keep finalized sessions for a short time
                continue

        for session_id in expired:
            self.delete_session(session_id)

        self._last_cleanup = now


# Module-level convenience
def get_session_manager() -> SessionStateManager:
    """Get the singleton SessionStateManager instance."""
    return SessionStateManager.get_instance()
