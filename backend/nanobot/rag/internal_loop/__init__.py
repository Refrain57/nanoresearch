"""Internal RAG Loop module.

Provides a state-machine based RAG loop with:
- Phase-based tool exposure
- Token-efficient message cleanup
- Multi-round retrieval with verification
- Automatic citation building
"""

from nanobot.rag.internal_loop.state import (
    SessionState,
    SessionStateManager,
    RoundState,
    PlanResult,
    SubQuery,
    get_session_manager,
)
from nanobot.rag.internal_loop.runner import (
    RAGLoopRunner,
    RAGLoopResult,
    run_rag_loop,
    classify_complexity,
)
from nanobot.rag.internal_loop.tools import (
    InternalTools,
    get_internal_tools,
)
from nanobot.rag.internal_loop.cleanup import (
    cleanup_messages_for_next_round,
    estimate_messages_tokens,
)

__all__ = [
    # State
    "SessionState",
    "SessionStateManager",
    "RoundState",
    "PlanResult",
    "SubQuery",
    "get_session_manager",
    # Runner
    "RAGLoopRunner",
    "RAGLoopResult",
    "run_rag_loop",
    "classify_complexity",
    # Tools
    "InternalTools",
    "get_internal_tools",
    # Cleanup
    "cleanup_messages_for_next_round",
    "estimate_messages_tokens",
]
