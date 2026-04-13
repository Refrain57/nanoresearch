"""MCP Server Tools.

This package exports Agentic RAG tools that allow agents to:
- Plan queries and select retrieval strategies
- Perform atomic retrieval (dense, sparse, hybrid)
- Control fusion and reranking
- Verify if results answer the query (self-reflection)
- Manage multi-turn search sessions

All tools are atomic and composable by agents for iterative retrieval.
"""

from nanobot.rag.mcp_server.tools.agentic import (
    # Retrieval
    RetrieveDenseTool,
    RetrieveSparseTool,
    RetrieveHybridTool,
    register_retrieval_tools,
    # Fusion
    FuseResultsTool,
    register_fusion_tools,
    # Reranking
    RerankResultsTool,
    register_reranking_tools,
    # Verification (self-reflection)
    VerifyResultsTool,
    register_verification_tools,
    # Query Planning
    PlanQueryTool,
    ProcessQueryTool,
    register_planning_tools,
    # Citations
    BuildCitationsTool,
    register_citation_tools,
    # Collections
    ListCollectionsTool,
    ListDocumentsTool,
    IngestDocumentTool,
    register_collection_tools,
    # Session Management
    CreateSessionTool,
    GetSessionTool,
    UpdateSessionTool,
    CloseSessionTool,
    ListSessionsTool,
    register_session_tools,
    # All-in-one registration
    register_all_agentic_tools,
)

__all__ = [
    # Retrieval
    "RetrieveDenseTool",
    "RetrieveSparseTool",
    "RetrieveHybridTool",
    "register_retrieval_tools",
    # Fusion
    "FuseResultsTool",
    "register_fusion_tools",
    # Reranking
    "RerankResultsTool",
    "register_reranking_tools",
    # Verification
    "VerifyResultsTool",
    "register_verification_tools",
    # Query Planning
    "PlanQueryTool",
    "ProcessQueryTool",
    "register_planning_tools",
    # Citations
    "BuildCitationsTool",
    "register_citation_tools",
    # Collections
    "ListCollectionsTool",
    "ListDocumentsTool",
    "IngestDocumentTool",
    "register_collection_tools",
    # Session Management
    "CreateSessionTool",
    "GetSessionTool",
    "UpdateSessionTool",
    "CloseSessionTool",
    "ListSessionsTool",
    "register_session_tools",
    # All-in-one registration
    "register_all_agentic_tools",
]
