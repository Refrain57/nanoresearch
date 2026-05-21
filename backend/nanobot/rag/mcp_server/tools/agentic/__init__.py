"""Agentic RAG MCP Tools.

This package contains atomic tools for Agentic RAG, allowing agents to:
- Plan queries with LLM analysis
- Perform raw retrieval (dense, sparse, hybrid)
- Control fusion of results
- Apply reranking
- Verify if results answer the query
- Build structured citations
- Manage multi-turn search sessions

Tools are designed to be composed by agents for iterative retrieval.
"""

from nanobot.rag.mcp_server.tools.agentic.retrieval import (
    RetrieveDenseTool,
    RetrieveSparseTool,
    RetrieveHybridTool,
    FetchSectionTool,
    FetchNeighborsTool,
    register_tools as register_retrieval_tools,
)

from nanobot.rag.mcp_server.tools.agentic.fusion import (
    FuseResultsTool,
    register_tools as register_fusion_tools,
)

from nanobot.rag.mcp_server.tools.agentic.reranking import (
    RerankResultsTool,
    register_tools as register_reranking_tools,
)

from nanobot.rag.mcp_server.tools.agentic.verification import (
    VerifyResultsTool,
    register_tools as register_verification_tools,
)

from nanobot.rag.mcp_server.tools.agentic.query_planning import (
    PlanQueryTool,
    ProcessQueryTool,
    register_tools as register_planning_tools,
)

from nanobot.rag.mcp_server.tools.agentic.citations import (
    BuildCitationsTool,
    register_tools as register_citation_tools,
)

from nanobot.rag.mcp_server.tools.agentic.collections import (
    ListCollectionsTool,
    ListDocumentsTool,
    IngestDocumentTool,
    register_tools as register_collection_tools,
)

from nanobot.rag.mcp_server.tools.agentic.session import (
    CreateSessionTool,
    GetSessionTool,
    UpdateSessionTool,
    CloseSessionTool,
    ListSessionsTool,
    register_tools as register_session_tools,
)


def register_all_agentic_tools(protocol_handler) -> None:
    """Register all agentic RAG tools with the protocol handler.

    Args:
        protocol_handler: The MCP protocol handler instance
    """
    register_retrieval_tools(protocol_handler)
    register_fusion_tools(protocol_handler)
    register_reranking_tools(protocol_handler)
    register_verification_tools(protocol_handler)
    register_planning_tools(protocol_handler)
    register_citation_tools(protocol_handler)
    register_collection_tools(protocol_handler)
    register_session_tools(protocol_handler)


__all__ = [
    # Retrieval
    "RetrieveDenseTool",
    "RetrieveSparseTool",
    "RetrieveHybridTool",
    "FetchSectionTool",
    "FetchNeighborsTool",
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