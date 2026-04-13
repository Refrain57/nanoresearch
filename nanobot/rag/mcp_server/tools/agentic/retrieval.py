"""Agentic retrieval tools.

Provides atomic retrieval tools for fine-grained agent control:
- retrieve_dense: Pure vector similarity search
- retrieve_sparse: Pure BM25 keyword search
- retrieve_hybrid: Hybrid search with full intermediate results
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

from nanobot.rag.core.settings import Settings, load_settings
from nanobot.rag.core.query_engine.hybrid_search import HybridSearch, create_hybrid_search
from nanobot.rag.core.query_engine.dense_retriever import DenseRetriever, create_dense_retriever
from nanobot.rag.core.query_engine.sparse_retriever import SparseRetriever, create_sparse_retriever
from nanobot.rag.core.query_engine.query_processor import QueryProcessor
from nanobot.rag.ingestion.storage.bm25_indexer import BM25Indexer
from nanobot.rag.libs.embedding.embedding_factory import EmbeddingFactory
from nanobot.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
from nanobot.rag.core.settings import resolve_path
from nanobot.rag.core.response.response_builder import MCPToolResponse
from nanobot.rag.mcp_server.tools.agentic.shared import (
    results_to_dict_list,
    build_json_response,
    build_structured_response,
    format_markdown_results,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Metadata
# =============================================================================

RETRIEVE_DENSE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query for semantic similarity search",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return (default: 20)",
            "default": 20,
        },
        "collection": {
            "type": "string",
            "description": "Collection name (default: default)",
            "default": "default",
        },
        "domain": {
            "type": "string",
            "description": "Filter by document domain: papers, nanobot, notes, general (optional)",
        },
        "filters": {
            "type": "object",
            "description": "Metadata filters to apply (e.g., {\"source_path\": \"doc.pdf\"})",
        },
        "return_text": {
            "type": "boolean",
            "description": "Include text content in results (default: true)",
            "default": True,
        },
    },
    "required": ["query"],
}

RETRIEVE_SPARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keywords for BM25 search (auto-extracted if query provided)",
        },
        "query": {
            "type": "string",
            "description": "Query to extract keywords from (alternative to keywords)",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return (default: 20)",
            "default": 20,
        },
        "collection": {
            "type": "string",
            "description": "Collection name (default: default)",
            "default": "default",
        },
        "domain": {
            "type": "string",
            "description": "Filter by document domain: papers, nanobot, notes, general (optional)",
        },
        "return_text": {
            "type": "boolean",
            "description": "Include text content in results (default: true)",
            "default": True,
        },
    },
    "required": [],
}

RETRIEVE_HYBRID_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query",
        },
        "session_key": {
            "type": "string",
            "description": "Main agent session key for multi-turn context",
        },
        "top_k": {
            "type": "integer",
            "description": "Final number of results (default: 10)",
            "default": 10,
        },
        "dense_top_k": {
            "type": "integer",
            "description": "Number of dense results to retrieve (default: 20)",
            "default": 20,
        },
        "sparse_top_k": {
            "type": "integer",
            "description": "Number of sparse results to retrieve (default: 20)",
            "default": 20,
        },
        "collection": {
            "type": "string",
            "description": "Collection name (default: default)",
            "default": "default",
        },
        "domain": {
            "type": "string",
            "description": "Filter by document domain: papers, nanobot, notes, general (optional)",
        },
        "enable_dense": {
            "type": "boolean",
            "description": "Enable dense retrieval (default: true)",
            "default": True,
        },
        "enable_sparse": {
            "type": "boolean",
            "description": "Enable sparse retrieval (default: true)",
            "default": True,
        },
        "filters": {
            "type": "object",
            "description": "Metadata filters for dense retrieval",
        },
        "rrf_k": {
            "type": "integer",
            "description": "RRF smoothing parameter k (default: 60)",
        },
        "return_intermediate": {
            "type": "boolean",
            "description": "Return intermediate dense/sparse results (default: true)",
            "default": True,
        },
    },
    "required": ["query"],
}

# =============================================================================
# Structure-Aware Retrieval Schema
# =============================================================================

FETCH_SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "section_path": {
            "type": "string",
            "description": "Section path to fetch (e.g., '/RAG/检索策略')",
        },
        "collection": {
            "type": "string",
            "description": "Collection name (default: default)",
            "default": "default",
        },
        "include_neighbors": {
            "type": "boolean",
            "description": "Also fetch prev/next chunks (default: true)",
            "default": True,
        },
        "max_chunks": {
            "type": "integer",
            "description": "Maximum chunks to return (default: 10)",
            "default": 10,
        },
    },
    "required": ["section_path"],
}

FETCH_NEIGHBORS_SCHEMA = {
    "type": "object",
    "properties": {
        "chunk_id": {
            "type": "string",
            "description": "Chunk ID to get neighbors for",
        },
        "collection": {
            "type": "string",
            "description": "Collection name (default: default)",
            "default": "default",
        },
        "window": {
            "type": "integer",
            "description": "Number of chunks before/after (default: 1)",
            "default": 1,
        },
    },
    "required": ["chunk_id"],
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RetrievalConfig:
    """Configuration for retrieval tools."""
    default_collection: str = "default"
    default_top_k: int = 10
    max_top_k: int = 100
    default_dense_top_k: int = 20
    default_sparse_top_k: int = 20


# =============================================================================
# Dense Retrieval Tool
# =============================================================================

class RetrieveDenseTool:
    """Tool for pure dense (vector) retrieval.

    This tool provides access to the semantic similarity search without
    any keyword matching or fusion. Useful when you want strictly
    semantic results.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._settings = settings
        self._config = config or RetrievalConfig()
        self._embedding_client = None
        self._retriever = None
        self._current_collection = None
        self._vector_store = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _ensure_initialized(self, collection: str) -> None:
        """Initialize components for the given collection."""
        if self._current_collection == collection and self._retriever is not None:
            return

        self._embedding_client = EmbeddingFactory.create(self.settings)
        self._vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        self._retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=self._embedding_client,
            vector_store=self._vector_store,
        )
        self._current_collection = collection

    async def execute(
        self,
        query: str,
        top_k: int = 20,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        return_text: bool = True,
    ) -> MCPToolResponse:
        """Execute dense retrieval.

        Args:
            query: Search query
            top_k: Number of results
            collection: Collection name
            filters: Metadata filters
            return_text: Include text content

        Returns:
            MCPToolResponse with retrieval results
        """
        effective_collection = collection or self._config.default_collection
        effective_top_k = min(top_k, self._config.max_top_k)

        logger.info(f"retrieve_dense: query='{query}', collection={effective_collection}")

        try:
            # Initialize components in thread
            await asyncio.to_thread(
                self._ensure_initialized, effective_collection
            )

            # Perform retrieval in thread
            def _retrieve():
                return self._retriever.retrieve(
                    query=query,
                    top_k=effective_top_k,
                    filters=filters,
                )

            results = await asyncio.to_thread(_retrieve)

            # Convert to dict format
            result_dicts = results_to_dict_list(
                results,
                include_text=return_text,
            )

            # Build response
            response_data = {
                "method": "dense",
                "query": query,
                "collection": effective_collection,
                "total_results": len(results),
                "top_k_requested": effective_top_k,
                "results": result_dicts,
                "embedding_model": self.settings.embedding.model,
                "dimension": 1024,  # text-embedding-v3
            }

            # Add markdown for human readability
            markdown = format_markdown_results(
                results,
                query,
                max_results=min(5, len(results)),
            )

            return MCPToolResponse(
                content=json.dumps(response_data, ensure_ascii=False, indent=2)
                + "\n\n"
                + markdown,
                metadata=response_data,
                is_empty=len(results) == 0,
            )

        except Exception as e:
            logger.exception(f"retrieve_dense error: {e}")
            return MCPToolResponse(
                content=f"Error in dense retrieval: {e}",
                is_empty=False,
            )


# =============================================================================
# Sparse Retrieval Tool
# =============================================================================

class RetrieveSparseTool:
    """Tool for pure sparse (BM25) retrieval.

    This tool provides access to keyword-based search using BM25.
    Useful for exact term matching and when semantic search isn't needed.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._settings = settings
        self._config = config or RetrievalConfig()
        self._retriever = None
        self._current_collection = None
        self._vector_store = None
        self._bm25_indexer = None
        self._query_processor = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _ensure_initialized(self, collection: str) -> None:
        """Initialize components for the given collection."""
        if self._current_collection == collection and self._retriever is not None:
            return

        self._vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        self._bm25_indexer = BM25Indexer(
            index_dir=str(resolve_path(f"~/.nanobot/rag/bm25/{collection}"))
        )
        self._retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=self._bm25_indexer,
            vector_store=self._vector_store,
        )
        self._query_processor = QueryProcessor()
        self._current_collection = collection

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query using QueryProcessor."""
        processed = self._query_processor.process(query)
        return processed.keywords

    async def execute(
        self,
        keywords: Optional[List[str]] = None,
        query: Optional[str] = None,
        top_k: int = 20,
        collection: Optional[str] = None,
        return_text: bool = True,
    ) -> MCPToolResponse:
        """Execute sparse retrieval.

        Args:
            keywords: Direct BM25 keywords
            query: Query to extract keywords from
            top_k: Number of results
            collection: Collection name
            return_text: Include text content

        Returns:
            MCPToolResponse with retrieval results
        """
        effective_collection = collection or self._config.default_collection
        effective_top_k = min(top_k, self._config.max_top_k)

        # Extract keywords from query if not provided
        if keywords is None:
            if query is None:
                return MCPToolResponse(
                    content="Error: Must provide either 'keywords' or 'query' parameter",
                    is_empty=False,
                )
            await asyncio.to_thread(self._ensure_initialized, effective_collection)
            keywords = await asyncio.to_thread(self._extract_keywords, query)
        else:
            await asyncio.to_thread(self._ensure_initialized, effective_collection)

        logger.info(
            f"retrieve_sparse: keywords={keywords}, "
            f"collection={effective_collection}"
        )

        try:
            # Perform retrieval in thread
            def _retrieve():
                return self._retriever.retrieve(
                    keywords=keywords,
                    top_k=effective_top_k,
                    collection=effective_collection,
                )

            results = await asyncio.to_thread(_retrieve)

            # Convert to dict format
            result_dicts = results_to_dict_list(
                results,
                include_text=return_text,
            )

            # Build response
            response_data = {
                "method": "sparse",
                "keywords": keywords,
                "query": query,
                "collection": effective_collection,
                "total_results": len(results),
                "top_k_requested": effective_top_k,
                "results": result_dicts,
            }

            # Add markdown for human readability
            markdown = format_markdown_results(
                results,
                query or " ".join(keywords),
                max_results=min(5, len(results)),
            )

            return MCPToolResponse(
                content=json.dumps(response_data, ensure_ascii=False, indent=2)
                + "\n\n"
                + markdown,
                metadata=response_data,
                is_empty=len(results) == 0,
            )

        except Exception as e:
            logger.exception(f"retrieve_sparse error: {e}")
            return MCPToolResponse(
                content=f"Error in sparse retrieval: {e}",
                is_empty=False,
            )


# =============================================================================
# Hybrid Retrieval Tool (Core Agentic Tool)
# =============================================================================

class RetrieveHybridTool:
    """Tool for hybrid search with full intermediate results.

    This is the core agentic retrieval tool. It returns:
    - Raw dense results (before fusion)
    - Raw sparse results (before fusion)
    - Fused results (after RRF)
    - Reranked results (if enabled)

    This allows the agent to inspect and control each step of the retrieval.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._settings = settings
        self._config = config or RetrievalConfig()
        self._embedding_client = None
        self._reranker = None
        self._hybrid_search = None
        self._current_collection = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _ensure_initialized(self, collection: str) -> None:
        """Initialize components for the given collection."""
        if self._current_collection == collection and self._hybrid_search is not None:
            return

        # Create components (reusing QueryKnowledgeHubTool pattern)
        self._embedding_client = EmbeddingFactory.create(self.settings)

        # Import here to avoid circular imports
        from nanobot.rag.core.query_engine.dense_retriever import create_dense_retriever
        from nanobot.rag.core.query_engine.sparse_retriever import create_sparse_retriever
        from nanobot.rag.core.query_engine.reranker import create_core_reranker
        from nanobot.rag.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )

        dense_retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=self._embedding_client,
            vector_store=vector_store,
        )

        bm25_indexer = BM25Indexer(
            index_dir=str(resolve_path(f"~/.nanobot/rag/bm25/{collection}"))
        )
        sparse_retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        from nanobot.rag.core.query_engine.query_processor import QueryProcessor
        query_processor = QueryProcessor()

        self._hybrid_search = create_hybrid_search(
            settings=self.settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        self._reranker = create_core_reranker(settings=self.settings)

        self._current_collection = collection

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        collection: Optional[str] = None,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        rrf_k: Optional[int] = None,
        return_intermediate: bool = True,
    ) -> MCPToolResponse:
        """Execute hybrid retrieval with full details.

        Args:
            query: Search query
            top_k: Final number of results
            dense_top_k: Number of dense results to retrieve
            sparse_top_k: Number of sparse results to retrieve
            collection: Collection name
            enable_dense: Enable dense retrieval
            enable_sparse: Enable sparse retrieval
            filters: Metadata filters
            rrf_k: RRF smoothing parameter
            return_intermediate: Return raw dense/sparse results

        Returns:
            MCPToolResponse with full retrieval details
        """
        effective_collection = collection or self._config.default_collection
        effective_top_k = min(top_k, self._config.max_top_k)

        logger.info(
            f"retrieve_hybrid: query='{query}', collection={effective_collection}, "
            f"top_k={effective_top_k}, dense={enable_dense}, sparse={enable_sparse}"
        )

        try:
            # Initialize components in thread
            await asyncio.to_thread(
                self._ensure_initialized, effective_collection
            )

            # Perform search in thread
            def _search():
                return self._hybrid_search.search(
                    query=query,
                    top_k=effective_top_k,
                    filters=filters,
                    return_details=True,
                )

            result = await asyncio.to_thread(_search)

            # Convert results
            final_results = results_to_dict_list(result.results)

            # Build response data
            response_data: Dict[str, Any] = {
                "method": "hybrid",
                "query": query,
                "collection": effective_collection,
                "final_results": final_results,
                "final_results_count": len(final_results),
                "used_fallback": result.used_fallback,
            }

            # Add intermediate results if requested
            if return_intermediate:
                if result.dense_results is not None:
                    response_data["dense_retrieval"] = {
                        "enabled": enable_dense,
                        "results_count": len(result.dense_results),
                        "results": results_to_dict_list(result.dense_results),
                        "error": result.dense_error,
                    }
                else:
                    response_data["dense_retrieval"] = {
                        "enabled": enable_dense,
                        "results_count": 0,
                        "results": [],
                    }

                if result.sparse_results is not None:
                    response_data["sparse_retrieval"] = {
                        "enabled": enable_sparse,
                        "results_count": len(result.sparse_results),
                        "results": results_to_dict_list(result.sparse_results),
                        "error": result.sparse_error,
                    }
                else:
                    response_data["sparse_retrieval"] = {
                        "enabled": enable_sparse,
                        "results_count": 0,
                        "results": [],
                    }

                response_data["fusion"] = {
                    "method": "rrf",
                    "k": rrf_k or 60,
                    "input_counts": {
                        "dense": len(result.dense_results) if result.dense_results else 0,
                        "sparse": len(result.sparse_results) if result.sparse_results else 0,
                    },
                }

            # Add processed query info
            if result.processed_query is not None:
                response_data["processed_query"] = {
                    "original_query": result.processed_query.original_query,
                    "keywords": result.processed_query.keywords,
                    "filters": result.processed_query.filters,
                }

            # Add errors if any
            if result.dense_error or result.sparse_error:
                response_data["errors"] = {
                    "dense": result.dense_error,
                    "sparse": result.sparse_error,
                }

            # Markdown summary
            markdown = format_markdown_results(
                result.results,
                query,
                max_results=min(5, len(result.results)),
            )

            return MCPToolResponse(
                content=json.dumps(response_data, ensure_ascii=False, indent=2)
                + "\n\n"
                + markdown,
                metadata={
                    "query": query,
                    "collection": effective_collection,
                    "result_count": len(final_results),
                },
                is_empty=len(final_results) == 0,
            )

        except Exception as e:
            logger.exception(f"retrieve_hybrid error: {e}")
            return MCPToolResponse(
                content=f"Error in hybrid retrieval: {e}",
                is_empty=False,
            )


# =============================================================================
# Module-level Tool Instances
# =============================================================================

_dense_tool: Optional[RetrieveDenseTool] = None
_sparse_tool: Optional[RetrieveSparseTool] = None
_hybrid_tool: Optional[RetrieveHybridTool] = None


def get_dense_tool(settings: Optional[Settings] = None) -> RetrieveDenseTool:
    """Get or create the dense retrieval tool instance."""
    global _dense_tool
    if _dense_tool is None:
        _dense_tool = RetrieveDenseTool(settings=settings)
    return _dense_tool


def get_sparse_tool(settings: Optional[Settings] = None) -> RetrieveSparseTool:
    """Get or create the sparse retrieval tool instance."""
    global _sparse_tool
    if _sparse_tool is None:
        _sparse_tool = RetrieveSparseTool(settings=settings)
    return _sparse_tool


def get_hybrid_tool(settings: Optional[Settings] = None) -> RetrieveHybridTool:
    """Get or create the hybrid retrieval tool instance."""
    global _hybrid_tool
    if _hybrid_tool is None:
        _hybrid_tool = RetrieveHybridTool(settings=settings)
    return _hybrid_tool


# =============================================================================
# MCP Handlers
# =============================================================================

async def retrieve_dense_handler(
    query: str,
    top_k: int = 20,
    collection: str = "default",
    domain: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    return_text: bool = True,
) -> types.CallToolResult:
    """MCP handler for retrieve_dense."""
    # Convert domain to filters
    if domain and not filters:
        filters = {"domain": domain}
    elif domain and filters:
        filters = {**filters, "domain": domain}

    tool = get_dense_tool()
    try:
        result = await tool.execute(
            query=query,
            top_k=top_k,
            collection=collection,
            filters=filters,
            return_text=return_text,
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=result.is_empty,
        )
    except Exception as e:
        logger.exception(f"retrieve_dense handler error: {e}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


async def retrieve_sparse_handler(
    keywords: Optional[List[str]] = None,
    query: Optional[str] = None,
    top_k: int = 20,
    collection: str = "default",
    return_text: bool = True,
) -> types.CallToolResult:
    """MCP handler for retrieve_sparse."""
    tool = get_sparse_tool()
    try:
        result = await tool.execute(
            keywords=keywords,
            query=query,
            top_k=top_k,
            collection=collection,
            return_text=return_text,
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=result.is_empty,
        )
    except Exception as e:
        logger.exception(f"retrieve_sparse handler error: {e}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


async def retrieve_hybrid_handler(
    query: str,
    session_key: Optional[str] = None,
    top_k: int = 10,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    collection: str = "default",
    domain: Optional[str] = None,
    enable_dense: bool = True,
    enable_sparse: bool = True,
    filters: Optional[Dict[str, Any]] = None,
    rrf_k: Optional[int] = None,
    return_intermediate: bool = True,
) -> types.CallToolResult:
    """MCP handler for retrieve_hybrid with query rewriting support."""
    # Query rewriting for multi-turn context
    logger.info(f"retrieve_hybrid_handler called: query='{query}', session_key='{session_key}', domain='{domain}'")

    # Convert domain to filters
    if domain and not filters:
        filters = {"domain": domain}
    elif domain and filters:
        filters = {**filters, "domain": domain}

    rewritten_query = query
    rewrite_info = {}

    # P5: Query rewriting for multi-turn context
    if session_key:
        try:
            from nanobot.rag.mcp_server.tools.agentic.query_planning import PlanQueryTool
            plan_tool = PlanQueryTool()
            plan_result = await plan_tool.execute(query=query, session_key=session_key)

            # Extract rewritten query from plan result
            if plan_result and plan_result.content:
                try:
                    plan_data = json.loads(plan_result.content)
                    rewritten_query = plan_data.get("rewritten_query", query)
                    rewrite_info = {
                        "original_query": query,
                        "rewritten_query": rewritten_query,
                        "history_used": plan_data.get("history_used", False),
                    }
                    logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse plan result, using original query")
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}, using original query")

    tool = get_hybrid_tool()
    try:
        result = await tool.execute(
            query=rewritten_query,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            collection=collection,
            enable_dense=enable_dense,
            enable_sparse=enable_sparse,
            filters=filters,
            rrf_k=rrf_k,
            return_intermediate=return_intermediate,
        )

        # Add rewrite info to result if available
        if rewrite_info:
            try:
                result_data = json.loads(result.content)
                result_data["query_rewrite"] = rewrite_info
                result.content = json.dumps(result_data, ensure_ascii=False)
            except Exception:
                pass  # Keep original result if parsing fails

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=result.is_empty,
        )
    except Exception as e:
        logger.exception(f"retrieve_hybrid handler error: {e}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(protocol_handler) -> None:
    """Register all retrieval tools with the protocol handler."""
    protocol_handler.register_tool(
        name="retrieve_dense",
        description="Semantic search using vector embeddings - finds documents by MEANING.\n\n"
                   "When to use:\n"
                   "- Concepts, ideas, paraphrased queries\n"
                   "- User asks about 'how to do X' or 'best practices for Y'\n"
                   "- Query uses different words than documents\n\n"
                   "When NOT to use:\n"
                   "- Exact technical terms, code names, specific IDs\n"
                   "- Very short queries (1-2 words) - may be too broad\n\n"
                   "ALWAYS call verify_results after this to check quality.",
        input_schema=RETRIEVE_DENSE_SCHEMA,
        handler=retrieve_dense_handler,
    )
    logger.info("Registered MCP tool: retrieve_dense")

    protocol_handler.register_tool(
        name="retrieve_sparse",
        description="Keyword search (BM25) - finds documents by EXACT TERMS.\n\n"
                   "When to use:\n"
                   "- Technical terms, product names, code identifiers\n"
                   "- User asks about specific thing by name\n"
                   "- Query contains exact words from documents\n\n"
                   "When NOT to use:\n"
                   "- Conceptual questions, synonyms, paraphrasing\n\n"
                   "ALWAYS call verify_results after this to check quality.",
        input_schema=RETRIEVE_SPARSE_SCHEMA,
        handler=retrieve_sparse_handler,
    )
    logger.info("Registered MCP tool: retrieve_sparse")

    protocol_handler.register_tool(
        name="retrieve_hybrid",
        description="Combined semantic + keyword search (RECOMMENDED default).\n\n"
                   "Why use this:\n"
                   "- Best of both worlds: meaning AND exact terms\n"
                   "- Returns dense_results, sparse_results, AND fused_results\n"
                   "- Most robust for general queries\n\n"
                   "When to use:\n"
                   "- Default choice for most queries\n"
                   "- When unsure which strategy fits\n"
                   "- Complex queries with both concepts and terms\n\n"
                   "ALWAYS call verify_results after this to check quality.",
        input_schema=RETRIEVE_HYBRID_SCHEMA,
        handler=retrieve_hybrid_handler,
    )
    logger.info("Registered MCP tool: retrieve_hybrid")

    # Register new structure-aware tools
    protocol_handler.register_tool(
        name="fetch_section",
        description="Fetch chunks from a specific document section by path.\n\n"
                   "When to use:\n"
                   "- Agent knows the exact section path (e.g., '/RAG/检索策略')\n"
                   "- Need all content from a specific section\n"
                   "- Building comprehensive answer about a topic\n\n"
                   "Args:\n"
                   "- section_path: Section path like '/Chapter 1/Section 1.1'\n"
                   "- include_neighbors: Also fetch prev/next chunks (default: true)",
        input_schema=FETCH_SECTION_SCHEMA,
        handler=fetch_section_handler,
    )
    logger.info("Registered MCP tool: fetch_section")

    protocol_handler.register_tool(
        name="fetch_neighbors",
        description="Fetch neighboring chunks for context.\n\n"
                   "When to use:\n"
                   "- Found a relevant chunk but need more context\n"
                   "- User asks 'what else is around this content?'\n"
                   "- Building narrative from partial match\n\n"
                   "Args:\n"
                   "- chunk_id: The chunk ID to get neighbors for\n"
                   "- window: Number of chunks before/after (default: 1)",
        input_schema=FETCH_NEIGHBORS_SCHEMA,
        handler=fetch_neighbors_handler,
    )
    logger.info("Registered MCP tool: fetch_neighbors")


# =============================================================================
# Structure-Aware Retrieval Tools
# =============================================================================


class FetchSectionTool:
    """Tool for fetching chunks by section path.

    This tool enables direct navigation to document sections
    without needing to search by keywords.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._settings = settings
        self._config = config or RetrievalConfig()
        self._vector_store = None
        self._current_collection = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _ensure_initialized(self, collection: str) -> None:
        """Initialize vector store for the given collection."""
        if self._current_collection == collection and self._vector_store is not None:
            return

        self._vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        self._current_collection = collection

    async def execute(
        self,
        section_path: str,
        collection: Optional[str] = None,
        include_neighbors: bool = True,
        max_chunks: int = 10,
    ) -> MCPToolResponse:
        """Fetch chunks from a specific section.

        Args:
            section_path: Section path to fetch
            collection: Collection name
            include_neighbors: Include neighbor chunks
            max_chunks: Maximum chunks to return

        Returns:
            MCPToolResponse with section chunks
        """
        effective_collection = collection or self._config.default_collection

        logger.info(
            f"fetch_section: section_path='{section_path}', "
            f"collection={effective_collection}"
        )

        try:
            await asyncio.to_thread(
                self._ensure_initialized, effective_collection
            )

            def _fetch():
                # Query by section_path filter
                # Note: ChromaDB's $contains may not work well with Chinese,
                # so we get all and filter in Python
                all_results = self._vector_store.collection.get(
                    limit=1000,
                    include=["metadatas", "documents"],
                )

                # Filter chunks matching the section path
                matching_chunks = []
                for i, chunk_id in enumerate(all_results.get("ids", [])):
                    meta = all_results.get("metadatas", [])[i] if i < len(all_results.get("metadatas", [])) else {}
                    doc = all_results.get("documents", [])[i] if i < len(all_results.get("documents", [])) else ""
                    sp = meta.get("section_path", "")

                    # Check if section_path contains the search term
                    if section_path in sp or sp.endswith(section_path):
                        matching_chunks.append({
                            "chunk_id": chunk_id,
                            "text": doc,
                            "metadata": meta,
                        })

                return matching_chunks

            matching_chunks = await asyncio.to_thread(_fetch)

            if not matching_chunks:
                return MCPToolResponse(
                    content=f"No chunks found for section path: {section_path}",
                    is_empty=True,
                )

            # Use matching_chunks as chunks
            chunks = matching_chunks

            # Collect neighbor IDs if requested
            if include_neighbors:
                neighbor_ids = set()
                for chunk in chunks:
                    prev_id = chunk.get("metadata", {}).get("prev_chunk_id")
                    next_id = chunk.get("metadata", {}).get("next_chunk_id")
                    if prev_id:
                        neighbor_ids.add(prev_id)
                    if next_id:
                        neighbor_ids.add(next_id)

                # Fetch neighbors
                if neighbor_ids:
                    def _fetch_neighbors():
                        return self._vector_store.collection.get(
                            ids=list(neighbor_ids),
                            include=["metadatas", "documents"],
                        )

                    neighbor_results = await asyncio.to_thread(_fetch_neighbors)

                    for i, chunk_id in enumerate(neighbor_results.get("ids", [])):
                        doc = neighbor_results.get("documents", [])[i] if i < len(neighbor_results.get("documents", [])) else ""
                        metadata = neighbor_results.get("metadatas", [])[i] if i < len(neighbor_results.get("metadatas", [])) else {}

                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": doc,
                            "metadata": {**metadata, "is_neighbor": True},
                        })

            # Sort by chunk_index if available
            chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

            # Limit to max_chunks
            chunks = chunks[:max_chunks]

            response_data = {
                "method": "fetch_section",
                "section_path": section_path,
                "collection": effective_collection,
                "total_chunks": len(chunks),
                "chunks": chunks,
            }

            return MCPToolResponse(
                content=json.dumps(response_data, ensure_ascii=False, indent=2),
                metadata=response_data,
                is_empty=len(chunks) == 0,
            )

        except Exception as e:
            logger.exception(f"fetch_section error: {e}")
            return MCPToolResponse(
                content=f"Error fetching section: {e}",
                is_empty=False,
            )


class FetchNeighborsTool:
    """Tool for fetching neighboring chunks.

    This tool enables context retrieval around a specific chunk.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._settings = settings
        self._config = config or RetrievalConfig()
        self._vector_store = None
        self._current_collection = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _ensure_initialized(self, collection: str) -> None:
        """Initialize vector store for the given collection."""
        if self._current_collection == collection and self._vector_store is not None:
            return

        self._vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        self._current_collection = collection

    async def execute(
        self,
        chunk_id: str,
        collection: Optional[str] = None,
        window: int = 1,
    ) -> MCPToolResponse:
        """Fetch neighboring chunks.

        Args:
            chunk_id: Center chunk ID
            collection: Collection name
            window: Number of chunks before/after

        Returns:
            MCPToolResponse with neighbor chunks
        """
        effective_collection = collection or self._config.default_collection

        logger.info(
            f"fetch_neighbors: chunk_id='{chunk_id}', "
            f"window={window}, collection={effective_collection}"
        )

        try:
            await asyncio.to_thread(
                self._ensure_initialized, effective_collection
            )

            def _fetch_center():
                return self._vector_store.collection.get(
                    ids=[chunk_id],
                    include=["metadatas", "documents"],
                )

            center_result = await asyncio.to_thread(_fetch_center)

            if not center_result or not center_result.get("ids"):
                return MCPToolResponse(
                    content=f"Chunk not found: {chunk_id}",
                    is_empty=True,
                )

            # Get center chunk metadata
            center_metadata = center_result.get("metadatas", [{}])[0]
            center_doc = center_result.get("documents", [""])[0]

            chunks = [{
                "chunk_id": chunk_id,
                "text": center_doc,
                "metadata": center_metadata,
                "position": "center",
            }]

            # Collect neighbors by following prev/next links
            collected_ids = {chunk_id}
            frontier = [chunk_id]
            current_window = 0

            while current_window < window and frontier:
                new_frontier = []
                for fid in frontier:
                    # Get chunk's neighbors
                    def _get_meta(cid):
                        result = self._vector_store.collection.get(
                            ids=[cid],
                            include=["metadatas"],
                        )
                        return result.get("metadatas", [{}])[0] if result.get("ids") else {}

                    meta = _get_meta(fid)
                    prev_id = meta.get("prev_chunk_id")
                    next_id = meta.get("next_chunk_id")

                    if prev_id and prev_id not in collected_ids:
                        collected_ids.add(prev_id)
                        new_frontier.append(prev_id)

                    if next_id and next_id not in collected_ids:
                        collected_ids.add(next_id)
                        new_frontier.append(next_id)

                frontier = new_frontier
                current_window += 1

            # Fetch all collected neighbors
            if len(collected_ids) > 1:
                def _fetch_all():
                    return self._vector_store.collection.get(
                        ids=list(collected_ids),
                        include=["metadatas", "documents"],
                    )

                all_results = await asyncio.to_thread(_fetch_all)

                for i, cid in enumerate(all_results.get("ids", [])):
                    if cid == chunk_id:
                        continue  # Already added as center

                    doc = all_results.get("documents", [])[i] if i < len(all_results.get("documents", [])) else ""
                    metadata = all_results.get("metadatas", [])[i] if i < len(all_results.get("metadatas", [])) else {}

                    chunks.append({
                        "chunk_id": cid,
                        "text": doc,
                        "metadata": metadata,
                        "position": "neighbor",
                    })

            # Sort by chunk_index
            chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

            response_data = {
                "method": "fetch_neighbors",
                "center_chunk_id": chunk_id,
                "window": window,
                "collection": effective_collection,
                "total_chunks": len(chunks),
                "chunks": chunks,
            }

            return MCPToolResponse(
                content=json.dumps(response_data, ensure_ascii=False, indent=2),
                metadata=response_data,
                is_empty=len(chunks) == 0,
            )

        except Exception as e:
            logger.exception(f"fetch_neighbors error: {e}")
            return MCPToolResponse(
                content=f"Error fetching neighbors: {e}",
                is_empty=False,
            )


# Tool instances
_fetch_section_tool: Optional[FetchSectionTool] = None
_fetch_neighbors_tool: Optional[FetchNeighborsTool] = None


def get_fetch_section_tool(settings: Optional[Settings] = None) -> FetchSectionTool:
    """Get or create the fetch_section tool instance."""
    global _fetch_section_tool
    if _fetch_section_tool is None:
        _fetch_section_tool = FetchSectionTool(settings=settings)
    return _fetch_section_tool


def get_fetch_neighbors_tool(settings: Optional[Settings] = None) -> FetchNeighborsTool:
    """Get or create the fetch_neighbors tool instance."""
    global _fetch_neighbors_tool
    if _fetch_neighbors_tool is None:
        _fetch_neighbors_tool = FetchNeighborsTool(settings=settings)
    return _fetch_neighbors_tool


async def fetch_section_handler(
    section_path: str,
    collection: str = "default",
    include_neighbors: bool = True,
    max_chunks: int = 10,
) -> types.CallToolResult:
    """MCP handler for fetch_section."""
    tool = get_fetch_section_tool()
    try:
        result = await tool.execute(
            section_path=section_path,
            collection=collection,
            include_neighbors=include_neighbors,
            max_chunks=max_chunks,
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=result.is_empty,
        )
    except Exception as e:
        logger.exception(f"fetch_section handler error: {e}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


async def fetch_neighbors_handler(
    chunk_id: str,
    collection: str = "default",
    window: int = 1,
) -> types.CallToolResult:
    """MCP handler for fetch_neighbors."""
    tool = get_fetch_neighbors_tool()
    try:
        result = await tool.execute(
            chunk_id=chunk_id,
            collection=collection,
            window=window,
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=result.is_empty,
        )
    except Exception as e:
        logger.exception(f"fetch_neighbors handler error: {e}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )