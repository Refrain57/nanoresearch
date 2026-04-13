"""Reranking tool for Agentic RAG.

Allows Agent to apply reranking to retrieval results.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from nanobot.rag.mcp_server.tools.agentic.shared import (
    build_json_response,
    results_to_dict_list,
)

logger = logging.getLogger(__name__)


class RerankResultsTool:
    """MCP tool for reranking retrieval results.

    Allows Agent to:
    - Apply reranking to improve result relevance
    - Configure reranking parameters
    - Use different reranking strategies
    """

    def __init__(self):
        self._reranker = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize the reranker if not already done."""
        if self._initialized:
            return

        try:
            from nanobot.rag.core.query_engine.reranker import CoreReranker
            from nanobot.rag.core.settings import get_settings

            settings = get_settings()
            self._reranker = CoreReranker(settings.reranker.model_dump())
            self._initialized = True
            logger.info("Reranker initialized for agentic tool")
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            self._initialized = True  # Mark as attempted

    @property
    def name(self) -> str:
        return "rerank_results"

    @property
    def description(self) -> str:
        return """Rerank retrieval results to improve relevance.

Args:
    results: JSON string of retrieval results to rerank
    query: The original query for relevance scoring
    top_k: Number of results to return after reranking (default: 10)
    rerank_strategy: Strategy - "default", "diverse", "fresh" (default: "default")

Returns:
    JSON with reranked results and reranking metadata."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "string",
                    "description": "JSON string of retrieval results to rerank",
                },
                "query": {
                    "type": "string",
                    "description": "The original query for relevance scoring",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of results to return after reranking",
                },
                "rerank_strategy": {
                    "type": "string",
                    "enum": ["default", "diverse", "fresh"],
                    "default": "default",
                    "description": "Reranking strategy to use",
                },
            },
            "required": ["results", "query"],
        }

    async def execute(
        self,
        results: str,
        query: str,
        top_k: int = 10,
        rerank_strategy: str = "default",
    ) -> "MCPToolResponse":
        """Execute the reranking tool.

        Args:
            results: JSON string of results to rerank
            query: The original query
            top_k: Number of results to return
            rerank_strategy: Reranking strategy

        Returns:
            MCPToolResponse with reranked results
        """
        import asyncio

        # Parse input results
        try:
            results_list = json.loads(results) if results else []
        except json.JSONDecodeError as e:
            return build_json_response(
                {"error": f"Invalid JSON input: {e}"},
                is_empty=True,
            )

        if not results_list:
            return build_json_response(
                {"error": "No results to rerank"},
                is_empty=True,
            )

        # Initialize reranker
        await asyncio.to_thread(self._ensure_initialized)

        if not self._reranker:
            return build_json_response({
                "error": "Reranker not available",
                "original_results": results_list[:top_k],
            })

        try:
            # Normalize results
            normalized = self._normalize_results(results_list)

            # Apply reranking
            reranked = await asyncio.to_thread(
                self._rerank,
                query=query,
                results=normalized,
                top_k=top_k,
                strategy=rerank_strategy,
            )

            return build_json_response({
                "query": query,
                "rerank_strategy": rerank_strategy,
                "total_input": len(results_list),
                "total_output": len(reranked),
                "results": reranked,
            })

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return build_json_response({
                "error": f"Reranking failed: {e}",
                "original_results": results_list[:top_k],
            })

    def _normalize_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Normalize results to standard dict format.

        Args:
            results: List of results (dict or object)

        Returns:
            Normalized list of dicts
        """
        normalized = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                normalized.append({
                    "chunk_id": r.get("chunk_id", f"result_{i}"),
                    "score": float(r.get("score", 0.0)),
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata", {}),
                })
            else:
                normalized.append({
                    "chunk_id": getattr(r, "chunk_id", f"result_{i}"),
                    "score": float(getattr(r, "score", 0.0)),
                    "text": getattr(r, "text", ""),
                    "metadata": getattr(r, "metadata", {}),
                })
        return normalized

    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        strategy: str,
    ) -> List[Dict[str, Any]]:
        """Apply reranking using CoreReranker.

        Args:
            query: The query
            results: Results to rerank
            top_k: Number to return
            strategy: Reranking strategy

        Returns:
            Reranked results
        """
        # Build documents for reranker
        documents = []
        for r in results:
            text = r.get("text", "")
            if not text:
                # Try to get from metadata
                metadata = r.get("metadata", {})
                text = metadata.get("text", "")
            documents.append(text)

        # Call reranker
        rerank_results = self._reranker.rerank(
            query=query,
            documents=documents,
            top_k=min(top_k, len(documents)),
        )

        # Build final results
        reranked = []
        for rr in rerank_results:
            idx = rr.index if hasattr(rr, "index") else rr.get("index", 0)
            score = rr.relevance_score if hasattr(rr, "relevance_score") else rr.get("relevance_score", 0.0)

            original = results[idx].copy()
            original["rerank_score"] = float(score)
            original["original_rank"] = idx + 1
            reranked.append(original)

        return reranked


# MCP Tool Handler
async def rerank_results_handler(
    results: str,
    query: str,
    top_k: int = 10,
    rerank_strategy: str = "default",
) -> "MCPToolResponse":
    """Handler for rerank_results MCP tool."""
    tool = RerankResultsTool()
    return await tool.execute(
        results=results,
        query=query,
        top_k=top_k,
        rerank_strategy=rerank_strategy,
    )


def register_tools(protocol_handler: Any) -> None:
    """Register reranking tools with the protocol handler.

    Args:
        protocol_handler: The MCP protocol handler instance
    """
    tool = RerankResultsTool()
    protocol_handler.register_tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        handler=rerank_results_handler,
    )
    logger.info(f"Registered agentic tool: {tool.name}")
