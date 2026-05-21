"""Fusion tool for Agentic RAG.

Allows Agent to control how dense and sparse results are combined.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from nanobot.rag.core.types_agentic import FusionResult
from nanobot.rag.mcp_server.tools.agentic.shared import (
    build_json_response,
    build_structured_response,
    results_to_dict_list,
)

logger = logging.getLogger(__name__)


class FuseResultsTool:
    """MCP tool for fusing dense and sparse retrieval results.

    Allows Agent to:
    - Combine results from retrieve_dense and retrieve_sparse
    - Control fusion method (RRF, weighted, etc.)
    - Adjust fusion parameters
    """

    def __init__(self):
        self._rrf_k = 60

    @property
    def name(self) -> str:
        return "fuse_results"

    @property
    def description(self) -> str:
        return """Fuse dense and sparse retrieval results using configurable methods.

Args:
    dense_results: Results from retrieve_dense (JSON list)
    sparse_results: Results from retrieve_sparse (JSON list)
    method: Fusion method - "rrf" (default), "weighted", "interleave"
    top_k: Number of results to return (default: 10)
    rrf_k: RRF parameter k (default: 60, only for RRF method)
    dense_weight: Weight for dense results (default: 0.5, for weighted method)
    sparse_weight: Weight for sparse results (default: 0.5, for weighted method)

Returns:
    JSON with fused results and fusion metadata."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dense_results": {
                    "type": "string",
                    "description": "JSON string of dense retrieval results",
                },
                "sparse_results": {
                    "type": "string",
                    "description": "JSON string of sparse retrieval results",
                },
                "method": {
                    "type": "string",
                    "enum": ["rrf", "weighted", "interleave"],
                    "default": "rrf",
                    "description": "Fusion method to use",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of results to return",
                },
                "rrf_k": {
                    "type": "integer",
                    "default": 60,
                    "description": "RRF parameter k (only for RRF method)",
                },
                "dense_weight": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Weight for dense scores (weighted method)",
                },
                "sparse_weight": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Weight for sparse scores (weighted method)",
                },
            },
            "required": ["dense_results", "sparse_results"],
        }

    async def execute(
        self,
        dense_results: str,
        sparse_results: str,
        method: str = "rrf",
        top_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> "MCPToolResponse":
        """Execute the fusion tool.

        Args:
            dense_results: JSON string of dense results
            sparse_results: JSON string of sparse results
            method: Fusion method
            top_k: Number of results to return
            rrf_k: RRF parameter
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores

        Returns:
            MCPToolResponse with fused results
        """
        import json

        # Parse input results
        try:
            dense_list = json.loads(dense_results) if dense_results else []
            sparse_list = json.loads(sparse_results) if sparse_results else []
        except json.JSONDecodeError as e:
            return build_json_response(
                {"error": f"Invalid JSON input: {e}"},
                is_empty=True,
            )

        if not dense_list and not sparse_list:
            return build_json_response(
                {"error": "No results to fuse"},
                is_empty=True,
            )

        # Normalize results to dict format
        dense_normalized = self._normalize_results(dense_list, "dense")
        sparse_normalized = self._normalize_results(sparse_list, "sparse")

        # Apply fusion method
        if method == "rrf":
            fused = self._rrf_fusion(dense_normalized, sparse_normalized, rrf_k, top_k)
        elif method == "weighted":
            fused = self._weighted_fusion(
                dense_normalized, sparse_normalized, dense_weight, sparse_weight, top_k
            )
        elif method == "interleave":
            fused = self._interleave_fusion(dense_normalized, sparse_normalized, top_k)
        else:
            fused = self._rrf_fusion(dense_normalized, sparse_normalized, rrf_k, top_k)

        # Build result
        result = FusionResult(
            results=fused,
            method=method,
            weights={"dense": dense_weight, "sparse": sparse_weight} if method == "weighted" else None,
            rrf_k=rrf_k if method == "rrf" else None,
            input_counts={"dense": len(dense_normalized), "sparse": len(sparse_normalized)},
            unique_chunks=len(set(r["chunk_id"] for r in fused)),
        )

        return build_json_response(result.to_dict())

    def _normalize_results(
        self, results: List[Any], source: str
    ) -> List[Dict[str, Any]]:
        """Normalize results to standard dict format.

        Args:
            results: List of results (dict or object)
            source: Source type for scoring normalization

        Returns:
            Normalized list of dicts
        """
        normalized = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                chunk_id = r.get("chunk_id", f"{source}_{i}")
                score = float(r.get("score", 0.0))
                text = r.get("text", "")
                metadata = r.get("metadata", {})
            else:
                chunk_id = getattr(r, "chunk_id", f"{source}_{i}")
                score = float(getattr(r, "score", 0.0))
                text = getattr(r, "text", "")
                metadata = getattr(r, "metadata", {})

            normalized.append({
                "chunk_id": chunk_id,
                "score": score,
                "text": text,
                "metadata": metadata,
                "source": source,
            })

        return normalized

    def _rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each list the doc appears in.

        Args:
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results
            k: RRF parameter
            top_k: Number of results to return

        Returns:
            Fused and sorted results
        """
        # Build rank maps
        dense_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}
        sparse_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(sparse_results)}

        # Combine all unique chunk IDs
        all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Calculate RRF scores
        scores = {}
        chunk_data = {}

        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                rrf_score += 1.0 / (k + sparse_ranks[chunk_id])

            scores[chunk_id] = rrf_score

            # Get chunk data (prefer dense, fallback to sparse)
            if chunk_id in dense_ranks:
                for r in dense_results:
                    if r["chunk_id"] == chunk_id:
                        chunk_data[chunk_id] = r
                        break
            else:
                for r in sparse_results:
                    if r["chunk_id"] == chunk_id:
                        chunk_data[chunk_id] = r
                        break

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results
        fused = []
        for chunk_id in sorted_ids[:top_k]:
            data = chunk_data[chunk_id].copy()
            data["fusion_score"] = scores[chunk_id]
            data["fusion_method"] = "rrf"
            fused.append(data)

        return fused

    def _weighted_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float,
        sparse_weight: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Weighted score fusion.

        Combined score = dense_weight * dense_score + sparse_weight * sparse_score.

        Args:
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores
            top_k: Number of results to return

        Returns:
            Fused and sorted results
        """
        # Normalize scores within each list
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)

        # Build score maps
        dense_scores = {r["chunk_id"]: r["normalized_score"] for r in dense_normalized}
        sparse_scores = {r["chunk_id"]: r["normalized_score"] for r in sparse_normalized}

        # Combine all unique chunk IDs
        all_chunk_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

        # Calculate weighted scores
        scores = {}
        chunk_data = {}

        for chunk_id in all_chunk_ids:
            weighted_score = 0.0
            if chunk_id in dense_scores:
                weighted_score += dense_weight * dense_scores[chunk_id]
            if chunk_id in sparse_scores:
                weighted_score += sparse_weight * sparse_scores[chunk_id]

            scores[chunk_id] = weighted_score

            # Get chunk data
            if chunk_id in dense_scores:
                for r in dense_normalized:
                    if r["chunk_id"] == chunk_id:
                        chunk_data[chunk_id] = r
                        break
            else:
                for r in sparse_normalized:
                    if r["chunk_id"] == chunk_id:
                        chunk_data[chunk_id] = r
                        break

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results
        fused = []
        for chunk_id in sorted_ids[:top_k]:
            data = chunk_data[chunk_id].copy()
            data["fusion_score"] = scores[chunk_id]
            data["fusion_method"] = "weighted"
            fused.append(data)

        return fused

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize scores to [0, 1] range using min-max scaling.

        Args:
            results: Results with scores

        Returns:
            Results with normalized_score added
        """
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        for r in results:
            r["normalized_score"] = (r["score"] - min_score) / score_range

        return results

    def _interleave_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Interleave fusion - alternate between dense and sparse results.

        Args:
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results
            top_k: Number of results to return

        Returns:
            Interleaved results
        """
        fused = []
        seen_ids = set()
        i = 0

        while len(fused) < top_k:
            # Add from dense
            if i < len(dense_results):
                r = dense_results[i]
                if r["chunk_id"] not in seen_ids:
                    r_copy = r.copy()
                    r_copy["fusion_method"] = "interleave"
                    r_copy["fusion_rank"] = i
                    fused.append(r_copy)
                    seen_ids.add(r["chunk_id"])

            if len(fused) >= top_k:
                break

            # Add from sparse
            if i < len(sparse_results):
                r = sparse_results[i]
                if r["chunk_id"] not in seen_ids:
                    r_copy = r.copy()
                    r_copy["fusion_method"] = "interleave"
                    r_copy["fusion_rank"] = i
                    fused.append(r_copy)
                    seen_ids.add(r["chunk_id"])

            i += 1

            # Stop if both lists exhausted
            if i >= len(dense_results) and i >= len(sparse_results):
                break

        return fused


# MCP Tool Handler
async def fuse_results_handler(
    dense_results: str,
    sparse_results: str,
    method: str = "rrf",
    top_k: int = 10,
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> "MCPToolResponse":
    """Handler for fuse_results MCP tool."""
    tool = FuseResultsTool()
    return await tool.execute(
        dense_results=dense_results,
        sparse_results=sparse_results,
        method=method,
        top_k=top_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )


def register_tools(protocol_handler: Any) -> None:
    """Register fusion tools with the protocol handler.

    Args:
        protocol_handler: The MCP protocol handler instance
    """
    tool = FuseResultsTool()
    protocol_handler.register_tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        handler=fuse_results_handler,
    )
    logger.info(f"Registered agentic tool: {tool.name}")
