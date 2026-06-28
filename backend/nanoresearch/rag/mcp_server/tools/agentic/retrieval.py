"""Agentic retrieval tools.

Provides structure-aware retrieval tools used internally by the RAG loop:
- FetchSectionTool: fetch chunks by document section path
- FetchNeighborsTool: fetch neighboring chunks for context expansion

These tools are NOT registered as MCP tools exposed to the outer Agent.
They are called directly by the internal RAG loop (internal_loop/tools.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

from nanoresearch.rag.core.settings import Settings, load_settings
from nanoresearch.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
from nanoresearch.rag.core.response.response_builder import MCPToolResponse

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def _batch_fetch_chunks_cached(
    vector_store,
    namespace: str,
    chunk_ids: list[str],
) -> list[dict]:
    """Fetch chunks by ID with Redis chunk cache. Returns list of {chunk_id, text, metadata}."""
    if not chunk_ids:
        return []

    try:
        from nanoresearch.bus.redis_client import get_redis
        from nanoresearch.bus.redis_keys import RedisKeys
        redis = get_redis()
        pipe = redis.pipeline()
        for cid in chunk_ids:
            pipe.hgetall(RedisKeys.chunk(namespace, cid))
        results = await pipe.execute()
        cached: dict[str, dict] = {}
        for cid, h in zip(chunk_ids, results):
            if h:
                cached[cid] = {
                    "chunk_id": cid,
                    "text": h.get("text", ""),
                    "metadata": json.loads(h.get("metadata") or "{}"),
                }
        uncached_ids = [cid for cid in chunk_ids if cid not in cached]
        hit_count = len(cached)
        miss_count = len(uncached_ids)
        if hit_count:
            logger.debug(
                "chunk cache: %d hit(s), %d miss(es) for namespace %s",
                hit_count, miss_count, namespace,
                extra={"event": "chunk_cache_hit", "cache_layer": "chunk_cache",
                       "hit_count": hit_count, "miss_count": miss_count},
            )
        if miss_count:
            logger.debug(
                "chunk cache: %d miss(es) for namespace %s",
                miss_count, namespace,
                extra={"event": "chunk_cache_miss", "cache_layer": "chunk_cache",
                       "miss_count": miss_count},
            )
    except Exception:
        cached = {}
        uncached_ids = list(chunk_ids)

    if not uncached_ids:
        return [cached[cid] for cid in chunk_ids]

    def _fetch():
        return vector_store.collection.get(
            ids=uncached_ids,
            include=["metadatas", "documents"],
        )

    raw = await asyncio.to_thread(_fetch)
    fetched = []
    for i, cid in enumerate(raw.get("ids", [])):
        doc = raw.get("documents", [])[i] if i < len(raw.get("documents", [])) else ""
        meta = raw.get("metadatas", [])[i] if i < len(raw.get("metadatas", [])) else {}
        fetched.append({"chunk_id": cid, "text": doc, "metadata": meta})

    try:
        from nanoresearch.bus.redis_client import get_redis
        from nanoresearch.bus.redis_keys import RedisKeys
        redis = get_redis()
        pipe = redis.pipeline()
        for entry in fetched:
            ck = RedisKeys.chunk(namespace, entry["chunk_id"])
            pipe.hset(ck, mapping={
                "text": entry["text"],
                "metadata": json.dumps(entry.get("metadata", {}), ensure_ascii=False),
            })
            pipe.expire(ck, RedisKeys.CHUNK_TTL)
        await pipe.execute()
    except Exception:
        pass

    combined: list[dict] = []
    for cid in chunk_ids:
        entry = cached.get(cid) or next((e for e in fetched if e["chunk_id"] == cid), None)
        if entry:
            combined.append(entry)
    return combined


# =============================================================================
# Schemas (kept for potential future use)
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


# =============================================================================
# Structure-Aware Retrieval Tools
# =============================================================================

class FetchSectionTool:
    """Fetch chunks from a specific document section by path.

    Used internally by the RAG loop for comparison queries:
    when intent == "comparison", fetches structurally relevant sections
    before fuse/verify to ensure comprehensive coverage.
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
        effective_collection = collection or self._config.default_collection

        logger.info(
            f"fetch_section: section_path='{section_path}', "
            f"collection={effective_collection}"
        )

        try:
            await asyncio.to_thread(self._ensure_initialized, effective_collection)

            def _fetch():
                all_results = self._vector_store.collection.get(
                    limit=1000,
                    include=["metadatas", "documents"],
                )
                matching_chunks = []
                for i, chunk_id in enumerate(all_results.get("ids", [])):
                    meta = all_results.get("metadatas", [])[i] if i < len(all_results.get("metadatas", [])) else {}
                    doc = all_results.get("documents", [])[i] if i < len(all_results.get("documents", [])) else ""
                    sp = meta.get("section_path", "")
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

            chunks = matching_chunks

            if include_neighbors:
                neighbor_ids = set()
                for chunk in chunks:
                    prev_id = chunk.get("metadata", {}).get("prev_chunk_id")
                    next_id = chunk.get("metadata", {}).get("next_chunk_id")
                    if prev_id:
                        neighbor_ids.add(prev_id)
                    if next_id:
                        neighbor_ids.add(next_id)

                if neighbor_ids:
                    neighbor_list = await _batch_fetch_chunks_cached(
                        self._vector_store, effective_collection, list(neighbor_ids),
                    )
                    for entry in neighbor_list:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": doc,
                            "metadata": {**metadata, "is_neighbor": True},
                        })

            chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
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
    """Fetch neighboring chunks for context expansion.

    Used internally by the RAG loop as a fallback when verification fails:
    expands the chunk pool by fetching context around retrieved chunks.
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
        effective_collection = collection or self._config.default_collection

        logger.info(
            f"fetch_neighbors: chunk_id='{chunk_id}', "
            f"window={window}, collection={effective_collection}"
        )

        try:
            await asyncio.to_thread(self._ensure_initialized, effective_collection)

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

            center_metadata = center_result.get("metadatas", [{}])[0]
            center_doc = center_result.get("documents", [""])[0]

            chunks = [{
                "chunk_id": chunk_id,
                "text": center_doc,
                "metadata": center_metadata,
                "position": "center",
            }]

            collected_ids = {chunk_id}
            frontier = [chunk_id]
            current_window = 0

            while current_window < window and frontier:
                new_frontier = []
                for fid in frontier:
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

            if len(collected_ids) > 1:
                all_chunks = await _batch_fetch_chunks_cached(
                    self._vector_store, effective_collection, list(collected_ids),
                )
                for entry in all_chunks:
                    if entry["chunk_id"] == chunk_id:
                        continue
                    chunks.append({
                        "chunk_id": entry["chunk_id"],
                        "text": entry["text"],
                        "metadata": entry["metadata"],
                        "position": "neighbor",
                    })

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


# =============================================================================
# Module-level Tool Instances
# =============================================================================

_fetch_section_tool: Optional[FetchSectionTool] = None
_fetch_neighbors_tool: Optional[FetchNeighborsTool] = None


def get_fetch_section_tool(settings: Optional[Settings] = None) -> FetchSectionTool:
    global _fetch_section_tool
    if _fetch_section_tool is None:
        _fetch_section_tool = FetchSectionTool(settings=settings)
    return _fetch_section_tool


def get_fetch_neighbors_tool(settings: Optional[Settings] = None) -> FetchNeighborsTool:
    global _fetch_neighbors_tool
    if _fetch_neighbors_tool is None:
        _fetch_neighbors_tool = FetchNeighborsTool(settings=settings)
    return _fetch_neighbors_tool


# =============================================================================
# Handlers (for potential future use)
# =============================================================================

async def fetch_section_handler(
    section_path: str,
    collection: str = "default",
    include_neighbors: bool = True,
    max_chunks: int = 10,
) -> types.CallToolResult:
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


# =============================================================================
# Tool Registration (no MCP tools registered — these are internal-loop tools)
# =============================================================================

def register_tools(protocol_handler) -> None:
    """No MCP tools registered from this module.

    FetchSectionTool and FetchNeighborsTool are internal RAG loop tools,
    called directly by internal_loop/tools.py, not exposed to the outer Agent.
    """
    pass
