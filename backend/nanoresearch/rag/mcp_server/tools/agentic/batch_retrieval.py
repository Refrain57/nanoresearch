"""Batch retrieval tools for concurrent multi-round retrieval.

This module provides tools for:
- execute_retrieval_batch: Execute multiple retrieval tasks concurrently
- fuse_and_fetch_round: Fuse results and return full text
- get_round_status: Query round status

Design Principles:
- Auto-create round: execute_retrieval_batch creates round if not provided
- Concurrent execution: Uses asyncio.gather for parallel retrieval
- Summary-only responses: Don't return full text during batch execution
- Full text at fusion: Only fuse_and_fetch_round returns complete text
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

from nanoresearch.rag.core.settings import Settings, load_settings
from nanoresearch.rag.core.types import RetrievalResult
from nanoresearch.rag.core.query_engine.query_processor import QueryProcessor
from nanoresearch.rag.ingestion.storage.bm25_indexer import BM25Indexer
from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory
from nanoresearch.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
from nanoresearch.rag.core.settings import resolve_path
from nanoresearch.rag.core.response.response_builder import MCPToolResponse
from nanoresearch.rag.mcp_server.tools.agentic.round_state import (
    RoundStateManager,
    RetrievalTask,
    get_round_manager,
)
from nanoresearch.rag.mcp_server.tools.agentic.shared import (
    results_to_dict_list,
    build_json_response,
    format_markdown_results,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Schemas
# =============================================================================

EXECUTE_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "round_id": {
            "type": "string",
            "description": "Existing round ID to add tasks to. "
                          "If not provided, a new round will be auto-created.",
        },
        "tasks": {
            "type": "array",
            "description": "List of retrieval tasks to execute concurrently.",
            "items": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for this task",
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Retrieval strategy: 'dense', 'sparse', or 'hybrid'",
                        "enum": ["dense", "sparse", "hybrid"],
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results for this task (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query", "strategy"],
            },
        },
        "top_k": {
            "type": "integer",
            "description": "Default top_k for all tasks if not specified per-task (default: 10)",
            "default": 10,
        },
        "collection": {
            "type": "string",
            "description": "Collection name (default: default)",
            "default": "default",
        },
    },
    "required": ["tasks"],
}

FUSE_ROUND_SCHEMA = {
    "type": "object",
    "properties": {
        "round_id": {
            "type": "string",
            "description": "Round ID to fuse and return results from",
        },
        "strategy": {
            "type": "string",
            "description": "Fusion strategy: 'rrf', 'weighted', or 'interleave' (default: rrf)",
            "enum": ["rrf", "weighted", "interleave"],
            "default": "rrf",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum results to return (default: 20)",
            "default": 20,
        },
    },
    "required": ["round_id"],
}

GET_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "round_id": {
            "type": "string",
            "description": "Round ID to query status for",
        },
    },
    "required": ["round_id"],
}


# =============================================================================
# Concurrent Retrieval Engine
# =============================================================================

class ConcurrentRetrievalEngine:
    """Engine for executing multiple retrieval tasks concurrently.

    This class manages the lifecycle of concurrent retrieval:
    - Creates/fetches rounds
    - Executes tasks in parallel using asyncio.gather
    - Stores results in round state
    - Returns summaries (not full text)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        default_collection: str = "default",
    ) -> None:
        self._settings = settings
        self._default_collection = default_collection
        self._query_processor = QueryProcessor()

        # Cache retriever components to avoid creating multiple ChromaDB clients
        # for the same collection/strategy combination (ChromaDB 1.x doesn't support this)
        self._retriever_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()  # Thread-safe cache access

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _get_or_create_retriever(
        self,
        strategy: str,
        collection: str,
    ) -> tuple[Any, Any, Any]:
        """Get or create retriever components based on strategy.

        Uses caching with thread lock to avoid creating multiple ChromaDB clients
        for the same collection/strategy combination (ChromaDB 1.x doesn't support this).

        Returns:
            Tuple of (retriever, vector_store, query_processor)
        """
        cache_key = f"{collection}:{strategy}"

        # Thread-safe cache access
        with self._cache_lock:
            if cache_key in self._retriever_cache:
                cached = self._retriever_cache[cache_key]
                return cached["retriever"], cached["vector_store"], cached["query_processor"]

            # Create components (inside lock to prevent race condition)
            vector_store = VectorStoreFactory.create(
                self.settings,
                collection_name=collection,
            )

            query_processor = self._query_processor

            if strategy == "dense":
                embedding_client = EmbeddingFactory.create(self.settings)
                from nanoresearch.rag.core.query_engine.dense_retriever import create_dense_retriever
                retriever = create_dense_retriever(
                    settings=self.settings,
                    embedding_client=embedding_client,
                    vector_store=vector_store,
                )
            elif strategy == "sparse":
                bm25_indexer = BM25Indexer(
                    index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{collection}"))
                )
                from nanoresearch.rag.core.query_engine.sparse_retriever import create_sparse_retriever
                retriever = create_sparse_retriever(
                    settings=self.settings,
                    bm25_indexer=bm25_indexer,
                    vector_store=vector_store,
                )
            else:  # hybrid
                embedding_client = EmbeddingFactory.create(self.settings)
                bm25_indexer = BM25Indexer(
                    index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{collection}"))
                )
                from nanoresearch.rag.core.query_engine.dense_retriever import create_dense_retriever
                from nanoresearch.rag.core.query_engine.sparse_retriever import create_sparse_retriever
                from nanoresearch.rag.core.query_engine.hybrid_search import create_hybrid_search

                dense_retriever = create_dense_retriever(
                    settings=self.settings,
                    embedding_client=embedding_client,
                    vector_store=vector_store,
                )
                sparse_retriever = create_sparse_retriever(
                    settings=self.settings,
                    bm25_indexer=bm25_indexer,
                    vector_store=vector_store,
                )
                hybrid_search = create_hybrid_search(
                    settings=self.settings,
                    query_processor=query_processor,
                    dense_retriever=dense_retriever,
                    sparse_retriever=sparse_retriever,
                )
                self._retriever_cache[cache_key] = {
                    "retriever": hybrid_search,
                    "vector_store": vector_store,
                    "query_processor": query_processor,
                }
                return hybrid_search, vector_store, query_processor

            # Cache for non-hybrid strategies
            self._retriever_cache[cache_key] = {
                "retriever": retriever,
                "vector_store": vector_store,
                "query_processor": query_processor,
            }

            return retriever, vector_store, query_processor

    def _execute_single_task(
        self,
        task: Dict[str, Any],
        collection: str,
    ) -> tuple[str, List[RetrievalResult]]:
        """Execute a single retrieval task (synchronous).

        Args:
            task: Task dict with query, strategy, top_k
            collection: Collection name

        Returns:
            Tuple of (task_id, results)
        """
        task_id = task.get("task_id", str(uuid.uuid4())[:8])
        query = task["query"]
        strategy = task["strategy"]
        top_k = task.get("top_k", 10)

        try:
            if strategy == "dense":
                retriever, _, _ = self._get_or_create_retriever(strategy, collection)
                results = retriever.retrieve(
                    query=query,
                    top_k=top_k,
                )
            elif strategy == "sparse":
                retriever, _, query_processor = self._get_or_create_retriever(strategy, collection)
                keywords = query_processor.process(query).keywords
                results = retriever.retrieve(
                    keywords=keywords,
                    top_k=top_k,
                    collection=collection,
                )
            else:  # hybrid
                retriever, _, _ = self._get_or_create_retriever(strategy, collection)
                results = retriever.search(
                    query=query,
                    top_k=top_k,
                )

            return task_id, results

        except Exception as e:
            logger.warning(f"Task {task_id} failed: {e}")
            return task_id, []

    async def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        round_id: Optional[str],
        top_k: int,
        collection: str,
    ) -> Dict[str, Any]:
        """Execute multiple retrieval tasks concurrently.

        Args:
            tasks: List of task dicts with query, strategy, top_k
            round_id: Existing round ID or None to auto-create
            top_k: Default top_k for tasks
            collection: Collection name

        Returns:
            Summary dict with round_id, task count, total chunks, etc.
        """
        manager = get_round_manager()

        # Auto-create round if not provided or doesn't exist
        if round_id is None:
            round_id = manager.create_round()
            logger.info(f"Auto-created round: {round_id}")
        elif round_id not in manager.list_rounds():
            # Round ID provided but doesn't exist in manager - create it
            manager.get_or_create_round(round_id)
            logger.info(f"Created missing round: {round_id}")

        # Register tasks and execute concurrently
        async_tasks = []
        task_ids = []

        for task_dict in tasks:
            # Create RetrievalTask for state tracking
            task_id = str(uuid.uuid4())[:8]
            task_ids.append(task_id)

            retrieval_task = RetrievalTask(
                task_id=task_id,
                query=task_dict["query"],
                strategy=task_dict["strategy"],
            )
            manager.add_task(round_id, retrieval_task)

            # Store task_id in dict for concurrent execution
            task_dict["task_id"] = task_id

            # Create async task for concurrent execution
            async_tasks.append(
                self._execute_task_async(
                    task=task_dict,
                    collection=collection,
                    round_id=round_id,
                )
            )

        # Execute all tasks concurrently
        logger.info(f"Executing {len(async_tasks)} tasks concurrently in round {round_id}")
        await asyncio.gather(*async_tasks, return_exceptions=True)

        # Get summary status
        status = manager.get_status(round_id)

        return {
            "round_id": round_id,
            "tasks_executed": len(tasks),
            "total_chunks": status.total_chunks if status else 0,
            "task_ids": task_ids,
            "is_new_round": round_id not in manager.list_rounds() or True,  # Simplified
            "summary": self._build_summary(status),
        }

    async def _execute_task_async(
        self,
        task: Dict[str, Any],
        collection: str,
        round_id: str,
    ) -> None:
        """Execute a single task asynchronously.

        Args:
            task: Task dict with query, strategy, top_k, task_id
            collection: Collection name
            round_id: Round ID to store results in
        """
        task_id = task["task_id"]

        try:
            # Run synchronous retrieval in thread pool
            _, results = await asyncio.to_thread(
                self._execute_single_task,
                task,
                collection,
            )

            # Store results in round state
            manager = get_round_manager()
            manager.add_results(round_id, task_id, results)

            logger.debug(f"Task {task_id} completed with {len(results)} results")

        except Exception as e:
            logger.error(f"Task {task_id} execution failed: {e}")
            # Store empty results on failure
            manager = get_round_manager()
            manager.add_results(round_id, task_id, [])

    def _build_summary(self, status: Any) -> str:
        """Build a human-readable summary from round status.

        Args:
            status: RoundStatus object

        Returns:
            Summary string
        """
        if status is None:
            return "Round not found or expired."

        task_summaries = []
        for task in status.tasks:
            task_summaries.append(
                f"- [{task['strategy']}] {task['query'][:50]}... ({task['chunks']} chunks)"
            )

        summary = f"Round {status.round_id}: {status.total_chunks} total chunks\n"
        summary += f"Tasks: {len(status.tasks)}, Fused: {status.fused}\n"
        summary += "\n".join(task_summaries)

        return summary


# =============================================================================
# Tool Classes
# =============================================================================

class ExecuteRetrievalBatchTool:
    """Tool for concurrent execution of multiple retrieval tasks.

    This tool:
    1. Auto-creates a round if round_id not provided
    2. Registers all tasks with the round
    3. Executes tasks concurrently using asyncio.gather
    4. Stores results in round state
    5. Returns a SUMMARY (not full text)

    Example:
        >>> tool = ExecuteRetrievalBatchTool()
        >>> result = await tool.execute(
        ...     tasks=[
        ...         {"query": "Transformer", "strategy": "dense", "top_k": 5},
        ...         {"query": "RNN", "strategy": "sparse", "top_k": 5},
        ...     ],
        ...     top_k=10,
        ...     collection="papers"
        ... )
        >>> print(result.metadata["round_id"])  # "abc123"
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings
        self._engine: Optional[ConcurrentRetrievalEngine] = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _get_engine(self) -> ConcurrentRetrievalEngine:
        """Get or create the retrieval engine."""
        if self._engine is None:
            self._engine = ConcurrentRetrievalEngine(settings=self.settings)
        return self._engine

    async def execute(
        self,
        round_id: Optional[str],
        tasks: List[Dict[str, Any]],
        top_k: int = 10,
        collection: str = "default",
    ) -> MCPToolResponse:
        """Execute batch retrieval tasks.

        Args:
            round_id: Existing round ID or None to auto-create
            tasks: List of task dicts with query, strategy, top_k
            top_k: Default top_k for all tasks
            collection: Collection name

        Returns:
            MCPToolResponse with round summary (not full text)
        """
        logger.info(
            f"execute_retrieval_batch: {len(tasks)} tasks, "
            f"round_id={round_id}, collection={collection}"
        )

        try:
            # Validate tasks
            for i, task in enumerate(tasks):
                if "query" not in task:
                    return MCPToolResponse(
                        content=f"Error: Task {i} missing required 'query' field",
                        is_empty=False,
                    )
                if "strategy" not in task:
                    return MCPToolResponse(
                        content=f"Error: Task {i} missing required 'strategy' field",
                        is_empty=False,
                    )
                if task["strategy"] not in ("dense", "sparse", "hybrid"):
                    return MCPToolResponse(
                        content=f"Error: Task {i} invalid strategy '{task['strategy']}'. "
                               f"Must be 'dense', 'sparse', or 'hybrid'",
                        is_empty=False,
                    )
                # Set default top_k if not specified
                if "top_k" not in task:
                    task["top_k"] = top_k

            # Execute batch
            engine = self._get_engine()
            result = await engine.execute_batch(
                tasks=tasks,
                round_id=round_id,
                top_k=top_k,
                collection=collection,
            )

            # Build response
            response_data = {
                "method": "execute_retrieval_batch",
                "round_id": result["round_id"],
                "tasks_executed": result["tasks_executed"],
                "total_chunks": result["total_chunks"],
                "task_ids": result["task_ids"],
                "collection": collection,
                "message": "Tasks executed. Use get_round_status to check progress, "
                         "or fuse_and_fetch_round to get full results.",
            }

            # Add markdown summary
            markdown = f"## Batch Retrieval Executed\n\n"
            markdown += f"**Round ID:** `{result['round_id']}`\n"
            markdown += f"**Tasks:** {result['tasks_executed']}\n"
            markdown += f"**Total Chunks:** {result['total_chunks']}\n"
            markdown += f"**Collection:** {collection}\n\n"
            markdown += f"### Next Steps:\n"
            markdown += f"1. Call `get_round_status('{result['round_id']}')` to check status\n"
            markdown += f"2. Call `fuse_and_fetch_round('{result['round_id']}')` to get full results\n"

            return MCPToolResponse(
                content=json.dumps(response_data, ensure_ascii=False, indent=2)
                + "\n\n"
                + markdown,
                metadata={
                    "round_id": result["round_id"],
                    "tasks_executed": result["tasks_executed"],
                    "total_chunks": result["total_chunks"],
                },
                is_empty=False,
            )

        except Exception as e:
            logger.exception(f"execute_retrieval_batch error: {e}")
            return MCPToolResponse(
                content=f"Error executing batch retrieval: {e}",
                is_empty=False,
            )


class FuseAndFetchRoundTool:
    """Tool for fusing round results and returning full text.

    This tool:
    1. Gets all results from the specified round
    2. Applies fusion strategy (RRF by default)
    3. Returns COMPLETE text for all results
    4. Marks round as fused

    Example:
        >>> tool = FuseAndFetchRoundTool()
        >>> result = await tool.execute(
        ...     round_id="abc123",
        ...     strategy="rrf",
        ...     top_k=20
        ... )
        >>> print(len(result.metadata["results"]))  # 20
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    async def execute(
        self,
        round_id: str,
        strategy: str = "rrf",
        top_k: int = 20,
    ) -> MCPToolResponse:
        """Fuse and return full results from a round.

        Args:
            round_id: Round ID to fuse
            strategy: Fusion strategy ('rrf', 'weighted', 'interleave')
            top_k: Maximum results to return

        Returns:
            MCPToolResponse with complete results
        """
        logger.info(f"fuse_and_fetch_round: round_id={round_id}, strategy={strategy}")

        try:
            manager = get_round_manager()

            # Check if round exists
            round_state = manager.get_round(round_id)
            if round_state is None:
                return MCPToolResponse(
                    content=f"Round not found: {round_id}",
                    is_empty=True,
                )

            # Fuse results
            fused_results = manager.fuse(
                round_id=round_id,
                strategy=strategy,
                top_k=top_k,
            )

            if not fused_results:
                return MCPToolResponse(
                    content=f"No results to fuse in round {round_id}",
                    is_empty=True,
                )

            # Build response with FULL text
            response_data = {
                "method": "fuse_and_fetch_round",
                "round_id": round_id,
                "fusion_strategy": strategy,
                "total_results": len(fused_results),
                "top_k_requested": top_k,
                "results": fused_results,
            }

            # Build markdown with full text
            markdown = f"## Fused Results\n\n"
            markdown += f"**Round ID:** `{round_id}`\n"
            markdown += f"**Strategy:** {strategy}\n"
            markdown += f"**Total Results:** {len(fused_results)}\n\n"

            for i, result in enumerate(fused_results[:10], 1):
                score = result.get("score", 0.0)
                fusion_score = result.get("fusion_score", score)
                chunk_id = result.get("chunk_id", "unknown")
                text = result.get("text", "")
                source = result.get("metadata", {}).get("source_path", "unknown")

                markdown += f"### {i}. Score: {fusion_score:.4f}\n"
                markdown += f"**Chunk:** `{chunk_id}`\n"
                markdown += f"**Source:** `{source}`\n"
                markdown += f"\n> {text}\n\n"

            if len(fused_results) > 10:
                markdown += f"\n*...还有 {len(fused_results) - 10} 条结果*\n"

            return MCPToolResponse(
                content=json.dumps(response_data, ensure_ascii=False, indent=2)
                + "\n\n"
                + markdown,
                metadata={
                    "round_id": round_id,
                    "fusion_strategy": strategy,
                    "result_count": len(fused_results),
                    "results": fused_results,
                },
                is_empty=False,
            )

        except Exception as e:
            logger.exception(f"fuse_and_fetch_round error: {e}")
            return MCPToolResponse(
                content=f"Error fusing round: {e}",
                is_empty=False,
            )


class GetRoundStatusTool:
    """Tool for querying round status.

    Returns a summary of the round including:
    - Number of tasks and chunks
    - Task details (query, strategy, chunk count)
    - Whether round has been fused
    - Age in minutes

    Example:
        >>> tool = GetRoundStatusTool()
        >>> result = await tool.execute(round_id="abc123")
        >>> print(result.metadata["total_chunks"])  # 20
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    async def execute(
        self,
        round_id: str,
    ) -> MCPToolResponse:
        """Get status of a round.

        Args:
            round_id: Round ID to query

        Returns:
            MCPToolResponse with round status
        """
        logger.info(f"get_round_status: round_id={round_id}")

        try:
            manager = get_round_manager()
            status = manager.get_status(round_id)

            if status is None:
                return MCPToolResponse(
                    content=f"Round not found or expired: {round_id}",
                    is_empty=True,
                )

            # Build response
            response_data = status.to_dict()
            response_data["method"] = "get_round_status"

            # Build markdown
            markdown = f"## Round Status\n\n"
            markdown += f"**Round ID:** `{status.round_id}`\n"
            markdown += f"**Total Chunks:** {status.total_chunks}\n"
            markdown += f"**Fused:** {'Yes' if status.fused else 'No'}\n"
            markdown += f"**Age:** {status.age_minutes:.1f} minutes\n\n"

            markdown += f"### Tasks ({len(status.tasks)})\n"
            for task in status.tasks:
                markdown += f"- **[{task['strategy']}]** {task['query'][:60]}...\n"
                markdown += f"  - Task ID: `{task['task_id']}`\n"
                markdown += f"  - Chunks: {task['chunks']}\n"

            return MCPToolResponse(
                content=json.dumps(response_data, ensure_ascii=False, indent=2)
                + "\n\n"
                + markdown,
                metadata=response_data,
                is_empty=False,
            )

        except Exception as e:
            logger.exception(f"get_round_status error: {e}")
            return MCPToolResponse(
                content=f"Error getting round status: {e}",
                is_empty=False,
            )


# =============================================================================
# Module-level Tool Instances
# =============================================================================

_batch_tool: Optional[ExecuteRetrievalBatchTool] = None
_fuse_tool: Optional[FuseAndFetchRoundTool] = None
_status_tool: Optional[GetRoundStatusTool] = None


def get_batch_tool(settings: Optional[Settings] = None) -> ExecuteRetrievalBatchTool:
    """Get or create the batch tool instance."""
    global _batch_tool
    if _batch_tool is None:
        _batch_tool = ExecuteRetrievalBatchTool(settings=settings)
    return _batch_tool


def get_fuse_tool(settings: Optional[Settings] = None) -> FuseAndFetchRoundTool:
    """Get or create the fuse tool instance."""
    global _fuse_tool
    if _fuse_tool is None:
        _fuse_tool = FuseAndFetchRoundTool(settings=settings)
    return _fuse_tool


def get_status_tool(settings: Optional[Settings] = None) -> GetRoundStatusTool:
    """Get or create the status tool instance."""
    global _status_tool
    if _status_tool is None:
        _status_tool = GetRoundStatusTool(settings=settings)
    return _status_tool


# =============================================================================
# MCP Handlers
# =============================================================================

async def execute_retrieval_batch_handler(
    tasks: List[Dict[str, Any]],
    top_k: int = 10,
    collection: str = "default",
    round_id: Optional[str] = None,
) -> types.CallToolResult:
    """MCP handler for execute_retrieval_batch."""
    tool = get_batch_tool()
    try:
        result = await tool.execute(
            round_id=round_id,
            tasks=tasks,
            top_k=top_k,
            collection=collection,
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=result.is_empty,
        )
    except Exception as e:
        logger.exception(f"execute_retrieval_batch handler error: {e}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


async def fuse_and_fetch_round_handler(
    round_id: str,
    strategy: str = "rrf",
    top_k: int = 20,
) -> types.CallToolResult:
    """MCP handler for fuse_and_fetch_round."""
    tool = get_fuse_tool()
    try:
        result = await tool.execute(
            round_id=round_id,
            strategy=strategy,
            top_k=top_k,
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=result.is_empty,
        )
    except Exception as e:
        logger.exception(f"fuse_and_fetch_round handler error: {e}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


async def get_round_status_handler(
    round_id: str,
) -> types.CallToolResult:
    """MCP handler for get_round_status."""
    tool = get_status_tool()
    try:
        result = await tool.execute(round_id=round_id)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=result.is_empty,
        )
    except Exception as e:
        logger.exception(f"get_round_status handler error: {e}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(protocol_handler) -> None:
    """Register all batch retrieval tools with the protocol handler."""
    protocol_handler.register_tool(
        name="execute_retrieval_batch",
        description="Execute multiple retrieval tasks CONCURRENTLY in a single call.\n\n"
                   "This is the PRIMARY tool for complex queries that need multiple searches.\n\n"
                   "Key features:\n"
                   "- Auto-creates a round if not provided\n"
                   "- Runs all tasks in parallel (fast!)\n"
                   "- Returns SUMMARY only (not full text - protects context)\n"
                   "- Use get_round_status to check progress\n"
                   "- Use fuse_and_fetch_round to get FULL results\n\n"
                   "When to use:\n"
                   "- Complex queries needing multiple search strategies\n"
                   "- Comparing results across different approaches\n"
                   "- When you need results from multiple collections\n\n"
                   "Workflow:\n"
                   "1. execute_retrieval_batch → get round_id\n"
                   "2. get_round_status(round_id) → check chunks\n"
                   "3. fuse_and_fetch_round(round_id) → get full results",
        input_schema=EXECUTE_BATCH_SCHEMA,
        handler=execute_retrieval_batch_handler,
    )
    logger.info("Registered MCP tool: execute_retrieval_batch")

    protocol_handler.register_tool(
        name="fuse_and_fetch_round",
        description="Fuse all results from a round and return FULL text.\n\n"
                   "This tool:\n"
                   "- Retrieves all chunks stored in the round\n"
                   "- Applies fusion strategy (RRF, weighted, or interleave)\n"
                   "- Returns COMPLETE text for each result\n"
                   "- Marks round as fused (prevents duplicate fetches)\n\n"
                   "When to use:\n"
                   "- After execute_retrieval_batch\n"
                   "- When ready to use results for answer generation\n"
                   "- This is your final step in the retrieval phase",
        input_schema=FUSE_ROUND_SCHEMA,
        handler=fuse_and_fetch_round_handler,
    )
    logger.info("Registered MCP tool: fuse_and_fetch_round")

    protocol_handler.register_tool(
        name="get_round_status",
        description="Check the status of a retrieval round.\n\n"
                   "Returns:\n"
                   "- Number of tasks executed\n"
                   "- Total chunks retrieved\n"
                   "- Per-task details (query, strategy, chunk count)\n"
                   "- Whether round has been fused\n"
                   "- Age of the round (for TTL awareness)\n\n"
                   "When to use:\n"
                   "- After execute_retrieval_batch\n"
                   "- Before fuse_and_fetch_round\n"
                   "- To decide if more retrieval is needed",
        input_schema=GET_STATUS_SCHEMA,
        handler=get_round_status_handler,
    )
    logger.info("Registered MCP tool: get_round_status")
