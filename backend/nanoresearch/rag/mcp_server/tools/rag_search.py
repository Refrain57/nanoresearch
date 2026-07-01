"""RAG Search Tool - Single entry point for RAG retrieval.

This tool provides a unified interface for RAG search:
- Simple queries: Direct hybrid retrieval
- Complex queries: Internal loop with verification

The tool handles:
- Query complexity classification
- Multi-round retrieval
- Result verification
- Citation building
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, Optional

from nanoresearch.rag.mcp_server.tools.agentic.shared import build_json_response, build_citations_from_chunks


# Import from submodule to avoid circular import
def _get_batch_tool():
    """Lazy import ExecuteRetrievalBatchTool to avoid circular dependency."""
    from nanoresearch.rag.mcp_server.tools.agentic.batch_retrieval import ExecuteRetrievalBatchTool
    return ExecuteRetrievalBatchTool


# Import from submodule to avoid circular import
def _get_rag_loop_components():
    """Lazy import to avoid circular dependency."""
    from nanoresearch.rag.internal_loop.runner import RAGLoopRunner, RAGLoopResult, classify_complexity, run_rag_loop
    return RAGLoopRunner, RAGLoopResult, classify_complexity, run_rag_loop


def _log(msg: str) -> None:
    """Log to stderr."""
    print(f"[RAG] {msg}", flush=True, file=sys.stderr)


class RAGSearchTool:
    """MCP tool for unified RAG search.

    This is the single entry point for external agents.
    It internally handles:
    1. Query complexity classification
    2. Simple queries: Direct retrieval
    3. Complex queries: Internal loop with verification

    Example:
        >>> tool = RAGSearchTool()
        >>> result = await tool.execute(
        ...     query="PGSR 和 SuGaR 的差异",
        ...     collection="papers",
        ... )
    """

    def __init__(self):
        self._batch_tool = None
        self._loop_runner = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize tools lazily."""
        if self._initialized:
            return

        _log("_ensure_initialized called")

        try:
            ExecuteRetrievalBatchTool = _get_batch_tool()
            _log("Got batch tool class")
            self._batch_tool = ExecuteRetrievalBatchTool()
            _log("Batch tool created")

            _, RAGLoopRunner, _, _ = _get_rag_loop_components()
            _log("Got RAGLoopRunner class")
            self._loop_runner = RAGLoopRunner()
            _log("Loop runner created")

            self._initialized = True
            _log("Initialization complete")
        except Exception as e:
            _log(f"Initialization failed: {e}")
            traceback.print_exc(file=sys.stderr)
            self._initialized = True  # Mark as attempted

    @property
    def name(self) -> str:
        return "kb_search"

    @property
    def description(self) -> str:
        return """查询用户上传的知识库文档和历史研究报告。

研究报告在生成后会自动入库，因此可以通过此工具搜索到之前的研究成果。
当用户问到可能和历史研究相关的问题时（如"之前研究过什么"、"上次关于 X 的结论"），优先使用此工具查询。

根据查询复杂度自动选择检索路径：简单查询直接检索，复杂查询启动多轮内部检索循环。

若需要自己控制每一跳检索（如多跳推理），请用 kb_retrieve。

参数：
- query: 查询内容（必填）
- kb_id: 知识库 ID（可选，有多个绑定知识库时指定；只有一个时自动使用）
- context: 外部上下文，用于解析"它""这个"等指代词（可选）
- max_iterations: 内部最大检索轮数（默认 5）"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户查询",
                },
                "context": {
                    "type": "string",
                    "description": "外部上下文，用于解析指代词（如'它'、'这个'）",
                },
                "max_iterations": {
                    "type": "integer",
                    "default": 5,
                    "description": "最大迭代次数（复杂查询时使用）",
                },
                "kb_id": {
                    "type": "string",
                    "description": "知识库 ID，不传时自动搜索默认知识库",
                },
                "session_key": {
                    "type": "string",
                    "description": "Main agent session key (channel:chat_id) for multi-turn context",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        collection: str = "default",
        context: Optional[str] = None,
        max_iterations: int = 5,
        session_key: Optional[str] = None,
    ) -> "MCPToolResponse":
        """Execute RAG search.

        Args:
            query: User query
            collection: Collection to search
            context: External context for query rewriting
            max_iterations: Maximum iterations for complex queries
            session_key: Outer agent session key (channel:chat_id), forwarded to
                internal loop / plan_query for history-based query rewrite

        Returns:
            MCPToolResponse with retrieval results
        """
        self._ensure_initialized()

        # Lazy import to avoid circular dependency
        _, _, classify_complexity, _ = _get_rag_loop_components()

        _log(f"Query: {query[:50]}... (context={context is not None})")

        # Classify query complexity
        complexity = classify_complexity(query, context)
        _log(f"Complexity: {complexity}")

        if complexity == "simple":
            # Fast path: Direct hybrid retrieval
            _log("Taking simple path (direct retrieval)")
            return await self._execute_simple(query, collection)
        else:
            # Complex path: Internal loop
            _log("Taking complex path (internal loop)")
            return await self._execute_complex(
                query, collection, context, max_iterations, session_key
            )

    async def _execute_simple(
        self,
        query: str,
        collection: str,
    ) -> "MCPToolResponse":
        """Execute simple query: Direct hybrid retrieval.

        Args:
            query: User query
            collection: Collection to search

        Returns:
            MCPToolResponse with results
        """
        _log(f"_execute_simple: query='{query}', collection='{collection}'")

        try:
            # Create a round for this simple query
            from nanoresearch.rag.mcp_server.tools.agentic.round_state import get_round_manager
            manager = get_round_manager()
            round_id = manager.create_round()

            # Use batch tool for single hybrid retrieval
            tasks = [{
                "query": query,
                "strategy": "hybrid",
                "top_k": 10,
            }]

            _log(f"Calling batch tool with {len(tasks)} tasks")
            result = await self._batch_tool.execute(
                tasks=tasks,
                round_id=round_id,
                collection=collection,
            )

            _log(f"Batch tool returned: {type(result)}")

            # Get fused results from the round
            fused_chunks = manager.fuse(round_id=round_id, strategy="rrf", top_k=10)
            _log(f"Got {len(fused_chunks)} fused chunks")

            # Auto-expand neighbors for high-confidence hits (score >= 0.85).
            # When a chunk scores this high it's a genuine hit; the data we need
            # (e.g. a table) is likely in the adjacent chunk.  This removes the
            # dependency on classify_complexity for structural coverage.
            high_score = [c for c in fused_chunks if c.get("score", 0) >= 0.85]
            if high_score:
                try:
                    from nanoresearch.rag.internal_loop.tools import InternalTools
                    neighbor_chunks = await InternalTools().expand_with_neighbors(
                        fused_chunks=high_score,
                        collection=collection,
                    )
                    if neighbor_chunks:
                        existing_ids = {c.get("chunk_id") for c in fused_chunks}
                        new_neighbors = [c for c in neighbor_chunks if c.get("chunk_id") not in existing_ids]
                        fused_chunks = fused_chunks + new_neighbors
                        _log(f"Auto-expanded {len(new_neighbors)} neighbor chunks (high-score={len(high_score)})")
                except Exception as e:
                    _log(f"Neighbor expansion skipped: {e}")

            return build_json_response({
                "success": True,
                "chunks": fused_chunks,
                "citations": build_citations_from_chunks(fused_chunks),
                "summary": f"Simple retrieval completed with {len(fused_chunks)} chunks",
                "iterations": 1,
            })

        except Exception as e:
            _log(f"Simple retrieval failed: {e}")
            traceback.print_exc(file=sys.stderr)
            return build_json_response({
                "success": False,
                "error": str(e),
                "chunks": [],
            })

    async def _execute_complex(
        self,
        query: str,
        collection: str,
        context: Optional[str],
        max_iterations: int,
        session_key: Optional[str] = None,
    ) -> "MCPToolResponse":
        """Execute complex query: Internal loop with verification.

        Args:
            query: User query
            collection: Collection to search
            context: External context
            max_iterations: Maximum iterations
            session_key: Outer agent session key for multi-turn context

        Returns:
            MCPToolResponse with results
        """
        try:
            # Lazy import to avoid circular dependency
            _, _, _, run_rag_loop = _get_rag_loop_components()

            _log("Running internal loop...")
            result = await run_rag_loop(
                query=query,
                context=context,
                collection=collection,
                max_iterations=max_iterations,
                session_key=session_key,
            )

            _log(f"Loop completed: success={result.success}, chunks={len(result.chunks)}")
            result_dict = result.to_dict()
            result_dict["citations"] = build_citations_from_chunks(result.chunks)
            return build_json_response(result_dict)

        except Exception as e:
            _log(f"Complex retrieval failed: {e}")
            traceback.print_exc(file=sys.stderr)
            return build_json_response({
                "success": False,
                "error": str(e),
                "chunks": [],
            })


# MCP Tool Handler
async def rag_search_handler(
    query: str,
    collection: str = "default",
    context: Optional[str] = None,
    max_iterations: int = 5,
    kb_id: Optional[str] = None,
    session_key: Optional[str] = None,
) -> "MCPToolResponse":
    """Handler for kb_search MCP tool.

    Note: 'collection' is injected by the main process (mcp.py) and
    should NOT be passed by the Agent. The Agent should use 'kb_id' instead.
    """
    tool = RAGSearchTool()
    return await tool.execute(
        query=query,
        collection=collection,
        context=context,
        max_iterations=max_iterations,
        session_key=session_key,
    )


def register_tools(protocol_handler: Any) -> None:
    """Register RAG search tool with the protocol handler.

    Args:
        protocol_handler: The MCP protocol handler instance
    """
    tool = RAGSearchTool()
    protocol_handler.register_tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        handler=rag_search_handler,
    )
    _log(f"Registered MCP tool: {tool.name}")
