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

from nanobot.rag.mcp_server.tools.agentic.shared import build_json_response


# Import from submodule to avoid circular import
def _get_batch_tool():
    """Lazy import ExecuteRetrievalBatchTool to avoid circular dependency."""
    from nanobot.rag.mcp_server.tools.agentic.batch_retrieval import ExecuteRetrievalBatchTool
    return ExecuteRetrievalBatchTool


# Import from submodule to avoid circular import
def _get_rag_loop_components():
    """Lazy import to avoid circular dependency."""
    from nanobot.rag.internal_loop.runner import RAGLoopRunner, RAGLoopResult, classify_complexity, run_rag_loop
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
        return "rag_search"

    @property
    def description(self) -> str:
        return """统一的 RAG 检索入口，用于从知识库中检索相关信息。

## 何时使用
- 需要从文档库中查找信息时
- 需要进行多轮检索以获取完整信息时
- 需要对比、分析多个主题时

## 功能特点

1. **智能查询分类**
   - 简单查询：直接执行混合检索，快速返回
   - 复杂查询：启动内部循环，多轮检索验证

2. **自动验证**
   - 检索结果自动验证充分性
   - 结果不足时自动补充检索

3. **引用构建**
   - 自动构建结构化引用
   - 便于追溯信息来源

## 参数说明

- query: 用户查询（必填）
- collection: 检索集合名称（默认 "default"）
- context: 外部上下文，用于解析指代词（可选）
- max_iterations: 最大迭代次数（默认 5）

## 返回格式

返回 JSON 格式结果，包含：
- success: 是否成功
- chunks: 检索到的文档片段
- citations: 引用信息
- summary: 检索摘要
- iterations: 实际迭代次数

## 示例

```python
# 简单查询
rag_search(query="PGSR 的核心思想是什么？")

# 带上下文的查询
rag_search(
    query="它的性能如何？",
    context="用户之前在讨论 PGSR 方法"
)

# 复杂对比查询
rag_search(query="PGSR 和 SuGaR 的性能对比")
```"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户查询",
                },
                "collection": {
                    "type": "string",
                    "default": "default",
                    "description": "检索集合名称",
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
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        collection: str = "default",
        context: Optional[str] = None,
        max_iterations: int = 5,
    ) -> "MCPToolResponse":
        """Execute RAG search.

        Args:
            query: User query
            collection: Collection to search
            context: External context for query rewriting
            max_iterations: Maximum iterations for complex queries

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
                query, collection, context, max_iterations
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
            from nanobot.rag.mcp_server.tools.agentic.round_state import get_round_manager
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
            return build_json_response({
                "success": True,
                "chunks": fused_chunks,
                "citations": None,
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
    ) -> "MCPToolResponse":
        """Execute complex query: Internal loop with verification.

        Args:
            query: User query
            collection: Collection to search
            context: External context
            max_iterations: Maximum iterations

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
            )

            _log(f"Loop completed: success={result.success}, chunks={len(result.chunks)}")
            return build_json_response(result.to_dict())

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
) -> "MCPToolResponse":
    """Handler for rag_search MCP tool."""
    tool = RAGSearchTool()
    return await tool.execute(
        query=query,
        collection=collection,
        context=context,
        max_iterations=max_iterations,
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
    _log(f"Registered RAG tool: {tool.name}")
