"""Internal tools for RAG loop.

Wraps existing agentic tools for use in the internal loop.
Each tool returns structured results that can be used by the runner.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from nanobot.rag.internal_loop.state import RoundState, SessionState, SubQuery

# Lazy imports to avoid circular dependency
def _get_batch_tool():
    from nanobot.rag.mcp_server.tools.agentic.batch_retrieval import ExecuteRetrievalBatchTool
    return ExecuteRetrievalBatchTool

def _get_plan_tool():
    from nanobot.rag.mcp_server.tools.agentic.query_planning import PlanQueryTool
    return PlanQueryTool

def _get_verify_tool():
    from nanobot.rag.mcp_server.tools.agentic.verification import VerifyResultsTool
    return VerifyResultsTool

def _get_rerank_tool():
    from nanobot.rag.mcp_server.tools.agentic.reranking import RerankResultsTool
    return RerankResultsTool

def _get_citations_tool():
    from nanobot.rag.mcp_server.tools.agentic.citations import BuildCitationsTool
    return BuildCitationsTool

def _get_round_manager():
    from nanobot.rag.mcp_server.tools.agentic.round_state import RoundStateManager
    return RoundStateManager

def _get_fetch_section_tool():
    from nanobot.rag.mcp_server.tools.agentic.retrieval import get_fetch_section_tool
    return get_fetch_section_tool()

def _get_fetch_neighbors_tool():
    from nanobot.rag.mcp_server.tools.agentic.retrieval import get_fetch_neighbors_tool
    return get_fetch_neighbors_tool()


# Keywords that trigger fetch_section for comparison queries
_COMPARISON_SECTION_KEYWORDS = ["对比", "比较", "vs", "comparison", "区别", "差异"]


class InternalTools:
    """Wrapper for internal RAG loop tools.

    Provides a unified interface for:
    - plan_query: Query analysis and planning
    - execute_batch: Batch retrieval execution
    - expand_with_sections: Fetch comparison-relevant sections (before fuse)
    - fuse_results: Result fusion
    - verify_results: Verification
    - expand_with_neighbors: Context expansion on verify failure (after fuse)
    - rerank_results: Reranking
    - build_citations: Citation building
    """

    def __init__(self):
        self._plan_tool = None
        self._batch_tool = None
        self._verify_tool = None
        self._rerank_tool = None
        self._citations_tool = None
        self._round_manager = None

        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize tools lazily."""
        if self._initialized:
            return

        try:
            self._plan_tool = _get_plan_tool()()
            self._batch_tool = _get_batch_tool()()
            self._verify_tool = _get_verify_tool()()
            self._rerank_tool = _get_rerank_tool()()
            self._citations_tool = _get_citations_tool()()
            self._round_manager = _get_round_manager().get_instance()
            self._initialized = True
            logger.info("Internal tools initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise

    # =========================================================================
    # Tool Schemas (for LLM tool definitions)
    # =========================================================================

    def get_plan_tool_schema(self) -> Dict[str, Any]:
        """Get schema for plan_query tool."""
        return {
            "type": "function",
            "function": {
                "name": "plan_query",
                "description": "分析查询并分解为子查询，标注每个子查询的检索策略。必须在搜索开始前调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用户查询",
                        },
                        "context": {
                            "type": "string",
                            "description": "外部上下文（可选）",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def get_execute_batch_tool_schema(self) -> Dict[str, Any]:
        """Get schema for execute_batch tool."""
        return {
            "type": "function",
            "function": {
                "name": "execute_batch",
                "description": "并发执行多个检索任务。每个任务需要指定 query 和 strategy。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "strategy": {"type": "string", "enum": ["dense", "sparse", "hybrid"]},
                                    "top_k": {"type": "integer", "default": 5},
                                },
                                "required": ["query", "strategy"],
                            },
                            "description": "检索任务列表",
                        },
                        "round_id": {
                            "type": "string",
                            "description": "当前 round 的 ID",
                        },
                        "collection": {
                            "type": "string",
                            "default": "default",
                            "description": "检索集合",
                        },
                    },
                    "required": ["tasks", "round_id"],
                },
            },
        }

    def get_rerank_tool_schema(self) -> Dict[str, Any]:
        """Get schema for rerank_results tool."""
        return {
            "type": "function",
            "function": {
                "name": "rerank_results",
                "description": "对检索结果进行重排序以提高相关性。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "string",
                            "description": "JSON 字符串形式的检索结果",
                        },
                        "query": {
                            "type": "string",
                            "description": "原始查询",
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 10,
                        },
                    },
                    "required": ["results", "query"],
                },
            },
        }

    def get_search_tools(self) -> List[Dict[str, Any]]:
        """Get schemas for Phase 2 search tools.

        Returns:
            List of tool schemas for search phase
        """
        return [
            self.get_execute_batch_tool_schema(),
            self.get_rerank_tool_schema(),
        ]

    def get_plan_tools(self) -> List[Dict[str, Any]]:
        """Get schemas for Phase 1 plan tools.

        Returns:
            List of tool schemas for plan phase
        """
        return [
            self.get_plan_tool_schema(),
        ]

    # =========================================================================
    # Tool Execution Methods
    # =========================================================================

    async def plan_query(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute plan_query tool.

        Args:
            query: The user query
            context: External context (optional)

        Returns:
            Plan result with sub_queries and strategy annotations
        """
        self._ensure_initialized()

        try:
            result = await self._plan_tool.execute(
                query=query,
                context=context,
            )

            # Parse result
            if hasattr(result, "content"):
                content = result.content
                if isinstance(content, list):
                    # Extract text from content blocks
                    text = ""
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            text += block["text"]
                    return json.loads(text) if text else {}
                elif isinstance(content, str):
                    return json.loads(content)
            return {}

        except Exception as e:
            logger.error(f"plan_query failed: {e}")
            # Return fallback plan
            return {
                "complexity": "complex",
                "sub_queries": [
                    {"query": query, "strategy": "hybrid", "reason": "fallback"}
                ],
            }

    async def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        round_id: str,
        collection: str = "default",
        session_state: Optional[SessionState] = None,
    ) -> Dict[str, Any]:
        """Execute batch retrieval.

        Args:
            tasks: List of retrieval tasks
            round_id: Current round ID
            collection: Collection name
            session_state: Session state for tracking

        Returns:
            Execution results
        """
        self._ensure_initialized()

        try:
            # Ensure round exists in RoundStateManager
            self._round_manager.get_or_create_round(round_id)

            # Execute retrieval
            result = await self._batch_tool.execute(
                tasks=tasks,
                round_id=round_id,
                collection=collection,
            )

            # Extract total_chunks from metadata if available
            total_chunks = 0
            if hasattr(result, "metadata"):
                total_chunks = result.metadata.get("total_chunks", 0)

            # Try to parse content as JSON for results
            parsed = self._parse_tool_result(result)

            # If content is not valid JSON, construct result from metadata
            if "total_chunks" not in parsed:
                parsed["total_chunks"] = total_chunks
            if "round_id" not in parsed:
                parsed["round_id"] = round_id

            return parsed

        except Exception as e:
            logger.error(f"execute_batch failed: {e}")
            return {"error": str(e), "results": {}, "total_chunks": 0}

    async def expand_with_sections(
        self,
        round_id: str,
        collection: str = "default",
    ) -> int:
        """Fetch comparison-relevant sections and add to round (call BEFORE fuse).

        Tries each comparison keyword as a section_path filter.
        New chunks are added to the round state so they participate in fuse.

        Returns:
            Number of new chunks added to the round.
        """
        self._ensure_initialized()

        fetch_tool = _get_fetch_section_tool()
        added = 0

        for keyword in _COMPARISON_SECTION_KEYWORDS:
            try:
                result = await fetch_tool.execute(
                    section_path=keyword,
                    collection=collection,
                    include_neighbors=False,
                    max_chunks=5,
                )
                if result.is_empty:
                    continue

                data = json.loads(result.content)
                chunks = data.get("chunks", [])
                if not chunks:
                    continue

                import uuid as _uuid
                task_id = f"section_{keyword}_{_uuid.uuid4().hex[:6]}"
                self._round_manager.add_results(round_id, task_id, chunks)
                added += len(chunks)
                logger.info(f"expand_with_sections: added {len(chunks)} chunks for keyword='{keyword}'")
            except Exception as e:
                logger.warning(f"expand_with_sections failed for keyword '{keyword}': {e}")

        return added

    async def expand_with_neighbors(
        self,
        fused_chunks: List[Dict[str, Any]],
        collection: str = "default",
        top_n: int = 5,
        window: int = 1,
    ) -> List[Dict[str, Any]]:
        """Expand context by fetching neighbors of top fused chunks (call AFTER fuse).

        Does NOT touch round state (round is already fused).
        Returns new chunks only (dedup against existing fused_chunks by chunk_id).

        Args:
            fused_chunks: Already-fused chunks from the round.
            collection: Collection name.
            top_n: How many top chunks to expand around.
            window: Neighbor window size.

        Returns:
            List of new chunks not already in fused_chunks.
        """
        self._ensure_initialized()

        fetch_tool = _get_fetch_neighbors_tool()
        existing_ids = {c.get("chunk_id") for c in fused_chunks}
        new_chunks: List[Dict[str, Any]] = []

        candidates = fused_chunks[:top_n]
        for chunk in candidates:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            try:
                result = await fetch_tool.execute(
                    chunk_id=chunk_id,
                    collection=collection,
                    window=window,
                )
                if result.is_empty:
                    continue

                data = json.loads(result.content)
                for c in data.get("chunks", []):
                    cid = c.get("chunk_id")
                    if cid and cid not in existing_ids:
                        existing_ids.add(cid)
                        new_chunks.append(c)
            except Exception as e:
                logger.warning(f"expand_with_neighbors failed for chunk '{chunk_id}': {e}")

        logger.info(f"expand_with_neighbors: added {len(new_chunks)} new neighbor chunks")
        return new_chunks

    async def fuse_results(
        self,
        round_id: str,
        strategy: str = "rrf",
        top_k: int = 20,
        session_state: Optional[SessionState] = None,
    ) -> List[Dict[str, Any]]:
        """Fuse results from a round.

        Args:
            round_id: Round ID to fuse
            strategy: Fusion strategy
            top_k: Maximum results
            session_state: Session state for tracking

        Returns:
            Fused results
        """
        self._ensure_initialized()

        try:
            fused = self._round_manager.fuse(
                round_id=round_id,
                strategy=strategy,
                top_k=top_k,
            )

            # Store in session state if provided
            if session_state and session_state.current_round:
                session_state.current_round.fused = True
                session_state.current_round.fused_chunks = fused
                session_state.fused_chunks = fused

            return fused

        except Exception as e:
            logger.error(f"fuse_results failed: {e}")
            return []

    async def verify_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        """Verify retrieval results.

        Args:
            results: Results to verify
            query: Original query

        Returns:
            Verification result with confidence and next_actions
        """
        self._ensure_initialized()

        try:
            # Convert results to JSON string
            results_json = json.dumps(results, ensure_ascii=False)

            result = await self._verify_tool.execute(
                results=results_json,
                query=query,
            )

            # Parse result
            parsed = self._parse_tool_result(result)

            # Ensure required fields exist
            if "confidence" not in parsed:
                parsed["confidence"] = 0.5
            if "answered" not in parsed:
                parsed["answered"] = parsed.get("confidence", 0.5) >= 0.7
            if "next_actions" not in parsed:
                parsed["next_actions"] = []
            if "missing_aspects" not in parsed:
                parsed["missing_aspects"] = []

            return parsed

        except Exception as e:
            logger.error(f"verify_results failed: {e}")
            return {
                "answered": False,
                "confidence": 0.3,
                "summary": f"Verification failed: {e}",
                "next_actions": [],
                "missing_aspects": [],
            }

    async def rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Rerank retrieval results.

        Args:
            results: Results to rerank
            query: Original query
            top_k: Maximum results to return

        Returns:
            Reranked results
        """
        self._ensure_initialized()

        try:
            results_json = json.dumps(results, ensure_ascii=False)

            result = await self._rerank_tool.execute(
                results=results_json,
                query=query,
                top_k=top_k,
            )

            parsed = self._parse_tool_result(result)
            return parsed.get("results", results[:top_k])

        except Exception as e:
            logger.error(f"rerank_results failed: {e}")
            return results[:top_k]

    async def build_citations(
        self,
        results: List[Dict[str, Any]],
        format: str = "structured",
        include_text: bool = False,
    ) -> Dict[str, Any]:
        """Build citations from results.

        Args:
            results: Retrieval results
            format: Citation format
            include_text: Whether to include text snippets

        Returns:
            Citations result
        """
        self._ensure_initialized()

        try:
            results_json = json.dumps(results, ensure_ascii=False)

            result = await self._citations_tool.execute(
                results=results_json,
                format=format,
                include_text=include_text,
            )

            return self._parse_tool_result(result)

        except Exception as e:
            logger.error(f"build_citations failed: {e}")
            return {"citations": [], "total": 0}

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_tool_result(self, result: Any) -> Dict[str, Any]:
        """Parse tool result into dictionary.

        Args:
            result: Tool result (MCPToolResponse)

        Returns:
            Parsed dictionary
        """
        if result is None:
            return {}

        # Handle MCPToolResponse
        if hasattr(result, "content"):
            content = result.content

            if isinstance(content, list):
                # Extract text from content blocks
                text = ""
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text += block["text"]

                if text:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"text": text}

            elif isinstance(content, str):
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"text": content}

            elif isinstance(content, dict):
                return content

        # Handle dict directly
        if isinstance(result, dict):
            return result

        return {}


# Singleton instance
_internal_tools: Optional[InternalTools] = None


def get_internal_tools() -> InternalTools:
    """Get the singleton InternalTools instance.

    Returns:
        InternalTools instance
    """
    global _internal_tools
    if _internal_tools is None:
        _internal_tools = InternalTools()
    return _internal_tools
