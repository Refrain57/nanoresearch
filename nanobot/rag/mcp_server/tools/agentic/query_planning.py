"""Query planning tools for Agentic RAG.

Allows Agent to plan and process queries before retrieval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nanobot.rag.core.types_agentic import QueryPlan
from nanobot.rag.mcp_server.tools.agentic.shared import (
    build_json_response,
    safe_json_loads,
)

logger = logging.getLogger(__name__)

# Rewrite prompt for resolving pronouns and references
REWRITE_PROMPT = """对话历史：
{history}

当前问题：{query}

将当前问题改写为独立完整的检索查询，解析所有指代和省略。

注意：如果最近的对话已经切换了话题，优先使用最近的上下文解析指代词，
忽略更早的话题锚点。
如果问题已经完整清晰，原样返回。

只输出改写后的查询，不要其他内容。"""


class PlanQueryTool:
    """MCP tool for planning query strategy.

    Allows Agent to:
    - Analyze query complexity
    - Get decomposition suggestions
    - Understand optimal retrieval strategy
    """

    # Planning prompt template
    PLANNING_PROMPT = """你是一个RAG系统查询分析专家。请分析用户查询并建议检索策略。

## 用户查询
{query}

## 分析要求
1. 判断查询的复杂度（简单/中等/复杂）
2. 如果是复杂查询，分解为多个子查询
3. 建议最优的检索策略

## 可用检索策略
- dense: 纯向量检索，适合语义理解类查询
- sparse: 纯BM25检索，适合精确关键词匹配
- hybrid: 混合检索（推荐），结合语义和关键词

## 输出格式 (JSON)
请输出一个JSON对象，格式如下：
```json
{{
  "complexity": "simple/medium/complex",
  "suggested_strategy": "dense/sparse/hybrid",
  "reason": "策略选择原因",
  "decomposition": {{
    "sub_queries": ["子查询1", "子查询2"],
    "search_order": ["建议的搜索顺序"]
  }},
  "keywords": ["关键术语1", "关键术语2"],
  "filters": {{
    "可能的过滤器": "值"
  }}
}}
```

请只输出JSON，不要有其他内容。"""

    def __init__(self):
        self._llm_client = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize LLM client if not already done."""
        if self._initialized:
            return

        try:
            from nanobot.rag.core.settings import get_settings

            settings = get_settings()
            if hasattr(settings, "llm") and settings.llm:
                self._init_llm_client(settings.llm.model_dump())
            elif hasattr(settings, "embedding") and settings.embedding:
                self._init_llm_client(settings.embedding.model_dump())

            self._initialized = True
            logger.info("LLM client initialized for query planning")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            self._initialized = True

    def _init_llm_client(self, config: Dict[str, Any]) -> None:
        """Initialize LLM client from config."""
        provider = config.get("provider", "dashscope")

        if provider == "dashscope" or provider == "aliyun":
            import dashscope
            from dashscope import Generation

            api_key = config.get("api_key") or config.get("dashscope_api_key")
            if api_key:
                dashscope.api_key = api_key
                self._llm_client = Generation
                self._llm_model = config.get("model", "qwen-turbo")

        elif provider == "openai":
            import openai

            api_key = config.get("api_key")
            base_url = config.get("base_url")
            if api_key:
                self._llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)
                self._llm_model = config.get("model", "gpt-3.5-turbo")

    @property
    def name(self) -> str:
        return "plan_query"

    @property
    def description(self) -> str:
        return """Analyze query complexity and decide retrieval strategy.

When to use:
- BEFORE first retrieval for complex queries (multi-part questions, comparisons)
- When query contains multiple aspects or terms like "vs", "compare", "and"
- When simple retrieval didn't work well

When NOT needed:
- Simple factual lookups (just call retrieve_hybrid directly)

Why:
- Complex queries often need multiple searches (sub-queries)
- Suggested strategy (dense/sparse/hybrid) helps optimize retrieval
- Saves time by planning before executing

Args:
    query: The user query to analyze

Returns:
    JSON with:
    - complexity: simple/medium/complex
    - suggested_strategy: dense/sparse/hybrid
    - suggested_queries: List of sub-queries to search (if complex)
    - keywords: Important terms extracted
    - reason: Why this strategy was chosen"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user query to analyze",
                },
                "session_key": {
                    "type": "string",
                    "description": "Main agent session key (channel:chat_id) for multi-turn context",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, session_key: str = None) -> "MCPToolResponse":
        """Execute the query planning tool.

        Args:
            query: The user query
            session_key: Optional session key for multi-turn context

        Returns:
            MCPToolResponse with query plan
        """
        import asyncio

        if not query:
            return build_json_response(
                {"error": "Empty query"},
                is_empty=True,
            )

        # 1. Get conversation history for query rewriting
        history = self._get_conversation_history(session_key) if session_key else []

        # 2. Rewrite query to resolve pronouns
        rewritten_query = await self._rewrite_query(query, history)

        # Use rewritten query for subsequent analysis
        analysis_query = rewritten_query

        # Initialize LLM
        await asyncio.to_thread(self._ensure_initialized)

        if not self._llm_client:
            # Fallback: simple heuristic planning
            plan = self._heuristic_planning(analysis_query)
            plan["original_query"] = query
            plan["rewritten_query"] = rewritten_query
            plan["history_used"] = len(history) > 0
            return build_json_response(plan)

        try:
            # Build prompt with rewritten query
            prompt = self.PLANNING_PROMPT.format(query=analysis_query)

            # Call LLM
            llm_response = await asyncio.to_thread(
                self._call_llm,
                prompt,
            )

            # Parse response
            plan = self._parse_llm_response(llm_response, analysis_query)

            # Add rewrite metadata to result
            plan["original_query"] = query
            plan["rewritten_query"] = rewritten_query
            plan["history_used"] = len(history) > 0

            return build_json_response(plan)

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            plan = self._heuristic_planning(analysis_query)
            plan["original_query"] = query
            plan["rewritten_query"] = rewritten_query
            plan["history_used"] = len(history) > 0
            return build_json_response(plan)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for planning."""
        if hasattr(self._llm_client, "call"):
            response = self._llm_client.call(
                model=self._llm_model,
                prompt=prompt,
            )
            return response.output.text if hasattr(response, "output") else str(response)

        elif hasattr(self._llm_client, "chat"):
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content

        else:
            raise ValueError("Unknown LLM client type")

    def _get_conversation_history(self, session_key: str) -> List[str]:
        """Get conversation history from main Agent session.

        Strategy:
        1. Always include initial_query (first user message) as topic anchor
        2. Get recent 5 rounds
        3. Deduplicate if initial_query is already in recent 5

        User message content field can be:
        1. String: use directly
        2. List: extract {"type": "text", "text": "..."} blocks
        """
        logger.info(f"_get_conversation_history called: session_key='{session_key}'")
        try:
            from nanobot.session.manager import SessionManager
            from nanobot.config.paths import get_workspace

            manager = SessionManager(get_workspace())
            session = manager.get_or_create(session_key)

            # Get history messages
            history = session.get_history()

            # Extract user message text content
            user_messages = []
            for msg in history:
                if msg.get("role") != "user":
                    continue

                content = msg.get("content")
                if isinstance(content, str):
                    user_messages.append(content)
                elif isinstance(content, list):
                    # Multimodal list: extract text blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    if text_parts:
                        user_messages.append(" ".join(text_parts))

            if not user_messages:
                return []

            # Get initial_query (first user message)
            initial_query = user_messages[0]

            # Get recent 5 rounds
            recent = user_messages[-5:]

            # Combine: initial_query + recent, deduplicate
            if initial_query in recent:
                # initial_query already in window, return as is
                return recent
            else:
                # initial_query outside window, prepend to beginning
                return [initial_query] + recent

        except Exception as e:
            logger.warning(f"Failed to get conversation history: {e}")
            return []

    async def _rewrite_query(self, query: str, history: List[str]) -> str:
        """Rewrite query to resolve pronouns using conversation history."""
        if not history:
            return query

        import asyncio

        # Format history
        history_text = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(history)
        )

        prompt = REWRITE_PROMPT.format(
            history=history_text,
            query=query
        )

        # Call LLM asynchronously
        try:
            # Ensure LLM is initialized
            await asyncio.to_thread(self._ensure_initialized)

            if not self._llm_client:
                return query

            response = await asyncio.to_thread(self._call_llm, prompt)

            # Parse returned text
            rewritten = response.strip()

            # Fallback to original query if empty or only punctuation
            if not rewritten or rewritten in [".", "。"]:
                return query

            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query

    def _parse_llm_response(self, response: str, query: str) -> Dict[str, Any]:
        """Parse LLM response into query plan."""
        import re

        # Extract JSON
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        try:
            data = json.loads(json_str)
            return {
                "original_query": query,
                "complexity": data.get("complexity", "simple"),
                "suggested_strategy": data.get("suggested_strategy", "hybrid"),
                "suggested_queries": data.get("decomposition", {}).get("sub_queries", [query]),
                "reason": data.get("reason", ""),
                "keywords": data.get("keywords", []),
                "filters": data.get("filters", {}),
                "decomposition": data.get("decomposition", {}),
            }
        except json.JSONDecodeError:
            return self._heuristic_planning(query)

    def _heuristic_planning(self, query: str) -> Dict[str, Any]:
        """Simple heuristic planning when LLM is unavailable."""
        # Check query length and structure
        words = query.split()
        has_questions = any(w in query for w in ["什么", "如何", "怎么", "为什么", "是否", "what", "how", "why"])

        complexity = "simple"
        if len(words) > 10 or "?" in query or has_questions:
            complexity = "medium"
        if "和" in query or "与" in query or "以及" in query or " vs " in query.lower():
            complexity = "complex"

        strategy = "hybrid"
        if len(words) < 5 and not has_questions:
            strategy = "dense"  # Short queries benefit from semantic search

        # Structure-aware planning
        structure_hints = self._extract_structure_hints(query)
        retrieval_steps = self._plan_retrieval_steps(query, complexity, structure_hints)

        return {
            "original_query": query,
            "complexity": complexity,
            "suggested_strategy": strategy,
            "suggested_queries": [query],
            "reason": f"Heuristic: {complexity} query, recommending {strategy}",
            "keywords": words[:5],
            "filters": structure_hints.get("filters", {}),
            "decomposition": {"sub_queries": [], "search_order": [query]},
            "structure_hints": structure_hints,
            "retrieval_steps": retrieval_steps,
            "stop_conditions": {
                "max_hops": 5,
                "overlap_threshold": 0.8,
                "confidence_threshold": 0.9,
            },
        }

    def _extract_structure_hints(self, query: str) -> Dict[str, Any]:
        """Extract structure-aware hints from query.

        Analyzes query to determine optimal retrieval strategy based on
        document structure metadata (section_level, content_type, etc.).

        Args:
            query: User query string.

        Returns:
            Dict with structure hints including filters and preferences.
        """
        query_lower = query.lower()
        hints = {
            "filters": {},
            "preferred_section_levels": [],
            "preferred_content_types": [],
            "query_intent": "search",
        }

        # Detect code-related queries
        code_keywords = ["代码", "code", "函数", "function", "实现", "implement",
                         "示例", "example", "api", "语法", "syntax"]
        if any(kw in query_lower for kw in code_keywords):
            hints["preferred_content_types"].append("code")
            hints["query_intent"] = "code_lookup"

        # Detect overview/summary queries
        overview_keywords = ["概述", "overview", "简介", "introduction", "总结",
                            "summary", "什么是", "what is", "介绍", "概览"]
        if any(kw in query_lower for kw in overview_keywords):
            hints["preferred_section_levels"] = [1, 2]  # Prefer high-level sections
            hints["query_intent"] = "overview"

        # Detect comparison queries
        comparison_keywords = ["比较", "compare", "对比", "vs", "versus", "区别", "difference"]
        if any(kw in query_lower for kw in comparison_keywords):
            hints["query_intent"] = "comparison"
            hints["preferred_section_levels"] = [2]  # Comparison sections often at level 2

        # Detect detail/deep-dive queries
        detail_keywords = ["详细", "detail", "具体", "specific", "深入", "deep"]
        if any(kw in query_lower for kw in detail_keywords):
            hints["preferred_section_levels"] = [3, 4, 5]  # Prefer detail sections
            hints["query_intent"] = "detail"

        return hints

    def _plan_retrieval_steps(
        self,
        query: str,
        complexity: str,
        structure_hints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Plan retrieval steps based on query and structure hints.

        Args:
            query: User query string.
            complexity: Query complexity (simple/medium/complex).
            structure_hints: Extracted structure hints.

        Returns:
            List of retrieval step dicts.
        """
        steps = []
        intent = structure_hints.get("query_intent", "search")

        # Step 1: Initial retrieval
        initial_step = {
            "step": 1,
            "action": "retrieve_hybrid",
            "query": query,
            "reason": "Initial hybrid search",
        }

        # Add structure-based filters if applicable
        if structure_hints.get("preferred_section_levels"):
            initial_step["filters"] = {
                "section_level": {"$in": structure_hints["preferred_section_levels"]}
            }
            initial_step["reason"] = f"Filtered to section levels {structure_hints['preferred_section_levels']}"

        if structure_hints.get("preferred_content_types"):
            # Note: content_type filter may be applied post-retrieval
            initial_step["preferred_content_types"] = structure_hints["preferred_content_types"]

        steps.append(initial_step)

        # Step 2: Structure expansion (for complex queries)
        if complexity == "complex":
            steps.append({
                "step": 2,
                "action": "expand_structure",
                "reason": "Complex query - fetch neighbors and related sections",
                "max_neighbors": 2,
            })

        # Step 3: Comparison handling (for comparison queries)
        if intent == "comparison":
            steps.append({
                "step": len(steps) + 1,
                "action": "fetch_section",
                "section_pattern": "对比|比较|vs|comparison",
                "reason": "Comparison query - fetch comparison sections",
            })

        return steps


class ProcessQueryTool:
    """MCP tool for processing queries.

    Exposes QueryProcessor functionality for keyword extraction.
    """

    def __init__(self):
        self._query_processor = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize QueryProcessor if not already done."""
        if self._initialized:
            return

        try:
            from nanobot.rag.core.query_engine.query_processor import QueryProcessor
            from nanobot.rag.core.settings import get_settings

            settings = get_settings()
            self._query_processor = QueryProcessor()
            self._initialized = True
            logger.info("QueryProcessor initialized for agentic tool")
        except Exception as e:
            logger.warning(f"Failed to initialize QueryProcessor: {e}")
            self._initialized = True

    @property
    def name(self) -> str:
        return "process_query"

    @property
    def description(self) -> str:
        return """Process a query to extract keywords, filters, and normalized form.

Args:
    query: The user query to process

Returns:
    JSON with processed query data including keywords and filters."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user query to process",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str) -> "MCPToolResponse":
        """Execute the query processing tool.

        Args:
            query: The user query

        Returns:
            MCPToolResponse with processed query data
        """
        import asyncio

        if not query:
            return build_json_response(
                {"error": "Empty query"},
                is_empty=True,
            )

        # Initialize processor
        await asyncio.to_thread(self._ensure_initialized)

        if not self._query_processor:
            return build_json_response(
                self._basic_processing(query)
            )

        try:
            # Process query
            processed = await asyncio.to_thread(
                self._query_processor.process,
                query,
            )

            # Build response
            result = {
                "original_query": query,
                "processed_query": processed.normalized_query if hasattr(processed, "normalized_query") else query,
                "keywords": processed.keywords if hasattr(processed, "keywords") else [],
                "filters": processed.filters if hasattr(processed, "filters") else {},
                "expanded_terms": processed.expanded_terms if hasattr(processed, "expanded_terms") else [],
                "intent": processed.intent if hasattr(processed, "intent") else "search",
            }

            return build_json_response(result)

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return build_json_response(
                self._basic_processing(query)
            )

    def _basic_processing(self, query: str) -> Dict[str, Any]:
        """Basic query processing without QueryProcessor."""
        # Simple keyword extraction
        import re

        # Remove punctuation
        clean_query = re.sub(r"[^\w\s]", " ", query)

        # Split into words
        words = clean_query.split()

        # Filter out short words
        keywords = [w for w in words if len(w) > 1]

        return {
            "original_query": query,
            "processed_query": query.strip(),
            "keywords": keywords[:10],
            "filters": {},
            "expanded_terms": [],
            "intent": "search",
        }


# =============================================================================
# Retrieval Controller - Multi-hop Termination Conditions
# =============================================================================

@dataclass
class RetrievalContext:
    """Context for multi-hop retrieval.

    Tracks the state across multiple retrieval iterations.
    """
    original_query: str
    hops: int = 0
    max_hops: int = 5
    collected_chunk_ids: set = field(default_factory=set)
    collected_results: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    overlap_threshold: float = 0.8
    confidence_threshold: float = 0.9


@dataclass
class StopReason:
    """Reason for stopping multi-hop retrieval."""
    reason: str
    should_stop: bool
    details: Dict[str, Any] = field(default_factory=dict)


class RetrievalController:
    """Controller for multi-hop retrieval with termination conditions.

    This class implements the stopping logic for iterative retrieval:
    1. Max hops limit
    2. Overlap threshold (new results overlap with existing)
    3. Confidence threshold (results are good enough)

    Example:
        >>> controller = RetrievalController(max_hops=5, overlap_threshold=0.8)
        >>> context = RetrievalContext(original_query="RAG architecture")
        >>> while True:
        ...     results = retrieve(query)
        ...     context = controller.update_context(context, results)
        ...     stop_reason = controller.should_stop(context)
        ...     if stop_reason.should_stop:
        ...         break
    """

    def __init__(
        self,
        max_hops: int = 5,
        overlap_threshold: float = 0.8,
        confidence_threshold: float = 0.9,
    ):
        """Initialize RetrievalController.

        Args:
            max_hops: Maximum number of retrieval iterations.
            overlap_threshold: Stop if new results overlap exceeds this.
            confidence_threshold: Stop if confidence exceeds this.
        """
        self.max_hops = max_hops
        self.overlap_threshold = overlap_threshold
        self.confidence_threshold = confidence_threshold

    def should_stop(self, context: RetrievalContext) -> StopReason:
        """Check if multi-hop retrieval should stop.

        Args:
            context: Current retrieval context.

        Returns:
            StopReason with should_stop flag and details.
        """
        # Check max hops
        if context.hops >= self.max_hops:
            return StopReason(
                reason="max_hops_reached",
                should_stop=True,
                details={"hops": context.hops, "max_hops": self.max_hops},
            )

        # Check confidence threshold
        if context.confidence >= self.confidence_threshold:
            return StopReason(
                reason="confidence_threshold_reached",
                should_stop=True,
                details={
                    "confidence": context.confidence,
                    "threshold": self.confidence_threshold,
                },
            )

        # Check overlap (handled in update_context)
        # This is checked when we add new results

        return StopReason(
            reason="continue",
            should_stop=False,
            details={"hops": context.hops},
        )

    def update_context(
        self,
        context: RetrievalContext,
        new_results: List[Dict[str, Any]],
        confidence: Optional[float] = None,
    ) -> Tuple[RetrievalContext, StopReason]:
        """Update context with new retrieval results.

        Args:
            context: Current retrieval context.
            new_results: New retrieval results.
            confidence: Optional confidence score for results.

        Returns:
            Tuple of (updated_context, stop_reason).
        """
        # Increment hop count
        context.hops += 1

        # Calculate overlap
        new_chunk_ids = {r.get("chunk_id") for r in new_results if r.get("chunk_id")}
        overlap = len(new_chunk_ids & context.collected_chunk_ids)
        overlap_ratio = overlap / len(new_chunk_ids) if new_chunk_ids else 0.0

        # Check high overlap
        if overlap_ratio >= self.overlap_threshold and context.hops > 1:
            return context, StopReason(
                reason="high_overlap",
                should_stop=True,
                details={
                    "overlap_ratio": overlap_ratio,
                    "threshold": self.overlap_threshold,
                    "new_chunks": len(new_chunk_ids),
                    "overlapping_chunks": overlap,
                },
            )

        # Update collected results
        context.collected_chunk_ids.update(new_chunk_ids)
        context.collected_results.extend(new_results)

        # Update confidence if provided
        if confidence is not None:
            context.confidence = max(context.confidence, confidence)

        # Check other stop conditions
        stop_reason = self.should_stop(context)

        return context, stop_reason

    def calculate_overlap(
        self,
        existing_ids: set,
        new_ids: set,
    ) -> float:
        """Calculate overlap ratio between existing and new results.

        Args:
            existing_ids: Set of existing chunk IDs.
            new_ids: Set of new chunk IDs.

        Returns:
            Overlap ratio (0.0 to 1.0).
        """
        if not new_ids:
            return 0.0
        overlap = len(existing_ids & new_ids)
        return overlap / len(new_ids)


# MCP Tool Handlers
async def plan_query_handler(query: str) -> "MCPToolResponse":
    """Handler for plan_query MCP tool."""
    tool = PlanQueryTool()
    return await tool.execute(query=query)


async def process_query_handler(query: str) -> "MCPToolResponse":
    """Handler for process_query MCP tool."""
    tool = ProcessQueryTool()
    return await tool.execute(query=query)


def register_tools(protocol_handler: Any) -> None:
    """Register query planning tools with the protocol handler.

    Args:
        protocol_handler: The MCP protocol handler instance
    """
    plan_tool = PlanQueryTool()
    protocol_handler.register_tool(
        name=plan_tool.name,
        description=plan_tool.description,
        input_schema=plan_tool.input_schema,
        handler=plan_query_handler,
    )
    logger.info(f"Registered agentic tool: {plan_tool.name}")

    process_tool = ProcessQueryTool()
    protocol_handler.register_tool(
        name=process_tool.name,
        description=process_tool.description,
        input_schema=process_tool.input_schema,
        handler=process_query_handler,
    )
    logger.info(f"Registered agentic tool: {process_tool.name}")