"""RAG Internal Loop Runner.

Implements the state machine for internal RAG loop:
- Phase 1: Plan → Analyze query, create sub-queries
- Phase 2: Search → Execute retrieval (loop until done)
- Phase 3: Fuse + Verify → System-forced verification
- Phase 4: Finalize → Build citations and return
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


# 指代词列表：需要上下文才能理解的词
PRONOUN_KEYWORDS = ["它", "这个", "那个", "这些", "那些", "其", "上文", "上面", "之前"]
COMPARISON_KEYWORDS = ["对比", "比较", "差异", "vs", "区别"]


def classify_complexity(query: str, context: Optional[str]) -> str:
    """轻量规则判断查询复杂度（不用 LLM）。

    Args:
        query: 用户查询
        context: 外部上下文

    Returns:
        "simple" 或 "complex"
    """
    # 有指代词且无上下文 → 需要进入 loop
    if context is None and any(kw in query for kw in PRONOUN_KEYWORDS):
        return "complex"

    # 简单规则：短查询
    if len(query) < 20:
        return "simple"

    # 对比关键词
    if any(kw in query for kw in ["对比", "比较", "差异", "vs", "和", "以及"]):
        return "complex"

    # 多个子问题
    if query.count("？") > 1 or query.count("?") > 1:
        return "complex"

    # 包含"分别"、"各自"等
    if any(kw in query for kw in ["分别", "各自", "同时"]):
        return "complex"

    return "simple"


@dataclass
class RAGLoopResult:
    """Result from RAG loop execution.

    Attributes:
        success: Whether the loop completed successfully
        chunks: Retrieved and fused chunks
        citations: Built citations
        verification: Final verification result
        iterations: Number of iterations used
        summary: Summary of the retrieval process
    """
    success: bool = False
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    citations: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    iterations: int = 0
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "chunks": self.chunks,
            "citations": self.citations,
            "verification": self.verification,
            "iterations": self.iterations,
            "summary": self.summary,
            "error": self.error,
        }


class RAGLoopRunner:
    """Runner for internal RAG loop.

    Implements the state machine:
    ```
    Phase 1: Plan → plan_query
    Phase 2: Search → execute_batch (loop)
    Phase 3: Fuse + Verify → (system forced)
    Phase 4: Finalize → build_citations
    ```

    Example:
        >>> runner = RAGLoopRunner(llm_provider)
        >>> result = await runner.run(
        ...     query="PGSR 和 SuGaR 的差异",
        ...     collection="papers",
        ... )
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        model: str = "qwen3.5-plus-2026-04-20",
    ):
        """Initialize the runner.

        Args:
            llm_provider: LLM provider for the internal loop
            model: Model name to use
        """
        self.llm_provider = llm_provider
        self.model = model
        self._tools = None
        self._session_manager = None
        self._round_manager = None
        self._initialized = False

    @property
    def tools(self):
        """Lazy load tools."""
        if self._tools is None:
            from nanoresearch.rag.internal_loop.tools import InternalTools
            self._tools = InternalTools()
        return self._tools

    @property
    def session_manager(self):
        """Lazy load session manager."""
        if self._session_manager is None:
            from nanoresearch.rag.internal_loop.state import SessionStateManager
            self._session_manager = SessionStateManager()
        return self._session_manager

    @property
    def round_manager(self):
        """Lazy load round manager."""
        if self._round_manager is None:
            from nanoresearch.rag.mcp_server.tools.agentic.round_state import RoundStateManager
            self._round_manager = RoundStateManager.get_instance()
        return self._round_manager

    def _ensure_initialized(self) -> None:
        """Initialize LLM provider if needed."""
        if self._initialized:
            return

        logger.info("[RAG] Initializing LLM provider...")

        if self.llm_provider is None:
            # Try to get from settings
            try:
                from nanoresearch.rag.core.settings import get_settings
                from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider
                from nanoresearch.providers.registry import find_by_name

                settings = get_settings()
                if hasattr(settings, "llm") and settings.llm:
                    llm_config = settings.llm
                    # Convert dataclass to dict
                    config_dict = {
                        "api_key": llm_config.api_key,
                        "api_base": llm_config.base_url,
                        "default_model": llm_config.model,
                    }

                    # Find provider spec by name
                    spec = find_by_name(llm_config.provider)
                    config_dict["spec"] = spec

                    self.llm_provider = OpenAICompatProvider(**config_dict)
                    self.model = llm_config.model
                    logger.info(f"[RAG] LLM provider created: {self.model}")
                else:
                    logger.warning("[RAG] No LLM config found")
            except Exception as e:
                logger.warning(f"[RAG] Could not initialize LLM provider: {e}")

        self._initialized = True

    async def run(
        self,
        query: str,
        context: Optional[str] = None,
        collection: str = "default",
        max_iterations: int = 5,
        session_key: Optional[str] = None,
    ) -> RAGLoopResult:
        """Run the RAG loop.

        Args:
            query: User query
            context: External context from main agent
            collection: Collection to search
            max_iterations: Maximum iterations
            session_key: Outer agent session key (channel:chat_id); propagated to
                plan_query for history-based rewrite. None disables rewrite.

        Returns:
            RAGLoopResult with chunks and citations
        """
        self._ensure_initialized()

        # Import here to avoid circular dependency
        from nanoresearch.rag.internal_loop.state import SessionState

        # Create session
        session = self.session_manager.create_session(
            query=query,
            context=context,
            max_iterations=max_iterations,
            caller_session_key=session_key,
        )

        logger.info(f"Starting RAG loop for query: {query[:50]}...")

        # Import prompts
        from nanoresearch.rag.internal_loop.prompts import build_system_prompt, build_task_instruction
        from nanoresearch.rag.internal_loop.cleanup import cleanup_messages_for_next_round

        # Build initial messages
        messages = self._build_initial_messages(query, context, build_system_prompt)

        try:
            # === Phase 1: Plan ===
            plan_result = await self._run_plan_phase(session, messages)
            session.plan = plan_result
            session.current_phase = "search"

            # === Phase 2-4: Search + Verify Loop ===
            for iteration in range(max_iterations):
                session.iteration = iteration
                logger.debug(f"Starting iteration {iteration + 1}")

                # Create new round for this iteration
                round_state = session.add_round()

                # === Phase 2: Search ===
                search_done = await self._run_search_phase(
                    session, round_state, messages, collection
                )

                if not search_done:
                    # No search executed, might be an error
                    break

                # === Phase 2.5: expand_with_sections (comparison queries, before fuse) ===
                if any(kw in query for kw in COMPARISON_KEYWORDS):
                    await self.tools.expand_with_sections(
                        round_id=round_state.round_id,
                        collection=collection,
                    )

                # === Phase 3: Fuse + Verify (System Forced) ===
                fused_chunks = await self.tools.fuse_results(
                    round_id=round_state.round_id,
                    strategy="rrf",
                    top_k=20,
                    session_state=session,
                )

                if not fused_chunks:
                    logger.warning("No chunks after fusion")
                    continue

                # Verify
                verify_result = await self.tools.verify_results(
                    results=fused_chunks,
                    query=query,
                )
                session.verification_results.append(verify_result)

                logger.info(f"Verification: confidence={verify_result.get('confidence', 0):.2f}")

                # Check termination conditions
                if verify_result.get("answered") or verify_result.get("confidence", 0) >= 0.7:
                    # === Phase 4: Finalize ===
                    session.current_phase = "finalize"
                    citations = await self.tools.build_citations(fused_chunks)

                    return RAGLoopResult(
                        success=True,
                        chunks=fused_chunks,
                        citations=citations,
                        verification=verify_result,
                        iterations=iteration + 1,
                        summary=verify_result.get("summary", "Retrieval completed"),
                    )

                # === Phase 3.5: expand_with_neighbors (verify failed, before next iteration) ===
                neighbor_chunks = await self.tools.expand_with_neighbors(
                    fused_chunks=fused_chunks,
                    collection=collection,
                )
                if neighbor_chunks:
                    fused_chunks = fused_chunks + neighbor_chunks
                    logger.info(f"Expanded chunk pool to {len(fused_chunks)} with neighbors")

                # Not sufficient: inject next_actions
                next_actions = verify_result.get("next_actions", [])
                missing_aspects = verify_result.get("missing_aspects", [])

                if next_actions:
                    # Build task instruction
                    task_instruction = build_task_instruction(
                        confidence=verify_result.get("confidence", 0),
                        missing_aspects=missing_aspects,
                        next_actions=next_actions,
                    )

                    # Inject as system message
                    messages.append({
                        "role": "system",
                        "content": task_instruction,
                    })

                    # Cleanup messages for token control
                    messages = cleanup_messages_for_next_round(messages, verify_result)

                    # Add next_actions to round's sub_queries
                    from nanoresearch.rag.internal_loop.state import SubQuery
                    for action in next_actions:
                        if action.get("action") == "search":
                            sq = SubQuery(
                                query=action.get("query", ""),
                                strategy=action.get("strategy", "hybrid"),
                                reason=action.get("reason", ""),
                            )
                            round_state.sub_queries.append(sq)

                session.current_phase = "search"
                continue

            # Reached max iterations
            logger.warning(f"Reached max iterations: {max_iterations}")
            return RAGLoopResult(
                success=False,
                chunks=session.fused_chunks,
                iterations=max_iterations,
                summary="Reached maximum iterations without sufficient results",
            )

        except Exception as e:
            logger.error(f"RAG loop error: {e}")
            return RAGLoopResult(
                success=False,
                chunks=[],
                iterations=session.iteration,
                error=str(e),
                summary=f"Error: {e}",
            )

    def _build_initial_messages(
        self,
        query: str,
        context: Optional[str],
        build_system_prompt,  # Pass as argument to avoid import
    ) -> List[Dict[str, Any]]:
        """Build initial messages for the loop."""
        messages = []

        # System prompt
        system_prompt = build_system_prompt(query, context)
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        # External context (if provided)
        if context:
            messages.append({
                "role": "system",
                "content": f"[对话上下文]\n{context}",
            })

        # User query
        messages.append({
            "role": "user",
            "content": query,
        })

        return messages

    async def _run_plan_phase(
        self,
        session: "SessionState",
        messages: List[Dict[str, Any]],
    ) -> "PlanResult":
        """Run Phase 1: Plan."""
        # Import here
        from nanoresearch.rag.internal_loop.state import PlanResult, SubQuery

        # Use tools.plan_query
        plan_dict = await self.tools.plan_query(
            query=session.original_query,
            context=session.context,
        )

        # Convert to PlanResult
        sub_queries = []
        for sq_dict in plan_dict.get("sub_queries", []):
            sub_queries.append(SubQuery(
                query=sq_dict.get("query", ""),
                strategy=sq_dict.get("strategy", "hybrid"),
                reason=sq_dict.get("reason", ""),
            ))

        # If no sub_queries, create one from original query
        if not sub_queries:
            sub_queries.append(SubQuery(
                query=session.original_query,
                strategy="hybrid",
                reason="fallback",
            ))

        return PlanResult(
            complexity=plan_dict.get("complexity", "complex"),
            sub_queries=sub_queries,
            context_aware=session.context is not None,
        )

    async def _run_search_phase(
        self,
        session: "SessionState",
        round_state: "RoundState",
        messages: List[Dict[str, Any]],
        collection: str,
    ) -> bool:
        """Run Phase 2: Search."""
        # Import here
        from nanoresearch.rag.internal_loop.state import SubQuery

        # Prepare search tasks from plan
        tasks = []
        for sq in session.plan.sub_queries:
            if not sq.completed:
                tasks.append({
                    "query": sq.query,
                    "strategy": sq.strategy,
                    "top_k": 5,
                })
                sq.completed = True

        # Also add tasks from round_state.sub_queries (from next_actions)
        for sq in round_state.sub_queries:
            if not sq.completed:
                tasks.append({
                    "query": sq.query,
                    "strategy": sq.strategy,
                    "top_k": 5,
                })
                sq.completed = True

        if not tasks:
            logger.warning("No search tasks to execute")
            return False

        # Execute batch retrieval
        result = await self.tools.execute_batch(
            tasks=tasks,
            round_id=round_state.round_id,
            collection=collection,
            session_state=session,
        )

        if result.get("error"):
            logger.error(f"Batch retrieval error: {result['error']}")
            return False

        # Check total_chunks from batch result summary
        total_chunks = result.get("total_chunks", 0)
        logger.info(f"Search returned {total_chunks} total chunks")
        return total_chunks > 0


# Convenience function
async def run_rag_loop(
    query: str,
    context: Optional[str] = None,
    collection: str = "default",
    max_iterations: int = 5,
    session_key: Optional[str] = None,
) -> RAGLoopResult:
    """Run the RAG loop.

    Args:
        query: User query
        context: External context
        collection: Collection to search
        max_iterations: Maximum iterations
        session_key: Outer agent session key

    Returns:
        RAGLoopResult
    """
    runner = RAGLoopRunner()
    return await runner.run(
        query=query,
        context=context,
        collection=collection,
        max_iterations=max_iterations,
        session_key=session_key,
    )
