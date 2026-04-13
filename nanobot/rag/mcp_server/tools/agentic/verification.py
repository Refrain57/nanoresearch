"""Verification tool for Agentic RAG.

Uses LLM to verify if retrieval results answer the query.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from nanobot.rag.core.types_agentic import VerificationResult
from nanobot.rag.mcp_server.tools.agentic.shared import (
    build_json_response,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


class VerifyResultsTool:
    """MCP tool for verifying retrieval results.

    Allows Agent to:
    - Check if results answer the query
    - Get suggestions for query refinement
    - Understand result coverage
    """

    # Verification prompt template
    VERIFICATION_PROMPT = """你是一个RAG系统评估专家。请分析以下检索结果是否能回答用户的查询。

## 用户查询
{query}

## 检索结果 ({total_results} 条)
{results_text}

## 分析要求
1. 判断这些结果是否能完整回答用户的问题
2. 评估每个结果的相关性
3. 如果不能完全回答，建议改进的查询方向

## 输出格式 (JSON)
请输出一个JSON对象，格式如下：
```json
{{
  "answered": true/false,
  "confidence": 0.0-1.0,
  "summary": "简要说明结果覆盖情况",
  "per_result": [
    {{"index": 0, "relevant": true, "reason": "相关原因"}},
    ...
  ],
  "suggestions": {{
    "refined_queries": ["改进查询建议1", "改进查询建议2"],
    "missing_aspects": ["缺失的方面1"],
    "additional_searches": ["建议的额外搜索"]
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
            # Try to get LLM configuration
            if hasattr(settings, "llm") and settings.llm:
                self._init_llm_client(settings.llm.model_dump())
            elif hasattr(settings, "embedding") and settings.embedding:
                # Fallback to embedding provider config
                self._init_llm_client(settings.embedding.model_dump())

            self._initialized = True
            logger.info("LLM client initialized for verification tool")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            self._initialized = True

    def _init_llm_client(self, config: Dict[str, Any]) -> None:
        """Initialize LLM client from config.

        Args:
            config: LLM configuration dict
        """
        provider = config.get("provider", "dashscope")

        if provider == "dashscope" or provider == "aliyun":
            import dashscope
            from dashscope import Generation

            api_key = config.get("api_key") or config.get("dashscope_api_key")
            if api_key:
                dashscope.api_key = api_key
                self._llm_client = Generation
                self._llm_model = config.get("model", "qwen-turbo")
            else:
                logger.warning("No API key for DashScope")

        elif provider == "openai":
            import openai

            api_key = config.get("api_key")
            base_url = config.get("base_url")
            if api_key:
                self._llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)
                self._llm_model = config.get("model", "gpt-3.5-turbo")

        else:
            logger.warning(f"Unknown LLM provider: {provider}")

    @property
    def name(self) -> str:
        return "verify_results"

    @property
    def description(self) -> str:
        return """Evaluate if retrieval results actually answer the user's query.

This is the CRITICAL step after retrieval - always call this to check quality.

When to use:
- After ANY retrieval operation (retrieve_dense, retrieve_sparse, retrieve_hybrid)
- When you're not sure if results are sufficient
- Before returning final answer to user

Why:
- Low confidence means you need to refine query or try different retrieval strategy
- Suggestions tell you exactly what's missing and how to improve

Args:
    results: JSON string of retrieval results (from retrieve_* tools)
    query: The original user question
    max_results_to_analyze: How many results to analyze (default: 5)

Returns:
    JSON with:
    - answered: Whether results cover the query (boolean)
    - confidence: 0.0-1.0 score for answer quality
    - summary: Brief assessment
    - suggestions.refined_queries: Better search terms if confidence is low
    - suggestions.missing_aspects: Topics not covered in results

IMPORTANT: If confidence < 0.7, you MUST:
1. Use suggestions.refined_queries to retry retrieval
2. Or try a different retrieval strategy (dense vs sparse)
3. Never ignore low confidence and assume results are good"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "string",
                    "description": "JSON string of retrieval results",
                },
                "query": {
                    "type": "string",
                    "description": "The original query",
                },
                "max_results_to_analyze": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum results to include in analysis",
                },
            },
            "required": ["results", "query"],
        }

    async def execute(
        self,
        results: str,
        query: str,
        max_results_to_analyze: int = 5,
    ) -> "MCPToolResponse":
        """Execute the verification tool.

        Args:
            results: JSON string of results to verify
            query: The original query
            max_results_to_analyze: Maximum results to analyze

        Returns:
            MCPToolResponse with verification result
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
                VerificationResult(
                    query=query,
                    total_results=0,
                    answered=False,
                    confidence=0.0,
                    summary="No results to verify",
                    suggestions={"refined_queries": [query]},
                ).to_dict()
            )

        # Initialize LLM
        await asyncio.to_thread(self._ensure_initialized)

        if not self._llm_client:
            # Fallback: simple heuristic verification
            return build_json_response(
                self._heuristic_verification(query, results_list, max_results_to_analyze)
            )

        try:
            # Prepare results text
            results_text = self._format_results_for_prompt(results_list[:max_results_to_analyze])

            # Build prompt
            prompt = self.VERIFICATION_PROMPT.format(
                query=query,
                total_results=len(results_list),
                results_text=results_text,
            )

            # Call LLM
            llm_response = await asyncio.to_thread(
                self._call_llm,
                prompt,
            )

            # Parse response
            verification = self._parse_llm_response(llm_response, query, len(results_list))

            return build_json_response(verification)

        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return build_json_response(
                self._heuristic_verification(query, results_list, max_results_to_analyze)
            )

    def _format_results_for_prompt(self, results: List[Any]) -> str:
        """Format results for LLM prompt.

        Args:
            results: List of results

        Returns:
            Formatted text
        """
        lines = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                text = r.get("text", "")
                score = r.get("score", 0.0)
            else:
                text = getattr(r, "text", "")
                score = getattr(r, "score", 0.0)

            # Truncate long text
            if len(text) > 500:
                text = text[:500] + "..."

            lines.append(f"### 结果 {i + 1} (相关度: {score:.2%})\n{text}\n")

        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for verification.

        Args:
            prompt: The verification prompt

        Returns:
            LLM response text
        """
        if hasattr(self._llm_client, "call"):
            # DashScope style
            response = self._llm_client.call(
                model=self._llm_model,
                prompt=prompt,
            )
            return response.output.text if hasattr(response, "output") else str(response)

        elif hasattr(self._llm_client, "chat"):
            # OpenAI style
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content

        else:
            raise ValueError("Unknown LLM client type")

    def _parse_llm_response(
        self, response: str, query: str, total_results: int
    ) -> Dict[str, Any]:
        """Parse LLM response into verification result.

        Args:
            response: LLM response text
            query: Original query
            total_results: Total number of results

        Returns:
            Verification dict
        """
        import re

        # Try to extract JSON
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON directly
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        try:
            data = json.loads(json_str)
            return {
                "query": query,
                "total_results": total_results,
                "answered": data.get("answered", False),
                "confidence": float(data.get("confidence", 0.5)),
                "summary": data.get("summary", ""),
                "per_result": data.get("per_result", []),
                "suggestions": data.get("suggestions", {}),
            }
        except json.JSONDecodeError:
            # Fallback parsing
            answered = "answered\": true" in response.lower() or "\"answered\":true" in response.lower()
            return {
                "query": query,
                "total_results": total_results,
                "answered": answered,
                "confidence": 0.6 if answered else 0.3,
                "summary": "LLM analysis completed",
                "per_result": [],
                "suggestions": {"refined_queries": [query]},
            }

    def _heuristic_verification(
        self, query: str, results: List[Any], max_results: int
    ) -> Dict[str, Any]:
        """Simple heuristic verification when LLM is unavailable.

        Args:
            query: The query
            results: Results list
            max_results: Max to analyze

        Returns:
            Verification dict
        """
        # Check if results have reasonable scores
        scores = []
        for r in results[:max_results]:
            if isinstance(r, dict):
                scores.append(float(r.get("score", 0.0)))
            else:
                scores.append(float(getattr(r, "score", 0.0)))

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Simple heuristic: high scores = likely answered
        answered = avg_score > 0.5
        confidence = min(avg_score, 0.8) if answered else max(avg_score, 0.3)

        return {
            "query": query,
            "total_results": len(results),
            "answered": answered,
            "confidence": confidence,
            "summary": f"Average relevance score: {avg_score:.2%}",
            "per_result": [],
            "suggestions": {
                "refined_queries": [query],
                "note": "LLM verification unavailable, using heuristic",
            },
        }


# MCP Tool Handler
async def verify_results_handler(
    results: str,
    query: str,
    max_results_to_analyze: int = 5,
) -> "MCPToolResponse":
    """Handler for verify_results MCP tool."""
    tool = VerifyResultsTool()
    return await tool.execute(
        results=results,
        query=query,
        max_results_to_analyze=max_results_to_analyze,
    )


def register_tools(protocol_handler: Any) -> None:
    """Register verification tools with the protocol handler.

    Args:
        protocol_handler: The MCP protocol handler instance
    """
    tool = VerifyResultsTool()
    protocol_handler.register_tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        handler=verify_results_handler,
    )
    logger.info(f"Registered agentic tool: {tool.name}")
