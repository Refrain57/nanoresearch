"""Shared utilities for Agentic RAG tools.

Common functions used across multiple agentic tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from nanobot.rag.core.types_agentic import retrieval_result_to_dict

logger = logging.getLogger(__name__)


def build_structured_response(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    is_empty: bool = False,
    citations: Optional[List[Any]] = None,
) -> "MCPToolResponse":
    """Build a structured MCP tool response.

    Args:
        content: The main content (usually JSON string)
        metadata: Additional metadata
        is_empty: Whether the result is empty
        citations: Optional citations

    Returns:
        MCPToolResponse instance
    """
    from nanobot.rag.core.response.response_builder import MCPToolResponse

    return MCPToolResponse(
        content=content,
        metadata=metadata or {},
        is_empty=is_empty,
        citations=citations or [],
    )


def build_json_response(
    data: Dict[str, Any],
    is_empty: bool = False,
) -> "MCPToolResponse":
    """Build a JSON-formatted MCP tool response.

    Args:
        data: Dictionary to serialize as JSON
        is_empty: Whether the result is empty

    Returns:
        MCPToolResponse with JSON content
    """
    return build_structured_response(
        content=json.dumps(data, ensure_ascii=False, indent=2),
        metadata={"format": "json"},
        is_empty=is_empty,
    )


def results_to_dict_list(
    results: List[Any],
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    """Convert retrieval results to dictionary list.

    Args:
        results: List of RetrievalResult objects
        include_text: Whether to include text content

    Returns:
        List of dictionaries with result data
    """
    output = []
    for result in results:
        if isinstance(result, dict):
            item = retrieval_result_to_dict(result)
        else:
            item = {
                "chunk_id": getattr(result, "chunk_id", ""),
                "score": getattr(result, "score", 0.0),
                "text": getattr(result, "text", "") if include_text else "",
                "metadata": getattr(result, "metadata", {}),
            }
        output.append(item)
    return output


def format_markdown_results(
    results: List[Any],
    query: str,
    max_results: int = 5,
    max_text_length: int = 300,
) -> str:
    """Format results as Markdown text.

    Args:
        results: List of RetrievalResult objects
        query: The original query
        max_results: Maximum results to display
        max_text_length: Maximum text length per result

    Returns:
        Markdown-formatted string
    """
    if not results:
        return f"## 查询结果\n\n未找到与 **\"{query}\"** 相关的结果。"

    lines = [
        f"## 查询结果\n",
        f"针对查询 **\"{query}\"** 找到 {len(results)} 条结果:\n",
    ]

    for i, result in enumerate(results[:max_results], 1):
        if isinstance(result, dict):
            score = result.get("score", 0.0)
            text = result.get("text", "")
            metadata = result.get("metadata", {})
        else:
            score = getattr(result, "score", 0.0)
            text = getattr(result, "text", "")
            metadata = getattr(result, "metadata", {})

        source = metadata.get("source_path", "unknown")
        page = metadata.get("page_num")

        # Truncate text
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."

        lines.append(f"### {i}. 相关度: {score:.2%}")
        lines.append(f"**来源:** `{source}`")
        if page:
            lines.append(f"**页码:** {page}")
        lines.append(f"\n> {text}\n")

    if len(results) > max_results:
        lines.append(f"\n*...还有 {len(results) - max_results} 条结果*\n")

    return "\n".join(lines)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON text.

    Args:
        text: JSON string to parse

    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation).

    Uses a simple heuristic: ~4 characters per token for Chinese,
    ~1.3 for English.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    # Simple heuristic: count Chinese chars separately
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    # Chinese: ~1.5 tokens per char, English: ~4 chars per token
    return int(chinese_chars * 1.5 + other_chars / 4)
