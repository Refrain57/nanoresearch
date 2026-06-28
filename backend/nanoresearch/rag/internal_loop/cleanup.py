"""Token cleanup logic for RAG internal loop.

Provides functions to prune messages and reduce token consumption
while preserving essential information for the loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Token-related constants
MAX_TEXT_LENGTH = 200  # Truncate text fields longer than this
MAX_RESULTS_TO_KEEP = 3  # Keep only top N results from each tool call
MAX_MESSAGES_TO_KEEP = 10  # Maximum messages to keep in conversation


def cleanup_messages_for_next_round(
    messages: List[Dict[str, Any]],
    verify_result: Optional[Dict[str, Any]] = None,
    keep_rounds: int = 2,
) -> List[Dict[str, Any]]:
    """Clean up messages for the next iteration to control token usage.

    Strategy:
    - Keep all system messages (they provide context)
    - Keep the original user query (first user message)
    - Keep the latest verify result summary
    - Prune tool results to keep messages small

    Args:
        messages: Original messages list
        verify_result: Latest verification result (optional)
        keep_rounds: Number of recent rounds to preserve

    Returns:
        Cleaned messages list
    """
    cleaned: List[Dict[str, Any]] = []
    user_queries: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            # Keep all system messages
            cleaned.append(msg)

        elif role == "user":
            # Collect user queries
            user_queries.append(msg)

        elif role in ("tool", "assistant"):
            # Collect tool/assistant results
            tool_results.append(msg)

    # Keep the first user query (original query)
    if user_queries:
        cleaned.append(user_queries[0])

    # Prune and add tool results
    pruned_tool_results = _prune_tool_results(tool_results, keep_rounds)
    cleaned.extend(pruned_tool_results)

    logger.debug(f"Cleaned messages: {len(messages)} -> {len(cleaned)}")

    return cleaned


def _prune_tool_results(
    tool_results: List[Dict[str, Any]],
    keep_rounds: int,
) -> List[Dict[str, Any]]:
    """Prune tool results to reduce token usage.

    Args:
        tool_results: List of tool result messages
        keep_rounds: Number of recent rounds to keep

    Returns:
        Pruned tool results
    """
    if not tool_results:
        return []

    # Keep only the last N tool result blocks
    # Each "round" might have multiple tool calls, so we need to identify
    # round boundaries. For simplicity, we keep the last `keep_rounds` blocks.

    # Group tool results by approximate round
    # Each tool result is usually followed by an assistant response
    # For now, just take the last few

    max_results = keep_rounds * 2  # Rough estimate: 2 tool calls per round

    if len(tool_results) <= max_results:
        # Not too many, prune text fields only
        return [_prune_single_result(r) for r in tool_results]

    # Too many, keep recent ones and prune aggressively
    recent = tool_results[-max_results:]
    pruned_recent = [_prune_single_result(r) for r in recent]

    return pruned_recent


def _prune_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Prune a single tool result to reduce size.

    Args:
        result: Tool result dictionary

    Returns:
        Pruned result
    """
    pruned = result.copy()

    # Handle content that might be JSON
    if isinstance(pruned.get("content"), str):
        try:
            import json
            content_data = json.loads(pruned["content"])
            if isinstance(content_data, dict):
                pruned_content = _prune_dict_content(content_data)
                pruned["content"] = json.dumps(pruned_content, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, truncate if too long
            content = pruned["content"]
            if len(content) > MAX_TEXT_LENGTH * 2:
                pruned["content"] = content[:MAX_TEXT_LENGTH * 2] + "..."

    return pruned


def _prune_dict_content(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively prune a dictionary to reduce token usage.

    Args:
        data: Dictionary to prune

    Returns:
        Pruned dictionary
    """
    if not isinstance(data, dict):
        return data

    pruned = {}

    for key, value in data.items():
        # Skip keys that are not useful for next iteration
        if key in ("text", "full_text", "content"):
            # Truncate long text
            if isinstance(value, str) and len(value) > MAX_TEXT_LENGTH:
                pruned[key] = value[:MAX_TEXT_LENGTH] + "..."
            else:
                pruned[key] = value

        elif key == "results" and isinstance(value, list):
            # Keep only top results
            pruned[key] = _prune_results_list(value)

        elif key == "summary" and isinstance(value, str):
            # Keep summary but truncate if too long
            if len(value) > MAX_TEXT_LENGTH:
                pruned[key] = value[:MAX_TEXT_LENGTH] + "..."
            else:
                pruned[key] = value

        elif isinstance(value, dict):
            # Recursively prune nested dicts
            nested = _prune_dict_content(value)
            if nested:  # Only add if not empty
                pruned[key] = nested

        elif isinstance(value, list):
            # For other lists, limit length
            if len(value) > MAX_RESULTS_TO_KEEP:
                pruned[key] = value[:MAX_RESULTS_TO_KEEP]
            else:
                pruned[key] = value

        else:
            # Keep other values as-is
            pruned[key] = value

    return pruned


def _prune_results_list(results: List[Any]) -> List[Any]:
    """Prune a list of retrieval results.

    Args:
        results: List of result dictionaries

    Returns:
        Pruned list
    """
    if not isinstance(results, list):
        return results

    # Keep only top N results
    if len(results) > MAX_RESULTS_TO_KEEP:
        results = results[:MAX_RESULTS_TO_KEEP]

    pruned = []
    for result in results:
        if isinstance(result, dict):
            pruned_result = {}
            # Keep essential fields
            essential_keys = ["chunk_id", "score", "metadata", "source", "title"]
            for key in essential_keys:
                if key in result:
                    value = result[key]
                    # Truncate text
                    if key == "text" and isinstance(value, str) and len(value) > MAX_TEXT_LENGTH:
                        value = value[:MAX_TEXT_LENGTH] + "..."
                    pruned_result[key] = value
            if pruned_result:
                pruned.append(pruned_result)
        else:
            pruned.append(result)

    return pruned


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate token count for messages.

    Rough estimation: 1 token ≈ 4 characters for Chinese,
    1 token ≈ 0.75 words for English.

    Args:
        messages: Messages to estimate

    Returns:
        Estimated token count
    """
    total_chars = 0

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if isinstance(text, str):
                        total_chars += len(text)

    # Rough estimation
    return total_chars // 3
