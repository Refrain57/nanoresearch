"""Search history tool — search user_memory for conversation history.

DEPRECATED: This tool is no longer needed. User memory is now automatically
retrieved via RAG in build_history_context(). The tool file is kept for
backward compatibility but should not be registered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nanoresearch.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanoresearch.providers.base import LLMProvider


class SearchHistoryTool(Tool):
    """DEPRECATED: Use RAG-based user_memory retrieval instead."""

    name = "search_history"
    description = (
        "[DEPRECATED] 搜索对话历史记录。"
        "历史记忆现在通过 RAG 自动检索，无需手动搜索。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "limit": {
                "type": "integer",
                "default": 5,
                "description": "返回条数上限（默认 5）",
            },
        },
        "required": ["query"],
    }

    def __init__(self, workspace_path: str):
        pass  # No-op for backward compatibility

    def execute(self, query: str, limit: int = 5) -> str:
        return (
            "search_history 工具已废弃。"
            "历史记忆现在通过 RAG 自动检索，相关内容会自动注入到上下文中。"
        )
