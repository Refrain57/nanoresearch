"""Session management tools for Agentic RAG.

Manages multi-turn search sessions with state persistence.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from nanobot.rag.core.session import FileSessionStore, MemorySessionStore
from nanobot.rag.core.types_agentic import SearchSession
from nanobot.rag.mcp_server.tools.agentic.shared import (
    build_json_response,
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Manager for search sessions.

    Provides session creation, retrieval, update, and cleanup.
    Uses file-based storage for persistence.
    """

    _instance = None
    _store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._store is None:
            self._init_store()

    def _init_store(self) -> None:
        """Initialize session store."""
        try:
            from nanobot.rag.core.settings import get_settings, resolve_path

            settings = get_settings()

            # Check for configured session store
            if hasattr(settings, "session_store"):
                store_config = settings.session_store.model_dump()
                store_type = store_config.get("type", "file")

                if store_type == "file":
                    sessions_dir = store_config.get("path", "./data/sessions")
                    sessions_dir = resolve_path(sessions_dir)
                    self._store = FileSessionStore(sessions_dir=sessions_dir)
                else:
                    self._store = MemorySessionStore()
            else:
                # Default to file store
                sessions_dir = "./data/sessions"
                try:
                    sessions_dir = resolve_path(sessions_dir)
                except Exception:
                    pass  # Use relative path
                self._store = FileSessionStore(sessions_dir=sessions_dir)

            logger.info(f"Session store initialized: {type(self._store).__name__}")
        except Exception as e:
            logger.warning(f"Failed to initialize file session store: {e}")
            self._store = MemorySessionStore()


# Global session manager
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get the global session manager.

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


class CreateSessionTool:
    """MCP tool for creating a new search session."""

    @property
    def name(self) -> str:
        return "create_session"

    @property
    def description(self) -> str:
        return """Create a new multi-turn search session.

Args:
    initial_query: The initial search query
    collection: Collection to search (default: "default")
    metadata: Optional session metadata (JSON string)

Returns:
    JSON with session_id and session state."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "initial_query": {
                    "type": "string",
                    "description": "The initial search query",
                },
                "collection": {
                    "type": "string",
                    "default": "default",
                    "description": "Collection to search",
                },
                "metadata": {
                    "type": "string",
                    "description": "Optional session metadata (JSON string)",
                },
            },
            "required": ["initial_query"],
        }

    async def execute(
        self,
        initial_query: str,
        collection: str = "default",
        metadata: Optional[str] = None,
    ) -> "MCPToolResponse":
        """Execute the create session tool.

        Args:
            initial_query: The initial query
            collection: Target collection
            metadata: Optional metadata JSON

        Returns:
            MCPToolResponse with session data
        """
        manager = get_session_manager()

        # Parse metadata if provided
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                meta_dict = {"raw_metadata": metadata}

        # Create session
        session_id = str(uuid.uuid4())[:8]  # Short ID for usability
        session = SearchSession(
            session_id=session_id,
            created_at=datetime.now(),
            initial_query=initial_query,
            collection=collection,
            current_query=initial_query,
            query_history=[initial_query],
            metadata=meta_dict,
        )

        # Store session
        manager._store.create(session)

        return build_json_response({
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "initial_query": initial_query,
            "collection": collection,
            "state": "active",
            "query_count": 1,
        })


class GetSessionTool:
    """MCP tool for retrieving session state."""

    @property
    def name(self) -> str:
        return "get_session"

    @property
    def description(self) -> str:
        return """Get the current state of a search session.

Args:
    session_id: The session ID to retrieve
    include_results: Whether to include retrieval results (default: true)

Returns:
    JSON with full session state including query history and results."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to retrieve",
                },
                "include_results": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include retrieval results",
                },
            },
            "required": ["session_id"],
        }

    async def execute(
        self,
        session_id: str,
        include_results: bool = True,
    ) -> "MCPToolResponse":
        """Execute the get session tool.

        Args:
            session_id: Session ID to get
            include_results: Whether to include results

        Returns:
            MCPToolResponse with session state
        """
        manager = get_session_manager()

        session = manager._store.get(session_id)

        if not session:
            return build_json_response(
                {"error": f"Session not found: {session_id}"},
                is_empty=True,
            )

        # Build response
        data = {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "initial_query": session.initial_query,
            "current_query": session.current_query,
            "collection": session.collection,
            "query_history": session.query_history,
            "query_count": len(session.query_history),
            "total_results": len(session.all_results),
            "state": "active",
        }

        if include_results:
            data["retrieval_results"] = {
                k: v for k, v in session.retrieval_results.items()
            }
            data["all_results"] = session.all_results[:10]  # Limit for response

        return build_json_response(data)


class UpdateSessionTool:
    """MCP tool for updating session state."""

    @property
    def name(self) -> str:
        return "update_session"

    @property
    def description(self) -> str:
        return """Update a session with new query or results.

Args:
    session_id: The session ID to update
    query: New query to add to history (optional)
    results: New results to store (JSON string, optional)
    results_key: Key to store results under (default: "latest")

Returns:
    JSON with updated session state."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to update",
                },
                "query": {
                    "type": "string",
                    "description": "New query to add to history",
                },
                "results": {
                    "type": "string",
                    "description": "New results to store (JSON string)",
                },
                "results_key": {
                    "type": "string",
                    "default": "latest",
                    "description": "Key to store results under",
                },
            },
            "required": ["session_id"],
        }

    async def execute(
        self,
        session_id: str,
        query: Optional[str] = None,
        results: Optional[str] = None,
        results_key: str = "latest",
    ) -> "MCPToolResponse":
        """Execute the update session tool.

        Args:
            session_id: Session ID to update
            query: New query to add
            results: New results to store
            results_key: Key for results

        Returns:
            MCPToolResponse with updated state
        """
        manager = get_session_manager()

        session = manager._store.get(session_id)

        if not session:
            return build_json_response(
                {"error": f"Session not found: {session_id}"},
                is_empty=True,
            )

        # Update query if provided
        if query:
            session.current_query = query
            session.query_history.append(query)

        # Update results if provided
        if results:
            try:
                results_list = json.loads(results)
                session.retrieval_results[results_key] = results_list
                session.all_results.extend(results_list)
            except json.JSONDecodeError:
                logger.warning(f"Invalid results JSON for session {session_id}")

        # Save updated session
        manager._store.update(session)

        return build_json_response({
            "session_id": session_id,
            "updated": True,
            "query_count": len(session.query_history),
            "current_query": session.current_query,
            "total_results": len(session.all_results),
        })


class CloseSessionTool:
    """MCP tool for closing a search session."""

    @property
    def name(self) -> str:
        return "close_session"

    @property
    def description(self) -> str:
        return """Close and optionally delete a search session.

Args:
    session_id: The session ID to close
    delete: Whether to delete the session data (default: false)

Returns:
    JSON confirming session closure."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to close",
                },
                "delete": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to delete session data",
                },
            },
            "required": ["session_id"],
        }

    async def execute(
        self,
        session_id: str,
        delete: bool = False,
    ) -> "MCPToolResponse":
        """Execute the close session tool.

        Args:
            session_id: Session ID to close
            delete: Whether to delete

        Returns:
            MCPToolResponse confirming closure
        """
        manager = get_session_manager()

        session = manager._store.get(session_id)

        if not session:
            return build_json_response(
                {"error": f"Session not found: {session_id}"},
                is_empty=True,
            )

        if delete:
            manager._store.delete(session_id)
            return build_json_response({
                "session_id": session_id,
                "closed": True,
                "deleted": True,
                "message": f"Session {session_id} closed and deleted",
            })
        else:
            return build_json_response({
                "session_id": session_id,
                "closed": True,
                "deleted": False,
                "message": f"Session {session_id} marked as closed",
                "query_count": len(session.query_history),
                "total_results": len(session.all_results),
            })


class ListSessionsTool:
    """MCP tool for listing active sessions."""

    @property
    def name(self) -> str:
        return "list_sessions"

    @property
    def description(self) -> str:
        return """List all active search sessions.

Args:
    limit: Maximum sessions to return (default: 10)

Returns:
    JSON with list of active sessions."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum sessions to return",
                },
            },
        }

    async def execute(self, limit: int = 10) -> "MCPToolResponse":
        """Execute the list sessions tool.

        Args:
            limit: Maximum sessions to list

        Returns:
            MCPToolResponse with session list
        """
        manager = get_session_manager()

        sessions = manager._store.list_active()

        # Build summary list
        session_list = []
        for session in sessions[:limit]:
            session_list.append({
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "initial_query": session.initial_query,
                "current_query": session.current_query,
                "collection": session.collection,
                "query_count": len(session.query_history),
            })

        return build_json_response({
            "sessions": session_list,
            "total": len(sessions),
            "limit": limit,
        })


# MCP Tool Handlers
async def create_session_handler(
    initial_query: str,
    collection: str = "default",
    metadata: Optional[str] = None,
) -> "MCPToolResponse":
    """Handler for create_session MCP tool."""
    tool = CreateSessionTool()
    return await tool.execute(
        initial_query=initial_query,
        collection=collection,
        metadata=metadata,
    )


async def get_session_handler(
    session_id: str,
    include_results: bool = True,
) -> "MCPToolResponse":
    """Handler for get_session MCP tool."""
    tool = GetSessionTool()
    return await tool.execute(
        session_id=session_id,
        include_results=include_results,
    )


async def update_session_handler(
    session_id: str,
    query: Optional[str] = None,
    results: Optional[str] = None,
    results_key: str = "latest",
) -> "MCPToolResponse":
    """Handler for update_session MCP tool."""
    tool = UpdateSessionTool()
    return await tool.execute(
        session_id=session_id,
        query=query,
        results=results,
        results_key=results_key,
    )


async def close_session_handler(
    session_id: str,
    delete: bool = False,
) -> "MCPToolResponse":
    """Handler for close_session MCP tool."""
    tool = CloseSessionTool()
    return await tool.execute(
        session_id=session_id,
        delete=delete,
    )


async def list_sessions_handler(limit: int = 10) -> "MCPToolResponse":
    """Handler for list_sessions MCP tool."""
    tool = ListSessionsTool()
    return await tool.execute(limit=limit)


def register_tools(protocol_handler: Any) -> None:
    """Register session management tools with the protocol handler.

    Args:
        protocol_handler: The MCP protocol handler instance
    """
    tools = [
        CreateSessionTool(),
        GetSessionTool(),
        UpdateSessionTool(),
        CloseSessionTool(),
        ListSessionsTool(),
    ]

    handlers = {
        "create_session": create_session_handler,
        "get_session": get_session_handler,
        "update_session": update_session_handler,
        "close_session": close_session_handler,
        "list_sessions": list_sessions_handler,
    }

    for tool in tools:
        protocol_handler.register_tool(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=handlers[tool.name],
        )
        logger.info(f"Registered agentic tool: {tool.name}")