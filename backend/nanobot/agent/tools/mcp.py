"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


def _extract_nullable_branch(options: Any) -> tuple[dict[str, Any], bool] | None:
    """Return the single non-null branch for nullable unions."""
    if not isinstance(options, list):
        return None

    non_null: list[dict[str, Any]] = []
    saw_null = False
    for option in options:
        if not isinstance(option, dict):
            return None
        if option.get("type") == "null":
            saw_null = True
            continue
        non_null.append(option)

    if saw_null and len(non_null) == 1:
        return non_null[0], True
    return None


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    """Normalize only nullable JSON Schema patterns for tool definitions."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)

    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        non_null = [item for item in raw_type if item != "null"]
        if "null" in raw_type and len(non_null) == 1:
            normalized["type"] = non_null[0]
            normalized["nullable"] = True

    for key in ("oneOf", "anyOf"):
        nullable_branch = _extract_nullable_branch(normalized.get(key))
        if nullable_branch is not None:
            branch, _ = nullable_branch
            merged = {k: v for k, v in normalized.items() if k != key}
            merged.update(branch)
            normalized = merged
            normalized["nullable"] = True
            break

    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            name: _normalize_schema_for_openai(prop)
            if isinstance(prop, dict)
            else prop
            for name, prop in normalized["properties"].items()
        }

    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_schema_for_openai(normalized["items"])

    if normalized.get("type") != "object":
        return normalized

    normalized.setdefault("properties", {})
    normalized.setdefault("required", [])
    return normalized


# MCP tool original names (from MCP server, before server-name prefix) confirmed read-only.
# Each entry must be verified to make no writes before being added.
# Conservative default: any MCP tool NOT listed here is treated as side-effect.
_QUERY_MCP_TOOL_ORIGINAL_NAMES: frozenset[str] = frozenset({
    # RAG retrieval tools — query ChromaDB / BM25 indexes only, no writes
    "retrieve_hybrid",
    "retrieve_dense",
    "retrieve_sparse",
    "kb_search",
    "kb_retrieve",
})


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a nanobot Tool."""

    def __init__(self, session, server_name: str, tool_def, tool_timeout: int = 30):
        self._session = session
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        raw_schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._parameters = _normalize_schema_for_openai(raw_schema)
        self._tool_timeout = tool_timeout
        self._session_key: str | None = None  # Auto-injected for query rewriting
        self._kb_map: dict[str, str] = {}  # {kb_id: chroma_collection}, set per-run

    def set_kb_map(self, kb_map: dict[str, str]) -> None:
        """Set the kb_id → chroma_collection mapping for the current run."""
        self._kb_map = kb_map

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def side_effect(self) -> bool:
        return self._original_name not in _QUERY_MCP_TOOL_ORIGINAL_NAMES

    def set_session_key(self, session_key: str) -> None:
        """Set session key for automatic injection into query rewriting tools."""
        self._session_key = session_key

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        if self._original_name == "kb_search":
            logger.debug(f"MCP execute: _session_key={self._session_key}, kwargs={kwargs}")

        # list_collections: always intercept — never expose ChromaDB internals.
        # Returns only the user's bound KBs; empty list when no KBs are configured.
        if self._original_name == "list_collections":
            import json as _json
            collections = [{"name": col, "type": "chroma", "kb_id": kid}
                           for kid, col in self._kb_map.items()]
            return _json.dumps({"collections": collections, "total": len(collections)}, ensure_ascii=False)

        # Resolve kb_id → chroma_collection for KB tools (kb_search, kb_retrieve, list_documents).
        # The Agent passes kb_id; mcp.py converts to collection before forwarding to MCP subprocess.
        if self._original_name in ("kb_search", "kb_retrieve", "list_documents"):
            _kb_id = kwargs.pop("kb_id", None)
            if _kb_id and _kb_id in self._kb_map:
                kwargs["collection"] = self._kb_map[_kb_id]
                logger.debug("MCP {}: resolved kb_id={} → collection={!r}", self._original_name, _kb_id, kwargs["collection"])
            elif not _kb_id and len(self._kb_map) == 1:
                _col = next(iter(self._kb_map.values()))
                kwargs["collection"] = _col
                logger.debug("MCP {}: single KB auto-inject → collection={!r}", self._original_name, _col)
            elif _kb_id and _kb_id not in self._kb_map:
                if len(self._kb_map) == 1:
                    _col = next(iter(self._kb_map.values()))
                    kwargs["collection"] = _col
                    logger.debug("MCP {}: unknown kb_id={!r}, falling back to single KB → collection={!r}",
                                 self._original_name, _kb_id, _col)
                else:
                    logger.warning("MCP {}: unknown kb_id={!r}, not in kb_map", self._original_name, _kb_id)

        # Whitelist guard: collection must belong to this user's kb_map.
        # Prevents access to other users' collections even if collection name is known.
        if self._original_name in ("kb_search", "kb_retrieve", "list_documents"):
            import json as _json
            _allowed = set(self._kb_map.values())
            _requested_col = kwargs.get("collection")
            if not _allowed:
                return _json.dumps({
                    "error": "No knowledge base configured for this agent. Please create a knowledge base in the Web UI and bind it to the agent.",
                    "access_denied": True,
                }, ensure_ascii=False)
            if not _requested_col or _requested_col not in _allowed:
                logger.warning("MCP {}: access denied — collection={!r} not in kb_map", self._original_name, _requested_col)
                return _json.dumps({
                    "error": f"Access denied: collection {_requested_col!r} is not bound to this agent.",
                    "access_denied": True,
                }, ensure_ascii=False)

        # memory_search: fixed collections, no kb_map resolution needed — pass through directly

        # ingest_document: enforce kb_id rules — auto-inject for single KB, error for ambiguous/missing.
        if self._original_name == "ingest_document":
            import json as _json
            _kb_id = kwargs.get("kb_id")
            if not _kb_id:
                if not self._kb_map:
                    return _json.dumps({
                        "error": "No knowledge base configured for this agent. Please create a knowledge base in the Web UI and bind it to the agent.",
                        "accepted": False,
                    }, ensure_ascii=False)
                elif len(self._kb_map) == 1:
                    kwargs["kb_id"] = next(iter(self._kb_map.keys()))
                    logger.debug("MCP ingest_document: single KB auto-inject → kb_id={!r}", kwargs["kb_id"])
                else:
                    available = [{"kb_id": kid, "collection": col} for kid, col in self._kb_map.items()]
                    return _json.dumps({
                        "error": "ingest_document requires kb_id when multiple knowledge bases are available. Specify kb_id.",
                        "available_kbs": available,
                        "accepted": False,
                    }, ensure_ascii=False)

        # delete_document: auto-inject kb_id for single KB.
        if self._original_name == "delete_document":
            _kb_id = kwargs.get("kb_id")
            if not _kb_id and len(self._kb_map) == 1:
                kwargs["kb_id"] = next(iter(self._kb_map.keys()))
                logger.debug("MCP delete_document: single KB auto-inject → kb_id={!r}", kwargs["kb_id"])

        effective_timeout = self._tool_timeout

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("MCP tool '{}' timed out after {}s", self._name, effective_timeout)
            return f"(MCP tool call timed out after {effective_timeout}s)"
        except asyncio.CancelledError:
            # MCP SDK's anyio cancel scopes can leak CancelledError on timeout/failure.
            # Re-raise only if our task was externally cancelled (e.g. /stop).
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
            return "(MCP tool call was cancelled)"
        except Exception as exc:
            logger.exception(
                "MCP tool '{}' failed: {}: {}",
                self._name,
                type(exc).__name__,
                exc,
            )
            return f"(MCP tool call failed: {type(exc).__name__})"

        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))

        raw_result = "\n".join(parts) or "(no output)"
        logger.info(f"MCP tool '{self._name}' raw result: {raw_result[:300]}")
        return raw_result


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry, stack: AsyncExitStack
) -> None:
    """Connect to configured MCP servers and register their tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    for name, cfg in mcp_servers.items():
        try:
            transport_type = cfg.type
            if not transport_type:
                if cfg.command:
                    transport_type = "stdio"
                elif cfg.url:
                    # Convention: URLs ending with /sse use SSE transport; others use streamableHttp
                    transport_type = (
                        "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
                    )
                else:
                    logger.warning("MCP server '{}': no command or url configured, skipping", name)
                    continue

            if transport_type == "stdio":
                import os as _os
                # Merge parent env with config overrides so the subprocess gets DATABASE_URL,
                # REDIS_URL, etc. while still allowing config.json to override PYTHONPATH etc.
                _stdio_env = {**_os.environ, **(cfg.env or {})}
                params = StdioServerParameters(
                    command=cfg.command, args=cfg.args, env=_stdio_env
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif transport_type == "sse":
                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    merged_headers = {**(cfg.headers or {}), **(headers or {})}
                    return httpx.AsyncClient(
                        headers=merged_headers or None,
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                    )

                read, write = await stack.enter_async_context(
                    sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                )
            elif transport_type == "streamableHttp":
                # Always provide an explicit httpx client so MCP HTTP transport does not
                # inherit httpx's default 5s timeout and preempt the higher-level tool timeout.
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,
                        follow_redirects=True,
                        timeout=None,
                    )
                )
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
            else:
                logger.warning("MCP server '{}': unknown transport type '{}'", name, transport_type)
                continue

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = await session.list_tools()
            enabled_tools = set(cfg.enabled_tools)
            allow_all_tools = "*" in enabled_tools
            registered_count = 0
            matched_enabled_tools: set[str] = set()
            available_raw_names = [tool_def.name for tool_def in tools.tools]
            available_wrapped_names = [f"mcp_{name}_{tool_def.name}" for tool_def in tools.tools]
            for tool_def in tools.tools:
                wrapped_name = f"mcp_{name}_{tool_def.name}"
                if (
                    not allow_all_tools
                    and tool_def.name not in enabled_tools
                    and wrapped_name not in enabled_tools
                ):
                    logger.debug(
                        "MCP: skipping tool '{}' from server '{}' (not in enabledTools)",
                        wrapped_name,
                        name,
                    )
                    continue
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.tool_timeout)
                registry.register(wrapper)
                logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)
                registered_count += 1
                if enabled_tools:
                    if tool_def.name in enabled_tools:
                        matched_enabled_tools.add(tool_def.name)
                    if wrapped_name in enabled_tools:
                        matched_enabled_tools.add(wrapped_name)

            if enabled_tools and not allow_all_tools:
                unmatched_enabled_tools = sorted(enabled_tools - matched_enabled_tools)
                if unmatched_enabled_tools:
                    logger.warning(
                        "MCP server '{}': enabledTools entries not found: {}. Available raw names: {}. "
                        "Available wrapped names: {}",
                        name,
                        ", ".join(unmatched_enabled_tools),
                        ", ".join(available_raw_names) or "(none)",
                        ", ".join(available_wrapped_names) or "(none)",
                    )

            logger.info("MCP server '{}': connected, {} tools registered", name, registered_count)
        except Exception as e:
            logger.error("MCP server '{}': failed to connect: {}", name, e)
