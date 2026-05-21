"""Citation building tool for Agentic RAG.

Generates structured citations from retrieval results.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from nanobot.rag.mcp_server.tools.agentic.shared import (
    build_json_response,
    build_structured_response,
)

logger = logging.getLogger(__name__)


class BuildCitationsTool:
    """MCP tool for building citations from results.

    Allows Agent to:
    - Generate structured citations
    - Format citations in different styles
    - Link results to source documents
    """

    @property
    def name(self) -> str:
        return "build_citations"

    @property
    def description(self) -> str:
        return """Build structured citations from retrieval results.

Args:
    results: JSON string of retrieval results
    format: Citation format - "structured" (default), "markdown", "numbered"
    include_text: Whether to include text snippets (default: false)
    max_text_length: Maximum text snippet length (default: 200)

Returns:
    JSON or Markdown with formatted citations."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "string",
                    "description": "JSON string of retrieval results",
                },
                "format": {
                    "type": "string",
                    "enum": ["structured", "markdown", "numbered"],
                    "default": "structured",
                    "description": "Citation format to use",
                },
                "include_text": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to include text snippets",
                },
                "max_text_length": {
                    "type": "integer",
                    "default": 200,
                    "description": "Maximum text snippet length",
                },
            },
            "required": ["results"],
        }

    async def execute(
        self,
        results: str,
        format: str = "structured",
        include_text: bool = False,
        max_text_length: int = 200,
    ) -> "MCPToolResponse":
        """Execute the citation building tool.

        Args:
            results: JSON string of results
            format: Citation format
            include_text: Whether to include text
            max_text_length: Max text length

        Returns:
            MCPToolResponse with citations
        """
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
                {"citations": [], "total": 0},
                is_empty=True,
            )

        # Normalize results
        normalized = self._normalize_results(results_list)

        # Build citations
        citations = self._build_citations(normalized, include_text, max_text_length)

        # Format output
        if format == "structured":
            return build_json_response({
                "citations": citations,
                "total": len(citations),
                "format": format,
            })

        elif format == "markdown":
            markdown = self._format_markdown(citations)
            return build_structured_response(
                content=markdown,
                metadata={"format": "markdown", "total": len(citations)},
            )

        elif format == "numbered":
            numbered = self._format_numbered(citations)
            return build_json_response({
                "citations": numbered,
                "total": len(citations),
                "format": format,
            })

        else:
            return build_json_response({
                "citations": citations,
                "total": len(citations),
            })

    def _normalize_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Normalize results to standard dict format.

        Args:
            results: List of results (dict or object)

        Returns:
            Normalized list of dicts
        """
        normalized = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                normalized.append({
                    "chunk_id": r.get("chunk_id", f"chunk_{i}"),
                    "score": float(r.get("score", 0.0)),
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata", {}),
                })
            else:
                normalized.append({
                    "chunk_id": getattr(r, "chunk_id", f"chunk_{i}"),
                    "score": float(getattr(r, "score", 0.0)),
                    "text": getattr(r, "text", ""),
                    "metadata": getattr(r, "metadata", {}),
                })
        return normalized

    def _build_citations(
        self,
        results: List[Dict[str, Any]],
        include_text: bool,
        max_text_length: int,
    ) -> List[Dict[str, Any]]:
        """Build structured citations from results.

        Args:
            results: Normalized results
            include_text: Whether to include text
            max_text_length: Max text length

        Returns:
            List of citation dicts
        """
        citations = []

        for r in results:
            metadata = r.get("metadata", {})

            citation = {
                "id": r.get("chunk_id", ""),
                "relevance_score": r.get("score", 0.0),
                "source": metadata.get("source_path", metadata.get("source", "unknown")),
                "title": metadata.get("title", ""),
                "page": metadata.get("page_num", metadata.get("page")),
                "section": metadata.get("section", ""),
                "chunk_index": metadata.get("chunk_index"),
                "doc_id": metadata.get("doc_id", ""),
            }

            # Add text snippet if requested
            if include_text:
                text = r.get("text", "")
                if len(text) > max_text_length:
                    text = text[:max_text_length] + "..."
                citation["text_snippet"] = text

            # Add timestamp if available
            if "created_at" in metadata:
                citation["created_at"] = metadata["created_at"]

            # Clean up None values
            citation = {k: v for k, v in citation.items() if v is not None}

            citations.append(citation)

        return citations

    def _format_markdown(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations as Markdown.

        Args:
            citations: List of citation dicts

        Returns:
            Markdown formatted string
        """
        lines = ["## 引用来源\n\n"]

        for i, citation in enumerate(citations, 1):
            score = citation.get("relevance_score", 0.0)
            source = citation.get("source", "unknown")
            page = citation.get("page")
            title = citation.get("title", "")
            text = citation.get("text_snippet", "")

            lines.append(f"### [{i}] {source}")
            if title:
                lines.append(f"**标题:** {title}")
            if page:
                lines.append(f"**页码:** {page}")
            lines.append(f"**相关度:** {score:.2%}")

            if text:
                lines.append(f"\n> {text}\n")

            lines.append("\n")

        return "\n".join(lines)

    def _format_numbered(self, citations: List[Dict[str, Any]]) -> List[str]:
        """Format citations as numbered references.

        Args:
            citations: List of citation dicts

        Returns:
            List of numbered citation strings
        """
        numbered = []

        for i, citation in enumerate(citations, 1):
            source = citation.get("source", "unknown")
            page = citation.get("page")
            title = citation.get("title", "")

            parts = [f"[{i}]"]
            if title:
                parts.append(title)
            parts.append(source)
            if page:
                parts.append(f"p.{page}")

            numbered.append(" ".join(parts))

        return numbered


# MCP Tool Handler
async def build_citations_handler(
    results: str,
    format: str = "structured",
    include_text: bool = False,
    max_text_length: int = 200,
) -> "MCPToolResponse":
    """Handler for build_citations MCP tool."""
    tool = BuildCitationsTool()
    return await tool.execute(
        results=results,
        format=format,
        include_text=include_text,
        max_text_length=max_text_length,
    )


def register_tools(protocol_handler: Any) -> None:
    """Register citation tools with the protocol handler.

    Args:
        protocol_handler: The MCP protocol handler instance
    """
    tool = BuildCitationsTool()
    protocol_handler.register_tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        handler=build_citations_handler,
    )
    logger.info(f"Registered agentic tool: {tool.name}")