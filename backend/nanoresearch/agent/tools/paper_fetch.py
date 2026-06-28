"""Paper fetch tool — download arxiv/paper PDFs to workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanoresearch.agent.tools.base import Tool

_USER_AGENT = "Mozilla/5.0 (compatible; NanoResearch/1.0)"
_TIMEOUT = 120
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB hard cap


class PaperFetchTool(Tool):
    """Download a paper PDF from a URL and save it to workspace/papers/.

    Intended for arxiv and other academic paper PDFs where WebFetchTool's
    text-extraction mode is not suitable for binary content.
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "fetch_paper"

    @property
    def description(self) -> str:
        return (
            "Download a paper PDF from a URL and save it locally for ingestion.\n"
            "Use this before calling ingest_document to get the PDF file.\n"
            "Returns the absolute local path on success, or an Error: message on failure.\n\n"
            "arxiv PDF URL format: https://arxiv.org/pdf/{arxiv_id}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Direct URL of the paper PDF (http/https only)",
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Filename to save as, e.g. '2506.12345.pdf'. "
                        "Saved under workspace/papers/"
                    ),
                },
            },
            "required": ["url", "filename"],
        }

    async def execute(self, url: str, filename: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "Error: only http/https URLs are supported"

        safe_name = Path(filename).name
        if not safe_name:
            return "Error: invalid filename"

        dest = self._workspace / "papers" / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info("PaperFetchTool: {} -> {}", url, dest)

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=_TIMEOUT,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    data = b""
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        data += chunk
                        if len(data) > _MAX_BYTES:
                            return f"Error: file exceeds {_MAX_BYTES // (1024 * 1024)} MB limit"

            dest.write_bytes(data)
            logger.info("PaperFetchTool: saved {} bytes to {}", len(data), dest)
            return str(dest)

        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} for {url}"
        except httpx.TimeoutException:
            return f"Error: timed out after {_TIMEOUT}s"
        except Exception as e:
            logger.exception("PaperFetchTool: unexpected error")
            return f"Error: {type(e).__name__}: {e}"
