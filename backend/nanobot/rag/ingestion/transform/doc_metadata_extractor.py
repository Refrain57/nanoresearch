"""Document-level metadata extractor (Layer 3.5).

Extracts structured metadata (title, authors, abstract) from the leading portion
of a document using an LLM. Runs once per document before chunking so all
descendant chunks inherit the enriched metadata.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from nanobot.rag.core.types import Document
from nanobot.rag.observability.logger import get_logger

if TYPE_CHECKING:
    from nanobot.rag.core.settings import Settings

logger = get_logger(__name__)

_EXTRACTION_PROMPT = """\
Read the text below and extract the following fields if present:
- title: the document title (string)
- authors: list of author names (array of strings)
- abstract: the abstract or opening summary paragraph (string)

Return ONLY a JSON object with keys "title", "authors", "abstract".
Use null for any field that is not found. No explanation, no markdown fences.

Text:
{text}"""

_MAX_INPUT_CHARS = 3000


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class DocumentMetadataExtractor:
    """Extracts document-level metadata using a text LLM.

    Adds title, authors, and abstract to document.metadata so all chunks
    created from this document inherit them via DocumentChunker's metadata
    inheritance.

    Gracefully skips extraction when the LLM is unavailable.
    """

    def __init__(self, settings: "Settings") -> None:
        self._llm = None
        try:
            from nanobot.rag.libs.llm.llm_factory import LLMFactory
            self._llm = LLMFactory.create(settings)
            logger.info("DocumentMetadataExtractor: LLM loaded")
        except Exception as exc:
            logger.warning(f"DocumentMetadataExtractor: LLM unavailable ({exc}), skipping extraction")

    def extract(self, document: Document) -> Document:
        """Extract title/authors/abstract and merge into document.metadata.

        Only adds fields that are not already set and that the LLM returns
        as non-null. Safe to call even when LLM is unavailable (no-op).
        """
        if not self._llm:
            return document

        lead = document.text[:_MAX_INPUT_CHARS]
        prompt = _EXTRACTION_PROMPT.format(text=lead)

        try:
            from nanobot.rag.libs.llm.base_llm import Message
            response = self._llm.chat([Message(role="user", content=prompt)])
            content = _strip_fences(response.content)
            extracted: dict = json.loads(content)

            for key in ("title", "authors", "abstract"):
                value = extracted.get(key)
                if value is not None and key not in document.metadata:
                    document.metadata[key] = value

            title_preview = str(document.metadata.get("title", ""))[:60]
            logger.info(f"DocumentMetadataExtractor: extracted title='{title_preview}'")

        except Exception as exc:
            logger.warning(f"DocumentMetadataExtractor: extraction failed ({exc})")

        return document
