"""Markdown Loader implementation.

Loads Markdown files and converts them to standardized Document format
with structure metadata.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from nanobot.rag.core.types import Document
from nanobot.rag.libs.loader.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """Markdown Loader for .md files.

    This loader:
    1. Reads Markdown file content
    2. Extracts title from first H1 heading or filename
    3. Extracts structure metadata (headings, code blocks count, etc.)
    4. Preserves Markdown structure for structure-aware chunking

    Note: Markdown files already have heading structure, which is ideal for
    structure-aware chunking (document_based strategy).
    """

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a Markdown file.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            Document with Markdown text and metadata.

        Raises:
            FileNotFoundError: If the Markdown file doesn't exist.
            ValueError: If the file is not a valid text file.
        """
        # Validate file
        path = self._validate_file(file_path)
        if path.suffix.lower() not in ('.md', '.markdown', '.mdown'):
            raise ValueError(f"File is not Markdown: {path}")

        # Read content
        try:
            text_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to other encodings
            try:
                text_content = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text_content = path.read_text(encoding="latin-1")

        if not text_content.strip():
            raise ValueError(f"Empty Markdown file: {path}")

        # Compute document hash for unique ID
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        # Initialize metadata
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "markdown",
            "doc_hash": doc_hash,
            "file_name": path.name,
        }

        # Extract title from first H1 heading
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title

        # Extract structure info for chunking
        headings = self._extract_headings(text_content)
        if headings:
            metadata["headings"] = headings
            metadata["heading_count"] = len(headings)

        # Count code blocks and tables
        code_blocks = len(re.findall(r'```[\s\S]*?```', text_content))
        tables = len(re.findall(r'\|[^\n]+\|\n', text_content))
        metadata["code_block_count"] = code_blocks
        metadata["table_count"] = tables

        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content.

        Args:
            file_path: Path to file.

        Returns:
            Hex string of SHA256 hash.
        """
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown H1 heading.

        Args:
            text: Markdown text content.

        Returns:
            Title string if found, None otherwise.
        """
        lines = text.split('\n')

        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        return None

    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """Extract all headings with their levels and positions.

        Args:
            text: Markdown text content.

        Returns:
            List of heading dicts with level, title, and line number.
        """
        headings = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if line.startswith('#'):
                # Parse heading level and title
                match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    headings.append({
                        "level": level,
                        "title": title,
                        "line": line_num,
                    })

        return headings
