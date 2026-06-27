"""Document chunking module - adapts libs.splitter for business layer.

This module serves as the adapter layer between libs.splitter (pure text splitting)
and Ingestion Pipeline (business object transformation). It transforms Document
objects into Chunk objects with proper ID generation, metadata inheritance, and
traceability.

Core Value-Add (vs libs.splitter):
1. Chunk ID Generation: Deterministic and unique IDs for each chunk
2. Metadata Inheritance: Propagates Document metadata to all chunks
3. chunk_index: Records sequential position within document
4. source_ref: Establishes parent-child traceability
5. Type Conversion: str → Chunk object (core.types contract)

Design Principles:
- Adapter Pattern: Bridges text splitter tool with business objects
- Config-Driven: Uses SplitterFactory for configuration-based strategy selection
- Deterministic: Same Document produces same Chunk IDs on repeat splits
- Type-Safe: Enforces core.types.Chunk contract
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

from nanobot.rag.core.types import Chunk, Document
from nanobot.rag.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from nanobot.rag.core.settings import Settings


def detect_chunk_strategy(document: Document) -> str:
    """Automatically detect the best chunking strategy based on document structure.

    Detection order (most reliable first):
    1. PDF bookmarks/TOC (from PyMuPDF extraction)
    2. Markdown headings with hierarchical levels
    3. Numbered section patterns (1.1, 1.2, etc.)

    Args:
        document: Document to analyze

    Returns:
        "structured" if structure detected, "fixed" otherwise
    """
    text = document.text or ""

    # 1. Check PDF bookmarks (most reliable)
    if document.metadata.get("bookmarks"):
        bookmark_count = len(document.metadata["bookmarks"])
        if bookmark_count >= 3:
            logger.debug(f"Detected {bookmark_count} PDF bookmarks, using structured")
            return "structured"

    # 2. Check Markdown headings with hierarchical structure
    if document.metadata.get("headings"):
        headings = document.metadata["headings"]
        if len(headings) >= 3:
            levels = set()
            for h in headings:
                level = h.get("level", 1)
                levels.add(level)
            if len(levels) >= 2:
                logger.debug(f"Detected {len(headings)} hierarchical headings, using structured")
                return "structured"

    # 3. Check Markdown headings directly in text
    heading_pattern = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
    headings_in_text = heading_pattern.findall(text)
    if len(headings_in_text) >= 3:
        levels = set()
        for h in headings_in_text:
            level = len(h) - len(h.lstrip('#'))
            levels.add(level)
        if len(levels) >= 2:
            logger.debug(f"Detected {len(headings_in_text)} hierarchical headings in text, using structured")
            return "structured"

    # 4. Check numbered section patterns (1.1, 1.2, 2.3.1, etc.)
    section_pattern = re.compile(r'^\d+(\.\d+)+\s+\S+', re.MULTILINE)
    numbered_sections = section_pattern.findall(text)
    if len(numbered_sections) >= 5:
        logger.debug(f"Detected {len(numbered_sections)} numbered sections, using structured")
        return "structured"

    # Default: fixed-size chunking
    logger.debug("No clear structure detected, using fixed")
    return "fixed"


class DocumentChunker:
    """Converts Documents into Chunks with business-level enrichment.

    This class wraps a text splitter (from libs) and adds business logic:
    - Generates stable chunk IDs
    - Inherits and extends metadata
    - Maintains document traceability

    Supports two chunking strategies:
    - "fixed" / "recursive": Traditional fixed-size chunking
    - "document_based": Structure-aware chunking based on Markdown headings

    Attributes:
        _splitter: The underlying text splitter from libs layer
        _settings: Configuration settings for chunking behavior
        _structure_chunker: Optional DocumentStructureChunker for structure-aware mode

    Example:
        >>> from nanobot.rag.core.settings import load_settings
        >>> from nanobot.rag.core.types import Document
        >>> settings = load_settings("config/settings.yaml")
        >>> chunker = DocumentChunker(settings)
        >>> document = Document(
        ...     id="doc_123",
        ...     text="Long document content...",
        ...     metadata={"source_path": "data/report.pdf"}
        ... )
        >>> chunks = chunker.split_document(document)
        >>> print(f"Generated {len(chunks)} chunks")
        >>> print(f"First chunk ID: {chunks[0].id}")
        >>> print(f"First chunk index: {chunks[0].metadata['chunk_index']}")
    """

    def __init__(self, settings: Settings, auto_detect: bool = True, chunk_strategy_override: Optional[str] = None):
        """Initialize DocumentChunker with configuration.

        Args:
            settings: Configuration settings containing splitter configuration.
                     The splitter config is expected at settings.splitter.*
            auto_detect: If True, automatically detect best chunking strategy
                        based on document structure (default: True)
            chunk_strategy_override: Per-KB override ("auto", "fixed", "structured").
                        "auto" keeps auto-detect; others force a specific strategy.
                        "document_based" is accepted as a legacy alias for "structured".

        Raises:
            ValueError: If splitter configuration is invalid or provider unknown
        """
        self._settings = settings
        # Normalise legacy alias
        if chunk_strategy_override == "document_based":
            chunk_strategy_override = "structured"
        self._chunk_strategy_override = chunk_strategy_override if chunk_strategy_override and chunk_strategy_override != "auto" else None
        self._auto_detect = auto_detect if self._chunk_strategy_override is None else False
        self._structure_chunker: Optional[StructuredChunker] = None
        self._splitter = None

        # If auto_detect is disabled, use the configured strategy
        if not auto_detect:
            chunk_strategy = self._get_chunk_strategy()
            if chunk_strategy in ("structured", "document_based"):
                ingestion = settings.ingestion
                self._structure_chunker = StructuredChunker(
                    chunk_token_num=ingestion.chunk_size if ingestion else 512,
                    overlapped_percent=0,
                )
            else:
                self._splitter = SplitterFactory.create(settings)

    def _get_chunk_strategy(self) -> str:
        """Get chunk_strategy from settings."""
        if self._settings.ingestion:
            return self._settings.ingestion.chunk_strategy or "fixed"
        return "fixed"

    def _get_structure_chunker(self) -> "StructuredChunker":
        """Get or create StructuredChunker (lazy initialization)."""
        if self._structure_chunker is None:
            ingestion = self._settings.ingestion
            self._structure_chunker = StructuredChunker(
                chunk_token_num=ingestion.chunk_size if ingestion else 512,
                overlapped_percent=0,
            )
        return self._structure_chunker

    def _get_semantic_chunker(self):
        """Get or create SemanticChunker (lazy initialization)."""
        if not hasattr(self, "_semantic_chunker") or self._semantic_chunker is None:
            from nanobot.rag.ingestion.chunking.semantic_chunker import SemanticChunker
            self._semantic_chunker = SemanticChunker(self._settings)
        return self._semantic_chunker

    def _get_splitter(self):
        """Get or create text splitter (lazy initialization)."""
        if self._splitter is None:
            self._splitter = SplitterFactory.create(self._settings)
        return self._splitter

    def split_document(self, document: Document) -> List[Chunk]:
        """Split a Document into Chunks with full business enrichment.

        This is the main entry point that orchestrates the transformation:
        - If auto_detect: Detects best strategy based on document structure
        - If chunk_strategy == "document_based": Uses structure-aware chunking
        - Otherwise: Uses traditional splitter + business enrichment

        Args:
            document: Source document to split into chunks

        Returns:
            List of Chunk objects with:
            - Unique, deterministic IDs
            - Inherited metadata + chunk_index + source_ref
            - For document_based: structure metadata (title, section_level, etc.)

        Raises:
            ValueError: If document has no text or invalid structure
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")

        # Determine chunking strategy
        detected_strategy = None
        if self._chunk_strategy_override:
            detected_strategy = self._chunk_strategy_override
            logger.info(f"Using KB-level chunk strategy override: {detected_strategy}")
        elif self._auto_detect:
            detected_strategy = detect_chunk_strategy(document)
            logger.info(f"Auto-detected chunk strategy: {detected_strategy} for document {document.id}")
        else:
            detected_strategy = self._get_chunk_strategy()

        # Normalise legacy alias at runtime too (e.g. value stored in DB)
        if detected_strategy == "document_based":
            detected_strategy = "structured"

        # Use structure-aware chunker if detected/configured
        if detected_strategy == "semantic":
            text_fragments = self._get_semantic_chunker().split_text(document.text)
            if not text_fragments:
                logger.warning(f"SemanticChunker returned no chunks for {document.id}, falling back to splitter")
                text_fragments = self._get_splitter().split_text(document.text)
        elif detected_strategy == "structured":
            text_fragments = self._get_structure_chunker().split_text(document.text)
            if not text_fragments:
                logger.warning(f"StructuredChunker returned no chunks for {document.id}, falling back to fixed")
                text_fragments = self._get_splitter().split_text(document.text)
            chunks: List[Chunk] = []
            for index, text in enumerate(text_fragments):
                chunk_id = self._generate_chunk_id(document.id, index, text)
                chunk_metadata = self._inherit_metadata(document, index, text)
                chunk_metadata["chunk_strategy_used"] = "structured"
                # Light structural annotation: extract heading from first line if present
                import re as _re
                first_line = text.strip().split("\n")[0].strip()
                m = _re.match(r"^(#{1,6})\s+(.+)$", first_line)
                if m:
                    chunk_metadata["section_level"] = len(m.group(1))
                    chunk_metadata["title"] = m.group(2).strip()
                chunks.append(Chunk(id=chunk_id, text=text, metadata=chunk_metadata))
            # Link neighbours
            for i, chunk in enumerate(chunks):
                if i > 0:
                    chunk.metadata["prev_chunk_id"] = chunks[i - 1].id
                if i < len(chunks) - 1:
                    chunk.metadata["next_chunk_id"] = chunks[i + 1].id
            return chunks

        # Traditional flow: Use underlying splitter to get text fragments
        text_fragments = self._get_splitter().split_text(document.text)

        if not text_fragments:
            raise ValueError(
                f"Splitter returned no chunks for document {document.id}. "
                f"Text length: {len(document.text)}"
            )

        # Step 2: Transform text fragments into Chunk objects with enrichment
        chunks: List[Chunk] = []
        for index, text in enumerate(text_fragments):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            chunk_metadata["chunk_strategy_used"] = detected_strategy

            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate unique and deterministic chunk ID.
        
        ID format: {doc_id}_{index:04d}_{content_hash}
        - doc_id: Parent document identifier
        - index: Sequential position (zero-padded to 4 digits)
        - content_hash: First 8 chars of text SHA256 hash
        
        This ensures:
        - Uniqueness: Combination of doc_id + index + content_hash
        - Determinism: Same input always produces same ID
        - Debuggability: Human-readable structure
        
        Args:
            doc_id: Parent document ID
            index: Sequential position of chunk (0-based)
            text: Chunk text content
        
        Returns:
            Unique chunk ID string
        
        Example:
            >>> chunker._generate_chunk_id("doc_123", 0, "Hello world")
            'doc_123_0000_c0535e4b'
        """
        # Compute content hash for uniqueness
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        
        # Format: {doc_id}_{index:04d}_{hash_8chars}
        return f"{doc_id}_{index:04d}_{content_hash}"
    
    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str = "") -> dict:
        """Inherit metadata from document and add chunk-specific fields.
        
        This creates a new metadata dict containing:
        - All fields from document.metadata (copied, not referenced)
        - chunk_index: Sequential position (0-based)
        - source_ref: Reference to parent document ID
        - image_refs: List of image IDs referenced in this chunk (extracted from placeholders)
        
        Note: The document-level 'images' field is intentionally excluded from chunk
        metadata as it would be redundant. Instead, chunk-specific 'image_refs' is
        populated based on [IMAGE: xxx] placeholders found in the chunk text.
        
        Args:
            document: Source document whose metadata to inherit
            chunk_index: Sequential position of this chunk
            chunk_text: The text content of this chunk (used to extract image_refs)
        
        Returns:
            Metadata dict with inherited and chunk-specific fields
        
        Example:
            >>> doc = Document(
            ...     id="doc_123",
            ...     text="Content",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> metadata = chunker._inherit_metadata(doc, 2, "See [IMAGE: img_001]")
            >>> metadata["source_path"]
            'file.pdf'
            >>> metadata["chunk_index"]
            2
            >>> metadata["source_ref"]
            'doc_123'
            >>> metadata["image_refs"]
            ['img_001']
        """
        import re
        
        # Copy all document metadata (shallow copy is sufficient for primitives)
        chunk_metadata = document.metadata.copy()
        
        # Get document-level images for lookup
        doc_images = document.metadata.get("images", [])
        
        # Remove document-level 'images' field - we'll add chunk-specific images below
        chunk_metadata.pop("images", None)
        
        # Add chunk-specific fields
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id
        
        # Extract image_refs from chunk text by finding [IMAGE: xxx] placeholders
        image_refs = []
        if chunk_text:
            # Pattern matches [IMAGE: image_id] placeholders
            pattern = r'\[IMAGE:\s*([^\]]+)\]'
            matches = re.findall(pattern, chunk_text)
            image_refs = [m.strip() for m in matches]
        
        chunk_metadata["image_refs"] = image_refs
        
        # Build chunk-specific 'images' list with full metadata for referenced images
        # This is needed by ImageCaptioner to access image paths for Vision API calls
        chunk_images = []
        if image_refs and doc_images:
            image_lookup = {img.get("id"): img for img in doc_images}
            for img_id in image_refs:
                if img_id in image_lookup:
                    chunk_images.append(image_lookup[img_id])
        
        if chunk_images:
            chunk_metadata["images"] = chunk_images
        
        # Try to determine page_num from the first referenced image
        if chunk_images:
            chunk_metadata["page_num"] = chunk_images[0].get("page")

        return chunk_metadata


# ============================================================================
# Structured Chunker — RAGFlow-like hierarchical merging
# ============================================================================


class StructuredChunker:
    """Structure-aware chunker using RAGFlow-like hierarchical merging.

    Replaces the old DocumentStructureChunker. Key improvements:
    - False-positive heading filtering (length / punctuation heuristics)
    - Hierarchical tree merge: small sibling sections are merged
    - Token-based sizing (vs character-based)
    - TOC detection and removal
    - Fallback to naive merge when no structure detected

    Interface mirrors BaseSplitter.split_text() so it can be used inline
    within DocumentChunker.split_document().
    """

    def __init__(self, chunk_token_num: int = 512, overlapped_percent: int = 0, **_: object) -> None:
        self.chunk_token_num = chunk_token_num
        self.overlapped_percent = overlapped_percent

    def split_text(self, text: str) -> list[str]:
        from nanobot.rag.libs.splitter.ragflow.book import chunk_markdown
        return chunk_markdown(
            text,
            parser_config={
                "chunk_token_num": self.chunk_token_num,
                "overlapped_percent": self.overlapped_percent,
            },
        )


# Keep old name as alias so splitter_factory import doesn't break immediately
DocumentStructureChunker = StructuredChunker
