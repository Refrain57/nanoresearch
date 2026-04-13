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
from typing import TYPE_CHECKING, List, Optional

from nanobot.rag.core.types import Chunk, Document
from nanobot.rag.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from nanobot.rag.core.settings import Settings


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

    def __init__(self, settings: Settings):
        """Initialize DocumentChunker with configuration.

        Args:
            settings: Configuration settings containing splitter configuration.
                     The splitter config is expected at settings.splitter.*

        Raises:
            ValueError: If splitter configuration is invalid or provider unknown
        """
        self._settings = settings
        self._structure_chunker: Optional[DocumentStructureChunker] = None

        # Check chunk_strategy to determine which chunker to use
        chunk_strategy = self._get_chunk_strategy()

        if chunk_strategy == "document_based":
            # Use structure-aware chunker directly
            ingestion = settings.ingestion
            min_chunk = ingestion.min_chunk_length if ingestion else 100
            max_chunk = ingestion.max_chunk_length if ingestion else 2000
            self._structure_chunker = DocumentStructureChunker(
                min_chunk_length=min_chunk,
                max_chunk_length=max_chunk,
            )
            self._splitter = None  # Not used for document_based strategy
        else:
            # Use traditional splitter via factory
            self._splitter = SplitterFactory.create(settings)

    def _get_chunk_strategy(self) -> str:
        """Get chunk_strategy from settings."""
        if self._settings.ingestion:
            return self._settings.ingestion.chunk_strategy or "fixed"
        return "fixed"

    def split_document(self, document: Document) -> List[Chunk]:
        """Split a Document into Chunks with full business enrichment.

        This is the main entry point that orchestrates the transformation:
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

        # Use structure-aware chunker if configured
        if self._structure_chunker is not None:
            chunks = self._structure_chunker.split_document(document)
            # Ensure inherited metadata from document
            for chunk in chunks:
                # Merge document metadata (excluding 'images' which is chunk-specific)
                doc_meta = document.metadata.copy()
                doc_meta.pop("images", None)
                # Merge with structure metadata (structure metadata takes precedence)
                chunk.metadata = {**doc_meta, **chunk.metadata}
                chunk.metadata["source_ref"] = document.id
            return chunks

        # Traditional flow: Use underlying splitter to get text fragments
        text_fragments = self._splitter.split_text(document.text)

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
# Document Structure Chunker - Structure-Aware Chunking
# ============================================================================

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ContentType(Enum):
    """Content type enumeration for structure-aware chunking."""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"


@dataclass
class Block:
    """Represents a content block with type and boundaries.

    Used for identifying special blocks (code, table, list) that should
    not be split across chunks.
    """
    type: ContentType
    content: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Section:
    """Represents a document section with heading and content.

    Used for structure-aware chunking based on Markdown headings.
    """
    level: int  # 1-6 for # to ######
    title: str
    content: str
    start: int
    end: int
    path: str = ""  # e.g., "/Chapter 1/Section 1.1"
    parent_title: str = ""


class DocumentStructureChunker:
    """Structure-aware document chunker that preserves document semantics.

    This chunker leverages Markdown document structure (headings, code blocks,
    tables, lists) to create semantically coherent chunks with rich metadata.

    Key Features:
    1. Heading-based splitting: Chunks align with document sections
    2. Special block protection: Code blocks, tables, lists are kept intact
    3. Rich metadata: title, section_level, section_path, content_type
    4. Neighbor linking: prev_chunk_id, next_chunk_id for context retrieval

    Design Principles:
    - Semantic Coherence: Chunks represent complete semantic units
    - Structure Preservation: Document hierarchy is maintained in metadata
    - Retrieval Optimization: Metadata enables structure-aware retrieval
    - Fallback Strategy: Falls back to recursive splitting for oversized sections

    Example:
        >>> from nanobot.rag.core.types import Document
        >>> doc = Document(
        ...     id="doc_001",
        ...     text="# Chapter 1\\n\\n## Section 1.1\\n\\nContent here...",
        ...     metadata={"source_path": "doc.md"}
        ... )
        >>> chunker = DocumentStructureChunker(min_chunk_length=100, max_chunk_length=2000)
        >>> chunks = chunker.split_document(doc)
        >>> chunks[0].metadata["title"]
        'Chapter 1'
        >>> chunks[0].metadata["section_level"]
        1
    """

    # Regex patterns for special blocks
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    TABLE_PATTERN = re.compile(r'(?:\|[^\n]+\|\n)+', re.MULTILINE)
    LIST_PATTERN = re.compile(r'(?:^[ \t]*[-*+][ \t]+.+\n)+', re.MULTILINE)
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(
        self,
        min_chunk_length: int = 100,
        max_chunk_length: int = 2000,
        fallback_splitter: Optional[Any] = None,
    ):
        """Initialize DocumentStructureChunker.

        Args:
            min_chunk_length: Minimum characters per chunk. Smaller sections
                              will be merged with siblings if possible.
            max_chunk_length: Maximum characters per chunk. Larger sections
                              will be split using fallback splitter.
            fallback_splitter: Optional splitter for oversized chunks.
                               If None, uses RecursiveCharacterTextSplitter.
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self._fallback_splitter = fallback_splitter

        # Lazy load fallback splitter
        self._fallback = None

    def _get_fallback_splitter(self):
        """Get or create fallback splitter for oversized chunks."""
        if self._fallback is None:
            if self._fallback_splitter is not None:
                self._fallback = self._fallback_splitter
            else:
                # Use recursive splitter as fallback
                try:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    self._fallback = RecursiveCharacterTextSplitter(
                        chunk_size=self.max_chunk_length,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", ". ", " ", ""],
                    )
                except ImportError:
                    self._fallback = None
        return self._fallback

    def split_document(self, document: Document) -> List[Chunk]:
        """Split a Document into structure-aware Chunks.

        This method:
        1. Identifies special blocks (code, tables, lists)
        2. Parses document structure (headings)
        3. Creates chunks aligned with sections
        4. Protects special blocks from being split
        5. Adds neighbor links for context retrieval

        Args:
            document: Source document to split.

        Returns:
            List of Chunk objects with rich metadata including:
            - title: Section heading
            - section_level: Heading level (1-6)
            - section_path: Full path like "/Chapter/Section"
            - content_type: text, code, table, or list
            - prev_chunk_id / next_chunk_id: Neighbor links
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")

        # Step 1: Identify special blocks
        blocks = self._identify_blocks(document.text)

        # Step 2: Parse document structure
        sections = self._parse_sections(document.text, blocks)

        # Step 3: Create chunks from sections
        chunks = self._create_chunks(document, sections, blocks)

        # Step 4: Link neighbors
        chunks = self._link_neighbors(chunks)

        return chunks

    def _identify_blocks(self, text: str) -> List[Block]:
        """Identify special blocks that should not be split.

        Args:
            text: Document text.

        Returns:
            List of Block objects with type and boundaries.
        """
        blocks = []
        occupied_ranges = set()  # Track character positions already assigned

        # 1. Identify code blocks
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            blocks.append(Block(
                type=ContentType.CODE,
                content=match.group(),
                start=match.start(),
                end=match.end(),
                metadata={"language": self._extract_code_language(match.group())},
            ))
            occupied_ranges.update(range(match.start(), match.end()))

        # 2. Identify tables
        for match in self.TABLE_PATTERN.finditer(text):
            # Skip if overlaps with code block
            if any(pos in occupied_ranges for pos in range(match.start(), match.end())):
                continue
            blocks.append(Block(
                type=ContentType.TABLE,
                content=match.group(),
                start=match.start(),
                end=match.end(),
            ))
            occupied_ranges.update(range(match.start(), match.end()))

        # 3. Identify lists
        for match in self.LIST_PATTERN.finditer(text):
            # Skip if overlaps with existing blocks
            if any(pos in occupied_ranges for pos in range(match.start(), match.end())):
                continue
            blocks.append(Block(
                type=ContentType.LIST,
                content=match.group(),
                start=match.start(),
                end=match.end(),
            ))
            occupied_ranges.update(range(match.start(), match.end()))

        # Sort by position
        blocks.sort(key=lambda b: b.start)

        return blocks

    def _extract_code_language(self, code_block: str) -> str:
        """Extract language from code block opening."""
        first_line = code_block.split('\n')[0]
        if first_line.startswith('```'):
            lang = first_line[3:].strip()
            return lang if lang else "unknown"
        return "unknown"

    def _parse_sections(self, text: str, blocks: List[Block]) -> List[Section]:
        """Parse document into sections based on headings.

        Args:
            text: Document text.
            blocks: Special blocks identified earlier.

        Returns:
            List of Section objects with heading info.
        """
        sections = []
        heading_stack = []  # Track current heading hierarchy

        # Find all headings
        heading_matches = list(self.HEADING_PATTERN.finditer(text))

        if not heading_matches:
            # No headings - treat entire document as one section
            return [Section(
                level=0,
                title="",
                content=text,
                start=0,
                end=len(text),
                path="",
                parent_title="",
            )]

        # Create sections between headings
        for i, match in enumerate(heading_matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()

            # End at next heading or document end
            end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)

            # Update heading stack
            heading_stack = heading_stack[:level - 1] + [title]
            path = "/" + "/".join(heading_stack)
            parent_title = heading_stack[-2] if len(heading_stack) > 1 else ""

            # Content starts after heading line
            content_start = match.end()
            content = text[content_start:end].strip()

            sections.append(Section(
                level=level,
                title=title,
                content=content,
                start=start,
                end=end,
                path=path,
                parent_title=parent_title,
            ))

        return sections

    def _create_chunks(
        self,
        document: Document,
        sections: List[Section],
        blocks: List[Block],
    ) -> List[Chunk]:
        """Create Chunk objects from sections.

        Args:
            document: Source document.
            sections: Parsed sections.
            blocks: Special blocks to protect.

        Returns:
            List of Chunk objects.
        """
        chunks = []
        chunk_index = 0

        for section in sections:
            # Check if section contains special blocks
            section_blocks = [
                b for b in blocks
                if b.start >= section.start and b.end <= section.end
            ]

            if section.content:
                # Create chunks from section content
                section_chunks = self._split_section(
                    document=document,
                    section=section,
                    blocks=section_blocks,
                    chunk_index=chunk_index,
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)

        return chunks

    def _split_section(
        self,
        document: Document,
        section: Section,
        blocks: List[Block],
        chunk_index: int,
    ) -> List[Chunk]:
        """Split a section into chunks, protecting special blocks.

        Args:
            document: Source document.
            section: Section to split.
            blocks: Special blocks in this section.
            chunk_index: Starting chunk index.

        Returns:
            List of Chunk objects for this section.
        """
        chunks = []
        content = section.content

        # Check if section is small enough to be one chunk
        if len(content) <= self.max_chunk_length:
            chunk = self._create_chunk(
                document=document,
                content=content,
                section=section,
                chunk_index=chunk_index,
                content_type=self._determine_content_type(content, blocks),
            )
            return [chunk]

        # Section is too large - need to split
        # Protect special blocks and split remaining text
        protected_ranges = [(b.start, b.end, b) for b in blocks]

        # Split content while respecting protected blocks
        # For simplicity, use fallback splitter with protection
        sub_chunks = self._protected_split(
            document=document,
            content=content,
            section=section,
            blocks=blocks,
            chunk_index=chunk_index,
        )

        return sub_chunks

    def _protected_split(
        self,
        document: Document,
        content: str,
        section: Section,
        blocks: List[Block],
        chunk_index: int,
    ) -> List[Chunk]:
        """Split content while protecting special blocks.

        Strategy:
        1. Extract special blocks as their own chunks
        2. Split remaining text using fallback splitter
        3. Merge small chunks with neighbors if needed

        Args:
            document: Source document.
            content: Content to split.
            section: Parent section.
            blocks: Special blocks to protect.
            chunk_index: Starting chunk index.

        Returns:
            List of protected chunks.
        """
        chunks = []
        current_index = chunk_index

        # Sort blocks by position in content
        sorted_blocks = sorted(blocks, key=lambda b: content.find(b.content) if b.content in content else len(content))

        # Split content around special blocks
        parts = []
        last_end = 0

        for block in sorted_blocks:
            # Find block in content
            block_start = content.find(block.content)
            if block_start == -1:
                continue

            # Text before block
            if block_start > last_end:
                parts.append((content[last_end:block_start], ContentType.TEXT))

            # Block itself
            parts.append((block.content, block.type))
            last_end = block_start + len(block.content)

        # Remaining text after last block
        if last_end < len(content):
            parts.append((content[last_end:], ContentType.TEXT))

        # If no blocks, just split the whole content
        if not sorted_blocks:
            parts = [(content, ContentType.TEXT)]

        # Create chunks from parts
        for part_content, part_type in parts:
            if not part_content.strip():
                continue

            if len(part_content) <= self.max_chunk_length:
                # Part fits in one chunk
                chunk = self._create_chunk(
                    document=document,
                    content=part_content,
                    section=section,
                    chunk_index=current_index,
                    content_type=part_type,
                )
                chunks.append(chunk)
                current_index += 1
            else:
                # Part too large - use fallback splitter
                if part_type == ContentType.CODE or part_type == ContentType.TABLE:
                    # Code and tables should NOT be split - keep as is with warning
                    import logging
                    logging.warning(
                        f"Large {part_type.value} block ({len(part_content)} chars) "
                        f"exceeds max_chunk_length but kept intact to preserve structure"
                    )
                    chunk = self._create_chunk(
                        document=document,
                        content=part_content,
                        section=section,
                        chunk_index=current_index,
                        content_type=part_type,
                    )
                    chunks.append(chunk)
                    current_index += 1
                else:
                    # Use fallback splitter for text
                    fallback = self._get_fallback_splitter()
                    if fallback:
                        sub_texts = fallback.split_text(part_content)
                        for sub_text in sub_texts:
                            if sub_text.strip():
                                chunk = self._create_chunk(
                                    document=document,
                                    content=sub_text,
                                    section=section,
                                    chunk_index=current_index,
                                    content_type=ContentType.TEXT,
                                )
                                chunks.append(chunk)
                                current_index += 1
                    else:
                        # No fallback - keep as is
                        chunk = self._create_chunk(
                            document=document,
                            content=part_content,
                            section=section,
                            chunk_index=current_index,
                            content_type=part_type,
                        )
                        chunks.append(chunk)
                        current_index += 1

        return chunks

    def _create_chunk(
        self,
        document: Document,
        content: str,
        section: Section,
        chunk_index: int,
        content_type: ContentType,
    ) -> Chunk:
        """Create a Chunk object with rich metadata.

        Args:
            document: Source document.
            content: Chunk text content.
            section: Parent section.
            chunk_index: Index in the chunk sequence.
            content_type: Type of content (text, code, table, list).

        Returns:
            Chunk with structure-aware metadata.
        """
        # Generate chunk ID
        chunk_id = self._generate_chunk_id(document.id, chunk_index, content)

        # Build metadata
        metadata = document.metadata.copy()

        # Structure metadata
        metadata["title"] = section.title
        metadata["section_level"] = section.level
        metadata["section_path"] = section.path
        metadata["parent_section"] = section.parent_title

        # Content metadata
        metadata["content_type"] = content_type.value

        # Position metadata
        metadata["chunk_index"] = chunk_index
        metadata["source_ref"] = document.id

        # Neighbor links (will be filled later)
        metadata["prev_chunk_id"] = None
        metadata["next_chunk_id"] = None

        # Remove document-level images field (too large for each chunk)
        metadata.pop("images", None)

        return Chunk(
            id=chunk_id,
            text=content,
            metadata=metadata,
        )

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate unique chunk ID.

        Format: {doc_id}_{index:04d}_{content_hash}
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    def _determine_content_type(self, content: str, blocks: List[Block]) -> ContentType:
        """Determine the primary content type of a chunk.

        Args:
            content: Chunk content.
            blocks: Special blocks in this section.

        Returns:
            Primary ContentType.
        """
        if not blocks:
            return ContentType.TEXT

        # Check if content is primarily a special block
        for block in blocks:
            if block.content == content:
                return block.type

        return ContentType.TEXT

    def _link_neighbors(self, chunks: List[Chunk]) -> List[Chunk]:
        """Link adjacent chunks with prev/next IDs.

        Args:
            chunks: List of chunks to link.

        Returns:
            Same list with neighbor IDs populated.
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata["prev_chunk_id"] = chunks[i - 1].id
            if i < len(chunks) - 1:
                chunk.metadata["next_chunk_id"] = chunks[i + 1].id

        return chunks
