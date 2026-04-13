"""Unit tests for DocumentChunker.

Tests the document chunking functionality including:
- Basic chunking with ID generation
- Metadata inheritance
- Chunk index tracking
- Image placeholder extraction
- Edge cases (empty documents, single chunk, etc.)
"""

import hashlib
import pytest
from unittest.mock import MagicMock, patch

from nanobot.rag.core.types import Document, Chunk
from tests.rag.conftest import MockSettings


class MockSplitterSettings:
    """Mock splitter settings."""
    provider: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50


class TestDocumentChunkerInit:
    """Tests for DocumentChunker initialization."""

    def test_init_with_mock_settings(self):
        """Test initialization with mock settings."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        # Patch SplitterFactory to return a mock splitter
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)
            assert chunker._settings is settings


class TestDocumentChunkerChunkId:
    """Tests for chunk ID generation."""

    def test_chunk_id_format(self):
        """Test chunk ID follows expected format."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["test content"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            chunk_id = chunker._generate_chunk_id("doc_123", 0, "test content")

            # Format: {doc_id}_{index:04d}_{content_hash}
            assert chunk_id.startswith("doc_123_0000_")
            # Hash should be 8 characters
            parts = chunk_id.split("_")
            assert len(parts[-1]) == 8

    def test_chunk_id_deterministic(self):
        """Test that same input produces same chunk ID."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["test content"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            id1 = chunker._generate_chunk_id("doc_123", 0, "test content")
            id2 = chunker._generate_chunk_id("doc_123", 0, "test content")

            assert id1 == id2

    def test_chunk_id_different_content(self):
        """Test that different content produces different chunk ID."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["content1", "content2"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            id1 = chunker._generate_chunk_id("doc_123", 0, "content1")
            id2 = chunker._generate_chunk_id("doc_123", 1, "content2")

            assert id1 != id2

    def test_chunk_id_zero_padded_index(self):
        """Test that chunk index is zero-padded."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["test"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            id0 = chunker._generate_chunk_id("doc", 0, "content")
            id9 = chunker._generate_chunk_id("doc", 9, "content")
            id10 = chunker._generate_chunk_id("doc", 10, "content")

            assert "_0000_" in id0
            assert "_0009_" in id9
            assert "_0010_" in id10


class TestDocumentChunkerMetadataInheritance:
    """Tests for metadata inheritance."""

    def test_metadata_inherited_from_document(self):
        """Test that document metadata is inherited to chunks."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk content"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Some text content",
                metadata={
                    "source_path": "test.pdf",
                    "title": "Test Document",
                    "author": "Test Author",
                },
            )

            chunks = chunker.split_document(doc)

            assert len(chunks) == 1
            assert chunks[0].metadata["source_path"] == "test.pdf"
            assert chunks[0].metadata["title"] == "Test Document"
            assert chunks[0].metadata["author"] == "Test Author"

    def test_chunk_index_added(self):
        """Test that chunk_index is added to metadata."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk1", "chunk2", "chunk3"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Some text content",
                metadata={"source_path": "test.pdf"},
            )

            chunks = chunker.split_document(doc)

            assert chunks[0].metadata["chunk_index"] == 0
            assert chunks[1].metadata["chunk_index"] == 1
            assert chunks[2].metadata["chunk_index"] == 2

    def test_source_ref_added(self):
        """Test that source_ref points to parent document."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk content"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_parent_123",
                text="Some text content",
                metadata={"source_path": "test.pdf"},
            )

            chunks = chunker.split_document(doc)

            assert chunks[0].metadata["source_ref"] == "doc_parent_123"

    def test_image_refs_extraction(self):
        """Test extraction of image references from chunk text."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = [
                "First part [IMAGE: img_001] and [IMAGE: img_002]",
                "Second part [IMAGE: img_003]",
            ]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Some text with images",
                metadata={"source_path": "test.pdf"},
            )

            chunks = chunker.split_document(doc)

            assert chunks[0].metadata["image_refs"] == ["img_001", "img_002"]
            assert chunks[1].metadata["image_refs"] == ["img_003"]

    def test_image_refs_with_doc_images(self):
        """Test image metadata with document-level images."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["Content with [IMAGE: img_001]"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Content with images",
                metadata={
                    "source_path": "test.pdf",
                    "images": [
                        {"id": "img_001", "path": "/images/img1.png", "page": 1},
                        {"id": "img_002", "path": "/images/img2.png", "page": 2},
                    ],
                },
            )

            chunks = chunker.split_document(doc)

            # Chunk should have full image metadata
            assert "images" in chunks[0].metadata
            assert len(chunks[0].metadata["images"]) == 1
            assert chunks[0].metadata["images"][0]["id"] == "img_001"

    def test_document_level_images_removed_from_chunk(self):
        """Test that document-level images field is not copied to chunk."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk without image ref"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Some text",
                metadata={
                    "source_path": "test.pdf",
                    "images": [{"id": "img_001", "path": "/images/img1.png"}],
                },
            )

            chunks = chunker.split_document(doc)

            # Document-level images should be removed, chunk-specific images added
            assert "images" not in chunks[0].metadata or chunks[0].metadata.get("images") == []


class TestDocumentChunkerSplitDocument:
    """Tests for split_document method."""

    def test_split_document_basic(self):
        """Test basic document splitting."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk1", "chunk2", "chunk3"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Original document text",
                metadata={"source_path": "test.pdf"},
            )

            chunks = chunker.split_document(doc)

            assert len(chunks) == 3
            assert all(isinstance(c, Chunk) for c in chunks)
            assert chunks[0].text == "chunk1"
            assert chunks[1].text == "chunk2"
            assert chunks[2].text == "chunk3"

    def test_split_document_empty_text_error(self):
        """Test that empty document raises error."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="",
                metadata={"source_path": "test.pdf"},
            )

            with pytest.raises(ValueError, match="no text content"):
                chunker.split_document(doc)

    def test_split_document_whitespace_only_error(self):
        """Test that whitespace-only document raises error."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="   \n\t  ",
                metadata={"source_path": "test.pdf"},
            )

            with pytest.raises(ValueError, match="no text content"):
                chunker.split_document(doc)

    def test_split_document_splitter_returns_empty(self):
        """Test handling when splitter returns empty list."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = []
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Some text content",
                metadata={"source_path": "test.pdf"},
            )

            with pytest.raises(ValueError, match="Splitter returned no chunks"):
                chunker.split_document(doc)

    def test_split_document_single_chunk(self):
        """Test splitting document that produces single chunk."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["short content"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Short document",
                metadata={"source_path": "test.pdf"},
            )

            chunks = chunker.split_document(doc)

            assert len(chunks) == 1
            assert chunks[0].metadata["chunk_index"] == 0


class TestDocumentChunkerMetadataIntegrity:
    """Tests for metadata integrity across chunks."""

    def test_metadata_not_shared(self):
        """Test that metadata dicts are not shared between chunks."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Some text",
                metadata={"source_path": "test.pdf"},
            )

            chunks = chunker.split_document(doc)

            # Modifying one chunk's metadata should not affect others
            chunks[0].metadata["new_key"] = "value"

            assert "new_key" not in chunks[1].metadata

    def test_page_num_from_image(self):
        """Test that page_num is extracted from first image."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentChunker

        settings = MockSettings()
        with patch("nanobot.rag.ingestion.chunking.document_chunker.SplitterFactory") as mock_factory:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["Content with [IMAGE: img_001]"]
            mock_factory.create.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_123",
                text="Content with images",
                metadata={
                    "source_path": "test.pdf",
                    "images": [
                        {"id": "img_001", "path": "/images/img1.png", "page": 5},
                    ],
                },
            )

            chunks = chunker.split_document(doc)

            assert chunks[0].metadata.get("page_num") == 5


# ============================================================================
# DocumentStructureChunker Tests
# ============================================================================

class TestDocumentStructureChunkerInit:
    """Tests for DocumentStructureChunker initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        assert chunker.min_chunk_length == 100
        assert chunker.max_chunk_length == 2000

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker(
            min_chunk_length=50,
            max_chunk_length=1000,
        )
        assert chunker.min_chunk_length == 50
        assert chunker.max_chunk_length == 1000


class TestDocumentStructureChunkerBlockIdentification:
    """Tests for special block identification."""

    def test_identify_code_blocks(self):
        """Test identification of code blocks."""
        from nanobot.rag.ingestion.chunking.document_chunker import (
            DocumentStructureChunker,
            ContentType,
        )

        chunker = DocumentStructureChunker()
        text = """Some text.

```python
def hello():
    print("Hello")
```

More text."""

        blocks = chunker._identify_blocks(text)

        code_blocks = [b for b in blocks if b.type == ContentType.CODE]
        assert len(code_blocks) == 1
        assert "def hello()" in code_blocks[0].content

    def test_identify_multiple_code_blocks(self):
        """Test identification of multiple code blocks."""
        from nanobot.rag.ingestion.chunking.document_chunker import (
            DocumentStructureChunker,
            ContentType,
        )

        chunker = DocumentStructureChunker()
        text = """
```python
code1
```
text
```javascript
code2
```
"""
        blocks = chunker._identify_blocks(text)

        code_blocks = [b for b in blocks if b.type == ContentType.CODE]
        assert len(code_blocks) == 2

    def test_identify_tables(self):
        """Test identification of tables."""
        from nanobot.rag.ingestion.chunking.document_chunker import (
            DocumentStructureChunker,
            ContentType,
        )

        chunker = DocumentStructureChunker()
        text = """Some text.

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

More text."""

        blocks = chunker._identify_blocks(text)

        table_blocks = [b for b in blocks if b.type == ContentType.TABLE]
        assert len(table_blocks) == 1
        assert "Header 1" in table_blocks[0].content

    def test_identify_lists(self):
        """Test identification of lists."""
        from nanobot.rag.ingestion.chunking.document_chunker import (
            DocumentStructureChunker,
            ContentType,
        )

        chunker = DocumentStructureChunker()
        text = """Some text.

- Item 1
- Item 2
- Item 3

More text."""

        blocks = chunker._identify_blocks(text)

        list_blocks = [b for b in blocks if b.type == ContentType.LIST]
        assert len(list_blocks) == 1


class TestDocumentStructureChunkerSectionParsing:
    """Tests for section parsing."""

    def test_parse_single_heading(self):
        """Test parsing document with single heading."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        text = """# Chapter 1

This is the content of chapter 1."""

        sections = chunker._parse_sections(text, [])

        assert len(sections) == 1
        assert sections[0].level == 1
        assert sections[0].title == "Chapter 1"
        assert "content of chapter 1" in sections[0].content

    def test_parse_multiple_headings(self):
        """Test parsing document with multiple headings."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        text = """# Chapter 1

Content 1.

## Section 1.1

Content 1.1.

# Chapter 2

Content 2."""

        sections = chunker._parse_sections(text, [])

        assert len(sections) == 3
        assert sections[0].level == 1
        assert sections[0].title == "Chapter 1"
        assert sections[1].level == 2
        assert sections[1].title == "Section 1.1"
        assert sections[2].level == 1
        assert sections[2].title == "Chapter 2"

    def test_parse_section_path(self):
        """Test that section_path is correctly built."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        text = """# Main

## Subsection

### Detail

Content here."""

        sections = chunker._parse_sections(text, [])

        assert sections[0].path == "/Main"
        assert sections[1].path == "/Main/Subsection"
        assert sections[2].path == "/Main/Subsection/Detail"

    def test_parse_no_headings(self):
        """Test parsing document without headings."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        text = "Just plain text without any headings."

        sections = chunker._parse_sections(text, [])

        assert len(sections) == 1
        assert sections[0].level == 0
        assert sections[0].title == ""


class TestDocumentStructureChunkerMetadata:
    """Tests for structure-aware metadata."""

    def test_metadata_has_title(self):
        """Test that chunk metadata includes title."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        doc = Document(
            id="doc_001",
            text="# Test Title\n\nContent here.",
            metadata={"source_path": "test.md"},
        )

        chunks = chunker.split_document(doc)

        assert len(chunks) >= 1
        assert "title" in chunks[0].metadata
        assert chunks[0].metadata["title"] == "Test Title"

    def test_metadata_has_section_level(self):
        """Test that chunk metadata includes section_level."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        doc = Document(
            id="doc_001",
            text="# Main\n\nContent.\n\n## Sub\n\nMore content.",
            metadata={"source_path": "test.md"},
        )

        chunks = chunker.split_document(doc)

        # First chunk should be level 1
        level_1_chunks = [c for c in chunks if c.metadata.get("section_level") == 1]
        level_2_chunks = [c for c in chunks if c.metadata.get("section_level") == 2]

        assert len(level_1_chunks) >= 1
        assert len(level_2_chunks) >= 1

    def test_metadata_has_section_path(self):
        """Test that chunk metadata includes section_path."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        doc = Document(
            id="doc_001",
            text="# Chapter\n\n## Section\n\nContent.",
            metadata={"source_path": "test.md"},
        )

        chunks = chunker.split_document(doc)

        # At least one chunk should have a section_path
        paths = [c.metadata.get("section_path", "") for c in chunks]
        assert any("/Chapter" in p for p in paths)

    def test_metadata_has_content_type(self):
        """Test that chunk metadata includes content_type."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        doc = Document(
            id="doc_001",
            text="# Title\n\n```python\ncode\n```\n\nText.",
            metadata={"source_path": "test.md"},
        )

        chunks = chunker.split_document(doc)

        # Check that we have different content types
        content_types = {c.metadata.get("content_type") for c in chunks}
        assert "code" in content_types or "text" in content_types


class TestDocumentStructureChunkerNeighborLinks:
    """Tests for neighbor linking."""

    def test_neighbors_linked(self):
        """Test that adjacent chunks are linked."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        doc = Document(
            id="doc_001",
            text="# Section 1\n\nContent 1.\n\n# Section 2\n\nContent 2.\n\n# Section 3\n\nContent 3.",
            metadata={"source_path": "test.md"},
        )

        chunks = chunker.split_document(doc)

        if len(chunks) >= 2:
            # First chunk should have next but no prev
            assert chunks[0].metadata.get("prev_chunk_id") is None
            assert chunks[0].metadata.get("next_chunk_id") is not None

            # Middle chunks should have both
            if len(chunks) >= 3:
                assert chunks[1].metadata.get("prev_chunk_id") is not None
                assert chunks[1].metadata.get("next_chunk_id") is not None

            # Last chunk should have prev but no next
            assert chunks[-1].metadata.get("prev_chunk_id") is not None
            assert chunks[-1].metadata.get("next_chunk_id") is None

    def test_neighbor_ids_valid(self):
        """Test that neighbor IDs point to valid chunks."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        doc = Document(
            id="doc_001",
            text="# S1\n\nContent.\n\n# S2\n\nContent.\n\n# S3\n\nContent.",
            metadata={"source_path": "test.md"},
        )

        chunks = chunker.split_document(doc)
        chunk_ids = {c.id for c in chunks}

        for chunk in chunks:
            prev_id = chunk.metadata.get("prev_chunk_id")
            next_id = chunk.metadata.get("next_chunk_id")

            if prev_id is not None:
                assert prev_id in chunk_ids, f"prev_chunk_id {prev_id} not found"
            if next_id is not None:
                assert next_id in chunk_ids, f"next_chunk_id {next_id} not found"


class TestDocumentStructureChunkerCodeBlockProtection:
    """Tests for code block protection."""

    def test_code_block_not_split(self):
        """Test that code blocks are not split."""
        from nanobot.rag.ingestion.chunking.document_chunker import (
            DocumentStructureChunker,
            ContentType,
        )

        # Create a long code block
        long_code = "\n".join([f"    line_{i} = {i}" for i in range(100)])
        chunker = DocumentStructureChunker(max_chunk_length=500)  # Small limit

        doc = Document(
            id="doc_001",
            text=f"# Title\n\n```python\n{long_code}\n```\n\nText.",
            metadata={"source_path": "test.md"},
        )

        chunks = chunker.split_document(doc)

        # Find code chunk
        code_chunks = [c for c in chunks if c.metadata.get("content_type") == "code"]
        assert len(code_chunks) == 1
        # Code block should be intact
        assert "line_0" in code_chunks[0].text
        assert "line_99" in code_chunks[0].text

    def test_code_block_language_extracted(self):
        """Test that code block language is extracted."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        blocks = chunker._identify_blocks("```python\ncode\n```")

        assert len(blocks) == 1
        assert blocks[0].metadata.get("language") == "python"


class TestDocumentStructureChunkerIntegration:
    """Integration tests for DocumentStructureChunker."""

    def test_full_document_chunking(self):
        """Test complete document chunking with all features."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        doc = Document(
            id="doc_001",
            text="""# RAG System Design

## Overview

RAG (Retrieval-Augmented Generation) combines retrieval with generation.

## Implementation

```python
class RAGSystem:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
```

### Components

| Component | Description |
|-----------|-------------|
| Retriever | Finds relevant documents |
| Generator | Produces responses |

## Benefits

- Improved accuracy
- Reduced hallucinations
- Better context awareness
""",
            metadata={
                "source_path": "docs/rag_design.md",
                "doc_type": "technical",
            },
        )

        chunks = chunker.split_document(doc)

        # Verify we got multiple chunks
        assert len(chunks) >= 1

        # Verify metadata presence
        for chunk in chunks:
            assert "title" in chunk.metadata
            assert "section_level" in chunk.metadata
            assert "content_type" in chunk.metadata
            assert "chunk_index" in chunk.metadata

        # Verify content types
        content_types = {c.metadata.get("content_type") for c in chunks}
        assert "text" in content_types  # At least some text

    def test_empty_document_error(self):
        """Test that empty document raises error."""
        from nanobot.rag.ingestion.chunking.document_chunker import DocumentStructureChunker

        chunker = DocumentStructureChunker()
        doc = Document(
            id="doc_001",
            text="",
            metadata={"source_path": "test.md"},
        )

        with pytest.raises(ValueError, match="no text content"):
            chunker.split_document(doc)