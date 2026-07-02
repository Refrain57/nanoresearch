"""Unit tests for chunk↔block grounding alignment (grounding.py).

The aligner maps each chunk back to the MinerU source blocks it came from,
by order-preserving normalized-text matching, and writes grounding info onto
chunk.metadata (page + grounding rectangles). It must be robust to the RAGFlow
chunker's whitespace/markdown reflow, and degrade gracefully when a chunk can't
be matched.
"""

from __future__ import annotations

from nanoresearch.rag.core.types import Chunk
from nanoresearch.rag.ingestion.grounding import align_chunks_to_blocks


def _chunk(text: str) -> Chunk:
    return Chunk(id="c", text=text, metadata={"source_path": "x.pdf"})


def test_verbatim_single_block_grounds_to_that_block():
    blocks = [
        {"text": "Introduction to widgets.", "page": 1, "bbox": [0.1, 0.1, 0.9, 0.2]},
        {"text": "Widgets are useful.", "page": 1, "bbox": [0.1, 0.25, 0.9, 0.35]},
    ]
    chunks = [_chunk("Introduction to widgets.")]

    align_chunks_to_blocks(chunks, blocks)

    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["grounding"] == [
        {"page": 1, "bbox": [0.1, 0.1, 0.9, 0.2]},
    ]


def test_chunk_spanning_two_blocks_lists_both():
    blocks = [
        {"text": "First sentence here.", "page": 2, "bbox": [0.1, 0.1, 0.9, 0.2]},
        {"text": "Second sentence here.", "page": 2, "bbox": [0.1, 0.3, 0.9, 0.4]},
    ]
    chunks = [_chunk("First sentence here. Second sentence here.")]

    align_chunks_to_blocks(chunks, blocks)

    assert chunks[0].metadata["page"] == 2
    assert chunks[0].metadata["grounding"] == [
        {"page": 2, "bbox": [0.1, 0.1, 0.9, 0.2]},
        {"page": 2, "bbox": [0.1, 0.3, 0.9, 0.4]},
    ]


def test_reflowed_whitespace_and_markdown_still_matches():
    # Block text is plain; chunk has a markdown heading marker + collapsed/extra
    # whitespace + newlines, as the structured chunker would produce.
    blocks = [
        {"text": "Method Overview", "page": 1, "bbox": [0.1, 0.1, 0.9, 0.15]},
        {"text": "We propose a novel approach.", "page": 1, "bbox": [0.1, 0.2, 0.9, 0.3]},
    ]
    chunks = [_chunk("## Method   Overview\n\nWe propose a novel approach.")]

    align_chunks_to_blocks(chunks, blocks)

    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["grounding"] == [
        {"page": 1, "bbox": [0.1, 0.1, 0.9, 0.15]},
        {"page": 1, "bbox": [0.1, 0.2, 0.9, 0.3]},
    ]


def test_cross_page_chunk_primary_page_is_first_block():
    blocks = [
        {"text": "End of page one paragraph.", "page": 3, "bbox": [0.1, 0.8, 0.9, 0.9]},
        {"text": "Start of page two paragraph.", "page": 4, "bbox": [0.1, 0.1, 0.9, 0.2]},
    ]
    chunks = [_chunk("End of page one paragraph. Start of page two paragraph.")]

    align_chunks_to_blocks(chunks, blocks)

    assert chunks[0].metadata["page"] == 3
    assert [g["page"] for g in chunks[0].metadata["grounding"]] == [3, 4]


def test_unmatched_chunk_left_without_grounding():
    blocks = [
        {"text": "Completely unrelated content.", "page": 1, "bbox": [0.1, 0.1, 0.9, 0.2]},
    ]
    chunks = [_chunk("This text does not appear in any block whatsoever.")]

    align_chunks_to_blocks(chunks, blocks)

    assert "grounding" not in chunks[0].metadata
    assert "page" not in chunks[0].metadata


def test_repeated_block_text_assigned_in_document_order():
    # Two identical block texts; two identical chunks must map to distinct blocks
    # in order (cursor advances), not both to the first.
    blocks = [
        {"text": "See Table 1.", "page": 1, "bbox": [0.1, 0.1, 0.5, 0.2]},
        {"text": "See Table 1.", "page": 5, "bbox": [0.1, 0.6, 0.5, 0.7]},
    ]
    chunks = [_chunk("See Table 1."), _chunk("See Table 1.")]

    align_chunks_to_blocks(chunks, blocks)

    assert chunks[0].metadata["grounding"] == [{"page": 1, "bbox": [0.1, 0.1, 0.5, 0.2]}]
    assert chunks[1].metadata["grounding"] == [{"page": 5, "bbox": [0.1, 0.6, 0.5, 0.7]}]


def test_chunk_with_image_placeholder_still_grounds_surrounding_text():
    # Chunks frequently embed a markdown image placeholder whose path chars are
    # NOT present in any block's text. The aligner must ignore that syntax and
    # still match the surrounding paragraphs.
    blocks = [
        {"text": "Figure 1 shows the pipeline.", "page": 1, "bbox": [0.1, 0.1, 0.9, 0.2]},
        {"text": "The results are strong.", "page": 1, "bbox": [0.1, 0.5, 0.9, 0.6]},
    ]
    chunks = [
        _chunk(
            "Figure 1 shows the pipeline.\n\n"
            "![fig](rag/images/abc/fig1.png)\n\n"
            "The results are strong."
        )
    ]

    align_chunks_to_blocks(chunks, blocks)

    assert [g["page"] for g in chunks[0].metadata["grounding"]] == [1, 1]


def test_empty_blocks_list_is_noop():
    chunks = [_chunk("Anything at all.")]

    align_chunks_to_blocks(chunks, [])

    assert "grounding" not in chunks[0].metadata
    assert "page" not in chunks[0].metadata
