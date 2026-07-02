"""Chunk ↔ source-block grounding alignment.

MinerU emits per-block layout (page + bbox) in reading order, but the chunker
reflows that text into chunks, so chunks lose their link to the source region.
This module rebuilds that link by order-preserving normalized-text matching:
each chunk is located within the concatenation of block texts, and every block
whose span overlaps the match is recorded as the chunk's grounding.

Grounding is written onto ``chunk.metadata`` (``page`` + ``grounding``) and must
be computed *before* any text-rewriting transform (e.g. ChunkRefiner) so the
chunk text still matches the source blocks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from nanoresearch.rag.core.types import Chunk

# Markdown image/link constructs — dropped whole so their URL/path chars (which
# never appear in the source blocks' text) don't break the contiguous match.
_MD_LINK = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
# Reduce text to comparable content characters: lowercase, drop all whitespace,
# punctuation, markdown markers and underscores. Keeps unicode word chars
# (letters incl. CJK/accented, digits), so it is robust to the chunker's
# whitespace/markdown reflow.
_NON_CONTENT = re.compile(r"[\W_]+", re.UNICODE)

Block = Dict[str, Any]


def _normalize(text: str) -> str:
    return _NON_CONTENT.sub("", _MD_LINK.sub("", text.lower()))


def align_chunks_to_blocks(chunks: List[Chunk], blocks: List[Block]) -> None:
    """Attach page + bbox grounding to each chunk, in place.

    Args:
        chunks: Chunks in document order (``chunk.text`` still original).
        blocks: Source blocks in reading order, each ``{text, page, bbox}`` with
            ``bbox`` a ``[x0, y0, x1, y1]`` fraction in ``[0, 1]``.

    For each chunk that can be located in the block stream, sets
    ``chunk.metadata["grounding"]`` = ``[{"page", "bbox"}, ...]`` (all overlapping
    blocks, in order) and ``chunk.metadata["page"]`` = the first block's page.
    Chunks that cannot be matched are left untouched (graceful degradation).
    """
    if not chunks or not blocks:
        return

    # Build a single normalized string of all block texts, recording each
    # block's [start, end) span within it.
    spans: List[tuple[int, int, Block]] = []
    parts: List[str] = []
    pos = 0
    for block in blocks:
        norm = _normalize(block.get("text", ""))
        spans.append((pos, pos + len(norm), block))
        parts.append(norm)
        pos += len(norm)
    concat = "".join(parts)
    if not concat:
        return

    cursor = 0
    for chunk in chunks:
        needle = _normalize(chunk.text)
        if not needle:
            continue

        idx = concat.find(needle, cursor)
        if idx == -1:
            # Order broke (e.g. chunk reordering); retry from the start.
            idx = concat.find(needle)
            if idx == -1:
                continue

        m_start, m_end = idx, idx + len(needle)
        matched = [
            block
            for (b_start, b_end, block) in spans
            if b_start < m_end and b_end > m_start
        ]
        if not matched:
            continue

        grounding = [{"page": b["page"], "bbox": b["bbox"]} for b in matched]
        chunk.metadata["grounding"] = grounding
        chunk.metadata["page"] = grounding[0]["page"]
        cursor = m_end
