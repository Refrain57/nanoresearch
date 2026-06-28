"""Semantic chunker: NLTK sentence tokenization + agglomerative clustering.

Layer 1 of the Yuxi-aligned chunking pipeline:
- Mixed Chinese/English sentence splitting
- AgglomerativeClustering on sentence embeddings
- Size bounds: min_tokens=50, max_tokens from settings.ingestion.chunk_size
- Falls back to simple sentence merging when embedding unavailable
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from nanoresearch.rag.libs.splitter.base_splitter import BaseSplitter
from nanoresearch.rag.observability.logger import get_logger

if TYPE_CHECKING:
    from nanoresearch.rag.core.settings import Settings

logger = get_logger(__name__)

_PUNKT_CHECKED = False


def _ensure_punkt() -> None:
    global _PUNKT_CHECKED
    if _PUNKT_CHECKED:
        return
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    _PUNKT_CHECKED = True


def _split_chinese(text: str) -> list[str]:
    pattern = r'(?<=[。！？\u201d\u2019])|(?<=[。！？])(?![\u201c\u2018\u201d\u2019])'
    parts = re.split(pattern, text)
    return [s.strip() for s in parts if s.strip()]


def _split_mixed(text: str) -> list[str]:
    """Split mixed Chinese/English text into sentences."""
    _ensure_punkt()
    from nltk.tokenize import sent_tokenize

    sentences: list[str] = []
    for chunk in re.split(r"(\n+)", text):
        if not chunk.strip():
            continue
        if re.search(r"[A-Za-z]", chunk):
            sentences.extend(s.strip() for s in sent_tokenize(chunk) if s.strip())
        else:
            parts = _split_chinese(chunk)
            if parts:
                sentences.extend(parts)
            else:
                sentences.extend(
                    p.strip() for p in re.split(r"(?<=[。！？])", chunk) if p.strip()
                )
    return sentences


def _token_count(text: str) -> int:
    """Approximate token count: 4 chars ≈ 1 token."""
    return max(1, len(text) // 4)


def _embed_batched(
    sentences: list[str],
    embed_fn: Callable[[list[str]], Any],
    batch_size: int = 8,
) -> list:
    """Call embed_fn in small batches to respect provider batch size limits."""
    results = []
    for i in range(0, len(sentences), batch_size):
        results.extend(embed_fn(sentences[i : i + batch_size]))
    return results


def _cluster_and_chunk(
    sentences: list[str],
    embed_fn: Callable[[list[str]], Any],
    max_tokens: int,
) -> list[str]:
    """Embed sentences, cluster by semantics, reassemble into chunks."""
    sentence_tokens = [_token_count(s) for s in sentences]
    total_tokens = sum(sentence_tokens)

    embeddings = np.array(_embed_batched(sentences, embed_fn), dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    best_k = max(2, min((total_tokens + max_tokens - 1) // max_tokens, len(sentences)))

    labels = AgglomerativeClustering(
        n_clusters=best_k, metric="cosine", linkage="average"
    ).fit_predict(embeddings)

    chunks: list[str] = []
    current = ""
    current_tokens = 0
    current_label = int(labels[0])

    for sentence, label, tokens in zip(sentences, labels, sentence_tokens):
        label = int(label)
        if label != current_label or current_tokens + tokens > max_tokens:
            if current.strip():
                chunks.append(current.strip())
            current = sentence
            current_tokens = tokens
            current_label = label
        else:
            current = (current + " " + sentence) if current else sentence
            current_tokens += tokens

    if current.strip():
        chunks.append(current.strip())
    return chunks


def _simple_merge(sentences: list[str], max_tokens: int) -> list[str]:
    """Merge sentences sequentially without embeddings (fallback)."""
    chunks: list[str] = []
    current = ""
    current_tokens = 0
    for s in sentences:
        t = _token_count(s)
        if current_tokens + t > max_tokens and current:
            chunks.append(current.strip())
            current = s
            current_tokens = t
        else:
            current = (current + " " + s) if current else s
            current_tokens += t
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _enforce_size_bounds(chunks: list[str], min_tokens: int) -> list[str]:
    """Merge chunks smaller than min_tokens into their nearest neighbor."""
    if len(chunks) <= 1:
        return chunks

    result: list[str] = []
    for chunk in chunks:
        if _token_count(chunk) < min_tokens and result:
            result[-1] = result[-1] + "\n" + chunk
        else:
            result.append(chunk)

    # Handle leading undersized chunk
    if len(result) >= 2 and _token_count(result[0]) < min_tokens:
        result[1] = result[0] + "\n" + result[1]
        result = result[1:]

    return result


class SemanticChunker(BaseSplitter):
    """Semantic text splitter using sentence embeddings + agglomerative clustering.

    Splits text by semantic similarity rather than fixed character counts.
    Enforces size bounds: merges chunks < min_tokens, caps chunks at max_tokens.
    Falls back to simple sentence merging when the embedding service is unavailable.
    """

    def __init__(
        self,
        settings: "Settings",
        min_tokens: int = 50,
    ) -> None:
        self._min_tokens = min_tokens
        self._max_tokens = settings.ingestion.chunk_size if settings.ingestion else 512
        self._embed_fn: Optional[Callable[[list[str]], Any]] = None

        try:
            from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory
            _embedding = EmbeddingFactory.create(settings)
            self._embed_fn = _embedding.embed
            logger.info("SemanticChunker: embedding function loaded")
        except Exception as exc:
            logger.warning(f"SemanticChunker: embedding unavailable ({exc}), using simple merge")

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        self.validate_text(text)

        sentences = _split_mixed(text)
        if len(sentences) < 2:
            return [text.strip()]

        total_tokens = sum(_token_count(s) for s in sentences)

        if self._embed_fn is not None and total_tokens > self._max_tokens:
            try:
                chunks = _cluster_and_chunk(sentences, self._embed_fn, self._max_tokens)
            except Exception as exc:
                logger.warning(f"SemanticChunker clustering failed ({exc}), falling back to simple merge")
                chunks = _simple_merge(sentences, self._max_tokens)
        else:
            chunks = _simple_merge(sentences, self._max_tokens)

        chunks = _enforce_size_bounds(chunks, self._min_tokens)
        return [c for c in chunks if c.strip()]
