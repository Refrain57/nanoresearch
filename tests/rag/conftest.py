"""Shared fixtures for RAG tests.

This module provides mock fixtures and test utilities for testing the RAG module
without requiring actual LLM API calls or vector database connections.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from nanobot.rag.core.types import (
    Document,
    Chunk,
    ChunkRecord,
    ProcessedQuery,
    RetrievalResult,
)


# ============================================================================
# Mock Settings
# ============================================================================

@dataclass
class MockSplitterSettings:
    """Mock splitter settings."""
    provider: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class MockEmbeddingSettings:
    """Mock embedding settings."""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536


@dataclass
class MockVectorStoreSettings:
    """Mock vector store settings."""
    provider: str = "chroma"
    persist_directory: str = "./test_data/chroma"


@dataclass
class MockRerankSettings:
    """Mock rerank settings."""
    enabled: bool = False
    top_k: int = 5


@dataclass
class MockRetrievalSettings:
    """Mock retrieval settings."""
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    rrf_k: int = 60


@dataclass
class MockIngestionSettings:
    """Mock ingestion settings."""
    batch_size: int = 100
    chunk_strategy: str = "fixed"  # Use traditional splitter for tests
    min_chunk_length: int = 100
    max_chunk_length: int = 2000


@dataclass
class MockLLMSettings:
    """Mock LLM settings."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"


@dataclass
class MockSettings:
    """Mock Settings object for testing.

    Provides all necessary configuration fields without requiring
    actual settings.yaml file or API connections.
    """
    splitter: MockSplitterSettings = field(default_factory=MockSplitterSettings)
    embedding: MockEmbeddingSettings = field(default_factory=MockEmbeddingSettings)
    vector_store: MockVectorStoreSettings = field(default_factory=MockVectorStoreSettings)
    rerank: MockRerankSettings = field(default_factory=MockRerankSettings)
    retrieval: MockRetrievalSettings = field(default_factory=MockRetrievalSettings)
    ingestion: MockIngestionSettings = field(default_factory=MockIngestionSettings)
    llm: MockLLMSettings = field(default_factory=MockLLMSettings)


@pytest.fixture
def mock_settings() -> MockSettings:
    """Provide mock settings for testing."""
    return MockSettings()


# ============================================================================
# Mock Components
# ============================================================================

@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """Mock embedding client that returns fixed vectors."""
    client = MagicMock()
    # Return predictable vectors for testing
    def mock_embed(texts: List[str]) -> List[List[float]]:
        return [[0.1 * i for i in range(1536)] for _ in texts]
    client.embed = mock_embed
    client.embed_batch = mock_embed
    return client


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock vector store for testing."""
    store = MagicMock()
    # In-memory storage for testing
    store._data: Dict[str, Dict[str, Any]] = {}

    def mock_upsert(records: List[ChunkRecord]) -> List[str]:
        ids = []
        for record in records:
            store._data[record.id] = {
                "id": record.id,
                "text": record.text,
                "metadata": record.metadata,
                "dense_vector": record.dense_vector,
            }
            ids.append(record.id)
        return ids

    def mock_query(vector: List[float], top_k: int = 10, filters: Optional[Dict] = None) -> List[RetrievalResult]:
        # Return mock results
        results = []
        for chunk_id, data in list(store._data.items())[:top_k]:
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=0.8,
                text=data["text"],
                metadata=data["metadata"],
            ))
        return results

    def mock_get_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
        return [store._data.get(id, {}) for id in ids if id in store._data]

    store.upsert = mock_upsert
    store.query = mock_query
    store.get_by_ids = mock_get_by_ids
    store.delete = MagicMock()
    store.clear = MagicMock()

    return store


@pytest.fixture
def mock_bm25_indexer() -> MagicMock:
    """Mock BM25 indexer for testing."""
    indexer = MagicMock()
    indexer._index: Dict[str, Dict[str, float]] = {}

    def mock_add_documents(docs: List[Dict], collection: str = "default") -> None:
        for doc in docs:
            indexer._index[doc.get("chunk_id", "unknown")] = doc.get("term_frequencies", {})

    def mock_search(query_terms: List[str], top_k: int = 10, collection: str = "default") -> List[RetrievalResult]:
        # Return mock results
        results = []
        for i, chunk_id in enumerate(list(indexer._index.keys())[:top_k]):
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=float(5 - i),  # BM25-like score
                text=f"Mock text for {chunk_id}",
                metadata={"source_path": "test.pdf"},
            ))
        return results

    indexer.add_documents = mock_add_documents
    indexer.search = mock_search
    indexer.save = MagicMock()
    indexer.load = MagicMock()

    return indexer


@pytest.fixture
def mock_reranker() -> MagicMock:
    """Mock reranker for testing."""
    reranker = MagicMock()

    def mock_rerank(query: str, candidates: List[Dict], **kwargs) -> List[Dict]:
        # Return candidates in reverse order (mock reranking effect)
        reranked = candidates.copy()
        # Add rerank_score
        for i, c in enumerate(reranked):
            c["rerank_score"] = 1.0 - (i * 0.1)
        return reranked

    reranker.rerank = mock_rerank
    reranker.__class__.__name__ = "MockReranker"

    return reranker


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Mock LLM provider for testing."""
    provider = MagicMock()
    provider.provider_name = "mock"

    def mock_generate(prompt: str, **kwargs) -> str:
        return "Mock LLM response"

    def mock_generate_structured(prompt: str, schema: Dict, **kwargs) -> Dict:
        return {"result": "mock structured output"}

    provider.generate = mock_generate
    provider.generate_structured = mock_generate_structured

    return provider


# ============================================================================
# Sample Data
# ============================================================================

@pytest.fixture
def sample_document() -> Document:
    """Provide sample document for testing."""
    return Document(
        id="doc_test_001",
        text="# Test Document\n\nThis is a test document for unit testing.\n\n"
             "## Section 1\n\nContent of section 1 with some keywords: Azure, OpenAI, configuration.\n\n"
             "## Section 2\n\nContent of section 2 discussing vector databases and embeddings.",
        metadata={
            "source_path": "test/documents/test.pdf",
            "doc_type": "pdf",
            "title": "Test Document",
            "page_count": 2,
        },
    )


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Provide sample chunks for testing."""
    return [
        Chunk(
            id="chunk_001_0000",
            text="# Test Document\n\nThis is a test document for unit testing.",
            metadata={
                "source_path": "test/documents/test.pdf",
                "chunk_index": 0,
                "source_ref": "doc_test_001",
            },
        ),
        Chunk(
            id="chunk_001_0001",
            text="## Section 1\n\nContent of section 1 with some keywords: Azure, OpenAI, configuration.",
            metadata={
                "source_path": "test/documents/test.pdf",
                "chunk_index": 1,
                "source_ref": "doc_test_001",
            },
        ),
        Chunk(
            id="chunk_001_0002",
            text="## Section 2\n\nContent of section 2 discussing vector databases and embeddings.",
            metadata={
                "source_path": "test/documents/test.pdf",
                "chunk_index": 2,
                "source_ref": "doc_test_001",
            },
        ),
    ]


@pytest.fixture
def sample_retrieval_results() -> List[RetrievalResult]:
    """Provide sample retrieval results for testing."""
    return [
        RetrievalResult(
            chunk_id="chunk_001",
            score=0.95,
            text="This is highly relevant content about Azure configuration.",
            metadata={"source_path": "docs/azure.pdf", "chunk_index": 1},
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            score=0.85,
            text="OpenAI API setup and configuration guide.",
            metadata={"source_path": "docs/openai.pdf", "chunk_index": 5},
        ),
        RetrievalResult(
            chunk_id="chunk_003",
            score=0.75,
            text="Vector database best practices.",
            metadata={"source_path": "docs/vector.pdf", "chunk_index": 2},
        ),
    ]


@pytest.fixture
def sample_dense_results() -> List[RetrievalResult]:
    """Provide sample dense retrieval results."""
    return [
        RetrievalResult(
            chunk_id="a",
            score=0.9,
            text="Dense result A",
            metadata={"source_path": "a.pdf"},
        ),
        RetrievalResult(
            chunk_id="b",
            score=0.8,
            text="Dense result B",
            metadata={"source_path": "b.pdf"},
        ),
        RetrievalResult(
            chunk_id="c",
            score=0.7,
            text="Dense result C",
            metadata={"source_path": "c.pdf"},
        ),
    ]


@pytest.fixture
def sample_sparse_results() -> List[RetrievalResult]:
    """Provide sample sparse retrieval results."""
    return [
        RetrievalResult(
            chunk_id="b",
            score=5.2,
            text="Sparse result B",
            metadata={"source_path": "b.pdf"},
        ),
        RetrievalResult(
            chunk_id="c",
            score=4.1,
            text="Sparse result C",
            metadata={"source_path": "c.pdf"},
        ),
        RetrievalResult(
            chunk_id="d",
            score=3.5,
            text="Sparse result D",
            metadata={"source_path": "d.pdf"},
        ),
    ]


@pytest.fixture
def sample_processed_query() -> ProcessedQuery:
    """Provide sample processed query."""
    return ProcessedQuery(
        original_query="如何配置 Azure OpenAI？",
        keywords=["配置", "Azure", "OpenAI"],
        filters={"collection": "docs"},
    )


# ============================================================================
# Utility Functions
# ============================================================================

def create_mock_retrieval_result(chunk_id: str, score: float, text: str = "") -> RetrievalResult:
    """Helper to create mock retrieval results."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"Mock text for {chunk_id}",
        metadata={"source_path": f"{chunk_id}.pdf"},
    )


def create_mock_chunks(count: int, prefix: str = "test") -> List[Chunk]:
    """Helper to create multiple mock chunks."""
    chunks = []
    for i in range(count):
        chunks.append(Chunk(
            id=f"{prefix}_chunk_{i:04d}",
            text=f"Content of chunk {i}. This chunk contains some test data.",
            metadata={
                "source_path": f"test/{prefix}.pdf",
                "chunk_index": i,
            },
        ))
    return chunks


def create_mock_dense_vectors(count: int, dimension: int = 1536) -> List[List[float]]:
    """Helper to create mock dense vectors."""
    vectors = []
    for i in range(count):
        # Create deterministic vectors for testing
        vector = [0.1 * (i + 1) + 0.01 * j for j in range(dimension)]
        vectors.append(vector)
    return vectors