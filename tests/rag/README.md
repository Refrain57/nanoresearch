# RAG Module Tests

This directory contains tests for the RAG (Retrieval-Augmented Generation) module.

## Test Structure

```
tests/rag/
├── conftest.py           # Shared fixtures and test utilities
├── unit/                 # Unit tests for individual components
│   ├── test_fusion.py          # RRF Fusion algorithm tests
│   ├── test_query_processor.py # Query preprocessing tests
│   ├── test_document_chunker.py # Document chunking tests
│   ├── test_hybrid_search.py   # Hybrid search tests
│   └── test_reranker.py        # Reranking tests
├── integration/          # Integration tests for complete flows
│   ├── test_ingestion_pipeline.py # End-to-end ingestion tests
│   └── test_query_flow.py        # Query flow tests
└── stress/               # Stress tests for performance
    ├── test_large_document_ingest.py # Large document tests
    ├── test_concurrent_queries.py    # Concurrent query tests
    └── test_memory_usage.py          # Memory usage tests
```

## Running Tests

### Run All Tests
```bash
pytest tests/rag/ -v
```

### Run Only Unit Tests
```bash
pytest tests/rag/unit/ -v
```

### Run Integration Tests
```bash
pytest tests/rag/integration/ -v
```

### Run Stress Tests (Skip by Default)
```bash
# Stress tests are marked and can be skipped
pytest tests/rag/ -v -m "not stress"

# Or run only stress tests
pytest tests/rag/stress/ -v -m stress
```

### Run with Coverage
```bash
pytest tests/rag/ --cov=nanobot.rag --cov-report=html
```

## Test Categories

### Unit Tests
Test individual components in isolation using mocks:
- `test_fusion.py`: RRF algorithm correctness, edge cases
- `test_query_processor.py`: Chinese/English tokenization, stopword filtering
- `test_document_chunker.py`: Chunk ID generation, metadata inheritance
- `test_hybrid_search.py`: Dense/Sparse retrieval coordination, fallback behavior
- `test_reranker.py`: Reranking logic, error handling

### Integration Tests
Test complete flows with mocked external dependencies:
- `test_ingestion_pipeline.py`: Full ingestion pipeline (load → chunk → embed → store)
- `test_query_flow.py`: Query processing → retrieval → reranking

### Stress Tests
Test performance under load:
- `test_large_document_ingest.py`: Documents with 1000+ chunks
- `test_concurrent_queries.py`: 50+ concurrent queries
- `test_memory_usage.py`: Memory profiling and leak detection

## Fixtures

Key fixtures in `conftest.py`:

- `mock_settings`: Mock Settings object for configuration
- `mock_embedding_client`: Mock embedding client with deterministic vectors
- `mock_vector_store`: In-memory vector store for testing
- `mock_bm25_indexer`: Mock BM25 indexer
- `sample_document`: Sample Document object for testing
- `sample_chunks`: List of sample Chunk objects
- `sample_retrieval_results`: Sample retrieval results

## Writing New Tests

1. **Unit Tests**: Use fixtures from `conftest.py`, mock external dependencies
2. **Integration Tests**: Mock heavy dependencies (LLM, vector DB) but test real flow
3. **Stress Tests**: Mark with `@pytest.mark.stress`, measure time/memory

Example:
```python
def test_my_component(mock_settings):
    """Test description."""
    from nanobot.rag.core.my_module import MyComponent

    component = MyComponent(mock_settings)
    result = component.process("input")

    assert result.success
```
