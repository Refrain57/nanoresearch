# Step 2: Unified Ingestion Module — Implementation Plan

## Context

Per `docs/unified_ingestion_design.md` (v3), create `unified.py` implementing the `ingest_document` interface. This step creates the module + unit tests; it does **not** wire it into callers (that's Step 3 for worker, Step 4 for MCP).

## Prerequisites — 5 files need small changes before unified.py can work

### 1. `backend/nanobot/storage/models.py` (~line 162)
Add `content_hash` column to `KbDocument` (nullable, String(64), for backward compat):
```python
content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
```

### 2. `backend/nanobot/storage/repositories/knowledge_repo.py`
Three changes:
- **`create_document`** (line 190): add `content_hash: str | None = None` param, pass to constructor
- **New method `find_by_content_hash`** (~line 205): `SELECT ... WHERE kb_id=? AND content_hash=? AND status!='failed'` → `KbDocument | None`
- **New method `reset_for_reprocessing`** (~line 222): set status='reprocessing', clear chunk_count/error_msg, update file_path/content_hash if provided

### 3. `backend/nanobot/rag/core/types.py` (~line 121, after `Chunk`)
Add `ChunkPayload` dataclass:
```python
@dataclass
class ChunkPayload:
    chroma_id: str
    text: str
    token_count: int
    char_start: int | None = None
    char_end: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 4. `backend/nanobot/rag/ingestion/storage/vector_upserter.py`
- `upsert()` (line 73): change return type from `List[str]` to `tuple[List[str], List[ChunkPayload]]`. In the same loop that builds records, also build a `ChunkPayload` per chunk (chroma_id from `_generate_chunk_id`, text from `chunk.text`, token_count = `max(1, len(text)//4)`, char_start/end from `chunk.start_offset`/`chunk.end_offset`, metadata from `chunk.metadata`)
- `upsert_batch()` (line 170): update return type to match (aggregate payloads from flattened call, or just call `upsert` and return both)

### 5. `backend/nanobot/rag/ingestion/pipeline.py`
- `PipelineResult.__init__` (line 71): add `chunk_payloads: Optional[list] = None` param, store as `self.chunk_payloads = chunk_payloads or []`
- `Pipeline.run()` (line 597): capture new return: `vector_ids, chunk_payloads = self.vector_upserter.upsert(...)`
- `Pipeline.run()` return (line 696): add `chunk_payloads=chunk_payloads` to PipelineResult construction
- Error return path (line 710): no change needed (default `[]` is fine)

### 6. `backend/nanobot/storage/database.py` (~line 70)
Add `("kb_documents", "content_hash")` to CHECKS list for startup validation.

### 7. Migration script: `backend/scripts/migrate_content_hash.sql`
```sql
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
```

## New File: `backend/nanobot/rag/ingestion/unified.py`

### Errors
```python
class KbNotFoundError(RuntimeError): ...
class IngestFailedError(RuntimeError):
    def __init__(self, doc_id: str, cause: str): ...
```

### IngestResult (matching design doc exactly)
```python
@dataclass
class IngestResult:
    kb_id: str
    doc_id: str
    collection: str
    chunk_count: int
    status: Literal["created", "skipped_duplicate", "replaced"]
    duplicate_of: str | None = None
```

### `ingest_document()` signature
```python
async def ingest_document(
    *,
    kb_id: str,
    file_path: str,           # absolute, permanent path
    original_filename: str,
    content_hash: str,        # SHA256 hex, pre-computed by caller
    pdf_parser: Literal["mineru", "marker", "markitdown"] = "mineru",
    chunk_strategy: str = "auto",
    force: bool = False,
    uid: str = "",
    metadata: dict[str, Any] | None = None,
    # Injected dependencies (for testability):
    repo: KnowledgeRepository | None = None,
    settings: Settings | None = None,
    session_factory: Any = None,
) -> IngestResult:
```

### Flow implementation

**Step 0** — Validate `file_path`: must be absolute, must exist, must not be under `tempfile.gettempdir()`. Raises `ValueError`.

**Step A** — Resolve KB: `kb = await repo.get(uuid.UUID(kb_id))`. None → `KbNotFoundError`. Get `chroma_collection = kb.chroma_collection or kb_id`.

**Step B** — Dedup: `existing = await repo.find_by_content_hash(kb_uuid, content_hash)`.
- `existing.status == 'indexed'` and `force=False` → return `IngestResult(status="skipped_duplicate", duplicate_of=str(existing.id))`
- `existing.status == 'processing'` or `force=True` → fall through to Step C (reprocessing path)

**Step C** — PG record:
- *Reprocessing* (existing and (force or status=='processing')):
  1. `await repo.reset_for_reprocessing(existing.id, file_path, content_hash)`
  2. Clear old ChromaDB entries: `VectorStoreFactory.create(settings, collection_name=chroma_collection).delete_by_metadata({"source_path": existing.file_path or file_path})`
  3. `doc_uuid = existing.id`
- *New* (no existing):
  1. `doc = await repo.create_document(kb_uuid, original_filename, file_path, content_hash=content_hash, pdf_parser=pdf_parser)`
  2. `await repo.update_document_status(doc.id, "processing")`
  3. `doc_uuid = doc.id`

**Step D** — Pipeline:
```python
pipeline = IngestionPipeline(settings, collection=chroma_collection, force=True,
    pdf_parser=pdf_parser, chunk_strategy_override=chunk_strategy)
result = await asyncio.get_running_loop().run_in_executor(None, lambda: pipeline.run(file_path))
```
On failure:
1. `await repo.update_document_status(doc_uuid, "failed", error_msg=result.error)`
2. Clean up ChromaDB partial writes: `vs.delete_by_metadata({"source_path": file_path})`
3. `raise IngestFailedError(str(doc_uuid), result.error)`

**Step E** — Write chunks & finalize:
1. Build `list[KbChunk]` from `result.chunk_payloads`:
   ```python
   for idx, p in enumerate(result.chunk_payloads):
       meta = dict(p.metadata)
       meta["source_path"] = str(file_path)  # permanent path — required for delete_by_metadata cleanup
       meta["original_filename"] = original_filename  # display name preserved separately
       KbChunk(kb_id=kb_uuid, document_id=doc_uuid, chroma_id=p.chroma_id,
               chunk_index=idx, content=p.text, token_count=p.token_count,
               char_start=p.char_start, char_end=p.char_end, chunk_metadata=meta)
   ```
2. `await repo.create_chunks(chunk_rows)`
3. `await repo.update_document_status(doc_uuid, "indexed", chunk_count=len(chunk_rows))`
4. `await repo.increment_counts(kb_uuid, doc_delta=1, chunk_delta=len(chunk_rows))`
5. Return `IngestResult(status="created", ...)`

## New File: `backend/tests/unit/rag/ingestion/test_unified.py`

Using `pytest` + `pytest-asyncio` + `unittest.mock` (AsyncMock for repo, MagicMock for pipeline).

### Test cases (10 total)

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `test_path_not_absolute_raises` | `file_path="relative/path"` → `ValueError` |
| 2 | `test_path_in_temp_raises` | `file_path` under `gettempdir()` → `ValueError` |
| 3 | `test_path_not_exists_raises` | Non-existent file → `ValueError` |
| 4 | `test_kb_not_found_raises` | `repo.get()` returns `None` → `KbNotFoundError` |
| 5 | `test_dedup_skips` | Existing doc with status='indexed', force=False → `status="skipped_duplicate"` |
| 6 | `test_force_reprocess` | `force=True`, existing doc → calls `reset_for_reprocessing`, clears ChromaDB, runs pipeline → `status="replaced"` |
| 7 | `test_processing_residue_retries` | Existing doc with status='processing' → falls through to reprocessing (crashed last time) |
| 8 | `test_fresh_ingest_success` | No dedup → creates doc, runs pipeline, writes chunks, increments counts → `status="created"` |
| 9 | `test_pipeline_failure_rollback` | Pipeline returns `success=False` → doc marked 'failed', ChromaDB cleaned, `IngestFailedError` raised |
| 10 | `test_empty_chunks` | Pipeline returns 0 chunks → no crash, chunk_count=0, status still 'indexed' |

### Mock strategy
- `KnowledgeRepository` — all methods are `AsyncMock`
- `IngestionPipeline` — patched at class level, `.run()` returns controlled `PipelineResult`
- `VectorStoreFactory.create().delete_by_metadata()` — `MagicMock` returning int
- Settings — minimal `Settings()` object (or MagicMock)
- A temp file is created via `tempfile.NamedTemporaryFile` for path validation tests

## Sequencing

1. Types: add `ChunkPayload` to `types.py`
2. Model: add `content_hash` to `KbDocument`
3. Repo: add `find_by_content_hash`, `reset_for_reprocessing`, extend `create_document`
4. Pipeline: add `chunk_payloads` to `PipelineResult`
5. VectorUpserter: change `upsert()` return type
6. Pipeline: wire `chunk_payloads` through `run()`
7. DB checks: add `kb_documents.content_hash`
8. Migration: create SQL script
9. **Create `unified.py`** (the main deliverable)
10. **Create `test_unified.py`** (verify independently)

Items 1-8 are prerequisites; 9-10 are the deliverable. No caller changes (worker.py, MCP) in this step.

## Verification

```bash
cd backend
python -m pytest tests/unit/rag/ingestion/test_unified.py -v
```

All 10 tests pass independently, proving the unified flow is correct end-to-end without touching any caller code.
