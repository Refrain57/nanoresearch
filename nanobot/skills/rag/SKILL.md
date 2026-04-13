---
name: rag
description: RAG knowledge base search with advanced retrieval capabilities.
always: false
---

# RAG Search

## When to Use What

- **Simple query** → `retrieve_hybrid(query="...")`
- **Results incomplete** → `verify_results` to check, then refine
- **Complex/multi-part query** → `plan_query` to decompose first
- **Need custom fusion** → `fuse_results` with dense/sparse separately
- **Need citations** → `build_citations`
- **Multiple queries to compare/summarize** → Session tools (see below)

## Basic Usage

For simple queries:
```
retrieve_hybrid(query="your query")
```

## Document Management

### ingest_document
**When to use**: User wants to add a document to the knowledge base.

```
ingest_document(file_path="path/to/document.pdf", collection="default")
```

- Only PDF files are supported
- File path can be absolute or relative to workspace
- Large files may take time (chunking + embedding)

**Example workflow**:
```
1. ingest_document(file_path="papers/PGSR.pdf")
2. Wait for success confirmation
3. retrieve_hybrid(query="PGSR method")
```

### list_collections / list_documents
**When to use**: Explore what's in the knowledge base.

```
list_collections()  → see all collections
list_documents(collection="default")  → see documents in a collection
```

## Advanced Tools

### verify_results
**When to use**: Results don't fully answer the question, or you need to check if more searching is needed.

```
verify_results(results=json_result, query="original query")
```

Returns: `answered` (true/false), `confidence`, `suggestions.refined_queries`

**Workflow**:
```
1. retrieve_hybrid(query="X")
2. If results seem incomplete → verify_results(results=result, query="X")
3. If verify.answered=false → try verify.suggestions.refined_queries
```

### fuse_results
**When to use**: You want to control how dense and sparse results are combined.

```
fuse_results(dense_results=..., sparse_results=..., method="rrf", rrf_k=60)
```

Methods: "rrf" (default), "weighted", "interleave"

### plan_query
**When to use**: Complex multi-part questions that might benefit from decomposition.

```
plan_query(query="complex question")
```

Returns: `suggested_strategy`, `suggested_queries`, `keywords`, `structure_hints`, `retrieval_steps`

## Structure-Aware Tools

These tools leverage document structure metadata (section levels, headings, content types) for intelligent navigation.

### fetch_section
**When to use**: You know the document structure and need content from a specific section.

```
fetch_section(section_path="/Chapter 1/Section 1.1", collection="default")
```

Parameters:
- `section_path`: The section path in the document (e.g., `/RAG/检索策略`)
- `include_neighbors`: Also fetch adjacent chunks for context (default: true)
- `max_chunks`: Maximum chunks to return (default: 10)

**When to use**:
- Agent knows the exact section path from document structure
- Need comprehensive content from a specific topic
- Building detailed answer about a known section

**Example**:
```
1. retrieve_hybrid(query="RAG retrieval strategy") → found chunk from section "/RAG/检索策略"
2. fetch_section(section_path="/RAG/检索策略") → get full section content
```

### fetch_neighbors
**When to use**: Found a relevant chunk but need more context around it.

```
fetch_neighbors(chunk_id="doc_abc123_0005", window=1)
```

Parameters:
- `chunk_id`: The chunk ID to get neighbors for
- `window`: How many chunks before/after to fetch (default: 1)
- `collection`: Collection name (default: "default")

**When to use**:
- User asks "what else is around this content?"
- Building narrative flow from partial match
- Need context before/after a specific section

**Example**:
```
1. retrieve_hybrid(query="embedding model") → got chunk_id "doc_xyz_0010"
2. fetch_neighbors(chunk_id="doc_xyz_0010", window=2) → get 2 chunks before/after
```

### Structure Hints (from plan_query)

When you use `plan_query`, it returns `structure_hints` that guide which tools to use:

| Hint | Suggested Tool | Why |
|------|----------------|-----|
| `prefer_high_level` | `fetch_section` with overview sections | Overview queries benefit from high-level sections |
| `prefer_code` | Filter by `content_type=code` | Code queries need code blocks |
| `need_context` | `fetch_neighbors` | Need surrounding context |

**Example with structure hints**:
```
1. plan_query(query="Explain RAG architecture overview")
2. If structure_hints.shows prefer_high_level:
   → fetch_section(section_path="/RAG/Architecture")
3. Else if need_context:
   → fetch_neighbors(chunk_id=found_chunk_id)
```

## Query Planning with Structure

`plan_query` returns enhanced information for complex queries:

```
plan_query(query="Compare different RAG retrieval strategies")
```

Returns:
- `suggested_strategy`: "multi_hop", "direct", "comparison"
- `suggested_queries`: Array of sub-queries to execute
- `structure_hints`: Hints for structure-aware retrieval
- `retrieval_steps`: Suggested sequence of retrieval operations
- `stop_conditions`: When to stop multi-hop retrieval
  - `max_hops`: Maximum number of hops (default: 5)
  - `overlap_threshold`: Stop if overlap > 80%
  - `confidence_threshold`: Stop if confidence > 90%

### build_citations
**When to use**: User needs formatted references/citations.

```
build_citations(results=..., format="markdown")
```

## Iterative Search Pattern

When initial results are unsatisfactory:

```
1. retrieve_hybrid(query="X")
2. verify_results(results=result, query="X")
3. If not answered:
   - Check verify.suggestions.refined_queries
   - retrieve_hybrid(query=refined_query)
   - Or try retrieve_dense / retrieve_sparse separately
```

## Examples

**Simple search**:
- User: "What is PGSR?"
- Action: `retrieve_hybrid(query="PGSR")`

**Unsatisfied with results**:
- User: "Tell me more about PGSR's technical details"
- Action: `retrieve_hybrid(query="PGSR technical details")` → results seem incomplete
- Then: `verify_results(results=..., query="PGSR technical details")`
- If needed: Try refined query from suggestions

**Complex question**:
- User: "Compare PGSR with other 3D reconstruction methods"
- Action: `plan_query(query="compare PGSR with other methods")`
- Then: Multiple `retrieve_hybrid` calls based on suggested_queries

**Need citations**:
- User: "Give me references for this"
- Action: `build_citations(results=..., format="markdown")`

**Compare multiple queries**:
- User: "Compare PGSR and SuGaR training time"
- Action:
  ```
  1. create_session(initial_query="compare PGSR and SuGaR")
  2. retrieve_hybrid(query="PGSR training time")
  3. update_session(results=..., results_key="PGSR")
  4. retrieve_hybrid(query="SuGaR training time")
  5. update_session(results=..., results_key="SuGaR")
  6. get_session() → returns both results for comparison
  ```

## Session Tools

**Purpose**: Save and compare results from multiple queries.

When you need to:
- Compare results across different queries
- Summarize findings from multiple searches
- Keep history of all queries in a research task

### Tools

| Tool | Purpose |
|------|---------|
| `create_session` | Start a new session for tracking multiple queries |
| `update_session` | Save query results with a key for later reference |
| `get_session` | Retrieve all saved results for comparison/summary |
| `close_session` | End the session |
| `list_sessions` | See all active sessions |

### Workflow

```
1. create_session(initial_query="research topic")
2. retrieve_hybrid(query="aspect A") → update_session(results=..., results_key="A")
3. retrieve_hybrid(query="aspect B") → update_session(results=..., results_key="B")
4. get_session() → get all saved results
5. Compare/summarize the results
```
