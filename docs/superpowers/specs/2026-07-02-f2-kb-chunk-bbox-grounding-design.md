# F2 — KB chunk ↔ PDF bbox grounding viewer

**Date:** 2026-07-02
**Branch:** `worktree-f2-grounding-viewer`
**Status:** design approved (user authorized direct execution)

## Problem

In the KB document viewer, selecting a chunk should highlight the region of the
source PDF that the chunk came from (bbox grounding). Today there is no
grounding: the parser produces per-block bbox/page but the loader discards it,
so `Chunk.position` is never populated and nothing reaches the KB API or UI.

## What already exists (do not rebuild)

- **Frontend preview modal** (`web/src/views/KnowledgeDetailView.vue`) already
  renders the real PDF client-side via `vue-pdf-embed` (`:source=pdfBlobUrl`,
  `:page=currentPdfPage`, page nav) with a right-side chunk list. `selectPreviewChunk`
  already jumps to `chunk.metadata.page_num ?? page` — but `page` is never
  populated, and there is **no bbox overlay**.
- **MinerU emits grounding natively.** `content_list.json` is a flattened,
  reading-order list where every block carries `page_idx` + `bbox`
  ([x0,y0,x1,y1], normalized to 0–1000, top-left origin) + `page_size`. The
  loader already calls `pipe_result.get_content_list(...)` (in
  `_extract_image_types`) — it just keeps `img_path`+type and drops the rest.
- **Metadata rides end-to-end for free.** `VectorUpserter` passes
  `dict(chunk.metadata)` → `ChunkPayload` → `KbChunk.chunk_metadata` (JSONB) →
  `_chunk_to_dict` returns `metadata`. Anything on `chunk.metadata` reaches the
  KB API with no schema/API change.

## Decisions (locked)

1. **Substrate:** overlay bbox rectangles on the live `vue-pdf-embed` page. No
   page-image rendering/storage/serving.
2. **Parser scope (v1):** MinerU only. Other parsers / pre-feature docs degrade
   gracefully (PDF + chunk list still work; no jump/highlight).
3. **Chunk↔block mapping:** **Approach A — align, keep the chunker.** Capture
   blocks in the loader; map chunk→blocks by order-preserving normalized-text
   match; compute grounding *before* transforms so LLM refinement can't break it.
   (Approach B, block-native chunking, is a documented future follow-up.)

## Architecture / data flow

### 1. Loader — capture blocks (`mineru_loader.py`)

Alongside the markdown, build:

```
document.metadata["mineru_blocks"] = [
  {"text": str, "page": int (1-based), "bbox": [x0, y0, x1, y1]},  # each in [0,1]
  ...
]  # in reading order
```

- **Source = `middle.json`, NOT `content_list`.** The installed package is
  **magic-pdf 1.x**, whose `content_list.json` has no bbox (bbox in content_list
  is a `mineru` 2.x feature). Use `pipe_result.get_middle_json()` →
  `json.loads(...)["pdf_info"]`; per page: `page_idx` (0-based), `page_size`
  `[w,h]`, and `para_blocks[].bbox` = `[x0,y0,x1,y1]` **absolute** coords,
  top-left origin. Block text = concatenation of `lines[].spans[].content`
  (image/table blocks nest one level deeper under `blocks[]`).
  - v0.6.x UNIPipe fallback: same structure at `pipe.pdf_mid_data["pdf_info"]`.
- `page = page_idx + 1` (1-based, matches `vue-pdf-embed :page` and existing UI).
- `bbox = [x0/w, y0/h, x1/w, y1/h]` — normalize absolute coords by `page_size`
  → per-axis fraction in `[0,1]`, top-left origin.
- Text blocks contribute their text; table/image/equation blocks contribute
  caption/empty text but keep bbox+page (so figure/table chunks still ground).
- **HTTP mode:** the loader posts to the repo's own `backend/scripts/mineru_server.py`,
  which currently returns only `{markdown, images}`. To support grounding in
  http mode, extend that server's `extract()` response with
  `middle_json = pipe_result.get_middle_json()` (pipe_result already built there)
  and parse it in the loader's http path. If a deployment's server predates this,
  http-mode docs degrade gracefully (no grounding).

### 2. Pipeline — grounding alignment (`pipeline.py`, Stage 3)

Run right after chunking and **before** Stage 4 transforms (same place the
existing `image_types` propagation runs, ~line 449). New module
`nanoresearch/rag/ingestion/grounding.py`:

```
def align_chunks_to_blocks(chunks, blocks) -> None:
    # mutates each chunk.metadata in place
```

Algorithm (order-preserving two-pointer):
1. Build a normalized concatenation of block texts, recording each block's
   `[start, end)` span in the normalized string. Normalization: lowercase,
   collapse whitespace, strip markdown markers/punctuation to content chars —
   robust to the RAGFlow "structured" chunker's reflow.
2. For each chunk (in order), normalize its text and `find()` it in the
   normalized block string starting from a running cursor (respects order,
   handles repeats). Map the matched `[start, end)` → overlapping block indices.
   Advance the cursor.
3. Fallback: if not found from cursor, try a global find; if still not found,
   leave grounding empty (graceful — that chunk gets no highlight).

Attach:
```
chunk.metadata["grounding"] = [{"page": int, "bbox": [x0,y0,x1,y1]}, ...]  # all matched blocks
chunk.metadata["page"] = <first matched block's page>   # lights up existing jump-to-page
```
Then `document.metadata.pop("mineru_blocks", None)` so it isn't inherited into
every chunk (mirrors the `image_types` pop).

### 3. Storage / API — no change

Grounding rides `chunk.metadata` → `KbChunk.chunk_metadata` → KB API `metadata`.
No DB migration, no new endpoint.

### 4. Frontend — bbox overlay (`KnowledgeDetailView.vue`)

Confirmed: `vue-pdf-embed@2.1.4` (pdfjs 4.10.38), single-page mode. Render tree:
`.vue-pdf-embed` (root) → wrapper → `.vue-pdf-embed__page` (position:relative) →
`<canvas>` + `.textLayer`.

- `page` now populated → existing `selectPreviewChunk` jump-to-page works.
- **Wrapper:** add a `.pdf-stage` (`position:relative; overflow:auto`) around
  `vue-pdf-embed`, and **move** the current `overflow-y:auto` off `.pdf-embed`
  onto `.pdf-stage`. Add a sibling `.bbox-overlay` (absolute, `pointer-events:none`)
  measured to the canvas box; rects are children positioned in **percentages** of
  the overlay so they auto-track resize.
- **Measure on `@rendered`** (NOT `@loaded` — canvas isn't sized at `loaded`),
  inside `nextTick` + `requestAnimationFrame` (a-modal settle). Read the box via
  `canvas.getBoundingClientRect()` — **never `canvas.width/height`** (those are
  `× devicePixelRatio`).
- **Resize-safe:** `ResizeObserver` on the canvas, re-bound inside the measure fn
  each render (page change replaces the canvas node). Overlay lives inside the
  scroll container so no scroll listener needed. Disconnect on preview close /
  unmount (mirror `closePreview`).
- **Rect mapping** (top-left origin, matches MinerU): for bbox `[x0,y0,x1,y1]`,
  `left:x0·100%, top:y0·100%, width:(x1-x0)·100%, height:(y1-y0)·100%`.
- **Current-page filter:** draw only grounding entries with `page === currentPdfPage`.
- Highlight style: semi-transparent clay fill + border (`--nr-accent`).
- Multi-page chunk: jump to first page; only current-page rects drawn.
- Edge cases to note (not blocking v1): library doesn't auto re-render on
  container resize (canvas stretches until a prop/`:key` changes) — optional
  `resizeKey`; rotated pages would need a rotation transform on bbox.

## Graceful degradation

- Non-MinerU / pre-feature docs: no `grounding`/`page` → viewer shows PDF +
  chunk list, no jump/highlight, no errors. Reprocess (force) with MinerU to
  backfill grounding.
- MinerU chunk that didn't align: no highlight for that chunk only.

## Testing

- **Aligner (highest-risk, TDD first):** unit tests for verbatim chunks,
  reflowed/merged chunks, multi-block chunks, cross-page chunks, and unmatched
  chunks (assert graceful empty). Pure function, no heavy deps.
- **Loader:** test with a mocked `content_list` (magic_pdf not installed) →
  assert `mineru_blocks` captured with 1-based page + normalized [0,1] bbox.
- **Overlay:** pure coord→pixel mapping unit test; manual verify in the app that
  rects land on the right region and survive resize/page change.

## Verify-first risks — RESOLVED by the two research agents

1. **MinerU accessor:** installed = magic-pdf **1.x** → use `get_middle_json()`
   (`pdf_info[].para_blocks[].bbox`, absolute, top-left, normalize by
   `page_size`). `content_list` has no bbox on 1.x. HTTP mode needs
   `mineru_server.py` to also return `middle_json` (now in scope).
2. **Frontend:** `vue-pdf-embed@2.1.4` → measure on `@rendered` via
   `getBoundingClientRect()` (DPR-safe), overlay in percentages, `ResizeObserver`
   on the canvas rebated each render.

Remaining verify-at-implementation: confirm on one real doc that (a) `page_size`
matches the bbox unit and (b) top-left origin renders correctly (flip Y only if
boxes appear vertically mirrored).

## Out of scope (v1)

- Marker/MarkItDown grounding.
- Block-native chunking (Approach B) — separate future spec; would give exact
  grounding by construction and help table extraction (Q1).
- Hover-preview highlight (nice-to-have); v1 is click-to-highlight.
- Backfilling grounding for already-ingested docs beyond the existing
  reprocess-with-force path.
