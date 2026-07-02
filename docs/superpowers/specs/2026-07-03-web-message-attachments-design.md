# Web Message Attachments (outbound) — Design

- **Date:** 2026-07-03
- **Status:** Approved (brainstorm complete)
- **Scope:** Outbound only — deliver files the agent attaches via `message(media=[...])` to the **web** chat user. The reverse direction (user uploads a file → workspace) is explicitly a separate follow-up spec.

## Problem

When the agent calls the `message` tool with a `media` attachment on the **web** channel, the file never reaches the user. The attachment is dropped at three layers:

1. **Live SSE** — `worker.py` `_web_message_sink` emits `{"type":"agent_message","content": m.content}` and drops `m.media`.
2. **Persistence** — `loop.py` web bridge (`if msg.channel == "web":`) folds only `MessageTool.sent_contents()` (text) into the saved turn; media is never stored.
3. **Frontend** — `web/src` has zero `media` handling; nothing would render it even if it arrived.

Result: the agent says "📎 here's your file, download it" but the user sees only text, no file.

Other channels (dingtalk/whatsapp/weixin) already handle `msg.media` in their own `send()`; **this design touches the web channel only.**

## Non-goals (YAGNI)

- Inbound (user uploads a file, "put it in my workspace") — separate spec.
- Server-side conversion of office/binary docs to previewable formats.
- A dedicated attachments table or object store (see Persistence).
- Changing how non-web channels handle media.

## Type policy (web rendering)

Governing rule: **the attachment card is always downloadable; preview is best-effort by type; anything the browser can't render falls back to a download-only card.** New/unknown types automatically land on "download" with no code change.

| Type | Web treatment |
|------|---------------|
| Images (png/jpg/jpeg/gif/webp/svg/bmp/ico) | inline `<img>` (click → large view) + download |
| `.md` | card → open shared preview modal (rendered via `marked`) + download |
| `.pdf` | card → open shared preview modal (VuePdfEmbed) + download |
| **everything else** (txt/json/csv/py/html/pptx/docx/… and unknown) | **download-only card** (icon + name + size) |

## Transmission — reference descriptor, not bytes

Do **not** ship file bytes over SSE or into the DB (base64 of a report = MBs per message; bloats SSE + JSONB). Instead ship a small **descriptor**; the frontend lazily fetches bytes on demand from the existing authenticated workspace endpoint.

**Descriptor shape** (mirrors the existing workspace list entry `_file_entry`):

```json
{ "path": "<workspace-relative posix path>", "name": "<basename>", "size": 12345 }
```

- `path` is relative to the user's workspace root — the same kind of path `GET /api/workspace/files` returns and `GET /api/workspace/files/{path}` serves.
- Frontend derives type/icon from the extension (reuse `extOf(name)`), fetches bytes with the existing `fetchWorkspaceFileBlob(path)` (JWT-authenticated blob) only when the user previews/downloads; images fetch eagerly to render inline.

## Abs → workspace-relative mapping (shared helper + security)

`m.media` entries are absolute paths (e.g. `…\.nanoresearch\workspace\users\admin\report.md`). A single helper maps each to a descriptor **or drops it**:

```
build_attachment_descriptors(media: list[str], workspace_root: Path) -> list[dict]:
  for p in media:
    rp = Path(p).resolve()
    try: rel = rp.relative_to(workspace_root.resolve())
    except ValueError: continue          # outside this user's workspace → drop (security)
    if not rp.is_file(): continue         # missing → drop
    yield {"path": rel.as_posix(), "name": rp.name, "size": rp.stat().st_size}
```

- `workspace_root` = the user's workspace root, same convention as `workspace_paths.user_workspace` (`base_workspace / "users" / uid`).
- Files outside the user's workspace are **never** served (relies on the same containment guarantee as `safe_resolve`). No traversal, no reading arbitrary disk paths.
- Dropped attachments simply don't appear; optionally the card can show a greyed "文件不可用" if we later want that signal (not required for v1).

The helper is used by **both** consumers below so live and persisted views agree.

## Backend changes

1. **SSE sink** — `worker.py` `_web_message_sink`: include mapped `media` in the event:
   `{"type":"agent_message","content": m.content, "media": build_attachment_descriptors(m.media, workspace_root)}`.
   The sink closure must capture the per-user `workspace_root` (available at run setup from `base_workspace` + uid).

2. **Persistence** — carry media through the web bridge (`loop.py`, `if msg.channel == "web":`). `MessageTool` currently records only text in `sent_contents()`; extend it to record `(content, media)` per delivered message. The bridge then folds assistant messages as
   `{"role":"assistant","content": c, "media": <descriptors>}` (map abs→rel with the same helper). Empty/absent `media` → omit the key (unchanged shape for text-only sends).

3. **Helper location** — put `build_attachment_descriptors` next to `workspace_paths` (it's the same domain) so router, worker, and loop share it.

No new table, no migration (see Persistence).

## Frontend changes

1. **SSE ingest** — the `onAgentMessage(event)` handler (wired at `useRunStream.js:39`) attaches `event.media` to the assistant message object it pushes into the chat store.
2. **History reload** — messages loaded from `GET /api/conversations/{id}/messages` carry `media` inside the persisted message dict; the store surfaces it on the message object the same way.
3. **Render** — `MessageList.vue` renders an attachments area under an assistant message's text: images inline; `.md`/`.pdf`/other as cards. Cards show icon + name + formatted size; `.md`/`.pdf`/images are clickable to preview.
4. **Shared preview component** — extract the preview modal currently inside `WorkspaceFiles.vue` into `components/FilePreviewModal.vue` (props: a descriptor / `{path,name}`; it fetches the blob, renders image / PDF / rendered-markdown, and offers download). `WorkspaceFiles.vue` and `MessageList.vue` both use it. This keeps the `.md` rendering + frontmatter-strip logic (from the issue-1 fix) single-sourced instead of duplicated.

## Persistence

Attachment descriptors ride inside `Message.content` (already `JSONB`, "full message dict"). No schema change, no migration. On reload the frontend re-fetches bytes from the workspace endpoint; if the agent later deleted the file, the card shows unavailable/download-fails — acceptable. (User has sanctioned a table if ever needed; it isn't for this feature. If that changes, a migration script + CHECKS entry is required — `serve` does not run `create_all`.)

## Testing

- **Backend unit:**
  - `build_attachment_descriptors`: in-workspace file → correct rel path/name/size; path outside workspace → dropped; missing file → dropped; nested subdir path preserved.
  - `_web_message_sink` emits `media` in the event when `m.media` is set.
  - Web-bridge persistence round-trip: a `message(media=[...])` send results in a saved assistant message dict carrying `media` descriptors.
  - Run from `<worktree>/backend`; first confirm `python -c "import nanoresearch, sys; print(nanoresearch.__file__)"` resolves to the **worktree** (editable install may point at the original repo).
- **Frontend:** build passes; manual visual — agent-attached `.md` renders in modal, image inline, pdf previews, `.pptx`/unknown gives download; refresh keeps attachments.

## Reuse summary

- Path scope/guard: `workspace_paths.user_workspace` / `safe_resolve` semantics.
- Byte delivery: existing `GET /api/workspace/files/{path}` + `fetchWorkspaceFileBlob`.
- Markdown/PDF/image rendering: existing `marked` + `VuePdfEmbed`, extracted into `FilePreviewModal`.
- Descriptor shape: existing `_file_entry`.
