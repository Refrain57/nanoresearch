# Web Message Attachments (outbound) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver files the agent attaches via `message(media=[...])` to the web chat user (currently dropped at SSE, persistence, and frontend).

**Architecture:** Ship a lightweight descriptor `{path,name,size}` (workspace-relative) over SSE and inside the persisted message JSONB; the frontend lazily fetches bytes from the existing authenticated `/api/workspace/files/{path}`. A shared `build_attachment_descriptors` helper maps absolute media paths to workspace-relative descriptors and drops anything outside the user's workspace. Frontend renders images inline and md/pdf/other as cards opening a shared `FilePreviewModal`.

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy (backend), Vue 3 + Vite + ant-design-vue + `marked` + `vue-pdf-embed` (frontend), Redis streams for SSE.

## Global Constraints

- Web channel only. Do NOT touch dingtalk/whatsapp/weixin media handling.
- No new DB table, no migration. Attachment descriptors ride inside `Message.content` (already JSONB).
- Descriptor shape is exactly `{"path": <workspace-relative posix str>, "name": <basename str>, "size": <int bytes>}`. No other keys.
- Security: never serve/read a media path outside the user's workspace root (`AgentLoop.workspace` = `base_workspace/users/{uid}`). Mapping uses `Path.relative_to`; `ValueError` ⇒ drop.
- Type policy (frontend): images inline; `.md`/`.pdf` preview modal; everything else download-only card. All types downloadable.
- Backend test env: `D:/Code/nanobot/backend/.venv/Scripts/python.exe` (pytest 9.1.0). Run tests FROM the worktree's `backend/` dir with `-m pytest` — e.g. `cd <worktree>/backend && D:/Code/nanobot/backend/.venv/Scripts/python.exe -m pytest tests/xxx.py -v`. Because `python -m` puts the worktree `backend/` on `sys.path[0]`, `import nanoresearch` resolves to the **worktree** (verified: `nanoresearch.__file__` points inside the worktree). **Do NOT run `pip install -e`** — it would repoint the shared editable install and break the original checkout's running backend.
- Frontend build gate: `npm run build` from the worktree's `web/` must pass. (Node deps already installed; do not reinstall unless a missing-module error occurs.)

---

### Task 1: Attachment descriptor helper (backend, pure)

**Files:**
- Modify: `backend/nanoresearch/server/routers/workspace_paths.py`
- Test: `backend/tests/test_workspace_paths.py` (create)

**Interfaces:**
- Produces: `build_attachment_descriptors(media: list[str] | None, workspace_root: Path) -> list[dict]`, each dict `{"path": str, "name": str, "size": int}`. Drops entries that are outside `workspace_root`, are not existing files, or unreadable.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_workspace_paths.py
from pathlib import Path
from nanoresearch.server.routers.workspace_paths import build_attachment_descriptors


def test_in_workspace_file_maps_to_descriptor(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    f = ws / "report.md"
    f.write_text("hello", encoding="utf-8")
    out = build_attachment_descriptors([str(f)], ws)
    assert out == [{"path": "report.md", "name": "report.md", "size": 5}]


def test_nested_subdir_path_preserved(tmp_path):
    ws = tmp_path / "users" / "alice"
    sub = ws / "sub" / "deep"
    sub.mkdir(parents=True)
    f = sub / "a.pdf"
    f.write_bytes(b"1234")
    out = build_attachment_descriptors([str(f)], ws)
    assert out == [{"path": "sub/deep/a.pdf", "name": "a.pdf", "size": 4}]


def test_outside_workspace_dropped(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    other = tmp_path / "users" / "bob" / "secret.txt"
    other.parent.mkdir(parents=True)
    other.write_text("x", encoding="utf-8")
    assert build_attachment_descriptors([str(other)], ws) == []


def test_missing_file_dropped(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    assert build_attachment_descriptors([str(ws / "nope.md")], ws) == []


def test_none_and_empty(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    assert build_attachment_descriptors(None, ws) == []
    assert build_attachment_descriptors([], ws) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_workspace_paths.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_attachment_descriptors'`.

- [ ] **Step 3: Write minimal implementation**

Append to `backend/nanoresearch/server/routers/workspace_paths.py`:

```python
def build_attachment_descriptors(
    media: list[str] | None, workspace_root: Path
) -> list[dict]:
    """Map absolute media paths to workspace-relative attachment descriptors.

    Drops any path outside workspace_root, non-existent, or not a file.
    """
    root = workspace_root.resolve()
    out: list[dict] = []
    for p in media or []:
        try:
            rp = Path(p).resolve()
            rel = rp.relative_to(root)
        except (ValueError, OSError):
            continue
        try:
            if not rp.is_file():
                continue
            size = rp.stat().st_size
        except OSError:
            continue
        out.append({"path": rel.as_posix(), "name": rp.name, "size": size})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_workspace_paths.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/server/routers/workspace_paths.py backend/tests/test_workspace_paths.py
git commit -m "feat(workspace): add build_attachment_descriptors helper"
```

---

### Task 2: MessageTool records media per send (backend)

**Files:**
- Modify: `backend/nanoresearch/agent/tools/message.py`
- Test: `backend/tests/test_message_tool.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `MessageTool.sent_media() -> list[list[str]]` — absolute media-path lists, index-aligned with `sent_contents()`. `start_turn()` resets both.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_message_tool.py
import pytest
from nanoresearch.agent.tools.message import MessageTool
from nanoresearch.bus.events import OutboundMessage


@pytest.mark.asyncio
async def test_sent_media_aligned_with_contents():
    sent: list[OutboundMessage] = []

    async def cb(m):
        sent.append(m)

    t = MessageTool(send_callback=cb, default_channel="web", default_chat_id="c1")
    t.start_turn()
    await t.execute(content="here", media=["/ws/users/alice/a.md"])
    await t.execute(content="no file")
    assert t.sent_contents() == ["here", "no file"]
    assert t.sent_media() == [["/ws/users/alice/a.md"], []]


@pytest.mark.asyncio
async def test_start_turn_resets_media():
    async def cb(m):
        return None

    t = MessageTool(send_callback=cb, default_channel="web", default_chat_id="c1")
    t.start_turn()
    await t.execute(content="x", media=["/ws/users/alice/a.md"])
    t.start_turn()
    assert t.sent_media() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_message_tool.py -v`
Expected: FAIL — `AttributeError: 'MessageTool' object has no attribute 'sent_media'`.

- [ ] **Step 3: Write minimal implementation**

In `message.py`:

`__init__`: add `self._sent_media: list[list[str]] = []` next to `self._sent_contents`.

`start_turn`:
```python
    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False
        self._sent_contents = []
        self._sent_media = []
```

Add method next to `sent_contents`:
```python
    def sent_media(self) -> list[list[str]]:
        """Media-path lists for sends to the default channel/chat this turn,
        index-aligned with sent_contents()."""
        return [list(m) for m in self._sent_media]
```

In `execute`, in the `if channel == self._default_channel and chat_id == self._default_chat_id:` block (right after `self._sent_contents.append(content)`):
```python
                self._sent_contents.append(content)
                self._sent_media.append(list(media or []))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_message_tool.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/agent/tools/message.py backend/tests/test_message_tool.py
git commit -m "feat(message-tool): track per-send media alongside content"
```

---

### Task 3: SSE sink includes media (backend)

**Files:**
- Modify: `backend/nanoresearch/worker.py` (the inline `_web_message_sink` closure near line 591)
- Test: `backend/tests/test_worker_agent_message_event.py` (create)

**Interfaces:**
- Consumes: `build_attachment_descriptors` (Task 1); `AgentLoop.workspace`.
- Produces: module-level factory `_make_web_message_sink(redis, stream_key: str, workspace_root) -> Callable[[Any], Awaitable[None]]`. The returned callback, for messages with `channel == "web"`, writes `{"type":"agent_message","content": m.content, "media": build_attachment_descriptors(m.media, workspace_root)}` via `xadd_event`.

- [ ] **Step 1: Write the failing test** (mirrors `test_worker_citations_event.py` `_FakeRedis` pattern)

```python
# backend/tests/test_worker_agent_message_event.py
from __future__ import annotations
import json
from pathlib import Path
import pytest


class _FakeRedis:
    def __init__(self):
        self._streams: dict[str, list[dict]] = {}
    async def xadd(self, key, fields):
        self._streams.setdefault(key, []).append(json.loads(fields["data"]))
    async def expire(self, key, ttl):
        pass
    def xadds_for(self, key):
        return self._streams.get(key, [])


class _Msg:
    def __init__(self, channel, content, media):
        self.channel = channel
        self.content = content
        self.media = media


@pytest.mark.asyncio
async def test_web_sink_emits_media_descriptors(tmp_path):
    from nanoresearch.worker import _make_web_message_sink
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    (ws / "r.md").write_text("hi", encoding="utf-8")

    fake = _FakeRedis()
    sink = _make_web_message_sink(fake, "run_events:t1", ws)
    await sink(_Msg("web", "see file", [str(ws / "r.md")]))

    ev = next(e for e in fake.xadds_for("run_events:t1") if e["type"] == "agent_message")
    assert ev["content"] == "see file"
    assert ev["media"] == [{"path": "r.md", "name": "r.md", "size": 2}]


@pytest.mark.asyncio
async def test_web_sink_drops_out_of_workspace_media(tmp_path):
    from nanoresearch.worker import _make_web_message_sink
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x", encoding="utf-8")

    fake = _FakeRedis()
    sink = _make_web_message_sink(fake, "run_events:t2", ws)
    await sink(_Msg("web", "hi", [str(outside)]))

    ev = next(e for e in fake.xadds_for("run_events:t2") if e["type"] == "agent_message")
    assert ev["media"] == []


@pytest.mark.asyncio
async def test_web_sink_ignores_non_web_channel(tmp_path):
    from nanoresearch.worker import _make_web_message_sink
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    fake = _FakeRedis()
    sink = _make_web_message_sink(fake, "run_events:t3", ws)
    await sink(_Msg("telegram", "hi", []))
    assert fake.xadds_for("run_events:t3") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_worker_agent_message_event.py -v`
Expected: FAIL — `ImportError: cannot import name '_make_web_message_sink'`.

- [ ] **Step 3: Write minimal implementation**

Add a module-level factory in `worker.py` (near `_make_on_citations`):

```python
def _make_web_message_sink(redis, stream_key: str, workspace_root):
    from nanoresearch.server.routers.workspace_paths import build_attachment_descriptors

    async def _sink(m) -> None:
        if getattr(m, "channel", None) != "web":
            return
        await xadd_event(redis, stream_key, {
            "type": "agent_message",
            "content": m.content,
            "media": build_attachment_descriptors(getattr(m, "media", None), workspace_root),
        })
    return _sink
```

Replace the inline closure at ~591 so the wiring uses the factory:

```python
        from nanoresearch.agent.tools.message import MessageTool as _MessageTool
        _mt = loop.tools.get("message")
        if isinstance(_mt, _MessageTool):
            _mt.set_send_callback(_make_web_message_sink(redis, run_stream_key, loop.workspace))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_worker_agent_message_event.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/worker.py backend/tests/test_worker_agent_message_event.py
git commit -m "feat(worker): include workspace attachment descriptors in agent_message SSE event"
```

---

### Task 4: Persist media in the web bridge (backend)

**Files:**
- Modify: `backend/nanoresearch/agent/loop.py` (web-bridge fold, ~949-956, inside `_process_message`)
- Test: `backend/tests/test_web_attachment_persist.py` (create)

**Interfaces:**
- Consumes: `build_attachment_descriptors` (Task 1); `MessageTool.sent_contents()` + `sent_media()` (Task 2); `AgentLoop.workspace`.
- Produces: module-level helper in `loop.py`: `build_sent_attachment_messages(contents: list[str], medias: list[list[str]], workspace_root) -> list[dict]` — one `{"role":"assistant","content": c}` per content, adding `"media": [descriptors]` only when non-empty.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_web_attachment_persist.py
from pathlib import Path
from nanoresearch.agent.loop import build_sent_attachment_messages


def test_build_messages_adds_media_when_present(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    (ws / "r.md").write_text("hi", encoding="utf-8")
    out = build_sent_attachment_messages(
        ["see file", "plain"],
        [[str(ws / "r.md")], []],
        ws,
    )
    assert out == [
        {"role": "assistant", "content": "see file",
         "media": [{"path": "r.md", "name": "r.md", "size": 2}]},
        {"role": "assistant", "content": "plain"},
    ]


def test_out_of_workspace_media_omitted(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    outside = tmp_path / "x.txt"
    outside.write_text("x", encoding="utf-8")
    out = build_sent_attachment_messages(["hi"], [[str(outside)]], ws)
    assert out == [{"role": "assistant", "content": "hi"}]  # media dropped -> key omitted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_web_attachment_persist.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_sent_attachment_messages'`.

- [ ] **Step 3: Write minimal implementation**

Add a module-level helper in `loop.py` (top-level, near other module helpers):

```python
def build_sent_attachment_messages(contents, medias, workspace_root):
    """Build assistant message dicts for web-bridge persistence, attaching
    workspace-relative media descriptors when present."""
    from nanoresearch.server.routers.workspace_paths import build_attachment_descriptors
    msgs = []
    for i, c in enumerate(contents):
        m = {"role": "assistant", "content": c}
        abs_media = medias[i] if i < len(medias) else []
        desc = build_attachment_descriptors(abs_media, workspace_root)
        if desc:
            m["media"] = desc
        msgs.append(m)
    return msgs
```

Wire the web-bridge fold (replace the `_send_msgs` construction at ~949-956):

```python
        if msg.channel == "web":
            _mt = self.tools.get("message")
            if isinstance(_mt, MessageTool) and (_sent := _mt.sent_contents()):
                _send_msgs = build_sent_attachment_messages(
                    _sent, _mt.sent_media(), self.workspace)
                if all_msgs and all_msgs[-1].get("role") == "assistant":
                    all_msgs = all_msgs[:-1] + _send_msgs + [all_msgs[-1]]
                else:
                    all_msgs = all_msgs + _send_msgs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_web_attachment_persist.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full backend suite touched here (regression)**

Run: `cd backend && python -m pytest tests/test_workspace_paths.py tests/test_message_tool.py tests/test_worker_agent_message_event.py tests/test_web_attachment_persist.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/agent/loop.py backend/tests/test_web_attachment_persist.py
git commit -m "feat(loop): persist web message attachments into saved turn"
```

---

### Task 5: Extract shared FilePreviewModal (frontend)

**Files:**
- Create: `web/src/components/FilePreviewModal.vue`
- Modify: `web/src/components/WorkspaceFiles.vue` (use the shared modal; remove the now-duplicated modal/preview logic)

**Interfaces:**
- Produces: `<FilePreviewModal v-model:open="open" :file="file" />` where `file` is `{ path: string, name: string }` (workspace-relative path). The component fetches the blob via `fetchWorkspaceFileBlob(file.path)` and renders: image inline / pdf via `VuePdfEmbed` / `.md` via `marked` (strip leading YAML frontmatter) / else a download-only view. Provides a download button. Emits nothing else.

- [ ] **Step 1: Create `FilePreviewModal.vue`**

Move the preview markup + logic out of `WorkspaceFiles.vue` into this component. It owns: `previewType` derivation from extension (`image`/`pdf`/`markdown`/`other`), blob fetch with a seq-guard, `renderedMd` (frontmatter strip + `marked.parse`), object-URL lifecycle, and download. Full component:

```vue
<template>
  <a-modal :open="open" :title="file?.name" :footer="null" width="80%"
           wrap-class-name="fp-modal" @update:open="v => emit('update:open', v)">
    <a-spin :spinning="loading">
      <div class="fp-body">
        <img v-if="kind === 'image' && url" :src="url" class="fp-img" />
        <VuePdfEmbed v-else-if="kind === 'pdf' && url" :source="url" class="fp-pdf" />
        <div v-else-if="kind === 'markdown'" class="fp-md" v-html="renderedMd"></div>
        <pre v-else-if="kind === 'text'" class="fp-text">{{ text }}</pre>
        <div v-else class="fp-other">此文件类型不支持预览，请下载后查看。</div>
      </div>
    </a-spin>
    <div class="fp-actions">
      <a-button size="small" @click="download"><download-outlined /> 下载</a-button>
    </div>
  </a-modal>
</template>

<script setup>
import { ref, watch, onBeforeUnmount } from 'vue'
import { message } from 'ant-design-vue'
import { DownloadOutlined } from '@ant-design/icons-vue'
import VuePdfEmbed from 'vue-pdf-embed'
import 'vue-pdf-embed/dist/styles/textLayer.css'
import 'vue-pdf-embed/dist/styles/annotationLayer.css'
import { marked } from 'marked'
import { fetchWorkspaceFileBlob } from '@/apis/workspace'

marked.setOptions({ breaks: true, gfm: true })

const props = defineProps({ open: Boolean, file: { type: Object, default: null } })
const emit = defineEmits(['update:open'])

const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'ico']
const TEXT_EXTS = ['txt', 'json', 'log', 'csv', 'yaml', 'yml', 'js', 'ts', 'jsx', 'tsx', 'vue', 'py', 'css', 'html', 'xml', 'sh', 'toml', 'ini']

const loading = ref(false)
const kind = ref('')
const url = ref('')
const text = ref('')
const renderedMd = ref('')
let blob = null
let seq = 0

function extOf(name) {
  const i = (name || '').lastIndexOf('.')
  return i >= 0 ? name.slice(i + 1).toLowerCase() : ''
}
function revoke() { if (url.value) { URL.revokeObjectURL(url.value); url.value = '' } }

async function load(file) {
  const my = ++seq
  revoke(); text.value = ''; renderedMd.value = ''; blob = null
  const ext = extOf(file.name)
  kind.value = IMAGE_EXTS.includes(ext) ? 'image'
    : ext === 'pdf' ? 'pdf'
    : ext === 'md' ? 'markdown'
    : TEXT_EXTS.includes(ext) ? 'text' : 'other'
  loading.value = true
  try {
    const b = await fetchWorkspaceFileBlob(file.path)
    if (my !== seq) return
    blob = b
    if (kind.value === 'markdown' || kind.value === 'text') {
      const raw = await b.text()
      if (my !== seq) return
      if (kind.value === 'markdown') {
        const body = raw.replace(/^﻿/, '').replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, '')
        renderedMd.value = marked.parse(body)
      } else { text.value = raw }
    } else {
      url.value = URL.createObjectURL(b)
    }
  } catch (e) {
    if (my === seq) { message.error(`预览 ${file.name} 失败：` + (e.message || '')); emit('update:open', false) }
  } finally {
    if (my === seq) loading.value = false
  }
}

function download() {
  if (!blob || !props.file) return
  const u = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = u; a.download = props.file.name
  document.body.appendChild(a); a.click(); a.remove()
  URL.revokeObjectURL(u)
}

watch(() => [props.open, props.file], ([open, file]) => {
  if (open && file) load(file); else if (!open) { seq++; revoke() }
})
onBeforeUnmount(revoke)
</script>

<style scoped>
.fp-body { max-height: 70vh; overflow: auto; display: flex; justify-content: center; }
.fp-img { max-width: 100%; height: auto; }
.fp-pdf { width: 100%; }
.fp-text { width: 100%; margin: 0; padding: 12px 14px; background: var(--nr-canvas, #faf7f2); color: var(--nr-ink, #2b2b2b); border-radius: 6px; font-size: 12.5px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
.fp-other { padding: 24px; color: var(--nr-ink-3); }
.fp-md { width: 100%; padding: 4px 8px 12px; font-size: 13.5px; line-height: 1.7; color: var(--nr-ink, #2b2b2b); word-break: break-word; text-align: left; }
.fp-md :deep(h1), .fp-md :deep(h2), .fp-md :deep(h3), .fp-md :deep(h4) { margin: 1em 0 0.5em; line-height: 1.3; font-weight: 600; }
.fp-md :deep(h1) { font-size: 1.5em; }
.fp-md :deep(h2) { font-size: 1.3em; border-bottom: 1px solid var(--nr-border); padding-bottom: 0.3em; }
.fp-md :deep(h3) { font-size: 1.15em; }
.fp-md :deep(p) { margin: 0.5em 0; }
.fp-md :deep(ul), .fp-md :deep(ol) { padding-left: 1.5em; margin: 0.5em 0; }
.fp-md :deep(li) { margin: 0.25em 0; }
.fp-md :deep(code) { background: var(--nr-border, #eee); padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.9em; }
.fp-md :deep(pre) { background: var(--nr-canvas, #faf7f2); padding: 10px 12px; border-radius: 6px; overflow-x: auto; }
.fp-md :deep(pre code) { background: none; padding: 0; }
.fp-md :deep(a) { color: var(--nr-clay, #b5651d); }
.fp-md :deep(blockquote) { margin: 0.5em 0; padding-left: 1em; border-left: 3px solid var(--nr-border); color: var(--nr-ink-3); }
.fp-md :deep(table) { border-collapse: collapse; margin: 0.5em 0; }
.fp-md :deep(th), .fp-md :deep(td) { border: 1px solid var(--nr-border); padding: 4px 8px; }
.fp-md :deep(img) { max-width: 100%; }
.fp-md :deep(hr) { border: none; border-top: 1px solid var(--nr-border); margin: 1em 0; }
.fp-actions { margin-top: 12px; display: flex; justify-content: flex-end; }
</style>
```

- [ ] **Step 2: Refactor `WorkspaceFiles.vue` to use it**

Remove the inline `<a-modal … wf-preview-modal>` block, the preview `<script>` state (`previewOpen/previewType/previewName/previewUrl/previewText/previewBlob/openSeq/renderedMd/revokePreviewUrl/closePreview/downloadPreview` and the marked import + IMAGE_EXTS/TEXT_EXTS/extOf duplicates used only by preview), and the preview styles (`.wf-preview-*` and `.wf-preview-md*`). Keep list/dir/download/delete logic. Replace `openFile` so text/image/pdf/md open the shared modal, and non-previewable still downloads:

```js
import FilePreviewModal from '@/components/FilePreviewModal.vue'
// ...
const previewOpen = ref(false)
const previewFile = ref(null)
const PREVIEWABLE = ['png','jpg','jpeg','gif','webp','svg','bmp','ico','pdf','md','txt','json','log','csv','yaml','yml','js','ts','jsx','tsx','vue','py','css','html','xml','sh','toml','ini']
function openFile(entry) {
  const ext = extOf(entry.name)  // keep a small extOf in this file
  if (PREVIEWABLE.includes(ext)) { previewFile.value = { path: entry.path, name: entry.name }; previewOpen.value = true }
  else downloadFile(entry)
}
```

Add `<FilePreviewModal v-model:open="previewOpen" :file="previewFile" />` in the template where the old modal was.

- [ ] **Step 3: Build**

Run: `cd web && npm run build`
Expected: build succeeds, no unresolved-import / template errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/FilePreviewModal.vue web/src/components/WorkspaceFiles.vue
git commit -m "refactor(web): extract shared FilePreviewModal used by workspace panel"
```

---

### Task 6: Render message attachments in chat (frontend)

**Files:**
- Modify: `web/src/views/ChatView.vue` (`onAgentMessage` handler ~517-527)
- Modify: `web/src/stores/chat.js` (surface `media` from persisted messages, same path as `citations`)
- Modify: `web/src/components/MessageList.vue` (render attachments; use `FilePreviewModal`)

**Interfaces:**
- Consumes: SSE `agent_message` event now has `event.media` (Task 3); persisted assistant messages carry `media` in their content dict (Task 4); `FilePreviewModal` (Task 5).
- Produces: assistant message objects carry `msg.media = [{path,name,size}]`.

- [ ] **Step 1: Live SSE — attach media (ChatView.vue)**

In `onAgentMessage`, add `media` to the pushed message:

```js
    onAgentMessage: (event) => {
      if (chatStore.currentConvId !== convId) return
      chatStore.messages.push({
        id: `agent-msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        role: 'assistant',
        content: { text: event.content },
        media: event.media || [],
        seq: chatStore.messages.length,
      })
    },
```

- [ ] **Step 2: History reload — surface media (stores/chat.js)**

Find where DB messages are mapped to frontend message objects (the same mapper that sets `citations`/`toolCalls` — grep `citations` in `web/src/stores/chat.js`). Where it reads `citations` from the persisted message dict, also read `media`:

```js
// alongside the existing citations extraction:
media: m.content?.media || m.media || [],
```

(Match the exact access pattern used for `citations` in that mapper; `media` lives in the persisted message dict written by Task 4.)

- [ ] **Step 3: Render attachments (MessageList.vue)**

Import and register the modal, add state:

```js
import FilePreviewModal from '@/components/FilePreviewModal.vue'
const previewOpen = ref(false)
const previewFile = ref(null)
const PREVIEW_EXTS = ['png','jpg','jpeg','gif','webp','svg','bmp','ico','pdf','md']
function attExt(name){ const i=(name||'').lastIndexOf('.'); return i>=0?name.slice(i+1).toLowerCase():'' }
function isImage(a){ return ['png','jpg','jpeg','gif','webp','svg','bmp','ico'].includes(attExt(a.name)) }
function canPreview(a){ return PREVIEW_EXTS.includes(attExt(a.name)) }
function fmtSize(b){ if(b<1024)return b+'B'; if(b<1048576)return (b/1024).toFixed(1)+'K'; return (b/1048576).toFixed(1)+'M' }
function openAtt(a){ previewFile.value={path:a.path,name:a.name}; previewOpen.value=true }
```

Add an attachments block right after the assistant bubble/tool-calls (inside the `msg.role !== 'tool'` template, after the tool-calls-panel `</div>`):

```html
      <div v-if="msg.role === 'assistant' && msg.media?.length" class="attachments">
        <template v-for="(a, i) in msg.media" :key="i">
          <img v-if="isImage(a)" :src="attImageUrl(a)" class="att-img" @click="openAtt(a)" />
          <div v-else class="att-card" :class="{ clickable: canPreview(a) }" @click="canPreview(a) ? openAtt(a) : downloadAtt(a)">
            <file-outlined class="att-icon" />
            <span class="att-name">{{ a.name }}</span>
            <span class="att-size">{{ fmtSize(a.size) }}</span>
            <download-outlined class="att-dl" @click.stop="downloadAtt(a)" />
          </div>
        </template>
      </div>
```

Images need bytes: fetch lazily into an object URL cache (auth blob, not a bare src):

```js
import { fetchWorkspaceFileBlob } from '@/apis/workspace'
const imgUrls = ref({})
function attImageUrl(a){
  if(!imgUrls.value[a.path]){
    imgUrls.value[a.path] = ''
    fetchWorkspaceFileBlob(a.path).then(b => { imgUrls.value = { ...imgUrls.value, [a.path]: URL.createObjectURL(b) } }).catch(()=>{})
  }
  return imgUrls.value[a.path]
}
async function downloadAtt(a){
  try { const b = await fetchWorkspaceFileBlob(a.path); const u=URL.createObjectURL(b); const el=document.createElement('a'); el.href=u; el.download=a.name; document.body.appendChild(el); el.click(); el.remove(); URL.revokeObjectURL(u) }
  catch(e){ /* message.error optional */ }
}
```

Add `<FilePreviewModal v-model:open="previewOpen" :file="previewFile" />` near the root of the template, ensure `FileOutlined`/`DownloadOutlined` are imported from `@ant-design/icons-vue`, and add minimal styles:

```css
.attachments { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 0 44px; }
.att-img { max-width: 220px; max-height: 160px; border-radius: 8px; cursor: pointer; border: 1px solid var(--nr-border); }
.att-card { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--nr-border); border-radius: 8px; background: #fff; max-width: 280px; }
.att-card.clickable { cursor: pointer; }
.att-card:hover { background: var(--nr-clay-soft); }
.att-icon { color: var(--nr-ink-3); }
.att-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12.5px; }
.att-size { font-size: 11px; color: var(--nr-ink-3); }
.att-dl { font-size: 12px; opacity: 0.6; cursor: pointer; }
.att-dl:hover { opacity: 1; }
```

- [ ] **Step 4: Build**

Run: `cd web && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/views/ChatView.vue web/src/stores/chat.js web/src/components/MessageList.vue
git commit -m "feat(web): render agent message attachments (inline images, cards, preview)"
```

- [ ] **Step 6: Manual visual verification (dev server)**

With backend + `web` dev server running, have an agent send a message with `.md`, an image, a `.pdf`, and a `.pptx` attachment. Confirm: image inline; `.md` card opens rendered modal; `.pdf` card previews; `.pptx` card downloads only; refresh the page — attachments persist (loaded from DB).

---

## Notes for the executor

- Backend Tasks 1–4 are pure/factory unit tests — no live Redis/DB/LLM needed.
- Frontend Tasks 5–6 gate on `npm run build`; Task 6 Step 6 is human visual verification.
- If the chat store maps DB messages in more than one place (initial load vs pagination `loadOlder`), surface `media` in each, mirroring `citations`.
