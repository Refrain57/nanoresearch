# Workspace 文件预览浮窗 + 删除功能 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让工作区文件通过带鉴权的浮窗预览（图片/文本/PDF）取代裸 `target="_blank"` 导航（根治 401），并新增带鉴权 + 越界防护的 DELETE 端点与前端删除按钮。

**Architecture:** 后端在 `workspace_router.py` 增加 `DELETE /api/workspace/files/{file_path:path}`，复用既有 `_safe_resolve` 越界防护，系统文件（`EDITABLE_FILES`）与工作区根禁止删除。前端 `WorkspaceFiles.vue` 改为经 `apis/workspace.js`（走 `apis/base.js` 的带鉴权 fetch）拉取 blob，在 `a-modal` 内按扩展名分流预览，删除用 `a-popconfirm` 确认。

**Tech Stack:** FastAPI + pytest（后端）；Vue 3 `<script setup>` + ant-design-vue（`a-modal`/`a-popconfirm`/`message`）+ `vue-pdf-embed`（前端，无单测框架，验证靠 `npm run build` + 手动）。

## Global Constraints

- 后端路由前缀与既有一致：`/api/workspace/files/{file_path:path}`。
- 越界防护必须复用 `_safe_resolve(workspace, rel_path)`（越界抛 `HTTPException(403, "非法路径")`）。
- 系统文件保护判定与既有 `write_file` 一致：`Path(file_path).name in EDITABLE_FILES`（值：`SOUL.md`, `AGENTS.md`, `USER.md`, `TOOLS.md`）。
- 前端所有鉴权请求走 `apis/workspace.js` → `apis/base.js`（`apiRequest`/`apiDelete`），不再手写裸 `<a href>` 或裸 `fetch`。
- 前端确认框统一用 `a-popconfirm`（`ok-text="删除" cancel-text="取消" ok-type="danger"`），与 `KnowledgeView.vue` 一致；触发元素加 `@click.stop` 防冒泡到可点击行。
- 文本预览用 `<pre>{{ text }}</pre>`（Vue 插值自动转义）；SVG 用 `<img>` 而非 `<iframe>`（不执行内嵌脚本）。
- 每个 `URL.createObjectURL` 都要在关闭/卸载时 `revokeObjectURL`。

---

### Task 1: 后端 DELETE 端点

**Files:**
- Modify: `backend/nanoresearch/server/routers/workspace_router.py`（顶部加 `import shutil`；文件末尾加 `delete_file` 端点）
- Test: `backend/tests/test_workspace_router.py`（新建）

**Interfaces:**
- Consumes（既有，勿改）：`_user_workspace(request, uid) -> Path`、`_safe_resolve(workspace, rel_path) -> Path`、`EDITABLE_FILES: set[str]`、`get_current_user`（`nanoresearch.server.middleware.auth`）。
- Produces：`DELETE /api/workspace/files/{file_path:path}` → 成功 `200 {"deleted": file_path}`；系统文件/根 `403`；越界 `403`；不存在 `404`。

- [ ] **Step 1: 写失败测试**

新建 `backend/tests/test_workspace_router.py`：

```python
"""Tests for workspace file DELETE endpoint + traversal guard."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.server.routers.workspace_router import router, _safe_resolve


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.loop_config = {"base_workspace": str(tmp_path)}
    app.dependency_overrides[get_current_user] = lambda: "u1"
    ws = tmp_path / "users" / "u1"
    ws.mkdir(parents=True)
    return TestClient(app), ws


def test_delete_file(client):
    c, ws = client
    (ws / "note.txt").write_text("hi", encoding="utf-8")
    resp = c.delete("/api/workspace/files/note.txt")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "note.txt"}
    assert not (ws / "note.txt").exists()


def test_delete_directory_recursive(client):
    c, ws = client
    d = ws / "sub"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")
    resp = c.delete("/api/workspace/files/sub")
    assert resp.status_code == 200
    assert not d.exists()


def test_delete_system_file_forbidden(client):
    c, ws = client
    (ws / "SOUL.md").write_text("soul", encoding="utf-8")
    resp = c.delete("/api/workspace/files/SOUL.md")
    assert resp.status_code == 403
    assert (ws / "SOUL.md").exists()


def test_delete_missing_file(client):
    c, ws = client
    resp = c.delete("/api/workspace/files/nope.txt")
    assert resp.status_code == 404


def test_delete_workspace_root_forbidden(client):
    c, ws = client
    resp = c.delete("/api/workspace/files/.")
    assert resp.status_code == 403


def test_safe_resolve_blocks_traversal(tmp_path):
    # 直接测越界守卫，避开 HTTP 客户端对 URL 中 ".." 的规范化
    with pytest.raises(HTTPException) as ei:
        _safe_resolve(tmp_path, "../../secret.txt")
    assert ei.value.status_code == 403
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/test_workspace_router.py -v`
Expected: 5 个 delete 测试 FAIL（`405 Method Not Allowed`，端点不存在）；`test_safe_resolve_blocks_traversal` PASS（守卫已存在）。

- [ ] **Step 3: 实现 DELETE 端点**

在 `backend/nanoresearch/server/routers/workspace_router.py` 顶部 import 区加：

```python
import shutil
```

在文件末尾（`download_file` 之后）追加：

```python
@router.delete("/api/workspace/files/{file_path:path}")
async def delete_file(
    file_path: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    if Path(file_path).name in EDITABLE_FILES:
        raise HTTPException(status_code=403, detail="系统文件不允许删除")
    workspace = _user_workspace(request, uid)
    resolved = _safe_resolve(workspace, file_path)
    if resolved == workspace.resolve():
        raise HTTPException(status_code=403, detail="不允许删除工作区根目录")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()
    return {"deleted": file_path}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/test_workspace_router.py -v`
Expected: 6 passed。

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/server/routers/workspace_router.py backend/tests/test_workspace_router.py
git commit -m "feat(workspace): DELETE 端点(递归删目录+系统文件/越界/根防护)"
```

---

### Task 2: 前端预览浮窗 + 下载 + 删除

**Files:**
- Modify: `web/src/apis/workspace.js`（加 `deleteWorkspaceFile`、`fetchWorkspaceFileBlob`，import 补 `apiDelete`）
- Modify: `web/src/components/WorkspaceFiles.vue`（重写文件行为 `<div @click>`、加预览 `a-modal`、加删除 `a-popconfirm`、改用 workspace api 助手）

**Interfaces:**
- Consumes（Task 1 产出）：`DELETE /api/workspace/files/{path}`；既有 `GET /api/workspace/files`（列目录）、`GET /api/workspace/files/{path}`（取内容）。
- Consumes（既有）：`apis/base.js` 的 `apiDelete(url)`、`apiRequest(url, opts, requiresAuth, responseType)`（当 `responseType` 非 `'json'`/`'text'` 时返回原始 `Response`）。
- Produces：`deleteWorkspaceFile(path) -> Promise`、`fetchWorkspaceFileBlob(path) -> Promise<Blob>`。

- [ ] **Step 1: 加 workspace api 助手**

编辑 `web/src/apis/workspace.js`，把首行 import 改为含 `apiDelete`，并在文件末尾追加两个导出。改后完整内容：

```js
import { apiGet, apiPut, apiDelete } from './base'
import { apiRequest } from './base'

export const listWorkspaceFiles = (dir = '') =>
  apiGet(`/api/workspace/files${dir ? '?dir=' + encodeURIComponent(dir) : ''}`)

export const getWorkspaceFile = (path) =>
  apiRequest(`/api/workspace/files/${path}`, { method: 'GET' }, true, 'text')

export const updateWorkspaceFile = (path, content) =>
  apiPut(`/api/workspace/files/${path}`, { content })

export const deleteWorkspaceFile = (path) =>
  apiDelete(`/api/workspace/files/${path}`)

export const fetchWorkspaceFileBlob = (path) =>
  apiRequest(`/api/workspace/files/${path}`, { method: 'GET' }, true, 'blob')
    .then((res) => res.blob())
```

- [ ] **Step 2: 重写 `WorkspaceFiles.vue`**

用以下完整内容替换 `web/src/components/WorkspaceFiles.vue`：

```vue
<template>
  <div class="wf-panel">
    <div class="wf-header">
      <a-button v-if="currentDir" size="small" type="text" class="wf-back" @click="goUp">
        <left-outlined />
      </a-button>
      <folder-open-outlined class="wf-header-icon" />
      <span class="wf-header-title">工作区</span>
      <span v-if="currentDir" class="wf-breadcrumb">/ {{ currentDir }}</span>
      <a-button size="small" type="text" class="wf-refresh" @click="refresh" :loading="loading">
        <reload-outlined />
      </a-button>
    </div>

    <a-spin :spinning="loading" size="small">
      <div class="wf-body">
        <a-empty v-if="!loading && !entries.length" description="暂无文件" :image="Empty.PRESENTED_IMAGE_SIMPLE" style="margin: 20px 0" />

        <div v-else class="wf-list">
          <!-- 返回上级 -->
          <div v-if="currentDir" class="wf-row wf-dir" @click="goUp">
            <folder-outlined class="wf-icon" />
            <span class="wf-name">..</span>
          </div>

          <!-- 目录 -->
          <div
            v-for="entry in dirs"
            :key="entry.path"
            class="wf-row wf-dir"
            @click="enterDir(entry.path)"
          >
            <folder-outlined class="wf-icon wf-icon-dir" />
            <span class="wf-name">{{ entry.name }}</span>
            <a-popconfirm
              :title="`删除目录 ${entry.name}？将递归删除其全部内容。`"
              ok-text="删除"
              cancel-text="取消"
              ok-type="danger"
              placement="left"
              @confirm="deleteEntry(entry)"
            >
              <delete-outlined class="wf-del" @click.stop />
            </a-popconfirm>
            <right-outlined class="wf-arrow" />
          </div>

          <!-- 文件 -->
          <div
            v-for="entry in files"
            :key="entry.path"
            class="wf-row wf-file"
            @click="openFile(entry)"
          >
            <file-outlined class="wf-icon" />
            <span class="wf-name">{{ entry.name }}</span>
            <span class="wf-size">{{ formatSize(entry.size) }}</span>
            <download-outlined class="wf-dl" @click.stop="downloadFile(entry)" />
            <a-popconfirm
              :title="`删除文件 ${entry.name}？`"
              ok-text="删除"
              cancel-text="取消"
              ok-type="danger"
              placement="left"
              @confirm="deleteEntry(entry)"
            >
              <delete-outlined class="wf-del" @click.stop />
            </a-popconfirm>
          </div>
        </div>
      </div>
    </a-spin>

    <!-- 文件预览浮窗 -->
    <a-modal
      v-model:open="previewOpen"
      :title="previewName"
      :footer="null"
      width="80%"
      wrap-class-name="wf-preview-modal"
      @cancel="closePreview"
    >
      <a-spin :spinning="previewLoading">
        <div class="wf-preview-body">
          <img v-if="previewType === 'image' && previewUrl" :src="previewUrl" class="wf-preview-img" />
          <VuePdfEmbed v-else-if="previewType === 'pdf' && previewUrl" :source="previewUrl" class="wf-preview-pdf" />
          <pre v-else-if="previewType === 'text'" class="wf-preview-text">{{ previewText }}</pre>
        </div>
      </a-spin>
      <div class="wf-preview-actions">
        <a-button size="small" @click="downloadPreview">
          <download-outlined /> 下载
        </a-button>
      </div>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { Empty, message } from 'ant-design-vue'
import {
  FolderOutlined, FolderOpenOutlined, FileOutlined,
  DownloadOutlined, ReloadOutlined, RightOutlined, LeftOutlined,
  DeleteOutlined,
} from '@ant-design/icons-vue'
import VuePdfEmbed from 'vue-pdf-embed'
import 'vue-pdf-embed/dist/styles/textLayer.css'
import 'vue-pdf-embed/dist/styles/annotationLayer.css'
import { listWorkspaceFiles, deleteWorkspaceFile, fetchWorkspaceFileBlob } from '@/apis/workspace'

const loading = ref(false)
const entries = ref([])
const currentDir = ref('')

const dirs = computed(() => entries.value.filter(e => e.is_dir))
const files = computed(() => entries.value.filter(e => !e.is_dir))

// ── 预览状态 ──
const previewOpen = ref(false)
const previewLoading = ref(false)
const previewType = ref('')   // 'image' | 'pdf' | 'text'
const previewName = ref('')
const previewUrl = ref('')     // 图片/PDF 的 object URL
const previewText = ref('')
let previewBlob = null         // 供浮窗内「下载」复用

const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'ico']
const TEXT_EXTS = ['md', 'txt', 'json', 'log', 'csv', 'yaml', 'yml', 'js', 'ts', 'jsx', 'tsx', 'vue', 'py', 'css', 'html', 'xml', 'sh', 'toml', 'ini']

function extOf(name) {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i + 1).toLowerCase() : ''
}

async function fetchDir(dir = '') {
  loading.value = true
  try {
    entries.value = await listWorkspaceFiles(dir)
  } catch (e) {
    message.error('加载工作区失败：' + (e.message || ''))
  } finally {
    loading.value = false
  }
}

function enterDir(path) {
  currentDir.value = path
  fetchDir(path)
}

function goUp() {
  const parts = currentDir.value.split('/').filter(Boolean)
  parts.pop()
  const parent = parts.join('/')
  currentDir.value = parent
  fetchDir(parent)
}

function refresh() {
  fetchDir(currentDir.value)
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}K`
  return `${(bytes / 1024 / 1024).toFixed(1)}M`
}

// ── 预览 / 下载 / 删除 ──
function revokePreviewUrl() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = ''
  }
}

function closePreview() {
  previewOpen.value = false
  revokePreviewUrl()
  previewText.value = ''
  previewBlob = null
}

async function openFile(entry) {
  const ext = extOf(entry.name)
  const kind = IMAGE_EXTS.includes(ext) ? 'image'
    : ext === 'pdf' ? 'pdf'
    : TEXT_EXTS.includes(ext) ? 'text'
    : 'other'

  if (kind === 'other') {
    downloadFile(entry)
    return
  }

  revokePreviewUrl()
  previewText.value = ''
  previewBlob = null
  previewName.value = entry.name
  previewType.value = kind
  previewOpen.value = true
  previewLoading.value = true
  try {
    const blob = await fetchWorkspaceFileBlob(entry.path)
    previewBlob = blob
    if (kind === 'text') {
      previewText.value = await blob.text()
    } else {
      previewUrl.value = URL.createObjectURL(blob)
    }
  } catch (e) {
    message.error(`预览 ${entry.name} 失败：` + (e.message || ''))
    closePreview()
  } finally {
    previewLoading.value = false
  }
}

function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

async function downloadFile(entry) {
  try {
    const blob = await fetchWorkspaceFileBlob(entry.path)
    triggerBlobDownload(blob, entry.name)
  } catch (e) {
    message.error(`下载 ${entry.name} 失败：` + (e.message || ''))
  }
}

function downloadPreview() {
  if (previewBlob) triggerBlobDownload(previewBlob, previewName.value)
}

async function deleteEntry(entry) {
  try {
    await deleteWorkspaceFile(entry.path)
    message.success(`已删除 ${entry.name}`)
    refresh()
  } catch (e) {
    message.error(`删除 ${entry.name} 失败：` + (e.message || ''))
  }
}

onMounted(() => fetchDir())
onBeforeUnmount(revokePreviewUrl)
</script>

<style scoped>
.wf-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--nr-rail);
  border-left: 1px solid var(--nr-border);
}

.wf-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 12px 14px 10px;
  border-bottom: 1px solid var(--nr-border);
  background: #fff;
  flex-shrink: 0;
}
.wf-header-icon { color: var(--nr-gold); font-size: 15px; }
.wf-header-title { font-size: 13px; font-weight: 600; color: var(--nr-ink); }
.wf-breadcrumb { font-size: 12px; color: var(--nr-ink-3); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.wf-back { flex-shrink: 0; }
.wf-refresh { margin-left: auto; }

.wf-body { flex: 1; overflow-y: auto; padding: 6px 0; }

.wf-list { display: flex; flex-direction: column; }

.wf-row {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 5px 14px;
  font-size: 13px;
  cursor: pointer;
  text-decoration: none;
  color: var(--nr-ink);
  transition: background 0.12s;
  min-width: 0;
}
.wf-row:hover { background: var(--nr-border); }
.wf-file:hover { background: var(--nr-clay-soft); color: var(--nr-clay); }

.wf-icon { font-size: 13px; flex-shrink: 0; color: var(--nr-ink-3); }
.wf-icon-dir { color: var(--nr-gold); }
.wf-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12.5px; }
.wf-size { font-size: 11px; color: var(--nr-ink-3); flex-shrink: 0; }
.wf-arrow { font-size: 10px; color: var(--nr-ink-3); flex-shrink: 0; }
.wf-dl { font-size: 12px; flex-shrink: 0; opacity: 0; transition: opacity 0.12s; }
.wf-file:hover .wf-dl { opacity: 0.6; }

.wf-del {
  font-size: 12px;
  flex-shrink: 0;
  opacity: 0;
  color: var(--nr-ink-3);
  transition: opacity 0.12s, color 0.12s;
}
.wf-row:hover .wf-del { opacity: 0.6; }
.wf-del:hover { color: #cf1322; opacity: 1; }

.wf-preview-body {
  max-height: 70vh;
  overflow: auto;
  display: flex;
  justify-content: center;
}
.wf-preview-img { max-width: 100%; height: auto; }
.wf-preview-pdf { width: 100%; }
.wf-preview-text {
  width: 100%;
  margin: 0;
  padding: 12px 14px;
  background: var(--nr-canvas, #faf7f2);
  color: var(--nr-ink, #2b2b2b);
  border-radius: 6px;
  font-size: 12.5px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}
.wf-preview-actions {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}
</style>
```

- [ ] **Step 3: 构建验证**

Run: `cd web && npm run build`
Expected: 构建成功，无 import 报错（`vue-pdf-embed`、图标、api 助手均已就位）。
（若 worktree 无 `node_modules`，先 `cd web && npm install`。）

- [ ] **Step 4: 手动验证清单**（`npm run dev` + 登录，或在部署环境）

- [ ] 点击图片文件 → 浮窗内显示图片；点击 pdf → 浮窗内渲染 PDF；点击 `.md`/`.txt`/`.json` → 浮窗内 `<pre>` 显示文本，且不再报 401、不再开新标签。
- [ ] 点击未列入扩展名的文件（如 `.zip`）→ 直接下载，不开浮窗。
- [ ] 浮窗内「下载」按钮可下载当前预览文件。
- [ ] 文件行 hover 显示下载 + 删除图标；删除弹 `a-popconfirm`，确认后文件消失并 `message.success`。
- [ ] 目录行 hover 显示删除图标；确认框提示「将递归删除」，确认后目录消失。
- [ ] 尝试删除 `SOUL.md` → 后端 403，前端 `message.error`。

- [ ] **Step 5: 提交**

```bash
git add web/src/apis/workspace.js web/src/components/WorkspaceFiles.vue
git commit -m "feat(workspace): 带鉴权浮窗预览(图/文/PDF)+下载+删除按钮"
```

---

## Self-Review

**1. Spec coverage:**
- Item 4（401 + 浮窗预览）→ Task 2（去裸链、`fetchWorkspaceFileBlob` 带鉴权、`a-modal` 按扩展名分流、SVG 走 `<img>`、文本 `<pre>` 转义、objectURL 回收）。✓
- Item 5（DELETE 端点 + 前端删除）→ Task 1（端点 + 系统文件/根/越界/404 分支 + 递归删目录）+ Task 2（`a-popconfirm` 删除按钮）。✓
- spec「非目标」（不做编辑/重命名/批量删）→ 计划未引入。✓

**2. Placeholder scan:** 无 TBD/TODO；所有代码步骤含完整代码；命令含预期输出。✓

**3. Type consistency:** `deleteWorkspaceFile` / `fetchWorkspaceFileBlob` 在 Task 2 Step 1 定义，Task 2 Step 2 组件按同名 import；后端 `delete_file` 返回 `{"deleted": file_path}` 与 Task 1 测试断言一致；`previewType` 取值 `'image'|'pdf'|'text'` 在 template 与 script 一致。✓
