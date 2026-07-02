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
          <div v-else-if="previewType === 'markdown'" class="wf-preview-md" v-html="renderedMd"></div>
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
import { marked } from 'marked'
import { listWorkspaceFiles, deleteWorkspaceFile, fetchWorkspaceFileBlob } from '@/apis/workspace'

marked.setOptions({ breaks: true, gfm: true })

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
let openSeq = 0                // 递增序号，用于丢弃过期的预览请求

// .md 预览：剥掉开头的 YAML frontmatter 再交给 marked 渲染
const renderedMd = computed(() => {
  if (previewType.value !== 'markdown') return ''
  const body = previewText.value
    .replace(/^﻿/, '')
    .replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, '')
  return marked.parse(body)
})

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
  openSeq++
  revokePreviewUrl()
  previewText.value = ''
  previewBlob = null
}

async function openFile(entry) {
  const ext = extOf(entry.name)
  const kind = IMAGE_EXTS.includes(ext) ? 'image'
    : ext === 'pdf' ? 'pdf'
    : ext === 'md' ? 'markdown'
    : TEXT_EXTS.includes(ext) ? 'text'
    : 'other'

  if (kind === 'other') {
    downloadFile(entry)
    return
  }

  const mySeq = ++openSeq
  revokePreviewUrl()
  previewText.value = ''
  previewBlob = null
  previewName.value = entry.name
  previewType.value = kind
  previewOpen.value = true
  previewLoading.value = true
  try {
    const blob = await fetchWorkspaceFileBlob(entry.path)
    if (mySeq !== openSeq) return
    if (kind === 'text' || kind === 'markdown') {
      const text = await blob.text()
      if (mySeq !== openSeq) return
      previewText.value = text
    } else {
      previewUrl.value = URL.createObjectURL(blob)
    }
    previewBlob = blob
  } catch (e) {
    if (mySeq !== openSeq) return
    message.error(`预览 ${entry.name} 失败：` + (e.message || ''))
    closePreview()
  } finally {
    if (mySeq === openSeq) previewLoading.value = false
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
.wf-preview-md {
  width: 100%;
  padding: 4px 8px 12px;
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--nr-ink, #2b2b2b);
  word-break: break-word;
  text-align: left;
}
.wf-preview-md :deep(h1),
.wf-preview-md :deep(h2),
.wf-preview-md :deep(h3),
.wf-preview-md :deep(h4) { margin: 1em 0 0.5em; line-height: 1.3; font-weight: 600; }
.wf-preview-md :deep(h1) { font-size: 1.5em; }
.wf-preview-md :deep(h2) { font-size: 1.3em; border-bottom: 1px solid var(--nr-border); padding-bottom: 0.3em; }
.wf-preview-md :deep(h3) { font-size: 1.15em; }
.wf-preview-md :deep(p) { margin: 0.5em 0; }
.wf-preview-md :deep(ul),
.wf-preview-md :deep(ol) { padding-left: 1.5em; margin: 0.5em 0; }
.wf-preview-md :deep(li) { margin: 0.25em 0; }
.wf-preview-md :deep(code) {
  background: var(--nr-border, #eee);
  padding: 0.1em 0.35em;
  border-radius: 4px;
  font-size: 0.9em;
}
.wf-preview-md :deep(pre) {
  background: var(--nr-canvas, #faf7f2);
  padding: 10px 12px;
  border-radius: 6px;
  overflow-x: auto;
}
.wf-preview-md :deep(pre code) { background: none; padding: 0; }
.wf-preview-md :deep(a) { color: var(--nr-clay, #b5651d); }
.wf-preview-md :deep(blockquote) {
  margin: 0.5em 0;
  padding-left: 1em;
  border-left: 3px solid var(--nr-border);
  color: var(--nr-ink-3);
}
.wf-preview-md :deep(table) { border-collapse: collapse; margin: 0.5em 0; }
.wf-preview-md :deep(th),
.wf-preview-md :deep(td) { border: 1px solid var(--nr-border); padding: 4px 8px; }
.wf-preview-md :deep(img) { max-width: 100%; }
.wf-preview-md :deep(hr) { border: none; border-top: 1px solid var(--nr-border); margin: 1em 0; }

.wf-preview-actions {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}
</style>
