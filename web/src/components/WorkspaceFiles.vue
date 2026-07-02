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

    <!-- 文件预览浮窗（共享组件） -->
    <FilePreviewModal v-model:open="previewOpen" :file="previewFile" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Empty, message } from 'ant-design-vue'
import {
  FolderOutlined, FolderOpenOutlined, FileOutlined,
  DownloadOutlined, ReloadOutlined, RightOutlined, LeftOutlined,
  DeleteOutlined,
} from '@ant-design/icons-vue'
import { listWorkspaceFiles, deleteWorkspaceFile, fetchWorkspaceFileBlob } from '@/apis/workspace'
import FilePreviewModal from '@/components/FilePreviewModal.vue'

const loading = ref(false)
const entries = ref([])
const currentDir = ref('')

const dirs = computed(() => entries.value.filter(e => e.is_dir))
const files = computed(() => entries.value.filter(e => !e.is_dir))

// ── 预览状态（预览逻辑在 FilePreviewModal 内） ──
const previewOpen = ref(false)
const previewFile = ref(null)
const PREVIEWABLE = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'ico', 'pdf', 'md', 'txt', 'json', 'log', 'csv', 'yaml', 'yml', 'js', 'ts', 'jsx', 'tsx', 'vue', 'py', 'css', 'html', 'xml', 'sh', 'toml', 'ini']

function extOf(name) {
  const i = (name || '').lastIndexOf('.')
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
function openFile(entry) {
  const ext = extOf(entry.name)
  if (PREVIEWABLE.includes(ext)) {
    previewFile.value = { path: entry.path, name: entry.name }
    previewOpen.value = true
  } else {
    downloadFile(entry)
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
</style>
