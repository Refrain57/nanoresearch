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
            <right-outlined class="wf-arrow" />
          </div>

          <!-- 文件 -->
          <a
            v-for="entry in files"
            :key="entry.path"
            :href="`/api/workspace/files/${entry.path}`"
            target="_blank"
            class="wf-row wf-file"
          >
            <file-outlined class="wf-icon" />
            <span class="wf-name">{{ entry.name }}</span>
            <span class="wf-size">{{ formatSize(entry.size) }}</span>
            <download-outlined class="wf-dl" />
          </a>
        </div>
      </div>
    </a-spin>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Empty } from 'ant-design-vue'
import {
  FolderOutlined, FolderOpenOutlined, FileOutlined,
  DownloadOutlined, ReloadOutlined, RightOutlined, LeftOutlined,
} from '@ant-design/icons-vue'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const loading = ref(false)
const entries = ref([])
const currentDir = ref('')

const dirs = computed(() => entries.value.filter(e => e.is_dir))
const files = computed(() => entries.value.filter(e => !e.is_dir))

async function fetchDir(dir = '') {
  loading.value = true
  try {
    const params = dir ? `?dir=${encodeURIComponent(dir)}` : ''
    const res = await fetch(`/api/workspace/files${params}`, {
      headers: userStore.getAuthHeaders(),
    })
    if (res.ok) entries.value = await res.json()
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
</style>
