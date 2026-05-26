<template>
  <app-layout>
    <div class="kb-detail-page">
      <!-- Header -->
      <div class="page-header">
        <div class="header-left">
          <a-button type="text" @click="router.push('/knowledge')" style="padding: 0; margin-right: 8px">
            <arrow-left-outlined />
          </a-button>
          <h2>{{ kb?.name || '知识库' }}</h2>
          <a-tag color="green" v-if="kb?.status === 'active'" style="margin-left: 8px">活跃</a-tag>
        </div>
        <div class="header-right">
          <span class="stat-chip"><file-text-outlined /> {{ kb?.doc_count ?? 0 }} 篇</span>
          <span class="stat-chip"><database-outlined /> {{ kb?.chunk_count ?? 0 }} Chunk</span>
          <a-button @click="router.push(`/knowledge/${kbId}/eval`)">评估</a-button>
        </div>
      </div>

      <a-tabs v-model:activeKey="activeTab">
        <!-- Tab 1: Documents -->
        <a-tab-pane key="docs" tab="文档">
          <div class="tab-toolbar">
            <a-upload
              :show-upload-list="false"
              :before-upload="handleUpload"
              accept=".pdf,.md,.txt,.docx"
              :multiple="true"
            >
              <a-button type="primary"><upload-outlined /> 上传文档</a-button>
            </a-upload>
          </div>

          <a-table
            :data-source="documents"
            :loading="docsLoading"
            :columns="docColumns"
            row-key="id"
            :pagination="false"
            size="middle"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'status'">
                <a-tag :color="statusColor(record.status)" size="small">{{ statusLabel(record.status) }}</a-tag>
                <a-spin v-if="isIndexing(record.status)" size="small" style="margin-left: 6px" />
              </template>
              <template v-if="column.key === 'file_size'">
                {{ formatSize(record.file_size) }}
              </template>
              <template v-if="column.key === 'action'">
                <a-button size="small" type="link" @click="openDocChunks(record)">查看 Chunks</a-button>
                <a-popconfirm title="确定删除？" ok-type="danger" @confirm="removeDoc(record.id)">
                  <a-button size="small" type="link" danger>删除</a-button>
                </a-popconfirm>
              </template>
            </template>
          </a-table>
        </a-tab-pane>

        <!-- Tab 2: Chunk Browser -->
        <a-tab-pane key="chunks" tab="Chunk 浏览">
          <div class="chunk-browser">
            <div class="chunk-doc-list">
              <div
                v-for="doc in documents"
                :key="doc.id"
                class="chunk-doc-item"
                :class="{ active: selectedDocId === doc.id }"
                @click="selectDoc(doc)"
              >
                <file-text-outlined />
                <span class="chunk-doc-name">{{ doc.filename }}</span>
                <a-badge :count="doc.chunk_count" :overflow-count="999" color="#1677ff" />
              </div>
              <a-empty v-if="!documents.length" description="暂无文档" style="padding: 32px 0" />
            </div>

            <div class="chunk-list-panel">
              <a-spin :spinning="chunksLoading">
                <div v-if="docChunks.length" class="chunk-items">
                  <div
                    v-for="chunk in docChunks"
                    :key="chunk.id"
                    class="chunk-item"
                    @click="openChunkDetail(chunk)"
                  >
                    <div class="chunk-header">
                      <span class="chunk-index">#{{ chunk.chunk_index }}</span>
                      <span class="chunk-tokens" v-if="chunk.token_count">{{ chunk.token_count }} tokens</span>
                    </div>
                    <div class="chunk-preview">{{ chunk.content.slice(0, 120) }}{{ chunk.content.length > 120 ? '…' : '' }}</div>
                  </div>
                </div>
                <a-empty v-else-if="selectedDocId" description="该文档暂无 Chunk" />
                <div v-else class="chunk-placeholder">← 选择左侧文档查看 Chunk</div>
              </a-spin>
            </div>
          </div>
        </a-tab-pane>

        <!-- Tab 3: Test Query -->
        <a-tab-pane key="query" tab="测试检索">
          <div class="query-panel">
            <div class="query-input-row">
              <a-input
                v-model:value="queryText"
                placeholder="输入检索问题..."
                @pressEnter="runQuery"
                style="flex: 1"
              />
              <a-input-number v-model:value="queryTopK" :min="1" :max="20" :step="1" style="width: 80px" />
              <a-button type="primary" :loading="queryLoading" @click="runQuery">检索</a-button>
            </div>

            <a-spin :spinning="queryLoading">
              <div v-if="queryResults.length" class="query-results">
                <div
                  v-for="(r, i) in queryResults"
                  :key="r.chunk_id"
                  class="query-result-item"
                >
                  <div class="result-header">
                    <span class="result-rank">#{{ i + 1 }}</span>
                    <a-tag color="blue" size="small">相似度 {{ (r.score * 100).toFixed(1) }}%</a-tag>
                    <span class="result-source">{{ r.metadata?.source_path?.split('/').pop() || '' }}</span>
                  </div>
                  <div class="result-text">{{ r.text }}</div>
                </div>
              </div>
              <a-empty v-else-if="queryDone" description="未找到相关内容" />
            </a-spin>
          </div>
        </a-tab-pane>
      </a-tabs>

      <!-- Chunk detail modal -->
      <a-modal v-model:open="chunkModalOpen" title="Chunk 详情" :footer="null" width="720">
        <div v-if="selectedChunk">
          <div class="chunk-meta">
            <span>索引 #{{ selectedChunk.chunk_index }}</span>
            <span v-if="selectedChunk.token_count">{{ selectedChunk.token_count }} tokens</span>
            <span v-if="selectedChunk.char_start !== null">字符 {{ selectedChunk.char_start }}–{{ selectedChunk.char_end }}</span>
          </div>
          <pre class="chunk-content">{{ selectedChunk.content }}</pre>
          <div v-if="selectedChunk.metadata && Object.keys(selectedChunk.metadata).length">
            <div class="chunk-meta-title">元数据</div>
            <pre class="chunk-meta-json">{{ JSON.stringify(selectedChunk.metadata, null, 2) }}</pre>
          </div>
        </div>
      </a-modal>
    </div>
  </app-layout>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  ArrowLeftOutlined, UploadOutlined, FileTextOutlined, DatabaseOutlined
} from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import { useKnowledgeStore } from '@/stores/knowledge'
import { listDocumentChunks, testQuery } from '@/apis/knowledge'

const route = useRoute()
const router = useRouter()
const kbStore = useKnowledgeStore()

const kbId = route.params.id
const kb = computed(() => kbStore.current)

const activeTab = ref('docs')
const documents = ref([])
const docsLoading = ref(false)

const selectedDocId = ref(null)
const docChunks = ref([])
const chunksLoading = ref(false)

const queryText = ref('')
const queryTopK = ref(5)
const queryLoading = ref(false)
const queryResults = ref([])
const queryDone = ref(false)

const chunkModalOpen = ref(false)
const selectedChunk = ref(null)

let pollTimer = null

const docColumns = [
  { title: '文件名', dataIndex: 'filename', key: 'filename' },
  { title: '大小', key: 'file_size', width: 100 },
  { title: 'Chunk 数', dataIndex: 'chunk_count', key: 'chunk_count', width: 100 },
  { title: '状态', key: 'status', width: 110 },
  { title: '操作', key: 'action', width: 180 },
]

onMounted(async () => {
  await kbStore.fetchOne(kbId)
  await loadDocs()
  startPoll()
})

onUnmounted(() => stopPoll())

async function loadDocs() {
  docsLoading.value = true
  try { documents.value = await kbStore.fetchDocuments(kbId) }
  finally { docsLoading.value = false }
}

function startPoll() {
  pollTimer = setInterval(() => {
    if (documents.value.some(d => isIndexing(d.status))) loadDocs()
  }, 3000)
}

function stopPoll() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

async function handleUpload(file) {
  try {
    await kbStore.uploadDoc(kbId, file)
    await loadDocs()
    message.success(`${file.name} 上传成功，正在解析…`)
  } catch (e) {
    message.error(e.message || '上传失败')
  }
  return false // prevent default upload
}

async function removeDoc(docId) {
  try {
    await kbStore.removeDocument(kbId, docId)
    await loadDocs()
    message.success('已删除')
  } catch (e) {
    message.error(e.message || '删除失败')
  }
}

async function selectDoc(doc) {
  selectedDocId.value = doc.id
  chunksLoading.value = true
  try {
    docChunks.value = await listDocumentChunks(kbId, doc.id)
  } finally {
    chunksLoading.value = false
  }
}

async function openDocChunks(doc) {
  activeTab.value = 'chunks'
  await selectDoc(doc)
}

function openChunkDetail(chunk) {
  selectedChunk.value = chunk
  chunkModalOpen.value = true
}

async function runQuery() {
  if (!queryText.value.trim()) return
  queryLoading.value = true
  queryDone.value = false
  queryResults.value = []
  try {
    const res = await testQuery(kbId, queryText.value, queryTopK.value)
    queryResults.value = res.results || []
    queryDone.value = true
  } catch (e) {
    message.error(e.message || '检索失败')
  } finally {
    queryLoading.value = false
  }
}

function statusColor(s) {
  return { uploaded: 'default', parsing: 'orange', indexing: 'blue', indexed: 'green', error: 'red' }[s] || 'default'
}

function statusLabel(s) {
  return { uploaded: '待处理', parsing: '解析中', indexing: '索引中', indexed: '已索引', error: '错误' }[s] || s
}

function isIndexing(s) { return s === 'parsing' || s === 'indexing' }

function formatSize(bytes) {
  if (!bytes) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
</script>

<style scoped>
.kb-detail-page { padding: 32px; }
.page-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.header-left { display: flex; align-items: center; }
.header-left h2 { font-size: 22px; font-weight: 700; margin: 0; }
.header-right { display: flex; align-items: center; gap: 12px; }
.stat-chip { font-size: 13px; color: #666; display: flex; align-items: center; gap: 4px; }

.tab-toolbar { margin-bottom: 16px; }

/* Chunk browser */
.chunk-browser { display: flex; gap: 0; border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden; height: 560px; }

.chunk-doc-list {
  width: 220px;
  border-right: 1px solid #f0f0f0;
  overflow-y: auto;
  background: #fafafa;
}
.chunk-doc-item {
  display: flex; align-items: center; gap: 8px; padding: 10px 14px;
  cursor: pointer; font-size: 13px; transition: background 0.15s;
  border-bottom: 1px solid #f0f0f0;
}
.chunk-doc-item:hover { background: #f0f8ff; }
.chunk-doc-item.active { background: #e6f4ff; font-weight: 600; color: #1677ff; }
.chunk-doc-name { flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }

.chunk-list-panel { flex: 1; overflow-y: auto; padding: 12px; }
.chunk-placeholder { display: flex; align-items: center; justify-content: center; height: 100%; color: #bbb; font-size: 14px; }
.chunk-items { display: flex; flex-direction: column; gap: 8px; }
.chunk-item {
  border: 1px solid #f0f0f0; border-radius: 6px; padding: 10px 12px;
  cursor: pointer; transition: all 0.15s;
}
.chunk-item:hover { border-color: #91caff; background: #f0f8ff; }
.chunk-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.chunk-index { font-size: 11px; font-weight: 700; color: #1677ff; background: #e6f4ff; border-radius: 4px; padding: 1px 6px; }
.chunk-tokens { font-size: 11px; color: #999; }
.chunk-preview { font-size: 12px; color: #555; line-height: 1.5; }

/* Query */
.query-panel { max-width: 800px; }
.query-input-row { display: flex; gap: 8px; margin-bottom: 20px; }
.query-results { display: flex; flex-direction: column; gap: 12px; }
.query-result-item { border: 1px solid #f0f0f0; border-radius: 8px; padding: 14px; }
.result-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.result-rank { font-size: 12px; font-weight: 700; color: #1677ff; }
.result-source { font-size: 11px; color: #999; }
.result-text { font-size: 13px; color: #444; line-height: 1.6; }

/* Chunk modal */
.chunk-meta { display: flex; gap: 16px; font-size: 12px; color: #888; margin-bottom: 12px; }
.chunk-content { background: #f8f8f8; border-radius: 6px; padding: 12px; font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; max-height: 400px; overflow-y: auto; }
.chunk-meta-title { font-size: 13px; font-weight: 600; margin: 12px 0 6px; }
.chunk-meta-json { background: #f8f8f8; border-radius: 6px; padding: 10px; font-size: 12px; max-height: 200px; overflow-y: auto; }
</style>
