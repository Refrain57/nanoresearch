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
                      <a-tag
                        v-if="chunk.metadata?.content_type && chunk.metadata.content_type !== 'text'"
                        :color="contentTypeColor(chunk.metadata.content_type)"
                        size="small"
                        style="margin: 0"
                      >{{ chunk.metadata.content_type }}</a-tag>
                      <span class="chunk-tokens" v-if="chunk.token_count">{{ chunk.token_count }} tokens</span>
                    </div>
                    <div v-if="chunk.metadata?.section_path" class="chunk-section-path">
                      {{ chunk.metadata.section_path }}
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
              <a-segmented v-model:value="queryMode" :options="queryModeOptions" />
              <a-button type="primary" :loading="queryLoading" @click="runQuery">检索</a-button>
            </div>

            <a-spin :spinning="queryLoading">
              <div v-if="queryResults.length" class="query-results">
                <div
                  v-for="group in groupedQueryResults"
                  :key="group.filename"
                  class="result-file-group"
                >
                  <div class="result-file-header" @click="toggleGroup(group.filename)">
                    <file-text-outlined style="color: #1677ff" />
                    <span class="result-file-name">{{ group.filename }}</span>
                    <a-badge :count="group.chunks.length" color="#1677ff" :overflow-count="99" />
                    <caret-right-outlined
                      class="group-toggle-icon"
                      :class="{ expanded: !collapsedGroups.has(group.filename) }"
                    />
                  </div>
                  <div v-if="!collapsedGroups.has(group.filename)" class="result-file-chunks">
                    <div
                      v-for="r in group.chunks"
                      :key="r.chunk_id"
                      class="query-result-item"
                      @click="openResultDetail(r)"
                    >
                      <div class="result-header">
                        <span class="result-rank">#{{ r._rank }}</span>
                        <a-tooltip title="RRF 融合分">
                          <a-tag color="blue" size="small">{{ (r.score * 100).toFixed(2) }}%</a-tag>
                        </a-tooltip>
                        <a-tooltip v-if="r.dense_score != null" title="语义检索分">
                          <a-tag color="purple" size="small">D {{ (r.dense_score * 100).toFixed(1) }}%</a-tag>
                        </a-tooltip>
                        <a-tooltip v-if="r.sparse_score != null" title="关键词检索分">
                          <a-tag color="cyan" size="small">S {{ (r.sparse_score * 100).toFixed(1) }}%</a-tag>
                        </a-tooltip>
                        <a-tag v-if="r.dense_score == null" size="small" color="default">仅关键词</a-tag>
                        <a-tag v-if="r.sparse_score == null" size="small" color="default">仅语义</a-tag>
                      </div>
                      <div class="result-text">{{ r.text }}</div>
                    </div>
                  </div>
                </div>
              </div>
              <a-empty v-else-if="queryDone" description="未找到相关内容" />
            </a-spin>
          </div>
        </a-tab-pane>
      </a-tabs>

      <!-- Query result detail modal -->
      <a-modal v-model:open="resultDetailOpen" title="检索结果详情" :footer="null" width="760">
        <div v-if="selectedResult">
          <div class="chunk-meta-pills">
            <span class="meta-pill">排名 #{{ selectedResult._rank }}</span>
            <a-tooltip title="RRF 融合分">
              <a-tag color="blue">{{ (selectedResult.score * 100).toFixed(2) }}%</a-tag>
            </a-tooltip>
            <a-tooltip v-if="selectedResult.dense_score != null" title="语义检索分">
              <a-tag color="purple">D {{ (selectedResult.dense_score * 100).toFixed(1) }}%</a-tag>
            </a-tooltip>
            <a-tooltip v-if="selectedResult.sparse_score != null" title="关键词检索分">
              <a-tag color="cyan">S {{ (selectedResult.sparse_score * 100).toFixed(1) }}%</a-tag>
            </a-tooltip>
          </div>
          <div class="chunk-content markdown-body" v-html="renderMarkdown(selectedResult.text)"></div>
          <template v-if="selectedResult.metadata && Object.keys(selectedResult.metadata).length">
            <div class="chunk-meta-title">元数据</div>
            <div class="chunk-meta-structured">
              <template v-for="(val, key) in structuredMetaFields(selectedResult.metadata)" :key="key">
                <div class="meta-row">
                  <span class="meta-key">{{ key }}</span>
                  <span class="meta-val">{{ val }}</span>
                </div>
              </template>
              <template v-if="Object.keys(rawMetaRemainder(selectedResult.metadata)).length">
                <div class="meta-row meta-row-json">
                  <span class="meta-key">其他</span>
                  <pre class="meta-val-json">{{ JSON.stringify(rawMetaRemainder(selectedResult.metadata), null, 2) }}</pre>
                </div>
              </template>
            </div>
          </template>
        </div>
      </a-modal>

      <!-- Chunk detail modal -->
      <a-modal v-model:open="chunkModalOpen" title="Chunk 详情" :footer="null" width="720">
        <div v-if="selectedChunk">
          <div class="chunk-meta-pills">
            <span class="meta-pill">#{{ selectedChunk.chunk_index }}</span>
            <span class="meta-pill" v-if="selectedChunk.token_count">{{ selectedChunk.token_count }} tokens</span>
            <span class="meta-pill" v-if="selectedChunk.char_start != null">字符 {{ selectedChunk.char_start }}–{{ selectedChunk.char_end }}</span>
            <a-tag
              v-if="selectedChunk.metadata?.content_type"
              :color="contentTypeColor(selectedChunk.metadata.content_type)"
              size="small"
            >{{ selectedChunk.metadata.content_type }}</a-tag>
          </div>
          <div class="chunk-content markdown-body" v-html="renderMarkdown(selectedChunk.content)"></div>
          <template v-if="selectedChunk.metadata && Object.keys(selectedChunk.metadata).length">
            <div class="chunk-meta-title">元数据</div>
            <div class="chunk-meta-structured">
              <template v-for="(val, key) in structuredMetaFields(selectedChunk.metadata)" :key="key">
                <div class="meta-row">
                  <span class="meta-key">{{ key }}</span>
                  <span class="meta-val">{{ val }}</span>
                </div>
              </template>
              <template v-if="Object.keys(rawMetaRemainder(selectedChunk.metadata)).length">
                <div class="meta-row meta-row-json">
                  <span class="meta-key">其他</span>
                  <pre class="meta-val-json">{{ JSON.stringify(rawMetaRemainder(selectedChunk.metadata), null, 2) }}</pre>
                </div>
              </template>
            </div>
          </template>
        </div>
      </a-modal>
    </div>
  </app-layout>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, reactive } from 'vue'
import { marked } from 'marked'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  ArrowLeftOutlined, UploadOutlined, FileTextOutlined, DatabaseOutlined, CaretRightOutlined
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
const queryMode = ref('hybrid')
const queryModeOptions = [
  { label: '混合', value: 'hybrid' },
  { label: '语义', value: 'dense' },
  { label: '关键词', value: 'sparse' },
]
const queryLoading = ref(false)
const queryResults = ref([])
const queryDone = ref(false)

const chunkModalOpen = ref(false)
const selectedChunk = ref(null)
const resultDetailOpen = ref(false)
const selectedResult = ref(null)
const collapsedGroups = reactive(new Set())

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

const groupedQueryResults = computed(() => {
  const map = new Map()
  queryResults.value.forEach((r, i) => {
    const filename = r.metadata?.source_path?.split('/').pop() || r.metadata?.source_path || '未知来源'
    if (!map.has(filename)) map.set(filename, [])
    map.get(filename).push({ ...r, _rank: i + 1 })
  })
  return Array.from(map.entries()).map(([filename, chunks]) => ({ filename, chunks }))
})

function toggleGroup(filename) {
  if (collapsedGroups.has(filename)) collapsedGroups.delete(filename)
  else collapsedGroups.add(filename)
}

function contentTypeColor(type) {
  return { code: 'orange', table: 'green', list: 'cyan', text: 'default' }[type] || 'default'
}

const STRUCTURED_META_KEYS = ['title', 'section_path', 'section_level', 'content_type', 'chunk_strategy_used', 'page_num', 'tags', 'summary', 'refined_by', 'enriched_by', 'prev_chunk_id', 'next_chunk_id']

function structuredMetaFields(meta) {
  const result = {}
  for (const key of STRUCTURED_META_KEYS) {
    if (meta[key] != null) {
      const val = Array.isArray(meta[key]) ? meta[key].join(', ') : String(meta[key])
      if (val) result[key] = val
    }
  }
  return result
}

function openResultDetail(r) {
  selectedResult.value = r
  resultDetailOpen.value = true
}

function renderMarkdown(text) {
  return marked.parse(text || '')
}

function rawMetaRemainder(meta) {
  const knownKeys = new Set([...STRUCTURED_META_KEYS, 'source_path', 'source_ref', 'chunk_index', 'doc_type'])
  const remainder = {}
  for (const [k, v] of Object.entries(meta)) {
    if (!knownKeys.has(k)) remainder[k] = v
  }
  return remainder
}

async function runQuery() {
  if (!queryText.value.trim()) return
  queryLoading.value = true
  queryDone.value = false
  queryResults.value = []
  try {
    const res = await testQuery(kbId, queryText.value, queryTopK.value, queryMode.value)
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

/* Chunk list */
.chunk-items { display: flex; flex-direction: column; gap: 8px; }
.chunk-item {
  border: 1px solid #f0f0f0; border-radius: 6px; padding: 10px 12px;
  cursor: pointer; transition: all 0.15s;
}
.chunk-item:hover { border-color: #91caff; background: #f0f8ff; }
.chunk-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.chunk-index { font-size: 11px; font-weight: 700; color: #1677ff; background: #e6f4ff; border-radius: 4px; padding: 1px 6px; }
.chunk-tokens { font-size: 11px; color: #999; }
.chunk-section-path { font-size: 11px; color: #888; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chunk-preview { font-size: 12px; color: #555; line-height: 1.5; }

/* Query */
.query-panel { max-width: 900px; }
.query-input-row { display: flex; gap: 8px; margin-bottom: 20px; }
.query-results { display: flex; flex-direction: column; gap: 8px; }

.result-file-group { border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden; }
.result-file-header {
  display: flex; align-items: center; gap: 8px; padding: 10px 14px;
  background: #fafafa; cursor: pointer; user-select: none;
  border-bottom: 1px solid #f0f0f0;
}
.result-file-header:hover { background: #f0f8ff; }
.result-file-name { flex: 1; font-size: 13px; font-weight: 600; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.group-toggle-icon { font-size: 12px; color: #999; transition: transform 0.2s; }
.group-toggle-icon.expanded { transform: rotate(90deg); }
.result-file-chunks { display: flex; flex-direction: column; gap: 0; }
.query-result-item { padding: 12px 14px; border-bottom: 1px solid #f8f8f8; cursor: pointer; transition: background 0.15s; }
.query-result-item:hover { background: #f0f8ff; }
.query-result-item:last-child { border-bottom: none; }
.result-header { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; flex-wrap: wrap; }
.result-rank { font-size: 12px; font-weight: 700; color: #1677ff; }
.result-text { font-size: 13px; color: #444; line-height: 1.6; }

/* Chunk modal */
.chunk-meta-pills { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.meta-pill { font-size: 12px; color: #888; background: #f5f5f5; border-radius: 4px; padding: 2px 8px; }
.chunk-content { background: #f8f8f8; border-radius: 6px; padding: 12px 16px; font-size: 13px; line-height: 1.7; max-height: 400px; overflow-y: auto; word-break: break-word; }
.chunk-content :deep(h1),.chunk-content :deep(h2),.chunk-content :deep(h3) { font-weight: 600; margin: 8px 0 4px; }
.chunk-content :deep(h1) { font-size: 16px; }
.chunk-content :deep(h2) { font-size: 14px; }
.chunk-content :deep(h3) { font-size: 13px; }
.chunk-content :deep(p) { margin: 4px 0; }
.chunk-content :deep(pre) { background: #e8e8e8; border-radius: 4px; padding: 8px; overflow-x: auto; font-size: 12px; }
.chunk-content :deep(code) { background: #e8e8e8; border-radius: 3px; padding: 1px 4px; font-size: 12px; }
.chunk-content :deep(pre code) { background: none; padding: 0; }
.chunk-content :deep(table) { border-collapse: collapse; width: 100%; font-size: 12px; margin: 6px 0; }
.chunk-content :deep(th),.chunk-content :deep(td) { border: 1px solid #ddd; padding: 4px 8px; }
.chunk-content :deep(th) { background: #eee; }
.chunk-content :deep(ul),.chunk-content :deep(ol) { padding-left: 20px; margin: 4px 0; }
.chunk-content :deep(blockquote) { border-left: 3px solid #ccc; margin: 4px 0; padding-left: 10px; color: #666; }
.chunk-meta-title { font-size: 13px; font-weight: 600; margin: 12px 0 6px; }
.chunk-meta-structured { border: 1px solid #f0f0f0; border-radius: 6px; overflow: hidden; }
.meta-row { display: flex; gap: 0; border-bottom: 1px solid #f8f8f8; font-size: 12px; }
.meta-row:last-child { border-bottom: none; }
.meta-key { width: 160px; flex-shrink: 0; padding: 6px 10px; background: #fafafa; color: #888; font-family: monospace; }
.meta-val { flex: 1; padding: 6px 10px; color: #333; word-break: break-all; }
.meta-row-json { align-items: flex-start; }
.meta-val-json { flex: 1; padding: 6px 10px; font-size: 11px; margin: 0; background: none; white-space: pre-wrap; word-break: break-word; max-height: 160px; overflow-y: auto; }
</style>
