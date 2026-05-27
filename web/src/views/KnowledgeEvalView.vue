<template>
  <app-layout>
    <div class="eval-page">
      <!-- Header -->
      <div class="page-header">
        <div class="header-left">
          <a-button type="text" @click="router.push(`/knowledge/${kbId}`)" style="padding: 0; margin-right: 8px">
            <arrow-left-outlined />
          </a-button>
          <h2>{{ kbName }} — 评估</h2>
        </div>
      </div>

      <div class="eval-layout">
        <!-- Left: Datasets -->
        <div class="datasets-panel">
          <div class="panel-header">
            <h3>评估数据集</h3>
            <a-upload
              :show-upload-list="false"
              :before-upload="handleDatasetUpload"
              accept=".jsonl"
            >
              <a-button size="small"><upload-outlined /> 上传 JSONL</a-button>
            </a-upload>
          </div>

          <a-spin :spinning="datasetsLoading">
            <div v-if="datasets.length" class="dataset-list">
              <div
                v-for="ds in datasets"
                :key="ds.id"
                class="dataset-item"
                :class="{ active: selectedDataset?.id === ds.id }"
                @click="selectedDataset = ds"
              >
                <div class="ds-name">{{ ds.name }}</div>
                <div class="ds-meta">{{ ds.item_count }} 条 · {{ fmtDate(ds.created_at) }}</div>
                <div class="ds-actions">
                  <a-button
                    size="small"
                    type="primary"
                    @click.stop="startRun(ds)"
                    :loading="startingRun === ds.id + ':quick'"
                  >
                    Quick
                  </a-button>
                  <a-button
                    size="small"
                    @click.stop="startRagasRun(ds)"
                    :loading="startingRun === ds.id + ':ragas'"
                  >
                    RAGAS
                  </a-button>
                  <a-popconfirm title="确定删除数据集？" ok-type="danger" @confirm="removeDataset(ds.id)">
                    <a-button size="small" danger type="text" @click.stop>
                      <delete-outlined />
                    </a-button>
                  </a-popconfirm>
                </div>
              </div>
            </div>
            <a-empty v-else description="暂无数据集" style="padding: 24px 0" />
          </a-spin>

          <div class="jsonl-hint">
            JSONL 格式：每行 <code>{"query":"...","gold_answer":"...","gold_chunk_ids":[]}</code>
          </div>
        </div>

        <!-- Right: Eval Runs -->
        <div class="runs-panel">
          <div class="panel-header">
            <h3>评估历史</h3>
            <a-button size="small" @click="loadRuns"><reload-outlined /> 刷新</a-button>
          </div>

          <a-spin :spinning="runsLoading">
            <div v-if="evalRuns.length" class="runs-list">
              <div
                v-for="run in evalRuns"
                :key="run.id"
                class="run-card"
                :class="{ expanded: expandedRun === run.id }"
              >
                <!-- Run header -->
                <div class="run-header" @click="toggleRun(run)">
                  <div class="run-left">
                    <span class="run-name">{{ run.name }}</span>
                    <a-tag :color="run.eval_type === 'ragas' ? 'purple' : 'cyan'" size="small">{{ run.eval_type || 'quick' }}</a-tag>
                    <a-tag :color="runStatusColor(run.status)" size="small">{{ runStatusLabel(run.status) }}</a-tag>
                  </div>
                  <div class="run-right">
                    <span v-if="run.overall_score !== null" class="run-score">
                      总分 {{ (run.overall_score * 100).toFixed(1) }}%
                    </span>
                    <a-progress
                      v-if="run.status === 'running'"
                      :percent="runProgress(run)"
                      size="small"
                      style="width: 100px"
                    />
                    <a-popconfirm title="确定删除？" ok-type="danger" @confirm.stop="removeRun(run)">
                      <a-button size="small" type="text" danger @click.stop>
                        <delete-outlined />
                      </a-button>
                    </a-popconfirm>
                  </div>
                </div>

                <!-- Metrics summary -->
                <div v-if="run.metrics && Object.keys(run.metrics).length" class="run-metrics">
                  <span v-for="(v, k) in run.metrics" :key="k" class="metric-chip">
                    {{ k }}: {{ (v * 100).toFixed(1) }}%
                  </span>
                </div>

                <!-- Expanded: item details -->
                <div v-if="expandedRun === run.id" class="run-items">
                  <a-spin :spinning="itemsLoading">
                    <div v-if="runItems.length" class="items-table">
                      <div class="items-header">
                        <span style="flex: 2">问题</span>
                        <span style="flex: 2">参考答案</span>
                        <span v-if="expandedRunType === 'ragas'" style="flex: 2">生成答案</span>
                        <span v-for="k in metricKeys" :key="k" style="width: 80px; text-align: center">{{ k }}</span>
                      </div>
                      <div v-for="item in runItems" :key="item.id" class="items-row">
                        <span style="flex: 2" class="item-text">{{ item.query }}</span>
                        <span style="flex: 2" class="item-text">{{ item.gold_answer || '-' }}</span>
                        <span v-if="expandedRunType === 'ragas'" style="flex: 2" class="item-text">{{ item.generated_answer || '-' }}</span>
                        <span
                          v-for="k in metricKeys" :key="k"
                          style="width: 80px; text-align: center; font-size: 12px"
                          :class="metricColor(item.metrics?.[k])"
                        >
                          {{ item.metrics?.[k] !== undefined ? (item.metrics[k] * 100).toFixed(0) + '%' : '-' }}
                        </span>
                      </div>
                    </div>
                    <a-empty v-else description="暂无逐条结果" style="padding: 16px 0" />
                  </a-spin>
                </div>
              </div>
            </div>
            <a-empty v-else description="暂无评估运行" style="padding: 40px 0" />
          </a-spin>
        </div>
      </div>
    </div>
  </app-layout>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  ArrowLeftOutlined, UploadOutlined, DeleteOutlined, ReloadOutlined
} from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import { useKnowledgeStore } from '@/stores/knowledge'
import {
  listDatasets, uploadDataset, deleteDataset,
  listEvalRuns, createEvalRun, createRagasRun, getEvalRun, deleteEvalRun
} from '@/apis/knowledge'

const route = useRoute()
const router = useRouter()
const kbStore = useKnowledgeStore()

const kbId = route.params.id
const kbName = computed(() => kbStore.current?.name || '知识库')

const datasets = ref([])
const datasetsLoading = ref(false)
const selectedDataset = ref(null)

const evalRuns = ref([])
const runsLoading = ref(false)
const startingRun = ref(null)

const expandedRun = ref(null)
const expandedRunType = ref('quick')
const runItems = ref([])
const itemsLoading = ref(false)

const metricKeys = computed(() => {
  const keys = new Set()
  runItems.value.forEach(i => Object.keys(i.metrics || {}).forEach(k => keys.add(k)))
  return [...keys]
})

let pollTimer = null

onMounted(async () => {
  await kbStore.fetchOne(kbId)
  await Promise.all([loadDatasets(), loadRuns()])
  startPoll()
})

onUnmounted(() => stopPoll())

function startPoll() {
  pollTimer = setInterval(() => {
    if (evalRuns.value.some(r => r.status === 'running')) loadRuns()
  }, 3000)
}

function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null } }

async function loadDatasets() {
  datasetsLoading.value = true
  try { datasets.value = await listDatasets(kbId) }
  finally { datasetsLoading.value = false }
}

async function loadRuns() {
  runsLoading.value = true
  try { evalRuns.value = await listEvalRuns(kbId) }
  finally { runsLoading.value = false }
}

async function handleDatasetUpload(file) {
  const name = file.name.replace(/\.jsonl$/, '')
  try {
    await uploadDataset(kbId, name, file)
    await loadDatasets()
    message.success('数据集上传成功')
  } catch (e) {
    message.error(e.message || '上传失败')
  }
  return false
}

async function removeDataset(id) {
  try {
    await deleteDataset(id)
    datasets.value = datasets.value.filter(d => d.id !== id)
    if (selectedDataset.value?.id === id) selectedDataset.value = null
    message.success('已删除')
  } catch (e) {
    message.error(e.message || '删除失败')
  }
}

async function startRun(ds) {
  startingRun.value = ds.id + ':quick'
  try {
    const run = await createEvalRun(kbId, {
      dataset_id: ds.id,
      name: `${ds.name} - Quick`,
      top_k: 5,
    })
    evalRuns.value.unshift(run)
    message.success('Quick 评估已启动')
  } catch (e) {
    message.error(e.message || '启动失败')
  } finally {
    startingRun.value = null
  }
}

async function startRagasRun(ds) {
  startingRun.value = ds.id + ':ragas'
  try {
    const run = await createRagasRun(kbId, {
      dataset_id: ds.id,
      name: `${ds.name} - RAGAS`,
      top_k: 5,
    })
    evalRuns.value.unshift(run)
    message.success('RAGAS 评估已启动')
  } catch (e) {
    message.error(e.message || '启动失败')
  } finally {
    startingRun.value = null
  }
}

async function toggleRun(run) {
  if (expandedRun.value === run.id) {
    expandedRun.value = null
    return
  }
  expandedRun.value = run.id
  expandedRunType.value = run.eval_type || 'quick'
  itemsLoading.value = true
  try {
    const detail = await getEvalRun(kbId, run.id)
    runItems.value = detail.items || []
  } finally {
    itemsLoading.value = false
  }
}

async function removeRun(run) {
  try {
    await deleteEvalRun(kbId, run.id)
    evalRuns.value = evalRuns.value.filter(r => r.id !== run.id)
    if (expandedRun.value === run.id) expandedRun.value = null
    message.success('已删除')
  } catch (e) {
    message.error(e.message || '删除失败')
  }
}

function runProgress(run) {
  if (!run.total_items) return 0
  return Math.round((run.completed_items / run.total_items) * 100)
}

function runStatusColor(s) {
  return { pending: 'default', running: 'blue', completed: 'green', failed: 'red' }[s] || 'default'
}

function runStatusLabel(s) {
  return { pending: '待运行', running: '运行中', completed: '完成', failed: '失败' }[s] || s
}

function metricColor(v) {
  if (v === undefined) return ''
  if (v >= 0.7) return 'metric-high'
  if (v >= 0.4) return 'metric-mid'
  return 'metric-low'
}

function fmtDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}
</script>

<style scoped>
.eval-page { padding: 32px; }
.page-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.header-left { display: flex; align-items: center; }
.header-left h2 { font-size: 22px; font-weight: 700; margin: 0; }

.eval-layout { display: grid; grid-template-columns: 300px 1fr; gap: 24px; align-items: start; }

.panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.panel-header h3 { font-size: 15px; font-weight: 700; margin: 0; }

/* Datasets */
.datasets-panel { background: #fff; border: 1px solid #f0f0f0; border-radius: 10px; padding: 16px; }

.dataset-list { display: flex; flex-direction: column; gap: 8px; }
.dataset-item {
  border: 1px solid #f0f0f0; border-radius: 8px; padding: 10px 12px;
  cursor: pointer; transition: all 0.15s;
}
.dataset-item:hover { border-color: #91caff; background: #f0f8ff; }
.dataset-item.active { border-color: #1677ff; background: #e6f4ff; }
.ds-name { font-size: 13px; font-weight: 600; margin-bottom: 2px; }
.ds-meta { font-size: 11px; color: #999; margin-bottom: 8px; }
.ds-actions { display: flex; align-items: center; gap: 6px; }

.jsonl-hint { font-size: 11px; color: #bbb; margin-top: 12px; line-height: 1.6; }
.jsonl-hint code { background: #f5f5f5; padding: 1px 4px; border-radius: 3px; font-size: 10px; }

/* Runs */
.runs-panel { background: #fff; border: 1px solid #f0f0f0; border-radius: 10px; padding: 16px; }

.runs-list { display: flex; flex-direction: column; gap: 10px; }

.run-card { border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden; }
.run-card.expanded { border-color: #91caff; }

.run-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 14px; cursor: pointer; transition: background 0.15s;
}
.run-header:hover { background: #fafafa; }
.run-left { display: flex; align-items: center; gap: 8px; }
.run-name { font-size: 13px; font-weight: 600; }
.run-right { display: flex; align-items: center; gap: 10px; }
.run-score { font-size: 13px; font-weight: 700; color: #1677ff; }

.run-metrics { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 14px 10px; }
.metric-chip { font-size: 11px; background: #f0f8ff; border: 1px solid #91caff; border-radius: 4px; padding: 2px 8px; color: #1677ff; }

.run-items { padding: 0 14px 12px; border-top: 1px solid #f0f0f0; }

.items-table { font-size: 12px; }
.items-header { display: flex; gap: 8px; padding: 8px 0; border-bottom: 1px solid #f0f0f0; font-weight: 600; color: #888; }
.items-row { display: flex; gap: 8px; padding: 8px 0; border-bottom: 1px solid #fafafa; align-items: flex-start; }
.item-text { overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }

.metric-high { color: #52c41a; font-weight: 600; }
.metric-mid  { color: #faad14; }
.metric-low  { color: #ff4d4f; }
</style>
