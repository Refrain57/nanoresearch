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
                    @click.stop="openRagasConfig(ds)"
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
            <div style="display: flex; gap: 8px; align-items: center">
              <a-button
                v-if="compareSelected.length === 2"
                type="primary"
                size="small"
                :loading="compareLoading"
                @click="openCompareModal"
              >对比选中</a-button>
              <a-button size="small" @click="loadRuns"><reload-outlined /> 刷新</a-button>
            </div>
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
                    <a-checkbox
                      :checked="!!compareSelected.find(r => r.id === run.id)"
                      :disabled="isCompareDisabled(run)"
                      @click.stop="toggleCompare(run)"
                    />
                    <a-popconfirm title="确定删除？" ok-type="danger" @confirm.stop="removeRun(run)">
                      <a-button size="small" type="text" danger @click.stop>
                        <delete-outlined />
                      </a-button>
                    </a-popconfirm>
                  </div>
                </div>

                <!-- Metrics summary（过滤私有字段） -->
                <div v-if="run.metrics && Object.keys(run.metrics).filter(k => !k.startsWith('_')).length" class="run-metrics">
                  <template v-for="(v, k) in run.metrics" :key="k">
                    <span v-if="!k.startsWith('_')" class="metric-chip">
                      {{ k }}: {{ (v * 100).toFixed(1) }}%
                    </span>
                  </template>
                </div>
                <!-- 私有指标行 -->
                <div v-if="run.metrics?._avg_hops !== undefined" class="private-metrics">
                  <span class="private-chip">平均跳数: {{ run.metrics._avg_hops?.toFixed(2) }}</span>
                </div>

                <!-- Expanded: item details -->
                <div v-if="expandedRun === run.id" class="run-items">
                  <a-spin :spinning="itemsLoading">
                    <template v-if="runItems.length">
                      <!-- question_type 筛选 Tab -->
                      <a-tabs v-model:active-key="filterType" size="small" style="margin-bottom: 8px">
                        <a-tab-pane key="all" tab="全部" />
                        <a-tab-pane key="single_hop" tab="单跳" />
                        <a-tab-pane key="multi_context" tab="多跳" />
                      </a-tabs>

                      <div class="items-table">
                        <div class="items-header">
                          <span style="width: 60px">题型</span>
                          <span style="flex: 2">问题</span>
                          <span style="flex: 2">参考答案</span>
                          <span v-if="expandedRunType === 'ragas' || expandedRunType === 'agent'" style="flex: 2">生成答案</span>
                          <span v-for="k in metricKeys" :key="k" style="width: 80px; text-align: center">{{ k }}</span>
                        </div>
                        <div v-for="item in filteredRunItems" :key="item.id" class="items-row">
                          <span style="width: 60px; font-size: 11px; color: #888">{{ item.question_type === 'multi_context' ? '多跳' : item.question_type === 'single_hop' ? '单跳' : '-' }}</span>
                          <span style="flex: 2" class="item-text">{{ item.query }}</span>
                          <span style="flex: 2" class="item-text">{{ item.gold_answer || '-' }}</span>
                          <span v-if="expandedRunType === 'ragas' || expandedRunType === 'agent'" style="flex: 2" class="item-text">{{ item.generated_answer || '-' }}</span>
                          <span
                            v-for="k in metricKeys" :key="k"
                            style="width: 80px; text-align: center; font-size: 12px"
                            :class="metricColor(item.metrics?.[k])"
                          >
                            {{ item.metrics?.[k] !== undefined ? (item.metrics[k] * 100).toFixed(0) + '%' : '-' }}
                          </span>
                        </div>
                      </div>

                      <!-- 交叉表：按题型分组指标均值 -->
                      <div v-if="crossTable.length > 1" class="cross-table-wrap">
                        <div class="cross-table-title">按题型指标均值</div>
                        <table class="cross-table">
                          <thead>
                            <tr>
                              <th>题型</th>
                              <th v-for="k in metricKeys" :key="k">{{ k }}</th>
                              <th>_hops</th>
                              <th>数量</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr v-for="row in crossTable" :key="row.type">
                              <td>{{ row.type === 'multi_context' ? '多跳' : row.type === 'single_hop' ? '单跳' : row.type }}</td>
                              <td v-for="k in metricKeys" :key="k" :class="metricColor(row.metrics[k])">
                                {{ row.metrics[k] !== undefined ? (row.metrics[k] * 100).toFixed(1) + '%' : '-' }}
                              </td>
                              <td>{{ row.avg_hops != null ? row.avg_hops.toFixed(2) : '-' }}</td>
                              <td>{{ row.count }}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </template>
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

    <!-- RAGAS 配置弹窗 -->
    <a-modal
      v-model:open="ragasConfigOpen"
      title="RAGAS 评估配置"
      ok-text="启动"
      cancel-text="取消"
      @ok="startRagasRun"
      width="420"
    >
      <a-form layout="vertical" style="margin-top: 16px">
        <a-form-item label="Generator 模型">
          <a-auto-complete
            v-model:value="ragasForm.generator_model"
            :options="settingsStore.allModelOptions"
            placeholder="留空使用默认（qwen-plus）"
            allow-clear
            style="width: 100%"
          />
          <div class="field-hint">用于根据检索结果生成答案</div>
        </a-form-item>
        <a-form-item label="Evaluator 模型">
          <a-auto-complete
            v-model:value="ragasForm.evaluator_model"
            :options="settingsStore.allModelOptions"
            placeholder="留空使用默认（qwen-max）"
            allow-clear
            style="width: 100%"
          />
          <div class="field-hint">用于评判答案质量（Faithfulness / AnswerRelevancy 等）</div>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- A/B 对比弹窗 -->
    <a-modal
      v-model:open="compareModalOpen"
      title="Run 对比"
      :footer="null"
      width="960"
    >
      <template v-if="compareRunData.run1 && compareRunData.run2">
        <!-- 顶部：两个 run 并排 -->
        <div class="compare-header">
          <div class="compare-run-info">
            <div class="crn">{{ compareRunData.run1.name }}</div>
            <div class="crs">{{ compareRunData.run1.overall_score != null ? (compareRunData.run1.overall_score * 100).toFixed(1) + '%' : 'N/A' }}</div>
            <a-tag :color="runStatusColor(compareRunData.run1.status)" size="small">{{ compareRunData.run1.eval_type }}</a-tag>
          </div>
          <div class="compare-vs">VS</div>
          <div class="compare-run-info">
            <div class="crn">{{ compareRunData.run2.name }}</div>
            <div class="crs">{{ compareRunData.run2.overall_score != null ? (compareRunData.run2.overall_score * 100).toFixed(1) + '%' : 'N/A' }}</div>
            <a-tag :color="runStatusColor(compareRunData.run2.status)" size="small">{{ compareRunData.run2.eval_type }}</a-tag>
          </div>
        </div>

        <!-- 聚合指标对比 -->
        <div class="compare-agg">
          <div v-for="k in compareMetricKeys" :key="k" class="agg-row">
            <span class="agg-key">{{ k }}</span>
            <span>{{ scoreDisplay(compareRunData.run1.metrics?.[k]) }}</span>
            <span :class="compareColor(compareRunData.run1.metrics?.[k], compareRunData.run2.metrics?.[k])">
              {{ compareDiff(compareRunData.run1.metrics?.[k], compareRunData.run2.metrics?.[k]) }}
            </span>
            <span>{{ scoreDisplay(compareRunData.run2.metrics?.[k]) }}</span>
          </div>
        </div>

        <!-- question_type 筛选 -->
        <a-tabs v-model:active-key="compareFilterType" size="small" style="margin-top: 8px">
          <a-tab-pane key="all" tab="全部" />
          <a-tab-pane key="single_hop" tab="单跳" />
          <a-tab-pane key="multi_context" tab="多跳" />
        </a-tabs>

        <!-- 逐题对比表 -->
        <div class="compare-items">
          <table class="ct-table">
            <thead>
              <tr>
                <th style="width: 60px">类型</th>
                <th style="min-width: 180px">问题</th>
                <th v-for="k in compareMetricKeys" :key="'r1-'+k" style="width: 70px">R1 {{ k }}</th>
                <th v-for="k in compareMetricKeys" :key="'r2-'+k" style="width: 70px">R2 {{ k }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="pair in alignedItems" :key="pair.query">
                <td style="font-size: 11px; color: #888">{{ pair.question_type === 'multi_context' ? '多跳' : pair.question_type === 'single_hop' ? '单跳' : '-' }}</td>
                <td class="ct-query">{{ pair.query }}</td>
                <td v-for="k in compareMetricKeys" :key="'r1v-'+k" :class="metricColor(pair.item1?.metrics?.[k])">
                  {{ pair.item1?.metrics?.[k] !== undefined ? (pair.item1.metrics[k] * 100).toFixed(0) + '%' : '-' }}
                </td>
                <td v-for="k in compareMetricKeys" :key="'r2v-'+k" :class="metricColor(pair.item2?.metrics?.[k])">
                  {{ pair.item2?.metrics?.[k] !== undefined ? (pair.item2.metrics[k] * 100).toFixed(0) + '%' : '-' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </a-modal>
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
import { useSettingsStore } from '@/stores/settings'
import {
  listDatasets, uploadDataset, deleteDataset,
  listEvalRuns, createEvalRun, createRagasRun, getEvalRun, deleteEvalRun
} from '@/apis/knowledge'

const route = useRoute()
const router = useRouter()
const kbStore = useKnowledgeStore()
const settingsStore = useSettingsStore()

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

const ragasConfigOpen = ref(false)
const ragasConfigDs = ref(null)
const ragasForm = ref({ generator_model: null, evaluator_model: null })

const metricKeys = computed(() => {
  const keys = new Set()
  runItems.value.forEach(i => Object.keys(i.metrics || {}).filter(k => !k.startsWith('_')).forEach(k => keys.add(k)))
  return [...keys]
})

const filterType = ref('all')
const filteredRunItems = computed(() =>
  filterType.value === 'all'
    ? runItems.value
    : runItems.value.filter(i => i.question_type === filterType.value)
)

const crossTable = computed(() => {
  const groups = {}
  runItems.value.forEach(item => {
    const t = item.question_type || 'unknown'
    if (!groups[t]) groups[t] = { type: t, count: 0, sums: {}, hopSum: 0 }
    groups[t].count++
    groups[t].hopSum += item.metrics?._hops ?? 0
    metricKeys.value.forEach(k => {
      if (item.metrics?.[k] !== undefined)
        groups[t].sums[k] = (groups[t].sums[k] ?? 0) + item.metrics[k]
    })
  })
  return Object.values(groups).map(g => ({
    type: g.type, count: g.count,
    avg_hops: g.hopSum / g.count,
    metrics: Object.fromEntries(
      metricKeys.value.map(k => [k, g.sums[k] !== undefined ? g.sums[k] / g.count : undefined])
    ),
  }))
})

const compareSelected = ref([])
const compareModalOpen = ref(false)
const compareRunData = ref({ run1: null, run2: null })
const compareLoading = ref(false)
const compareFilterType = ref('all')

const compareMetricKeys = computed(() => {
  const keys = new Set()
  const r1 = compareRunData.value.run1
  const r2 = compareRunData.value.run2
  if (r1?.metrics) Object.keys(r1.metrics).filter(k => !k.startsWith('_')).forEach(k => keys.add(k))
  if (r2?.metrics) Object.keys(r2.metrics).filter(k => !k.startsWith('_')).forEach(k => keys.add(k))
  return [...keys]
})

const alignedItems = computed(() => {
  const r1 = compareRunData.value.run1
  const r2 = compareRunData.value.run2
  if (!r1?.items || !r2?.items) return []
  const map2 = {}
  r2.items.forEach(i => { map2[i.query] = i })
  const all = r1.items.map(i => ({
    query: i.query,
    question_type: i.question_type || map2[i.query]?.question_type,
    item1: i,
    item2: map2[i.query] ?? null,
  }))
  if (compareFilterType.value === 'all') return all
  return all.filter(p => p.question_type === compareFilterType.value)
})

function toggleCompare(run) {
  const idx = compareSelected.value.findIndex(r => r.id === run.id)
  if (idx >= 0) compareSelected.value.splice(idx, 1)
  else if (compareSelected.value.length < 2) compareSelected.value.push(run)
}

function isCompareDisabled(run) {
  return compareSelected.value.length >= 2 && !compareSelected.value.find(r => r.id === run.id)
}

async function openCompareModal() {
  const [r1, r2] = compareSelected.value
  if (r1.dataset_id !== r2.dataset_id) {
    message.warning('两个 Run 来自不同数据集，对齐结果可能不完整')
  }
  compareLoading.value = true
  try {
    const [d1, d2] = await Promise.all([getEvalRun(kbId, r1.id), getEvalRun(kbId, r2.id)])
    compareRunData.value = { run1: d1, run2: d2 }
    compareFilterType.value = 'all'
    compareModalOpen.value = true
  } catch (e) {
    message.error(e.message || '加载失败')
  } finally {
    compareLoading.value = false
  }
}

const scoreDisplay = v => v != null ? (v * 100).toFixed(1) + '%' : 'N/A'
const compareDiff = (v1, v2) => (v1 == null || v2 == null) ? '-' : ((v1 - v2 >= 0 ? '+' : '') + ((v1 - v2) * 100).toFixed(1) + '%')
const compareColor = (v1, v2) => v1 == null || v2 == null ? '' : v1 > v2 ? 'compare-better' : v1 < v2 ? 'compare-worse' : 'compare-same'

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

function openRagasConfig(ds) {
  ragasConfigDs.value = ds
  ragasForm.value = { generator_model: null, evaluator_model: null }
  ragasConfigOpen.value = true
}

async function startRagasRun() {
  const ds = ragasConfigDs.value
  if (!ds) return
  ragasConfigOpen.value = false
  startingRun.value = ds.id + ':ragas'
  try {
    const run = await createRagasRun(kbId, {
      dataset_id: ds.id,
      name: `${ds.name} - RAGAS`,
      top_k: 5,
      generator_model: ragasForm.value.generator_model || undefined,
      evaluator_model: ragasForm.value.evaluator_model || undefined,
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
.field-hint { font-size: 11px; color: #aaa; margin-top: 4px; }

.private-metrics { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 14px 10px; }
.private-chip { font-size: 11px; background: #f9f0ff; border: 1px solid #d3adf7; border-radius: 4px; padding: 2px 8px; color: #722ed1; }

.cross-table-wrap { margin-top: 16px; }
.cross-table-title { font-size: 12px; font-weight: 600; color: #888; margin-bottom: 6px; }
.cross-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.cross-table th, .cross-table td { border: 1px solid #f0f0f0; padding: 5px 8px; text-align: center; }
.cross-table th { background: #fafafa; font-weight: 600; color: #666; }

.compare-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.compare-run-info { flex: 1; text-align: center; }
.crn { font-size: 14px; font-weight: 700; margin-bottom: 4px; }
.crs { font-size: 24px; font-weight: 700; color: #1677ff; margin-bottom: 6px; }
.compare-vs { font-size: 18px; font-weight: 700; color: #bbb; padding: 0 16px; }

.compare-agg { display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px; background: #fafafa; border-radius: 8px; padding: 12px; }
.agg-row { display: flex; align-items: center; gap: 12px; font-size: 13px; }
.agg-key { width: 140px; color: #888; font-size: 12px; flex-shrink: 0; }
.agg-row span { flex: 1; text-align: center; }

.compare-better { color: #52c41a; font-weight: 700; }
.compare-worse  { color: #ff4d4f; font-weight: 700; }
.compare-same   { color: #888; }

.compare-items { max-height: 400px; overflow-y: auto; margin-top: 8px; }
.ct-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.ct-table th, .ct-table td { border: 1px solid #f0f0f0; padding: 5px 8px; }
.ct-table th { background: #fafafa; font-weight: 600; color: #666; position: sticky; top: 0; }
.ct-query { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
