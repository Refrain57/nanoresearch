<template>
  <app-layout>
    <div class="conv-detail-page">
      <div class="page-header">
        <a-button type="text" @click="router.back()" class="back-btn">
          <arrow-left-outlined /> 返回
        </a-button>
        <h2>对话详情</h2>
        <span class="conv-id-hint">{{ route.params.id }}</span>
      </div>

      <a-spin :spinning="loading">
        <!-- 概览 -->
        <div class="overview-row" v-if="!loading">
          <div class="ov-card">
            <div class="ov-num">{{ runs.length }}</div>
            <div class="ov-label">总 Run 数</div>
          </div>
          <div class="ov-card">
            <div class="ov-num">{{ agentSummaries.length }}</div>
            <div class="ov-label">参与 Agent</div>
          </div>
          <div class="ov-card">
            <div class="ov-num">{{ totalToolCalls }}</div>
            <div class="ov-label">工具调用次数</div>
          </div>
          <div class="ov-card">
            <div class="ov-num" :class="overallRate >= 80 ? 'good' : overallRate >= 50 ? 'warn' : 'bad'">
              {{ overallRate }}%
            </div>
            <div class="ov-label">工具成功率</div>
          </div>
        </div>

        <!-- Agent 覆盖配置 -->
        <div class="override-card" v-if="!loading">
          <div class="override-header">
            <span class="override-title">对话参数覆盖</span>
            <a-button type="link" size="small" @click="openOverride">
              <edit-outlined /> 编辑
            </a-button>
          </div>
          <div class="override-body">
            <span v-if="overrideData.model" class="override-tag">模型：{{ overrideData.model }}</span>
            <span v-if="overrideData.max_iterations" class="override-tag">最大迭代：{{ overrideData.max_iterations }}</span>
            <span v-if="!overrideData.model && !overrideData.max_iterations" class="override-empty">使用 Agent 默认配置</span>
          </div>
        </div>

        <!-- 各 Agent 卡片 -->
        <div v-for="ag in agentSummaries" :key="ag.agentId" class="agent-block">
          <div class="agent-block-header">
            <div class="agent-block-title">
              <robot-outlined />
              <router-link v-if="ag.agentId" :to="`/agents/${ag.agentId}`" class="agent-name-link">
                {{ ag.agentName }}
              </router-link>
              <span v-else class="agent-name-plain">{{ ag.agentName }}</span>
            </div>
            <div class="agent-block-meta">
              <span>{{ ag.runCount }} 次运行</span>
              <span>{{ ag.toolCallCount }} 次工具调用</span>
              <a-tag :color="ag.successRate >= 80 ? 'green' : ag.successRate >= 50 ? 'orange' : 'red'" size="small">
                成功率 {{ ag.successRate }}%
              </a-tag>
            </div>
          </div>

          <!-- 该 agent 的工具调用统计表 -->
          <table v-if="ag.toolStats.length" class="tool-stats-table">
            <thead>
              <tr><th>工具</th><th>调用</th><th>成功</th><th>失败</th><th>成功率</th></tr>
            </thead>
            <tbody>
              <tr v-for="t in ag.toolStats" :key="t.name">
                <td class="tool-name">{{ t.name }}</td>
                <td>{{ t.total }}</td>
                <td class="ok">{{ t.success }}</td>
                <td class="err">{{ t.error }}</td>
                <td>
                  <div class="rate-wrap">
                    <div class="rate-bar" :style="{ width: t.rate + '%' }" />
                    <span>{{ t.rate }}%</span>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
          <div v-else class="no-tools">本 Agent 无工具调用记录</div>

          <!-- 该 agent 的 run 时间线 -->
          <a-collapse ghost class="run-collapse">
            <a-collapse-panel
              v-for="run in ag.runs"
              :key="run.id"
              :header="`Run · ${formatTime(run.started_at)}${run.duration_ms ? ' · ' + (run.duration_ms/1000).toFixed(1) + 's' : ''}`"
            >
              <template #extra>
                <a-tag :color="statusColor(run.status)" size="small">{{ statusLabel(run.status) }}</a-tag>
              </template>
              <run-timeline :run-id="run.id" />
            </a-collapse-panel>
          </a-collapse>
        </div>

        <a-empty v-if="!loading && runs.length === 0" description="暂无运行记录" style="margin-top: 60px" />
      </a-spin>
    </div>
  </app-layout>

  <a-modal
    v-model:open="overrideOpen"
    title="对话参数覆盖"
    ok-text="保存"
    cancel-text="取消"
    :confirm-loading="overrideSaving"
    @ok="saveOverride"
    width="400"
  >
    <a-form layout="vertical" style="margin-top: 16px">
      <a-form-item label="模型">
        <a-auto-complete
          v-model:value="overrideForm.model"
          :options="settingsStore.allModelOptions"
          placeholder="留空则使用 Agent 默认模型"
          allow-clear
          style="width: 100%"
        />
        <div class="field-hint">覆盖此对话的模型，留空恢复 Agent 默认</div>
      </a-form-item>
      <a-form-item label="最大迭代次数">
        <a-input-number
          v-model:value="overrideForm.max_iterations"
          :min="1"
          :max="200"
          placeholder="留空则使用 Agent 默认"
          style="width: 100%"
        />
        <div class="field-hint">覆盖此对话的最大 Agent 迭代次数</div>
      </a-form-item>
    </a-form>
  </a-modal>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { ArrowLeftOutlined, RobotOutlined, EditOutlined } from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import RunTimeline from '@/components/RunTimeline.vue'
import { getConversationRuns, getConversation, updateAgentOverride } from '@/apis/conversations'
import { getAgent } from '@/apis/agents'
import { useSettingsStore } from '@/stores/settings'

const route = useRoute()
const router = useRouter()
const settingsStore = useSettingsStore()

const runs = ref([])
const loading = ref(false)
const agentNameCache = ref({})
const overrideData = ref({})
const overrideOpen = ref(false)
const overrideSaving = ref(false)
const overrideForm = ref({ model: null, max_iterations: null })

function openOverride() {
  overrideForm.value = {
    model: overrideData.value.model || null,
    max_iterations: overrideData.value.max_iterations || null,
  }
  overrideOpen.value = true
}

async function saveOverride() {
  overrideSaving.value = true
  try {
    const res = await updateAgentOverride(route.params.id, {
      model: overrideForm.value.model || '',
      max_iterations: overrideForm.value.max_iterations || null,
    })
    overrideData.value = res.agent_override
    overrideOpen.value = false
    message.success('已保存')
  } catch (e) {
    message.error('保存失败：' + (e.message || '未知错误'))
  } finally {
    overrideSaving.value = false
  }
}

onMounted(async () => {
  loading.value = true
  try {
    const [runsData, conv] = await Promise.all([
      getConversationRuns(route.params.id),
      getConversation(route.params.id),
    ])
    runs.value = runsData
    overrideData.value = conv.agent_override || {}
    const ids = [...new Set(runsData.map(r => r.agent_id).filter(Boolean))]
    const entries = await Promise.all(ids.map(async id => {
      try { return [id, (await getAgent(id))?.name || id.slice(0, 8)] }
      catch { return [id, id.slice(0, 8)] }
    }))
    agentNameCache.value = Object.fromEntries(entries)
  } finally {
    loading.value = false
  }
})

const agentSummaries = computed(() => {
  const groups = {}
  for (const run of runs.value) {
    const key = run.agent_id || '__default__'
    if (!groups[key]) groups[key] = { agentId: run.agent_id, runs: [], toolCalls: [] }
    groups[key].runs.push(run)
    groups[key].toolCalls.push(...(run.tool_calls || []))
  }
  return Object.values(groups).map(g => {
    const toolMap = {}
    for (const tc of g.toolCalls) {
      if (!toolMap[tc.name]) toolMap[tc.name] = { name: tc.name, total: 0, success: 0, error: 0 }
      toolMap[tc.name].total++
      if (tc.status === 'success') toolMap[tc.name].success++
      else toolMap[tc.name].error++
    }
    const toolStats = Object.values(toolMap).map(t => ({
      ...t, rate: t.total ? Math.round(t.success / t.total * 100) : 0
    })).sort((a, b) => b.total - a.total)

    const total = g.toolCalls.length
    const ok = g.toolCalls.filter(tc => tc.status === 'success').length
    return {
      agentId: g.agentId,
      agentName: g.agentId ? (agentNameCache.value[g.agentId] || g.agentId.slice(0, 8)) : '默认 Agent',
      runCount: g.runs.length,
      toolCallCount: total,
      successRate: total ? Math.round(ok / total * 100) : 100,
      toolStats,
      runs: g.runs,
    }
  })
})

const totalToolCalls = computed(() => runs.value.reduce((s, r) => s + (r.tool_calls?.length || 0), 0))
const overallRate = computed(() => {
  const all = runs.value.flatMap(r => r.tool_calls || [])
  if (!all.length) return 100
  return Math.round(all.filter(tc => tc.status === 'success').length / all.length * 100)
})

const statusColor = s => ({ completed: 'green', failed: 'red', running: 'blue', pending: 'orange' }[s] || 'default')
const statusLabel = s => ({ completed: '完成', failed: '失败', running: '运行中', pending: '待运行' }[s] || s)
const formatTime = iso => iso ? new Date(iso).toLocaleString('zh-CN', { hour12: false }) : ''
</script>

<style scoped>
.conv-detail-page { padding: 32px; max-width: 900px; }
.page-header { display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }
.page-header h2 { margin: 0; font-size: 20px; font-weight: 700; }
.back-btn { color: #666; }
.conv-id-hint { font-size: 12px; color: #bbb; font-family: monospace; margin-left: auto; }

.overview-row { display: flex; gap: 16px; margin-bottom: 24px; }
.ov-card { flex: 1; background: #fafafa; border: 1px solid #f0f0f0; border-radius: 8px; padding: 16px; text-align: center; }
.ov-num { font-size: 28px; font-weight: 700; color: #1677ff; }
.ov-num.good { color: #52c41a; }
.ov-num.warn { color: #faad14; }
.ov-num.bad  { color: #f5222d; }
.ov-label { font-size: 12px; color: #999; margin-top: 4px; }

.agent-block { background: #fff; border: 1px solid #e8e8e8; border-radius: 10px; padding: 20px; margin-bottom: 16px; }
.agent-block-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px; }
.agent-block-title { display: flex; align-items: center; gap: 8px; font-size: 15px; font-weight: 600; }
.agent-name-link { color: #1677ff; text-decoration: none; }
.agent-name-link:hover { text-decoration: underline; }
.agent-name-plain { color: #333; }
.agent-block-meta { display: flex; align-items: center; gap: 12px; font-size: 13px; color: #888; }

.tool-stats-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 12px; }
.tool-stats-table th { text-align: left; padding: 5px 10px; color: #888; font-weight: 600; border-bottom: 1px solid #f0f0f0; }
.tool-stats-table td { padding: 7px 10px; border-bottom: 1px solid #fafafa; }
.tool-stats-table tr:last-child td { border-bottom: none; }
.tool-name { font-family: monospace; color: #1677ff; font-weight: 500; }
.ok  { color: #52c41a; }
.err { color: #f5222d; }
.rate-wrap { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #555; }
.rate-bar { height: 6px; border-radius: 3px; background: #52c41a; min-width: 2px; max-width: 80px; }
.no-tools { font-size: 13px; color: #bbb; padding: 8px 0 12px; }
.run-collapse { margin-top: 4px; }

.override-card {
  background: #fff; border: 1px solid #e8e8e8; border-radius: 10px;
  padding: 16px 20px; margin-bottom: 16px;
}
.override-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.override-title { font-size: 14px; font-weight: 600; color: #333; }
.override-body { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.override-tag { font-size: 12px; background: #e6f4ff; color: #1677ff; padding: 2px 10px; border-radius: 4px; }
.override-empty { font-size: 12px; color: #bbb; }
.field-hint { font-size: 11px; color: #aaa; margin-top: 4px; }
</style>
