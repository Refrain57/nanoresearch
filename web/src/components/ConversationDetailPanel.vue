<template>
  <div class="conv-detail-panel">
    <a-spin :spinning="loading" style="width:100%">
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
          <span v-if="overrideData.skills" class="override-tag">
            Skills：{{ overrideData.skills.length ? overrideData.skills.join(', ') : '全部禁用' }}
          </span>
          <span v-if="!overrideData.model && !overrideData.max_iterations && !overrideData.skills" class="override-empty">使用 Agent 默认配置</span>
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
      <a-form-item v-if="agentSkills.length" label="启用的 Skills">
        <div class="skill-override-list">
          <label
            v-for="s in agentSkills"
            :key="s.name"
            class="skill-override-item"
            :class="{ active: overrideForm.skills === null || overrideForm.skills.includes(s.name) }"
            @click="toggleSkillOverride(s.name)"
          >
            <span class="skill-override-name">{{ s.name }}</span>
            <check-outlined v-if="overrideForm.skills === null || overrideForm.skills.includes(s.name)" class="skill-check-icon" />
          </label>
        </div>
        <div class="field-hint">
          {{ overrideForm.skills === null ? '全选（使用 Agent 默认）' : `已选 ${overrideForm.skills.length} / ${agentSkills.length}` }}
          <a-button type="link" size="small" style="padding: 0; height: auto" @click="overrideForm.skills = null">重置为全部</a-button>
        </div>
      </a-form-item>
    </a-form>
  </a-modal>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { message } from 'ant-design-vue'
import { RobotOutlined, EditOutlined, CheckOutlined } from '@ant-design/icons-vue'
import RunTimeline from '@/components/RunTimeline.vue'
import { getConversationRuns, getConversation, updateAgentOverride } from '@/apis/conversations'
import { getAgent } from '@/apis/agents'
import { useSettingsStore } from '@/stores/settings'

const props = defineProps({
  convId: { type: String, required: true },
})

const emit = defineEmits(['override-saved'])

const settingsStore = useSettingsStore()

const runs = ref([])
const loading = ref(false)
const agentNameCache = ref({})
const overrideData = ref({})
const overrideOpen = ref(false)
const overrideSaving = ref(false)
const overrideForm = ref({ model: null, max_iterations: null, skills: null })
const agentSkills = ref([])

async function load(convId) {
  if (!convId) return
  loading.value = true
  runs.value = []
  agentNameCache.value = {}
  overrideData.value = {}
  agentSkills.value = []
  try {
    const [runsData, conv] = await Promise.all([
      getConversationRuns(convId),
      getConversation(convId),
    ])
    runs.value = runsData
    overrideData.value = conv.agent_override || {}

    // 收集所有需要查询的 agent id（run 里的 + conv 绑定的），一次性批量请求
    const runAgentIds = [...new Set(runsData.map(r => r.agent_id).filter(Boolean))]
    const allIds = conv.agent_id
      ? [...new Set([...runAgentIds, conv.agent_id])]
      : runAgentIds

    const agentMap = {}
    await Promise.all(allIds.map(async id => {
      try { agentMap[id] = await getAgent(id) }
      catch { agentMap[id] = null }
    }))

    agentNameCache.value = Object.fromEntries(
      runAgentIds.map(id => [id, agentMap[id]?.name || id.slice(0, 8)])
    )
    if (conv.agent_id && agentMap[conv.agent_id]) {
      agentSkills.value = (agentMap[conv.agent_id].skills || []).filter(s => s.enabled !== false)
    }
  } finally {
    loading.value = false
  }
}

watch(() => props.convId, (id) => load(id), { immediate: true })

defineExpose({ refresh: () => load(props.convId) })

function openOverride() {
  overrideForm.value = {
    model: overrideData.value.model || null,
    max_iterations: overrideData.value.max_iterations || null,
    skills: overrideData.value.skills ?? null,
  }
  overrideOpen.value = true
}

function toggleSkillOverride(name) {
  if (overrideForm.value.skills === null) {
    overrideForm.value.skills = agentSkills.value.map(s => s.name).filter(n => n !== name)
  } else {
    const idx = overrideForm.value.skills.indexOf(name)
    if (idx === -1) overrideForm.value.skills.push(name)
    else overrideForm.value.skills.splice(idx, 1)
    if (overrideForm.value.skills.length === agentSkills.value.length) {
      overrideForm.value.skills = null
    }
  }
}

async function saveOverride() {
  overrideSaving.value = true
  try {
    const res = await updateAgentOverride(props.convId, {
      model: overrideForm.value.model || '',
      max_iterations: overrideForm.value.max_iterations || null,
      skills: overrideForm.value.skills,
    })
    overrideData.value = res.agent_override
    overrideOpen.value = false
    message.success('已保存')
    emit('override-saved', res.agent_override)
  } catch (e) {
    message.error('保存失败：' + (e.message || '未知错误'))
  } finally {
    overrideSaving.value = false
  }
}

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
.conv-detail-panel { padding: 4px 0; }

.overview-row { display: flex; gap: 12px; margin-bottom: 20px; }
.ov-card { flex: 1; background: #fafafa; border: 1px solid #f0f0f0; border-radius: 8px; padding: 14px; text-align: center; }
.ov-num { font-size: 24px; font-weight: 700; color: #1677ff; }
.ov-num.good { color: #52c41a; }
.ov-num.warn { color: #faad14; }
.ov-num.bad  { color: #f5222d; }
.ov-label { font-size: 12px; color: #999; margin-top: 4px; }

.override-card { background: #fff; border: 1px solid #e8e8e8; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; }
.override-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.override-title { font-size: 14px; font-weight: 600; color: #333; }
.override-body { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.override-tag { font-size: 12px; background: #e6f4ff; color: #1677ff; padding: 2px 10px; border-radius: 4px; }
.override-empty { font-size: 12px; color: #bbb; }

.agent-block { background: #fff; border: 1px solid #e8e8e8; border-radius: 10px; padding: 18px; margin-bottom: 14px; }
.agent-block-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }
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

.field-hint { font-size: 11px; color: #aaa; margin-top: 4px; }
.skill-override-list { display: flex; flex-direction: column; gap: 4px; max-height: 220px; overflow-y: auto; }
.skill-override-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 10px; border: 1px solid #f0f0f0; border-radius: 6px;
  cursor: pointer; font-size: 13px; transition: all 0.15s;
}
.skill-override-item:hover { border-color: #91caff; background: #f0f8ff; }
.skill-override-item.active { border-color: #1677ff; background: #e6f4ff; }
.skill-override-name { flex: 1; }
.skill-check-icon { color: #1677ff; font-size: 12px; }
</style>
