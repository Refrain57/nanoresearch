<template>
  <app-layout>
    <div class="agent-detail-page">
      <a-spin :spinning="agentStore.loading">
<template v-if="agent">
          <!-- 头部 -->
          <div class="detail-header">
            <div>
              <h2>{{ agent.name }} <a-tag v-if="agent.is_default" color="blue">默认</a-tag></h2>
              <p class="desc">{{ agent.description || '暂无描述' }}</p>
              <div class="meta">v{{ agent.version }} · {{ agent.provider || '未知 provider' }} · {{ agent.model || '未设置模型' }}</div>
            </div>
            <div class="header-actions">
              <a-button @click="handleChat">开始对话</a-button>
              <a-button type="primary" @click="editModalOpen = true">编辑配置</a-button>
            </div>
          </div>

          <!-- 能力 + 统计 -->
          <div class="info-row">
            <a-card title="能力" :bordered="false" class="info-card">
              <div class="capabilities">
                <a-tag v-if="caps.streaming" color="green">流式输出</a-tag>
                <a-tag v-if="caps.web_search" color="blue">联网搜索</a-tag>
                <a-tag v-if="caps.code_execution" color="orange">代码执行</a-tag>
                <a-tag v-if="caps.knowledge_base" color="purple">知识库</a-tag>
                <a-tag v-if="caps.deep_research" color="red">深度研究</a-tag>
              </div>
            </a-card>

            <a-card title="统计" :bordered="false" class="info-card">
              <div class="stats-row">
                <div class="stat"><div class="num">{{ agent.stats?.total_conversations ?? 0 }}</div><div class="label">对话数</div></div>
                <div class="stat"><div class="num">{{ agent.stats?.total_runs ?? 0 }}</div><div class="label">运行数</div></div>
                <div class="stat">
                  <div class="num">{{ agent.stats?.avg_run_duration_ms ? (agent.stats.avg_run_duration_ms / 1000).toFixed(1) + 's' : '-' }}</div>
                  <div class="label">平均耗时</div>
                </div>
              </div>
            </a-card>
          </div>

          <!-- Skills -->
          <a-card :bordered="false" class="section-card">
            <template #title>
              <div class="card-title-row">
                <span>Skills</span>
                <a-button size="small" @click="addSkillOpen = true">
                  <plus-outlined /> 添加
                </a-button>
              </div>
            </template>
            <div v-if="agent.skills?.length">
              <div v-for="skill in agent.skills" :key="skill.id" class="skill-row">
                <div>
                  <span class="skill-name">{{ skill.name }}</span>
                  <span v-if="skill.description" class="skill-desc"> — {{ skill.description }}</span>
                  <div class="skill-tags">
                    <a-tag v-for="tag in (skill.tags || [])" :key="tag" size="small">{{ tag }}</a-tag>
                  </div>
                </div>
                <a-switch :checked="skill.enabled" size="small" @change="(v) => toggleSkill(skill.id, v)" />
              </div>
            </div>
            <a-empty v-else description="暂未配置 Skill" />
          </a-card>

          <!-- Knowledge Bases -->
          <a-card :bordered="false" class="section-card">
            <template #title>
              <div class="card-title-row">
                <span>知识库 <span class="kb-hint">（绑定后 Agent 在对话中可检索这些知识库）</span></span>
                <a-button size="small" @click="addKBOpen = true">
                  <plus-outlined /> 绑定
                </a-button>
              </div>
            </template>
            <div v-if="boundKBs.length">
              <div v-for="kb in boundKBs" :key="kb.id" class="skill-row">
                <div>
                  <span class="skill-name">{{ kb.name }}</span>
                  <span v-if="kb.description" class="skill-desc"> — {{ kb.description }}</span>
                </div>
                <a-popconfirm title="确定解绑这个知识库？" @confirm="handleUnbindKB(kb.id)">
                  <a-button size="small" danger type="link">解绑</a-button>
                </a-popconfirm>
              </div>
            </div>
            <a-empty v-else description="暂未绑定知识库" />
          </a-card>

          <!-- KB 添加 Modal -->
          <a-modal v-model:open="addKBOpen" title="绑定知识库" :footer="null" width="560">
            <div v-if="kbsToAdd.length" class="add-skill-list">
              <div
                v-for="kb in kbsToAdd"
                :key="kb.id"
                class="add-skill-row"
                @click="handleBindKB(kb.id)"
              >
                <div>
                  <div class="add-skill-name">{{ kb.name }}</div>
                  <div v-if="kb.description" class="add-skill-desc">{{ kb.description }}</div>
                </div>
              </div>
            </div>
            <a-empty v-else description="所有知识库已绑定" />
          </a-modal>

          <!-- Tools -->
          <a-card title="工具" :bordered="false" class="section-card" v-if="agent.tools?.length">
            <div v-for="tool in agent.tools" :key="tool.name" class="tool-row">
              <span>{{ tool.name }}</span>
              <a-switch :checked="tool.enabled" disabled size="small" />
            </div>
          </a-card>

          <!-- 工具调用统计 -->
          <a-card :bordered="false" class="section-card">
            <template #title>
              <div class="card-title-row">
                <span>工具调用统计</span>
                <a-button size="small" @click="loadToolStats">刷新</a-button>
              </div>
            </template>
            <a-spin :spinning="toolStatsLoading">
              <table v-if="toolStats.length" class="stats-table">
                <thead>
                  <tr><th>工具</th><th>调用次数</th><th>成功</th><th>失败</th><th>成功率</th></tr>
                </thead>
                <tbody>
                  <tr v-for="s in toolStats" :key="s.tool_name">
                    <td class="tool-name-cell">{{ s.tool_name }}</td>
                    <td>{{ s.total }}</td>
                    <td class="success-cell">{{ s.success }}</td>
                    <td class="error-cell">{{ s.error }}</td>
                    <td>
                      <div class="rate-bar-wrap">
                        <div class="rate-bar" :style="{ width: s.success_rate + '%' }" />
                        <span class="rate-text">{{ s.success_rate }}%</span>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
              <a-empty v-else description="暂无工具调用记录" />
            </a-spin>
          </a-card>

          <!-- Harness -->
          <a-card :bordered="false" class="section-card">
            <template #title>
              <div class="card-title-row">
                <span>Harness</span>
                <div style="display:flex;gap:6px">
                  <a-button size="small" :loading="promptPreviewLoading" @click="loadPromptPreview">预览 Prompt</a-button>
                  <a-button size="small" type="primary" :loading="harnessLoading" @click="saveHarness">保存</a-button>
                </div>
              </div>
            </template>

            <!-- 角色设定 -->
            <div class="harness-section">
              <div class="harness-label">角色设定 <span class="harness-hint">（替换默认身份，定义此 Agent 的专长与行为）</span></div>
              <a-textarea
                v-model:value="harnessForm.persona"
                :rows="6"
                placeholder="例：你是一个专注于代码审查的 Agent，精通 Python 和 TypeScript。你的职责是..."
                class="harness-textarea"
              />
            </div>

            <!-- 上下文 & 记忆参数 -->
            <div class="harness-section">
              <div class="harness-label">上下文 & 记忆参数</div>
              <div class="harness-grid">
                <div class="harness-field">
                  <div class="harness-field-label">Memory Token 预算 <span class="harness-hint">（注入系统提示的记忆+知识总 token 上限，默认 3000）</span></div>
                  <a-input-number v-model:value="harnessForm.total_token_budget" :min="500" :max="20000" :step="500" style="width:160px" />
                </div>
                <div class="harness-field">
                  <div class="harness-field-label">记忆占比 <span class="harness-hint">（记忆 vs 知识的分配比，默认 0.6）</span></div>
                  <a-input-number v-model:value="harnessForm.memory_budget_ratio" :min="0.1" :max="0.9" :step="0.1" :precision="1" style="width:120px" />
                </div>
                <div class="harness-field">
                  <div class="harness-field-label">上下文窗口覆盖 <span class="harness-hint">（token，留空使用模型默认值）</span></div>
                  <a-input-number v-model:value="harnessForm.context_window_tokens" :min="8192" :max="200000" :step="8192" :placeholder="'模型默认'" style="width:160px" allow-clear />
                </div>
                <div class="harness-field">
                  <div class="harness-field-label">压缩目标比例 <span class="harness-hint">（压缩后保留上下文窗口的几成，默认 0.5）</span></div>
                  <a-input-number v-model:value="harnessForm.compression_target_ratio" :min="0.2" :max="0.8" :step="0.1" :precision="1" style="width:120px" />
                </div>
              </div>
            </div>
          </a-card>

          <!-- Prompt 预览 Modal -->
          <a-modal v-model:open="promptPreviewOpen" title="System Prompt 预览" width="720" :footer="null">
            <pre class="prompt-preview-text">{{ promptPreviewText }}</pre>
          </a-modal>
        </template>
      </a-spin>

      <!-- 添加 Skill Modal -->
      <a-modal v-model:open="addSkillOpen" title="添加 Skill" :footer="null" width="560">
        <div v-if="skillsToAdd.length" class="add-skill-list">
          <div
            v-for="s in skillsToAdd"
            :key="s.name"
            class="add-skill-row"
            @click="addSkill(s)"
          >
            <div>
              <div class="add-skill-name">{{ s.name }}</div>
              <div class="add-skill-desc">{{ s.description }}</div>
            </div>
            <a-tag size="small" :color="s.source === 'builtin' ? 'blue' : 'green'">
              {{ s.source === 'builtin' ? '内置' : '自定义' }}
            </a-tag>
          </div>
        </div>
        <a-empty v-else description="所有可用 Skill 已添加" />
      </a-modal>

      <!-- 编辑 Modal -->
      <a-modal v-model:open="editModalOpen" title="编辑 Agent 配置" @ok="handleUpdate" :confirm-loading="saving">
        <a-form :model="editForm" layout="vertical" style="margin-top: 16px">
          <a-form-item label="名称"><a-input v-model:value="editForm.name" /></a-form-item>
          <a-form-item label="描述"><a-textarea v-model:value="editForm.description" :rows="3" /></a-form-item>
          <a-form-item label="默认模型">
            <a-auto-complete
              v-model:value="editForm.default_model"
              :options="settingsStore.allModelOptions"
              placeholder="例：anthropic/claude-opus-4-5"
              allow-clear
              style="width: 100%"
            />
          </a-form-item>
          <a-form-item label="最大迭代次数"><a-input-number v-model:value="editForm.max_iterations" :min="1" :max="100" /></a-form-item>
        </a-form>
      </a-modal>
    </div>
  </app-layout>
</template>

<script setup>
import { ref, computed, reactive, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { PlusOutlined } from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import { useAgentStore } from '@/stores/agent'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'
import { getAgentToolStats, getAgentPromptPreview, listBoundKBs, bindKB, unbindKB } from '@/apis/agents'
import { useKnowledgeStore } from '@/stores/knowledge'

const route = useRoute()
const router = useRouter()
const agentStore = useAgentStore()
const chatStore = useChatStore()
const settingsStore = useSettingsStore()
const kbStore = useKnowledgeStore()

const toolStats = ref([])
const toolStatsLoading = ref(false)

const editModalOpen = ref(false)
const saving = ref(false)
const editForm = reactive({ name: '', description: '', default_model: '', max_iterations: 40 })

// Harness state
const harnessLoading = ref(false)
const harnessForm = reactive({
  persona: '',
  total_token_budget: 3000,
  memory_budget_ratio: 0.6,
  context_window_tokens: null,
  compression_target_ratio: 0.5,
})
const promptPreviewOpen = ref(false)
const promptPreviewLoading = ref(false)
const promptPreviewText = ref('')

const addSkillOpen = ref(false)
const skillsToAdd = computed(() =>
  agentStore.skills.filter(s => !agent.value?.skills?.some(as => as.id === s.name))
)

// KB bindings
const boundKBs = ref([])
const bindKBSaving = ref(false)
const addKBOpen = ref(false)
const kbsToAdd = computed(() =>
  kbStore.kbs.filter(kb => !boundKBs.value.some(b => b.id === kb.id))
)

const agent = computed(() => agentStore.current)
const caps = computed(() => agent.value?.capabilities || {})

onMounted(async () => {
  try {
    await agentStore.fetchOne(route.params.id)
    if (!agentStore.skills.length) await agentStore.fetchSkills()
    if (!settingsStore.providers.length) settingsStore.fetchAll()
    if (!kbStore.kbs.length) await kbStore.fetchList()
    await loadToolStats()
    await loadBoundKBs()
  } catch (e) {
    console.error('[AgentDetail] mount error:', e)
    message.error('加载 Agent 详情失败: ' + (e.message || e))
  }
})

watch(agent, (a) => {
  if (a) {
    editForm.name = a.name
    editForm.description = a.description || ''
    editForm.default_model = a.model || ''
    editForm.max_iterations = a.max_iterations || 40
    // Sync harness form
    harnessForm.persona = a.persona || ''
    const h = a.harness || {}
    harnessForm.total_token_budget = h.total_token_budget ?? 3000
    harnessForm.memory_budget_ratio = h.memory_budget_ratio ?? 0.6
    harnessForm.context_window_tokens = h.context_window_tokens ?? null
    harnessForm.compression_target_ratio = h.compression_target_ratio ?? 0.5
  }
})

async function loadToolStats() {
  toolStatsLoading.value = true
  try { toolStats.value = await getAgentToolStats(route.params.id) }
  catch (e) { message.error('加载工具统计失败') }
  finally { toolStatsLoading.value = false }
}

async function handleChat() {
  const conv = await chatStore.newConversation(agent.value?.name, agent.value?.id)
  router.push(`/chat/${conv.id}`)
}

async function toggleSkill(skillId, enabled) {
  try {
    const newSkills = agent.value.skills.map(s => s.id === skillId ? { ...s, enabled } : s)
    await agentStore.update(route.params.id, { skills_config: newSkills })
  } catch (e) {
    message.error(e.message || '更新失败')
  }
}

async function addSkill(s) {
  try {
    const newSkills = [
      ...(agent.value.skills || []),
      { id: s.name, name: s.name, description: s.description, tags: [], enabled: true }
    ]
    await agentStore.update(route.params.id, { skills_config: newSkills })
    addSkillOpen.value = false
    message.success(`已添加 ${s.name}`)
  } catch (e) {
    message.error(e.message || '添加失败')
  }
}

async function handleUpdate() {
  saving.value = true
  try {
    await agentStore.update(route.params.id, { ...editForm })
    editModalOpen.value = false
    message.success('更新成功')
  } catch (e) {
    message.error(e.message || '更新失败')
  } finally {
    saving.value = false
  }
}

async function saveHarness() {
  harnessLoading.value = true
  try {
    const harness = {
      total_token_budget: harnessForm.total_token_budget,
      memory_budget_ratio: harnessForm.memory_budget_ratio,
      compression_target_ratio: harnessForm.compression_target_ratio,
    }
    if (harnessForm.context_window_tokens) {
      harness.context_window_tokens = harnessForm.context_window_tokens
    }
    await agentStore.update(route.params.id, {
      persona: harnessForm.persona,
      harness,
    })
    message.success('Harness 配置已保存')
  } catch (e) {
    message.error(e.message || '保存失败')
  } finally {
    harnessLoading.value = false
  }
}

async function loadBoundKBs() {
  try {
    boundKBs.value = await listBoundKBs(route.params.id)
  } catch (e) {
    message.error('加载知识库绑定失败')
  }
}

async function handleBindKB(kbId) {
  bindKBSaving.value = true
  try {
    await bindKB(route.params.id, kbId)
    await loadBoundKBs()
    addKBOpen.value = false
    message.success('绑定成功')
  } catch (e) {
    message.error(e.message || '绑定失败')
  } finally {
    bindKBSaving.value = false
  }
}

async function handleUnbindKB(kbId) {
  try {
    await unbindKB(route.params.id, kbId)
    await loadBoundKBs()
    message.success('已解绑')
  } catch (e) {
    message.error(e.message || '解绑失败')
  }
}

async function loadPromptPreview() {
  promptPreviewLoading.value = true
  try {
    const data = await getAgentPromptPreview(route.params.id)
    promptPreviewText.value = data.static_prompt
    promptPreviewOpen.value = true
  } catch (e) {
    message.error('加载失败：' + (e.message || e))
  } finally {
    promptPreviewLoading.value = false
  }
}
</script>

<style scoped>
.agent-detail-page { padding: 32px; max-width: 900px; }
.detail-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; }
.detail-header h2 { font-size: 24px; font-weight: 700; margin: 0 0 4px; }
.desc { color: #666; margin: 4px 0; }
.meta { color: #999; font-size: 13px; }
.header-actions { display: flex; gap: 8px; }
.info-row { display: flex; gap: 16px; margin-bottom: 16px; }
.info-card { flex: 1; background: #fafafa; border-radius: 8px; }
.capabilities { display: flex; flex-wrap: wrap; gap: 8px; }
.stats-row { display: flex; gap: 32px; }
.stat { text-align: center; }
.num { font-size: 24px; font-weight: 700; color: #1677ff; }
.label { font-size: 12px; color: #999; }
.section-card { background: #fafafa; border-radius: 8px; margin-bottom: 16px; }
.skill-row, .tool-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }
.skill-row:last-child, .tool-row:last-child { border-bottom: none; }
.skill-name { font-weight: 500; }
.skill-desc { color: #888; font-size: 13px; }
.skill-tags { margin-top: 4px; }
.card-title-row { display: flex; align-items: center; justify-content: space-between; }
.stats-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.stats-table th { text-align: left; padding: 6px 10px; color: #888; font-weight: 600; border-bottom: 1px solid #f0f0f0; }
.stats-table td { padding: 8px 10px; border-bottom: 1px solid #f9f9f9; }
.stats-table tr:last-child td { border-bottom: none; }
.tool-name-cell { font-family: monospace; color: #1677ff; font-weight: 500; }
.success-cell { color: #52c41a; }
.error-cell { color: #f5222d; }
.rate-bar-wrap { display: flex; align-items: center; gap: 8px; }
.rate-bar { height: 6px; border-radius: 3px; background: #52c41a; min-width: 2px; max-width: 80px; }
.rate-text { font-size: 12px; color: #555; }
.run-dur { font-size: 12px; color: #aaa; margin-left: 8px; }
.add-skill-list { display: flex; flex-direction: column; gap: 4px; max-height: 400px; overflow-y: auto; margin-top: 8px; }
.add-skill-row { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-radius: 6px; cursor: pointer; border: 1px solid #f0f0f0; }
.add-skill-row:hover { background: #e6f4ff; border-color: #91caff; }
.add-skill-name { font-size: 13px; font-weight: 600; color: #1677ff; }
.add-skill-desc { font-size: 12px; color: #888; margin-top: 2px; }

.harness-section { margin-bottom: 20px; }
.harness-section:last-child { margin-bottom: 0; }
.harness-label { font-size: 13px; font-weight: 600; color: #333; margin-bottom: 8px; }
.harness-hint { font-size: 12px; color: #aaa; font-weight: 400; }
.harness-textarea { font-family: monospace; font-size: 13px; }
.harness-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.harness-field { display: flex; flex-direction: column; gap: 6px; }
.harness-field-label { font-size: 12px; color: #555; }
.prompt-preview-text {
  font-family: monospace; font-size: 12px; white-space: pre-wrap; word-break: break-word;
  background: #f8f8f8; border-radius: 6px; padding: 12px; max-height: 60vh; overflow-y: auto;
  border: 1px solid #f0f0f0; margin: 0;
}
.kb-hint { font-size: 12px; color: #aaa; font-weight: 400; }
</style>
