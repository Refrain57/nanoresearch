<template>
  <app-layout>
    <div class="agents-page">
      <div class="page-header">
        <h2 class="nr-serif">Agent 画廊</h2>
        <a-button type="primary" @click="openCreate">
          <plus-outlined /> 新建 Agent
        </a-button>
      </div>

      <a-spin :spinning="agentStore.loading">
        <div v-if="agentStore.agents.length" class="agents-grid">
          <div v-for="agent in agentStore.agents" :key="agent.id" class="agent-card-wrap">
            <agent-card :agent="agent" @chat="handleChat" @detail="handleDetail" />
            <a-popconfirm
              v-if="!agent.is_default"
              title="确定删除此 Agent？"
              ok-text="删除"
              cancel-text="取消"
              ok-type="danger"
              @confirm="handleDelete(agent)"
            >
              <a-button class="delete-btn" size="small" danger type="text">
                <delete-outlined />
              </a-button>
            </a-popconfirm>
          </div>
        </div>
        <a-empty v-else description="暂无 Agent" style="margin-top: 80px" />
      </a-spin>

      <!-- 系统 Skills -->
      <div class="section-header">
        <h3>系统可用 Skills</h3>
        <span class="skill-count">{{ agentStore.skills.length }} 个</span>
      </div>
      <div class="skills-grid">
        <div v-for="skill in agentStore.skills" :key="skill.name" class="skill-card">
          <div class="skill-name">{{ skill.name }}</div>
          <div class="skill-desc">{{ skill.description }}</div>
          <a-tag size="small" :color="skill.source === 'builtin' ? 'blue' : 'green'">
            {{ skill.source === 'builtin' ? '内置' : '自定义' }}
          </a-tag>
        </div>
      </div>
    </div>

    <!-- 新建 Agent Modal（两步：基本信息 + 选择 Skill） -->
    <a-modal
      v-model:open="createOpen"
      title="新建 Agent"
      @ok="handleCreate"
      :confirm-loading="creating"
      width="560"
    >
      <a-form :model="createForm" layout="vertical" style="margin-top: 16px">
        <a-form-item label="名称" required>
          <a-input v-model:value="createForm.name" placeholder="例：Research Agent" />
        </a-form-item>
        <a-form-item label="描述">
          <a-textarea v-model:value="createForm.description" :rows="2" />
        </a-form-item>
        <a-form-item label="默认模型">
          <a-auto-complete
            v-model:value="createForm.default_model"
            :options="settingsStore.allModelOptions"
            placeholder="例：anthropic/claude-opus-4-5"
            allow-clear
            style="width: 100%"
          />
        </a-form-item>
        <a-form-item label="最大迭代次数">
          <a-input-number v-model:value="createForm.max_iterations" :min="1" :max="100" style="width: 100%" />
        </a-form-item>
        <a-form-item label="初始 Skills（可选）">
          <div class="skill-check-list">
            <label
              v-for="s in agentStore.skills"
              :key="s.name"
              class="skill-check-item"
              :class="{ selected: createForm.selectedSkills.includes(s.name) }"
              @click="toggleCreateSkill(s.name)"
            >
              <span class="skill-check-name">{{ s.name }}</span>
              <span class="skill-check-desc">{{ s.description }}</span>
              <check-outlined v-if="createForm.selectedSkills.includes(s.name)" class="skill-check-icon" />
            </label>
          </div>
        </a-form-item>
      </a-form>
    </a-modal>
  </app-layout>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { PlusOutlined, DeleteOutlined, CheckOutlined } from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import AgentCard from '@/components/AgentCard.vue'
import { useAgentStore } from '@/stores/agent'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'

const router = useRouter()
const agentStore = useAgentStore()
const chatStore = useChatStore()
const settingsStore = useSettingsStore()

const createOpen = ref(false)
const creating = ref(false)
const createForm = reactive({
  name: '', description: '', default_model: '', max_iterations: 40, selectedSkills: []
})

onMounted(() => {
  agentStore.fetchList()
  agentStore.fetchSkills()
  settingsStore.fetchAll()
})

function openCreate() {
  Object.assign(createForm, { name: '', description: '', default_model: '', max_iterations: 40, selectedSkills: [] })
  createOpen.value = true
}

function toggleCreateSkill(name) {
  const idx = createForm.selectedSkills.indexOf(name)
  if (idx === -1) createForm.selectedSkills.push(name)
  else createForm.selectedSkills.splice(idx, 1)
}

async function handleChat(agent) {
  const conv = await chatStore.newConversation(agent.name, agent.id)
  router.push(`/chat/${conv.id}`)
}

function handleDetail(agent) {
  router.push(`/agents/${agent.id}`)
}

async function handleDelete(agent) {
  try {
    await agentStore.remove(agent.id)
    message.success('已删除')
  } catch (e) {
    message.error(e.message || '删除失败')
  }
}

async function handleCreate() {
  if (!createForm.name.trim()) { message.warning('请填写名称'); return }
  creating.value = true
  try {
    const skillsConfig = createForm.selectedSkills.map(name => {
      const s = agentStore.skills.find(x => x.name === name)
      return { id: name, name, description: s?.description || '', tags: [], enabled: true }
    })
    const agent = await agentStore.create({
      name: createForm.name,
      description: createForm.description,
      default_model: createForm.default_model || null,
      max_iterations: createForm.max_iterations,
    })
    // 如果选了 skill，立即更新
    if (skillsConfig.length) {
      await agentStore.update(agent.id, { skills_config: skillsConfig })
    }
    // 确保 current 被设置，避免详情页空白
    await agentStore.fetchOne(agent.id)
    createOpen.value = false
    message.success('创建成功')
    router.push(`/agents/${agent.id}`)
  } catch (e) {
    message.error(e.message || '创建失败')
  } finally {
    creating.value = false
  }
}
</script>

<style scoped>
.agents-page { padding: 32px; }
.page-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.page-header h2 { font-size: 22px; font-weight: 700; margin: 0; }
.agents-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
  margin-bottom: 40px;
}
.agent-card-wrap { position: relative; }
.delete-btn { position: absolute; top: 8px; right: 8px; z-index: 1; opacity: 0; transition: opacity 0.15s; }
.agent-card-wrap:hover .delete-btn { opacity: 1; }

.section-header { display: flex; align-items: center; gap: 10px; margin: 32px 0 16px; border-top: 1px solid var(--nr-border); padding-top: 32px; }
.section-header h3 { font-size: 16px; font-weight: 700; margin: 0; }
.skill-count { font-size: 13px; color: var(--nr-ink-3); }
.skills-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.skill-card { background: var(--nr-rail); border: 1px solid var(--nr-border); border-radius: 8px; padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; }
.skill-name { font-size: 13px; font-weight: 600; color: var(--nr-clay); }
.skill-desc { font-size: 12px; color: var(--nr-ink-2); line-height: 1.5; flex: 1; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

.skill-check-list { display: flex; flex-direction: column; gap: 6px; max-height: 280px; overflow-y: auto; }
.skill-check-item {
  display: flex; align-items: center; gap: 8px; padding: 8px 12px;
  border: 1px solid var(--nr-border); border-radius: 6px; cursor: pointer;
  transition: all 0.15s;
}
.skill-check-item:hover { border-color: var(--nr-clay-soft); background: var(--nr-clay-tint); }
.skill-check-item.selected { border-color: var(--nr-clay); background: var(--nr-clay-soft); }
.skill-check-name { font-size: 13px; font-weight: 600; color: var(--nr-ink); white-space: nowrap; }
.skill-check-desc { font-size: 12px; color: var(--nr-ink-2); flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.skill-check-icon { color: var(--nr-clay); flex-shrink: 0; }
</style>
