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
          <a-card title="Skills" :bordered="false" class="section-card" v-if="agent.skills?.length">
            <div v-for="skill in agent.skills" :key="skill.id" class="skill-row">
              <div>
                <span class="skill-name">{{ skill.name }}</span>
                <span v-if="skill.description" class="skill-desc"> — {{ skill.description }}</span>
                <div class="skill-tags">
                  <a-tag v-for="tag in (skill.tags || [])" :key="tag" size="small">{{ tag }}</a-tag>
                </div>
              </div>
              <a-switch :checked="skill.enabled" disabled size="small" />
            </div>
          </a-card>

          <!-- Tools -->
          <a-card title="工具" :bordered="false" class="section-card" v-if="agent.tools?.length">
            <div v-for="tool in agent.tools" :key="tool.name" class="tool-row">
              <span>{{ tool.name }}</span>
              <a-switch :checked="tool.enabled" disabled size="small" />
            </div>
          </a-card>
        </template>
      </a-spin>

      <!-- 编辑 Modal -->
      <a-modal v-model:open="editModalOpen" title="编辑 Agent 配置" @ok="handleUpdate" :confirm-loading="saving">
        <a-form :model="editForm" layout="vertical" style="margin-top: 16px">
          <a-form-item label="名称"><a-input v-model:value="editForm.name" /></a-form-item>
          <a-form-item label="描述"><a-textarea v-model:value="editForm.description" :rows="3" /></a-form-item>
          <a-form-item label="默认模型"><a-input v-model:value="editForm.default_model" /></a-form-item>
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
import AppLayout from '@/layouts/AppLayout.vue'
import { useAgentStore } from '@/stores/agent'
import { useChatStore } from '@/stores/chat'

const route = useRoute()
const router = useRouter()
const agentStore = useAgentStore()
const chatStore = useChatStore()

const editModalOpen = ref(false)
const saving = ref(false)
const editForm = reactive({ name: '', description: '', default_model: '', max_iterations: 40 })

const agent = computed(() => agentStore.current)
const caps = computed(() => agent.value?.capabilities || {})

onMounted(() => agentStore.fetchOne(route.params.id))

watch(agent, (a) => {
  if (a) {
    editForm.name = a.name
    editForm.description = a.description || ''
    editForm.default_model = a.model || ''
    editForm.max_iterations = a.max_iterations || 40
  }
})

async function handleChat() {
  const conv = await chatStore.newConversation(agent.value?.name, agent.value?.id)
  router.push(`/chat/${conv.id}`)
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
</style>
