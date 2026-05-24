<template>
  <app-layout>
    <div class="chat-layout">
      <!-- 左侧会话列表 -->
      <div class="sidebar">
        <div class="sidebar-header">
          <span class="sidebar-title">对话</span>
          <a-button type="primary" size="small" @click="handleNew">新建</a-button>
        </div>
        <div class="conv-list">
          <div
            v-for="conv in chatStore.conversations"
            :key="conv.id"
            :class="['conv-item', { active: conv.id === chatStore.currentConvId }]"
            @click="handleSelect(conv.id)"
          >
            <div class="conv-title">{{ conv.title || '新对话' }}</div>
            <div class="conv-preview">{{ conv.last_message_preview || '' }}</div>
            <delete-outlined
              class="conv-delete"
              @click.stop="chatStore.removeConversation(conv.id)"
            />
          </div>
          <div v-if="!chatStore.conversations.length" class="empty-hint">暂无对话</div>
        </div>
      </div>

    <!-- 新建对话：选择 Agent -->
    <a-modal
      v-model:open="newConvOpen"
      title="新建对话"
      @ok="confirmNew"
      ok-text="开始对话"
      cancel-text="取消"
      :confirm-loading="newConvLoading"
      width="480"
    >
      <p style="margin: 12px 0 8px; color: #666; font-size: 13px;">选择要使用的 Agent：</p>
      <div class="agent-pick-list">
        <div
          v-for="agent in agentStore.agents"
          :key="agent.id"
          :class="['agent-pick-item', { selected: selectedAgentId === agent.id }]"
          @click="selectedAgentId = agent.id"
        >
          <div class="agent-pick-avatar" :style="avatarStyle(agent.name)">{{ (agent.name || 'A')[0].toUpperCase() }}</div>
          <div class="agent-pick-info">
            <span class="agent-pick-name">{{ agent.name }}</span>
            <span class="agent-pick-desc">{{ agent.description || agent.model || '无描述' }}</span>
          </div>
          <check-circle-filled v-if="selectedAgentId === agent.id" class="agent-pick-check" />
        </div>
        <div
          :class="['agent-pick-item', { selected: selectedAgentId === null }]"
          @click="selectedAgentId = null"
        >
          <div class="agent-pick-avatar" style="background:#bbb">?</div>
          <div class="agent-pick-info">
            <span class="agent-pick-name">不绑定 Agent</span>
            <span class="agent-pick-desc">使用系统默认配置</span>
          </div>
          <check-circle-filled v-if="selectedAgentId === null" class="agent-pick-check" />
        </div>
      </div>
    </a-modal>

      <!-- 右侧消息区 -->
      <div class="chat-main">
        <template v-if="chatStore.currentConvId">
          <!-- Agent 信息栏 -->
          <div class="agent-bar" v-if="currentAgent">
            <div class="agent-bar-avatar" :style="agentAvatarStyle">
              {{ agentInitial }}
            </div>
            <div class="agent-bar-info">
              <span class="agent-bar-name">{{ currentAgent.name }}</span>
              <span class="agent-bar-model">{{ currentAgent.model || '未配置模型' }}</span>
            </div>
            <div class="agent-bar-caps">
              <a-tag v-if="currentAgent.capabilities?.streaming"     color="green"  size="small">流式</a-tag>
              <a-tag v-if="currentAgent.capabilities?.web_search"    color="blue"   size="small">联网</a-tag>
              <a-tag v-if="currentAgent.capabilities?.deep_research" color="red"    size="small">深研</a-tag>
              <a-tag v-if="currentAgent.capabilities?.knowledge_base" color="purple" size="small">知识库</a-tag>
            </div>
            <router-link :to="`/agents/${currentAgent.id}`" class="agent-bar-link">
              <info-circle-outlined /> 详情
            </router-link>
          </div>

          <message-list
            :messages="chatStore.messages"
            :streaming-text="chatStore.streamingText"
            :tool-hint="toolHint"
            :agent-name="currentAgent?.name || 'Agent'"
            :user-name="userStore.uid || 'U'"
          />
          <div class="run-hint" v-if="lastRunId && !chatStore.streaming">
            <router-link :to="`/runs/${lastRunId}`" class="run-link">
              <bar-chart-outlined /> 查看运行详情
            </router-link>
          </div>
          <div class="input-area">
            <a-textarea
              v-model:value="inputText"
              :auto-size="{ minRows: 1, maxRows: 6 }"
              placeholder="输入消息，Ctrl+Enter 发送"
              :disabled="chatStore.streaming"
              @keydown.ctrl.enter="handleSend"
              class="chat-input"
            />
            <a-button
              type="primary"
              :loading="chatStore.streaming"
              :disabled="!inputText.trim()"
              @click="handleSend"
            >发送</a-button>
          </div>
        </template>
        <div v-else class="empty-state">
          <robot-outlined style="font-size: 48px; color: #ccc" />
          <p>选择或新建一个对话开始</p>
        </div>
      </div>
    </div>
  </app-layout>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { DeleteOutlined, RobotOutlined, BarChartOutlined, InfoCircleOutlined, CheckCircleFilled } from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import MessageList from '@/components/MessageList.vue'
import { useChatStore } from '@/stores/chat'
import { useAgentStore } from '@/stores/agent'
import { useUserStore } from '@/stores/user'
import { useRunStream } from '@/composables/useRunStream'

const route = useRoute()
const router = useRouter()
const chatStore = useChatStore()
const agentStore = useAgentStore()
const userStore = useUserStore()
const runStream = useRunStream()

const inputText = ref('')
const toolHint = ref('')
const lastRunId = ref('')
const newConvOpen = ref(false)
const newConvLoading = ref(false)
const selectedAgentId = ref(null)

// 当前会话绑定的 Agent
const currentAgent = computed(() => {
  const conv = chatStore.conversations.find(c => c.id === chatStore.currentConvId)
  if (!conv?.agent_id) return null
  return agentStore.agents.find(a => a.id === conv.agent_id) || null
})

const agentInitial = computed(() => (currentAgent.value?.name || 'A')[0].toUpperCase())
const AVATAR_COLORS = ['#1677ff','#52c41a','#faad14','#f5222d','#722ed1','#13c2c2','#eb2f96']
function avatarStyle(name) {
  let hash = 0
  for (const ch of (name || '')) hash = (hash * 31 + ch.charCodeAt(0)) & 0xffffffff
  return { background: AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length] }
}
const agentAvatarStyle = computed(() => avatarStyle(currentAgent.value?.name || ''))

onMounted(async () => {
  await Promise.all([chatStore.fetchConversations(), agentStore.fetchList()])
  if (route.params.id) await chatStore.selectConversation(route.params.id)
})

watch(() => route.params.id, async (id) => {
  if (id) await chatStore.selectConversation(id)
})

function handleNew() {
  selectedAgentId.value = agentStore.agents.find(a => a.is_default)?.id ?? (agentStore.agents[0]?.id ?? null)
  newConvOpen.value = true
}

async function confirmNew() {
  newConvLoading.value = true
  try {
    const agent = agentStore.agents.find(a => a.id === selectedAgentId.value)
    const conv = await chatStore.newConversation(agent?.name || null, selectedAgentId.value || null)
    newConvOpen.value = false
    router.push(`/chat/${conv.id}`)
    await chatStore.selectConversation(conv.id)
  } finally {
    newConvLoading.value = false
  }
}

async function handleSelect(id) {
  router.push(`/chat/${id}`)
  await chatStore.selectConversation(id)
}

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || chatStore.streaming) return
  chatStore.messages.push({ id: `u-${Date.now()}`, role: 'user', content: { text }, seq: chatStore.messages.length })
  inputText.value = ''
  toolHint.value = ''

  const run = await chatStore.sendMessage(text)
  if (!run) return

  lastRunId.value = run.run_id
  await runStream.start(run.run_id, {
    onDelta: (chunk) => chatStore.appendDelta(chunk),
    onToolHint: (hint) => { toolHint.value = hint },
    onEnd: () => { toolHint.value = ''; chatStore.finalizeStream() }
  })
}
</script>

<style scoped>
.chat-layout { display: flex; height: 100vh; }
.sidebar { width: 260px; border-right: 1px solid #f0f0f0; display: flex; flex-direction: column; background: #fafafa; }
.sidebar-header { display: flex; align-items: center; justify-content: space-between; padding: 16px; border-bottom: 1px solid #f0f0f0; }
.sidebar-title { font-weight: 600; font-size: 15px; }
.conv-list { flex: 1; overflow-y: auto; }
.conv-item { padding: 12px 16px; cursor: pointer; position: relative; border-bottom: 1px solid #f0f0f0; }
.conv-item:hover, .conv-item.active { background: #e6f4ff; }
.conv-title { font-size: 14px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conv-preview { font-size: 12px; color: #999; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }
.conv-delete { position: absolute; right: 12px; top: 50%; transform: translateY(-50%); color: #ccc; display: none; }
.conv-item:hover .conv-delete { display: block; color: #ff4d4f; }
.empty-hint { text-align: center; color: #ccc; padding: 24px; font-size: 13px; }

.chat-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

/* Agent 信息栏 */
.agent-bar {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px; border-bottom: 1px solid #f0f0f0;
  background: #fafafa; flex-shrink: 0;
}
.agent-bar-avatar {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: #fff; flex-shrink: 0;
}
.agent-bar-info { display: flex; flex-direction: column; line-height: 1.3; }
.agent-bar-name { font-size: 13px; font-weight: 600; color: #333; }
.agent-bar-model { font-size: 11px; color: #999; }
.agent-bar-caps { display: flex; gap: 4px; flex-wrap: wrap; flex: 1; }
.agent-bar-link { font-size: 12px; color: #1677ff; display: flex; align-items: center; gap: 3px; white-space: nowrap; }
.agent-bar-link:hover { color: #0958d9; }

.run-hint { padding: 4px 16px; border-top: 1px solid #f0f0f0; background: #fafafa; }
.run-link { font-size: 12px; color: #1677ff; display: flex; align-items: center; gap: 4px; width: fit-content; }
.run-link:hover { color: #0958d9; }
.input-area { display: flex; gap: 8px; padding: 16px; border-top: 1px solid #f0f0f0; }
.chat-input { flex: 1; }
.empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #ccc; gap: 12px; }

.agent-pick-list { display: flex; flex-direction: column; gap: 6px; max-height: 320px; overflow-y: auto; }
.agent-pick-item {
  display: flex; align-items: center; gap: 10px; padding: 10px 12px;
  border: 1px solid #f0f0f0; border-radius: 8px; cursor: pointer; transition: all 0.15s;
}
.agent-pick-item:hover { border-color: #91caff; background: #f0f8ff; }
.agent-pick-item.selected { border-color: #1677ff; background: #e6f4ff; }
.agent-pick-avatar {
  width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; color: #fff;
}
.agent-pick-info { display: flex; flex-direction: column; flex: 1; overflow: hidden; }
.agent-pick-name { font-size: 13px; font-weight: 600; color: #333; }
.agent-pick-desc { font-size: 12px; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.agent-pick-check { color: #1677ff; font-size: 16px; flex-shrink: 0; }
</style>
