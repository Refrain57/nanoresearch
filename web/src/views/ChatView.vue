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
      <p style="margin: 12px 0 8px; color: var(--nr-ink-2); font-size: 13px;">选择要使用的 Agent：</p>
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
            <span class="agent-pick-desc">{{ agent.description || '无描述' }}</span>
          </div>
          <a-select
            v-if="selectedAgentId === agent.id"
            v-model:value="agentModelOverrides[agent.id]"
            size="small"
            style="width: 160px"
            show-search
            allow-clear
            placeholder="默认模型"
            :options="settingsStore.allModelOptions"
            @click.stop
            @change="(v) => agentModelOverrides[agent.id] = v"
          />
          <check-circle-filled v-if="selectedAgentId === agent.id" class="agent-pick-check" />
        </div>
        <div
          :class="['agent-pick-item', { selected: selectedAgentId === null }]"
          @click="selectedAgentId = null"
        >
          <div class="agent-pick-avatar" style="background:var(--nr-ink-3)">?</div>
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
          <!-- 顶栏：agent 信息（可选）+ 详情/工作区按钮（始终显示） -->
          <div class="agent-bar">
            <template v-if="currentAgent">
              <div class="agent-bar-avatar" :style="agentAvatarStyle">
                {{ agentInitial }}
              </div>
              <div class="agent-bar-info">
                <span class="agent-bar-name">{{ currentAgent.name }}</span>
                <span class="agent-bar-model" @click="openModelPicker">
                  <a class="model-name-link">{{ effectiveModel || '选择模型' }}</a>
                  <span v-if="overrideModel" class="override-badge">覆盖</span>
                  <span v-if="getModelProvider(effectiveModel)" class="model-provider-tag">{{ getModelProvider(effectiveModel) }}</span>
                </span>
              </div>
              <div class="agent-bar-caps">
                <a-tag v-if="currentAgent.capabilities?.streaming"     color="green"  size="small">流式</a-tag>
                <a-tag v-if="currentAgent.capabilities?.web_search"    color="blue"   size="small">联网</a-tag>
                <a-tag v-if="currentAgent.capabilities?.deep_research" color="red"    size="small">深研</a-tag>
                <a-tag v-if="currentAgent.capabilities?.knowledge_base" color="purple" size="small">知识库</a-tag>
              </div>
            </template>
            <div v-else class="agent-bar-empty">无绑定 Agent</div>
            <a-button type="text" size="small" class="agent-bar-link" style="margin-left: auto" @click="showDetail = !showDetail">
              <bar-chart-outlined /> 详情
            </a-button>
            <a-tooltip :title="showWorkspace ? '关闭工作区' : '打开工作区'">
              <a-button type="text" size="small" class="agent-bar-edit" @click="showWorkspace = !showWorkspace">
                <folder-open-outlined />
              </a-button>
            </a-tooltip>
          </div>

          <message-list
            :messages="chatStore.messages"
            :streaming-text="chatStore.streamingText"
            :streaming="chatStore.streaming"
            :tool-hint="toolHint"
            :agent-name="currentAgent?.name || 'Agent'"
            :user-name="userStore.uid || 'U'"
          />
          <div class="input-area">
            <div class="input-toolbar">
              <a-segmented
                v-model:value="ragMode"
                :options="[
                  { label: 'RAG', value: 'simple' },
                  { label: '多跳 Agent', value: 'agentic' }
                ]"
                size="small"
                :disabled="chatStore.streaming"
              />
              <a-select
                v-if="ragMode === 'simple'"
                v-model:value="manualKbId"
                :options="kbOptions"
                placeholder="选择知识库"
                size="small"
                style="min-width: 140px; margin-left: 8px"
                allow-clear
              />
            </div>
            <div class="input-row">
              <a-tooltip
                :title="settingsStore.coverage.hasChat ? '' : '请到 Settings 添加 Chat provider'"
                :trigger="settingsStore.coverage.hasChat ? [] : ['hover', 'focus']"
              >
                <a-textarea
                  v-model:value="inputText"
                  :auto-size="{ minRows: 1, maxRows: 6 }"
                  :placeholder="settingsStore.coverage.hasChat ? '输入消息，Enter 发送，Alt+Enter 换行' : '请到 Settings 添加 Chat provider'"
                  :disabled="chatStore.streaming || !settingsStore.coverage.hasChat"
                  @keydown="handleKeydown"
                  class="chat-input"
                />
              </a-tooltip>
              <a-button
                type="primary"
                :loading="chatStore.streaming"
                :disabled="!inputText.trim() || !settingsStore.coverage.hasChat"
                @click="handleSend"
              >发送</a-button>
            </div>
          </div>
        </template>
        <div v-else class="empty-state">
          <robot-outlined style="font-size: 48px; color: var(--nr-ink-3)" />
          <p>选择或新建一个对话开始</p>
        </div>
      </div>

      <!-- 右侧详情面板 -->
      <transition name="workspace-slide">
        <div v-if="showDetail && chatStore.currentConvId" class="detail-panel">
          <div class="detail-panel-header">
            <span class="detail-panel-title">对话详情</span>
            <a-button type="text" size="small" @click="showDetail = false">✕</a-button>
          </div>
          <div class="detail-panel-body">
            <conversation-detail-panel
              ref="detailPanelRef"
              :conv-id="chatStore.currentConvId"
              @override-saved="onOverrideSaved"
            />
          </div>
        </div>
      </transition>

      <!-- 右侧工作区文件面板 -->
      <transition name="workspace-slide">
        <div v-if="showWorkspace" class="workspace-panel">
          <workspace-files />
        </div>
      </transition>
    </div>

    <!-- 模型快速切换弹窗 -->
    <a-modal
      v-model:open="modelPickerOpen"
      title="切换对话模型"
      @ok="saveModelPicker"
      ok-text="切换"
      cancel-text="取消"
      :confirm-loading="modelPickerSaving"
      width="400"
    >
      <a-form layout="vertical" style="margin-top: 16px">
        <a-form-item label="模型">
          <a-auto-complete
            v-model:value="modelPickerValue"
            :options="settingsStore.allModelOptions"
            placeholder="输入或选择模型名称"
            allow-clear
            style="width: 100%"
          />
          <div class="field-hint">留空则使用 Agent 默认模型</div>
        </a-form-item>
        <div v-if="modelPickerValue && getModelProvider(modelPickerValue)" style="font-size:12px;color:var(--nr-ink-2)">
          当前选择将通过 <strong>{{ getModelProvider(modelPickerValue) }}</strong> 供应商 API 调用
        </div>
      </a-form>
    </a-modal>
  </app-layout>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { DeleteOutlined, FolderOpenOutlined, RobotOutlined, BarChartOutlined, CheckCircleFilled } from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import MessageList from '@/components/MessageList.vue'
import WorkspaceFiles from '@/components/WorkspaceFiles.vue'
import ConversationDetailPanel from '@/components/ConversationDetailPanel.vue'
import { useChatStore } from '@/stores/chat'
import { useAgentStore } from '@/stores/agent'
import { useUserStore } from '@/stores/user'
import { useSettingsStore } from '@/stores/settings'
import { useKnowledgeStore } from '@/stores/knowledge'
import { useRunStream } from '@/composables/useRunStream'
import { updateAgentOverride } from '@/apis/conversations'

const route = useRoute()
const router = useRouter()
const chatStore = useChatStore()
const agentStore = useAgentStore()
const userStore = useUserStore()
const settingsStore = useSettingsStore()
const kbStore = useKnowledgeStore()
const runStream = useRunStream()

const inputText = ref('')
const toolHint = ref('')
const showWorkspace = ref(false)
const showDetail = ref(false)
const newConvOpen = ref(false)
const newConvLoading = ref(false)
const selectedAgentId = ref(null)
const agentModelOverrides = reactive({})
const detailPanelRef = ref(null)
const pendingToolCalls = ref([])
const pendingCitations = ref(null)
// convId → runId: tracks runs that were interrupted mid-stream by a conversation switch
const pendingRuns = reactive({})
const ragMode = ref('agentic')
const manualKbId = ref(null)
const modelPickerOpen = ref(false)
const modelPickerSaving = ref(false)
const modelPickerValue = ref('')

const currentConv = computed(() => chatStore.conversations.find(c => c.id === chatStore.currentConvId))
const overrideModel = computed(() => currentConv.value?.agent_override?.model || null)

const effectiveModel = computed(() => overrideModel.value || currentAgent.value?.model || settingsStore.baseModel || '')
function getModelProvider(model) {
  for (const p of settingsStore.providers) {
    if (p.models?.includes(model)) return p.name
  }
  return null
}

function onOverrideSaved(agentOverride) {
  const conv = chatStore.conversations.find(c => c.id === chatStore.currentConvId)
  if (conv) conv.agent_override = agentOverride
}

// 当前会话绑定的 Agent
const currentAgent = computed(() => {
  const conv = chatStore.conversations.find(c => c.id === chatStore.currentConvId)
  if (!conv?.agent_id) return null
  return agentStore.agents.find(a => a.id === conv.agent_id) || null
})

// KB ID extracted from agent's harness or tools_config for simple RAG mode
const currentKbId = computed(() => {
  // First check manual selection
  if (manualKbId.value) return manualKbId.value
  // Then try agent's harness or tools_config
  const agent = currentAgent.value
  if (agent?.harness?.kb_id) return agent.harness.kb_id
  const ragTool = (agent?.tools_config || []).find(t => t.kb_id || t.collection)
  if (ragTool) return ragTool.kb_id || null
  // Fallback: use first available KB when in agentic mode
  if (ragMode.value === 'agentic' && kbStore.kbs.length) return kbStore.kbs[0].id
  return null
})

const kbOptions = computed(() =>
  kbStore.kbs.map(k => ({ value: k.id, label: k.name }))
)

const agentInitial = computed(() => (currentAgent.value?.name || 'A')[0].toUpperCase())
const AVATAR_COLORS = ['#C15F3C','#5E7355','#9A7B2E','#B04434','#7C6A8E','#3F7A78','#A8566E']
function avatarStyle(name) {
  let hash = 0
  for (const ch of (name || '')) hash = (hash * 31 + ch.charCodeAt(0)) & 0xffffffff
  return { background: AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length] }
}
const agentAvatarStyle = computed(() => avatarStyle(currentAgent.value?.name || ''))

onMounted(async () => {
  await Promise.all([chatStore.fetchConversations(), agentStore.fetchList(), kbStore.fetchList()])
  if (route.params.id) await chatStore.selectConversation(route.params.id)
})
onUnmounted(() => runStream.stop())

watch(() => route.params.id, async (id) => {
  // handleSelect already calls selectConversation directly; only handle URL-driven navigation here
  console.log('[nav] route watcher id=', id, 'currentConvId=', chatStore.currentConvId)
  if (id && id !== chatStore.currentConvId) {
    console.log('[nav] route watcher triggering selectConversation for', id)
    runStream.stop()
    toolHint.value = ''
    pendingToolCalls.value = []
    await chatStore.selectConversation(id)
    if (pendingRuns[id]) {
      const { runId, dbMsgCount, userMsg } = pendingRuns[id]
      if (chatStore.messages.length <= dbMsgCount) {
        // DB hasn't grown yet — run still in progress, restore user msg and reconnect SSE
        if (userMsg) chatStore.messages.push({ ...userMsg, seq: chatStore.messages.length })
        chatStore.streaming = true
        await connectStream(runId, id)
      } else {
        // DB already has the messages saved — run completed, clean up
        delete pendingRuns[id]
      }
    }
  }
})

function handleNew() {
  const defaultId = agentStore.agents.find(a => a.is_default)?.id ?? (agentStore.agents[0]?.id ?? null)
  selectedAgentId.value = defaultId
  // Reset model overrides to each agent's default
  for (const a of agentStore.agents) {
    agentModelOverrides[a.id] = null
  }
  newConvOpen.value = true
}

async function confirmNew() {
  newConvLoading.value = true
  try {
    const agent = agentStore.agents.find(a => a.id === selectedAgentId.value)
    const overrideModel = selectedAgentId.value ? agentModelOverrides[selectedAgentId.value] : null
    const conv = await chatStore.newConversation(
      agent?.name || null,
      selectedAgentId.value || null,
      overrideModel || null,
    )
    newConvOpen.value = false
    router.push(`/chat/${conv.id}`)
    await chatStore.selectConversation(conv.id)
  } finally {
    newConvLoading.value = false
  }
}

function openModelPicker() {
  modelPickerValue.value = effectiveModel.value
  modelPickerOpen.value = true
}

async function saveModelPicker() {
  if (!chatStore.currentConvId) return
  modelPickerSaving.value = true
  try {
    const res = await updateAgentOverride(chatStore.currentConvId, { model: modelPickerValue.value || '' })
    const conv = chatStore.conversations.find(c => c.id === chatStore.currentConvId)
    if (conv) conv.agent_override = res.agent_override
    modelPickerOpen.value = false
    if (modelPickerValue.value) message.success(`模型已切换至 ${modelPickerValue.value}`)
  } catch (e) {
    message.error('切换模型失败：' + (e.message || '未知错误'))
  } finally {
    modelPickerSaving.value = false
  }
}

async function handleSelect(id) {
  if (id === chatStore.currentConvId) return
  console.log('[nav] handleSelect', id, '← from', chatStore.currentConvId)
  runStream.stop()
  toolHint.value = ''
  pendingToolCalls.value = []
  showDetail.value = false
  router.push(`/chat/${id}`)
  await chatStore.selectConversation(id)
  console.log('[nav] handleSelect done, currentConvId=', chatStore.currentConvId, 'messages.length=', chatStore.messages.length)
  // If this conversation had a run interrupted by a switch, reconnect to the SSE.
  // The backend will replay completed events or resume live streaming.
  if (pendingRuns[id]) {
    const { runId, dbMsgCount, userMsg } = pendingRuns[id]
    console.log('[nav] reconnecting to pending run', runId, 'dbMsgCount=', dbMsgCount, 'currentMsgs=', chatStore.messages.length)
    if (chatStore.messages.length <= dbMsgCount) {
      if (userMsg) chatStore.messages.push({ ...userMsg, seq: chatStore.messages.length })
      chatStore.streaming = true
      await connectStream(runId, id)
    } else {
      delete pendingRuns[id]
    }
  }
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.altKey && !e.shiftKey && !e.ctrlKey) {
    e.preventDefault()
    handleSend()
  } else if (e.key === 'Enter' && e.altKey) {
    inputText.value += '\n'
  }
}

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || chatStore.streaming) return

  // Simple RAG 需要 KB ID
  if (ragMode.value === 'simple' && !currentKbId.value) {
    if (kbStore.kbs.length) {
      manualKbId.value = kbStore.kbs[0].id
      message.info(`已自动选择知识库：${kbStore.kbs[0].name}`)
    } else {
      message.warning('当前无可用知识库，已切换为多跳 Agent 模式')
      ragMode.value = 'agentic'
    }
  }

  const userMsgEntry = { id: `u-${Date.now()}`, role: 'user', content: { text }, seq: chatStore.messages.length }
  chatStore.messages.push(userMsgEntry)
  inputText.value = ''
  toolHint.value = ''

  const convId = chatStore.currentConvId
  const run = await chatStore.sendMessage(text, ragMode.value, currentKbId.value)
  if (!run) return

  // dbMsgCount = messages already in DB before this run (exclude the user msg we just pushed)
  pendingRuns[convId] = { runId: run.run_id, dbMsgCount: chatStore.messages.length - 1, userMsg: userMsgEntry }
  await connectStream(run.run_id, convId)
}

// Shared SSE connection logic — used by handleSend and reconnect-on-return
async function connectStream(runId, convId) {
  pendingToolCalls.value = []
  pendingCitations.value = null
  let messageCompleted = false
  await runStream.start(runId, {
    onDelta: (chunk) => {
      if (chatStore.currentConvId !== convId) return
      chatStore.appendDelta(chunk)
    },
    onToolHint: (hint) => {
      if (chatStore.currentConvId !== convId) return
      toolHint.value = hint
    },
    onToolCall: (tc) => {
      if (chatStore.currentConvId !== convId) return
      pendingToolCalls.value.push(tc)
    },
    onCitations: (ev) => {
      if (chatStore.currentConvId !== convId) return
      pendingCitations.value = ev.items
    },
    onMessageComplete: () => {
      if (chatStore.currentConvId !== convId) return
      toolHint.value = ''
      chatStore.finalizeStream(pendingToolCalls.value, pendingCitations.value)
      pendingToolCalls.value = []
      pendingCitations.value = null
      messageCompleted = true
    },
    onSubagentResult: (event) => {
      if (chatStore.currentConvId !== convId) return
      chatStore.messages.push({
        id: `subagent-${Date.now()}`,
        role: 'assistant',
        content: { text: `**[${event.label}]**\n\n${event.content}` },
        seq: chatStore.messages.length,
      })
    },
    onEnd: async () => {
      delete pendingRuns[convId]
      if (chatStore.currentConvId !== convId) return
      toolHint.value = ''
      pendingToolCalls.value = []
      pendingCitations.value = null
      // Reload from DB — backend saves all messages before pushing run_end,
      // so DB is canonical at this point. This avoids any streaming-state artifacts
      // (wrong avatars, wrong ordering from multi-turn replays).
      await chatStore.selectConversation(convId)
      if (showDetail.value) detailPanelRef.value?.refresh()
    },
  })
}

</script>

<style scoped>
.chat-layout { display: flex; height: 100vh; overflow: hidden; }
.sidebar { width: 260px; border-right: 1px solid var(--nr-border); display: flex; flex-direction: column; background: var(--nr-rail); }
.sidebar-header { display: flex; align-items: center; justify-content: space-between; padding: 16px; border-bottom: 1px solid var(--nr-border); }
.sidebar-title { font-weight: 600; font-size: 15px; }
.conv-list { flex: 1; overflow-y: auto; }
.conv-item { padding: 12px 16px; cursor: pointer; position: relative; border-bottom: 1px solid var(--nr-border); }
.conv-item:hover, .conv-item.active { background: var(--nr-clay-soft); }
.conv-title { font-size: 14px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conv-preview { font-size: 12px; color: var(--nr-ink-3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }
.conv-delete { position: absolute; right: 12px; top: 50%; transform: translateY(-50%); color: var(--nr-ink-3); display: none; }
.conv-item:hover .conv-delete { display: block; color: var(--nr-danger); }
.empty-hint { text-align: center; color: var(--nr-ink-3); padding: 24px; font-size: 13px; }

.chat-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

/* Agent 信息栏 */
.agent-bar {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px; border-bottom: 1px solid var(--nr-border);
  background: var(--nr-rail); flex-shrink: 0;
}
.agent-bar-avatar {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: #fff; flex-shrink: 0;
}
.agent-bar-info { display: flex; flex-direction: column; line-height: 1.3; }
.agent-bar-name { font-size: 13px; font-weight: 600; color: var(--nr-ink); }
.agent-bar-model { font-size: 11px; color: var(--nr-ink-3); display: flex; align-items: center; gap: 4px; }
.agent-bar-caps { display: flex; gap: 4px; flex-wrap: wrap; flex: 1; }
.agent-bar-link { font-size: 12px; color: var(--nr-clay); display: flex; align-items: center; gap: 3px; white-space: nowrap; }
.agent-bar-link:hover { color: var(--nr-clay-hover); }
.agent-bar-edit { color: var(--nr-ink-3); padding: 0 4px; }
.agent-bar-edit:hover { color: var(--nr-clay); }
.override-badge { margin-left: 4px; font-size: 10px; color: var(--nr-gold); font-weight: 500; }
.model-provider-tag { margin-left: 4px; font-size: 10px; color: var(--nr-clay); background: var(--nr-clay-soft); padding: 0 4px; border-radius: 3px; }
.model-name-link { cursor: pointer; color: var(--nr-clay); text-decoration: none; border-bottom: 1px dashed var(--nr-clay-soft); }
.model-name-link:hover { color: var(--nr-clay-hover); border-bottom-color: var(--nr-clay); }
.field-hint { font-size: 11px; color: var(--nr-ink-3); margin-top: 4px; }

.run-hint { padding: 4px 16px; border-top: 1px solid var(--nr-border); background: var(--nr-rail); }
.run-link { font-size: 12px; color: var(--nr-clay); display: flex; align-items: center; gap: 4px; width: fit-content; }
.run-link:hover { color: var(--nr-clay-hover); }
.input-area { display: flex; flex-direction: column; gap: 8px; padding: 16px; border-top: 1px solid var(--nr-border); }
.input-row { display: flex; gap: 8px; }
.chat-input { flex: 1; }
.empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--nr-ink-3); gap: 12px; }

.workspace-panel {
  width: 240px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-left: 1px solid var(--nr-border);
}

.detail-panel {
  width: 420px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-left: 1px solid var(--nr-border);
  background: #fff;
}
.detail-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 1px solid var(--nr-border);
  flex-shrink: 0;
}
.detail-panel-title { font-size: 14px; font-weight: 600; color: var(--nr-ink); }
.detail-panel-body { flex: 1; overflow-y: auto; padding: 16px; }

.workspace-slide-enter-active,
.workspace-slide-leave-active { transition: width 0.2s ease, opacity 0.2s ease; }
.workspace-slide-enter-from,
.workspace-slide-leave-to { width: 0; opacity: 0; }

.agent-pick-list { display: flex; flex-direction: column; gap: 6px; max-height: 320px; overflow-y: auto; }
.agent-pick-item {
  display: flex; align-items: center; gap: 10px; padding: 10px 12px;
  border: 1px solid var(--nr-border); border-radius: 8px; cursor: pointer; transition: all 0.15s;
}
.agent-pick-item:hover { border-color: var(--nr-clay-soft); background: var(--nr-clay-tint); }
.agent-pick-item.selected { border-color: var(--nr-clay); background: var(--nr-clay-soft); }
.agent-pick-avatar {
  width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; color: #fff;
}
.agent-pick-info { display: flex; flex-direction: column; flex: 1; overflow: hidden; }
.agent-pick-name { font-size: 13px; font-weight: 600; color: var(--nr-ink); }
.agent-pick-desc { font-size: 12px; color: var(--nr-ink-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.agent-pick-check { color: var(--nr-clay); font-size: 16px; flex-shrink: 0; }
</style>
