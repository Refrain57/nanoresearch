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

      <!-- 右侧消息区 -->
      <div class="chat-main">
        <template v-if="chatStore.currentConvId">
          <message-list
            :messages="chatStore.messages"
            :streaming-text="chatStore.streamingText"
            :tool-hint="toolHint"
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
import { ref, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { DeleteOutlined, RobotOutlined, BarChartOutlined } from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import MessageList from '@/components/MessageList.vue'
import { useChatStore } from '@/stores/chat'
import { useRunStream } from '@/composables/useRunStream'

const route = useRoute()
const router = useRouter()
const chatStore = useChatStore()
const runStream = useRunStream()

const inputText = ref('')
const toolHint = ref('')
const lastRunId = ref('')

onMounted(async () => {
  await chatStore.fetchConversations()
  if (route.params.id) {
    await chatStore.selectConversation(route.params.id)
  }
})

watch(() => route.params.id, async (id) => {
  if (id) await chatStore.selectConversation(id)
})

async function handleNew() {
  const conv = await chatStore.newConversation()
  router.push(`/chat/${conv.id}`)
  await chatStore.selectConversation(conv.id)
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
.run-hint { padding: 4px 16px; border-top: 1px solid #f0f0f0; background: #fafafa; }
.run-link { font-size: 12px; color: #1677ff; display: flex; align-items: center; gap: 4px; width: fit-content; }
.run-link:hover { color: #0958d9; }
.input-area { display: flex; gap: 8px; padding: 16px; border-top: 1px solid #f0f0f0; }
.chat-input { flex: 1; }
.empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #ccc; gap: 12px; }
</style>
