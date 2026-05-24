<template>
  <div class="message-list" ref="listRef">
    <div v-for="msg in messages" :key="msg.id" :class="['message', msg.role]">
      <!-- 头像 -->
      <div v-if="msg.role === 'assistant'" class="avatar agent-avatar" :style="agentAvatarStyle">
        {{ agentInitial }}
      </div>

      <div class="bubble">
        <span v-if="msg.role === 'user'">{{ msgText(msg) }}</span>
        <span v-else style="white-space: pre-wrap">{{ msgText(msg) }}</span>
      </div>

      <div v-if="msg.role === 'user'" class="avatar user-avatar">
        {{ userInitial }}
      </div>
    </div>

    <!-- 流式输出气泡 -->
    <div v-if="streamingText" class="message assistant">
      <div class="avatar agent-avatar" :style="agentAvatarStyle">{{ agentInitial }}</div>
      <div class="bubble streaming">
        <span style="white-space: pre-wrap">{{ streamingText }}</span>
        <span class="cursor">▌</span>
      </div>
    </div>

    <div v-if="toolHint" class="tool-hint">
      <loading-outlined spin /> {{ toolHint }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { LoadingOutlined } from '@ant-design/icons-vue'

const props = defineProps({
  messages: { type: Array, default: () => [] },
  streamingText: { type: String, default: '' },
  toolHint: { type: String, default: '' },
  agentName: { type: String, default: 'Agent' },
  userName: { type: String, default: 'U' },
})

const listRef = ref(null)

const agentInitial = computed(() => (props.agentName || 'A')[0].toUpperCase())
const userInitial  = computed(() => (props.userName  || 'U')[0].toUpperCase())

// 根据 agentName hash 生成固定颜色
const agentAvatarStyle = computed(() => {
  const colors = ['#1677ff','#52c41a','#faad14','#f5222d','#722ed1','#13c2c2','#eb2f96']
  let hash = 0
  for (const ch of props.agentName) hash = (hash * 31 + ch.charCodeAt(0)) & 0xffffffff
  return { background: colors[Math.abs(hash) % colors.length] }
})

function msgText(msg) {
  if (!msg.content) return ''
  if (typeof msg.content === 'string') return msg.content
  return msg.content.text || msg.content.content || JSON.stringify(msg.content)
}

watch(() => [props.messages.length, props.streamingText], async () => {
  await nextTick()
  if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight
})
</script>

<style scoped>
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.message { display: flex; align-items: flex-end; gap: 8px; }
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }

.avatar {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; color: #fff;
  flex-shrink: 0;
}
.agent-avatar { }
.user-avatar { background: #595959; }

.bubble {
  max-width: 68%;
  padding: 10px 16px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
}
.message.user .bubble { background: #1677ff; color: #fff; border-bottom-right-radius: 4px; }
.message.assistant .bubble { background: #f5f5f5; color: #333; border-bottom-left-radius: 4px; }
.bubble.streaming { background: #e6f4ff; }
.cursor { animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
.tool-hint {
  text-align: center;
  color: #999;
  font-size: 12px;
  padding: 4px 0;
}
</style>
