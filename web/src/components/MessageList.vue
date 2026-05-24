<template>
  <div class="message-list" ref="listRef">
    <div v-for="msg in messages" :key="msg.id" :class="['message', msg.role]">
      <div class="bubble">
        <span v-if="msg.role === 'user'">{{ msgText(msg) }}</span>
        <span v-else style="white-space: pre-wrap">{{ msgText(msg) }}</span>
      </div>
    </div>

    <!-- 流式输出气泡 -->
    <div v-if="streamingText" class="message assistant">
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
import { ref, watch, nextTick } from 'vue'
import { LoadingOutlined } from '@ant-design/icons-vue'

const props = defineProps({
  messages: { type: Array, default: () => [] },
  streamingText: { type: String, default: '' },
  toolHint: { type: String, default: '' }
})

const listRef = ref(null)

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
.message { display: flex; }
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }
.bubble {
  max-width: 72%;
  padding: 10px 16px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
}
.message.user .bubble { background: #1677ff; color: #fff; }
.message.assistant .bubble { background: #f5f5f5; color: #333; }
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
