<template>
  <div class="message-list" ref="listRef">
    <div v-for="msg in messages" :key="msg.id" :class="['message-wrap', msg.role]">
      <!-- skip raw tool result messages — they're shown inside the tool-calls panel -->
      <template v-if="msg.role !== 'tool'">
      <div :class="['message', msg.role]">
        <!-- 头像 -->
        <div v-if="msg.role === 'assistant'" class="avatar agent-avatar" :style="agentAvatarStyle">
          {{ agentInitial }}
        </div>

        <div v-if="msg.role !== 'assistant' || msgText(msg)" class="bubble">
          <span v-if="msg.role === 'user'">{{ msgText(msg) }}</span>
          <div v-else class="md-body" v-html="renderMd(msgText(msg))" />
        </div>

        <div v-if="msg.role === 'user'" class="avatar user-avatar">
          {{ userInitial }}
        </div>
      </div>

      <!-- 工具调用折叠面板（仅 assistant 消息） -->
      <div v-if="msg.role === 'assistant' && msg.toolCalls?.length" class="tool-calls-panel">
        <a-collapse size="small" :bordered="false">
          <a-collapse-panel
            v-for="(tc, idx) in msg.toolCalls"
            :key="idx"
            :header="`🔧 ${tc.name}${tc.status === 'error' ? ' ❌' : ''}`"
          >
            <div class="tc-input"><b>输入：</b>{{ JSON.stringify(tc.input, null, 2) }}</div>
            <div class="tc-output"><b>输出：</b>{{ tc.output_summary }}</div>
          </a-collapse-panel>
        </a-collapse>
      </div>
      </template>
    </div>

    <!-- 流式输出气泡：streaming=true 时即使还没文本也显示占位光标 -->
    <div v-if="streamingText || props.streaming" class="message assistant">
      <div class="avatar agent-avatar" :style="agentAvatarStyle">{{ agentInitial }}</div>
      <div class="bubble streaming">
        <div v-if="streamingText" class="md-body" v-html="renderMd(streamingText)" />
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
import { marked } from 'marked'

marked.setOptions({ breaks: true, gfm: true })

function renderMd(text) {
  if (!text) return ''
  return marked.parse(text)
}

const props = defineProps({
  messages: { type: Array, default: () => [] },
  streamingText: { type: String, default: '' },
  streaming: { type: Boolean, default: false },
  toolHint: { type: String, default: '' },
  agentName: { type: String, default: 'Agent' },
  userName: { type: String, default: 'U' },
})

const listRef = ref(null)

const agentInitial = computed(() => (props.agentName || 'A')[0].toUpperCase())
const userInitial  = computed(() => (props.userName  || 'U')[0].toUpperCase())

// 根据 agentName hash 生成固定颜色
const agentAvatarStyle = computed(() => {
  const colors = ['#C15F3C','#5E7355','#9A7B2E','#B04434','#7C6A8E','#3F7A78','#A8566E']
  let hash = 0
  for (const ch of props.agentName) hash = (hash * 31 + ch.charCodeAt(0)) & 0xffffffff
  return { background: colors[Math.abs(hash) % colors.length] }
})

function msgText(msg) {
  if (!msg.content) return ''
  if (typeof msg.content === 'string') return msg.content
  return msg.content.text || msg.content.content || ''
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
.message-wrap { display: flex; flex-direction: column; gap: 4px; }
.message { display: flex; align-items: flex-end; gap: 8px; }
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }
.tool-calls-panel { margin-left: 40px; max-width: 68%; }
.tc-input, .tc-output { font-size: 12px; white-space: pre-wrap; word-break: break-all; color: var(--nr-ink-2); margin-bottom: 4px; }

.avatar {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; color: #fff;
  flex-shrink: 0;
}
.agent-avatar { }
.user-avatar { background: #736C5E; }

.bubble {
  max-width: 68%;
  padding: 10px 16px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
}
.message.user .bubble { background: var(--nr-clay); color: #fff; border-bottom-right-radius: 4px; }
.message.assistant .bubble { background: var(--nr-card); color: var(--nr-ink); border: 1px solid var(--nr-border); border-bottom-left-radius: 4px; }
.bubble.streaming { background: var(--nr-clay-soft); }
.cursor { animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
.tool-hint {
  text-align: center;
  color: var(--nr-ink-3);
  font-size: 12px;
  padding: 4px 0;
}

.md-body { line-height: 1.7; }
.md-body :deep(p) { margin: 0 0 8px; }
.md-body :deep(p:last-child) { margin-bottom: 0; }
.md-body :deep(h1),:deep(h2),:deep(h3),:deep(h4) { margin: 12px 0 6px; font-weight: 700; line-height: 1.3; }
.md-body :deep(h1) { font-size: 1.25em; }
.md-body :deep(h2) { font-size: 1.1em; }
.md-body :deep(h3) { font-size: 1em; }
.md-body :deep(ul),:deep(ol) { margin: 6px 0; padding-left: 20px; }
.md-body :deep(li) { margin: 2px 0; }
.md-body :deep(code) { background: rgba(0,0,0,0.07); border-radius: 3px; padding: 1px 5px; font-family: monospace; font-size: 0.9em; }
.md-body :deep(pre) { background: #2A2622; color: #E8E2D6; border-radius: 6px; padding: 12px 14px; overflow-x: auto; margin: 8px 0; }
.md-body :deep(pre code) { background: none; padding: 0; font-size: 0.88em; }
.md-body :deep(blockquote) { border-left: 3px solid var(--nr-border-strong); margin: 6px 0; padding: 2px 12px; color: var(--nr-ink-2); }
.md-body :deep(table) { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 0.9em; }
.md-body :deep(th),:deep(td) { border: 1px solid var(--nr-border); padding: 5px 10px; }
.md-body :deep(th) { background: var(--nr-rail); font-weight: 600; }
.md-body :deep(a) { color: var(--nr-clay); text-decoration: none; }
.md-body :deep(a:hover) { text-decoration: underline; }
.md-body :deep(hr) { border: none; border-top: 1px solid var(--nr-border); margin: 10px 0; }
.message.assistant .md-body :deep(code) { background: rgba(0,0,0,0.06); }
</style>
