<template>
  <div class="message-list" ref="listRef" @scroll="onScroll">
    <div v-if="loadingOlder" class="loading-older">加载更早消息…</div>
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
          <CitationText v-else :text="msgText(msg)" :citations="msg.citations || []" />
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

      <!-- 消息附件（agent 通过 message 工具发送的文件） -->
      <div v-if="msg.role === 'assistant' && msg.media?.length" class="attachments">
        <template v-for="(a, i) in msg.media" :key="i">
          <img v-if="isImage(a)" :src="imgUrls[a.path] || ''" class="att-img" @click="openAtt(a)" />
          <div v-else class="att-card" :class="{ clickable: canPreview(a) }" @click="canPreview(a) ? openAtt(a) : downloadAtt(a)">
            <file-outlined class="att-icon" />
            <span class="att-name">{{ a.name }}</span>
            <span class="att-size">{{ fmtSize(a.size) }}</span>
            <download-outlined class="att-dl" @click.stop="downloadAtt(a)" />
          </div>
        </template>
      </div>

      </template>
    </div>

    <!-- 流式输出气泡：streaming=true 时即使还没文本也显示占位光标 -->
    <div v-if="streamingText || props.streaming" class="message assistant">
      <div class="avatar agent-avatar" :style="agentAvatarStyle">{{ agentInitial }}</div>
      <div class="bubble streaming">
        <div v-if="streamingText" class="md-body" v-html="renderStreamingMd(streamingText)" />
        <span class="cursor">▌</span>
      </div>
    </div>

    <div v-if="toolHint" class="tool-hint">
      <loading-outlined spin /> {{ toolHint }}
    </div>

    <FilePreviewModal v-model:open="previewOpen" :file="previewFile" />
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { LoadingOutlined, FileOutlined, DownloadOutlined } from '@ant-design/icons-vue'
import { marked } from 'marked'
import CitationText from './CitationText.vue'
import FilePreviewModal from './FilePreviewModal.vue'
import { fetchWorkspaceFileBlob } from '@/apis/workspace'

marked.setOptions({ breaks: true, gfm: true })

// Streaming text has no citations — plain marked render only.
function renderStreamingMd(text) {
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
  hasMore: { type: Boolean, default: false },
  loadingOlder: { type: Boolean, default: false },
})

const emit = defineEmits(['load-older'])

const listRef = ref(null)

const _prevScrollHeight = ref(0)
const _prevScrollTop = ref(0)
const _prepending = ref(false)

function onScroll() {
  const el = listRef.value
  if (!el) return
  if (el.scrollTop < 80 && props.hasMore && !props.loadingOlder && !_prepending.value) {
    _prevScrollHeight.value = el.scrollHeight
    _prevScrollTop.value = el.scrollTop
    _prepending.value = true
    emit('load-older')
  }
}

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

// ── 消息附件（agent 用 message 工具发送的文件） ──
const previewOpen = ref(false)
const previewFile = ref(null)
const imgUrls = ref({})
const IMG_EXTS = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'ico']
const PREVIEW_EXTS = [...IMG_EXTS, 'pdf', 'md']
function attExt(name) { const i = (name || '').lastIndexOf('.'); return i >= 0 ? name.slice(i + 1).toLowerCase() : '' }
function isImage(a) { return IMG_EXTS.includes(attExt(a.name)) }
function canPreview(a) { return PREVIEW_EXTS.includes(attExt(a.name)) }
function fmtSize(b) { if (b < 1024) return b + 'B'; if (b < 1048576) return (b / 1024).toFixed(1) + 'K'; return (b / 1048576).toFixed(1) + 'M' }
function openAtt(a) { previewFile.value = { path: a.path, name: a.name }; previewOpen.value = true }
async function downloadAtt(a) {
  try {
    const b = await fetchWorkspaceFileBlob(a.path)
    const u = URL.createObjectURL(b)
    const el = document.createElement('a')
    el.href = u; el.download = a.name
    document.body.appendChild(el); el.click(); el.remove()
    URL.revokeObjectURL(u)
  } catch (e) { /* ignore download error */ }
}

// 内联图片附件：按需拉取带鉴权的 blob，缓存 object URL
watch(() => props.messages, (msgs) => {
  for (const m of msgs || []) {
    if (m.role === 'assistant' && m.media?.length) {
      for (const a of m.media) {
        if (isImage(a) && !(a.path in imgUrls.value)) {
          imgUrls.value = { ...imgUrls.value, [a.path]: '' }
          fetchWorkspaceFileBlob(a.path)
            .then(b => { imgUrls.value = { ...imgUrls.value, [a.path]: URL.createObjectURL(b) } })
            .catch(() => {})
        }
      }
    }
  }
}, { deep: true, immediate: true })

watch(() => [props.messages.length, props.streamingText, props.loadingOlder], async () => {
  await nextTick()
  const el = listRef.value
  if (!el) return
  if (_prepending.value) {
    // 加载更早消息中/刚结束。等 loadingOlder 落定再恢复视口位置，
    // 这样即使这一页返回 0 行（messages 长度不变），也能清掉 _prepending。
    if (!props.loadingOlder) {
      // 高度差锚定视口；0 行时 delta=0，不跳动
      el.scrollTop = _prevScrollTop.value + (el.scrollHeight - _prevScrollHeight.value)
      _prepending.value = false
    }
  } else {
    el.scrollTop = el.scrollHeight
  }
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

/* streaming md-body (no citations, plain marked output) */
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

/* 消息附件 */
.attachments { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 0 40px; max-width: 68%; }
.att-img { max-width: 220px; max-height: 160px; border-radius: 8px; cursor: pointer; border: 1px solid var(--nr-border); }
.att-card { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--nr-border); border-radius: 8px; background: var(--nr-card); max-width: 280px; }
.att-card.clickable { cursor: pointer; }
.att-card:hover { background: var(--nr-clay-soft); }
.att-icon { color: var(--nr-ink-3); flex-shrink: 0; }
.att-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12.5px; }
.att-size { font-size: 11px; color: var(--nr-ink-3); flex-shrink: 0; }
.att-dl { font-size: 12px; opacity: 0.6; cursor: pointer; flex-shrink: 0; }
.att-dl:hover { opacity: 1; }
.md-body :deep(a) { color: var(--nr-clay); text-decoration: none; }
.md-body :deep(a:hover) { text-decoration: underline; }
.md-body :deep(hr) { border: none; border-top: 1px solid var(--nr-border); margin: 10px 0; }
.message.assistant .md-body :deep(code) { background: rgba(0,0,0,0.06); }
.loading-older { text-align: center; color: var(--nr-ink-3); font-size: 12px; padding: 4px 0; }
</style>
