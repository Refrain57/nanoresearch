import { defineStore } from 'pinia'
import { ref } from 'vue'
import { listConversations, createConversation, getMessages, deleteConversation } from '@/apis/conversations'
import { createRun } from '@/apis/runs'

export const useChatStore = defineStore('chat', () => {
  const conversations = ref([])
  const messages = ref([])
  const currentConvId = ref(null)
  const streaming = ref(false)
  const streamingText = ref('')

  const oldestSeq = ref(null)
  const hasMore = ref(false)
  const loadingOlder = ref(false)
  // 内部：已加载的原始（映射但未合并）消息，升序；以及当前 messages 中「合并派生」部分的长度
  let _rawLoaded = []
  let _mergedLen = 0
  const PAGE = 40

  async function fetchConversations() {
    conversations.value = await listConversations()
  }

  async function selectConversation(id) {
    streaming.value = false
    streamingText.value = ''
    currentConvId.value = id
    messages.value = []
    _rawLoaded = []
    _mergedLen = 0
    oldestSeq.value = null
    hasMore.value = false
    loadingOlder.value = false
    try {
      const resp = await getMessages(id, { limit: PAGE })
      // Guard against race: user switched conversations while fetching
      if (currentConvId.value !== id) return
      _rawLoaded = _toMapped(resp.messages)
      const merged = _mergeToolCallMessages(_rawLoaded)
      _mergedLen = merged.length
      messages.value = merged
      oldestSeq.value = _rawLoaded.length ? _rawLoaded[0].seq : null
      hasMore.value = !!resp.has_more
    } catch (e) {
      console.error('[chat] selectConversation failed:', e)
      if (currentConvId.value === id) messages.value = []
    }
  }

  async function loadOlder() {
    if (!currentConvId.value || !hasMore.value || loadingOlder.value || oldestSeq.value == null) return
    const id = currentConvId.value
    loadingOlder.value = true
    try {
      const resp = await getMessages(id, { limit: PAGE, before_seq: oldestSeq.value })
      if (currentConvId.value !== id) return
      const olderMapped = _toMapped(resp.messages)
      if (olderMapped.length) {
        _rawLoaded = [...olderMapped, ..._rawLoaded]
        const remerged = _mergeToolCallMessages(_rawLoaded)
        // 保留流式/重连追加的尾部消息（不属于 _rawLoaded 合并派生的部分）
        const tail = messages.value.slice(_mergedLen)
        messages.value = [...remerged, ...tail]
        _mergedLen = remerged.length
        oldestSeq.value = _rawLoaded[0].seq
      }
      hasMore.value = !!resp.has_more
    } catch (e) {
      console.error('[chat] loadOlder failed:', e)
    } finally {
      loadingOlder.value = false
    }
  }

  async function newConversation(title = null, agentId = null, model = null) {
    const conv = await createConversation({ title, agent_id: agentId, model })
    conversations.value.unshift(conv)
    return conv
  }

  async function removeConversation(id) {
    await deleteConversation(id)
    conversations.value = conversations.value.filter(c => c.id !== id)
    if (currentConvId.value === id) {
      currentConvId.value = null
      messages.value = []
    }
  }

  async function sendMessage(content, ragMode = 'agentic', kbId = null) {
    if (!currentConvId.value) return null
    const run = await createRun(currentConvId.value, content, ragMode, kbId)
    streaming.value = true
    streamingText.value = ''
    return run
  }

  function appendDelta(chunk) {
    streamingText.value += chunk
  }

  function finalizeStream(toolCalls = [], citations = null) {
    if (streamingText.value) {
      messages.value.push({
        id: `stream-${Date.now()}`,
        role: 'assistant',
        content: { text: streamingText.value },
        toolCalls: toolCalls.length ? [...toolCalls] : undefined,
        citations: citations?.length ? [...citations] : undefined,
        seq: messages.value.length
      })
    }
    streamingText.value = ''
    streaming.value = false
  }

  return {
    conversations, messages, currentConvId, streaming, streamingText,
    oldestSeq, hasMore, loadingOlder,
    fetchConversations, selectConversation, loadOlder, newConversation, removeConversation,
    sendMessage, appendDelta, finalizeStream
  }
})

function _normalizeToolCalls(tc) {
  if (!tc || !tc.length) return undefined
  return tc.map(t => ({
    name: t.function?.name || t.name || '?',
    input: (() => { try { return JSON.parse(t.function?.arguments || '{}') } catch { return {} } })(),
    output_summary: '',
    status: 'success',
  }))
}

function _mapRawMessage(m) {
  const stored = m.content
  const text = typeof stored === 'string'
    ? stored
    : (stored?.text ?? stored?.content ?? '')
  const tool_calls = m.tool_calls ?? stored?.tool_calls
  const citations = typeof stored === 'string' ? null : (stored?._citations ?? null)
  return {
    ...m,
    content: { text },
    tool_calls,
    toolCalls: _normalizeToolCalls(tool_calls),
    citations: citations?.length ? citations : undefined,
  }
}

// 后端已过滤 internal，这里 defense-in-depth 再过滤一次；映射为渲染形状（未合并）。
function _toMapped(apiMessages) {
  return (apiMessages || [])
    .filter(m => !(m.content && typeof m.content === 'object' && m.content.internal))
    .map(_mapRawMessage)
}

// 把「仅 tool_calls 的 assistant 消息」并入紧随其后的文本 assistant 消息。
function _mergeToolCallMessages(mapped) {
  const merged = []
  let pendingTc = null
  let pendingCitations = null
  for (const m of mapped) {
    if (m.role === 'assistant' && !m.content.text && m.toolCalls?.length) {
      pendingTc = m.toolCalls
      pendingCitations = m.citations ?? null
    } else if (m.role === 'assistant' && m.content.text) {
      merged.push(pendingTc
        ? { ...m, toolCalls: pendingTc, citations: m.citations ?? pendingCitations ?? undefined }
        : m)
      pendingTc = null
      pendingCitations = null
    } else {
      merged.push(m)
    }
  }
  return merged
}
