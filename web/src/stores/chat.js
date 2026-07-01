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

  async function fetchConversations() {
    conversations.value = await listConversations()
  }

  async function selectConversation(id) {
    streaming.value = false
    streamingText.value = ''
    currentConvId.value = id
    messages.value = []
    try {
      const raw = await getMessages(id, { limit: 100 })
      // Guard against race: if user switched again while fetching, discard stale result
      if (currentConvId.value !== id) {
        console.warn('[chat] selectConversation stale result discarded: wanted', id, 'but now on', currentConvId.value)
        return
      }
      console.log('[chat] selectConversation loaded', raw.length, 'msgs for', id,
        raw.map(m => ({ role: m.role, contentType: typeof m.content, hasToolCalls: !!(m.content?.tool_calls?.length) }))
      )
      const mapped = raw.map(m => {
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
      })

      // Merge tool-call-only assistant messages into the following text assistant message.
      // DB stores them as separate messages (tool_calls message then text message), but
      // the streaming path combines them into one — this makes loaded history match.
      const merged = []
      let pendingTc = null
      let pendingCitations = null
      for (const m of mapped) {
        if (m.role === 'assistant' && !m.content.text && m.toolCalls?.length) {
          pendingTc = m.toolCalls // hold, skip this message
          pendingCitations = m.citations ?? null
        } else if (m.role === 'assistant' && m.content.text) {
          merged.push(pendingTc
            ? { ...m, toolCalls: pendingTc, citations: m.citations ?? pendingCitations ?? undefined }
            : m)
          pendingTc = null
          pendingCitations = null
        } else {
          merged.push(m)
          // non-assistant messages (tool results etc.) don't reset pendingTc
        }
      }
      messages.value = merged
    } catch (e) {
      console.error('[chat] selectConversation failed:', e)
      if (currentConvId.value === id) messages.value = []
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
    fetchConversations, selectConversation, newConversation, removeConversation,
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
