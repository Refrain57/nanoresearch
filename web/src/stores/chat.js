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
    currentConvId.value = id
    messages.value = await getMessages(id, { limit: 100 })
    streamingText.value = ''
  }

  async function newConversation(title = null, agentId = null) {
    const conv = await createConversation({ title, agent_id: agentId })
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

  async function sendMessage(content) {
    if (!currentConvId.value) return null
    const run = await createRun(currentConvId.value, content)
    streaming.value = true
    streamingText.value = ''
    return run
  }

  function appendDelta(chunk) {
    streamingText.value += chunk
  }

  function finalizeStream() {
    if (streamingText.value) {
      messages.value.push({
        id: `stream-${Date.now()}`,
        role: 'assistant',
        content: { text: streamingText.value },
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
