import { apiGet, apiPost } from './base'

export const createRun = (conversationId, content, ragMode = 'agentic', kbId = null) =>
  apiPost('/api/runs', { conversation_id: conversationId, content, rag_mode: ragMode, kb_id: kbId })

export const getRun = (id) => apiGet(`/api/runs/${id}`)
