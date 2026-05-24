import { apiGet, apiPost } from './base'

export const createRun = (conversationId, content) =>
  apiPost('/api/runs', { conversation_id: conversationId, content })

export const getRun = (id) => apiGet(`/api/runs/${id}`)
