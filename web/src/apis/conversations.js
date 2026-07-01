import { apiGet, apiPost, apiPut, apiDelete } from './base'

export const listConversations = (params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return apiGet(`/api/conversations${qs ? '?' + qs : ''}`)
}
export const createConversation = (data = {}) => apiPost('/api/conversations', data)
export const getConversation = (id) => apiGet(`/api/conversations/${id}`)
export const getMessages = (id, params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return apiGet(`/api/conversations/${id}/messages${qs ? '?' + qs : ''}`)
}
export const deleteConversation  = (id) => apiDelete(`/api/conversations/${id}`)
export const getConversationRuns = (id) => apiGet(`/api/conversations/${id}/runs`)
export const getWorkboard = (id) => apiGet(`/api/conversations/${id}/workboard`)
export const updateAgentOverride = (id, data) => apiPut(`/api/conversations/${id}/agent-override`, data)
