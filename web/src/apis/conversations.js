import { apiGet, apiPost, apiDelete } from './base'

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
export const deleteConversation = (id) => apiDelete(`/api/conversations/${id}`)
