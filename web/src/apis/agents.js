import { apiGet, apiPost, apiPut, apiDelete } from './base'

export const listAgents  = ()         => apiGet('/api/agents')
export const getAgent    = (id)       => apiGet(`/api/agents/${id}`)
export const updateAgent = (id, data) => apiPut(`/api/agents/${id}`, data)
export const createAgent = (data)     => apiPost('/api/agents', data)
export const listSkills  = ()         => apiGet('/api/skills')
export const deleteAgent = (id)       => apiDelete(`/api/agents/${id}`)
export const listAgentRuns = (id, limit = 20) => apiGet(`/api/agents/${id}/runs?limit=${limit}`)
