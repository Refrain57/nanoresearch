import { apiGet, apiPost, apiPut, apiDelete } from './base'

export const listAgents  = ()         => apiGet('/api/agents')
export const getAgent    = (id)       => apiGet(`/api/agents/${id}`)
export const updateAgent = (id, data) => apiPut(`/api/agents/${id}`, data)
export const createAgent = (data)     => apiPost('/api/agents', data)
export const listSkills  = ()         => apiGet('/api/skills')
export const deleteAgent = (id)       => apiDelete(`/api/agents/${id}`)
export const listAgentRuns        = (id, limit = 20) => apiGet(`/api/agents/${id}/runs?limit=${limit}`)
export const getAgentToolStats    = (id)             => apiGet(`/api/agents/${id}/tool-stats`)
export const getAgentPromptPreview = (id)            => apiGet(`/api/agents/${id}/prompt-preview`)
export const listBoundKBs         = (id)             => apiGet(`/api/agents/${id}/knowledge`)
export const bindKB               = (id, kbId)       => apiPost(`/api/agents/${id}/knowledge`, { kb_id: kbId })
export const unbindKB             = (id, kbId)       => apiDelete(`/api/agents/${id}/knowledge/${kbId}`)
