import { apiGet, apiPut } from './base'

export const listAgents = () => apiGet('/api/agents')
export const getAgent = (id) => apiGet(`/api/agents/${id}`)
export const updateAgent = (id, data) => apiPut(`/api/agents/${id}`, data)
