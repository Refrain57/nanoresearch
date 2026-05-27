import { apiGet, apiPut } from './base'

export const getMySettings      = ()     => apiGet('/api/settings/me')
export const updateMySettings   = (data) => apiPut('/api/settings/me', data)
export const getAvailableModels = ()     => apiGet('/api/settings/available-models')
