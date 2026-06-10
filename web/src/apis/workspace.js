import { apiGet, apiPut } from './base'
import { apiRequest } from './base'

export const listWorkspaceFiles = (dir = '') =>
  apiGet(`/api/workspace/files${dir ? '?dir=' + encodeURIComponent(dir) : ''}`)

export const getWorkspaceFile = (path) =>
  apiRequest(`/api/workspace/files/${path}`, { method: 'GET' }, true, 'text')

export const updateWorkspaceFile = (path, content) =>
  apiPut(`/api/workspace/files/${path}`, { content })
