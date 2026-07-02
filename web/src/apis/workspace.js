import { apiGet, apiPut, apiDelete, apiRequest } from './base'

// 按段编码路径，保留 `/` 以匹配后端 {file_path:path} 路由
const encodePath = (p) => p.split('/').map(encodeURIComponent).join('/')

export const listWorkspaceFiles = (dir = '') =>
  apiGet(`/api/workspace/files${dir ? '?dir=' + encodeURIComponent(dir) : ''}`)

export const getWorkspaceFile = (path) =>
  apiRequest(`/api/workspace/files/${encodePath(path)}`, { method: 'GET' }, true, 'text')

export const updateWorkspaceFile = (path, content) =>
  apiPut(`/api/workspace/files/${encodePath(path)}`, { content })

export const deleteWorkspaceFile = (path) =>
  apiDelete(`/api/workspace/files/${encodePath(path)}`)

export const fetchWorkspaceFileBlob = (path) =>
  apiRequest(`/api/workspace/files/${encodePath(path)}`, { method: 'GET' }, true, 'blob')
    .then((res) => res.blob())
