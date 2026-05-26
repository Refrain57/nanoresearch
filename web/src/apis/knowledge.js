import { apiGet, apiPost, apiPut, apiDelete, apiRequest } from './base'

// Knowledge bases
export const listKnowledge    = ()           => apiGet('/api/knowledge')
export const getKnowledge     = (id)         => apiGet(`/api/knowledge/${id}`)
export const createKnowledge  = (data)       => apiPost('/api/knowledge', data)
export const updateKnowledge  = (id, data)   => apiPut(`/api/knowledge/${id}`, data)
export const deleteKnowledge  = (id)         => apiDelete(`/api/knowledge/${id}`)

// Documents
export const listDocuments    = (kbId)       => apiGet(`/api/knowledge/${kbId}/documents`)
export const deleteDocument   = (kbId, docId) => apiDelete(`/api/knowledge/${kbId}/documents/${docId}`)

export const uploadDocument = (kbId, file) => {
  const form = new FormData()
  form.append('file', file)
  return apiPost(`/api/knowledge/${kbId}/documents`, form)
}

// Chunks
export const listChunks        = (kbId, params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return apiGet(`/api/knowledge/${kbId}/chunks${qs ? '?' + qs : ''}`)
}
export const listDocumentChunks = (kbId, docId) => apiGet(`/api/knowledge/${kbId}/documents/${docId}/chunks`)

// Test retrieval
export const testQuery = (kbId, query, topK = 5) =>
  apiPost(`/api/knowledge/${kbId}/query/test`, { query, top_k: topK })

// Eval datasets
export const listDatasets   = (kbId)       => apiGet(`/api/eval/${kbId}/datasets`)
export const deleteDataset  = (datasetId)  => apiDelete(`/api/eval/datasets/${datasetId}`)

export const uploadDataset = (kbId, name, file) => {
  const form = new FormData()
  form.append('file', file)
  form.append('name', name)
  return apiPost(`/api/eval/${kbId}/datasets/upload?name=${encodeURIComponent(name)}`, form)
}

// Eval runs
export const listEvalRuns   = (kbId)       => apiGet(`/api/eval/${kbId}/runs`)
export const createEvalRun  = (kbId, data) => apiPost(`/api/eval/${kbId}/runs`, data)
export const getEvalRun     = (kbId, runId) => apiGet(`/api/eval/${kbId}/runs/${runId}`)
export const deleteEvalRun  = (kbId, runId) => apiDelete(`/api/eval/${kbId}/runs/${runId}`)
