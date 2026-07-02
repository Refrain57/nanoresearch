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

export const uploadDocument = (kbId, file, pdfParser = 'mineru') => {
  const form = new FormData()
  form.append('file', file)
  form.append('pdf_parser', pdfParser)
  return apiPost(`/api/knowledge/${kbId}/documents`, form)
}

// Chunks
export const listChunks        = (kbId, params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return apiGet(`/api/knowledge/${kbId}/chunks${qs ? '?' + qs : ''}`)
}
export const listDocumentChunks = (kbId, docId) => apiGet(`/api/knowledge/${kbId}/documents/${docId}/chunks`)

// Document file download (returns Blob)
export const getDocumentFileBlob = async (kbId, docId) => {
  const { useUserStore } = await import('@/stores/user')
  const userStore = useUserStore()
  const resp = await fetch(`/api/knowledge/${kbId}/documents/${docId}/file`, {
    headers: {
      ...userStore.getAuthHeaders(),
      'Accept-Encoding': 'identity',
    },
  })
  if (!resp.ok) throw new Error(await resp.text())
  return resp.blob()
}

// Knowledge Graph
export const buildDocGraph   = (kbId, docId) => apiPost(`/api/knowledge/${kbId}/documents/${docId}/graph/build`, {})
export const getDocEntities  = (kbId, docId) => apiGet(`/api/knowledge/${kbId}/documents/${docId}/graph/entities`)
export const getGraphStats   = (kbId)        => apiGet(`/api/knowledge/${kbId}/graph/stats`)
export const buildKbGraph    = (kbId)        => apiPost(`/api/knowledge/${kbId}/graph/build`, {})

export const listGraphEntities = (kbId, params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return apiGet(`/api/knowledge/${kbId}/graph/entities${qs ? '?' + qs : ''}`)
}
export const getGraphEntity  = (kbId, name)     => apiGet(`/api/knowledge/${kbId}/graph/entities/${encodeURIComponent(name)}`)
export const getTripleChunks = (kbId, tripleId) => apiGet(`/api/knowledge/${kbId}/graph/triples/${tripleId}/chunks`)
export const getEntityArticle      = (kbId, name) => apiGet(`/api/knowledge/${kbId}/graph/entities/${encodeURIComponent(name)}/article`)
export const generateEntityArticle = (kbId, name) => apiPost(`/api/knowledge/${kbId}/graph/entities/${encodeURIComponent(name)}/article`, {})

export const listConcepts           = (kbId)        => apiGet(`/api/knowledge/${kbId}/graph/concepts`)
export const getConceptArticle      = (kbId, topic) => apiGet(`/api/knowledge/${kbId}/graph/concept/article?topic=${encodeURIComponent(topic)}`)
export const generateConceptArticle = (kbId, topic) => apiPost(`/api/knowledge/${kbId}/graph/concept/article?topic=${encodeURIComponent(topic)}`, {})
export const getOverviewArticle     = (kbId)        => apiGet(`/api/knowledge/${kbId}/graph/overview/article`)
export const generateOverviewArticle = (kbId)       => apiPost(`/api/knowledge/${kbId}/graph/overview/article`, {})

// Test retrieval
export const testQuery = (kbId, query, topK = 5, mode = 'hybrid') =>
  apiPost(`/api/knowledge/${kbId}/query/test`, {
    query,
    top_k: topK,
    enable_dense: mode !== 'sparse',
    enable_sparse: mode !== 'dense',
  })

// Eval datasets
export const listDatasets   = (kbId)       => apiGet(`/api/eval/${kbId}/datasets`)
export const deleteDataset  = (datasetId)  => apiDelete(`/api/eval/datasets/${datasetId}`)

export const uploadDataset = (kbId, name, file) => {
  const form = new FormData()
  form.append('file', file)
  form.append('name', name)
  return apiPost(`/api/eval/${kbId}/datasets/upload?name=${encodeURIComponent(name)}`, form)
}

export const generateDataset = (kbId, data) => apiPost(`/api/eval/${kbId}/datasets/generate`, data)

// Eval runs
export const listEvalRuns   = (kbId)       => apiGet(`/api/eval/${kbId}/runs`)
export const createEvalRun  = (kbId, data) => apiPost(`/api/eval/${kbId}/runs`, data)
export const createRagasRun = (kbId, data) => apiPost(`/api/eval/${kbId}/runs/ragas`, data)
export const createAgentRun = (kbId, data) => apiPost(`/api/eval/${kbId}/runs/agent`, data)
export const getEvalRun     = (kbId, runId) => apiGet(`/api/eval/${kbId}/runs/${runId}`)
export const deleteEvalRun  = (kbId, runId) => apiDelete(`/api/eval/${kbId}/runs/${runId}`)
