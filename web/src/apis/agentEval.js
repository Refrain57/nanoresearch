import { apiGet, apiPost, apiDelete, apiRequest } from './base'

// Snapshots
export const getSnapshots = (params = {}) => {
  const q = new URLSearchParams()
  if (params.is_badcase != null) q.set('is_badcase', params.is_badcase)
  if (params.run_status) q.set('run_status', params.run_status)
  if (params.page) q.set('page', params.page)
  if (params.page_size) q.set('page_size', params.page_size)
  return apiGet(`/api/eval/agent/snapshots?${q}`)
}

export const getSnapshot = (id) => apiGet(`/api/eval/agent/snapshots/${id}`)

// Badcases
export const getBadcases = (params = {}) => {
  const q = new URLSearchParams()
  if (params.badcase_status) q.set('badcase_status', params.badcase_status)
  if (params.badcase_category) q.set('badcase_category', params.badcase_category)
  if (params.semantic_category) q.set('semantic_category', params.semantic_category)
  if (params.root_cause_auto) q.set('root_cause_auto', params.root_cause_auto)
  if (params.page) q.set('page', params.page)
  if (params.page_size) q.set('page_size', params.page_size)
  return apiGet(`/api/eval/agent/badcases?${q}`)
}

export const annotateBadcase = (id, data) =>
  apiRequest(`/api/eval/agent/badcases/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  })

// Test cases
export const getTestCases = (dataset_type) => {
  const q = dataset_type ? `?dataset_type=${dataset_type}` : ''
  return apiGet(`/api/eval/agent/testcases${q}`)
}

export const createTestCase = (data) => apiPost('/api/eval/agent/testcases', data)

export const deleteTestCase = (id) => apiDelete(`/api/eval/agent/testcases/${id}`)

// Stats
export const getEvalStats = () => apiGet('/api/eval/agent/stats')

// Eval Runs (batch evaluation)
export const triggerEvalRun = (data) => apiPost('/api/eval/agent/eval-runs', data)

export const getEvalRuns = (params = {}) => {
  const q = new URLSearchParams()
  if (params.page) q.set('page', params.page)
  if (params.page_size) q.set('page_size', params.page_size)
  return apiGet(`/api/eval/agent/eval-runs?${q}`)
}

export const getEvalRun = (id) => apiGet(`/api/eval/agent/eval-runs/${id}`)
export const deleteEvalRun = (id) => apiDelete(`/api/eval/agent/eval-runs/${id}`)

// Trends
export const getTrends = (days = 30) => apiGet(`/api/eval/agent/trends?days=${days}`)

// Snapshot replay
export const replaySnapshot = (id) =>
  apiRequest(`/api/eval/agent/snapshots/${id}/replay`, { method: 'POST' })

// Badcase: promote + batch classify
export const promoteBadcase = (id, data) => apiPost(`/api/eval/agent/badcases/${id}/promote`, data)

export const classifyBadcasesBatch = () =>
  apiRequest('/api/eval/agent/badcases/classify-batch', { method: 'POST' })

export const getClassifyBatchStatus = () =>
  apiGet('/api/eval/agent/badcases/classify-batch/status')

// Eval Run: regression + comparison
export const getEvalRunRegression = (id) => apiGet(`/api/eval/agent/eval-runs/${id}/regression`)

export const compareEvalRuns = (run_a, run_b) =>
  apiGet(`/api/eval/agent/eval-runs/comparison?run_a=${run_a}&run_b=${run_b}`)

// Judge Calibration
export const triggerCalibration = (data) => apiPost('/api/eval/agent/judge-calibration', data)

export const getCalibrationHistory = (params = {}) => {
  const q = new URLSearchParams()
  if (params.page) q.set('page', params.page)
  if (params.page_size) q.set('page_size', params.page_size)
  return apiGet(`/api/eval/agent/judge-calibration?${q}`)
}

// Optimization Proposals
export const triggerOptimization = (data) => apiPost('/api/eval/agent/optimize', data)

export const getOptimizations = (params = {}) => {
  const q = new URLSearchParams()
  if (params.status) q.set('status', params.status)
  if (params.page) q.set('page', params.page)
  if (params.page_size) q.set('page_size', params.page_size)
  return apiGet(`/api/eval/agent/optimize?${q}`)
}

export const updateOptimizationStatus = (id, status) =>
  apiRequest(`/api/eval/agent/optimize/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  })

// Root-cause closed-loop diagnosis
export const contextDiagnosis = (data) => apiPost('/api/eval/agent/badcases/context-diagnosis', data)

export const getToolErrorStats = (days = 7) => apiGet(`/api/eval/agent/tool-error-stats?days=${days}`)

export const userInputAnalysis = (data) => apiPost('/api/eval/agent/badcases/user-input-analysis', data)

export const getModelAnalysis = (limit = 20) => apiGet(`/api/eval/agent/badcases/model-analysis?limit=${limit}`)
