import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  listKnowledge, getKnowledge, createKnowledge, updateKnowledge, deleteKnowledge,
  listDocuments, uploadDocument, deleteDocument,
  listChunks, listDocumentChunks, testQuery,
  listDatasets, uploadDataset, deleteDataset,
  listEvalRuns, createEvalRun, getEvalRun, deleteEvalRun,
} from '@/apis/knowledge'

export const useKnowledgeStore = defineStore('knowledge', () => {
  const kbs = ref([])
  const current = ref(null)
  const documents = ref([])
  const chunks = ref([])
  const evalRuns = ref([])
  const loading = ref(false)

  async function fetchList() {
    loading.value = true
    try { kbs.value = await listKnowledge() }
    finally { loading.value = false }
  }

  async function fetchOne(id) {
    loading.value = true
    try { current.value = await getKnowledge(id) }
    finally { loading.value = false }
  }

  async function create(data) {
    const kb = await createKnowledge(data)
    kbs.value.unshift(kb)
    return kb
  }

  async function update(id, data) {
    const updated = await updateKnowledge(id, data)
    const idx = kbs.value.findIndex(k => k.id === id)
    if (idx !== -1) kbs.value[idx] = updated
    if (current.value?.id === id) current.value = updated
    return updated
  }

  async function remove(id) {
    await deleteKnowledge(id)
    kbs.value = kbs.value.filter(k => k.id !== id)
    if (current.value?.id === id) current.value = null
  }

  async function fetchDocuments(kbId) {
    documents.value = await listDocuments(kbId)
    return documents.value
  }

  async function uploadDoc(kbId, file, pdfParser = 'mineru') {
    const doc = await uploadDocument(kbId, file, pdfParser)
    documents.value.unshift(doc)
    return doc
  }

  async function removeDocument(kbId, docId) {
    await deleteDocument(kbId, docId)
    documents.value = documents.value.filter(d => d.id !== docId)
  }

  async function fetchEvalRuns(kbId) {
    evalRuns.value = await listEvalRuns(kbId)
    return evalRuns.value
  }

  async function startEvalRun(kbId, data) {
    const run = await createEvalRun(kbId, data)
    evalRuns.value.unshift(run)
    return run
  }

  return {
    kbs, current, documents, chunks, evalRuns, loading,
    fetchList, fetchOne, create, update, remove,
    fetchDocuments, uploadDoc, removeDocument,
    fetchEvalRuns, startEvalRun,
    // expose API calls directly for one-off use
    listChunks, listDocumentChunks, testQuery,
    listDatasets, uploadDataset, deleteDataset,
    getEvalRun, deleteEvalRun,
  }
})
