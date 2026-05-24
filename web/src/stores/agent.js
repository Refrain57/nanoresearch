import { defineStore } from 'pinia'
import { ref } from 'vue'
import { listAgents, getAgent, updateAgent } from '@/apis/agents'

export const useAgentStore = defineStore('agent', () => {
  const agents = ref([])
  const current = ref(null)
  const loading = ref(false)

  async function fetchList() {
    loading.value = true
    try { agents.value = await listAgents() }
    finally { loading.value = false }
  }

  async function fetchOne(id) {
    loading.value = true
    try { current.value = await getAgent(id) }
    finally { loading.value = false }
  }

  async function update(id, data) {
    const updated = await updateAgent(id, data)
    const idx = agents.value.findIndex(a => a.id === id)
    if (idx !== -1) agents.value[idx] = updated
    if (current.value?.id === id) current.value = updated
    return updated
  }

  return { agents, current, loading, fetchList, fetchOne, update }
})
