import { defineStore } from 'pinia'
import { ref } from 'vue'
import { listAgents, getAgent, updateAgent, createAgent, listSkills, deleteAgent } from '@/apis/agents'

export const useAgentStore = defineStore('agent', () => {
  const agents = ref([])
  const current = ref(null)
  const loading = ref(false)
  const skills = ref([])   // 系统可用 skills

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

  async function fetchSkills() {
    skills.value = await listSkills()
  }

  async function create(data) {
    const agent = await createAgent(data)
    agents.value.unshift(agent)
    return agent
  }

  async function remove(id) {
    await deleteAgent(id)
    agents.value = agents.value.filter(a => a.id !== id)
    if (current.value?.id === id) current.value = null
  }

  async function update(id, data) {
    const updated = await updateAgent(id, data)
    const idx = agents.value.findIndex(a => a.id === id)
    if (idx !== -1) agents.value[idx] = updated
    if (current.value?.id === id) current.value = updated
    return updated
  }

  return { agents, current, loading, skills, fetchList, fetchOne, fetchSkills, create, update, remove }
})
