<template>
  <app-layout>
    <div class="agents-page">
      <div class="page-header">
        <h2>Agent 画廊</h2>
      </div>

      <a-spin :spinning="agentStore.loading">
        <div v-if="agentStore.agents.length" class="agents-grid">
          <agent-card
            v-for="agent in agentStore.agents"
            :key="agent.id"
            :agent="agent"
            @chat="handleChat"
            @detail="handleDetail"
          />
        </div>
        <a-empty v-else description="暂无 Agent" style="margin-top: 80px" />
      </a-spin>
    </div>
  </app-layout>
</template>

<script setup>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import AppLayout from '@/layouts/AppLayout.vue'
import AgentCard from '@/components/AgentCard.vue'
import { useAgentStore } from '@/stores/agent'
import { useChatStore } from '@/stores/chat'

const router = useRouter()
const agentStore = useAgentStore()
const chatStore = useChatStore()

onMounted(() => agentStore.fetchList())

async function handleChat(agent) {
  const conv = await chatStore.newConversation(agent.name, agent.id)
  router.push(`/chat/${conv.id}`)
}

function handleDetail(agent) {
  router.push(`/agents/${agent.id}`)
}
</script>

<style scoped>
.agents-page { padding: 32px; }
.page-header { margin-bottom: 24px; }
.page-header h2 { font-size: 22px; font-weight: 700; margin: 0; }
.agents-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}
</style>
