import { createRouter, createWebHistory } from 'vue-router'
import { useUserStore } from '@/stores/user'

const routes = [
  { path: '/login', component: () => import('@/views/LoginView.vue'), meta: { requiresAuth: false } },
  { path: '/chat', component: () => import('@/views/ChatView.vue'), meta: { requiresAuth: true } },
  { path: '/chat/:id', component: () => import('@/views/ChatView.vue'), meta: { requiresAuth: true } },
  { path: '/agents', component: () => import('@/views/AgentsView.vue'), meta: { requiresAuth: true } },
  { path: '/agents/:id', component: () => import('@/views/AgentDetailView.vue'), meta: { requiresAuth: true } },
  { path: '/runs/:id', component: () => import('@/views/RunDetailView.vue'), meta: { requiresAuth: true } },
  { path: '/knowledge', component: () => import('@/views/KnowledgeView.vue'), meta: { requiresAuth: true } },
  { path: '/knowledge/:id', component: () => import('@/views/KnowledgeDetailView.vue'), meta: { requiresAuth: true } },
  { path: '/knowledge/:id/eval', redirect: to => `/knowledge/${to.params.id}` },
  { path: '/eval/agent', component: () => import('@/views/AgentEvalView.vue'), meta: { requiresAuth: true } },
  { path: '/', redirect: '/chat' }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to) => {
  const userStore = useUserStore()
  if (to.meta.requiresAuth !== false && !userStore.isLoggedIn) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  if (to.path === '/login' && userStore.isLoggedIn) {
    return { path: '/chat' }
  }
})

export default router
