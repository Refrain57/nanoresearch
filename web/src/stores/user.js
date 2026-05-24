import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useUserStore = defineStore('user', () => {
  const token = ref(localStorage.getItem('nr_token') || '')
  const uid = ref(localStorage.getItem('nr_uid') || '')

  const isLoggedIn = computed(() => !!token.value)

  function getAuthHeaders() {
    return { Authorization: `Bearer ${token.value}` }
  }

  function setToken(newToken, newUid) {
    token.value = newToken
    uid.value = newUid || ''
    localStorage.setItem('nr_token', newToken)
    if (newUid) localStorage.setItem('nr_uid', newUid)
  }

  function logout() {
    token.value = ''
    uid.value = ''
    localStorage.removeItem('nr_token')
    localStorage.removeItem('nr_uid')
  }

  return { token, uid, isLoggedIn, getAuthHeaders, setToken, logout }
})
