import { useUserStore } from '@/stores/user'
import { message } from 'ant-design-vue'

export async function apiRequest(url, options = {}, requiresAuth = true, responseType = 'json') {
  try {
    const isFormData = options?.body instanceof FormData
    const requestOptions = {
      ...options,
      headers: {
        ...(!isFormData ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers
      }
    }

    if (requiresAuth) {
      const userStore = useUserStore()
      if (!userStore.isLoggedIn) throw new Error('用户未登录')
      Object.assign(requestOptions.headers, userStore.getAuthHeaders())
    }

    const response = await fetch(url, requestOptions)

    if (!response.ok) {
      let errorMessage = `请求失败: ${response.status}`
      let errorData = null
      try {
        errorData = await response.json()
        errorMessage = errorData.detail || errorData.message || errorMessage
      } catch (_) {}

      const error = new Error(errorMessage)
      error.status = response.status
      error.data = errorData

      if (response.status === 401) {
        const userStore = useUserStore()
        message.error('登录已过期，请重新登录')
        userStore.logout()
        setTimeout(() => { window.location.href = '/login' }, 1000)
        throw error
      }
      throw error
    }

    if (responseType === 'json') return response.json()
    if (responseType === 'text') return response.text()
    return response
  } catch (error) {
    if (error.name !== 'AbortError') console.error('API请求错误:', error)
    throw error
  }
}

export const apiGet = (url, options = {}, requiresAuth = true) =>
  apiRequest(url, { method: 'GET', ...options }, requiresAuth)

export const apiPost = (url, data = {}, options = {}, requiresAuth = true) =>
  apiRequest(url, { method: 'POST', body: data instanceof FormData ? data : JSON.stringify(data), ...options }, requiresAuth)

export const apiPut = (url, data = {}, options = {}, requiresAuth = true) =>
  apiRequest(url, { method: 'PUT', body: JSON.stringify(data), ...options }, requiresAuth)

export const apiDelete = (url, options = {}, requiresAuth = true) =>
  apiRequest(url, { method: 'DELETE', ...options }, requiresAuth)
