<template>
  <div class="login-page">
    <div class="login-card">
      <h1 class="login-title">Nano Research</h1>
      <p class="login-subtitle">个人 AI 研究助手</p>

      <a-form :model="form" @finish="handleLogin" layout="vertical">
        <a-form-item name="username" :rules="[{ required: true, message: '请输入用户名' }]">
          <a-input v-model:value="form.username" placeholder="用户名" size="large" />
        </a-form-item>
        <a-form-item name="password" :rules="[{ required: true, message: '请输入密码' }]">
          <a-input-password v-model:value="form.password" placeholder="密码" size="large" />
        </a-form-item>
        <a-form-item>
          <a-button type="primary" html-type="submit" :loading="loading" block size="large">
            登录
          </a-button>
        </a-form-item>
      </a-form>

      <a-alert v-if="error" :message="error" type="error" show-icon style="margin-top: 8px" />
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useUserStore } from '@/stores/user'
import { login, getMe } from '@/apis/auth'

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

const form = reactive({ username: '', password: '' })
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  loading.value = true
  error.value = ''
  try {
    const data = await login(form.username, form.password)
    let uid = ''
    try { const me = await getMe(data.access_token); uid = me.uid } catch (_) {}
    userStore.setToken(data.access_token, uid)
    router.push(route.query.redirect || '/chat')
  } catch (err) {
    error.value = err.message || '登录失败'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f0f2f5;
}
.login-card {
  background: #fff;
  border-radius: 12px;
  padding: 48px 40px;
  width: 400px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.login-title {
  font-size: 28px;
  font-weight: 700;
  margin: 0 0 4px;
  color: #1a1a2e;
}
.login-subtitle {
  color: #888;
  margin: 0 0 32px;
}
</style>
