export async function login(username, password) {
  const form = new FormData()
  form.append('username', username)
  form.append('password', password)
  const resp = await fetch('/api/auth/token', { method: 'POST', body: form })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}))
    throw new Error(err.detail || '登录失败')
  }
  return resp.json()
}

export async function getMe(token) {
  const resp = await fetch('/api/auth/me', {
    headers: { Authorization: `Bearer ${token}` }
  })
  if (!resp.ok) throw new Error('获取用户信息失败')
  return resp.json()
}
