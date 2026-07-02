import { apiGet, apiPost, apiDelete } from './base'

// ClawHub marketplace (backend proxies clawhub.ai)
export const searchMarket   = (q, limit = 20) =>
  apiGet(`/api/skills/market/search?q=${encodeURIComponent(q)}&limit=${limit}`)
export const getMarketSkill  = (slug) =>
  apiGet(`/api/skills/market/${encodeURIComponent(slug)}`)
export const getMarketReadme = (slug) =>
  apiGet(`/api/skills/market/${encodeURIComponent(slug)}/readme`)

// Workspace skill pool
export const installSkill   = (slug) => apiPost('/api/skills/install', { slug })
export const uninstallSkill = (name) => apiDelete(`/api/skills/${encodeURIComponent(name)}`)
export const listSkills     = ()     => apiGet('/api/skills')
