<template>
  <div class="skill-market">
    <a-input-search
      v-model:value="query"
      placeholder="搜索技能市场，例如 web scraping"
      enter-button="搜索"
      :loading="loading"
      @search="doSearch"
    />

    <a-spin :spinning="loading">
      <div v-if="results.length" class="results">
        <div v-for="r in results" :key="r.slug" class="market-card">
          <div class="market-info">
            <div class="market-name">{{ r.name }}</div>
            <div class="market-meta">@{{ r.owner }} · v{{ r.version }}</div>
            <div v-if="r.summary" class="market-summary">{{ r.summary }}</div>
          </div>
          <a-button size="small" @click="preview(r.slug)">预览</a-button>
        </div>
      </div>
      <a-empty v-else-if="searched" description="没有找到匹配的 skill" />
      <div v-else class="hint">搜索 ClawHub 公共技能市场并安装到你的工作区。</div>
    </a-spin>

    <skill-preview-drawer
      v-model:open="drawerOpen"
      :slug="activeSlug"
      @installed="onInstalled"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { message } from 'ant-design-vue'
import { searchMarket } from '@/apis/skills'
import SkillPreviewDrawer from './SkillPreviewDrawer.vue'

const emit = defineEmits(['installed'])

const query = ref('')
const results = ref([])
const loading = ref(false)
const searched = ref(false)
const drawerOpen = ref(false)
const activeSlug = ref('')

async function doSearch() {
  if (!query.value.trim()) return
  loading.value = true
  try {
    results.value = await searchMarket(query.value.trim())
    searched.value = true
  } catch (e) {
    message.error(e.message || '搜索失败')
  } finally {
    loading.value = false
  }
}

function preview(slug) {
  activeSlug.value = slug
  drawerOpen.value = true
}

function onInstalled(slug) {
  emit('installed', slug)
}
</script>

<style scoped>
.results { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }
.market-card {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 12px; border: 1px solid var(--nr-border, #e6e3da); border-radius: 8px;
}
.market-name { font-weight: 600; }
.market-meta { font-size: 12px; opacity: 0.7; }
.market-summary { font-size: 12px; margin-top: 2px; }
.hint { margin-top: 16px; font-size: 13px; opacity: 0.7; }
</style>
