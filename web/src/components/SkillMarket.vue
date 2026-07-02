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
      <template v-if="results.length">
        <div class="sort-row">
          <span class="sort-label">排序</span>
          <a-radio-group v-model:value="sortBy" size="small">
            <a-radio-button value="score">相关度</a-radio-button>
            <a-radio-button value="downloads">下载量</a-radio-button>
            <a-radio-button value="updated">最近更新</a-radio-button>
          </a-radio-group>
        </div>
        <div class="results">
          <div v-for="r in sortedResults" :key="r.slug" class="market-card">
            <div class="market-info">
              <div class="market-name">{{ r.name }}</div>
              <div class="market-meta">
                @{{ r.owner }}<template v-if="r.version"> · v{{ r.version }}</template> · ↓ {{ r.downloads ?? 0 }}
              </div>
              <div v-if="r.summary" class="market-summary">{{ r.summary }}</div>
            </div>
            <a-button size="small" @click="preview(r.slug)">预览</a-button>
          </div>
        </div>
      </template>
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
import { ref, computed } from 'vue'
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
const sortBy = ref('score')

// 'score' keeps the registry's relevance order; the others re-sort the fetched page.
const sortedResults = computed(() => {
  const arr = [...results.value]
  if (sortBy.value === 'downloads') arr.sort((a, b) => (b.downloads ?? 0) - (a.downloads ?? 0))
  else if (sortBy.value === 'updated') arr.sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0))
  return arr
})

async function doSearch() {
  if (!query.value.trim()) return
  searched.value = false
  results.value = []
  loading.value = true
  try {
    results.value = await searchMarket(query.value.trim())
    searched.value = true
  } catch (e) {
    results.value = []
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
.sort-row { margin-top: 12px; display: flex; align-items: center; gap: 8px; }
.sort-label { font-size: 12px; opacity: 0.7; }
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
