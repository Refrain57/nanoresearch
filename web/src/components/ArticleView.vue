<template>
  <div class="wiki-article">
    <a-spin :spinning="loading">
      <template v-if="article">
        <CitationText :text="article.markdown" :citations="article.citations" />
        <div class="wiki-article-meta">
          <a-tag v-if="article.stale" color="orange">来源已更新</a-tag>
          <a-button size="small" type="link" @click="$emit('generate')">
            {{ article.stale ? '重新生成' : '重新生成词条' }}
          </a-button>
        </div>
      </template>
      <a-button v-else type="dashed" block @click="$emit('generate')">生成词条</a-button>
    </a-spin>
  </div>
</template>

<script setup>
import CitationText from '@/components/CitationText.vue'

defineProps({
  article: { type: Object, default: null },
  loading: { type: Boolean, default: false },
})

defineEmits(['generate'])
</script>

<style scoped>
.wiki-article { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #f0f0f0; }
.wiki-article-meta { margin-top: 8px; display: flex; align-items: center; gap: 8px; }
</style>
