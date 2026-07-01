<template>
  <app-layout>
    <div class="run-detail-page">
      <div class="page-header">
        <a-button type="text" @click="router.back()" class="back-btn">
          <arrow-left-outlined /> 返回
        </a-button>
        <h2 class="nr-serif">Run 详情</h2>
        <span class="run-id-hint">{{ route.params.id }}</span>
      </div>
      <run-timeline :run-id="route.params.id" />
    </div>
  </app-layout>
</template>

<script setup>
import { useRoute, useRouter } from 'vue-router'
import { ArrowLeftOutlined } from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import RunTimeline from '@/components/RunTimeline.vue'

const route = useRoute()
const router = useRouter()
</script>

<style scoped>
.run-detail-page { padding: 32px; max-width: 860px; }
.page-header { display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }
.page-header h2 { margin: 0; font-size: 20px; font-weight: 700; }
.back-btn { color: var(--nr-ink-2); }
.run-id-hint { font-size: 12px; color: var(--nr-ink-3); font-family: monospace; margin-left: auto; }
</style>
