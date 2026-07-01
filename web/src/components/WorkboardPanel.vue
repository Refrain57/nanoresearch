<template>
  <div class="wb-panel">
    <a-spin :spinning="loading">
      <div class="wb-header">
        <span class="wb-title">
          协作看板
          <a-tag v-if="roundActive" color="processing" size="small" style="margin-left: 8px">协作进行中</a-tag>
        </span>
        <a-button type="link" size="small" @click="load"><reload-outlined /> 刷新</a-button>
      </div>

      <a-empty v-if="!loading && !cards.length" description="本对话暂无多主协作看板" :image="simpleImage" style="margin: 24px 0" />

      <div v-for="group in groups" :key="group.status" class="wb-group">
        <div class="wb-group-head">
          <a-tag :color="group.meta.color" size="small">{{ group.meta.label }}</a-tag>
          <span class="wb-group-count">{{ group.cards.length }}</span>
        </div>
        <div v-for="c in group.cards" :key="c.id" class="wb-card" :class="'wb-' + c.status">
          <div class="wb-card-top">
            <span class="wb-card-title">{{ c.title }}</span>
            <span class="wb-card-owner">
              <robot-outlined />
              {{ (c.owner_agent && c.owner_agent.name) || (c.target_agent && c.target_agent.name) || '—' }}
            </span>
          </div>
          <div class="wb-card-meta">
            <span v-if="c.depends_on && c.depends_on.length" class="wb-dep">依赖 {{ c.depends_on.length }} 张前置卡</span>
            <span v-if="c.pass_count" class="wb-pass">被转派 {{ c.pass_count }} 次</span>
          </div>
          <a-collapse v-if="c.result || c.spec" ghost class="wb-collapse">
            <a-collapse-panel key="1" :header="c.result ? '产出' : '任务说明'">
              <pre class="wb-body">{{ c.result || c.spec }}</pre>
            </a-collapse-panel>
          </a-collapse>
        </div>
      </div>
    </a-spin>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { Empty } from 'ant-design-vue'
import { ReloadOutlined, RobotOutlined } from '@ant-design/icons-vue'
import { getWorkboard } from '@/apis/conversations'

const props = defineProps({ convId: { type: String, required: true } })

const simpleImage = Empty.PRESENTED_IMAGE_SIMPLE
const loading = ref(false)
const cards = ref([])
const roundActive = ref(false)

const STATUS_ORDER = ['running', 'ready', 'todo', 'done', 'blocked']
const STATUS_META = {
  running: { label: '进行中', color: 'blue' },
  ready: { label: '待认领', color: 'orange' },
  todo: { label: '等待依赖', color: 'default' },
  done: { label: '已完成', color: 'green' },
  blocked: { label: '受阻', color: 'red' },
}

const groups = computed(() =>
  STATUS_ORDER
    .map(status => ({
      status,
      meta: STATUS_META[status],
      cards: cards.value.filter(c => c.status === status),
    }))
    .filter(g => g.cards.length)
)

async function load() {
  if (!props.convId) return
  loading.value = true
  try {
    const data = await getWorkboard(props.convId)
    cards.value = data.cards || []
    roundActive.value = !!data.round_active
  } catch (e) {
    cards.value = []
    roundActive.value = false
  } finally {
    loading.value = false
  }
}

watch(() => props.convId, () => load(), { immediate: true })
defineExpose({ refresh: load })
</script>

<style scoped>
.wb-panel { padding: 4px 0; }
.wb-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.wb-title { font-size: 14px; font-weight: 600; color: #333; }
.wb-group { margin-bottom: 16px; }
.wb-group-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.wb-group-count { font-size: 12px; color: #999; }
.wb-card { background: #fff; border: 1px solid #e8e8e8; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; border-left: 3px solid #d9d9d9; }
.wb-card.wb-running { border-left-color: #1677ff; }
.wb-card.wb-ready { border-left-color: #faad14; }
.wb-card.wb-done { border-left-color: #52c41a; }
.wb-card.wb-blocked { border-left-color: #f5222d; }
.wb-card-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.wb-card-title { font-size: 13px; font-weight: 600; color: #333; flex: 1; }
.wb-card-owner { font-size: 12px; color: #1677ff; display: flex; align-items: center; gap: 3px; white-space: nowrap; }
.wb-card-meta { display: flex; gap: 10px; margin-top: 4px; }
.wb-dep, .wb-pass { font-size: 11px; color: #999; }
.wb-pass { color: #fa8c16; }
.wb-collapse { margin-top: 4px; margin-left: -16px; }
.wb-body { font-size: 12px; color: #555; white-space: pre-wrap; word-break: break-word; margin: 0; font-family: inherit; max-height: 240px; overflow-y: auto; }
</style>
