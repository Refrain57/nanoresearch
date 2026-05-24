<template>
  <div class="run-timeline">
    <a-spin :spinning="loading">
      <template v-if="run">
        <!-- 头部 -->
        <div class="run-header">
          <div class="run-meta">
            <a-tag :color="statusColor">{{ statusLabel }}</a-tag>
            <span class="model">{{ run.model_used || '未知模型' }}</span>
            <span class="duration" v-if="run.duration_ms">
              <clock-circle-outlined /> {{ (run.duration_ms / 1000).toFixed(2) }}s
            </span>
          </div>
          <div class="run-time">
            {{ formatTime(run.started_at) }}
          </div>
        </div>

        <!-- Token 用量 -->
        <a-card v-if="hasTokens" :bordered="false" class="token-card" size="small">
          <div class="token-bar-wrap">
            <div class="token-bar">
              <div class="seg input" :style="{ flex: tokens.input }" />
              <div class="seg output" :style="{ flex: tokens.output }" />
              <div class="seg cache" :style="{ flex: tokens.cache_read + tokens.cache_write }" />
            </div>
            <div class="token-legend">
              <span class="dot input" />输入 {{ tokens.input.toLocaleString() }}
              <span class="dot output" />输出 {{ tokens.output.toLocaleString() }}
              <span class="dot cache" v-if="tokens.cache_read + tokens.cache_write > 0" />
              <template v-if="tokens.cache_read + tokens.cache_write > 0">
                缓存读 {{ tokens.cache_read.toLocaleString() }} + 写 {{ tokens.cache_write.toLocaleString() }}
              </template>
            </div>
          </div>
        </a-card>

        <!-- 工具调用时间线 -->
        <template v-if="run.tool_calls?.length">
          <div class="section-title">工具调用</div>
          <a-collapse ghost>
            <a-collapse-panel
              v-for="(tc, i) in run.tool_calls"
              :key="i"
              :header="toolHeader(tc)"
            >
              <template #extra>
                <a-tag :color="tc.status === 'success' ? 'green' : 'red'" size="small">
                  {{ tc.status || 'unknown' }}
                </a-tag>
                <span v-if="tc.duration_ms" class="tc-duration">
                  {{ tc.duration_ms }}ms
                </span>
              </template>
              <div class="tc-body">
                <div class="tc-section">
                  <div class="tc-label">Input</div>
                  <pre class="tc-pre">{{ JSON.stringify(tc.input, null, 2) }}</pre>
                </div>
                <div class="tc-section" v-if="tc.output">
                  <div class="tc-label">Output</div>
                  <pre class="tc-pre">{{ typeof tc.output === 'string' ? tc.output : JSON.stringify(tc.output, null, 2) }}</pre>
                </div>
              </div>
            </a-collapse-panel>
          </a-collapse>
        </template>
        <a-empty v-else description="本次运行无工具调用" style="margin: 24px 0" />

        <!-- Artifacts -->
        <template v-if="run.artifacts?.length">
          <div class="section-title">产出文件</div>
          <div class="artifact-list">
            <div v-for="art in run.artifacts" :key="art.name" class="artifact-row">
              <file-outlined />
              <span class="art-name">{{ art.name }}</span>
              <span class="art-meta">{{ art.type }} · {{ formatSize(art.size_bytes) }}</span>
            </div>
          </div>
        </template>

        <!-- 错误信息 -->
        <a-alert v-if="run.error_message" type="error" :message="run.error_message" style="margin-top: 16px" />
      </template>
    </a-spin>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { ClockCircleOutlined, FileOutlined } from '@ant-design/icons-vue'
import { getRun } from '@/apis/runs'

const props = defineProps({ runId: { type: String, required: true } })

const run = ref(null)
const loading = ref(false)

async function fetchRun() {
  if (!props.runId) return
  loading.value = true
  try {
    run.value = await getRun(props.runId)
  } finally {
    loading.value = false
  }
}

onMounted(fetchRun)
watch(() => props.runId, fetchRun)

const statusColor = computed(() => ({
  completed: 'green', failed: 'red', pending: 'orange', running: 'blue'
}[run.value?.status] || 'default'))

const statusLabel = computed(() => ({
  completed: '已完成', failed: '失败', pending: '待执行', running: '运行中'
}[run.value?.status] || run.value?.status || '-'))

const tokens = computed(() => ({
  input: run.value?.tokens_used?.input || 0,
  output: run.value?.tokens_used?.output || 0,
  cache_read: run.value?.tokens_used?.cache_read || 0,
  cache_write: run.value?.tokens_used?.cache_write || 0,
}))

const hasTokens = computed(() =>
  tokens.value.input + tokens.value.output + tokens.value.cache_read + tokens.value.cache_write > 0
)

function toolHeader(tc) {
  const firstVal = tc.input && typeof tc.input === 'object' ? Object.values(tc.input)[0] : null
  if (typeof firstVal === 'string' && firstVal.length > 0) {
    const preview = firstVal.length > 60 ? firstVal.slice(0, 60) + '…' : firstVal
    return `${tc.name}("${preview}")`
  }
  return tc.name
}

function formatTime(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString('zh-CN', { hour12: false })
}

function formatSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
</script>

<style scoped>
.run-timeline { padding: 0; }
.run-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.run-meta { display: flex; align-items: center; gap: 12px; }
.model { font-size: 13px; color: #555; }
.duration { font-size: 13px; color: #888; display: flex; align-items: center; gap: 4px; }
.run-time { font-size: 12px; color: #aaa; }

.token-card { background: #fafafa; border-radius: 8px; margin-bottom: 16px; }
.token-bar-wrap { display: flex; flex-direction: column; gap: 8px; }
.token-bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: #f0f0f0; }
.seg { min-width: 2px; }
.seg.input  { background: #1677ff; }
.seg.output { background: #52c41a; }
.seg.cache  { background: #faad14; }
.token-legend { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #555; flex-wrap: wrap; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 3px; }
.dot.input  { background: #1677ff; }
.dot.output { background: #52c41a; }
.dot.cache  { background: #faad14; }

.section-title { font-size: 14px; font-weight: 600; color: #333; margin: 20px 0 8px; }

.tc-duration { font-size: 12px; color: #aaa; margin-left: 8px; }
.tc-body { padding: 4px 0; }
.tc-section { margin-bottom: 12px; }
.tc-label { font-size: 12px; font-weight: 600; color: #888; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.tc-pre { background: #f5f5f5; border-radius: 6px; padding: 10px 12px; font-size: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-all; margin: 0; max-height: 300px; overflow-y: auto; }

.artifact-list { display: flex; flex-direction: column; gap: 8px; }
.artifact-row { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: #fafafa; border-radius: 6px; }
.art-name { font-size: 13px; font-weight: 500; flex: 1; }
.art-meta { font-size: 12px; color: #aaa; }
</style>
