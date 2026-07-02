<template>
  <a-collapse accordion class="timeline">
    <a-collapse-panel
      v-for="item in items"
      :key="item.kind + '-' + item.order"
      :class="'tl-' + item.kind"
    >
      <template #header>
        <span class="tl-badge" :class="'tl-badge-' + item.kind">
          {{ item.kind === 'llm' ? 'LLM' : '工具' }}
        </span>
        <span class="mono tl-order">#{{ item.order }}</span>
        <template v-if="item.kind === 'tool'">
          <span class="tl-name">{{ item.data.name }}</span>
          <span v-if="item.data.error" class="tl-err">⚠</span>
        </template>
        <template v-else>
          <span class="tl-name">{{ item.data.model }}</span>
          <span class="muted tl-tokens">↑{{ item.data.input_tokens }} ↓{{ item.data.output_tokens }}</span>
        </template>
        <span class="tl-dur">{{ fmtMs(item.data.duration_ms) }}</span>
      </template>

      <!-- Tool step -->
      <template v-if="item.kind === 'tool'">
        <div class="tl-label">参数</div>
        <pre class="code-block">{{ prettyJson(item.data.params) }}</pre>
        <div class="tl-label" style="margin-top:8px">返回值</div>
        <pre class="code-block result">{{ item.data.result }}</pre>
        <div v-if="item.data.error" class="error-hint">⚠ 工具报错</div>
      </template>

      <!-- LLM step -->
      <template v-else>
        <template v-if="item.data.input_messages && item.data.input_messages.length">
          <div class="tl-label">输入 messages ({{ item.data.input_messages.length }})</div>
          <div class="msg-list">
            <div v-for="(m, mi) in item.data.input_messages" :key="mi" class="msg-row">
              <span class="msg-role" :class="'role-' + m.role">{{ m.role }}</span>
              <pre class="code-block msg-content">{{ m.content }}</pre>
            </div>
          </div>
        </template>
        <div v-else class="muted" style="margin-bottom:6px">（未记录输入，可能为旧快照）</div>

        <template v-if="item.data.output_reasoning">
          <div class="tl-label" style="margin-top:8px">推理</div>
          <pre class="code-block reasoning">{{ item.data.output_reasoning }}</pre>
        </template>

        <div class="tl-label" style="margin-top:8px">输出</div>
        <pre class="code-block result">{{ item.data.output_text || '（无文本，仅请求工具调用）' }}</pre>

        <template v-if="item.data.output_tool_calls && item.data.output_tool_calls.length">
          <div class="tl-label" style="margin-top:8px">请求工具调用</div>
          <pre class="code-block">{{ prettyJson(item.data.output_tool_calls) }}</pre>
        </template>
      </template>
    </a-collapse-panel>
  </a-collapse>
</template>

<script setup>
defineProps({
  items: { type: Array, default: () => [] },
})

function fmtMs(ms) {
  if (ms == null) return '—'
  return `${(ms / 1000).toFixed(2)} s`
}

function prettyJson(v) {
  try {
    return JSON.stringify(v, null, 2)
  } catch {
    return String(v)
  }
}
</script>

<style scoped>
.mono { font-family: monospace; font-size: 12px; }
.muted { color: var(--nr-ink-3); font-size: 12px; }

.timeline :deep(.ant-collapse-header) {
  font-family: monospace;
  font-size: 12px;
  align-items: center;
  gap: 8px;
  display: flex;
}
/* ant-design-vue wraps the #header slot content in .ant-collapse-header-text,
   which is the real flex child. Make THAT a flex row filling the width so
   .tl-dur's `margin-left:auto` actually right-aligns the duration. */
.timeline :deep(.ant-collapse-header-text) {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
/* Colored left rail differentiating LLM vs tool steps */
.timeline :deep(.tl-llm) { border-left: 3px solid var(--nr-sage, #5E7355); }
.timeline :deep(.tl-tool) { border-left: 3px solid var(--nr-gold, #B8912F); }

.tl-badge {
  font-size: 10px;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 3px;
  letter-spacing: .3px;
}
.tl-badge-llm { background: rgba(94,115,85,.16); color: var(--nr-sage, #5E7355); }
.tl-badge-tool { background: rgba(184,145,47,.16); color: var(--nr-gold, #8a6d1f); }
.tl-order { color: var(--nr-ink-3); }
.tl-name { font-weight: 600; color: var(--nr-ink-1); }
.tl-tokens { margin-left: 2px; }
.tl-err { color: var(--nr-danger); }
.tl-dur { margin-left: auto; font-size: 11px; color: var(--nr-ink-3); }

.tl-label { font-size: 11px; color: var(--nr-ink-2); margin-bottom: 4px; }

.msg-list { display: flex; flex-direction: column; gap: 6px; }
.msg-row { display: flex; gap: 8px; align-items: flex-start; }
.msg-role {
  flex: 0 0 62px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  color: var(--nr-ink-2);
  padding-top: 8px;
}
.msg-role.role-system { color: var(--nr-gold, #8a6d1f); }
.msg-role.role-user { color: var(--nr-sage, #5E7355); }
.msg-role.role-assistant { color: var(--nr-ink-1); }
.msg-content { flex: 1; margin: 0; max-height: 120px; }

.code-block {
  background: #2A2622;
  color: #E8E2D6;
  border-radius: 4px;
  padding: 8px 10px;
  font-size: 11px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 160px;
  overflow-y: auto;
  margin: 0;
}
.code-block.result { background: #232A1E; color: #D4E0C8; }
.code-block.reasoning { background: #2A2622; color: #C9B48A; font-style: italic; }
.error-hint { color: var(--nr-danger); font-size: 12px; margin-top: 4px; }
</style>
