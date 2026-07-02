<template>
  <a-modal :open="open" :title="file?.name" :footer="null" width="80%"
           wrap-class-name="fp-modal" @update:open="v => emit('update:open', v)">
    <a-spin :spinning="loading">
      <div class="fp-body">
        <img v-if="kind === 'image' && url" :src="url" class="fp-img" />
        <VuePdfEmbed v-else-if="kind === 'pdf' && url" :source="url" class="fp-pdf" />
        <div v-else-if="kind === 'markdown'" class="fp-md" v-html="renderedMd"></div>
        <pre v-else-if="kind === 'text'" class="fp-text">{{ text }}</pre>
        <div v-else class="fp-other">此文件类型不支持预览，请下载后查看。</div>
      </div>
    </a-spin>
    <div class="fp-actions">
      <a-button size="small" @click="download"><download-outlined /> 下载</a-button>
    </div>
  </a-modal>
</template>

<script setup>
import { ref, watch, onBeforeUnmount } from 'vue'
import { message } from 'ant-design-vue'
import { DownloadOutlined } from '@ant-design/icons-vue'
import VuePdfEmbed from 'vue-pdf-embed'
import 'vue-pdf-embed/dist/styles/textLayer.css'
import 'vue-pdf-embed/dist/styles/annotationLayer.css'
import { marked } from 'marked'
import { fetchWorkspaceFileBlob } from '@/apis/workspace'

marked.setOptions({ breaks: true, gfm: true })

const props = defineProps({ open: Boolean, file: { type: Object, default: null } })
const emit = defineEmits(['update:open'])

const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'ico']
const TEXT_EXTS = ['txt', 'json', 'log', 'csv', 'yaml', 'yml', 'js', 'ts', 'jsx', 'tsx', 'vue', 'py', 'css', 'html', 'xml', 'sh', 'toml', 'ini']

const loading = ref(false)
const kind = ref('')
const url = ref('')
const text = ref('')
const renderedMd = ref('')
let blob = null
let seq = 0

function extOf(name) {
  const i = (name || '').lastIndexOf('.')
  return i >= 0 ? name.slice(i + 1).toLowerCase() : ''
}
function revoke() { if (url.value) { URL.revokeObjectURL(url.value); url.value = '' } }

async function load(file) {
  const my = ++seq
  revoke(); text.value = ''; renderedMd.value = ''; blob = null
  const ext = extOf(file.name)
  kind.value = IMAGE_EXTS.includes(ext) ? 'image'
    : ext === 'pdf' ? 'pdf'
    : ext === 'md' ? 'markdown'
    : TEXT_EXTS.includes(ext) ? 'text' : 'other'
  loading.value = true
  try {
    const b = await fetchWorkspaceFileBlob(file.path)
    if (my !== seq) return
    blob = b
    if (kind.value === 'markdown' || kind.value === 'text') {
      const raw = await b.text()
      if (my !== seq) return
      if (kind.value === 'markdown') {
        const body = raw.replace(/^﻿/, '').replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, '')
        renderedMd.value = marked.parse(body)
      } else { text.value = raw }
    } else {
      url.value = URL.createObjectURL(b)
    }
  } catch (e) {
    if (my === seq) { message.error(`预览 ${file.name} 失败：` + (e.message || '')); emit('update:open', false) }
  } finally {
    if (my === seq) loading.value = false
  }
}

function download() {
  if (!blob || !props.file) return
  const u = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = u; a.download = props.file.name
  document.body.appendChild(a); a.click(); a.remove()
  URL.revokeObjectURL(u)
}

watch(() => [props.open, props.file], ([open, file]) => {
  if (open && file) load(file); else if (!open) { seq++; revoke() }
})
onBeforeUnmount(revoke)
</script>

<style scoped>
.fp-body { max-height: 70vh; overflow: auto; display: flex; justify-content: center; }
.fp-img { max-width: 100%; height: auto; }
.fp-pdf { width: 100%; }
.fp-text { width: 100%; margin: 0; padding: 12px 14px; background: var(--nr-canvas, #faf7f2); color: var(--nr-ink, #2b2b2b); border-radius: 6px; font-size: 12.5px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
.fp-other { padding: 24px; color: var(--nr-ink-3); }
.fp-md { width: 100%; padding: 4px 8px 12px; font-size: 13.5px; line-height: 1.7; color: var(--nr-ink, #2b2b2b); word-break: break-word; text-align: left; }
.fp-md :deep(h1), .fp-md :deep(h2), .fp-md :deep(h3), .fp-md :deep(h4) { margin: 1em 0 0.5em; line-height: 1.3; font-weight: 600; }
.fp-md :deep(h1) { font-size: 1.5em; }
.fp-md :deep(h2) { font-size: 1.3em; border-bottom: 1px solid var(--nr-border); padding-bottom: 0.3em; }
.fp-md :deep(h3) { font-size: 1.15em; }
.fp-md :deep(p) { margin: 0.5em 0; }
.fp-md :deep(ul), .fp-md :deep(ol) { padding-left: 1.5em; margin: 0.5em 0; }
.fp-md :deep(li) { margin: 0.25em 0; }
.fp-md :deep(code) { background: var(--nr-border, #eee); padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.9em; }
.fp-md :deep(pre) { background: var(--nr-canvas, #faf7f2); padding: 10px 12px; border-radius: 6px; overflow-x: auto; }
.fp-md :deep(pre code) { background: none; padding: 0; }
.fp-md :deep(a) { color: var(--nr-clay, #b5651d); }
.fp-md :deep(blockquote) { margin: 0.5em 0; padding-left: 1em; border-left: 3px solid var(--nr-border); color: var(--nr-ink-3); }
.fp-md :deep(table) { border-collapse: collapse; margin: 0.5em 0; }
.fp-md :deep(th), .fp-md :deep(td) { border: 1px solid var(--nr-border); padding: 4px 8px; }
.fp-md :deep(img) { max-width: 100%; }
.fp-md :deep(hr) { border: none; border-top: 1px solid var(--nr-border); margin: 1em 0; }
.fp-actions { margin-top: 12px; display: flex; justify-content: flex-end; }
</style>
