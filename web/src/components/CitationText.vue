<template>
  <div class="citation-text-root" ref="rootEl">
    <div class="md-body" v-html="rendered" @click="onCiteClick" />
    <div v-if="activeCite" class="cite-popover"
         :style="{ left: citePos.x + 'px', top: citePos.y + 'px' }">
      <div class="cite-pop-head">
        <span class="cite-pop-src">{{ activeCite.source }}</span>
        <span v-if="activeCite.page != null" class="cite-pop-page">p.{{ activeCite.page }}</span>
      </div>
      <div class="cite-pop-snippet">{{ activeCite.snippet }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { marked } from 'marked'

marked.setOptions({ breaks: true, gfm: true })

const props = defineProps({
  text:      { type: String, default: '' },
  citations: { type: Array,  default: () => [] },
})

// Turn [^n] markers into clickable <sup> — only for n present in validIndices,
// and only outside <pre>/<code> regions (avoids matching arr[1] etc. in code).
function linkifyCitations(html, validIndices) {
  if (!validIndices || validIndices.size === 0) return html
  // Split so odd-index segments are the <pre>…</pre> / <code>…</code> blocks (left untouched).
  const parts = html.split(/(<pre[\s\S]*?<\/pre>|<code[\s\S]*?<\/code>)/gi)
  // Renumber by first appearance: the model emits [^n] where n = retrieval-rank
  // citation index (not reading order), so display 1,2,3… in order of first
  // appearance while keeping the real index in data-cite for the popover lookup.
  const labelByIndex = new Map()
  let nextLabel = 1
  return parts.map((seg, i) => {
    if (i % 2 === 1) return seg
    return seg.replace(/\[\^(\d+)\]/g, (m, d) => {
      const n = Number(d)
      if (!validIndices.has(n)) return m
      let label = labelByIndex.get(n)
      if (label === undefined) { label = nextLabel++; labelByIndex.set(n, label) }
      return `<sup class="cite-ref" data-cite="${n}">[${label}]</sup>`
    })
  }).join('')
}

function renderMd(text, citations) {
  if (!text) return ''
  const html = marked.parse(text)
  const valid = new Set((citations || []).map(c => c.index))
  return linkifyCitations(html, valid)
}

const rendered = computed(() => renderMd(props.text, props.citations))

const rootEl     = ref(null)
const activeCite = ref(null)
const citePos    = ref({ x: 0, y: 0 })

function onCiteClick(e) {
  const el = e.target.closest('.cite-ref')
  if (!el) return
  const n = Number(el.dataset.cite)
  const cite = (props.citations || []).find(c => c.index === n)
  if (!cite) return
  const r = el.getBoundingClientRect()
  citePos.value = { x: r.left, y: r.bottom + 4 }
  activeCite.value = cite
}

function closeCite() { activeCite.value = null }
function onDocClick(e) {
  if (rootEl.value && rootEl.value.contains(e.target) && e.target.closest('.cite-ref')) return
  if (e.target.closest('.cite-popover')) return
  closeCite()
}
function onKey(e) { if (e.key === 'Escape') closeCite() }

onMounted(() => {
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
  document.removeEventListener('keydown', onKey)
})
</script>

<style scoped>
.citation-text-root { position: relative; }

.md-body { line-height: 1.7; }
.md-body :deep(.cite-ref) {
  color: var(--nr-clay);
  cursor: pointer;
  font-weight: 700;
  font-size: 0.72em;
  padding: 0 1px;
  user-select: none;
}
.md-body :deep(.cite-ref:hover) { text-decoration: underline; }
.md-body :deep(p) { margin: 0 0 8px; }
.md-body :deep(p:last-child) { margin-bottom: 0; }
.md-body :deep(h1),:deep(h2),:deep(h3),:deep(h4) { margin: 12px 0 6px; font-weight: 700; line-height: 1.3; }
.md-body :deep(h1) { font-size: 1.25em; }
.md-body :deep(h2) { font-size: 1.1em; }
.md-body :deep(h3) { font-size: 1em; }
.md-body :deep(ul),:deep(ol) { margin: 6px 0; padding-left: 20px; }
.md-body :deep(li) { margin: 2px 0; }
.md-body :deep(code) { background: rgba(0,0,0,0.07); border-radius: 3px; padding: 1px 5px; font-family: monospace; font-size: 0.9em; }
.md-body :deep(pre) { background: #2A2622; color: #E8E2D6; border-radius: 6px; padding: 12px 14px; overflow-x: auto; margin: 8px 0; }
.md-body :deep(pre code) { background: none; padding: 0; font-size: 0.88em; }
.md-body :deep(blockquote) { border-left: 3px solid var(--nr-border-strong); margin: 6px 0; padding: 2px 12px; color: var(--nr-ink-2); }
.md-body :deep(table) { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 0.9em; }
.md-body :deep(th),:deep(td) { border: 1px solid var(--nr-border); padding: 5px 10px; }
.md-body :deep(th) { background: var(--nr-rail); font-weight: 600; }
.md-body :deep(a) { color: var(--nr-clay); text-decoration: none; }
.md-body :deep(a:hover) { text-decoration: underline; }
.md-body :deep(hr) { border: none; border-top: 1px solid var(--nr-border); margin: 10px 0; }

.cite-popover {
  position: fixed;
  z-index: 50;
  max-width: 360px;
  background: var(--nr-card);
  border: 1px solid var(--nr-border-strong);
  border-radius: 8px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.12);
  padding: 8px 10px;
}
.cite-pop-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.cite-pop-src { font-size: 12px; font-weight: 600; color: var(--nr-ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cite-pop-page { font-size: 11px; color: var(--nr-ink-2); flex-shrink: 0; }
.cite-pop-snippet { font-size: 12px; color: var(--nr-ink-2); line-height: 1.5; max-height: 160px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
</style>
