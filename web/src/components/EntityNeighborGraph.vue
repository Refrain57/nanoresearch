<template>
  <svg :width="size" :height="size" class="eng">
    <line v-for="(p, i) in points" :key="'l'+i" :x1="c" :y1="c" :x2="p.x" :y2="p.y" class="eng-edge" />
    <g>
      <circle :cx="c" :cy="c" r="26" class="eng-center" />
      <text :x="c" :y="c" class="eng-label eng-center-label">{{ trunc(center) }}</text>
    </g>
    <g v-for="(p, i) in points" :key="'n'+i" class="eng-node" @click="$emit('select', p.name)">
      <circle :cx="p.x" :cy="p.y" r="20" />
      <text :x="p.x" :y="p.y" class="eng-label">{{ trunc(p.name) }}</text>
    </g>
  </svg>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({
  center: { type: String, default: '' },
  neighbors: { type: Array, default: () => [] },
})
defineEmits(['select'])
const size = 360
const c = size / 2
const R = 130
const points = computed(() => {
  const ns = props.neighbors.slice(0, 10)
  return ns.map((name, i) => {
    const a = (2 * Math.PI * i) / ns.length - Math.PI / 2
    return { name, x: c + R * Math.cos(a), y: c + R * Math.sin(a) }
  })
})
const trunc = (s) => (s && s.length > 8 ? s.slice(0, 7) + '…' : s)
</script>

<style scoped>
.eng { max-width: 100%; }
.eng-edge { stroke: #ddd; stroke-width: 1; }
.eng-center { fill: #C15F3C; }
.eng-node circle { fill: #5E7355; cursor: pointer; }
.eng-node:hover circle { fill: #6f875f; }
.eng-label { fill: #fff; font-size: 11px; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }
.eng-center-label { font-weight: 600; }
</style>
