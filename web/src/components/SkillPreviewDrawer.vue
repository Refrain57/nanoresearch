<template>
  <a-drawer
    :open="open"
    :title="skill?.name || slug"
    width="600"
    @close="$emit('update:open', false)"
  >
    <a-spin :spinning="loading">
      <template v-if="skill">
        <div class="trust-row">
          <a-tag>作者 @{{ skill.owner || skill.slug }}</a-tag>
          <a-tag v-if="skill.version">v{{ skill.version }}</a-tag>
          <a-tag v-if="skill.stats?.stars != null">★ {{ skill.stats.stars }}</a-tag>
          <a-tag :color="blocked ? 'red' : 'green'">
            审核: {{ skill.moderation?.state || 'unknown' }}
          </a-tag>
          <a-tag v-if="skill.has_scripts" color="orange">包含可执行脚本</a-tag>
        </div>

        <a-alert
          v-if="blocked"
          type="error"
          show-icon
          message="该 skill 已被市场标记，禁止安装"
          style="margin: 12px 0"
        />
        <a-alert
          v-else-if="skill.has_scripts"
          type="warning"
          show-icon
          message="此 skill 附带脚本文件，安装后 Agent 可能执行它们。请先阅读下方内容。"
          style="margin: 12px 0"
        />

        <div v-if="skill.files?.length" class="files">
          <div class="section-label">文件</div>
          <a-tag v-for="f in skill.files" :key="f" size="small">{{ f }}</a-tag>
        </div>

        <div class="section-label">SKILL.md</div>
        <div class="readme" v-html="readmeHtml"></div>
      </template>
    </a-spin>

    <template #footer>
      <a-space>
        <a-button @click="$emit('update:open', false)">取消</a-button>
        <a-popconfirm
          title="安装到你的工作区？安装后需新开会话才会加载。"
          ok-text="安装"
          @confirm="doInstall"
        >
          <a-button type="primary" :disabled="blocked" :loading="installing">安装</a-button>
        </a-popconfirm>
      </a-space>
    </template>
  </a-drawer>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { message } from 'ant-design-vue'
import { getMarketSkill, getMarketReadme, installSkill } from '@/apis/skills'

const props = defineProps({ open: Boolean, slug: String })
const emit = defineEmits(['update:open', 'installed'])

const loading = ref(false)
const installing = ref(false)
const skill = ref(null)
const readmeHtml = ref('')

const blocked = computed(() =>
  ['flagged', 'removed'].includes(skill.value?.moderation?.state)
)

watch(
  () => [props.open, props.slug],
  async ([open, slug]) => {
    if (!open || !slug) return
    loading.value = true
    skill.value = null
    readmeHtml.value = ''
    try {
      const [meta, readme] = await Promise.all([
        getMarketSkill(slug),
        getMarketReadme(slug).catch(() => ({ content: '' })),
      ])
      skill.value = meta
      readmeHtml.value = DOMPurify.sanitize(marked(readme.content || ''))
    } catch (e) {
      message.error(e.message || '加载 skill 详情失败')
      emit('update:open', false)
    } finally {
      loading.value = false
    }
  },
  { immediate: true }
)

async function doInstall() {
  installing.value = true
  try {
    await installSkill(props.slug)
    message.success('已安装，新开会话后生效')
    emit('installed', props.slug)
    emit('update:open', false)
  } catch (e) {
    message.error(e.message || '安装失败')
  } finally {
    installing.value = false
  }
}
</script>

<style scoped>
.trust-row { display: flex; flex-wrap: wrap; gap: 6px; }
.section-label { font-weight: 600; margin: 14px 0 6px; }
.files { margin-top: 12px; }
.readme { font-size: 13px; line-height: 1.6; }
.readme :deep(pre) { background: rgba(0,0,0,0.04); padding: 10px; border-radius: 6px; overflow: auto; }
</style>
