<template>
  <app-layout>
    <div class="knowledge-page">
      <div class="page-header">
        <h2>知识库</h2>
        <a-tooltip :title="settingsStore.coverage.hasEmbedding ? '' : '缺 embedding provider，请到 Settings 添加'">
          <a-button type="primary" :disabled="!settingsStore.coverage.hasEmbedding" @click="openCreate">
            <plus-outlined /> 新建知识库
          </a-button>
        </a-tooltip>
      </div>

      <a-spin :spinning="kbStore.loading">
        <div v-if="kbStore.kbs.length" class="kb-grid">
          <div v-for="kb in kbStore.kbs" :key="kb.id" class="kb-card-wrap">
            <div class="kb-card" @click="router.push(`/knowledge/${kb.id}`)">
              <div class="kb-card-header">
                <database-outlined class="kb-icon" />
                <span class="kb-name">{{ kb.name }}</span>
              </div>
              <div class="kb-desc">{{ kb.description || '暂无描述' }}</div>
              <div class="kb-stats">
                <span class="stat"><file-text-outlined /> {{ kb.doc_count }} 篇文档</span>
                <span class="stat"><database-outlined /> {{ kb.chunk_count }} 个 Chunk</span>
              </div>
              <div class="kb-footer">
                <a-tag :color="kb.status === 'active' ? 'green' : 'default'" size="small">
                  {{ kb.status === 'active' ? '活跃' : kb.status }}
                </a-tag>
              </div>
            </div>
            <a-popconfirm
              title="确定删除此知识库？所有文档和向量将被清除。"
              ok-text="删除"
              cancel-text="取消"
              ok-type="danger"
              @confirm="handleDelete(kb)"
            >
              <a-button class="delete-btn" size="small" danger type="text" @click.stop>
                <delete-outlined />
              </a-button>
            </a-popconfirm>
          </div>
        </div>
        <a-empty v-else description="暂无知识库" style="margin-top: 80px" />
      </a-spin>
    </div>

    <a-modal
      v-model:open="createOpen"
      title="新建知识库"
      @ok="handleCreate"
      :confirm-loading="creating"
      width="480"
    >
      <a-form :model="form" layout="vertical" style="margin-top: 16px">
        <a-form-item label="名称" required>
          <a-input v-model:value="form.name" placeholder="例：论文库" />
        </a-form-item>
        <a-form-item label="描述">
          <a-textarea v-model:value="form.description" :rows="2" placeholder="可选" />
        </a-form-item>
        <a-form-item label="分块策略">
          <a-select v-model:value="form.chunk_strategy">
            <a-select-option value="auto">自动检测（推荐）</a-select-option>
            <a-select-option value="semantic">语义分块（嵌入聚类）</a-select-option>
            <a-select-option value="fixed">固定大小</a-select-option>
            <a-select-option value="structured">结构感知（标题/章节）</a-select-option>
          </a-select>
          <div style="font-size: 12px; color: #999; margin-top: 4px">
            自动检测：根据文档结构自动选择；语义分块：按语义相似度聚类切分；固定大小：按 token 数均匀切分；结构感知：按标题层级保留文档结构
          </div>
        </a-form-item>
        <a-form-item label="Embedding 模型">
          <a-auto-complete
            v-model:value="form.embedding_model"
            :options="embeddingOptions"
            placeholder="留空使用默认配置"
            allow-clear
            style="width: 100%"
          />
          <div style="font-size: 12px; color: #999; margin-top: 4px">
            选择或输入 Embedding 模型名称，留空则使用全局默认配置
          </div>
        </a-form-item>
      </a-form>
    </a-modal>
  </app-layout>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { PlusOutlined, DeleteOutlined, DatabaseOutlined, FileTextOutlined } from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import { useKnowledgeStore } from '@/stores/knowledge'
import { useSettingsStore } from '@/stores/settings'
import { computed } from 'vue'

const router = useRouter()
const kbStore = useKnowledgeStore()
const settingsStore = useSettingsStore()

// Merge user-configured models with common embedding model presets
const EMBEDDING_PRESETS = [
  'text-embedding-v3', 'text-embedding-v2', 'text-embedding-ada-002',
  'text-embedding-3-small', 'text-embedding-3-large', 'bge-m3',
]
const embeddingOptions = computed(() => {
  const seen = new Set()
  const opts = []
  // User's provider models first
  for (const p of settingsStore.providers) {
    for (const m of (p.models || [])) {
      if (!seen.has(m)) { seen.add(m); opts.push({ value: m, label: m }) }
    }
  }
  // Common embedding presets as fallback
  for (const m of EMBEDDING_PRESETS) {
    if (!seen.has(m)) { seen.add(m); opts.push({ value: m, label: m }) }
  }
  return opts
})

const createOpen = ref(false)
const creating = ref(false)
const form = reactive({ name: '', description: '', chunk_strategy: 'auto', embedding_model: '' })

onMounted(() => { kbStore.fetchList(); settingsStore.fetchAll() })

function openCreate() {
  Object.assign(form, { name: '', description: '', chunk_strategy: 'auto', embedding_model: '' })
  createOpen.value = true
}

async function handleCreate() {
  if (!form.name.trim()) { message.warning('请填写名称'); return }
  creating.value = true
  try {
    const kb = await kbStore.create({
      name: form.name,
      description: form.description || null,
      chunk_strategy: form.chunk_strategy,
      embedding_model: form.embedding_model || null,
    })
    createOpen.value = false
    message.success('创建成功')
    router.push(`/knowledge/${kb.id}`)
  } catch (e) {
    message.error(e.message || '创建失败')
  } finally {
    creating.value = false
  }
}

async function handleDelete(kb) {
  try {
    await kbStore.remove(kb.id)
    message.success('已删除')
  } catch (e) {
    message.error(e.message || '删除失败')
  }
}
</script>

<style scoped>
.knowledge-page { padding: 32px; }
.page-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.page-header h2 { font-size: 22px; font-weight: 700; margin: 0; }

.kb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.kb-card-wrap { position: relative; }
.delete-btn { position: absolute; top: 8px; right: 8px; z-index: 1; opacity: 0; transition: opacity 0.15s; }
.kb-card-wrap:hover .delete-btn { opacity: 1; }

.kb-card {
  background: #fff;
  border: 1px solid #f0f0f0;
  border-radius: 12px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 140px;
}
.kb-card:hover { border-color: #1677ff; box-shadow: 0 4px 16px rgba(22, 119, 255, 0.1); }

.kb-card-header { display: flex; align-items: center; gap: 10px; }
.kb-icon { font-size: 20px; color: #1677ff; }
.kb-name { font-size: 16px; font-weight: 700; color: #111; }

.kb-desc { font-size: 13px; color: #666; line-height: 1.5; flex: 1;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

.kb-stats { display: flex; gap: 16px; }
.stat { font-size: 12px; color: #888; display: flex; align-items: center; gap: 4px; }

.kb-footer { display: flex; align-items: center; justify-content: space-between; }
</style>
