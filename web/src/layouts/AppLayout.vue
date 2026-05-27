<template>
  <a-layout style="min-height: 100vh">
    <a-layout-sider v-model:collapsed="collapsed" collapsible width="220" theme="dark">
      <div class="logo">{{ collapsed ? 'NR' : 'Nano Research' }}</div>

      <a-menu theme="dark" mode="inline" :selected-keys="[activeKey]" @click="navigate">
        <a-menu-item key="/chat">
          <comment-outlined />
          <span>对话</span>
        </a-menu-item>
        <a-menu-item key="/agents">
          <robot-outlined />
          <span>Agent</span>
        </a-menu-item>
        <a-menu-item key="/knowledge">
          <database-outlined />
          <span>知识库</span>
        </a-menu-item>
      </a-menu>

      <div class="sider-footer">
        <a-button type="text" class="footer-btn" @click="openSettings">
          <setting-outlined />
          <span v-if="!collapsed">供应商设置</span>
        </a-button>
        <a-button type="text" @click="logout" class="footer-btn">
          <logout-outlined />
          <span v-if="!collapsed">退出</span>
        </a-button>
      </div>
    </a-layout-sider>

    <a-layout>
      <a-layout-content>
        <slot />
      </a-layout-content>
    </a-layout>
  </a-layout>

  <!-- ============================================================
       供应商 & 评估设置 Drawer
  ============================================================ -->
  <a-drawer
    v-model:open="settingsOpen"
    title="供应商 & 评估设置"
    placement="left"
    width="400"
    :body-style="{ paddingTop: '16px' }"
  >
    <a-spin :spinning="settingsStore.loading">

      <!-- ── 供应商 ── -->
      <div class="section-header">
        <span class="section-title">API 供应商</span>
        <a-button type="link" size="small" @click="openProviderModal(null)">
          <plus-outlined /> 添加
        </a-button>
      </div>
      <div class="field-hint" style="margin-bottom: 10px">
        配置供应商后，模型名称可在 Agent 页面的模型选择中使用
      </div>

      <div v-if="settingsStore.providers.length === 0" class="empty-providers">
        尚未配置供应商，点击右侧「添加」
      </div>

      <div class="provider-list">
        <div v-for="p in settingsStore.providers" :key="p.id" class="provider-card">
          <div class="provider-card-body">
            <div class="provider-name">{{ p.name }}</div>
            <div class="provider-meta">
              <a-tag v-if="p.api_key_set" color="green" size="small">Key 已配置</a-tag>
              <span v-if="p.api_base" class="provider-base">{{ p.api_base }}</span>
            </div>
            <div v-if="p.models.length" class="provider-models">
              {{ p.models.join(' · ') }}
            </div>
          </div>
          <div class="provider-card-actions">
            <a-button type="text" size="small" @click="openProviderModal(p)">
              <edit-outlined />
            </a-button>
            <a-popconfirm title="确认删除？" ok-text="删除" cancel-text="取消" @confirm="deleteProvider(p.id)">
              <a-button type="text" size="small" danger>
                <delete-outlined />
              </a-button>
            </a-popconfirm>
          </div>
        </div>
      </div>

      <a-divider />

      <!-- ── RAGAS 离线评估 ── -->
      <div class="section-title" style="margin-bottom: 12px">RAGAS 离线评估模型</div>

      <a-form layout="vertical">
        <a-form-item label="Generator 模型">
          <a-auto-complete
            v-model:value="localRagasGen"
            :options="settingsStore.allModelOptions"
            placeholder="qwen-plus（默认）"
            allow-clear
            style="width: 100%"
          />
          <div class="field-hint">生成候选答案，推荐性价比高的模型</div>
        </a-form-item>
        <a-form-item label="Evaluator 模型">
          <a-auto-complete
            v-model:value="localRagasEval"
            :options="settingsStore.allModelOptions"
            placeholder="qwen-max（默认）"
            allow-clear
            style="width: 100%"
          />
          <div class="field-hint">评估 Faithfulness / Recall，推荐能力强的模型</div>
        </a-form-item>
        <a-form-item label="Embedding 模型">
          <a-auto-complete
            v-model:value="localRagasEmb"
            :options="embeddingOptions"
            placeholder="text-embedding-v3（默认）"
            allow-clear
            style="width: 100%"
          />
          <div class="field-hint">AnswerRelevancy 语义相似度所用向量模型</div>
        </a-form-item>
      </a-form>

      <a-button
        type="primary"
        block
        :loading="ragasSaving"
        @click="saveRagasSettings"
        style="margin-top: 4px"
      >
        保存评估配置
      </a-button>

    </a-spin>
  </a-drawer>

  <!-- ============================================================
       供应商编辑 Modal
  ============================================================ -->
  <a-modal
    v-model:open="providerModalOpen"
    :title="editingProvider ? '编辑供应商' : '添加供应商'"
    ok-text="保存"
    cancel-text="取消"
    :confirm-loading="providerSaving"
    @ok="saveProvider"
    width="440"
    :destroy-on-close="true"
  >
    <a-form layout="vertical" style="margin-top: 16px">
      <a-form-item label="供应商名称" required>
        <a-input v-model:value="providerForm.name" placeholder="如 通义千问、OpenAI、DeepSeek" />
      </a-form-item>
      <a-form-item label="API Key">
        <a-input-password
          v-model:value="providerForm.api_key"
          :placeholder="editingProvider?.api_key_set
            ? editingProvider.api_key_hint + '（留空保持不变）'
            : '请输入 API Key'"
          autocomplete="new-password"
        />
      </a-form-item>
      <a-form-item label="API Base URL">
        <a-input
          v-model:value="providerForm.api_base"
          placeholder="如 https://dashscope.aliyuncs.com/compatible-mode/v1"
        />
      </a-form-item>
      <a-form-item label="可用模型">
        <a-select
          v-model:value="providerForm.models"
          mode="tags"
          placeholder="输入模型名后按 Enter，如 qwen-plus"
          style="width: 100%"
          :token-separators="[',']"
        />
        <div class="field-hint">这些模型将出现在 Agent 的模型选择下拉框中</div>
      </a-form-item>
    </a-form>
  </a-modal>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  CommentOutlined, RobotOutlined, LogoutOutlined, DatabaseOutlined,
  SettingOutlined, PlusOutlined, EditOutlined, DeleteOutlined,
} from '@ant-design/icons-vue'
import { useUserStore } from '@/stores/user'
import { useSettingsStore } from '@/stores/settings'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()
const settingsStore = useSettingsStore()
const collapsed = ref(false)
const settingsOpen = ref(false)

// RAGAS local state
const localRagasGen  = ref(null)
const localRagasEval = ref(null)
const localRagasEmb  = ref(null)
const ragasSaving    = ref(false)

const embeddingOptions = [
  { value: 'text-embedding-v3', label: 'text-embedding-v3' },
  { value: 'text-embedding-v2', label: 'text-embedding-v2' },
  { value: 'text-embedding-ada-002', label: 'text-embedding-ada-002' },
]

function syncRagas() {
  localRagasGen.value  = settingsStore.ragasGeneratorModel
  localRagasEval.value = settingsStore.ragasEvaluatorModel
  localRagasEmb.value  = settingsStore.ragasEmbeddingModel
}

async function openSettings() {
  settingsOpen.value = true
  await settingsStore.fetchAll()
  syncRagas()
}

async function saveRagasSettings() {
  ragasSaving.value = true
  try {
    await settingsStore.saveRagasSettings({
      generatorModel: localRagasGen.value,
      evaluatorModel: localRagasEval.value,
      embeddingModel: localRagasEmb.value,
    })
    message.success('评估配置已保存')
  } catch (e) {
    message.error('保存失败：' + (e.message || '未知错误'))
  } finally {
    ragasSaving.value = false
  }
}

// ── Provider modal ──
const providerModalOpen = ref(false)
const providerSaving    = ref(false)
const editingProvider   = ref(null)
const providerForm      = ref({ name: '', api_key: '', api_base: '', models: [] })

function openProviderModal(p) {
  editingProvider.value = p
  providerForm.value = p
    ? { name: p.name, api_key: '', api_base: p.api_base || '', models: [...(p.models || [])] }
    : { name: '', api_key: '', api_base: '', models: [] }
  providerModalOpen.value = true
}

async function saveProvider() {
  if (!providerForm.value.name.trim()) {
    message.warning('请填写供应商名称')
    return
  }
  providerSaving.value = true
  try {
    const existing = settingsStore.providers.map(p => ({
      id: p.id,
      name: p.name,
      api_key: null,        // null = keep stored key
      api_base: p.api_base,
      models: p.models,
    }))

    let next
    if (editingProvider.value) {
      next = existing.map(p =>
        p.id === editingProvider.value.id
          ? {
              id: p.id,
              name: providerForm.value.name,
              api_key: providerForm.value.api_key || null,
              api_base: providerForm.value.api_base || null,
              models: providerForm.value.models,
            }
          : p
      )
    } else {
      next = [
        ...existing,
        {
          name: providerForm.value.name,
          api_key: providerForm.value.api_key || null,
          api_base: providerForm.value.api_base || null,
          models: providerForm.value.models,
        },
      ]
    }

    await settingsStore.saveProviders(next)
    providerModalOpen.value = false
    message.success('供应商已保存')
  } catch (e) {
    message.error('保存失败：' + (e.message || '未知错误'))
  } finally {
    providerSaving.value = false
  }
}

async function deleteProvider(id) {
  try {
    const next = settingsStore.providers
      .filter(p => p.id !== id)
      .map(p => ({ id: p.id, name: p.name, api_key: null, api_base: p.api_base, models: p.models }))
    await settingsStore.saveProviders(next)
    message.success('已删除')
  } catch (e) {
    message.error('删除失败：' + (e.message || ''))
  }
}

const activeKey = computed(() => {
  if (route.path.startsWith('/agents')) return '/agents'
  if (route.path.startsWith('/knowledge')) return '/knowledge'
  return '/chat'
})

onMounted(() => settingsStore.fetchAll())

function navigate({ key }) { router.push(key) }
function logout() { userStore.logout(); router.push('/login') }
</script>

<style scoped>
.logo {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 16px;
  font-weight: 700;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  white-space: nowrap;
  overflow: hidden;
}
.sider-footer {
  position: absolute;
  bottom: 48px;
  width: 100%;
  padding: 8px;
  border-top: 1px solid rgba(255,255,255,0.1);
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.footer-btn {
  color: rgba(255,255,255,0.65);
  width: 100%;
  text-align: left;
}
.footer-btn:hover { color: #fff; }

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}
.section-title { font-size: 13px; font-weight: 600; color: #333; }
.empty-providers { font-size: 12px; color: #bbb; padding: 8px 0 12px; }

.provider-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 4px; }
.provider-card {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  background: #fafafa;
}
.provider-card-body { flex: 1; min-width: 0; }
.provider-name { font-size: 13px; font-weight: 600; color: #222; }
.provider-meta { display: flex; align-items: center; gap: 6px; margin-top: 4px; flex-wrap: wrap; }
.provider-base { font-size: 11px; color: #888; }
.provider-models { font-size: 11px; color: #aaa; margin-top: 3px; line-height: 1.6; }
.provider-card-actions { display: flex; gap: 2px; flex-shrink: 0; }

.field-hint { font-size: 11px; color: #aaa; margin-top: 4px; }
</style>
