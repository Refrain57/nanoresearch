<template>
  <a-layout style="min-height: 100vh">
    <a-layout-sider v-model:collapsed="collapsed" collapsible width="220" theme="light">
      <div class="logo">{{ collapsed ? 'NR' : 'Nano Research' }}</div>

      <a-menu theme="light" mode="inline" :selected-keys="[activeKey]" @click="navigate">
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
        <a-menu-item key="/eval/agent">
          <experiment-outlined />
          <span>评测</span>
        </a-menu-item>
      </a-menu>

      <div class="sider-footer">
        <a-button type="text" class="footer-btn" @click="openSettings">
          <setting-outlined />
          <span v-if="!collapsed">系统设置</span>
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
       系统设置 Drawer
  ============================================================ -->
  <a-drawer
    v-model:open="settingsOpen"
    title="系统设置"
    placement="left"
    width="480"
    :body-style="{ paddingTop: '8px', paddingLeft: '16px', paddingRight: '16px' }"
  >
    <a-tabs v-model:activeKey="settingsTab">

      <!-- ── Tab 1: API 供应商 ── -->
      <a-tab-pane key="providers" tab="API 供应商">
        <a-spin :spinning="settingsStore.loading">
          <div class="section-header" style="margin-top: 8px">
            <span class="section-title">API 供应商</span>
            <a-button type="link" size="small" @click="openProviderModal(null)">
              <plus-outlined /> 添加
            </a-button>
          </div>
          <div class="field-hint" style="margin-bottom: 10px">
            配置供应商后，模型名称可在 Agent 页面的模型选择中使用
          </div>

          <a-alert
            type="info"
            show-icon
            style="margin-bottom: 12px; font-size: 12px"
            message="第一步：添加 API key"
            description="不同模型用途可以共用同一个 key，也可以分配不同 provider。下方「模型用途分配」决定每种调用走哪个 key。"
          />

          <div v-if="settingsStore.providers.length === 0" class="empty-providers">
            尚未配置供应商，点击右侧「添加」
          </div>

          <div class="provider-list">
            <div v-for="p in settingsStore.providers" :key="p.id" class="provider-card">
              <div class="provider-card-body">
                <div class="provider-name">
                  {{ p.name }}
                  <a-tag v-if="p.provider" size="small" style="margin-left: 6px">
                    {{ p.provider }}
                  </a-tag>
                </div>
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
        </a-spin>

        <!-- 模型用途分配 -->
        <div class="section-header" style="margin-top: 24px">
          <span class="section-title">模型用途分配</span>
        </div>
        <div class="field-hint" style="margin-bottom: 10px">
          每种调用使用哪个 provider + 哪个模型。留空时按 fallback 规则处理。
        </div>
        <div v-if="settingsStore.providers.length === 0" class="empty-providers">
          先添加 API key，再分配用途
        </div>
        <div v-else class="role-assignment-list">
          <div v-for="role in ROLE_LABELS" :key="role.key" class="role-row">
            <div class="role-label">
              <span class="role-title">{{ role.label }}</span>
              <span class="role-hint">{{ role.hint }}</span>
            </div>
            <div class="role-controls">
              <a-select
                :value="settingsStore.roles[role.key]?.provider_id || null"
                :options="providerSelectOptions"
                placeholder="未配置"
                allow-clear
                class="role-control"
                @change="(pid) => onRoleProviderChange(role.key, pid)"
              />
              <a-auto-complete
                :value="settingsStore.roles[role.key]?.model || ''"
                :options="modelOptionsForRole(role.key)"
                placeholder="模型名"
                allow-clear
                class="role-control"
                @change="(m) => onRoleModelChange(role.key, m)"
              />
            </div>
          </div>
        </div>

        <!-- Base 模型选择 -->
        <div class="section-header" style="margin-top: 24px">
          <span class="section-title">Base 模型</span>
        </div>
        <div class="field-hint" style="margin-bottom: 10px">
          设置全局默认模型，未绑定 Agent 或未指定模型时使用此设置
        </div>
        <div class="base-model-row">
          <a-auto-complete
            v-model:value="localBaseModel"
            :options="settingsStore.allModelOptions"
            placeholder="留空则根据供应商自动匹配"
            allow-clear
            style="width: 280px"
          />
          <a-button type="primary" size="small" :loading="baseModelSaving" @click="saveBaseModel">
            保存
          </a-button>
        </div>
      </a-tab-pane>

      <!-- ── Tab 2: 引导文件 ── -->
      <a-tab-pane key="bootstrap" tab="引导文件">
        <div style="margin-top: 8px">
          <div class="field-hint" style="margin-bottom: 12px">
            这些文件会注入到每次对话的系统提示词中（workspace 级别，对所有 Agent 生效）
          </div>
          <a-tabs v-model:activeKey="bootstrapTab" size="small" type="card">
            <a-tab-pane v-for="file in bootstrapFiles" :key="file.name" :tab="file.name">
              <div class="field-hint" style="margin: 8px 0">{{ file.description }}</div>
              <a-spin :spinning="file.loading">
                <a-textarea
                  v-model:value="file.content"
                  :rows="18"
                  :placeholder="file.placeholder"
                  style="font-family: monospace; font-size: 12px"
                />
              </a-spin>
              <div style="display: flex; align-items: center; gap: 10px; margin-top: 8px">
                <a-button type="primary" size="small" :loading="file.saving" @click="saveBootstrapFile(file)">
                  保存
                </a-button>
                <span v-if="file.saved" style="font-size: 12px; color: var(--nr-sage)">已保存</span>
              </div>
            </a-tab-pane>
          </a-tabs>
        </div>
      </a-tab-pane>

    </a-tabs>
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
      <a-form-item label="供应商类型" required>
        <a-select
          v-model:value="providerForm.provider"
          placeholder="选择供应商"
          :options="PROVIDER_PRESETS"
        />
      </a-form-item>
      <a-form-item label="自定义名称（备注）">
        <a-input v-model:value="providerForm.name" placeholder="如 我的 DeepSeek、团队 OpenAI" />
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
      <a-form-item :label="providerForm.provider === 'openai_compatible' ? 'API Base URL（必填）' : 'API Base URL'">
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
          :options="providerModelOptions"
        />
        <div class="field-hint">这些模型将出现在 Agent 的模型选择下拉框中</div>
      </a-form-item>
    </a-form>
  </a-modal>
</template>

<script setup>
import { ref, computed, onMounted, reactive } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  CommentOutlined, RobotOutlined, LogoutOutlined, DatabaseOutlined,
  SettingOutlined, PlusOutlined, EditOutlined, DeleteOutlined, ExperimentOutlined,
} from '@ant-design/icons-vue'
import { useUserStore } from '@/stores/user'
import { useSettingsStore } from '@/stores/settings'
import { getWorkspaceFile, updateWorkspaceFile } from '@/apis/workspace'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()
const settingsStore = useSettingsStore()
const collapsed = ref(false)
const settingsOpen = ref(false)
const settingsTab = ref('providers')
const bootstrapTab = ref('SOUL.md')

// ── Bootstrap files ──
const bootstrapFiles = reactive([
  {
    name: 'SOUL.md',
    description: '定义 AI 的身份、性格与价值观，是身份声明的唯一来源。',
    placeholder: '# 我是谁\n\n我是 nanobot，一个...',
    content: '', loading: false, saving: false, saved: false,
  },
  {
    name: 'AGENTS.md',
    description: '通用操作规范，适用于所有 Agent，包括协作模式、任务调度等。',
    placeholder: '# 操作规范\n\n## 任务调度\n...',
    content: '', loading: false, saving: false, saved: false,
  },
  {
    name: 'USER.md',
    description: '关于用户的背景信息、偏好和工作习惯。',
    placeholder: '# 用户信息\n\n## 背景\n...',
    content: '', loading: false, saving: false, saved: false,
  },
  {
    name: 'TOOLS.md',
    description: '工具使用补充说明和约束，与运行时工具列表配合使用。',
    placeholder: '# 工具使用说明\n\n## 文件操作\n...',
    content: '', loading: false, saving: false, saved: false,
  },
])

async function loadBootstrapFile(file) {
  file.loading = true
  try {
    file.content = await getWorkspaceFile(file.name)
  } catch (e) {
    if (e.status !== 404) message.error(`加载 ${file.name} 失败`)
    file.content = ''
  } finally {
    file.loading = false
  }
}

async function saveBootstrapFile(file) {
  file.saving = true
  file.saved = false
  try {
    await updateWorkspaceFile(file.name, file.content)
    file.saved = true
    setTimeout(() => { file.saved = false }, 2000)
  } catch (e) {
    message.error(`保存 ${file.name} 失败：` + (e.message || ''))
  } finally {
    file.saving = false
  }
}

const localBaseModel = ref('')
const baseModelSaving = ref(false)

async function saveBaseModel() {
  baseModelSaving.value = true
  try {
    await settingsStore.saveBaseModel(localBaseModel.value)
    message.success('Base 模型已保存')
  } catch (e) {
    message.error('保存失败：' + (e.message || '未知错误'))
  } finally {
    baseModelSaving.value = false
  }
}

async function openSettings() {
  settingsOpen.value = true
  await settingsStore.fetchAll()
  localBaseModel.value = settingsStore.baseModel || ''
  bootstrapFiles.forEach(loadBootstrapFile)
}

// ── Provider modal ──
const providerModalOpen = ref(false)
const providerSaving    = ref(false)
const editingProvider   = ref(null)
const providerForm      = ref({ provider: '', name: '', api_key: '', api_base: '', models: [] })

const PROVIDER_PRESETS = [
  { value: 'deepseek',          label: 'DeepSeek' },
  { value: 'openai',            label: 'OpenAI' },
  { value: 'anthropic',         label: 'Anthropic' },
  { value: 'dashscope',         label: '通义千问 (DashScope)' },
  { value: 'azure_openai',      label: 'Azure OpenAI' },
  { value: 'siliconflow',       label: 'SiliconFlow' },
  { value: 'openai_compatible', label: 'OpenAI 兼容 (自定义)' },
]

const ROLE_LABELS = [
  { key: 'chat',            label: '聊天 (chat)',           hint: '默认对话模型' },
  { key: 'ingestion_llm',   label: 'RAG 摄取',              hint: '处理知识库文档时使用' },
  { key: 'embedding',       label: '向量嵌入',              hint: '知识库检索需要' },
  { key: 'vision',          label: '视觉',                  hint: '图片理解；留空则关闭' },
  { key: 'eval_generator',  label: '评测 - 题目生成',       hint: '留空 fallback 到聊天模型' },
  { key: 'eval_evaluator',  label: '评测 - 打分',           hint: '留空 fallback 到聊天模型' },
]
const EMBEDDING_CAPABLE_PRESETS = new Set(['dashscope', 'openai', 'azure_openai', 'siliconflow'])

const PROVIDER_MODEL_PRESETS = {
  deepseek:  ['deepseek-v4-flash', 'deepseek-v4-pro', 'deepseek-reasoner', 'deepseek-v3', 'deepseek-r1'],
  openai:    ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'o3', 'o3-mini', 'o4-mini'],
  claude:    ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
  anthropic: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
  qwen:      ['qwen-max', 'qwen-plus', 'qwen-turbo', 'qwen-long', 'qwen-max-latest', 'qwen3-235b-a22b'],
  阿里云:    ['qwen-max', 'qwen-plus', 'qwen-turbo', 'qwen-long', 'qwen-max-latest', 'deepseek-v3', 'deepseek-r1'],
  dashscope: ['qwen-max', 'qwen-plus', 'qwen-turbo', 'deepseek-v3', 'deepseek-r1'],
  ollama:    ['llama3', 'llama3.1', 'mistral', 'mixtral', 'qwen2.5', 'deepseek-r1'],
  azure_openai:      [],
  siliconflow:       ['deepseek-v3', 'qwen-plus', 'BAAI/bge-large-zh-v1.5'],
  openai_compatible: [],
}

const providerModelOptions = computed(() => {
  const preset = providerForm.value.provider
  const presets = PROVIDER_MODEL_PRESETS[preset] || []
  return presets.map(m => ({ label: m, value: m }))
})

function openProviderModal(p) {
  editingProvider.value = p
  providerForm.value = p
    ? { provider: p.provider || '', name: p.name, api_key: '', api_base: p.api_base || '', models: [...(p.models || [])] }
    : { provider: '', name: '', api_key: '', api_base: '', models: [] }
  providerModalOpen.value = true
}

async function saveProvider() {
  if (!providerForm.value.name.trim()) {
    message.warning('请填写供应商名称')
    return
  }
  if (!providerForm.value.provider) {
    message.warning('请选择供应商类型')
    return
  }
  providerSaving.value = true
  try {
    const existing = settingsStore.providers.map(p => ({
      id: p.id, name: p.name, provider: p.provider, api_key: null, api_base: p.api_base, models: p.models,
    }))

    let next
    if (editingProvider.value) {
      next = existing.map(p =>
        p.id === editingProvider.value.id
          ? {
              id: p.id,
              name: providerForm.value.name,
              provider: providerForm.value.provider || null,
              api_key: providerForm.value.api_key || null,
              api_base: providerForm.value.api_base || null,
              models: providerForm.value.models,
            }
          : p
      )
    } else {
      next = [...existing, {
        name: providerForm.value.name,
        provider: providerForm.value.provider || null,
        api_key: providerForm.value.api_key || null,
        api_base: providerForm.value.api_base || null,
        models: providerForm.value.models,
      }]
    }

    await settingsStore.saveProviders(next)

    // Default role auto-assignment on first add or first embedding-capable add
    if (!editingProvider.value) {
      const updatedProviders = settingsStore.providers
      const added = updatedProviders.find(p =>
        p.name === providerForm.value.name && p.provider === (providerForm.value.provider || null)
      )
      if (added) {
        const nextRoles = { ...settingsStore.roles }
        let rolesChanged = false
        if (!nextRoles.chat) {
          nextRoles.chat = { provider_id: added.id, model: (added.models || [])[0] || '' }
          rolesChanged = true
        }
        if (!nextRoles.ingestion_llm) {
          nextRoles.ingestion_llm = { provider_id: added.id, model: (added.models || [])[0] || '' }
          rolesChanged = true
        }
        if (!nextRoles.embedding && EMBEDDING_CAPABLE_PRESETS.has(added.provider || '')) {
          const embModel = (added.models || []).find(m => /embed/i.test(m)) || ''
          nextRoles.embedding = { provider_id: added.id, model: embModel }
          rolesChanged = true
        }
        if (rolesChanged) {
          await settingsStore.saveRoles(nextRoles)
        }
      }
    }

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

const providerSelectOptions = computed(() =>
  settingsStore.providers.map(p => ({
    value: p.id,
    label: `${p.name}${p.provider ? ` (${p.provider})` : ''}`,
  }))
)

function modelOptionsForRole(roleKey) {
  const pid = settingsStore.roles[roleKey]?.provider_id
  if (!pid) return []
  const p = settingsStore.providers.find(x => x.id === pid)
  return (p?.models || []).map(m => ({ value: m, label: m }))
}

async function onRoleProviderChange(roleKey, providerId) {
  const next = { ...settingsStore.roles }
  if (!providerId) {
    next[roleKey] = null
  } else {
    const p = settingsStore.providers.find(x => x.id === providerId)
    const defaultModel = (p?.models || [])[0] || ''
    next[roleKey] = { provider_id: providerId, model: defaultModel }
  }
  try {
    await settingsStore.saveRoles(next)
  } catch (e) {
    message.error('保存失败：' + (e.message || ''))
  }
}

async function onRoleModelChange(roleKey, model) {
  const entry = settingsStore.roles[roleKey]
  if (!entry) return  // No provider chosen yet; the model field is disabled-ish
  const next = { ...settingsStore.roles, [roleKey]: { provider_id: entry.provider_id, model: model || '' } }
  try {
    await settingsStore.saveRoles(next)
  } catch (e) {
    message.error('保存失败：' + (e.message || ''))
  }
}

const activeKey = computed(() => {
  if (route.path.startsWith('/agents')) return '/agents'
  if (route.path.startsWith('/knowledge')) return '/knowledge'
  if (route.path.startsWith('/eval')) return '/eval/agent'
  return '/chat'
})

onMounted(() => settingsStore.fetchAll())

function navigate({ key }) { router.push(key) }
function logout() { userStore.logout(); router.push('/login') }
</script>

<style scoped>
.base-model-row { display: flex; align-items: center; gap: 10px; }
.logo {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--nr-ink);
  font-family: var(--nr-serif);
  font-size: 18px;
  font-weight: 500;
  letter-spacing: -.01em;
  border-bottom: 1px solid var(--nr-border);
  white-space: nowrap;
  overflow: hidden;
}
.sider-footer {
  position: absolute;
  bottom: 48px;
  width: 100%;
  padding: 8px;
  border-top: 1px solid var(--nr-border);
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.footer-btn {
  color: var(--nr-ink-2);
  width: 100%;
  text-align: left;
}
.footer-btn:hover { color: var(--nr-ink); }

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}
.section-title { font-size: 13px; font-weight: 600; color: var(--nr-ink); }
.empty-providers { font-size: 12px; color: var(--nr-ink-3); padding: 8px 0 12px; }

.provider-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 4px; }
.provider-card {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid var(--nr-border);
  border-radius: 8px;
  background: var(--nr-rail);
}
.provider-card-body { flex: 1; min-width: 0; }
.provider-name { font-size: 13px; font-weight: 600; color: var(--nr-ink); }
.provider-meta { display: flex; align-items: center; gap: 6px; margin-top: 4px; flex-wrap: wrap; }
.provider-base { font-size: 11px; color: var(--nr-ink-2); }
.provider-models { font-size: 11px; color: var(--nr-ink-3); margin-top: 3px; line-height: 1.6; }
.provider-card-actions { display: flex; gap: 2px; flex-shrink: 0; }

.field-hint { font-size: 11px; color: var(--nr-ink-3); margin-top: 4px; }

.role-assignment-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.role-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px 10px;
  background: var(--nr-rail);
  border-radius: 6px;
}
.role-label {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}
.role-title {
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
}
.role-hint {
  font-size: 11px;
  color: var(--nr-ink-2);
}
.role-controls {
  display: flex;
  gap: 8px;
}
.role-control {
  flex: 1;
  min-width: 0;
}
</style>
