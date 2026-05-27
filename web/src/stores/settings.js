import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getMySettings, updateMySettings, getAvailableModels } from '@/apis/settings'

export const useSettingsStore = defineStore('settings', () => {
  const providers = ref([])
  const ragasGeneratorModel = ref(null)
  const ragasEvaluatorModel = ref(null)
  const ragasEmbeddingModel = ref(null)
  const systemModels = ref([])
  const loading = ref(false)

  // All model names from all providers + system fallback
  const allModelOptions = computed(() => {
    const fromProviders = providers.value.flatMap(p => p.models || [])
    const fromSystem = systemModels.value
    const combined = [...new Set([...fromProviders, ...fromSystem])]
    return combined.map(m => ({ value: m, label: m }))
  })

  async function fetchAll() {
    loading.value = true
    try {
      const [s, m] = await Promise.all([getMySettings(), getAvailableModels()])
      providers.value = s.providers || []
      ragasGeneratorModel.value = s.ragas_generator_model
      ragasEvaluatorModel.value = s.ragas_evaluator_model
      ragasEmbeddingModel.value = s.ragas_embedding_model
      systemModels.value = m.models || []
    } finally {
      loading.value = false
    }
  }

  async function saveProviders(providerList) {
    const s = await updateMySettings({ providers: providerList })
    providers.value = s.providers || []
  }

  async function saveRagasSettings(data) {
    const s = await updateMySettings({
      ragas_generator_model: data.generatorModel || '',
      ragas_evaluator_model: data.evaluatorModel || '',
      ragas_embedding_model: data.embeddingModel || '',
    })
    ragasGeneratorModel.value = s.ragas_generator_model
    ragasEvaluatorModel.value = s.ragas_evaluator_model
    ragasEmbeddingModel.value = s.ragas_embedding_model
  }

  return {
    providers, systemModels, allModelOptions,
    ragasGeneratorModel, ragasEvaluatorModel, ragasEmbeddingModel,
    loading,
    fetchAll, saveProviders, saveRagasSettings,
  }
})
