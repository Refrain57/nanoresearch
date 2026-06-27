import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getMySettings, updateMySettings } from '@/apis/settings'

export const useSettingsStore = defineStore('settings', () => {
  const providers = ref([])
  const baseModel = ref(null)
  const ragasGeneratorModel = ref(null)
  const ragasEvaluatorModel = ref(null)
  const ragasEmbeddingModel = ref(null)
  const loading = ref(false)

  // All model names from user-configured providers only
  const allModelOptions = computed(() => {
    const fromProviders = providers.value.flatMap(p => p.models || [])
    return [...new Set(fromProviders)].map(m => ({ value: m, label: m }))
  })

  async function fetchAll() {
    loading.value = true
    try {
      const s = await getMySettings()
      providers.value = s.providers || []
      baseModel.value = s.model || null
      ragasGeneratorModel.value = s.ragas_generator_model
      ragasEvaluatorModel.value = s.ragas_evaluator_model
      ragasEmbeddingModel.value = s.ragas_embedding_model
    } finally {
      loading.value = false
    }
  }

  async function saveProviders(providerList) {
    const s = await updateMySettings({ providers: providerList })
    providers.value = s.providers || []
  }

  async function saveBaseModel(model) {
    const s = await updateMySettings({ model: model || '' })
    baseModel.value = s.model || null
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
    providers, allModelOptions, baseModel,
    ragasGeneratorModel, ragasEvaluatorModel, ragasEmbeddingModel,
    loading,
    fetchAll, saveProviders, saveBaseModel, saveRagasSettings,
  }
})
