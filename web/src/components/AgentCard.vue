<template>
  <div class="agent-card">
    <div class="card-header">
      <div>
        <h3 class="agent-name">{{ agent.name }}</h3>
        <p class="agent-desc">{{ agent.description || '暂无描述' }}</p>
      </div>
      <a-tag v-if="agent.is_default" color="blue">默认</a-tag>
    </div>

    <!-- 能力徽章 -->
    <div class="capabilities">
      <a-tag v-if="caps.streaming" color="green">流式</a-tag>
      <a-tag v-if="caps.web_search" color="blue">联网</a-tag>
      <a-tag v-if="caps.code_execution" color="orange">代码执行</a-tag>
      <a-tag v-if="caps.knowledge_base" color="purple">知识库</a-tag>
      <a-tag v-if="caps.deep_research" color="red">深度研究</a-tag>
    </div>

    <!-- Skills -->
    <a-collapse v-if="agent.skills?.length" ghost style="margin: 8px 0">
      <a-collapse-panel key="skills" header="Skills">
        <div v-for="skill in agent.skills" :key="skill.id" class="skill-item">
          <span class="skill-name">{{ skill.name }}</span>
          <a-tag :color="skill.enabled ? 'green' : 'default'" size="small">
            {{ skill.enabled ? '启用' : '停用' }}
          </a-tag>
        </div>
      </a-collapse-panel>
    </a-collapse>

    <!-- 统计 -->
    <div class="stats">
      <div class="stat"><span class="stat-num">{{ agent.stats?.total_conversations ?? 0 }}</span><span class="stat-label">对话</span></div>
      <div class="stat"><span class="stat-num">{{ agent.stats?.total_runs ?? 0 }}</span><span class="stat-label">运行</span></div>
      <div class="stat">
        <span class="stat-num">{{ agent.stats?.avg_run_duration_ms ? (agent.stats.avg_run_duration_ms / 1000).toFixed(1) + 's' : '-' }}</span>
        <span class="stat-label">平均耗时</span>
      </div>
    </div>

    <!-- 操作 -->
    <div class="card-actions">
      <a-button type="primary" size="small" @click="$emit('chat', agent)">开始对话</a-button>
      <a-button size="small" @click="$emit('detail', agent)">详情</a-button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({ agent: { type: Object, required: true } })
defineEmits(['chat', 'detail'])

const caps = computed(() => props.agent.capabilities || {})
</script>

<style scoped>
.agent-card {
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 12px;
  padding: 20px;
  transition: box-shadow 0.2s;
}
.agent-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
.card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
.agent-name { font-size: 16px; font-weight: 600; margin: 0 0 4px; }
.agent-desc { color: #888; font-size: 13px; margin: 0; }
.capabilities { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 4px; }
.skill-item { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; }
.skill-name { font-size: 13px; }
.stats { display: flex; gap: 24px; padding: 12px 0; border-top: 1px solid #f0f0f0; margin-top: 8px; }
.stat { display: flex; flex-direction: column; align-items: center; }
.stat-num { font-size: 18px; font-weight: 600; color: #1677ff; }
.stat-label { font-size: 11px; color: #999; }
.card-actions { display: flex; gap: 8px; margin-top: 12px; }
</style>
