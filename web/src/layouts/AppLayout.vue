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
      </a-menu>

      <div class="sider-footer">
        <a-button type="text" @click="logout" class="logout-btn">
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
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { CommentOutlined, RobotOutlined, LogoutOutlined } from '@ant-design/icons-vue'
import { useUserStore } from '@/stores/user'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()
const collapsed = ref(false)

const activeKey = computed(() => {
  if (route.path.startsWith('/agents')) return '/agents'
  return '/chat'
})

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
  bottom: 48px; /* above the default collapse trigger */
  width: 100%;
  padding: 8px;
  border-top: 1px solid rgba(255,255,255,0.1);
}
.logout-btn {
  color: rgba(255,255,255,0.65);
  width: 100%;
  text-align: left;
}
.logout-btn:hover { color: #fff; }
</style>
