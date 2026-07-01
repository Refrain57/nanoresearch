<template>
  <a-config-provider :theme="theme">
    <router-view />
  </a-config-provider>
</template>

<script setup>
// Anthropic 暖米白 + clay 全站主题 token（与 styles/theme.css 的 CSS 变量保持一致）
const theme = {
  token: {
    colorPrimary: '#C15F3C',
    colorInfo: '#C15F3C',
    colorSuccess: '#5E7355',
    colorWarning: '#9A7B2E',
    colorError: '#B04434',
    colorLink: '#C15F3C',
    colorLinkHover: '#A94E2F',
    colorTextBase: '#20201D',
    colorBgLayout: '#F4F3EE',
    colorBgContainer: '#FBFAF7',
    colorBgElevated: '#FFFFFF',
    colorBorder: '#D8D3C7',
    colorBorderSecondary: '#E6E3DA',
    borderRadius: 8,
    fontFamily: '"Hanken Grotesk", -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    fontSize: 14,
  },
  components: {
    Layout: {
      bodyBg: '#F4F3EE',
      siderBg: '#EFEDE4',
      headerBg: '#FBFAF7',
      triggerBg: '#E4DFD3',
      triggerColor: '#6B665C',
    },
    Menu: {
      itemBg: 'transparent',
      itemColor: '#6B665C',
      itemHoverColor: '#20201D',
      itemHoverBg: 'rgba(0,0,0,0.035)',
      itemSelectedColor: '#C15F3C',
      itemSelectedBg: 'rgba(193,95,60,0.10)',
      itemActiveBg: 'rgba(193,95,60,0.10)',
    },
    Button: {
      primaryShadow: '0 1px 2px rgba(193,95,60,0.30)',
    },
    Segmented: {
      itemSelectedBg: '#FFFFFF',
      itemSelectedColor: '#20201D',
      trackBg: 'rgba(0,0,0,0.045)',
    },
    Tabs: {
      itemSelectedColor: '#C15F3C',
      itemHoverColor: '#A94E2F',
      inkBarColor: '#C15F3C',
    },
    Input: {
      activeBorderColor: '#C15F3C',
      hoverBorderColor: '#D08C6E',
      activeShadow: '0 0 0 2px rgba(193,95,60,0.10)',
    },
    Table: {
      headerBg: 'rgba(0,0,0,0.025)',
      borderColor: '#E6E3DA',
    },
  },
}
</script>
