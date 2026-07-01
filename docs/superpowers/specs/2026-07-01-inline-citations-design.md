# 内联可点引用（Inline Citations, Option A）— Design

**Date:** 2026-07-01
**Status:** Approved (design) — pending spec review → writing-plans
**Depends on:** 已合入的 chat-citations 面板功能（`feat/chat-citations`）。本功能**复用**其后端引用捕获，**取代**其底部面板呈现。

## 1. 目标 / 非目标

**目标**：LLM 回答正文里的引用标记 `[^n]` 可点，点击就地弹 popover 显示该来源（文件名 + 页码 + 片段）。

**非目标（明确不做）**：
- 打开完整 chunk 原文 / 跳 Chunk 浏览（v1 只显 citation 里已有的 snippet）。
- 代码 B 方案（逐句 embedding 归因）——运行时太重，已否。
- 提升模型归因准确性（内联归因天生近似，见 §6）。
- 保留底部「引用来源」面板（本功能**删除**它，只留内联）。

## 2. 关键决策（已拍板）

| 决策 | 结论 |
|---|---|
| 点击行为 | 就地弹 **popover**（不滚动） |
| popover 内容 | 片段预览：`source`(文件名) + `page` + `snippet`（citation 里现成，零新接口） |
| 底部面板 | **删除**（直播 + 重进两处），只留内联 popover |
| 引用标记 | **`[^n]`（脚注式）**，n = kb_search citations 的 `index` |
| 归因方式 | A 方案：模型自标（prompt 引导），非代码归因 |

## 3. 防误认（3 层，核心）

bare `[n]` 会误伤 `arr[1]`、`[2020]`、列表标号等。三层防御：
1. **标记 `[^n]`**：模型只用脚注式标记，代码/正文极少自然出现 `[^`。
2. **有效索引过滤**：仅当 `n` ∈ 本条 assistant 消息的 `citations` 索引集合时才可点。
3. **跳过代码区**：后处理 marked 输出时不碰 `<code>` / `<pre>` 内内容。

三层叠加 → 误认≈0。

## 4. 设计

### 4.1 提示（后端）
`backend/nanoresearch/skills/rag/SKILL.md` 增加一条指令：用检索到的内容回答时，在相应处内联标注 `[^n]`，其中 `n` 取 kb_search 结果 `citations[].index`（模型上下文里已含该数组）。仅当确有来源支撑时标注；不确定就不标。

### 4.2 数据（前端，复用）
`msg.citations`（直播由 SSE `citations` 事件填、重进由 `content._citations` 填，均已实现）。每条 assistant 消息构建 `index → citation` 映射。

### 4.3 渲染（前端）
`renderMd(text, citations)`（`web/src/components/MessageList.vue`，现为 `marked.parse(text)`）改为接收 citations 并后处理：
- 对 marked 输出的 HTML，在**非 `<code>`/`<pre>` 区**匹配 `\[\^(\d+)\]`；
- 若 `n` 在 citation 映射内 → 替换为可点 `<sup class="cite-ref" data-cite="n">[n]</sup>`；
- 否则原样保留（纯文本）。

> **实现前置检查**：确认 `marked`（`gfm:true, breaks:true`）把 `[^n]` 保留为字面量（核心版无脚注扩展应如此）。若项目挂了 footnote 插件导致 `[^n]` 被转成脚注 DOM，则改为在该 DOM 上匹配，或对答案渲染关掉该扩展。

### 4.4 交互（前端）
`.md-body` 容器加 `@click` 委托：命中 `.cite-ref` → 读 `data-cite=n` → 弹 popover（定位在标记附近），显示 `citation.source` / `page` / `snippet`；点击空白或 Esc 关闭。单例 popover（同一时刻只开一个）。

### 4.5 删除底部面板
移除 `MessageList.vue` 的 `citations-panel`（`v-if msg.citations?.length` 那块）及其样式；直播 + 重进都不再渲染面板。**保留** `msg.citations` 数据链路与后端捕获（popover 依赖它）。

## 5. 降级
- 无 citations / 模型没标 `[^n]` → 无标记、无 popover，正常渲染。
- 无效 `[^n]`（n 不在集合）→ 保持字面量。
- 代码里的 `[^1]` / `arr[1]` → 不可点。

## 6. Caveat（记录，不修）
- **归因近似**：模型自标可能标错源/漏标（A 方案固有）。popover 点开的是"模型选的那条 citation"，是真 chunk（确定），但"这句是否真来自它"不保证。
- **多次 kb_search 重编号**：一轮多次检索合并去重会重编 index，模型看到的分段号可能与最终号错位（罕见，v1 接受）。

## 7. 测试
前端无单测框架 → `cd web && npm run build` 通过 + 手动 e2e：
1. agentic 提问 → 回答含 `[^n]` → 渲染成可点 `[n]`；点击 → popover 显示对应 来源/页/片段。
2. 代码块里的 `arr[1]` / `[^1]` → **不可点**。
3. 无效 `[^9]`（只有 3 条 citation）→ 保持字面量。
4. 刷新会话 → 内联可点仍在（`content._citations` 驱动）。
5. 底部面板已消失。
后端：改的是 prompt 文本，可加一个"`rag/SKILL.md` 含 `[^n]` 指令"的字符串冒烟。

## 8. Commit 切分
| Commit | 内容 | 文件 |
|---|---|---|
| C1 | prompt 指令：让模型内联标 `[^n]` | `skills/rag/SKILL.md` |
| C2 | 前端 `renderMd` 后处理 `[^n]`→可点(3 层防误认) + 删底部面板 | `MessageList.vue` |
| C3 | 前端 popover 组件 + 点击委托 + 关闭 | `MessageList.vue`(+ 可能小组件) |

顺序：C1 独立；C2→C3。

## 9. Spec 自检
- ✅ 无 TBD/占位；唯一"实现前置检查"(marked+`[^n]`)是明确的落地验证项，非空洞。
- ✅ 一致性：删面板 vs 复用数据链路 已澄清（删呈现、留数据）。
- ✅ 范围：单一小 feature，3 commit；wiki / 全 chunk / B 方案 明确划出。
- ✅ 歧义：标记(`[^n]`)、点击行为(popover)、面板(删)、内容(片段) 均定死。
