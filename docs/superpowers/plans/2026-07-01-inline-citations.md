# Inline Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 LLM 回答正文里的 `[^n]` 引用标记可点,点击弹 popover 显示该来源(文件名+页码+片段);删掉底部「引用来源」面板。

**Architecture:** 提示模型用检索结果 `citations[].index` 内联标 `[^n]` → 前端后处理 marked 出的 HTML,把"有效索引 + 非代码区"的 `[^n]` 换成可点 `<sup class="cite-ref" data-cite=n>` → `.md-body` 上点击委托 → 从该消息 `msg.citations` 取 citation 弹 popover。复用已有的引用捕获(后端不改)。

**Tech Stack:** Vue3 + Pinia;`marked` v18(核心版,无 footnote 扩展 → `[^n]` 保留为字面量)。

**Spec:** `docs/superpowers/specs/2026-07-01-inline-citations-design.md`

## Global Constraints

- 引用标记 = **`[^n]`**(脚注式),n = kb_search 结果 `citations[].index`。
- 防误认 3 层:`[^n]` 格式 + 仅**有效索引**(n ∈ 本条 `msg.citations` 的 index 集) + **跳过代码区**(`<pre>`/`<code>`)。
- 点击 = 就地 **popover**;内容 = `source`(文件名) + `page` + `snippet`(citation 现成字段,零新接口)。
- **删除**底部 `citations-panel`,但**保留** `msg.citations` 数据链路(renderMd + popover 都靠它)。
- 仅前端 + 一处 prompt;不碰后端引用捕获/SSE/持久化;不做全 chunk / B 方案。
- 无 JS 单测框架 → 前端任务以 `cd web && npm run build` 通过 + 明确的手动 e2e 用例验收(与已合入的 citations 前端任务同惯例)。

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `backend/nanoresearch/skills/rag/SKILL.md` | RAG 工具/回答指引 | 加"内联引用标注 `[^n]`"一节 |
| `web/src/components/MessageList.vue` | 消息渲染 | `renderMd(text,citations)` 后处理 `[^n]`;删面板;加 popover + 点击委托 + 样式 |

---

## Task 1: prompt —— 让模型内联标 `[^n]`

**Files:**
- Modify: `backend/nanoresearch/skills/rag/SKILL.md`(在 `## 工作原理` 之后插入新节)

**Interfaces:**
- Produces: 模型在使用 kb_search 内容回答时,于相应句尾输出 `[^n]`,n = 该来源在结果 `citations` 数组里的 `index`。前端(Task 2)据此识别。

- [ ] **Step 1: 在 `SKILL.md` 的 `## 工作原理` 小节之后,插入以下整节**

在文件中找到:
```
## 工作原理

`kb_search` 内部实现：
1. **Dense 检索**：向量相似度搜索
2. **Sparse 检索**：BM25 关键词匹配
3. **RRF 融合**：合并两种检索结果，按相关性排序
4. **可选重排序**：使用 reranker 进一步优化
```
在这一节**之后**插入:
```markdown

## 内联引用标注（重要）

当你**用 kb_search 检索到的内容**回答时，在引用到某来源的那句话末尾内联标注 `[^n]`：
- `n` 取该来源在 kb_search 返回结果 `citations` 数组里的 `index`（结果 JSON 中每条 citation 都带 `index`）。
- **只有确实引用了检索内容才标**；你自己的推断、常识、web 来源不标。
- 一句可标多个来源：`……因此性能更优[^1][^3]。`
- 必须用**确切格式 `[^n]`**（方括号 + 脱字符 + 数字），不要用 `[n]`、`(n)`、`【n】` 等其它写法。
- `n` 必须是 `citations` 里真实存在的 index；拿不准就不标。
```

- [ ] **Step 2: 冒烟校验(字符串在位)**

Run: `cd backend && python -c "import pathlib; t=pathlib.Path('nanoresearch/skills/rag/SKILL.md').read_text(encoding='utf-8'); assert '内联引用标注' in t and '[^n]' in t; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/nanoresearch/skills/rag/SKILL.md
git commit -m "feat(rag): instruct model to emit inline [^n] citations using citation index"
```

---

## Task 2: 前端渲染 —— `[^n]` 变可点 + 删底部面板

**Files:**
- Modify: `web/src/components/MessageList.vue`(`renderMd` `:75-78`;模板 `:14` 与 `:57`;删面板 `:36-49` + CSS `:130-137`;新增 `.cite-ref` 样式)

**Interfaces:**
- Consumes: `msg.citations`(数组,元素含 `index:int, source:str, page:int|null, snippet:str`,已由现有链路填充)。
- Produces: `.md-body` 内出现 `<sup class="cite-ref" data-cite="n">[n]</sup>` 可点元素(仅有效 n、非代码区);Task 3 的点击委托消费之。

- [ ] **Step 1: 在 `<script setup>` 里新增纯函数 `linkifyCitations` 并改写 `renderMd`**

把现有:
```js
function renderMd(text) {
  if (!text) return ''
  return marked.parse(text)
}
```
替换为:
```js
// Turn [^n] markers into clickable <sup> — only for n present in validIndices,
// and only outside <pre>/<code> regions (avoids matching arr[1] etc. in code).
function linkifyCitations(html, validIndices) {
  if (!validIndices || validIndices.size === 0) return html
  // Split so odd-index segments are the <pre>…</pre> / <code>…</code> blocks (left untouched).
  const parts = html.split(/(<pre[\s\S]*?<\/pre>|<code[\s\S]*?<\/code>)/gi)
  return parts.map((seg, i) => {
    if (i % 2 === 1) return seg
    return seg.replace(/\[\^(\d+)\]/g, (m, d) => {
      const n = Number(d)
      return validIndices.has(n)
        ? `<sup class="cite-ref" data-cite="${n}">[${n}]</sup>`
        : m
    })
  }).join('')
}

function renderMd(text, citations) {
  if (!text) return ''
  const html = marked.parse(text)
  const valid = new Set((citations || []).map(c => c.index))
  return linkifyCitations(html, valid)
}
```

- [ ] **Step 2: 模板传入 citations**

模板第 14 行(assistant 正文)由:
```html
<div v-else class="md-body" v-html="renderMd(msgText(msg))" />
```
改为:
```html
<div v-else class="md-body" v-html="renderMd(msgText(msg), msg.citations)" />
```
流式气泡第 57 行由:
```html
<div v-if="streamingText" class="md-body" v-html="renderMd(streamingText)" />
```
改为(流式期间无 citations,`[^n]` 暂显字面量,消息定稿后由上面的分支重渲染成可点):
```html
<div v-if="streamingText" class="md-body" v-html="renderMd(streamingText, [])" />
```

- [ ] **Step 3: 删除底部引用面板**

删掉模板第 36-49 行整段(`<!-- 引用来源折叠面板 -->` 到其 `</div>`):
```html
      <!-- 引用来源折叠面板（仅 assistant 消息） -->
      <div v-if="msg.role === 'assistant' && msg.citations?.length" class="citations-panel">
        ... (整段 a-collapse) ...
      </div>
```
并删掉 CSS 第 130-137 行(`.citations-panel` 到 `.cite-snippet` 的规则)。

- [ ] **Step 4: 新增 `.cite-ref` 样式**

在 `<style scoped>` 里(如 `.md-body` 规则附近)加:
```css
.md-body :deep(.cite-ref) {
  color: var(--nr-clay);
  cursor: pointer;
  font-weight: 700;
  font-size: 0.72em;
  padding: 0 1px;
  user-select: none;
}
.md-body :deep(.cite-ref:hover) { text-decoration: underline; }
```

- [ ] **Step 5: 构建校验**

Run: `cd web && npm run build`
Expected: 构建通过、0 error(chunk-size 警告是既有的)。

- [ ] **Step 6: 手动 e2e(渲染/防误认/面板)**

1. agentic 模式向已建 KB 提问 → 回答里模型标的 `[^1]` 渲染成上标 `[1]`(clay 色、可点样式)。
2. 让回答里含代码块且有 `arr[1]` / `[^1]`(可问"给个数组取值的代码例子") → 代码块内**不变可点**。
3. 若模型标了超范围 `[^9]`(只有 3 条 citation) → 保持字面量 `[^9]`。
4. 底部「引用来源」面板**已消失**。

- [ ] **Step 7: Commit**

```bash
git add web/src/components/MessageList.vue
git commit -m "feat(web): render inline [^n] citations as clickable refs; remove bottom panel"
```

---

## Task 3: 前端 popover —— 点 `[n]` 弹来源卡

**Files:**
- Modify: `web/src/components/MessageList.vue`(模板 `.md-body` 加 `@click`;新增 popover 元素、状态、关闭逻辑、样式)

**Interfaces:**
- Consumes: Task 2 产出的 `.cite-ref[data-cite=n]`;`msg.citations`(找 `index===n` 的条目)。

- [ ] **Step 1: `.md-body`(assistant 正文)加点击委托**

Task 2 改过的第 14 行:
```html
<div v-else class="md-body" v-html="renderMd(msgText(msg), msg.citations)" />
```
改为(带 `@click`,把该消息的 citations 传入):
```html
<div v-else class="md-body" v-html="renderMd(msgText(msg), msg.citations)"
     @click="onCiteClick($event, msg.citations)" />
```

- [ ] **Step 2: 新增 popover 状态 + 处理函数(script setup)**

在 `<script setup>` 里加:
```js
import { onMounted, onBeforeUnmount } from 'vue'   // 合并进已有的 vue import

const activeCite = ref(null)              // 当前展示的 citation 或 null
const citePos = ref({ x: 0, y: 0 })       // popover 左上定位(视口坐标)

function onCiteClick(e, citations) {
  const el = e.target.closest('.cite-ref')
  if (!el) return
  const n = Number(el.dataset.cite)
  const cite = (citations || []).find(c => c.index === n)
  if (!cite) return
  const r = el.getBoundingClientRect()
  citePos.value = { x: r.left, y: r.bottom + 4 }
  activeCite.value = cite
}

function closeCite() { activeCite.value = null }
function onDocClick(e) {
  // 点到别处(非 cite-ref、非 popover 内)就关
  if (e.target.closest('.cite-ref') || e.target.closest('.cite-popover')) return
  closeCite()
}
function onKey(e) { if (e.key === 'Escape') closeCite() }

onMounted(() => {
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
  document.removeEventListener('keydown', onKey)
})
```
> 注:`onCiteClick` 与 `onDocClick` 都会在同一次点击触发;`onDocClick` 里对 `.cite-ref` 直接 return,所以不会把刚打开的 popover 关掉。

- [ ] **Step 3: 新增 popover 模板(放在 `.message-list` 根 div 内的末尾,`</div>` 之前)**

在模板 `<div v-if="toolHint" ...>` 那段之后、根 `</div>` 之前加:
```html
    <div v-if="activeCite" class="cite-popover"
         :style="{ left: citePos.x + 'px', top: citePos.y + 'px' }">
      <div class="cite-pop-head">
        <span class="cite-pop-src">{{ activeCite.source }}</span>
        <span v-if="activeCite.page != null" class="cite-pop-page">p.{{ activeCite.page }}</span>
      </div>
      <div class="cite-pop-snippet">{{ activeCite.snippet }}</div>
    </div>
```

- [ ] **Step 4: 新增 popover 样式(`<style scoped>`)**

```css
.cite-popover {
  position: fixed;
  z-index: 50;
  max-width: 360px;
  background: var(--nr-card);
  border: 1px solid var(--nr-border-strong);
  border-radius: 8px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.12);
  padding: 8px 10px;
}
.cite-pop-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.cite-pop-src { font-size: 12px; font-weight: 600; color: var(--nr-ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cite-pop-page { font-size: 11px; color: var(--nr-ink-2); flex-shrink: 0; }
.cite-pop-snippet { font-size: 12px; color: var(--nr-ink-2); line-height: 1.5; max-height: 160px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
```

- [ ] **Step 5: 构建校验**

Run: `cd web && npm run build`
Expected: 通过、0 error。

- [ ] **Step 6: 手动 e2e(交互)**

1. 点正文里的 `[n]` → 标记下方弹出 popover,显示对应 来源名 / 页码 / 片段,内容与该 n 的 citation 一致。
2. 点 popover 外空白 → 关闭;按 Esc → 关闭;点另一个 `[m]` → 切到 m 的卡。
3. 刷新会话后历史消息的 `[n]` 仍可点弹卡(靠 `content._citations` 重进)。

- [ ] **Step 7: Commit**

```bash
git add web/src/components/MessageList.vue
git commit -m "feat(web): click inline citation ref -> source popover (snippet/page)"
```

---

## Self-Review

**Spec coverage:** §4.1 prompt→Task1;§4.3 渲染(linkify 3 层)→Task2;§4.5 删面板→Task2 Step3;§4.4 交互(popover+关闭)→Task3;§3 防误认(`[^n]`+有效索引+跳代码)→Task2 Step1 的 `linkifyCitations`;§5 降级(无效 n 字面量 / 无 citations 不动)→Task2 Step1 逻辑 + Step6 用例。§2 决策(popover/删面板/`[^n]`/片段)全覆盖。

**Placeholder scan:** 无 TBD/TODO;所有代码步给了完整代码;marked `[^n]` 保留为字面量已在 spec/plan 前言确认(marked v18 无 footnote 扩展)。

**Type consistency:** `renderMd(text, citations)`、`linkifyCitations(html, validIndices:Set)`、`onCiteClick(e, citations)`、citation 字段 `index/source/page/snippet` 全程一致;`.cite-ref[data-cite=n]` 由 Task2 产出、Task3 消费,选择器一致。

**顺序:** Task1 独立;Task2 → Task3(Task3 的点击委托依赖 Task2 产出的 `.cite-ref`)。
