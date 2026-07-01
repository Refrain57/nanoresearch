# 聊天回答内的结构化引用(Chat Citations)— Design

**Date:** 2026-06-30
**Status:** Approved (Option A) — plan at `docs/superpowers/plans/2026-07-01-chat-citations.md`
**Scope:** 在聊天气泡里给 LLM 的回答展示**结构化、可点击的引用来源**(provenance),数据全程持久化。复刻现有 `tool_call` 旁路机制,不发明新通道。

---

## 0. 证据等级

- **【代码确认】** 已读源码到行。
- **【需实现时确认】** 模式清楚、具体写入点待 plan 阶段钉死。

---

## 1. 目标与非目标

**目标**:用户向知识库提问 → agent 用检索到的 chunk 回答 → 聊天气泡里出现一个**引用面板**(仿现有工具调用折叠面板),列出每条回答依据的来源(文档名/路径 + 相关度 + 片段),可展开查看;**刷新/重进会话仍在**。

**非目标(v1 明确不做)**:
- 点击引用**跳转打开源文档全文**(无现成文档查看路由,留作后续)。v1 点击=展开看 `source_path` + 片段。
- 把 `source_path` 美化成漂亮标题(v1 原样显示路径/文件名)。
- LLM 正文内联 `[1]` 标记(那是已否决的方案 A;本设计是结构化面板,与正文解耦)。
- 矛盾检测、跨文档实体聚合(属 wiki/fact 层,另议)。

---

## 2. 现状(全部【代码确认】)

| 环节 | 现状 | 锚点 |
|---|---|---|
| 检索工具返回来源 | `kb_search` 返回 chunks(metadata 带 `source_path`),但 `citations: None` 未生成 | `rag_search.py:245-251` |
| 引用结构化能力 | `CitationGenerator` / `BuildCitationsTool` 已存在(index/source/score/snippet),但**未接入聊天答复链路** | `citation_generator.py`、`agentic/citations.py` |
| 工具结果捕获点 | `after_iteration` 钩子逐个 `zip(tool_calls, tool_results)` 发 `on_tool_call` | `loop.py:392-400` |
| SSE 旁路事件 | 已有 `message_delta / tool_hint / tool_call / message_complete / subagent_result / run_end` | `useRunStream.js:34-39` |
| 后端发事件 | `xadd_event(redis, run_stream_key, {"type": ...})` | `worker.py:245` |
| 消息持久化 | Message **无** tool_calls 列;tool_calls 塞在消息 `content`(OpenAI dict)里随消息存,前端从 `content.tool_calls` 读 | `chat.js:37`、`chat_router.py:147` |
| 观测副本 / run↔msg 链 | `AgentRun.tool_calls` 是另一份观测副本;`AgentRun.output_message_id` 声明但**全仓从未赋值**(无可用 run↔msg 链) | `models.py:130,132` |
| 前端渲染 | assistant 消息走 markdown;已有 `tool-calls-panel` 折叠面板挂在 `msg.toolCalls` | `MessageList.vue:14,23` |

**结论**:零件齐备但未连成线。本设计 = 把"生成→捕获→传输→存储→渲染"五段按现有 `tool_call` 同构补齐。

---

## 3. 数据结构

**Citation item**(单条引用,JSON):
```json
{
  "index": 1,
  "chunk_id": "…",
  "source": "docs/3dgs.pdf",
  "score": 0.87,
  "snippet": "3D Gaussian Splatting renders …",
  "page": 12,
  "doc_id": "…"
}
```
复用 `CitationGenerator.Citation.to_dict()` 的形状(`citation_generator.py:37`),只增 `doc_id`(为将来跳转预留,v1 不点开)。

---

## 4. 五段设计

### 4.1 生成 —— `kb_search` 填充 `citations`
- `rag_search.py:248` 现在写死 `"citations": None`。改为:从 `fused_chunks` 用 `CitationGenerator` 逻辑生成 citation 列表(取 `chunk_id` / `metadata.source_path` / `score` / 截断 snippet)填入。
- 复杂路径(`_execute_complex`,内循环)同样在其返回处填 `citations`。
- 责任划分:**工具生成引用**(它手里就是结构化 chunks),不把解析推给上层。

### 4.2 捕获/转发 —— `loop.py` 新增 `on_citations`
- `after_iteration`(`loop.py:392-400`)已拿到 `(tc, result)`。新增:当 `tc.name` 属于检索工具集(`kb_search` 等)且 `result` JSON 含 `citations` 数组时,提取并经新回调 `on_citations(payload)` 上抛。
- `_run_agent_loop` / `process_direct` 签名增加可选 `on_citations`(与 `on_tool_call` 平行,`loop.py:326,707,846`)。
- **去重 + 稳定序号**:一轮内多次 `kb_search` → 按 `chunk_id` 去重、累积到**本条 assistant 消息**,`index` 在该消息内全局重排(1..N)。

### 4.3 传输 —— 新 SSE 事件 `citations`
- worker 提供 `on_citations` 实现:`xadd_event(redis, run_stream_key, {"type": "citations", "message_id": <seq/id>, "items": [...]})`,与 `tool_call` 发法一致(`worker.py:245` 同构)。
- 与正文 `message_delta` 解耦,独立事件。

### 4.4 存储 —— 嵌入 `Message.content._citations`(Option A,2026-07-01 定)

**为何不加列**:Message 无 tool_calls 列;tool_calls 靠塞在 `content` dict 里随消息存、前端从 `content.tool_calls` 读(`chat.js:37`)。`AgentRun.output_message_id` 是死列,故 run↔msg JOIN 不可用。Option A 复刻 tool_calls 这条已验证的重进机制,**零迁移、零接口改动**。

**做法**:
- assistant 那条消息持久化时,在其 `content` dict 里加 `_citations`(下划线=内部字段),随 `_save_turn` 一同入库。
- `get_messages`(`chat_router.py:147`)本就返回整个 `content` → 前端读 `content._citations`(照搬 `content.tool_calls`)。
- **LLM 安全(关键)**:`content` 会被回放给 LLM。在 `providers/base.py` 的 `_sanitize_empty_content`(`:105`,已在剥离内部 `_meta`,见 `:128-129`)里**扩展为一并剥离顶层 `_citations`**,确保它不进 LLM 请求。这是本方案唯一的"防污染"改动。

### 4.5 渲染 —— 前端引用面板
- `useRunStream.js`:`dispatch` 增 `else if (event.type === 'citations') onCitations?.(event)`(仿 `:36`)。
- chat store:收到 `citations` 事件 → 挂到对应 assistant 消息 `msg.citations`(仿 `msg.toolCalls`)。
- `MessageList.vue`:新增 `<div v-if="msg.role==='assistant' && msg.citations?.length" class="citations-panel">`,折叠列出 `[index] source · score%`,展开看 snippet/page(仿 `tool-calls-panel` `:23`)。
- 历史消息:加载时 `msg.citations` 已带出,直接渲染(持久化生效)。

---

## 5. 数据流(全景)

```
kb_search(填 citations)  ──tool result──►  loop.after_iteration
                                              │ 去重+排序，累积到本条 assistant 消息
              ┌───────────────────────────────┤
              ▼                               ▼
   on_citations → xadd_event              持久化：assistant 消息 content
   {"type":"citations",...}               内嵌 _citations(随 _save_turn 入库)
              │                            (回放给 LLM 前由 _sanitize 剥离)
              ▼
   SSE → useRunStream(onCitations) → chat store(msg.citations) → MessageList 引用面板
                                                                        ▲
   重进会话：get_messages 返回 content → 前端读 content._citations ────────┘
```

---

## 6. 范围决策(已定 2026-06-30,按推荐)

| 决策 | 结论 |
|---|---|
| **覆盖哪条 RAG 路径** | **仅 agentic(走 `loop.py` 钩子)先做**;simple RAG(`worker.py` 简单路径)作为紧跟的第二个接入点,同一存储/事件、仅生成点不同。 |
| **点击引用行为** | **展开看片段 + page**;跳转打开源文档(无现成路由)留后续。 |
| **检索工具集范围** | **`kb_search` + `retrieve_by_entity`**(后者已在输出写 `来源`,顺带结构化)。 |

---

## 7. 测试

- **后端单测**:`kb_search` 由 chunks 生成 citations(simple+complex);`after_iteration` 对检索结果触发 `on_citations`、非检索工具不触发;一轮多次检索按 `chunk_id` 去重 + 序号连续。
- **持久化(Option A)**:assistant 消息 `content._citations` 往返(get_messages 返回 content → 前端读 `content._citations`);`_sanitize_empty_content` 后 LLM 请求不含 `_citations`;history 加载带出引用。无迁移。
- **前端**:`citations` 事件 → 面板渲染;刷新后历史消息仍显示引用(持久化关键用例)。
- **e2e**:向 KB 提问 → 引用面板出现、来源正确、展开见片段;刷新会话 → 引用仍在。

---

## 8. Commit 切分(独立可 revert)

| Commit | 内容 | 文件 |
|---|---|---|
| C1 | `_sanitize_empty_content` 扩展:剥离顶层 `_citations`(LLM 安全前置) | `providers/base.py` |
| C2 | `kb_search` 填充 `citations` | `rag_search.py` |
| C3 | `loop.py` `on_citations` 捕获/去重/转发 + 签名 | `loop.py` |
| C4 | worker 发 `citations` SSE 事件 + 把 `_citations` 嵌入 assistant 消息 content | `worker.py` + `_save_turn` 处 |
| C5 | 前端:`useRunStream` 分支 + store 挂载 + `MessageList` 面板 | `useRunStream.js`、`stores/chat.js`、`MessageList.vue` |

顺序:C1 先行(安全闸)→(C2,C3 并行)→C4→C5。每个可单独 revert。

---

## 9. 风险

- **R1 MCP JSON 往返**:citations 在 MCP 子进程生成、经 tool result JSON 回到 loop。Mitigation:结构简单(纯 dict 列表),已随 chunks 同路返回,无新协议。
- **R2 source_path 不可读**:可能是内部路径。v1 接受原样;美化留后续。
- **R3 simple RAG 路径漏覆盖**:v1 仅 agentic。Mitigation:范围决策已显式记录,第二步补 simple 路径(同存储/事件)。
- **R4 去重序号错位**:多次检索/多轮。Mitigation:按 `chunk_id` 去重、index 在单条 assistant 消息内重排,单测覆盖。
- **R5 `content._citations` 回放污染 LLM**:content 会被回放给模型。Mitigation:C1 先行,在 `_sanitize_empty_content` 剥离顶层 `_citations`;单测断言 sanitize 后无该 key。这是 C1 必须排在最前的原因。

---

## 10. Spec 自检

- ✅ 无 TBD/TODO/占位(唯一【需实现时确认】= `Message.tool_calls` 写入函数的精确位置,属 plan 阶段定位,非设计空洞)。
- ✅ 一致性:五段 × 数据流 × commit 切分对齐;持久化贯穿(列+写入+加载+测试)。
- ✅ 范围:单一功能、单 plan、5 commit 可 revert;wiki/fact 层、矛盾检测、内联 `[n]` 明确划出。
- ✅ 歧义:点击行为/路径覆盖/工具集三处可能的歧义已转成显式范围决策(§6)。
