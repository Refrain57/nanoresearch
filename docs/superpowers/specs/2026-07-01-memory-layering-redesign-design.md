# 记忆系统分层重构 — 设计文档

- 日期：2026-07-01
- 状态：设计已定，待落 plan
- 背景来源：`待修清单` M1（记忆 scope/路由混乱）+ 本轮代码勘察 + OpenClaw/Hermes 对比

---

## 1. 问题（已在代码中核实）

当前记忆是 **3 个文件 scope + 1 个混叠向量层**，没有清晰分层：

| scope | 路径 | 谁写 | 何时注入/召回 | 现状 |
|---|---|---|---|---|
| ① 孤儿 | `base/memory/MEMORY.md` | 迁移前旧 consolidation | 从不 | 全库无 reader，死数据 |
| ② 用户级(no-agent) | `base/users/{uid}/memory/MEMORY.md` | consolidation(agent_id=None) ＋ agent Write 工具 | 仅未绑 agent 时注入 | 半用 |
| ③ Agent级 | `base/users/{uid}/agents/{agent_id}/memory/MEMORY.md` | consolidation(agent_id=X) | 仅绑 X 时注入 | 半用 |
| L3 向量 | Chroma `user_memory{suffix}` | extractor(偏好/习惯/决策) ＋ consolidation summary ＋ raw_archive | 每轮 `<history>` 语义召回 | 混叠 |

三个硬伤：

1. **②/③ 二选一，永不拼接**（`context.py:337` 恒传 `agent_id`）→ 用户画像被按 scope 切碎、互相矛盾。
2. **写入口/注入口错位**：绑定时 consolidation 写 ③，但 agent 手动 Write 写 ②（`context.py:485`）→ 绑定态下 agent 自存的记忆不被注入。
3. **L3 混叠 + 抽取器有 bug**：`user_memory` 把画像/事件/摘要塞一个 collection、一套 metadata（无 conversation_id/topic）；`ConversationKnowledgeExtractor` 把所有抽取项硬编码 `is_evergreen=True`（`extractor.py:134`），而 `cleanup_old_user_memory` 只删非 evergreen → dev 人格永不过期。

**关键事实（澄清对比误判）**：原文并未丢失。PG `Message` 表（`conversation_id` + `seq` 有序）是每对话的全量真相源；`_db_save → replace_messages(conv.id, session.messages)` 写全量，Redis 只是压缩后的工作集。所以我们的病根不是"抽完丢原文"，而是**派生层混叠、不可从原文重建、画像被整篇覆写污染**。

---

## 2. 目标 / 非目标

### 目标
- 把记忆显式分成 **真相源层（只存不注入）** 和 **派生层（可从 PG 重建、可注入）**。
- 落地四层栈（见 §3）。
- 画像更新从"整篇覆写文件"升级为"对结构化 fact store 的增量 diff 自动应用"，带 provenance。
- 杀掉混叠的 `user_memory` 与 buggy extractor，换成职责单一的 events / conv-summary 两个 collection。

### 非目标（本轮明确不做，推后续）
- 人审队列、diff 风险分级（追加 vs 删改）、前端审批弹窗 / SSE 推送 diff。
- 画像的前端**人工编辑 UI**（schema 预留 `source=manual`/`edited_by`/`edited_at`，但本轮不建 UI，故本轮不产生 manual 记录）。
- 常驻全局"整体历史摘要"层（Model B：整体历史 = 对 events 的语义召回视图，不单列）。

---

## 3. 目标架构：四层栈

自底向上。只有 **PG 原文** 和 **画像 fact store** 是真相源；events / conv-summary / MEMORY.md 均为可重建派生物。

```
┌ TOP（User 全局，常驻注入）────────────────────────────────┐
│  ④ 画像：memory_facts store（真相源）→ MEMORY.md（单向投影）  │
├ MIDDLE（会话级）──────────────────────────────────────────┤
│  ② 会话摘要 mem_conv_summaries：keyed by conv_id+turn+time   │
│     近端确定性接回 ＋ 远端本对话早期语义召回（滑动窗口，有界）  │
├ BOTTOM（User 全局，原子，可检索）─────────────────────────┤
│  ① 历史事件 mem_events：{time, topic, action, result, conv_id} │
│     语义召回（取代旧 <history> 的 flat user_memory）           │
├ TRUTH ────────────────────────────────────────────────────┤
│  PG Message（全量对话日志，一切派生层的重建源）              │
└───────────────────────────────────────────────────────────┘
```

> Model B 决议：不做独立"整体历史摘要"层。"以前聊过啥"= 对 ① events 的语义召回。

---

## 4. 各层详细设计

### 4.1 ① 历史事件层（events）

- **存储**：Chroma collection `mem_events{suffix}`（沿用 `collection_suffix` 机制；按 uid 过滤，参考 per-uid 隔离约定）。
- **记录 schema**（metadata）：
  - `id: str`，`uid: str`，`conversation_id: str`
  - `time: iso8601`（事件发生时间，取 chunk 内消息时间）
  - `topic: str`（跟什么有关）
  - `action: str`（做了什么）
  - `result: str`（得到什么结果）
  - `type: "event"`
  - 向量化文本 = `f"{topic} | {action} | {result}"`
- **来源**：consolidation 一次 LLM pass 产出（§5），可从 PG 重建。
- **取代**：删除 `ConversationKnowledgeExtractor` 及其 `is_evergreen` 写入路径。events 不带 evergreen 概念，靠 decay + 语义相关性排序，不做永久沉淀。
- **召回**：`build_history_context` 改为查 `mem_events`，沿用现有 hybrid（BM25+vector+RRF+rerank+decay），uid 过滤，top_k=5。渲染进 `<history>`。

### 4.2 ② 会话摘要层（conversation summary）

- **存储**：Chroma collection `mem_conv_summaries{suffix}`。
- **记录 schema**：
  - `id: str`，`uid: str`，`conversation_id: str`
  - `turn_start: int`，`turn_end: int`（覆盖的 PG seq 范围）
  - `created_at: iso8601`，`topic: str`
  - `text: str`（该段摘要），`type: "conv_summary"`
- **来源**：与 events 同一次 consolidation pass 产出；切点搭现成的 `pick_consolidation_boundary`（已按 user-turn 边界切），一段 chunk 一条摘要。
- **两种消费（滑动窗口分两段，均限定本 `conversation_id`，只对已成摘要的历史段生效）—— 本轮均必做**：
  1. **近端固定窗口 → 确定性接回（保近期连续性、大小有界）**：取本 `conversation_id` 中 `turn_end <= last_consolidated` 的摘要里、`turn_end` **最接近压缩边界的最近若干段**，按 token 预算截断（默认占 conv_summary 预算约 **60%**），按 turn 升序注入 `<conversation_summary>`。这段必须确定性接回：用户下一句常指代刚压缩掉的最近段，语义上不一定相关，纯召回会漏。
  2. **远端早期摘要 → 语义召回（相关才进，不占死预算）**：近端窗口**之外**的本对话早期摘要进语义检索（**仍用 `conversation_id` 过滤**——是"本对话早期"召回，非 §8 的跨会话召回），默认占 conv_summary 预算约 **40%**。
- **不变量**：压缩边界**之后**未压缩的原始 turns 永远全量在场，**不参与任何召回**；语义召回只作用于已成摘要的历史段。
- **窗口大小**：60/40 为默认，做成可调（如 `CONV_SUMMARY_RECENT_RATIO`）。近端按 token 预算而非固定段数定大小，随预算自适应。

### 4.3 ④ 画像层（memory_facts store + MEMORY.md 投影）

**真相源从 `.md` 文件迁到结构化 store。** MEMORY.md 降为单向渲染投影。

- **存储**：新增 PG 表 `memory_facts`。
- **表 schema**：
  - `id: uuid pk`
  - `uid: str`（索引）
  - `section: enum(facts | user_profile | focus_areas)`
  - `text: str`
  - `source: enum(extracted | manual)`
  - `derived_from: json`（event id 列表，`source=extracted` 时填；`manual` 时为空）
  - `confidence: float | null`
  - `edited_by: str | null`（`source=manual` 时填）
  - `edited_at: timestamp | null`（`source=manual` 时填）
  - `active: bool default true`（软删除：diff 移除只置 false，保留 provenance 历史）
  - `created_at`，`updated_at`
- **投影**：`render_memory_md(uid)` 把 `active=true` 的记录按 section 分组渲染成现有 markdown 格式，写 `users/{uid}/memory/MEMORY.md`。每次 diff apply 后重渲染。**单向**：文件永不回读为真相。
- **注入**：`context.py` 的 `<memory>` 块改为读投影文件（注入路径基本不变），或直接从 store 渲染。始终注入（Model B：单一 user-global，`agent_id` 不再参与画像选择）。

**画像更新机制（本轮保留的地基）**：

- consolidation 产出 `profile_diff = {add: [...], remove: [...]}`（§5），**增量应用，永不整篇覆写**。
- `add` 项 → 插入新 `memory_facts` 记录：`source=extracted`，`derived_from=[本批 events 的 id]`，带 `confidence`，`section` 由 LLM 给。插入前做近重复去重（归一化 text 比对，跳过已存在的 active 记录）。
- `remove` 项 → 按**归一化 text 匹配** active 记录，置 `active=false`——**但仅当该记录 `source=extracted`**（匹配到 manual 记录则跳过，见下）。
- **manual 保护（数据层不变量，本轮保留）**：自动 diff 流程**不得删改 `source=manual` 的记录**。本轮虽无 UI 产生 manual 记录，该不变量仍在 diff-apply 逻辑中实现，为后续人工编辑铺路。
- 不建模 "modify"，改用 add+remove 表达。

---

## 5. Consolidation 重构

替换 `MemoryStore.consolidate()` 现在"produce `history_entry`(→L3) + `memory_update`(→MEMORY.md 整篇覆写)"的逻辑。

- **触发不变**：token 压缩 `maybe_consolidate_by_tokens` ＋ 启动 idle `plan_startup_consolidation`（保证短对话也留痕）。
- **一次 LLM pass**，`save_memory` 工具新 schema 返回：
  - `events: [{time, topic, action, result}]`
  - `summary: {text, topic}`（本 chunk 的会话摘要）
  - `profile_diff: {add: [{section, text, confidence}], remove: [text]}`
- **落库顺序**：先写 events（拿到 id）→ 写 conv_summary → 应用 profile_diff（`add` 的 `derived_from` 指向刚写的 events id）。
- **失败降级**：保留现有"连续失败 N 次 raw-archive"思路，但 raw-archive 落到 `mem_events`（或单独 raw 记录），不再进旧 `user_memory`。

---

## 6. 检索 / 注入

`_build_dynamic_suffix` 注入顺序与来源：

1. `<memory>` ← 画像投影（`memory_facts` active 记录），**始终**。
2. `<conversation_summary>` ← 本 `conversation_id` 已压缩段的 conv_summaries，**滑动窗口两段**（§4.2）：近端固定窗口确定性接回（约 60% conv_summary 预算）＋ 远端早期摘要语义召回（`conversation_id` 过滤，约 40%）。**大小有界，不随对话变长单调增长。**
3. `<history>` ← 对 `mem_events` 的语义召回（hybrid + decay，uid 过滤，topic=当前用户输入）。

预算沿用现有 `memory_budget_ratio` 切分；conv_summary 与 history 共享 knowledge_budget，conv_summary 内部再按 60/40 分近端/远端。

---

## 7. 一次性迁移（"全弃，从 PG 重生"）

- **弃**：清空/删除旧 `user_memory` collection；删除/忽略 ①②③ 三处 `MEMORY.md`（含富孤儿①的人工内容，已确认接受丢失）。
- **重建脚本**（`scripts/rebuild_memory_from_pg.py`）：对每个 uid：
  1. 读该 uid 全部 PG 对话（按 conversation_id 分组、seq 有序）。
  2. 每个对话按边界切 chunk，跑 §5 的 consolidation pass → 填 `mem_events` + `mem_conv_summaries` + `memory_facts`（全部 `source=extracted`）。
  3. `render_memory_md(uid)` 生成 MEMORY.md 投影。
- **幂等**：脚本可重跑；重跑前清空该 uid 的三个派生存储再重建（内容去重靠重建而非增量）。
- 本轮迁移全自动，无人审。

---

## 8. 可选增强（列出，非本轮）

- **跨会话**的会话摘要语义召回（跨 `conversation_id`）。注意：§4.2 的"本对话早期摘要语义召回"（限定 `conversation_id`）已提为本轮必做，此处仅指跨会话那半。
- events / summary 抽取质量校准（confidence 归一、assistant 消息剔除已在新 prompt 内处理）。

---

## 9. 明确推后（human-in-the-loop 全家桶）

- 人审队列；diff 风险分级（纯追加自动 / 删改需审）；前端审批弹窗；SSE/WS 推送 diff 到前端。
- 画像前端人工编辑 UI（schema 已预留 `source=manual`/`edited_by`/`edited_at` + manual 保护不变量）。

---

## 10. 分阶段（供 writing-plans 拆步）

- **P1 画像 store**：`memory_facts` 表 + diff-apply 引擎（add/remove/去重/manual 保护）+ `render_memory_md` 投影 + `context.py` 注入改读投影。
- **P2 events 层**：`mem_events` collection + consolidation 产 events + 删除旧 extractor + `build_history_context` 改查 events。
- **P3 会话摘要层**：`mem_conv_summaries` collection + consolidation 产 summary + `<conversation_summary>` 确定性接回。
- **P4 迁移**：`rebuild_memory_from_pg.py` + 清弃旧 `user_memory`/旧 MEMORY.md。

（P1–P3 之间弱依赖，可并行；P4 最后。consolidation 的 `save_memory` 新 schema 在 P2 引入、P3 扩展。）

---

## 11. 测试要点

- **diff-apply**：纯追加插入；remove 置 active=false；remove 命中 `source=manual` 记录时**跳过**（不变量）；近重复去重不重复插入。
- **投影**：`render_memory_md` 按 section 分组、只渲染 active、格式与旧 MEMORY.md 兼容。
- **consolidation pass**：`save_memory` 新 schema 解析（events/summary/profile_diff）；events 先写、profile_diff.add 的 `derived_from` 正确指向本批 events。
- **注入**：顺序 memory → conv_summary → history；conv_summary 只取本 conversation_id 已压缩段。
- **conv_summary 滑动窗口**：近端窗口按 token 预算截断、取 `turn_end` 最接近边界的段、按 turn 升序；远端早期段走语义召回且被 `conversation_id` 过滤；压缩边界之后未压缩的 raw turns 不进任何召回（不变量）。
- **迁移**：`rebuild_memory_from_pg` 幂等（重跑结果一致）；旧存储被清空。
- **回归**：确认无任何代码再读 `user_memory` / 旧 `agents/{id}/memory`。
