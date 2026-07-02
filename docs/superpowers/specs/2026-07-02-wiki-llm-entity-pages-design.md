# Wiki LLM 词条（Phase 2 MVP）— Design

**Date:** 2026-07-02
**Status:** Approved (design) — pending spec review → writing-plans
**Depends on:** Wiki Phase 1（已 merge，grounded 实体浏览器：实体列表 + 详情[事实/佐证/证据/邻居]）；引用功能的 `[^n]` 内联渲染（chat/inline-citations，需抽成共享件复用）；项目的 LLM provider 层（ModelFactory）。

## 1. 目标 / 非目标

**目标**：给 Phase-1 实体详情页**顶部**加一段**按需生成、缓存持久化**的 LLM 词条正文，正文带 `[^n]` 内联引用到该实体的证据；只依据该实体的 grounded 证据生成，用户可对照正文下方的 Phase-1 确定事实面板。

**非目标（明确不做，留后）**：
- **faithfulness 打分 / 事后校验**（是既定方向，MVP 不做；见 §7）。
- 概念页 / 对比页 / 总览页 / synthesis（Karpathy 的其它页型）。
- `[[wiki-links]]`（Phase-1 邻居已提供导航）。
- 预生成 / 建图 hook 生成 / cron / 定时（**cron 违反 hooks-not-cron 原则**；按需即可）。
- 异步 worker 生成（MVP 走同步）。

## 2. 已定决策
| 决策 | 结论 |
|---|---|
| 生成时机 | **按需 + 缓存持久化**：无缓存点「生成词条」才跑；源变 → 标 stale 提示重生成 |
| 执行方式 | **同步 API + 前端转圈**（单实体几秒；不引 worker） |
| grounding | **自由合成**，但输入**只喂该实体证据**；正文带 `[^n]`；下方 Phase-1 事实面板对照 |
| 校验 | **MVP 不做**（faithfulness/打分是未来方向） |
| 引用 | 正文内联 `[^n]`，**复用**现有引用渲染（抽共享件） |
| 展示位 | Phase-1 实体详情页顶部 |

## 3. 设计

### 3.1 生成管线（后端，同步）
1. **组装输入**：取该实体的证据 chunk（`graph_repo` 新增 `get_entity_evidence(kb_id, name) -> list[{chunk_id, content, source(filename), page}]`，经 `KgEntityMention → KbChunk → KbDocument` join；filename 复用文件名解析）+ 该实体的事实（现有 `get_entity_facts`）。给证据编号 1..N。
2. **LLM 合成**：prompt 要求"依据下列证据写一段该实体的中文词条正文；引用处标 `[^n]`，n = 证据编号；无证据支撑的不要写；不编造"。经项目 LLM provider（ModelFactory 解析的合成/ingestion 角色）同步调用。
3. **产出**：`markdown`（含 `[^n]`）+ `citations` 数组 `[{index, source, page, snippet}]`（由编号证据回填，形状与聊天 `msg.citations` 一致，供前端复用渲染）。

> **计划期需落地的锚点**：项目里"非 agent 的一次性 LLM completion"怎么调（参考 query-rewrite / eval RAGAS 的 provider 调用）；`get_chunks_by_entity_name` 现返回 chunk_id 列表，故新增 `get_entity_evidence` 取内容+文件名。

### 3.2 存储（新表）
不塞进 `KgEntity`（重建图会 `delete_by_kb` 清掉，且同名多 label）。新表：
```
kg_entity_articles(
  id, kb_id (FK, index), entity_name (str),
  markdown (Text), citations (JSONB),
  evidence_hash (str),        -- 该实体证据的签名（如排序后 chunk_id 集合的 hash）
  model (str), generated_at (timestamptz),
  UNIQUE(kb_id, entity_name)
)
```
- **stale 判定**：读取时比 `evidence_hash` 与当前证据签名；不等 → `stale=true`（前端提示"重新生成"）。
- 新表 = 一次 migration（跟随项目现有建表/迁移方式）。

### 3.3 API（knowledge_router，Graph 区）
- `GET /api/knowledge/{kb_id}/graph/entities/{name}/article` → `{article: {markdown, citations, model, generated_at, stale} | null}`。
- `POST /api/knowledge/{kb_id}/graph/entities/{name}/article` → 同步生成 + upsert + 返回同上 `article`。
- 均经 `_get_kb_or_404` + `get_current_user`；实体不存在 → 404。

### 3.4 前端（KnowledgeDetailView 实体详情顶部）
- 打开实体：`GET …/article`。有 → 渲染 `markdown` + `[^n]`（点开 popover 看来源）；`stale` → 显"重新生成"按钮。无 → 显「生成词条」按钮。
- 点生成/重生成 → `POST …/article`（转圈）→ 渲染。
- **`[^n]` 复用**：把 `MessageList.vue` 的 `linkifyCitations` + citation popover 抽成共享组件/composable（如 `useCitations` 或 `<CitationText>`），聊天与词条共用；抽取后 MessageList 改用共享件（等价重构，不改行为）。

### 3.5 grounding 模型（MVP）
三层非打分的防幻觉：① 输入约束（只喂该实体证据）；② 正文 `[^n]` 可追溯到证据；③ 正文下方即 Phase-1 确定事实/佐证/证据，用户肉眼对照。打分/faithfulness 留 §7。

## 4. 数据流
```
实体详情打开 → GET article
  命中缓存 → 渲染 markdown+[^n]（stale? 显重生成）
  未命中 → 「生成词条」按钮
点生成 → POST article
  组装证据(get_entity_evidence)+事实 → LLM 合成([^n]) → 存表(evidence_hash) → 返回 → 渲染
[^n] 点击 → popover(citation.source/page/snippet)  （复用聊天引用件）
```

## 5. 测试
- **后端单测**：`get_entity_evidence` 返回内容+文件名；article upsert/get；`evidence_hash` 变化 → stale=true。（真 PG，沿用 Phase-1 测试模式。）
- **prompt 冒烟**：生成函数在 mock/stub LLM 下产出 markdown+citations 结构。
- **前端**：`npm run build` + 手动 e2e（打开实体→生成→显示带 `[^n]`→点开 popover→改源后标 stale→重生成）。

## 6. Commit 切分（初估）
| C | 内容 |
|---|---|
| C1 | graph_repo `get_entity_evidence` + 单测 |
| C2 | 新表 `kg_entity_articles` + migration + repo（upsert/get + evidence_hash）+ 单测 |
| C3 | 生成服务（组装+prompt+LLM 调用+回填 citations） |
| C4 | 2 个 API 端点 |
| C5 | 前端：抽 `[^n]` 共享件 + MessageList 改用（等价重构） |
| C6 | 前端：实体页顶部词条区（生成/渲染/重生成） |

顺序：C1→C2→C3→C4→(C5→C6)。

## 7. 未来（不做，仅记）
- **faithfulness 校验/打分**：RAGAS 式 LLM 校验器逐句判证据支撑 → 总分 + 无支撑句标记（"自由合成 + 事后校验"的后半，本 MVP 只做前半）。
- 概念/对比/总览页、`[[wiki-links]]`、建图 hook 预生成（若接受成本，事件驱动非 cron）。

## 8. Spec 自检
- ✅ 无 TBD；新方法/表/端点/字段具名；两个计划期锚点（LLM completion 调法、get_entity_evidence）已标明待计划落地，非空洞。
- ✅ 一致性：citations 形状与聊天 `msg.citations` 对齐以复用渲染；存储不放 KgEntity 的理由（重建图清空）已说明。
- ✅ 范围：单一 MVP（生成+缓存+展示+复用引用）；打分/概念页/[[links]]/预生成/cron/async 明确划出。
- ✅ 反幻觉：MVP 无打分，靠 输入约束 + `[^n]` + Phase-1 面板对照；打分为后续。
- ✅ 歧义：时机(按需)、执行(同步)、引用(内联[^n])、展示(实体页顶)、存储(新表+evidence_hash stale) 均定死。
