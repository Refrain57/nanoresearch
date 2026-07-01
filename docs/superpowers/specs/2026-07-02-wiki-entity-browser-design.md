# Wiki 实体浏览器（Phase 1，grounded KG browser）— Design

**Date:** 2026-07-02
**Status:** Approved (design) — pending spec review → writing-plans
**Phasing:** 这是 wiki 页的 **Phase 1**。Phase 2（LLM 合成词条正文，Karpathy 式 `[[wiki-links]]`）单独一轮,不在本 spec。

## 1. 目标 / 非目标

**目标**：在 `KnowledgeDetailView` 新增「知识图谱/Wiki」tab —— 一个**基于已有 KG（KgEntity/KgTriple）的、确定性的结构化浏览器**:左侧实体列表(可搜/排序),右侧实体详情页(事实 + 每条事实的**佐证文档数** + 可跳转的邻居实体 + 可展开的证据 chunk),外加一个**可折叠的邻居关系图**。

**非目标（明确不做）**：
- LLM 合成词条正文（Phase 2）。
- `knowledge_lint`（Phase 1 只读浏览器不需要;现有 lint 是废弃 claims/insights 的死代码,应单独删除,非本 spec）。
- 全 KG 力导向大图（只做当前实体 + 1 跳邻居的**小**图）。
- 让建图自动化（沿用现有手动「重建知识图谱」;见 §依赖）。
- 跨 KB 视图（per-KB,和现有 KG 数据一致）。

## 2. 依赖（前置）
KG 需已填充。`KnowledgeDetailView` **已有**「重建知识图谱」按钮 + 图谱统计弹窗(走 `/graph/build`、`/graph/stats`;`enable_graph_expansion` 默认 False)。本 tab **复用**它:KG 为空时显示空态,引导点「重建知识图谱」。本期不改建图流程。

## 3. 已定决策
| 决策 | 结论 |
|---|---|
| 展现形式 | 列表/表格 + 实体详情页(维基式);外加**可折叠**邻居关系图(默认收起) |
| 详情粒度 | **事实级**:`X —关系→ Y` 三元组 |
| 佐证(corroboration) | 每条事实标 **"N 篇文档"** = 该 triple 的 mention 里**不同 `document_id` 的数量**(确定值) |
| 导航 | 邻居实体可点 → 跳该实体的详情页 |
| 证据 | 每条事实/实体可展开对应证据 chunk 片段(来源显示**原文件名**,复用文件名解析) |

## 4. 设计

### 4.1 后端 KG 读层（`storage/repositories/graph_repo.py` 新增查询）
现有:`get_stats`(top-20 实体+计数)、`get_chunks_by_entity_name`、`get_neighbor_chunks_via_entities`、`get_entities_by_doc`。新增:

- **`list_entities(kb_id, search=None, limit=50, offset=0) -> list[{name, label, mention_count}]`**：全量/分页/可搜的实体列表(按 mention_count desc)。（现有 get_stats 只 top-20,不够。）
- **`get_entity_facts(kb_id, name) -> list[{triple_id, source, label, target, doc_count}]`**：该实体作为 source **或** target 的所有 triple;`doc_count` = 该 triple 的 `KgTripleMention` join `KbChunk` 后 **distinct `document_id`** 计数(= 佐证)。**这是核心新查询。**
- **`get_entity_neighbors(kb_id, name) -> list[{name, label}]`**：由 triple 连接的去重邻居实体(可由 get_entity_facts 派生)。
- **`get_chunks_by_triple(triple_id) -> list[KbChunk]`**：某条事实的证据 chunk(经 `KgTripleMention`)。实体级证据复用现有 `get_chunks_by_entity_name`。

文件名:证据 chunk 的来源显示原文件名 —— 复用已实现的 `file_path → KbDocument.filename` 解析(与引用功能同源;可在 API 层按 doc_id/file_path 解析)。

### 4.2 API（`server/routers/knowledge_router.py`,已有 `/graph/*`）
- `GET /api/knowledge/{kb_id}/graph/entities?search=&limit=&offset=` → 实体列表。
- `GET /api/knowledge/{kb_id}/graph/entities/{name}` → 详情:`{name, label, mention_count, facts:[{triple_id, source, label, target, doc_count}], neighbors:[{name,label}]}`。
- `GET /api/knowledge/{kb_id}/graph/triples/{triple_id}/chunks` → 该事实证据 chunk(含 source 原文件名)。

### 4.3 前端（`web/src/views/KnowledgeDetailView.vue` + 可能拆小组件）
- 新增 tab「知识图谱/Wiki」,与「文档 / Chunk浏览 / 测试检索 / RAG评估」同级。
- 两栏:左 = 实体列表(搜索框 + 按提及数排序,点选);右 = 实体详情。
- 详情:
  - 头部:实体名 · label · 被 N 篇提及。
  - **事实**:逐条 `源 —关系→ 目标`,右侧「N 篇文档」佐证徽标;点 ▾ 展开该事实的证据 chunk 片段(带原文件名)。
  - **邻居实体**:chips,点击 → 切换到该实体详情(前端路由/状态,不整页刷新)。
  - **▸ 知识图谱**:默认收起;展开 = 当前实体 + 1 跳邻居的**小**关系图(轻量;SVG 径向或轻量图库,不引重型力导向)。
- 空态:KG 未建 → 提示并链到现有「重建知识图谱」。

### 4.4 数据流
```
KnowledgeDetailView「知识图谱/Wiki」tab
  列表:GET /graph/entities → 实体表
  选实体:GET /graph/entities/{name} → facts(带 doc_count 佐证)+ neighbors
  展开事实:GET /graph/triples/{id}/chunks → 证据(原文件名)
  邻居 chip 点击 → 复用 GET /graph/entities/{neighbor} → 切换详情
  ▾ 图谱:用 detail 的 neighbors 渲染小关系图
```

## 5. 测试
- **后端单测**:`get_entity_facts` 的 `doc_count` = distinct document(构造同一 triple 跨 2 文档 → doc_count=2);`list_entities` 搜索/分页/排序;`get_entity_neighbors` 去重;`get_chunks_by_triple` 返回正确 chunk。
- **API**:三个端点返回结构 + 空 KG 空态。
- **前端**:无单测框架 → `npm run build` + 手动 e2e(建图后:列表出现、点实体看事实+佐证数、展开见证据+真文件名、点邻居跳转、图谱折叠/展开)。

## 6. Commit 切分
| Commit | 内容 | 文件 |
|---|---|---|
| C1 | graph_repo 新增 4 查询 + 单测 | `graph_repo.py` + tests |
| C2 | 3 个 API 端点 | `knowledge_router.py` |
| C3 | 前端 tab + 列表 + 详情(事实/佐证/证据/邻居) | `KnowledgeDetailView.vue`(+ 组件) |
| C4 | 可折叠邻居关系图 | 同上(+ 轻量图组件) |

顺序:C1→C2→(C3→C4)。

## 7. Phase 2 预留（不做,仅记）
实体页顶部加 LLM 合成"词条正文"(Karpathy 式 `[[wiki-links]]`),用本期 grounded 事实/佐证当引用底 + 校验;复活 lint 概念查断链/孤儿/矛盾。

## 8. Spec 自检
- ✅ 无 TBD/占位;新查询/端点/字段均具名。
- ✅ 一致性:佐证=distinct document,贯穿 repo→API→前端;文件名解析复用引用功能。
- ✅ 范围:仅 Phase 1(浏览器);LLM 合成/lint/大图/自动建图 明确划出。
- ✅ 依赖:KG 需已建(复用现有手动建图),空态已处理。
- ✅ 歧义:展现(列表+详情+可折叠小图)、粒度(事实级)、佐证(distinct doc)、导航(邻居跳转)均定死。
