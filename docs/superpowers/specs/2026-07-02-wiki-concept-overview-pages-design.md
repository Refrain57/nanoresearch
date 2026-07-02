# Wiki 概念页 + 总览页（Wiki Phase 3）— Design

**Date:** 2026-07-02
**Status:** Approved (design) — 用户已 waive spec review,直接进 writing-plans。
**Depends on:** Phase 1(KG + 知识图谱/Wiki tab)+ **Phase 2**(`article_generator.py` / `CitationText.vue` / `kg_entity_articles` 表 / 按需+缓存+stale 机制)。**本分支基于 `feat/wiki-llm-entity-pages`(Phase 2)**。

## 1. 目标 / 非目标
**目标**:在知识图谱/Wiki tab 增加两类"宽范围合成页",与实体页并列,统一按需生成+缓存+`[^n]` 渲染:
- **概念页**:用户输入一个主题 → 对该主题做 RAG 检索 → 综合成 grounded 词条(引用检索到的 chunk)。
- **总览页**:全库一张地图 → 依据 KG 结构(top 实体 + 关系)综合成导览。

**非目标(仍不做)**:faithfulness 打分/校验、对比页、`[[wiki-links]]`、cron/预生成、按子领域分总览、概念自动派生(概念一律用户手动建)。

## 2. 已定决策
| 决策 | 结论 |
|---|---|
| 范围 | 概念 + 总览 同一 spec |
| UI | 知识图谱 tab 左栏分三区:**总览 / 概念 / 实体**;右侧统一 `CitationText` 渲染 |
| 概念来源 | **用户手动建**(输入主题);清单=已建概念 |
| 概念 grounding | RAG 检索该主题 → 检索到的 chunk 当证据 + `[^n]` |
| 总览范围 | **全库一张**;骨架=KG top 实体 + 其关系 |
| 生成时机 | 按需 + 缓存 + stale(同 Phase 2) |
| 存储 | **复用 `kg_entity_articles`,key 命名空间**(零迁移) |

## 3. 设计

### 3.1 统一 article 内核（复用 Phase 2）
三种页都是同一 "article":`{markdown(含 [^n]), citations[{index,source,page,snippet}], evidence_hash, model, generated_at, stale}`,存 `kg_entity_articles`,前端用 `CitationText` 渲染。**本期只加"概念/总览的证据组装 + prompt + 入口",内核全复用。**

### 3.2 存储（复用 kg_entity_articles,key 命名空间）
项目无 alembic,加列不会自动迁移 → **不改表**,用 `entity_name` 列作通用 key,命名空间隔离:
- 实体页:`entity_name = <归一化实体名>`(Phase 2,不变)
- 概念页:`entity_name = "concept::" + _normalize(topic)`
- 总览页:`entity_name = "__overview__"`
`UNIQUE(kb_id, entity_name)` 天然隔离,零 schema 变更。`citations`/`evidence_hash` 复用(概念=检索 chunk 签名;总览=KG 骨架签名)。

### 3.3 概念页生成（grounding = RAG）
1. 用户输入 `topic`。
2. **对 topic 做 RAG 检索**(复用 knowledge_router 现有 query-test 端点所用的检索路径:fusion top-k、dense+sparse)→ 拿回 top-k chunk。
3. 编号 → `article_generator` 新增 `build_concept_prompt(topic, evidence)`(依据检索结果写该主题词条、标 `[^n]`、不确定不写)→ LLM → markdown + `build_citations(evidence)`。
4. 存 `kg_entity_articles`(key=`concept::<norm>`,evidence_hash=检索 chunk 签名)。
> **计划期锚点**:server 端 RAG 检索怎么调 —— 复用 `knowledge_router` 的 query-test 端点(`/query/test`)背后的检索(它已构造 fusion/dense/sparse + session_factory + kb_id);plan 落地精确调用。

### 3.4 总览页生成（grounding = KG 结构）
1. 取 KG 骨架:`get_stats(kb_id).top_entities`(top 实体+提及数)+ 这些 top 实体的关系(对每个取 `get_entity_facts` 或取 top triples)。
2. `article_generator` 新增 `build_overview_prompt(top_entities, facts)`(依据"库里有哪些主要实体、怎么连"写导览,不放飞)→ LLM → markdown。
3. citations:总览引用的是"实体/关系"层,证据可弱(MVP:骨架里带的代表信息;不强制逐句 `[^n]`,有就标)。存 key=`__overview__`,evidence_hash=骨架签名(top 实体名+关系的哈希)。

### 3.5 生成服务扩展（`rag/wiki/article_generator.py`）
把 Phase 2 的 `generate_article` 的"LLM 调用核心"抽出(`_complete(llm_settings, system, user) -> str`),新增:
- `build_concept_prompt(topic, evidence)`、`build_overview_prompt(top_entities, facts)`
- `generate_concept_article(llm_settings, topic, evidence) -> (markdown, citations)`
- `generate_overview_article(llm_settings, top_entities, facts) -> (markdown, citations)`
- `concept_signature(evidence)`(复用 evidence_signature)、`overview_signature(top_entities, facts)`

### 3.6 API（`knowledge_router.py`,Graph 区,复用 `_article_dict`/`_resolve_rag_settings`）
- `GET /api/knowledge/{kb}/graph/concept/article?topic=...` → 缓存概念页(+stale)或 null
- `POST /api/knowledge/{kb}/graph/concept/article?topic=...` → 检索+生成+存
- `GET /api/knowledge/{kb}/graph/overview/article` → 缓存总览(+stale)或 null
- `POST /api/knowledge/{kb}/graph/overview/article` → 建骨架+生成+存
- 列已建概念:`GET /api/knowledge/{kb}/graph/concepts` → 从 `kg_entity_articles` 取 key 前缀 `concept::` 的列表。

### 3.7 前端（`KnowledgeDetailView.vue` 知识图谱 tab 左栏 + 复用 CitationText）
- 左栏三区:**总览**(单入口)/ **概念**(「+ 新建概念」输入 + 已建列表)/ **实体**(现有,不变)。
- 选任一 → 右侧渲染对应 article(把 Phase 2 实体页顶部的 article 区抽成可复用块 `ArticleView`,或直接复用 CitationText + 生成/重生成按钮),统一 `[^n]` popover。
- apis:`getConceptArticle/generateConceptArticle(kbId, topic)`、`getOverviewArticle/generateOverviewArticle(kbId)`、`listConcepts(kbId)`。

## 4. 数据流
```
概念:输入 topic → POST concept/article → RAG 检索 → article_generator(concept) → 存(concept::) → CitationText 渲染
总览:点总览 → GET overview/article(无则 POST) → get_stats 骨架 → article_generator(overview) → 存(__overview__)
左栏概念列表:GET graph/concepts(key 前缀 concept::)
```

## 5. 测试
- **后端单测**:`build_concept_prompt`/`build_overview_prompt`(含编号证据/骨架 + 依据指令,纯函数);`concept_signature`/`overview_signature` 稳定性;`kg_entity_articles` 用 `concept::`/`__overview__` key 存取不与实体页碰撞(真 PG);`listConcepts` 只返 `concept::` 前缀。
- **前端**:`npm run build`。
- **e2e(人工,需 LLM+服务+KG)**:输入概念→检索生成→`[^n]`;总览生成→导览;左栏三区切换;stale/重生成。

## 6. Commit 切分（初估）
| C | 内容 |
|---|---|
| C1 | article_generator 扩展:抽 `_complete` + concept/overview prompt+signature+generate + 纯函数单测 |
| C2 | 概念检索接线(server 端 RAG)+ 概念 API(GET/POST)+ listConcepts |
| C3 | 总览骨架组装 + 总览 API(GET/POST) |
| C4 | repo:按 key 前缀列概念(`list_articles_by_prefix`)+ 测试 |
| C5 | 前端:抽 ArticleView 复用块 + apis |
| C6 | 前端:左栏三区(总览/概念/实体)+ 接线 |

顺序:C1→(C2,C3,C4 后端)→(C5→C6 前端)。

## 7. Spec 自检
- ✅ 无 TBD;新方法/端点/key 命名空间/字段具名;计划期锚点(server RAG 检索调法)已标明待 plan 落地,非空洞。
- ✅ 一致性:article 形状与 Phase 2 一致(复用 CitationText/kg_entity_articles/_article_dict);存储零迁移(key 命名空间,不加列)。
- ✅ 范围:概念+总览两页型;打分/对比/[[links]]/cron/子领域总览/概念自动派生 明确划出。
- ✅ 依赖:基于 Phase 2 分支;server RAG 检索复用 query-test 路径。
- ✅ 歧义:概念来源(手动)、概念 grounding(RAG)、总览范围(全库一张)、存储(命名空间 key)、UI(左栏三区)均定死。
