# Wiki 知识库 — 路线图 / TODO

> Karpathy 式 "LLM wiki" 在本项目的落地路线。原则:**模型提候选、代码定佐证**;grounding 优先,范围越宽越要用结构/检索把证据框死;触发走**事件 hook,不用 cron**([[feedback_hooks_not_cron]])。

## 现状(已交付)

- ✅ **Phase 1 — grounded KG 浏览器**(已 merge 到 main)
  - KnowledgeDetailView「知识图谱/Wiki」tab:实体列表 + 详情(事实级 triple + 每条"N 篇文档"佐证 + 邻居可跳 + 证据 chunk 真文件名)+ 可折叠邻居图。
  - 后端 `graph_repo`:`list_entities` / `get_entity_summary` / `get_entity_facts`(佐证=distinct document)/ `get_chunks_by_triple`;3 个 `/graph/*` API。
- ✅ **Phase 2 MVP — LLM 实体词条**(已 PR:`feat/wiki-llm-entity-pages`)
  - 实体页顶部按需生成 + 缓存(`kg_entity_articles`,evidence_hash 判 stale)的 LLM 词条 + `[^n]` 内联引用(复用抽出的共享 `CitationText.vue`)。
  - 生成服务 `rag/wiki/article_generator.py`(同步 AsyncOpenAI,镜像 worker)。
  - **不打分**;grounding=只喂该实体证据 + `[^n]` + 下方 Phase-1 事实面板对照。

## 想做的下一步 — 页型扩展

- [ ] **概念页(concept)** —— 抽象主题一页(如"体渲染""显式 vs 隐式表示")。**grounding:对概念做一次 RAG 检索 → 拿回的 chunk 综合成词条 + `[^n]`**(把一次高质量 RAG 回答存成结构化页,靠检索到的 chunk 框证据,不靠单一实体)。待定:概念清单来源(手动指定 / 从 KG 高频实体·主题挑)。
- [ ] **总览页(overview)** —— 全库/子领域的一张地图(如"3D 重建方法总览")。**grounding:吃 KG 结构**(top 实体 + 三元组=骨架,每块配代表证据),让 LLM 讲导览而非放飞。待定:范围(整库一张 / 按子领域分)+ 骨架来源(KG 结构 / 检索聚类)。
- [ ] **(可选,已降级)对比页(comparison)** —— 两个及以上实体在共同维度对比(如"3DGS vs NeRF")。**grounding 最容易**:直接吃 KG 现成的跨实体三元组(如 `faster_than`)+ 两实体证据。用户当前更想要概念/总览,此项暂缓。

## 剩余的"可信 + 自维护"层(关键,决定 wiki 能否成完全体)

- [ ] **faithfulness 校验 / 打分 —— 最重要**。Phase 2 "自由合成 + 事后校验"里被特意跳过的后半:RAGAS 式 LLM 校验器逐句判"是否被证据支撑" → 总分 + 无支撑句标记。**不做这个,所有 LLM 词条就一直是"好看但没保证",与 grounded/反幻觉红线冲突。** 做时 report-only + 人审([[project_eval_observability_pivot]] 取向)。
- [ ] **lint**(Karpathy 第三操作)—— 查断 `[[链接]]`、孤儿页、矛盾;**report-only + 人审,绝不自动删库**。结构检查纯代码便宜;语义(矛盾)用 LLM、opt-in。现有 `knowledge_lint.py` 是废弃 claims/insights 的死代码,应删非复用。
- [ ] **`[[wiki-links]]`** —— 词条之间互链(现在导航只靠 Phase-1 邻居)。
- [ ] **新鲜度自动化** —— 现在:按需 + stale 标 + 手动重生成。可加:入库/建图**事件 hook** 自动刷(非 cron)。
- [ ] **KG 自动建图** —— 前置依赖,现在手动点「重建知识图谱」(默认 off)。自动化后 wiki 才有数据不用手点。

## 判断

页型层面(实体 ✅ / 概念 / 总览)做完 → "能用、像那么回事"了。但要到 Karpathy 那种"可信、自愈、越用越厚"的完全体,**faithfulness 校验是必补的关键闭环**,lint / 互链 / 新鲜度 / 自动建图是加分项。
