# NanoResearch 全项目工程蓝图 Spec — 撰写实施计划

> **For agentic workers:** 本计划交付物是**文档**（一份大 spec），不是代码。每个 Task 的循环是：
> 读该层真实代码 → 把该章写进 spec → 逐条核对 `文件:行号` 锚点与签名与代码一致（这是文档版的"测试"）。
> 用 superpowers:executing-plans 逐 Task 执行。Steps 用 checkbox 跟踪。

**Goal:** 产出 `docs/superpowers/specs/2026-06-29-nanoresearch-whole-project-blueprint.md`，
一份自顶向下、架构 + 关键代码、18 层全覆盖的 as-built 工程蓝图（中文，2000–4000+ 行）。

**Architecture:** 按蓝图 design（`2026-06-29-whole-project-blueprint-design.md`）的 18 章结构，
单文件追加式撰写。每章先 Read 对应代码再写，锚点格式 `backend/nanoresearch/<path>:<line>`，
关键算法给伪代码而非贴源码。

**Tech Stack:** 被记录的系统用 FastAPI / SQLAlchemy+asyncpg / Redis+arq / Chroma / MCP /
tiktoken / sentence-transformers / anthropic+openai / Vue+Pinia。

## Global Constraints

- 语言中文，与项目现有文档一致。
- as-built：以**当前代码**为准，与过时 `PROJECT_STRUCTURE.md`(nanobot/ 命名) 冲突时纠正之。
- 基线 commit `3b590a8f`，branch `feature/consolidation-compaction`。
- 每章固定骨架：职责一句话 → 关键组件(类/函数 + 锚点) → 数据流/时序 → 关键算法(伪代码) → 设计取舍/坑。
- 锚点必须真实存在；写前 Read，不靠记忆或旧文档。
- 前端(Ch12)、eval(Ch13) 与后端核心同等深度（用户点名）。
- 不臆造实现；拿不准的不写死。

---

### Task 1: 骨架 + Ch0–2（元信息 / 总览 / 进程拓扑）

**Files:**
- Create: `docs/superpowers/specs/2026-06-29-nanoresearch-whole-project-blueprint.md`
- Read: `backend/pyproject.toml`, `docker-compose.yml`, `Dockerfile`, `.env.example`,
  `backend/nanoresearch/__main__.py`, `worker.py`, `cli/commands.py`(头部), `config/paths.py`

**内容契约:**
- 顶部 TOC（18 章锚链接）+ Ch0 元信息（范围/读法/基线 commit）。
- Ch1：能力地图、分层 ASCII 架构图、**包→层对照表**（覆盖全部顶层包）、技术栈与关键依赖。
- Ch2：四类进程（gateway/agent/worker/RAG-MCP）职责与生命周期、端口与外部依赖拓扑、
  `NANORESEARCH_HOME` 多租户路径模型。

- [ ] Step 1: Read 上述文件，确认进程入口与端口/依赖事实。
- [ ] Step 2: 写 TOC + Ch0–2 进 spec。
- [ ] Step 3: 核对：包→层表覆盖 §3.2 全部包；端口/进程名与 docker-compose 一致。
- [ ] Step 4: Commit `docs(spec): blueprint Ch0-2 总览与进程拓扑`。

### Task 2: Ch3 Channels / Ch4 Bus / Ch5 Session

**Files:**
- Read: `channels/base.py`, `manager.py`, `registry.py`, 抽样 `telegram.py`/`slack.py`/`feishu.py`；
  `bus/events.py`, `queue.py`, `stream.py`, `redis_keys.py`, `pending_reaper.py`；`session/manager.py`

**内容契约:**
- Ch3：`BaseChannel` 契约 + `ChannelManager` 生命周期 + registry 插件发现；13 平台差异表；入/出站归一化。
- Ch4：Inbound/OutboundMessage 模型；queue/stream/redis_keys/pending_reaper 可靠投递与回收。
- Ch5：`SessionManager`(PG-backed) session_key 模型、aware-UTC、idle-gate、跨进程一致性。

- [ ] Step 1: Read channels/bus/session 关键文件取签名与行号。
- [ ] Step 2: 写 Ch3–5。
- [ ] Step 3: 核对锚点；13 平台列全。
- [ ] Step 4: Commit `docs(spec): blueprint Ch3-5 接入/总线/会话`。

### Task 3: Ch6 Agent 核心 / Ch7 记忆与压缩

**Files:**
- Read: `agent/loop.py`, `runner.py`, `context.py`, `subagent.py`, `hook.py`,
  `agent/tools/base.py`, `registry.py` + 13 工具抽样；`agent/memory.py`,
  `conversation_knowledge_extractor.py`；近期 consolidation 相关代码。

**内容契约:**
- Ch6：AgentLoop 主引擎 + `_connect_mcp` + turn 调度；AgentRunner ReAct(伪代码)；ContextBuilder 渐进式披露；
  13 内置工具逐个一行契约 + 锚点；subagent 异步回注；hooks。
- Ch7：MemoryStore + 抽取器；token 感知 consolidation/compaction（tail-protect/target-ratio/锚点保留/
  startup 计 turn + idle-gate）触发与算法（伪代码）。

- [ ] Step 1: Read agent/* 与 memory 相关代码。
- [ ] Step 2: 写 Ch6–7（ReAct 与 consolidation 给伪代码）。
- [ ] Step 3: 核对 13 工具与签名/行号；consolidation 常量与代码一致。
- [ ] Step 4: Commit `docs(spec): blueprint Ch6-7 Agent核心与记忆`。

### Task 4: Ch8 Providers

**Files:**
- Read: `providers/base.py`, `registry.py`, `model_factory.py`, `anthropic_provider.py`,
  `openai_compat_provider.py`, `azure_openai_provider.py`, `openai_codex_provider.py`, `transcription.py`

**内容契约:** LLMProvider 抽象 + registry + model_factory(role 显式)；各 provider 差异；多租户 LLM 配置。

- [ ] Step 1: Read providers/*。
- [ ] Step 2: 写 Ch8。
- [ ] Step 3: 核对抽象方法签名与 provider 列表。
- [ ] Step 4: Commit `docs(spec): blueprint Ch8 Providers`。

### Task 5: Ch9 RAG 子系统（最大，深写）

**Files:**
- Read: `rag/ingestion/pipeline.py`, `unified.py`, `chunking/*`, `transform/*`, `embedding/*`, `storage/*`,
  `graph/persist.py`；`rag/core/query_engine/*`(hybrid_search/dense/sparse/fusion/reranker/query_processor)；
  `rag/core/response/*`, `core/session/*`, `core/trace/*`；`rag/mcp_server/*`；`rag/internal_loop/*`

**内容契约:** 9.1 分区总览 → 9.2 摄入管道(parse→chunk→transform→embed→store+graph) →
9.3 查询引擎(dense+sparse 多路 → RRF fusion 伪代码 → Cross-Encoder rerank) →
9.4 MCP Server(协议/工具/async_tasks/per-uid 隔离) → 9.5 internal_loop 数据飞轮 →
9.6 query rewrite/指代消解(A 类透传链)。

- [ ] Step 1: Read rag/ 关键文件（分批）。
- [ ] Step 2: 写 Ch9 六小节，RRF 给伪代码。
- [ ] Step 3: 核对管道阶段顺序与类名/行号；per-uid 隔离落点。
- [ ] Step 4: Commit `docs(spec): blueprint Ch9 RAG 子系统`。

### Task 6: Ch10 Deep Research / Ch11 持久化

**Files:**
- Read: `research/planner.py`, `searcher.py`, `synthesizer.py`, `refiner.py`, `reporter.py`,
  `runner.py`, `types.py`, `knowledge_search.py`, `knowledge_lint.py`；
  `storage/database.py`, `models.py`, `repositories/*`(9 个)；`bus/redis_client.py`；
  `rag` 向量库落点；`backend/migrations/`

**内容契约:** Ch10 Planner→Searcher→Synthesizer→Refiner→Reporter 编排 + 覆盖度收敛(伪代码) + 引用溯源。
Ch11 三层持久化职责（PG models+9 repo / Redis bus+cache 写策略 / Chroma per-uid）+ migrations。

- [ ] Step 1: Read research/ 与 storage/。
- [ ] Step 2: 写 Ch10–11。
- [ ] Step 3: 核对 5 阶段类名、9 repo 列全、收敛逻辑。
- [ ] Step 4: Commit `docs(spec): blueprint Ch10-11 Research与持久化`。

### Task 7: Ch12 Web 服务与前端（认真写）

**Files:**
- Read: `server/main.py`, `server/routers/*`(7), `server/middleware/auth.py`；
  `web/src/main.js`, `router/index.js`, `App.vue`, `layouts/AppLayout.vue`,
  `stores/*`(5), `apis/base.js` + 抽样 apis, `composables/useRunStream.js`,
  `views/ChatView.vue`/`AgentEvalView.vue`/`RunDetailView.vue`, `components/RunTimeline.vue`

**内容契约:** 12.1 后端 server/main + 7 router + auth 中间件 + SSE/run-events；
12.2 前端 Vue 路由/布局、Pinia stores、apis 拦截器、useRunStream SSE 时序、关键 view 数据流与组件树；
12.3 前后端契约（REST + WS/SSE 端点清单、鉴权流）。

- [ ] Step 1: Read server/ 与 web/src/ 关键文件。
- [ ] Step 2: 写 Ch12 三小节（与后端核心同等深度）。
- [ ] Step 3: 核对 7 router 路径、stores/apis 列全、SSE 时序与 useRunStream 一致。
- [ ] Step 4: Commit `docs(spec): blueprint Ch12 Web服务与前端`。

### Task 8: Ch13 Eval 闭环（认真写）

**Files:**
- Read: `eval/badcase_detector.py`, `badcase_classifier.py`, `judge.py`, `context_diagnoser.py`,
  `optimizer.py`, `data_flywheel.py`, `regression_detector.py`, `test_runner.py`, `sandbox.py`,
  `snapshot.py`, `score_sample.py`, `tunable.py`, `evaluator.py`；触发 hooks 落点

**内容契约:** 闭环链路 detector→classifier→judge(LLM-as-judge)→diagnoser→optimizer→flywheel→regression；
test_runner/sandbox/snapshot/tunable 角色；事件驱动 hooks（非 cron）；A1 SDD 锚点(strict_replay+execution_sanity)。

- [ ] Step 1: Read eval/* 全量关键文件。
- [ ] Step 2: 写 Ch13（数据流图 + 各组件契约 + 触发机制）。
- [ ] Step 3: 核对组件链路与类名/行号；hooks 触发点。
- [ ] Step 4: Commit `docs(spec): blueprint Ch13 Eval闭环`。

### Task 9: Ch14–17（Cron/Heartbeat / Security / Config / 横切）

**Files:**
- Read: `cron/service.py`, `types.py`, `command/router.py`, `builtin.py`, `heartbeat/service.py`；
  `auth/jwt.py`, `password.py`, `server/middleware/auth.py`, `security/network.py`, `SECURITY.md`；
  `config/schema.py`, `loader.py`, `paths.py`, `migration.py`, `utils/env_compat.py`；
  `rag/observability/logger.py`, `core/trace/*`, `utils/cache_metrics.py`, `tests/`(结构), `.github/workflows`

**内容契约:** Ch14 cron/heartbeat（事件驱动 hooks 非 timer）；Ch15 auth/security(JWT/bcrypt/network)；
Ch16 config(schema/loader/paths/migration + env_compat 双读)；Ch17 横切（trace/可观测性/错误处理/测试策略/CI）。

- [ ] Step 1: Read 上述模块。
- [ ] Step 2: 写 Ch14–17。
- [ ] Step 3: 核对锚点；env_compat 双读、测试子域列全。
- [ ] Step 4: Commit `docs(spec): blueprint Ch14-17 定时/安全/配置/横切`。

### Task 10: Ch18 附录 + 全文终审

**Files:**
- Modify: spec 文件（补 Ch18 + 终审修订）

**内容契约:** 关键文件:行号索引、术语表、设计决策文档链接（specs/sdd/redis-* 系列）。
终审：TOC 链接通；无 TBD；18 层齐；抽样锚点核对；前端/eval 深度达标。

- [ ] Step 1: 写 Ch18 附录。
- [ ] Step 2: 全文终审（对照 design §7 验收清单逐条）。
- [ ] Step 3: 修订发现的问题。
- [ ] Step 4: Commit `docs(spec): blueprint Ch18 附录 + 终审`。

---

## Self-Review（计划对照 design spec）

- **覆盖**: design §4 的 0–18 章 → Task 1–10 全覆盖（Ch0-2→T1, 3-5→T2, 6-7→T3, 8→T4, 9→T5,
  10-11→T6, 12→T7, 13→T8, 14-17→T9, 18→T10）。无遗漏。
- **深度约束**: 前端 Ch12 独立成 Task 7、eval Ch13 独立成 Task 8，满足"认真写"。
- **占位扫描**: 各 Task 内容契约具体到文件与小节，无 TBD。
- **方法一致**: 每 Task 均 Read→写→核对锚点→commit，符合 design §5 方法论。
