# NanoResearch 全项目工程蓝图 Spec —— Design / 蓝图

> 本文件不是最终 spec，而是「最终大 spec 文档」的设计与契约（blueprint）。
> 它定义最终 spec 的目标、范围、组织结构、每层应覆盖的内容、撰写方法论与验收标准。
> 最终 spec 由 writing-plans 拆成的执行计划逐层撰写产出。

- **作者**: 协作产出（brainstorming 阶段）
- **日期**: 2026-06-29
- **基线 commit**: `3b590a8f`（branch `feature/consolidation-compaction`）
- **状态**: 待用户审阅 → 转 writing-plans

---

## 1. 目标与受众

写一份 **NanoResearch 整个项目的 as-built（已建成）工程蓝图**，单个 Markdown 大文档，
做到「越详细越好、每个层级都讲清楚」。

- **受众**: 完整工程蓝图读者（既要懂"为什么这样设计"，也要能落到"代码里有什么"）。
- **定位**: as-built，描述**当前代码真实的样子**，不是未来设计提案；不照抄已过时的
  `docs/PROJECT_STRUCTURE.md`（仍停留在 `nanobot/` 命名、2026-03-31）。
- **粒度**: 架构 + 关键代码。每层给：职责一句话 → 关键类/函数真实签名 + `文件:行号` 锚点
  → 数据流/时序 → 关键算法伪代码 → 设计取舍。**不逐函数罗列**。

### 用户明确的深度约束
- 整体比常规 spec **更详细**。
- **前端（Vue）和评测（eval）两层必须认真写**，不得略写。

---

## 2. 组织方式（已确认：方案 A 自顶向下分层）

按架构层自顶向下走，顺着「入站消息 → Agent → 工具/子系统 → 持久化」的请求流，
再补上不在单请求链路上的横切子系统（eval / cron / 安全 / 配置 / 部署）。

配一张 **「包 → 层」对照表**，让文档同时是一张代码地图：读者从任意 `backend/nanoresearch/<pkg>/`
都能定位到对应章节。

被否决的备选：B（按请求生命周期叙事，eval/cron 不好塞）、C（按目录逐包，分层感弱、像文件清单）。

---

## 3. 系统事实基线（探查所得，撰写时以代码复核为准）

### 3.1 进程拓扑（四类进程）
- `nr gateway` —— FastAPI HTTP :18790 + WebSocket :8765
- `nr agent` —— CLI 交互式 Agent 主循环
- `worker.py` —— arq 异步 worker（耗时任务 / subagent）
- RAG MCP Server —— `python -m nanoresearch.rag.mcp_server` 子进程，stdio 与 Agent 通信
- 外部依赖：Postgres 16 / Redis / ChromaDB

### 3.2 后端包清单（`backend/nanoresearch/`，py 文件数）
```
120 rag/        23 agent/      16 channels/   14 eval/      13 storage/
12  server/     10 research/    9 providers/   8 bus/        5 utils/
5   skills/      5 scripts/     5 config/      5 cli/        3 cron/
3   command/     3 auth/        2 session/     2 security/   2 heartbeat/  2 dashboard/
```
入口：`__main__.py`、`worker.py`(565)、`cli/commands.py`(1563)、`server/main.py`(151)。

### 3.3 前端（`web/src/`，Vue）
- `views/`: Chat / Agents / AgentDetail / AgentEval / Knowledge / KnowledgeDetail / KnowledgeEval / RunDetail / Login
- `stores/`: agent / chat / knowledge / settings / user（Pinia）
- `apis/`: agents / agentEval / auth / conversations / knowledge / runs / settings / workspace / base
- `composables/useRunStream.js`（SSE 流式）、`components/RunTimeline.vue` 等、`router/`、`layouts/`

### 3.4 部署
docker-compose：`postgres` + `nanoresearch-gateway`（build Dockerfile，挂 `~/.nanoresearch`，
`command: gateway`）+ `nanoresearch-cli`（profile=cli）。多租户经 `NANORESEARCH_HOME`。
包名 `nanoresearch-ai`，CLI 入口 `nr` / `nanoresearch`。

---

## 4. 最终 Spec 的章节大纲与每章内容契约

> 下列每一章在最终文档中都展开为完整小节。标 ★ 的为用户点名要"认真写"的层。

**0. 文档元信息** — 范围、读法、基线 commit、术语指引。

**1. 系统总览**
- 1.1 它是什么 / 能力地图：Deep Research、Agentic RAG、混合记忆、Subagent 异步。
- 1.2 分层架构图（ASCII，复刻并按当前代码修正 README 的图）。
- 1.3 **包 → 层 对照表**（每个 `nanoresearch/*` 包归到哪一层 + 一句话职责）。
- 1.4 技术栈与关键依赖（FastAPI/SQLAlchemy+asyncpg/Redis+arq/Chroma/MCP/tiktoken/
  sentence-transformers/anthropic+openai）。

**2. 进程与运行形态（部署即架构）**
- 2.1 四类进程职责、生命周期、谁拉起谁。
- 2.2 端口、外部依赖拓扑、docker-compose 服务图。
- 2.3 `NANORESEARCH_HOME` 路径模型与多租户隔离；`get_nanoresearch_home()` 与 loader fallback。

**3. 接入层 Channels（13 平台）**
- 3.1 `BaseChannel` 契约 + `ChannelManager` 生命周期 + `registry` 插件发现机制。
- 3.2 入站/出站消息归一化；13 平台差异表（telegram/discord/slack/feishu/dingtalk/
  wecom/weixin/qq/whatsapp/matrix/mochat/email）。

**4. 消息总线 Message Bus（Redis）**
- 4.1 `events.py`：InboundMessage / OutboundMessage 事件模型。
- 4.2 `queue` / `stream` / `redis_keys` / `redis_client` / `pending_reaper` / `redis_monitor`：
  可靠投递、pending 回收、监控。

**5. 会话层 Session**
- `SessionManager`（PG-backed）：`session_key` 模型、aware-UTC 归一、idle-gate、跨进程一致性。

**6. Agent 核心**
- 6.1 `AgentLoop`(955)：主引擎、`_connect_mcp()`、turn 调度、与 bus/session 的接线。
- 6.2 `AgentRunner`(326)：ReAct 循环（think→act→observe）。
- 6.3 `ContextBuilder`(616)：提示词组装、Skill 渐进式披露、上下文预算。
- 6.4 工具层：`base`/`registry` + 13 内置工具逐个给契约（filesystem/shell/spawn/cron/
  message/mcp/web/research/history/graph_retrieval/paper_fetch）。
- 6.5 `subagent.py`：异步执行 + MessageBus 回注主会话。
- 6.6 `hook.py`：Agent hooks（事件驱动扩展点）。

**7. 记忆与上下文压缩**
- 7.1 `MemoryStore`(679) + `conversation_knowledge_extractor`：每轮抽事实入库。
- 7.2 Token 感知 consolidation/compaction：tail-protect / target-ratio / 锚点保留 /
  startup 触发计 turn + idle-gate；confidence≥0.7 摘要持久化。（近期主线工作，重点写）

**8. LLM Providers**
- 8.1 `LLMProvider` 抽象 + `registry` + `model_factory`（role 显式解析）。
- 8.2 各 provider：anthropic / openai_compat / azure_openai / openai_codex / transcription；
  多租户 LLM 配置；20+ provider 兼容策略。

**9. RAG 子系统（最大，单独成章）** ★ 深写
- 9.1 总体分区：ingestion / core(query_engine+response+session+trace) / mcp_server /
  internal_loop / libs。
- 9.2 摄入管道 `ingestion/`：parsing → chunking(document/semantic) → transform(metadata/
  entity/image_caption/chunk_refiner) → embedding(dense/sparse/batch) → storage(vector_upserter/
  bm25_indexer/image_storage) + graph/persist；`unified.py` 统一摄入。
- 9.3 查询引擎 `core/query_engine/`：dense + sparse(BM25/jieba) 多路召回 → `fusion`(RRF) →
  `reranker`(Cross-Encoder) → `query_processor`。
- 9.4 RAG MCP Server：`protocol_handler` / `server` / `async_tasks`；工具集；**per-uid collection 隔离**。
- 9.5 `internal_loop/`：knowledge 数据飞轮（runner/state/tools/cleanup/prompts）。
- 9.6 query rewrite / 指代消解：A 类透传链（caller_session_key → plan_query），子进程 PG SessionManager。

**10. Deep Research 编排**
- `research/`：Planner → Searcher → Synthesizer → Refiner → Reporter；覆盖度阈值驱动收敛；
  `knowledge_search` / `knowledge_lint`；带引用溯源的报告生成。

**11. 持久化层**
- 11.1 Postgres：`storage/database` + `models` + 9 个 repo（agent/agent_eval/conversation/
  eval/graph/knowledge/run/user/user_settings）。
- 11.2 Redis：bus + 缓存（DEL+全量 RPUSH 写策略、volatile-lru）。
- 11.3 Chroma：向量库，per-uid 隔离。
- 11.4 migrations。

**12. Web 服务与前端** ★ 认真写
- 12.1 后端 `server/main`(151) + 7 router（agent / agent_eval / chat / eval / knowledge /
  settings / workspace）+ `middleware/auth`；SSE/run-events-stream。
- 12.2 前端 Vue：路由与布局、Pinia stores、apis 层与 `base.js` 拦截器、`useRunStream` SSE 时序、
  关键 view（Chat / AgentEval / KnowledgeEval / RunDetail）的数据流与组件树。
- 12.3 前后端契约：REST + WS/SSE 端点清单、鉴权流。

**13. 评测与优化闭环 Eval** ★ 认真写
- `eval/`：badcase_detector → badcase_classifier → judge（LLM-as-judge）→ context_diagnoser →
  optimizer → data_flywheel → regression_detector；test_runner / sandbox / snapshot / score_sample /
  tunable / high_risk_keywords。触发是事件驱动 hooks（非 cron）。A1 SDD 锚点：strict_replay +
  execution_sanity 防两洞，高 divergence 候选人审。

**14. 定时与心跳 Cron / Heartbeat**
- `cron/`(service/types) + `command` 路由 + `heartbeat/service`；触发用事件驱动 hooks，非 scheduled timer。

**15. 安全与鉴权 Auth / Security**
- `auth/`(jwt/password bcrypt) + `server/middleware/auth` + `security/network`；SECURITY.md 要点。

**16. 配置系统 Config**
- `schema`(Pydantic) / `loader` / `paths` / `migration` + `utils/env_compat`（NANOBOT_* → NANORESEARCH_* 双读）。

**17. 横切关注点**
- 可观测性 / trace（rag/observability、trace_collector/context、llm_trace、cache_metrics）；
  统一错误处理；测试策略（`tests/` 13 个子域：agent/channels/cli/config/cron/providers/rag/
  research/security/tools/utils 等）；CI（.github/workflows）。

**18. 附录**
- 关键文件:行号索引；术语表；已有设计决策文档链接（docs/superpowers/specs、docs/sdd、redis-* 系列）。

---

## 5. 撰写方法论（落实"以代码为准"）

1. **每章先读再写**：写每层前实际 Read 该层关键文件，签名 / 行号 / 算法均从代码取，不靠记忆或旧文档。
2. **锚点格式**：引用代码用 `backend/nanoresearch/<path>:<line>`，可点击跳转。
3. **算法用伪代码**：consolidation 触发、RRF fusion、ReAct 循环、覆盖度收敛等给伪代码而非贴整段源码。
4. **每章固定骨架**：职责 → 关键组件（类/函数 + 锚点）→ 数据流/时序 → 关键算法 → 设计取舍/坑。
5. **图**：架构图、进程拓扑、RAG 管道、请求时序用 ASCII 图。
6. **不臆造**：拿不准的实现不写死；与旧文档冲突时以当前代码为准并可标注差异。

## 6. 最终交付物

- 路径：`docs/superpowers/specs/2026-06-29-nanoresearch-whole-project-blueprint.md`（单文件）。
- 形态：顶部 TOC + 18 章分层结构，预计 2000–4000+ 行。
- 语言：中文（与项目现有文档一致）。

## 7. 验收标准

- [ ] 18 层全部成章，无 TBD / 占位。
- [ ] 「包 → 层」对照表覆盖 `nanoresearch/` 全部顶层包，无遗漏。
- [ ] 前端、eval 两章达到与后端核心同等深度（用户点名）。
- [ ] 每章含真实 `文件:行号` 锚点与关键签名，抽样核对与代码一致。
- [ ] 关键算法（consolidation / RRF / ReAct / 覆盖度收敛）有伪代码。
- [ ] 与过时 `PROJECT_STRUCTURE.md` 的命名/结构差异已纠正。
