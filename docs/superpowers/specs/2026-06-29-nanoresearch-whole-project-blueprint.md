# NanoResearch 全项目工程蓝图（as-built Spec）

> **基线 commit**：`3b590a8f`（branch `feature/consolidation-compaction`，代码截面）— 文档撰写于 2026-06-29。
> **范围**：后端 `backend/nanoresearch/` 全部子系统 + 前端 `web/`（Vue3）+ Docker/配置/部署层。
> **粒度**：架构 + 关键代码。每章给「职责一句话 → 关键组件（真实类/函数 + `文件:行号` 锚点）→ 数据流/时序 → 关键算法（伪代码）→ 设计取舍/坑」。
> **锚点约定**：后端代码引用形如 `backend/nanoresearch/<相对路径>:<行号>`，前端形如 `web/src/<相对路径>:<行号>`，行号指向基线代码、撰写时已人工核对。
> **撰写方式**：本文由 9 个并行子调研单元按层独立读码产出、统一装配；各层锚点经独立自检。与已过时的 `docs/PROJECT_STRUCTURE.md`（旧 `nanobot/` 命名）冲突处一律以当前代码为准。

## 目录

- [Ch0 文档元信息](#ch0-文档元信息)
- [Ch1 系统总览](#ch1-系统总览)
- [Ch2 进程与运行形态](#ch2-进程与运行形态)
- [Ch3 接入层 Channels](#ch3-接入层-channels)
- [Ch4 消息总线](#ch4-消息总线)
- [Ch5 会话层](#ch5-会话层)
- [Ch6 Agent 核心](#ch6-agent-核心)
- [Ch7 记忆与上下文压缩](#ch7-记忆与上下文压缩)
- [Ch8 LLM Providers](#ch8-llm-providers)
- [Ch9 RAG 子系统](#ch9-rag-子系统)
- [Ch10 Deep Research 编排](#ch10-deep-research-编排)
- [Ch11 持久化层](#ch11-持久化层)
- [Ch12 Web 服务与前端](#ch12-web-服务与前端)
- [Ch13 评测与优化闭环 Eval](#ch13-评测与优化闭环-eval)
- [Ch14 定时与心跳](#ch14-定时与心跳)
- [Ch15 安全与鉴权](#ch15-安全与鉴权)
- [Ch16 配置系统](#ch16-配置系统)
- [Ch17 横切关注点](#ch17-横切关注点)
- [Ch18 附录](#ch18-附录)

---


## Ch0 文档元信息

### 范围与目的

本文档是 **NanoResearch** 项目的 **as-built 工程蓝图**，记录截至基线提交 `3b590a8f` 时代码库的真实实现，供参与维护、扩展或审计的工程师快速定位任意模块的职责、接口与设计取舍。全文覆盖后端包 `backend/nanoresearch/`（Python 包名 `nanoresearch-ai`，版本 `0.1.4.post6`）、前端 `web/`（Vue3 + Pinia，见 Ch12）及配套的 Docker/配置层。

### 读法建议

- 首次阅读：按章顺序读 Ch0–Ch2，建立系统全景后按需跳读。
- 调试特定模块：直接跳到对应章节，利用「关键组件」列出的 `文件:行号` 锚点定位代码。
- 所有代码引用形如 `backend/nanoresearch/<相对路径>:<行号>`，行号已在写作时人工核对，指向截至基线的当前代码行。

### 章节速览

| 章 | 主题摘要 |
|---|---|
| Ch0 | 本文元信息、读法、术语约定（本章） |
| Ch1 | 系统总览：能力地图、分层架构图、包→层对照表、技术栈 |
| Ch2 | 进程与运行形态：四类进程、端口拓扑、NANORESEARCH\_HOME 路径模型 |
| Ch3 | 接入层 Channels：BaseChannel 契约、ChannelManager、12 平台适配 |
| Ch4 | 消息总线：InboundMessage/OutboundMessage、MessageBus、Redis Stream、PendingReaper |
| Ch5 | 会话层 Session：SessionManager、PG+Redis 混合存储、aware-UTC、idle-gate |
| Ch6 | Agent 核心：AgentLoop 主引擎、AgentRunner ReAct 循环、ContextBuilder、13 内置工具、Subagent |
| Ch7 | 记忆与上下文压缩：MemoryStore、知识抽取、token 感知 consolidation（tail-protect/target-ratio） |
| Ch8 | LLM Providers：LLMProvider 抽象、ModelFactory 多角色解析、各 provider 适配 |
| Ch9 | RAG 子系统：摄入流水线、混合检索（dense+sparse+RRF+rerank）、MCP Server、per-uid 隔离、内循环 |
| Ch10 | Deep Research 编排：Planner→Searcher→Synthesizer→Refiner→Reporter、覆盖度收敛、引用溯源 |
| Ch11 | 持久化层：PostgreSQL（models+9 repo）、Redis（DEL+RPUSH 策略）、ChromaDB、手动 SQL 迁移 |
| Ch12 | Web 服务与前端：FastAPI server + 7 router + SSE、Vue3 前端（router/stores/apis/useRunStream） |
| Ch13 | 评测与优化闭环 Eval：badcase 检测→分类→LLM judge→诊断→优化→沙箱回放→飞轮→回归 |
| Ch14 | 定时与心跳：CronService、HeartbeatService、事件驱动触发约束 |
| Ch15 | 安全与鉴权：JWT（HS256）、bcrypt、OAuth、SSRF 网络过滤 |
| Ch16 | 配置系统：Config Pydantic 模型、加载链、env\_compat 双读兼容 |
| Ch17 | 横切关注点：可观测性/trace、缓存指标、测试策略、CI |
| Ch18 | 附录：关键文件\:行号索引、术语表、设计决策文档链接 |

### 术语说明

文中「AgentLoop」特指 `backend/nanoresearch/agent/loop.py:AgentLoop` 这一类；「gateway」特指 `nr gateway` CLI 子命令所启动的进程（频道机器人 + 仪表盘，**不含** REST API）；「serve」特指 `nr serve` 启动的 FastAPI REST API 进程；「worker」特指 arq 后台 worker 进程（`WorkerSettings`）。

---

## Ch1 系统总览

**一句话职责**：NanoResearch 是面向个人/团队的 AI 研究助手后端，将 Deep Research、Agentic RAG、token 感知记忆压缩与多租户频道接入融合在单一 Python 包中，支持本地独立部署和多租户服务模式。

### 1.1 能力地图

| 能力 | 简述 |
|---|---|
| **Deep Research** | 给定研究问题，Planner 分解子问题，Searcher 网络检索，Synthesizer 聚合，Reporter 输出完整报告（`research/` 包，最多 3 轮迭代，默认阈值 6.0 分）。 |
| **Agentic RAG** | Agent 在循环中动态调用 RAG MCP Server（stdio 子进程）完成多轮向量/稀疏检索、重排序和引用生成，支持 Agentic 与 Simple 两种 RAG 路径。 |
| **混合记忆** | token 感知上下文构建（`agent/context.py`，预算 3000 token）+ 异步 MemoryConsolidator（`agent/memory.py`）对长会话执行摘要压缩，防止上下文溢出。 |
| **Subagent 异步** | SpawnTool 将子任务并行分发给独立 AgentLoop 实例（`agent/subagent.py`），所有子 agent 事件回写 Redis Stream，SSE 端点直接 XREAD。 |
| **多频道接入** | 支持 Telegram、DingTalk、飞书/Lark、Slack、QQ、WeCom、WeChat、Matrix、Socket.IO、Discord 等，由 `channels/` 包统一适配。 |

### 1.2 分层架构图（ASCII）

```
┌───────────────────────────────────────────────────────────┐
│                   接入层 (Channels / CLI)                   │
│  Telegram · DingTalk · 飞书 · Slack · QQ · WeCom · CLI   │
│  REST API (server/ :8000/18790)  ·  Dashboard (:8765)     │
└─────────────────────────┬─────────────────────────────────┘
                          │ InboundMessage / OutboundMessage
                          ▼
┌───────────────────────────────────────────────────────────┐
│                  消息总线 (bus/)                            │
│   MessageBus (asyncio.Queue)  ·  Redis Streams            │
│   PendingReaper  ·  RedisMonitor                          │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│               智能体循环 (agent/loop.py)                   │
│  ContextBuilder  ·  LLM Provider  ·  ToolRegistry        │
│  MemoryConsolidator  ·  SubagentManager  ·  CronTool     │
└──────────┬─────────────┬──────────────┬───────────────────┘
           │             │              │
           ▼             ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐
│  RAG MCP     │ │  Research    │ │  通用工具层             │
│  (rag/)      │ │  (research/) │ │  web · fs · exec       │
│  stdio 子进程 │ │  Deep        │ │  cron · message        │
│  Chroma      │ │  Research    │ │  paper_fetch · spawn   │
└──────────────┘ └──────────────┘ └────────────────────────┘
           │             │              │
           └─────────────┴──────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│                  持久化层                                  │
│  PostgreSQL 16 (storage/  +  SQLAlchemy/asyncpg)         │
│  Redis  (bus/stream.py 事件流 · session 缓存)             │
│  ChromaDB  (rag/libs/vector_store/)                       │
└───────────────────────────────────────────────────────────┘
```

### 1.3 包→层对照表

| 架构层 | 顶层包 | 职责一句话 |
|---|---|---|
| CLI 入口 | `cli` | Typer 命令集，定义 `gateway` / `serve` / `agent` / `status` 等子命令，内含 `build_loop_config()` 共享配置工厂 |
| 配置 | `config` | Pydantic Config 模型（schema.py）、JSON 加载/保存（loader.py）、路径辅助（paths.py）、迁移脚本 |
| 接入频道 | `channels` | Telegram / DingTalk / 飞书 / Slack / QQ / WeCom / WeChat / Matrix / Discord / Socket.IO 适配器 + ChannelManager |
| 消息总线 | `bus` | asyncio MessageBus 队列、Redis Stream 事件写入（stream.py）、PendingReaper、RedisMonitor、RedisKeys 命名空间 |
| 智能体核心 | `agent` | AgentLoop（loop.py）、ContextBuilder（context.py）、MemoryConsolidator（memory.py）、SubagentManager（subagent.py）、ToolRegistry 及全部内置工具（tools/） |
| 斜杠命令 | `command` | CommandRouter + CommandContext，处理 `/search` `/help` 等斜杠指令，解耦于 AgentLoop 主循环 |
| LLM 提供商 | `providers` | ModelFactory 多角色解析（chat/ingestion\_llm 等）、AnthropicProvider / OpenAICompatProvider / AzureOpenAI 适配器、PROVIDERS 注册表 |
| RAG 子系统 | `rag` | MCP Server（mcp\_server/，stdio 传输）、Agentic 检索引擎（core/query\_engine/）、文档摄入流水线（ingestion/）、向量库封装（libs/vector\_store/）、BM25 索引、嵌入/重排序工厂 |
| Deep Research | `research` | Planner → Searcher → Synthesizer → Reporter 四步管道，KnowledgeSearch 封装，支持自评估触发重试 |
| HTTP API | `server` | FastAPI `create_app()`（main.py）、路由（chat\_router / knowledge\_router / agent\_router / eval\_router 等）、SSE 流、arq 任务提交、静态文件挂载 |
| 会话管理 | `session` | SessionManager 读写 JSONL 会话文件 + Redis 混合存储，per-uid 隔离 |
| 数据持久化 | `storage` | SQLAlchemy 异步模型（models.py）+ ORM Repositories（用户/知识库/运行记录/对话/评估等），`init_engine()` + `get_session_factory()` |
| 认证 | `auth` | JWT 签发与校验（jwt.py）、bcrypt 密码哈希（password.py）、OAuth CLI 工具 |
| 安全 | `security` | 请求输入过滤、CORS 策略、速率限制辅助 |
| 评估 | `eval` | RAGAS 集成、Agent 自评估采样（`EVAL_SAMPLING_RATE`）、badcase 检测、评估 Repository |
| 计划任务 | `cron` | CronService（cron/service.py）——JSON 持久化、时间/间隔触发，触发器均为事件驱动（禁止 OS cron） |
| 心跳 | `heartbeat` | HeartbeatService 定时主动向 Agent 推送任务并把结论路由回用户频道 |
| 仪表盘 | `dashboard` | 轻量 FastAPI 应用（dashboard/server.py），端口 8765，提供会话/技能/定时任务/记忆预览 REST 接口 |
| 技能 | `skills` | Markdown 格式内置技能定义（SKILL.md），由 SkillsLoader 在 AgentLoop 启动时注入系统提示 |
| 脚本 | `scripts` | 数据库迁移、用户初始化、批量操作等一次性运维脚本 |
| 模板 | `templates` | Workspace 启动模板文件（AGENTS.md / SOUL.md / USER.md / TOOLS.md）|
| 工具函数 | `utils` | 环境变量兼容垫片（env\_compat.py）、token 统计、日志配置、cache metrics、通用辅助函数 |

### 1.4 技术栈与关键依赖

以下来源于 `backend/pyproject.toml`，按功能域归纳：

| 功能域 | 关键库 |
|---|---|
| Web 框架 | `fastapi>=0.115`、`uvicorn>=0.34`（ASGI） |
| 数据库 | `sqlalchemy[asyncio]>=2.0`、`asyncpg>=0.30`（PG 16 异步驱动） |
| 缓存 / 队列 | `redis[asyncio]>=5.0`、`arq>=0.26`（arq 异步任务队列） |
| 向量数据库 | `chromadb>=1.5.9` |
| MCP 协议 | `mcp>=1.26` |
| Token 计数 | `tiktoken>=0.12` |
| 嵌入模型 | `sentence-transformers>=5.6` |
| LLM 客户端 | `anthropic>=0.45`、`openai>=2.8` |
| CLI | `typer>=0.20`、`prompt-toolkit>=3.0`、`rich>=14` |
| 频道 SDK | `python-telegram-bot[socks]>=22.6`、`dingtalk-stream>=0.24`、`lark-oapi>=1.5`、`slack-sdk>=3.39`、`qq-botpy>=1.2`、`python-socketio>=5.16` |
| 认证 | `python-jose[cryptography]>=3.3`、`bcrypt>=4.0` |
| 评估 | `ragas>=0.4.3` |
| PDF 处理 | `pypdf>=6.10`、`pymupdf>=1.27`、`magic-pdf[full]>=0.6`、`markitdown[pdf]>=0.1.6` |
| 数据验证 | `pydantic>=2.12`、`pydantic-settings>=2.12` |
| 定时任务 | `croniter>=6.0` |
| 其他 | `loguru>=0.7`（日志）、`httpx>=0.28`（HTTP 客户端）、`json-repair>=0.57`（LLM 输出修复） |

---

## Ch2 进程与运行形态

**一句话职责**：NanoResearch 在运行时由四类独立进程组成——频道网关（`nr gateway`）、REST API 服务（`nr serve`）、CLI 交互代理（`nr agent`）和 ARQ 后台 Worker；RAG MCP Server 作为 AgentLoop 的 stdio 子进程动态派生，不算独立进程类型。

### 2.1 四类进程

#### 进程 1：频道网关（`nr gateway`）

**职责**：连接所有外部消息频道（Telegram / DingTalk / 飞书等），驱动 AgentLoop 处理入站消息，并对外暴露仪表盘。

**定义位置**：`backend/nanoresearch/cli/commands.py:576`

**启动方式**：
```
nr gateway [--port PORT] [--workspace PATH] [--config PATH]
```
在 Docker 中由 `nanoresearch-gateway` 服务以 `command: ["gateway"]` 启动（`docker-compose.yml:36`）。

**关键组件**：
- `AgentLoop`（`backend/nanoresearch/agent/loop.py:54`）：单实例，处理频道消息。
- `ChannelManager`（`backend/nanoresearch/channels/manager.py`）：并发启动所有已启用频道监听器。
- `CronService`（`backend/nanoresearch/cron/service.py`）：定时任务调度，JSON 持久化。
- `HeartbeatService`（`backend/nanoresearch/heartbeat/service.py`）：定时触发主动任务。
- 仪表盘 FastAPI App（`backend/nanoresearch/dashboard/server.py:24`）：绑定 **:8765**，提供会话/技能状态查询。

**注意**：`GatewayConfig.port`（默认 18790，`backend/nanoresearch/config/schema.py:104`）在 `nr gateway` 中被读取并打印（`cli/commands.py:596–598`），但**该命令本身不绑定 18790 端口**；18790 是 `nr serve` 对外暴露的 REST API 端口（可通过 `--port 18790` 指定）。这与 docker-compose 中同时暴露 18790 的配置相对应，但两者属于不同子命令。

**生命周期**：`asyncio.run(run())`（`cli/commands.py:828`）驱动三个并发协程——`_dashboard_server.serve()`、`agent.run()`、`channels.start_all()`；KeyboardInterrupt 触发有序关闭（close\_mcp / heartbeat.stop / cron.stop / channels.stop\_all）。

#### 进程 2：REST API 服务（`nr serve`）

**职责**：提供多租户 HTTP REST API 和 SSE 流式响应，接受 Web 前端请求，通过 arq 将 Agent 任务投递给 Worker 执行。

**定义位置**：`backend/nanoresearch/cli/commands.py:1339`

**启动方式**：
```
nr serve [--host HOST] [--port PORT] [--workspace PATH] [--config PATH]
```
默认端口 8000（typer option 默认值，`cli/commands.py:1342`）；生产环境推荐 `--port 18790`。

**关键组件**：
- `create_app()`（`backend/nanoresearch/server/main.py:32`）：FastAPI app 工厂，版本 2.0.0（`server/main.py:92`）。
- 路由：`chat_router`、`agent_router`、`knowledge_router`、`eval_router`、`settings_router`、`workspace_router`（`server/main.py:122–135`）。
- arq 连接池（`server/main.py:68`）：`app.state.arq_pool`，用于 `enqueue_job(run_agent_job, ...)`。
- 静态文件：`/rag-images`（rag 图片）、`/`（前端 `web/dist`，若存在）（`server/main.py:144–149`）。

**数据流**：HTTP 请求 → `chat_router.create_run()` → `arq_pool.enqueue_job(run_agent_job, ...)` → Worker 执行 → Redis Stream `run_events:{run_id}` → SSE 端点 XREAD → 客户端。

**生命周期**：`uvicorn.run(fastapi_app, host=host, port=port)`（`cli/commands.py:1467`）；lifespan 钩子（`server/main.py:34`）负责启动/停止 PendingReaper、RedisMonitor、arq pool。

#### 进程 3：CLI 交互代理（`nr agent`）

**职责**：终端交互式 Agent，直接对话无需频道或 HTTP，适合本地开发和单次调试。

**定义位置**：`backend/nanoresearch/cli/commands.py:839`

**启动方式**：
```
nr agent [-m "消息"] [-s SESSION_ID] [--workspace PATH]
```
两种模式：`-m` 指定消息时单次执行；否则进入 `prompt_toolkit` 交互 REPL。

**关键组件**：
- 独立的 `AgentLoop` 实例（`cli/commands.py:892`），不通过 Redis / arq。
- `StreamRenderer`（`cli/stream.py`）：终端流式渲染。
- `PromptSession`（`cli/commands.py:117`）：历史记录、粘贴模式、UTF-8 安全。

**生命周期**：单次 `asyncio.run()`，退出命令（`exit` / `quit` / `/exit`）或 Ctrl-C 终止。

#### 进程 4：ARQ 后台 Worker

**职责**：在独立进程中执行 AgentLoop 任务（`run_agent_job`）和文档摄入任务（`ingest_document_task`），所有运行事件写入 Redis Stream。

**定义位置**：`backend/nanoresearch/worker.py:558`（`WorkerSettings` 类）

**启动方式**：
```
arq nanoresearch.worker.WorkerSettings
```

**关键组件**：
- `WorkerSettings`（`worker.py:558`）：注册 `functions`、`redis_settings`、`on_startup`/`on_shutdown`、`max_jobs=10`、`job_timeout=7200s`。
- `run_agent_job()`（`worker.py:251`）：主 ARQ 任务，构建 `AgentLoop`（`_build_agent_loop()`，`worker.py:75`）并调用 `loop.process_direct()`；支持 Simple RAG 与 Agentic 两条路径。
- `ingest_document_task()`（`worker.py:457`）：文档摄入，调用 `rag/ingestion/unified.ingest_document()`。
- `startup()`（`worker.py:147`）：加载 `.env`、调用 `build_loop_config()`（`cli/commands.py:449`）初始化共享配置。
- 事件写入：所有流式 delta 通过 `xadd_event(redis, run_stream_key, ...)` 写入 `run_events:{run_id}` Redis Stream（`bus/stream.py:19`）；SSE 端点 XREAD 消费，不经过 Worker 中继。

**子进程：RAG MCP Server**（`backend/nanoresearch/rag/mcp_server/__main__.py:9`）

当 `config.json` 中配置了 `tools.mcp_servers` 且 `type=stdio`（或 command 指向 `nanoresearch.rag.mcp_server`）时，`AgentLoop._ensure_mcp_connected()`（`agent/loop.py:232`）通过 `connect_mcp_servers()`（`agent/tools/mcp.py:255`）以 stdio 管道启动子进程。MCP Server 以 JSON-RPC 2.0 over stdio 暴露 RAG 工具（向量检索、重排序、引用生成等），stdout 仅含协议消息，日志全部写 stderr（`rag/mcp_server/server.py:25`）。

### 2.2 端口与外部依赖拓扑（ASCII）

```
  用户 / 前端
     │
     ├──HTTP/SSE──► :18790  (nr serve，FastAPI REST API)
     │
     └──HTTP────► :8765  (nr gateway，仪表盘 Dashboard)

  nr gateway ──(InboundMsg)──► AgentLoop ──(tool)──► RAG MCP Server
                                                       (stdio 子进程)
                                    │
  nr serve ──(enqueue_job)──► Redis ──► ARQ Worker ──► AgentLoop
                                │                        │
                                ▼                        ▼
                          Redis Streams           ChromaDB (:default)
                          run_events:{run_id}     (向量存储，HTTP/本地)

  所有进程
     ├────► PostgreSQL 16  (:5432)
     │        DATABASE_URL = postgresql+asyncpg://...
     │
     └────► Redis          (:6379)
              REDIS_URL = redis://localhost:6379
```

**外部服务依赖**：

| 服务 | 默认地址 | 用途 |
|---|---|---|
| PostgreSQL 16 | `postgres:5432`（docker）/ `localhost:5432`（本地） | 用户/知识库/运行记录/对话/评估等持久化 |
| Redis | `redis://localhost:6379` | arq 任务队列、SSE 事件 Stream、session 缓存 |
| ChromaDB | 本地文件（由 RAG settings 配置）或 HTTP 模式 | 向量存储，per-uid collection 隔离 |

**注意**：docker-compose.yml（`docker-compose.yml:39`）`nanoresearch-gateway` 服务同时暴露 18790 和 8765，但实际上 `nr gateway` 命令只绑定 8765（仪表盘）；18790 是为 `nr serve` 预留，需在同一容器内或另起服务使用 `nr serve --port 18790` 才会绑定。

### 2.3 NANORESEARCH_HOME 路径模型与多租户隔离

**关键函数**：`get_nanoresearch_home()`（`backend/nanoresearch/config/loader.py:18`）

**行为**：
1. 调用 `apply_legacy_env_compat()`（`utils/env_compat.py`）——将旧 `NANOBOT_HOME` 环境变量透传为 `NANORESEARCH_HOME`（双读兼容，计划在 v0.3.0 后移除）。
2. 读取 `os.environ.get("NANORESEARCH_HOME")`；若存在则展开 `~` 后返回。
3. **唯一 fallback**：`Path.home() / ".nanoresearch"`（`loader.py:29`）。

```python
# loader.py:18-29（节选）
def get_nanoresearch_home() -> Path:
    apply_legacy_env_compat()
    raw = os.environ.get("NANORESEARCH_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".nanoresearch"   # ← 唯一 fallback
```

**多租户隔离**：每个租户设置不同的 `NANORESEARCH_HOME`（如 `/data/tenant_x`）并挂载独立卷（docker-compose.yml 注释第 31–35 行），所有运行时路径全部从该根派生：

| 路径辅助函数 | 路径 | 定义 |
|---|---|---|
| `get_config_path()` | `$HOME/.nanoresearch/config.json` | `loader.py:80` |
| `get_workspace_path()` | `$HOME/.nanoresearch/workspace` | `paths.py:38` |
| `get_runtime_subdir(name)` | `$HOME/.nanoresearch/<name>/` | `paths.py:17` |
| `get_media_dir(channel)` | `$HOME/.nanoresearch/media/<channel>/` | `paths.py:21` |
| `get_cron_dir()` | `$HOME/.nanoresearch/cron/` | `paths.py:27` |
| `get_logs_dir()` | `$HOME/.nanoresearch/logs/` | `paths.py:31` |
| `get_cli_history_path()` | `$HOME/.nanoresearch/history/cli_history` | `paths.py:51` |
| `get_legacy_sessions_dir()` | `$HOME/.nanoresearch/sessions/` | `paths.py:60` |

所有路径辅助函数均定义于 `backend/nanoresearch/config/paths.py`，**不含独立 fallback**——唯一的 fallback 收敛于 `loader.py:29` 的 `~/.nanoresearch`。

**部署模式**（`NANORESEARCH_MODE`，`loader.py:35`）：

- `local`（默认）：凭证读取链 = `user_settings.extra.providers` → `config.json` → `settings.yaml` → 环境变量。
- `server`：凭证唯一来源 = `user_settings.extra.providers`（DB 存储），禁止环境变量 fallback，适合多租户生产部署。

**工作区子目录结构**（`gateway` 启动后自动生成）：
```
~/.nanoresearch/
├── config.json          ← 主配置文件
├── workspace/           ← AgentLoop 工作目录（文件读写、代码执行沙箱）
│   ├── memory/MEMORY.md
│   ├── sessions/        ← JSONL 会话文件（legacy 单租户路径）
│   ├── skills/          ← 用户自定义技能
│   └── cron/jobs.json   ← 计划任务持久化
├── rag/
│   ├── documents/<kb_id>/  ← 摄入文档永久存储
│   └── images/             ← RAG 图片静态文件
└── history/cli_history ← CLI prompt_toolkit 历史
```


## Ch3 接入层 Channels

**职责**：将异构 IM 平台的原生事件统一归一化为 `InboundMessage`，并将 `OutboundMessage` 分发回对应平台；向上层完全屏蔽平台差异。

---

### 3.1 BaseChannel 抽象契约

`backend/nanoresearch/channels/base.py:15` 定义 `BaseChannel(ABC)`，所有平台实现必须继承并实现以下三个抽象方法：

| 方法签名 | 锚点 | 语义 |
|---|---|---|
| `async def start(self) -> None` | `base.py:65` | 长连接/轮询循环；阻塞直到 `stop()` 被调用 |
| `async def stop(self) -> None` | `base.py:77` | 清理资源、断开连接 |
| `async def send(self, msg: OutboundMessage) -> None` | `base.py:82` | 发送出站消息；失败时必须 raise 以触发 ChannelManager 重试 |

可选覆盖（默认 no-op，覆盖后启用流式）：

```
async def send_delta(self, chat_id: str, delta: str, metadata: dict | None = None) -> None
```
`base.py:94`：流式 chunk 投递。`supports_streaming` 属性（`base.py:107`）在 config 设置了 `streaming=True` 且子类真正覆盖了 `send_delta` 时返回 `True`。

`async def login(self, force: bool = False) -> bool`（`base.py:52`）：交互式登录（如微信二维码扫码），默认返回 `True`（无需登录）。

**核心辅助方法**

`_handle_message(sender_id, chat_id, content, media, metadata, session_key)` `base.py:123`：每个平台实现在收到消息后调用此方法。它完成：
1. `is_allowed(sender_id)` ACL 校验（`base.py:113`），`allow_from=[]` 拒绝所有人，`"*"` 开放；
2. 若 `supports_streaming` 为真，则在 metadata 中注入 `_wants_stream: True`；
3. 构造 `InboundMessage` 并调用 `bus.publish_inbound(msg)`（`base.py:167`）。

**类属性**

```python
name: str = "base"          # 平台标识符，必须唯一
display_name: str = "Base"  # 日志显示名
transcription_api_key: str  # Groq Whisper 音频转写密钥（由 ChannelManager 注入）
```
`base.py:23-25`

---

### 3.2 ChannelManager 生命周期

`backend/nanoresearch/channels/manager.py:19`，`ChannelManager` 负责：

**初始化**（`manager.py:37`，`_init_channels`）：
1. 调用 `discover_all()`（registry）获取所有已知 Channel 类；
2. 按 `config.channels.<name>.enabled` 过滤，跳过未启用的；
3. 实例化 channel，注入 Groq key（`manager.py:56`）；
4. 对空 `allow_from` 的 channel 调用 `_validate_allow_from()` 直接 `SystemExit`（`manager.py:64`）。

**启动**（`manager.py:79`，`start_all`）：
- 创建 `_dispatch_outbound` 协程 Task（`manager.py:86`）；
- 每个 channel 各一个 asyncio Task 运行 `channel.start()`（`manager.py:92`）；
- `asyncio.gather` 等待所有 Task（它们应永久运行）。

**停止**（`manager.py:97`，`stop_all`）：取消 dispatch Task，逐一调用 `channel.stop()`。

**出站调度循环**（`manager.py:117`，`_dispatch_outbound`）：
- 维护一个 `pending: list[OutboundMessage]` 缓冲区，用于 delta 合并后溢出的消息（因为 `asyncio.Queue` 不支持 push_front）；
- 对连续的 `_stream_delta` 消息调用 `_coalesce_stream_deltas()`（`manager.py:167`）做批量合并，降低 API 调用频次；
- 过滤进度/工具提示消息（依 `config.channels.send_progress` / `send_tool_hints`，`manager.py:137`）；
- 调用 `_send_with_retry(channel, msg)`（`manager.py:217`）投递。

**指数退避重试**（`manager.py:217`，`_send_with_retry`）：
```
delays = (1s, 2s, 4s)   # manager.py:16
最多 config.channels.send_max_retries 次
CancelledError 直接 re-raise（优雅关闭）
```

---

### 3.3 Registry 插件发现

`backend/nanoresearch/channels/registry.py:54`，`discover_all()` 两阶段合并：

```
阶段 1（内建）：pkgutil.iter_modules(nanoresearch.channels.__path__)
               过滤掉 {"base", "manager", "registry"}
               对每个模块名：importlib.import_module → 找第一个 BaseChannel 子类

阶段 2（外部）：importlib.metadata.entry_points(group="nanoresearch.channels")
               每个 entry point：ep.load() 得到 Channel 类

合并规则：内建优先（外部无法覆盖同名内建）
          `{**external, **builtin}` → builtin 在后，覆盖 external 的同名 key
```

关键函数：
- `discover_channel_names()` `registry.py:17`：零 import，仅 pkgutil 扫描返回模块名列表；
- `load_channel_class(module_name)` `registry.py:28`：动态 import 并反射出第一个 BaseChannel 子类；
- `discover_plugins()` `registry.py:40`：entry_points 发现，失败单独 warn 不影响其他插件。

---

### 3.4 入站/出站归一化路径

```
平台原生事件
    │
    ▼
channel._on_message() / _on_handler()
    │  ① 下载媒体到本地路径
    │  ② 拼装 content 文本
    │  ③ 可选：audio → Groq Whisper 转写
    ▼
BaseChannel._handle_message(sender_id, chat_id, content, media, metadata, session_key)
    │  ① ACL 校验 is_allowed()
    │  ② 注入 _wants_stream 标志
    │  ③ 构造 InboundMessage
    ▼
MessageBus.publish_inbound(msg)   → inbound asyncio.Queue

──────────────────────────────────────────────────────────────

Agent 生成 OutboundMessage
    ▼
MessageBus.publish_outbound(msg)  → outbound asyncio.Queue
    ▼
ChannelManager._dispatch_outbound()
    │  ① 过滤 progress/tool_hint
    │  ② 合并 stream delta
    │  ③ 查找 channels[msg.channel]
    ▼
channel.send(msg) 或 channel.send_delta(chat_id, delta, metadata)
    ▼
平台 API（HTTP / WebSocket / SMTP …）
```

`session_key` 默认值由 `InboundMessage.session_key` 属性计算（`events.py:23`）：
```python
return self.session_key_override or f"{self.channel}:{self.chat_id}"
```
Telegram topic 群组会通过 `_derive_topic_session_key()` 注入 `telegram:{chat_id}:topic:{thread_id}` 覆盖（`telegram.py:601`）。

---

### 3.5 平台差异表（12 平台）

> 注：代码中实际存在 12 个平台文件，规格书所列"13"与实际不符，下表以代码为准。

| 平台 | 文件 | 传输方式 | 特点 |
|---|---|---|---|
| Telegram | `telegram.py` | Long polling（getUpdates） | python-telegram-bot；支持流式（edit_message_text，0.6s 节流）；topic 群组线程隔离 session；media group 缓冲 0.6s 合并；emoji 反应 + typing 指示；Markdown→HTML 渲染 |
| Discord | `discord.py` | Gateway WebSocket（v10） | 原生 WebSocket 实现（无第三方 SDK 框架）；httpx HTTP 客户端发消息；附件上传 ≤20MB；intents=37377 |
| Slack | `slack.py` | Socket Mode（WebSocket） | slack_sdk；bot_token + app_token 双 token；thread 回复；react emoji；支持 group_policy=mention |
| Feishu | `feishu.py` | WebSocket 长连接（lark-oapi） | 飞书/Lark 官方 SDK；rich card / interactive 消息解析；图片/文件/音频类型映射；需运行时 import check |
| DingTalk | `dingtalk.py` | Stream Mode（dingtalk_stream SDK） | 官方 Stream 协议，无需公网 IP；httpx 下载附件；ClientID+ClientSecret 认证 |
| WeCom | `wecom.py` | WebSocket 长连接（wecom_aibot_sdk） | 企业微信 AI Bot 平台；bot_id+secret 认证；welcome_message；无需公网 IP |
| WeChat | `weixin.py` | HTTP 长轮询（ilinkai.weixin.qq.com） | 个人微信接入；逆向 openclaw-weixin v1.0.3 协议；QR 码登录获取 token；base64 图片上传 |
| QQ | `qq.py` | botpy SDK（官方 Bot API） | C2C（私信）+ Group（群）两种消息类型；rich media base64 上传（msg_type=7）；附件分 image/file 两类 |
| WhatsApp | `whatsapp.py` | WebSocket（Node.js 桥接） | Python 侧连 ws://localhost:3001；Node.js 桥用 @whiskeysockets/baileys 处理 WA Web 协议；bridge_token 鉴权 |
| Matrix | `matrix.py` | matrix-nio sync 长轮询 | E2E 加密（EncryptedMedia 解密）；nh3 HTML 消毒；mistune Markdown 渲染；InviteEvent 自动接受邀请；typing 30s 续期 |
| MoChat | `mochat.py` | Socket.IO（HTTP polling 备选） | 企微 SCRM；msgpack 可选序列化；cursor 去重（max 2000 已见 ID）；0.5s debounce 游标持久化 |
| Email | `email.py` | IMAP 轮询（30s）+ SMTP 回复 | imaplib/smtplib 标准库；DKIM/SPF 验签（Authentication-Results）；mark_seen；body 截断 12000 字符 |

---

### 3.6 设计取舍

- **内建优先于插件**：防止外部包意外替换核心平台实现（`registry.py:67-70`）。
- **流式合并在 ChannelManager 而非 Channel**：各平台实现只需 `send_delta`，合并/节流逻辑集中在 `_coalesce_stream_deltas`，避免重复。
- **typing 指示和 emoji 反应是 best-effort**：失败只 `logger.debug`，不阻塞主流程（`telegram.py:904`）。
- **坑：allow_from 空列表直接 SystemExit**（`manager.py:67`）：运维配错时会阻止启动，但能及早暴露配置问题，好于静默拒绝所有消息。

---

## Ch4 消息总线

**职责**：在 Channel 层与 Agent 层之间提供解耦的异步消息通道；在多进程/跨节点场景通过 Redis 补充可靠投递和事件回放。

---

### 4.1 InboundMessage / OutboundMessage 字段

`backend/nanoresearch/bus/events.py`

**InboundMessage**（`events.py:9`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `channel` | `str` | 平台标识（"telegram"、"slack" …） |
| `sender_id` | `str` | 发送者 ID |
| `chat_id` | `str` | 会话/群组 ID |
| `content` | `str` | 消息文本（媒体已转为文本标注） |
| `timestamp` | `datetime` | 本地接收时间（`datetime.now()`，非 aware-UTC） |
| `media` | `list[str]` | 本地媒体文件路径列表 |
| `metadata` | `dict[str, Any]` | 平台元数据（message_id、thread id 等） |
| `session_key_override` | `str \| None` | 线程级 session 覆盖 |

`session_key` 属性（`events.py:22`）：`session_key_override or f"{channel}:{chat_id}"`

**OutboundMessage**（`events.py:28`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `channel` | `str` | 目标平台 |
| `chat_id` | `str` | 目标会话 ID |
| `content` | `str` | 回复文本 |
| `reply_to` | `str \| None` | 可选：引用原消息 ID |
| `media` | `list[str]` | 附件路径 |
| `metadata` | `dict[str, Any]` | 控制信号（见下） |

**metadata 控制信号约定**（非结构化，运行时约定）：

| key | 含义 |
|---|---|
| `_wants_stream` | inbound：请求流式回复 |
| `_stream_delta` | outbound：这是一个流式 chunk |
| `_stream_end` | outbound：流式结束信号 |
| `_stream_id` | outbound：同一流的唯一标识（防多并发流串扰） |
| `_streamed` | outbound：内容已通过流式投递，`send()` 应跳过 |
| `_progress` | outbound：进度消息（可按配置过滤） |
| `_tool_hint` | outbound：工具执行提示（可按配置过滤） |

---

### 4.2 Queue 与 Stream 的角色

**MessageBus（queue.py）**：进程内通信，`asyncio.Queue`。

`backend/nanoresearch/bus/queue.py:8`

```
inbound:  asyncio.Queue[InboundMessage]
outbound: asyncio.Queue[OutboundMessage]

publish_inbound(msg)   → await inbound.put(msg)
consume_inbound()      → await inbound.get()
publish_outbound(msg)  → await outbound.put(msg)
consume_outbound()     → await outbound.get()
```

`MessageBus` 是**同进程内**（gateway 进程）Channel 和 AgentLoop 之间的零延迟通道，不经 Redis。

**stream.py**：跨进程事件回放，Redis Stream。

`backend/nanoresearch/bus/stream.py:19`

```
xadd_event(redis, stream_key, event)
    ├── 载荷 ≤ 8KB：单条 XADD
    └── 载荷 > 8KB：分块（chunk_group_id + chunk_index + total_chunks）
        每块 ≤ 8KB，独立 XADD，随后 expire(86400s)

xread_next(redis, stream_key, last_id, timeout_ms=5000)
    ├── XREAD count=20 block=timeout_ms
    ├── 重组 chunk_group_id 碎片
    └── 返回 (events: list[dict], new_last_id: str)

get_last_id(redis, stream_key)
    └── xrevrange count=1 → tail ID（用于子进程启动时锚定游标）
```

`run_events:{run_id}` stream 用于 gateway ↔ worker 的 Agent 执行事件回放（24h 窗口）。`chat_events:{chat_id}` stream 用于 SSE 推送（`redis_keys.py:68`）。

---

### 4.3 Redis Keys 命名约定

`backend/nanoresearch/bus/redis_keys.py`

| 前缀 | 示例 | TTL | 说明 |
|---|---|---|---|
| `pending:{session_key}` | `pending:telegram:12345` | 无（手动 DEL） | 悬挂任务 ID 集合；non-volatile 防 lru 驱逐 |
| `cancel:{session_key}` | `cancel:telegram:12345` | 无 | 取消信号 |
| `job:{job_id}` | `job:abc-uuid` | 无 | 任务元数据 |
| `run_events:{run_id}` | `run_events:abc` | 86400s | Agent 执行事件 stream |
| `session:msg:{uid}:{ch}:{chat_id}` | `session:msg:admin:telegram:12345` | 7200s | 会话消息列表（Redis List） |
| `session:meta:{uid}:{ch}:{chat_id}` | `session:meta:admin:telegram:12345` | 7200s | 会话元数据（Redis Hash） |
| `agent:{agent_id}` | `agent:xxx` | 1800s | Agent 配置热缓存 |
| `user_settings:{uid}` | `user_settings:admin` | 1800s | 用户设置缓存 |
| `kb:meta:{kb_id}` | `kb:meta:xxx` | 600s | 知识库元数据缓存 |
| `chunk:{ns}:{chunk_id}` | `chunk:admin:abc` | 21600s | RAG 文本块缓存 |
| `embedding:{text_hash}` | `embedding:sha256...` | 3600s | 向量缓存 |
| `chat_events:{chat_id}` | `chat_events:12345` | 无固定 | SSE 事件 stream |

`pending:*`、`cancel:*`、`job:*` 无 TTL，在 `volatile-lru` 驱逐策略下不会被驱逐（无 expire → non-volatile）。

---

### 4.4 PendingReaper：悬挂消息可靠回收

`backend/nanoresearch/bus/pending_reaper.py:22`

```
PendingReaper(interval=300s, idle_threshold=7200s)
    │
    └── 每 300s 执行一次 _reap()
            ├── SCAN cursor 遍历所有 pending:* key
            └── 对每个 key 调用 _stale_members(redis, key, now)
                    ├── 解析 member 格式："{task_id}:{unix_ts}"
                    ├── 条件 1：now - ts >= 7200s  （age guard）
                    └── 条件 2：redis.exists(chat_events:{chat_id}) == 0
                            → 双重门控确认 stream 已消亡
                    → 满足两个条件才标记为 stale
            ├── SREM 批量删除 stale members
            └── SCARD == 0 时 DEL 整个 key
```

**为何双重门控**（`pending_reaper.py:28-31` 注释）：stream 不存在可能是误报——新启动 session 的前一轮 stream 刚过期。age guard 防止误杀仍在运行的任务。

`_stale_members` 中的 `session_key` 解析（`pending_reaper.py:93`）：`key[len("pending:"):]` 得到 `{channel}:{chat_id}`，分割取 `chat_id` 构造 `chat_events` key 做存活检查。

---

### 4.5 RedisMonitor：健康监控

`backend/nanoresearch/bus/redis_monitor.py:43`

两个独立职责：

**3-A 驱逐告警**（`redis_monitor.py:92`，`_check_stats`）：每 60s `INFO stats`，对比 `evicted_keys` delta，delta > 0 立即 `logger.warning`。

**3-B 内存采样**（`redis_monitor.py:109`，`_scan_memory`）：每 300s 对 12 个前缀各 SCAN 采样最多 50 个 key，pipeline `MEMORY USAGE` 求平均，追加写入 `logs/redis_metrics.jsonl`（`redis_monitor.py:40`，路径可通过 `REDIS_METRICS_PATH` 环境变量覆盖）。

---

### 4.6 消息流转全景图

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  Gateway 进程                        │
┌──────────┐        │  ┌────────────┐        ┌───────────────────────┐   │
│ 平台 API │──事件──→│  │  Channel   │        │     AgentLoop         │   │
│(Telegram)│        │  │ _on_message│──put──→│ consume_inbound()     │   │
└──────────┘        │  └────────────┘        │ 查 Session / 调 Agent │   │
                    │                        │ publish_outbound(msg) │   │
┌──────────┐        │  ┌────────────┐        └───────────┬───────────┘   │
│ 平台 API │←send──│  │  Channel   │←get───  outbound Q │               │
│(Telegram)│        │  │   send()   │        ┌───────────┘               │
└──────────┘        │  └────────────┘        │ _dispatch_outbound         │
                    │                        │ (coalesce + retry)        │
                    └────────────────────────┼──────────────────────────-┘
                                             │
                              Redis Streams / run_events:{run_id}
                                             │
                    ┌────────────────────────┼───────────────────────────┐
                    │                Worker 进程                          │
                    │            xread_next(stream_key, last_id)         │
                    │            子 Agent 执行                            │
                    │            xadd_event(result)                      │
                    └─────────────────────────────────────────────────────┘

      PendingReaper (300s)     RedisMonitor (60s/300s)
            │                         │
      SCAN pending:*           INFO stats + MEMORY USAGE
      双重门控删除 stale        eviction alert + metrics.jsonl
```

---

### 4.7 设计取舍

- **Queue 不过 Redis**：同进程通信用 `asyncio.Queue`，避免网络 round-trip；跨进程才走 Redis Stream。
- **8KB 分块**（`stream.py:15`）：规避 Redis Big Key 问题（Problem 5），分块内含 `chunk_group_id` 保证有序重组。
- **`xread` cursor 不用 `"+"`**（`stream.py:53` 注释）：用 `get_last_id` 锚定启动游标防止漏读（Problem 2）。
- **`decode_responses=True` 全局**（`redis_client.py:18`）：统一字符串类型，避免 bytes/str 混用 bug。
- **坑**：`pending:*` 和 `cancel:*` 必须无 TTL；若误设 TTL 会在 `volatile-lru` 下被驱逐，导致任务状态丢失。

---

## Ch5 会话层

**职责**：跨请求维护用户对话历史；提供三级缓存（进程内 → Redis → PostgreSQL），保证多进程一致性和服务重启后历史不丢。

---

### 5.1 核心数据结构

`backend/nanoresearch/session/manager.py`

**Session**（`manager.py:19`）：

```python
@dataclass
class Session:
    key: str                        # "{channel}:{chat_id}"，e.g. "telegram:12345"
    messages: list[dict]            # 完整消息列表，append-only
    created_at: datetime            # aware-UTC
    updated_at: datetime            # aware-UTC，每次 add_message 刷新
    metadata: dict[str, Any]        # 任意会话级元数据
    last_consolidated: int = 0      # 已整合消息的下标偏移（记忆压缩用）
```

`session_key` 格式：`"{channel}:{chat_id}"`（`events.py:23`），topic 群组为 `"telegram:{chat_id}:topic:{thread_id}"`（`telegram.py:605`）。

**get_history**（`manager.py:67`）的切片逻辑：
```
unconsolidated = messages[last_consolidated:]   # 跳过已压缩部分
sliced = unconsolidated[-max_messages:]         # 取尾部 ≤ 500 条
起点调整：找到第一个 user 消息开始
_find_legal_start()：向前扫描确保 tool_call_id 有对应 tool_calls 声明，
                      否则跳过孤儿 tool result 防 API 400
```

---

### 5.2 三级缓存架构

`backend/nanoresearch/session/manager.py:116`

```
get_or_create(key)
    │
    ├── L1：self._cache[key]           进程内字典，零延迟
    │         命中 → 直接返回
    │
    ├── L2：_redis_load(key)           Redis，2h TTL
    │         命中 → 写回 L1，返回
    │
    └── L3：_load(key)                 PG（_db_load）或 JSONL（_file_load）
              命中 → _redis_save() 预热 L2，写回 L1，返回
              未命中 → 新建 Session()，同样预热 L2
```

**save(session)**（`manager.py:231`）：
```
L1 更新（self._cache）
→ _redis_save()（异步，fire-and-forget on error）
→ _db_save() 或 _file_save()（PG 或 JSONL）
```

---

### 5.3 Redis 写策略（DEL + 全量 RPUSH）

`backend/nanoresearch/session/manager.py:172`，`_redis_save`：

```python
async with redis.pipeline(transaction=True) as pipe:
    pipe.delete(msg_key)                        # 先删旧 List
    if session.messages:
        pipe.rpush(msg_key, *[json.dumps(m) for m in session.messages])  # 全量重写
    pipe.hset(meta_key, mapping={...})
    pipe.expire(msg_key, SESSION_TTL)           # 7200s
    pipe.expire(meta_key, SESSION_TTL)
    await pipe.execute()                        # MULTI/EXEC 原子执行
```

**为何存完整 messages 而非尾部**（`manager.py:176-183` 注释）：`last_consolidated` 是指向完整列表的下标偏移。若 Redis 只存 `messages[last_consolidated:]` 而偏移不变，`get_history` 从 Redis 加载后会对已截断的列表再次应用偏移，静默丢弃全部历史（2026-06-28 生产 bug）。同理，`_db_save` 调用 `repo.replace_messages(conv.id, session.messages)` 也传完整列表（`manager.py:283`）。

**注意**：这与 MEMORY.md 中记录的 Redis 缓存写策略一致——DEL + 全量 RPUSH，`volatile-lru` 环境下增量写是正确性 bug。

---

### 5.4 时间归一化与 idle-gate 关系

`backend/nanoresearch/utils/helpers.py:86`

```python
def utcnow_aware() -> datetime:
    return datetime.now(timezone.utc)          # helpers.py:87

def as_aware_utc(dt: datetime) -> datetime:   # helpers.py:91
    if dt.tzinfo is None:
        # 兜底：假设 UTC，记 warning（遗留无 tz 行）
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
```

`Session.updated_at` 在每次 `add_message()` 时由 `utcnow_aware()` 刷新（`manager.py:43`）。从 DB / Redis 加载时用 `as_aware_utc()` 归一（`manager.py:158-159`，`manager.py:264-265`），确保所有时间戳统一为 aware-UTC，避免 tz-naive vs tz-aware 比较抛 TypeError。

**与 idle-gate 的关系**：会话整合（consolidation）模块判断"空闲"（idle-gate）时，比较 `session.updated_at`（aware-UTC）与 `utcnow_aware()` 的差值。两端统一 aware-UTC 后差值计算不受夏令时/时区偏移影响，是 idle 判断正确性的前提。

---

### 5.5 PG-backed 持久化

`backend/nanoresearch/session/manager.py:256`

```
_db_load(key)
    → ConversationRepository.get_by_session_key(key)
    → repo.get_messages(conv.id)          → Session

_db_save(session)
    → repo.get_by_session_key(session.key)
        未找到 → repo.create(key, uid, metadata, created_at)
    → repo.replace_messages(conv.id, session.messages)   # 全量替换
    → repo.update_meta(conv.id, last_consolidated, metadata, updated_at)
```

无 PG 时（`session_factory is None`，`manager.py:134`）回退到 JSONL 文件（`_sessions_dir/{safe_filename(key)}.jsonl`）。JSONL 文件中第一行是 `_type=metadata` 的 JSON 头，其余行是消息记录。

---

### 5.6 跨进程会话一致性

**Gateway-Worker 分离模型**：Gateway 进程持有 `SessionManager`（L1 + L2 + L3），Worker 进程通过 Redis Stream 获取执行事件，**不直接读写 Session**。Agent 执行结果通过 `run_events` stream 发回 gateway，由 gateway 的 SessionManager 写入会话历史。

**多 Gateway 实例**：`invalidate:session` Pub/Sub 频道（`redis_keys.py:72`）用于跨实例的 L1 缓存失效通知——某实例修改 session 后 publish，其他实例收到后调用 `session_manager.invalidate(key)` 清除本地 L1 缓存，下次读走 L2/L3。

**坑**：L1 (`self._cache`) 是进程内字典，没有 TTL；若 Worker 直接修改 PG/Redis 而不触发 invalidate，Gateway 的 L1 会有陈旧数据直到进程重启。当前架构通过约定"只有 gateway 写 session"规避此问题。

---

### 5.7 设计取舍

- **append-only messages + last_consolidated 偏移**：整合器只移动指针，不修改或删除原始消息，保持 LLM prompt cache 友好（前缀不变）。
- **JSONL 回退**：无 PG 时（开发/测试）降级到文件存储，`_file_load` 支持 legacy 路径迁移（`manager.py:297-302`）。
- **Redis 失败非致命**（`manager.py:203`）：`_redis_save` 捕获所有异常只 warn，不阻塞 PG 写入，Redis 降级为"最佳努力"缓存。
- **坑：`InboundMessage.timestamp` 非 aware-UTC**（`events.py:16`，`datetime.now()` 无 tz）：仅用于消息 payload 内记录，不参与 idle-gate 计算，但若未来需要跨时区对比需注意。


## Ch6 Agent 核心

AgentLoop 是消息驱动的主引擎，AgentRunner 是无业务层的 ReAct 执行核，ContextBuilder 负责上下文组装。三者分工明确：AgentLoop 处理生命周期与 I/O 路由，AgentRunner 驱动 think→act→observe 收敛，ContextBuilder 组装每轮发给 LLM 的消息列表。

---

### 6.1 AgentLoop 主引擎

**职责一句话**：从 MessageBus 消费入站消息，为每个会话（session_key）串行调度 AgentRunner，将最终回复写回 MessageBus。

**关键类与锚点**

| 组件 | 文件:行 | 说明 |
|---|---|---|
| `AgentLoop` 类 | `agent/loop.py:54` | 主引擎，持有 bus/runner/tools/context/subagents/memory_consolidator |
| `AgentLoop.__init__` | `agent/loop.py:68` | 接线所有依赖；context_window_tokens 默认 65 536 |
| `AgentLoop._connect_mcp` | `agent/loop.py:230` | 惰性一次性连接 MCP servers，失败后下条消息重试 |
| `AgentLoop.run` | `agent/loop.py:427` | 主循环；`asyncio.wait_for(bus.consume_inbound(), 1.0)` 轮询 |
| `AgentLoop._dispatch` | `agent/loop.py:459` | 为每条消息创建 asyncio.Task；per-session asyncio.Lock + 全局 Semaphore 并发控制 |
| `AgentLoop._process_message` | `agent/loop.py:684` | 单条消息完整处理路径（见下文数据流） |
| `AgentLoop._run_agent_loop` | `agent/loop.py:305` | 封装 AgentRunSpec 并调 runner.run；收集 eval snapshot |
| `AgentLoop._save_turn` | `agent/loop.py:892` | 将 runner 产生的新消息追加入 session；截断超长工具结果（_TOOL_RESULT_MAX_CHARS=16 000 chars） |
| `STARTUP_CONSOLIDATION_IDLE_SECONDS` | `agent/loop.py:23` | 启动压缩空闲阈值，默认 1800 s（env 可覆盖） |
| `STARTUP_MIN_PENDING_TURNS` | `agent/loop.py:24` | 启动压缩最少 pending turns，默认 2 |

**数据流（一条普通用户消息）**

```
bus.consume_inbound()           ← MessageBus Redis 队列
        │
        ▼
_dispatch(msg)                  ← asyncio.Task，per-session Lock
        │
        ▼
_process_message(msg)
  ├─ sessions.get_or_create(key)
  ├─ _check_pending_consolidation()  ← T1 startup trigger（详见 Ch7）
  ├─ commands.dispatch()             ← /stop 等命令优先处理
  ├─ memory_consolidator.maybe_consolidate_by_tokens()  ← T2 token trigger
  ├─ context.build_messages(history, current_message, ...)
  ├─ _run_agent_loop(initial_messages)
  │       └─ runner.run(AgentRunSpec)  ← ReAct 循环（见 §6.2）
  ├─ _save_turn(session, all_msgs)
  ├─ sessions.save(session)
  └─ _schedule_background(maybe_consolidate_by_tokens)  ← 回程再检查
        │
        ▼
bus.publish_outbound(OutboundMessage)
```

**MCP 接线**：`_connect_mcp`（`agent/loop.py:230`）在 `run()` 或 `process_direct()` 首次调用时触发，用 `AsyncExitStack` 持有所有 MCP session 生命周期。每条消息处理时通过 `_set_tool_context` 将 `session_key` 和 `kb_map` 注入 MCP 工具包装器，以便查询改写逻辑能按 session 隔离。

**并发控制**：per-session `asyncio.Lock`（`_session_locks`）保证同一会话串行；跨 session 通过 `asyncio.Semaphore(_max)`（环境变量 `NANORESEARCH_MAX_CONCURRENT_REQUESTS`，默认 3）控制并发上限（`agent/loop.py:161`）。

---

### 6.2 AgentRunner — ReAct 执行核

**职责一句话**：无业务层关注的 think→tool_call→observe 迭代循环，收到 LLM 无工具调用或达到上限时退出。

**关键类与锚点**

| 组件 | 文件:行 | 说明 |
|---|---|---|
| `AgentRunner` 类 | `agent/runner.py:61` | 持有 provider；P0 逃生计数器 |
| `AgentRunSpec` | `agent/runner.py:28` | 执行配置：initial_messages / tools / model / max_iterations(默认 40) / hook / concurrent_tools |
| `AgentRunResult` | `agent/runner.py:48` | 执行结果：final_content / messages / tools_used / usage / stop_reason |
| `AgentRunner.run` | `agent/runner.py:94` | 主循环 |
| `AgentRunner._execute_tools` | `agent/runner.py:264` | 并发(gather)或串行执行工具列表 |
| `AgentRunner._run_tool` | `agent/runner.py:290` | 单工具执行；异常捕获为 "Error:" 字符串 |

**ReAct 循环伪代码**

```
messages = initial_messages
for iteration in range(max_iterations):          # 默认上限 40
    response = provider.chat(messages, tools)    # think
    
    if response.has_tool_calls:                  # act
        messages += [assistant_msg_with_tool_calls]
        results = execute_tools(response.tool_calls)   # concurrent 或 serial
        messages += [tool_result_msgs]               # observe
        
        # P0 逃生通道：同一工具连续失败 >= max_consecutive_failures(默认 3)
        if consecutive_failures >= max_consecutive_failures:
            return AgentRunResult(stop_reason="consecutive_failures")
        
        hook.after_iteration()
        continue                                 # 下一 think 轮
    
    # 无工具调用 → 收敛
    messages += [assistant_msg]
    final_content = hook.finalize_content(response.content)
    return AgentRunResult(stop_reason="completed", ...)
else:
    return AgentRunResult(stop_reason="max_iterations")
```

**收敛/退出条件**（`agent/runner.py:94-262`）：
- `stop_reason="completed"`：LLM 返回无工具调用的文本回复
- `stop_reason="max_iterations"`：迭代达 `max_iterations` 上限
- `stop_reason="error"`：LLM 返回 `finish_reason=="error"`
- `stop_reason="tool_error"`：工具抛出异常且 `fail_on_tool_error=True`（子 Agent 场景）
- `stop_reason="consecutive_failures"`：P0 逃生通道，同一工具连续失败 3 次（`agent/runner.py:198`）

**工具执行**：`concurrent_tools=True`（主 Agent 默认）时用 `asyncio.gather` 并发执行本轮所有工具调用（`agent/runner.py:269`）；子 Agent 串行执行以便逐步失败检测。

---

### 6.3 ContextBuilder — 上下文组装

**职责一句话**：将工作区身份、引导文件、技能摘要、历史记忆、语义召回拼装为 LLM 可消费的消息列表，支持 cache_control 三段分块以最大化 prompt caching 命中率。

**关键类与锚点**

| 组件 | 文件:行 | 说明 |
|---|---|---|
| `ContextBuilder` 类 | `agent/context.py:26` | 持有 workspace/skills/knowledge_search/uid |
| `build_messages` | `agent/context.py:516` | 入口：组装 `[system, *history, user]` 消息列表 |
| `build_system_prompt_blocks` | `agent/context.py:148` | 三段 cache_control 块（Anthropic/OpenRouter 路径） |
| `build_system_prompt` | `agent/context.py:114` | 单字符串（DashScope/其它不支持 cache_control 的路径） |
| `_build_workspace_block` | `agent/context.py:206` | Block 0（workspace 级）：身份 + 引导文件 + 工具列表 |
| `_build_agent_block` | `agent/context.py:235` | Block 1（per-agent）：persona + 技能摘要 + agent registry + KB bindings |
| `_build_dynamic_suffix` | `agent/context.py:314` | Block 2（动态，不缓存）：MEMORY.md + 历史召回 + always-on skills |
| `build_history_context` | `agent/context.py:44` | 从 user_memory 向量库按 uid 隔离召回语义相关历史 |
| `DEFAULT_TOTAL_BUDGET` | `agent/context.py:20` | 3000 tokens（memory + knowledge 共享预算） |
| `MEMORY_BUDGET_RATIO` | `agent/context.py:21` | 0.6（memory 占 60%） |
| `KNOWLEDGE_BUDGET_RATIO` | `agent/context.py:22` | 0.4（knowledge/history 占 40%） |
| `CHARS_PER_TOKEN` | `agent/context.py:23` | 估算系数 4 chars/token |

**系统提示三段结构**

```
┌─────────────────────────────────────────────────────────┐
│  Block 0 (workspace-level, cache_control=ephemeral)     │
│  身份(runtime/workspace/guidelines) + 引导文件          │
│  (AGENTS.md / SOUL.md / USER.md / TOOLS.md) + 工具列表  │
├─────────────────────────────────────────────────────────┤
│  Block 1 (per-agent, cache_control=ephemeral)           │
│  Persona + 技能摘要(渐进式披露) + Agent Registry        │
│  + KB bindings                                          │
├─────────────────────────────────────────────────────────┤
│  Block 2 (dynamic, 不缓存)                              │
│  <memory>MEMORY.md</memory>                             │
│  <history>语义召回历史</history>                        │
│  + always-on skills 全文                                │
└─────────────────────────────────────────────────────────┘
```

**Skill 渐进式披露**（`agent/context.py:254-261`）：  
Block 1 中只注入技能摘要（标题+一句话描述）。Agent 调用 `read_file` 读取 `skills/{name}/SKILL.md` 后才拿到完整指令。标记 `available="false"` 的技能需先安装依赖。always-on skills（`skills.get_always_skills()`）是例外——它们的全文在每轮都会注入到 Block 2。

**上下文预算**（`_build_dynamic_suffix`，`agent/context.py:314`）：
- `memory_budget = total_budget * 0.6`（默认 1800 tokens）
- `knowledge_budget = total_budget * 0.4`（默认 1200 tokens）
- 超出预算时 `_truncate_to_budget` 在行边界截断并附 `... (truncated)` 标记（`agent/context.py:378`）

**历史召回隔离**：`build_history_context` 调用 `knowledge_search.search_user_memory_sync(query, top_k=5, uid=uid)` 按 uid 隔离，避免跨租户泄漏（`agent/context.py:69`）。

---

### 6.4 内置工具表

**Tool 基类**（`agent/tools/base.py:7`）：抽象属性 `name / description / parameters`，抽象方法 `execute(**kwargs)`；`side_effect` 属性（默认 `True`）用于 eval sandbox 区分只读工具。`ToolRegistry`（`agent/tools/registry.py:43`）持有 `dict[str, Tool]`，`execute` 前先调 `cast_params` + `validate_params`，错误字符串追加诊断建议。

| 工具名 | 主类 | 文件 | 一句话职责 |
|---|---|---|---|
| `read_file` | `ReadFileTool` | `tools/filesystem.py:59` | 分页读文件；自动处理 image/PDF；只读 |
| `write_file` | `WriteFileTool` | `tools/filesystem.py:173` | 写文件，自动创建父目录 |
| `edit_file` | `EditFileTool` | `tools/filesystem.py:241` | 精确文本替换，支持空白宽容匹配与全局替换 |
| `list_dir` | `ListDirTool` | `tools/filesystem.py:341` | 列目录树，含元数据 |
| `exec` | `ExecTool` | `tools/shell.py:15` | 执行 shell 命令，内置拒绝模式（rm -rf 等），超时控制 |
| `web_search` | `WebSearchTool` | `tools/web.py:75` | 搜索网络（Brave/DuckDuckGo/Tavily/Searxng/Jina 可配）；只读 |
| `web_fetch` | `WebFetchTool` | `tools/web.py` | 抓取 URL 并提取文本；SSRF 防护；只读 |
| `fetch_paper` | `PaperFetchTool` | `tools/paper_fetch.py:18` | 下载学术论文 PDF（arxiv 等）至 workspace/papers/，最大 50 MB |
| `message` | `MessageTool` | `tools/message.py:9` | 向用户发消息，唯一可以推送文件给用户的工具 |
| `spawn` | `SpawnTool` | `tools/spawn.py:11` | 在后台启动子 Agent 执行耗时任务 |
| `cron` | `CronTool` | `tools/cron.py:12` | 注册定时提醒和周期任务（条件注册，需 cron_service） |
| `research` | `ResearchTool` | `tools/research.py:18` | 启动自主网络研究（start/status/list），通常配合 spawn 后台执行 |
| `retrieve_by_entity` | `RetrieveByEntityTool` | `tools/graph_retrieval.py:16` | 通过知识图谱跨文档追踪实体，不依赖向量相似度；只读 |
| `mcp_*` | `MCPToolWrapper` | `tools/mcp.py:90` | 动态包装 MCP server 工具，命名格式 `mcp_{server}_{tool}`；RAG 查询工具 `retrieve_hybrid` 等被白名单标记为只读 |

> 注：`SearchHistoryTool`（`tools/history.py:18`）已标记 DEPRECATED，不再注册，历史召回改由 `build_history_context` 的 RAG 自动完成。

**工具注册**（`AgentLoop._register_default_tools`，`agent/loop.py:183`）：  
filesystem 4 个工具无条件注册；`exec` 受 `exec_config.enable` 控制；`cron` 受 `cron_service is not None`；`research` 受 `research_config.enabled`；`retrieve_by_entity` 受 `session_factory is not None`；MCP 工具在 `_connect_mcp` 时动态注入注册表。

---

### 6.5 子 Agent 异步执行与 MessageBus 回注

**SubagentManager**（`agent/subagent.py:26`）：管理后台子 Agent 的生命周期。主 Agent 调用 `spawn` 工具后，`SpawnTool` 委托给 `SubagentManager.spawn`，立即返回任务 ID，不阻塞主对话。

**时序图**

```
主 Agent                   SubagentManager            Redis
    │                            │                      │
    │ spawn(task=...)            │                      │
    │──────────────────────────►│                      │
    │                            │ asyncio.create_task  │
    │                            │ (_run_subagent)      │
    │                            │ SADD pending:{sk}    │──►│
    │ "Subagent started (id:x)" │                      │
    │◄──────────────────────────│                      │
    │ (继续处理用户输入)         │                      │
    │                            │                      │
    │                            │ [后台] runner.run()  │
    │                            │ 工具调用...          │
    │                            │                      │
    │                            │ _announce_result()   │
    │                            │  web 路径：xadd到    │
    │                            │  run_events/         │──►│
    │                            │  chat_events Stream  │
    │                            │  SREM pending:{sk}   │──►│
    │                            │  非 web 路径：       │
    │                            │  publish_inbound(    │
    │                            │  channel="system")   │
    │◄──────────────────────────│                      │
    │ [收到 system 消息,         │                      │
    │  触发新一轮 _process_      │                      │
    │  message，回复用户]        │                      │
```

**子 Agent 工具集**（`agent/subagent.py:119`）：fs 四件套 + exec + web_search + web_fetch + research；无 message 工具（禁止直接回复用户），无 spawn 工具（禁止递归派生），无 cron 工具。

**取消**：`_SubagentHook.after_iteration` 每轮检查 Redis `cancel_key(session_key)` 是否存在，有则抛 `CancelledError`（`agent/subagent.py:166`）。

**crash 安全**：SREM pending 在 xadd 之后执行（Web 路径），保证 SSE reader 观测到事件后 pending 计数才降为 0（`agent/subagent.py:261`）。

---

### 6.6 AgentHook — 扩展点

**AgentHook**（`agent/hook.py:27`）是 AgentRunner 暴露给上层的纯虚回调接口，默认实现为空操作（no-op）。

| 钩子方法 | 触发时机 |
|---|---|
| `wants_streaming() → bool` | runner 据此决定是否调用流式 LLM API |
| `before_iteration(ctx)` | 每轮 LLM 调用前 |
| `on_stream(ctx, delta)` | 流式 token 到达时（per-delta） |
| `on_stream_end(ctx, resuming)` | 流式段结束；resuming=True 表示后续还有工具调用 |
| `before_execute_tools(ctx)` | 工具执行前（注入 session_key/kb_map，发 progress 提示） |
| `after_iteration(ctx)` | 工具执行并写回消息后 |
| `finalize_content(ctx, content)` | 最终文本后处理（如剥离 `<think>` 标签） |

主 Agent 使用内联的 `_LoopHook`（`agent/loop.py:335`）覆盖：`on_stream` 推送流式 delta 到 bus；`before_execute_tools` 设置工具路由上下文并发 progress 事件；`finalize_content` 调用 `strip_think` 剥离推理标签。子 Agent 使用 `_SubagentHook`（`agent/subagent.py:159`）：仅覆盖 `after_iteration` 轮询取消信号。

---

## Ch7 记忆与上下文压缩

NanoResearch 记忆系统由三个层次组成：MEMORY.md 静态长期事实、user_memory 向量库（对话知识与历史）、Redis session 消息列表。当 session 消息过长导致 prompt 超出 context window 时，consolidation/compaction 流程将历史消息提炼后归档，以维持 token 预算。

---

### 7.1 MemoryStore — 长期事实

**职责一句话**：以文件形式持久化稳定事实（MEMORY.md / HISTORY.md），并提供 LLM 驱动的压缩入口。

**关键类与锚点**

| 组件 | 文件:行 | 说明 |
|---|---|---|
| `MemoryStore` 类 | `agent/memory.py:184` | 管理单个 agent 或工作区的 MEMORY.md |
| `MemoryStore.consolidate` | `agent/memory.py:277` | 调 LLM 将消息块提炼为 history_entry + memory_update |
| `MemoryStore._fail_or_raw_archive` | `agent/memory.py:375` | 连续失败 3 次后降级为 raw_archive |
| `MemoryStore._raw_archive` | `agent/memory.py:384` | 降级路径：将消息原文写入 user_memory，不经 LLM 提炼 |
| `_MAX_FAILURES_BEFORE_RAW_ARCHIVE` | `agent/memory.py:187` | 常量 3：连续失败阈值 |
| `CONSOLIDATION_SUMMARY_CONFIDENCE` | `agent/memory.py:35` | 常量 **0.7**：写入 user_memory 的置信度阈值 |

**MEMORY.md 内容规范**（`_CONSOLIDATION_SYSTEM_PROMPT`，`agent/memory.py:44`）：
- `FACTS`：稳定事实（6 个月后仍成立），单行 grep 可搜索
- `USER_PROFILE`：用户背景，最多 3 句
- `FOCUS_AREAS`：长期关注领域，最多 5 条

**save_memory 工具模式**（`agent/memory.py:129`）：LLM 被强制调用 `save_memory({history_entry, memory_update})`。若 provider 不支持 `tool_choice`，自动降级为 `auto`（`agent/memory.py:311`）。

**存储路径**（`agent/memory.py:190`）：
- per-agent：`workspace/agents/{agent_id}/memory/MEMORY.md`
- 主 agent（无 agent_id）：`workspace/memory/MEMORY.md`

---

### 7.2 ConversationKnowledgeExtractor — 用户知识提取

**职责一句话**：从每轮压缩后的对话中抽取用户偏好（preference）、习惯（habit）、决策（decision）三类信息，写入 user_memory 向量库。

**关键类与锚点**

| 组件 | 文件:行 | 说明 |
|---|---|---|
| `ConversationKnowledgeExtractor` 类 | `agent/conversation_knowledge_extractor.py:83` | LLM 抽取 + knowledge_search 写入 |
| `extract_from_messages` | `agent/conversation_knowledge_extractor.py:102` | 入口：格式化消息 → 调 LLM → 过滤 → 写库 |
| `ExtractedUserInfo` | `agent/conversation_knowledge_extractor.py:75` | 数据类：content / type / confidence |

**触发时机**（`agent/memory.py:455`）：每次 `MemoryConsolidator.consolidate_messages` 成功后立即调用，非阻塞。confidence 由 LLM 自行估计（prompt 中示例值 0.9），`is_evergreen=True`（用户偏好常绿）。

**与 CONSOLIDATION_SUMMARY_CONFIDENCE 的关系**：`consolidate` 方法写入 user_memory 时固定使用常量 `CONSOLIDATION_SUMMARY_CONFIDENCE=0.7`（`agent/memory.py:35,358`），这是 consolidation_summary 和 raw_archive 的持久化门槛。`ConversationKnowledgeExtractor` 提取的条目置信度由 LLM 动态判定，写入时直接使用 LLM 返回的 confidence 字段。

**不提取内容**：一般性知识陈述、客观事实、技术概念解释、代码逻辑说明、Agent 的回复（只提取用户侧信息）。

---

### 7.3 MemoryConsolidator — 压缩策略与调度

**职责一句话**：拥有压缩策略、per-session asyncio.Lock、session 偏移更新，对 MemoryStore 的调用全部经由此类路由。

**关键类与锚点**

| 组件 | 文件:行 | 说明 |
|---|---|---|
| `MemoryConsolidator` 类 | `agent/memory.py:407` | 持有 provider/model/sessions/context_window_tokens |
| `maybe_consolidate_by_tokens` | `agent/memory.py:553` | T2 token 触发压缩（多轮循环） |
| `pick_consolidation_boundary` | `agent/memory.py:491` | 选取安全的用户轮边界 |
| `estimate_session_prompt_tokens` | `agent/memory.py:527` | 探针消息估算当前 prompt token 数 |
| `consolidate_messages` | `agent/memory.py:450` | 委托 MemoryStore.consolidate + 知识抽取 |
| `plan_startup_consolidation` | `agent/memory.py:649` | 模块级函数，计算 T1 startup 压缩范围 |
| `TOKEN_CONSOLIDATION_TARGET_RATIO` | `agent/memory.py:34` | 常量 **0.5**（env 可覆盖） |
| `CONSOLIDATION_TAIL_PROTECT` | `agent/memory.py:33` | 常量 **8**（env 可覆盖） |
| `_MAX_CONSOLIDATION_ROUNDS` | `agent/memory.py:410` | 常量 5：单次 T2 触发最多压缩轮数 |
| `_SAFETY_BUFFER` | `agent/memory.py:412` | 常量 1024 tokens：tokenizer 漂移保险 |

---

### 7.4 两类触发条件

#### T1 — Startup 触发（启动时检查上一会话积压）

**调用栈**：`_process_message` → `_check_pending_consolidation`（`agent/loop.py:535`）→ `plan_startup_consolidation`（`agent/memory.py:649`）

**判定伪代码**

```
# plan_startup_consolidation (agent/memory.py:649)
def plan_startup_consolidation(session, now_utc, idle_threshold,
                                min_turns, tail_protect, pick_boundary):

    # 空闲门（idle gate）：session 必须已静置足够久
    if now_utc - session.updated_at < idle_threshold:   # 默认 1800 s
        return None

    # 轮数门：pending 消息中用户轮数必须 >= min_turns（默认 2）
    pending = session.messages[session.last_consolidated:]
    pending_turns = count(m for m in pending if m.role == "user")
    if pending_turns < min_turns:
        return None

    # 边界选择：找一个 tail_protect（默认 8）条之前的用户轮边界
    boundary = pick_boundary(session, tokens_to_remove=1, tail_protect=tail_protect)
    if boundary is None or boundary.end_idx <= start:
        return None

    return (start, boundary.end_idx)   # 传给 consolidate_messages
```

**关键常量**（`agent/loop.py:23-24`，`agent/memory.py:33`）：
- `STARTUP_CONSOLIDATION_IDLE_SECONDS = 1800`（env: `STARTUP_CONSOLIDATION_IDLE_SECONDS`）
- `STARTUP_MIN_PENDING_TURNS = 2`（env: `STARTUP_MIN_PENDING_TURNS`）
- `CONSOLIDATION_TAIL_PROTECT = 8`（env: `CONSOLIDATION_TAIL_PROTECT`）

**防重复**：`AgentLoop._startup_consolidated: set[str]`（`agent/loop.py:157`）记录已 T1 压缩的 session_key，同次进程内每个 session 只触发一次。

#### T2 — Token 触发（超出 context window 时压缩）

**调用位置**：`_process_message` 在 context 组装前（`agent/loop.py:777`）和 runner 完成后（`agent/loop.py:839`）各调用一次 `maybe_consolidate_by_tokens`。

**判定与多轮压缩伪代码**

```
# MemoryConsolidator.maybe_consolidate_by_tokens (agent/memory.py:553)
async def maybe_consolidate_by_tokens(session):
    budget = context_window_tokens - max_completion_tokens - SAFETY_BUFFER(1024)
    target = int(budget * TOKEN_CONSOLIDATION_TARGET_RATIO)   # 默认 0.5 × budget

    estimated = estimate_session_prompt_tokens(session)
    if estimated < budget:
        # anti-shake: 若上次与本次 token 差 < 10%，跳过（仅 log）
        return

    for round in range(MAX_CONSOLIDATION_ROUNDS=5):
        if estimated <= target:
            return

        boundary = pick_consolidation_boundary(
            session,
            tokens_to_remove = estimated - target,
            tail_protect = CONSOLIDATION_TAIL_PROTECT(8)
        )
        if boundary is None:
            return   # 无安全边界可选

        chunk = session.messages[last_consolidated : boundary.end_idx]
        await consolidate_messages(chunk)

        # Lua LTRIM：Redis 层原子前移消息列表起点
        redis.eval(LUA_LTRIM, ..., keep_from_idx, timestamp)

        session.last_consolidated = boundary.end_idx
        sessions.save(session)

        estimated = estimate_session_prompt_tokens(session)
```

---

### 7.5 Boundary 选择：pick_consolidation_boundary

```
# agent/memory.py:491
def pick_consolidation_boundary(session, tokens_to_remove, tail_protect=8):
    start = session.last_consolidated           # head protect：已压缩位置之后
    max_end = len(session.messages) - tail_protect  # tail protect：保留尾部 8 条

    if start >= max_end or tokens_to_remove <= 0:
        return None

    removed_tokens = 0
    last_boundary = None
    for idx in range(start, max_end):
        if idx > start and messages[idx].role == "user":
            last_boundary = (idx, removed_tokens)
            if removed_tokens >= tokens_to_remove:
                return last_boundary   # 找到足够边界就提前退出
        removed_tokens += estimate_message_tokens(messages[idx])

    return last_boundary   # 返回范围内最后一个用户轮边界（可能 token 不足）
```

**三重保护**：

| 保护机制 | 实现 | 效果 |
|---|---|---|
| Head protect | `start = session.last_consolidated` | 已压缩内容不重复压缩；首轮交互与系统提示不受影响 |
| Tail protect | `max_end = len - CONSOLIDATION_TAIL_PROTECT(8)` | 最近 8 条消息（跨角色）永远留在 context，保留共指锚点 |
| 用户轮边界 | 只在 `role == "user"` 处切割 | 保证压缩块始终在语义上完整（不在 assistant/tool 消息中间截断） |

---

### 7.6 Redis 原子裁剪

压缩成功后，`maybe_consolidate_by_tokens` 执行 Lua 脚本 `_LUA_LTRIM`（`agent/memory.py:19`）原子地将 Redis 消息列表的起点前移到新 boundary，同时更新 `session:meta` 的 `updated_at`。若 Lua 失败（Redis 不可用），降级为 `sessions.save()` 全量覆写——非致命，仅下次启动会重新读完整消息列表再裁剪。

---

### 7.7 设计取舍与已知坑

**token 估算精度**：`estimate_session_prompt_tokens` 通过发送探针消息（`"[token-probe]"`）调 `estimate_prompt_tokens_chain` 得到近似值；对不返回 token 数的 provider（如本地模型）可能退化为字符估算，因此留了 `_SAFETY_BUFFER=1024` 的余量。

**anti-shake**：`_last_session_tokens` 记录上次检查时 token 数，若本次变化 < 10% 跳过压缩（`agent/memory.py:576`），防止在 token 边界附近反复触发小批量压缩。

**history.py 已废弃**：`SearchHistoryTool` 文件保留作向后兼容，但不注册也不使用；用户历史现由 `build_history_context` 的 RAG 路径（`agent/context.py:44`）自动完成，无需手动工具调用。

**子 Agent 无记忆**：`_run_subagent` 不持有 session，不触发任何压缩；子 Agent 完成后结果注入主 session，由主 session 的 T2 机制统一管理。

**置信度常量锚定（CONSOLIDATION_SUMMARY_CONFIDENCE=0.7）**：consolidation_summary 和 raw_archive 两条写入路径都使用同一常量（`agent/memory.py:35,358,399`），确保即使 LLM 提炼失败降级为 raw_archive 时，条目仍以 0.7 置信度持久化到 user_memory，不会因置信度过低被召回过滤器排除。


## Ch8 LLM Providers

NanoResearch 的 LLM Provider 层将所有外部模型 API 统一抽象为一个接口，通过 **ProviderSpec 注册表**描述 20+ 个 provider 的元数据，由 **ModelFactory** 按调用角色（role）显式解析凭据，最终由各 provider 实现类执行实际 API 调用。

---

### 8.1 LLMProvider 抽象契约

**职责**：定义所有 provider 必须遵守的同步/流式接口，并提供通用的重试、消息净化、错误处理基础设施。

核心数据类型（`backend/nanoresearch/providers/base.py`）：

| 类型 | 行号 | 说明 |
|---|---|---|
| `ToolCallRequest` | :14 | 工具调用请求；持有 `id/name/arguments` 及 provider 扩展字段 |
| `LLMResponse` | :43 | 统一响应；含 `content/tool_calls/finish_reason/usage/reasoning_content/thinking_blocks` |
| `GenerationSettings` | :58 | 冻结默认参数（`temperature=0.7, max_tokens=4096, reasoning_effort=None`）；存于 provider 实例，调用方可逐字段覆盖 |

**抽象方法**（子类必须实现）：

```python
# base.py:168
async def chat(
    self,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    reasoning_effort: str | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> LLMResponse: ...

# base.py:366
def get_default_model(self) -> str: ...
```

**可选覆盖**（有默认实现）：

- `chat_stream()` (`base.py:230`)：默认退化为 `chat()` 并将完整内容当单一 delta 传出；原生支持流式的 provider 应覆盖此方法。
- `_sanitize_empty_content()` (`base.py:104`)：清除空 content block、剥离内部 `_meta` 字段。
- `_sanitize_request_messages()` (`base.py:153`)：只保留 provider 允许的消息 key。

**公开重试包装器**：

- `chat_with_retry()` (`base.py:314`)：最多 4 次（初次 + 3 次重试），延迟序列 `(1, 2, 4)` 秒（`_CHAT_RETRY_DELAYS`，`base.py:81`）。判断是否为可重试的瞬态错误依赖 `_TRANSIENT_ERROR_MARKERS`（`base.py:82`）：含 `429/5xx/overloaded/timeout/connection` 等关键词。非瞬态错误若包含图片内容，则触发 `_strip_image_content()` (`base.py:200`) 去图重试一次。
- `chat_stream_with_retry()` (`base.py:266`)：与上同逻辑，但包装 `chat_stream()`。
- 两者均从 `self.generation` 读取默认参数，未显式传入的参数自动继承，无需调用方逐层透传。

---

### 8.2 ProviderSpec 注册表

**职责**：`PROVIDERS` 元组是 provider 元数据的单一可信来源，驱动环境变量解析、状态展示、provider 实例化等所有下游逻辑。

核心类：`ProviderSpec`（`backend/nanoresearch/providers/registry.py:21`，`frozen=True` dataclass）。关键字段：

| 字段 | 说明 |
|---|---|
| `name` | config 字段名，也是 `find_by_name()` 的查找键 |
| `backend` | 实现类标识：`"openai_compat"` / `"anthropic"` / `"azure_openai"` / `"openai_codex"` |
| `keywords` | 模型名关键词（小写），用于按模型名匹配 provider |
| `env_key` | API Key 环境变量名 |
| `is_gateway` | 网关型（可路由任意模型），如 OpenRouter、AiHubMix |
| `is_local` | 本地部署（vLLM、Ollama） |
| `is_oauth` | OAuth 认证（无 API Key），如 OpenAI Codex、GitHub Copilot |
| `detect_by_key_prefix` | 按 api_key 前缀匹配（如 `"sk-or-"` → OpenRouter） |
| `detect_by_base_keyword` | 按 api_base URL 子串匹配 |
| `default_api_base` | 该 provider 的默认 base URL |
| `strip_model_prefix` | 发请求前剥离 `"provider/"` 前缀（AiHubMix、VolcEngine Coding Plan 等） |
| `supports_prompt_caching` | 是否注入 `cache_control` 块（Anthropic、OpenRouter、DashScope） |
| `model_overrides` | 特定模型的参数强制覆盖（如 Moonshot kimi-k2.5 强制 `temperature=1.0`） |

`PROVIDERS` 元组（`registry.py:75`）按优先级排列，当前收录共 **22 个** ProviderSpec，顺序为：

```
custom → azure_openai → openrouter → aihubmix → siliconflow → volcengine →
volcengine_coding_plan → byteplus → byteplus_coding_plan →
anthropic → openai → openai_codex → github_copilot →
deepseek → gemini → zhipu → dashscope → moonshot → minimax → mistral →
stepfun → vllm → ollama → ovms → groq
```

网关型 provider（`is_gateway=True`）排在前面，保证网关在 fallback 路径中优先命中。

查找辅助函数：`find_by_name(name: str) -> ProviderSpec | None`（`registry.py:351`），入参规范化为 snake_case 后做线性扫描。

**新增 provider 路径**（registry.py 文件头注释）：向 `PROVIDERS` 添加一个 `ProviderSpec`，再在 `config/schema.py` 的 `ProvidersConfig` 加一个字段，即可自动接入环境变量解析与状态展示，无需其它改动。

---

### 8.3 ModelFactory：按角色显式解析

**职责**：纯解析层，读取三个配置源（`config.json`/`settings.yaml`/UserSettings DB 行），为每个调用角色返回一个 `ModelSpec`，不实例化 LLM 对象。

**模型角色**（`ModelRole` 枚举，`model_factory.py:35`）：

| 角色 | 值 | 默认 fallback 模型 |
|---|---|---|
| `CHAT` | `"chat"` | `gpt-4o` |
| `INGESTION_LLM` | `"ingestion_llm"` | settings.yaml → config defaults → `gpt-4o` |
| `EMBEDDING` | `"embedding"` | settings.yaml embedding 配置 |
| `VISION` | `"vision"` | settings.yaml vision_llm 配置 |
| `EVAL_GENERATOR` | `"eval_generator"` | `qwen-plus` |
| `EVAL_EVALUATOR` | `"eval_evaluator"` | `qwen-max` |

**核心入口**：

```python
# model_factory.py:116
@classmethod
def resolve(
    cls,
    role: ModelRole,
    *,
    config: Config | None = None,
    rag_settings: Settings | None = None,
    user_model: str | None = None,
    user_providers: list[dict] | None = None,
    user_roles: dict | None = None,
    mode: Literal["server", "local"] | None = None,
    **overrides: Any,
) -> ModelSpec: ...
```

`require_key()` (`model_factory.py:73`) 是 `resolve()` 的封装，若 `ModelSpec.api_key` 为空则抛出 `ModelResolutionError`，并在异常 `sources_checked` 字段中列出已检查的配置源。

#### 解析优先级（local 模式）

```
model_override kwarg
  → user_roles[role].provider_id (精确 provider id 绑定, :161)
    → _match_user_provider_by_model() (user_providers 中 models[] 列表匹配, :499)
      → 第一个有 api_key 的 user_provider (fallback)
        → config.json 对应 provider 配置
          → rag_settings (settings.yaml)
            → 硬编码 default_model
```

**server 模式**（`mode == "server"`，`model_factory.py:177`）：完全只读 `user_providers`，若匹配失败直接 raise `ModelResolutionError`，不回退到 `config.json` 或 `rag_settings`。这是多租户隔离的关键：server 模式下 config.json 中的系统级密钥对用户不可见。

#### 多租户 user-provided provider 匹配逻辑

`user_providers` 是来自 `user_settings.extra["providers"]` 的 list of dict，每个 dict 结构为：
```json
{"id": "...", "name": "...", "api_key": "...", "api_base": "...", "models": ["model-a", "model-b"]}
```

三种匹配方式（均在 `model_factory.py`）：

- `_match_user_provider_by_id(provider_id, user_providers)` (`:492`)：按 `id` 字段精确匹配，供 `user_roles` 显式绑定使用。
- `_match_user_provider_by_model(model, user_providers)` (`:499`)：先查 `models[]` 列表精确包含，未命中则 fallback 到第一个有 `api_key` 的 provider。
- `_match_user_provider_by_name(provider_name, user_providers)` (`:512`)：按 `name` 字段不区分大小写匹配，供 EMBEDDING 角色按 provider 名查找使用。

同时，`user_roles` dict（来自 `user_settings.extra["roles"]`）允许用户将某个角色显式绑定到指定 provider id，优先级高于所有其他路径（`:159-175`）。

#### patch_settings

`ModelFactory.patch_settings(rag_settings, role, spec)` (`model_factory.py:416`) 将解析结果反写回 `Settings` 副本，使下游的 `LLMFactory.create(settings)` 和 `EmbeddingFactory.create(settings)` 无需感知 ModelFactory 的存在。

---

### 8.4 各 Provider 适配差异

| Provider 类 | 文件 | backend 值 | 核心差异 |
|---|---|---|---|
| `AnthropicProvider` | `providers/anthropic_provider.py:23` | `"anthropic"` | 原生 Anthropic SDK；需将 OpenAI 消息格式转换为 Messages API；支持 prompt caching（`cache_control` 块）与 extended thinking（`budget_tokens`） |
| `OpenAICompatProvider` | `providers/openai_compat_provider.py:104` | `"openai_compat"` | 使用 `AsyncOpenAI` 客户端；接受 `ProviderSpec` 实例控制行为；覆盖所有使用 `base_url+api_key` 的 OpenAI 兼容端点 |
| `AzureOpenAIProvider` | `providers/azure_openai_provider.py:19` | `"azure_openai"` | 直接 `httpx` 调用（不用 SDK）；API 版本硬编码 `2024-10-21`；model 字段用作 Azure deployment name；使用 `api-key` 请求头（非 Bearer）；`max_completion_tokens` 替代 `max_tokens` |
| `OpenAICodexProvider` | `providers/openai_codex_provider.py:21` | `"openai_codex"` | OAuth 认证（`oauth_cli_kit.get_token`），无 API Key；调用 Codex Responses API（非 chat completions 端点）；消息格式完全不同，需 `_convert_messages()` 转换 |
| `GroqTranscriptionProvider` | `providers/transcription.py:10` | N/A | 不继承 `LLMProvider`；专用于语音转写（Groq Whisper API，模型固定为 `whisper-large-v3`）；只有 `transcribe(file_path)` 方法，不参与 chat 路径 |

#### AnthropicProvider 关键细节

消息格式转换（`anthropic_provider.py:62`）：`_convert_messages()` 将 OpenAI 格式的 `messages` 拆分出 `system` 字符串/块列表，并将 `tool` role 消息合并为 Anthropic 的 `tool_result` 块；`_merge_consecutive()` (`:189`) 处理 Anthropic 要求的 user/assistant 严格交替约束。

Extended Thinking（`:335`）：`reasoning_effort` 映射为 budget_tokens（low=1024, medium=4096, high=max(8192, max_tokens)），同时强制 `temperature=1.0`，tool_choice 降级为 `{"type": "auto"}`。

Prompt caching（`:254`）：`_apply_cache_control()` 在 system 块、倒数第二条消息、最后一个 tool definition 上分别打 `{"type": "ephemeral"}` 标记；若 ContextBuilder 已预置 `cache_control`，则直接透传。响应 usage 中额外回写 `cache_creation_input_tokens` / `cache_read_input_tokens`（`:390`）。

#### OpenAICompatProvider 关键细节

`ProviderSpec` 注入（`:117`）驱动以下行为：
- `spec.supports_prompt_caching` → 注入 `cache_control` 标记（`:139`）
- `spec.strip_model_prefix` → 发送前剥离 `"provider/"` 前缀（`:223`）
- `spec.model_overrides` → 匹配模型名后覆盖参数（`:238`），如 Moonshot kimi-k2.5 强制 temperature=1.0
- `spec.supports_max_completion_tokens` → 改用 `max_completion_tokens` 参数（`:232`）

推理内容提取（`:573`）：`_extract_reasoning_text()` 按顺序检查 `reasoning_content`（Kimi/DeepSeek-R1）、`thinking`（DashScope qwen3-thinking）、`reasoning`（部分硅基流动网关）三个字段名。

工具调用 ID 规范化（`:170`）：各 provider 生成的 tool_call_id 格式各异，`_normalize_tool_call_id()` 统一 SHA1 截取为 9 位字母数字（Mistral 等对 ID 长度敏感）。

OpenRouter 归因头（`:126-128`）：检测到 spec 为 openrouter 或 api_base 含 `"openrouter"` 时自动附加 `HTTP-Referer` / `X-OpenRouter-Title` 等请求头。

#### AzureOpenAIProvider 关键细节

URL 构造（`:52`）：`{api_base}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-10-21`，其中 `model` 参数直接作为 `deployment_name`。

温度支持检测（`:76`）：`_supports_temperature()` 对 `gpt-5`/`o1`/`o3`/`o4` 系列跳过 temperature 参数，避免 API 错误。SSE 流式解析自行实现（`_consume_stream()`，`:247`），不依赖 SDK。

#### OpenAICodexProvider 关键细节

认证流程（`:40`）：`asyncio.to_thread(get_codex_token)` 在线程池中同步获取 OAuth token，返回含 `account_id` 和 `access` 的对象，写入 `Authorization: Bearer` 和 `chatgpt-account-id` 请求头（`:106`）。

消息格式（`:151`）：Codex Responses API 使用完全不同的 `input_items` 格式，assistant tool call 转为 `function_call` 条目，tool result 转为 `function_call_output`，tool call ID 通过 `|` 分隔编码 `call_id` 和 `item_id`（`:213`）。

SSL 容错（`:67`）：若首次请求触发 `CERTIFICATE_VERIFY_FAILED`，自动以 `verify=False` 重试并打 warning 日志。

---

### 8.5 「支持 20+ Provider」的实现机制

支持数量来自两个正交维度：

**1. ProviderSpec 注册表驱动**：每个 provider 只需在 `PROVIDERS` 元组中添加一个 `ProviderSpec`，通过 `default_api_base` 字段指定端点 URL。无需新增 provider 实现类，所有 OpenAI 兼容端点共用 `OpenAICompatProvider`。

**2. `base_url + api_key` 模式**（`openai_compat` backend）：`AsyncOpenAI(api_key=..., base_url=...)` 只需换 URL 和 Key 即可对接任意 OpenAI 兼容 API。当前注册表中以此模式覆盖的 provider 包括（但不限于）：

| Provider | 端点 | registry.py 行 |
|---|---|---|
| DeepSeek | `api.deepseek.com` | :228 |
| Gemini | `generativelanguage.googleapis.com/v1beta/openai/` | :238 |
| 智谱 AI | `open.bigmodel.cn/api/paas/v4` | :246 |
| DashScope（通义/Qwen） | `dashscope.aliyuncs.com/compatible-mode/v1` | :256 |
| Moonshot（Kimi） | `api.moonshot.ai/v1` | :267 |
| MiniMax | `api.minimax.io/v1` | :274 |
| Mistral | `api.mistral.ai/v1` | :282 |
| Step Fun | `api.stepfun.com/v1` | :290 |
| OpenRouter（网关） | `openrouter.ai/api/v1` | :99 |
| AiHubMix（网关） | `aihubmix.com/v1` | :114 |
| SiliconFlow（网关） | `api.siliconflow.cn/v1` | :126 |
| VolcEngine（网关） | `ark.cn-beijing.volces.com/api/v3` | :136 |
| Ollama（本地） | `localhost:11434/v1` | :314 |
| vLLM（本地） | 用户自定义 | :302 |

网关型 provider（OpenRouter、AiHubMix 等）可进一步透传任意 provider 的模型，因此实际可达的模型数量远超注册表数量。

`custom` provider（`registry.py:77`，`is_direct=True`）作为兜底：用户可直接提供任意 `api_base` 和 `api_key`，无需注册表中有对应条目。

---

### 8.6 Provider 实例化数据流

```
UserSettings.extra["providers"]
        │
        ▼
ModelFactory.resolve(role, ...)          ← model_factory.py:116
        │  按角色选配置源，返回 ModelSpec
        │  (model, api_key, base_url, provider)
        ▼
_build_provider_from_spec(spec, fallback)  ← worker.py:40
        │  find_by_name(spec.provider) → ProviderSpec
        │  读取 p_spec.backend
        ├─ "anthropic"     → AnthropicProvider(api_key, api_base, ...)
        ├─ "azure_openai"  → AzureOpenAIProvider(api_key, api_base, ...)
        └─ (default)       → OpenAICompatProvider(api_key, api_base, spec=p_spec, ...)
                                      │
                                      ▼
                              LLMProvider.chat_with_retry()   ← base.py:314
                                  (重试、净化、错误处理)
```

注：`OpenAICodexProvider` 通过独立路径实例化（不经过 `_build_provider_from_spec`），因其无 API Key 无需 `ModelFactory` 解析凭据。

---

### 8.7 设计取舍

**role 显式解析 vs 隐式默认**：ModelFactory 为每个 role 维护独立的解析器（`_resolve_chat`、`_resolve_ingestion_llm` 等），而非统一的"用哪个都行"逻辑。代价是新增 role 需要新增分支；收益是每个 role 可以精确控制模型来源优先级和 fallback 链，避免评估用模型和对话用模型意外共用同一配置。

**server 模式完全隔离**：server 模式下 `config.json` 对用户不可见，强制所有凭据来自用户自带的 `user_providers`。这简化了多租户安全边界，但要求用户在调用任何 role 前都先配置好 provider，否则立即报错而非静默 fallback。

**`ProviderSpec` 驱动而非继承**：新增 OpenAI 兼容 provider 无需写新类，只需加一行 `ProviderSpec`；特殊行为（缓存、前缀剥离、参数覆盖）通过 spec 字段注入到 `OpenAICompatProvider`。代价是 spec 字段集随时间膨胀，每个新需求都可能新增字段。

**`GroqTranscriptionProvider` 独立于 LLMProvider 继承体系**：语音转写与 chat 接口语义不同（单文件输入 → 文本），刻意不套用 LLMProvider 接口，避免强行适配导致接口污染。


## Ch9 RAG 子系统

> 全项目最大、最核心的子系统，120+ 文件。本章以当前代码（`backend/nanoresearch/rag/`）为准，自顶向下拆为五大分区，逐分区给职责一句话 → 关键组件（类/函数 + 锚点）→ 数据流/时序 → 关键算法 → 设计取舍/坑。所有锚点形如 `backend/nanoresearch/<path>:<line>`，行号真实。

### 9.1 总体架构

**一句话**：RAG 子系统是一个「config-driven、可优雅降级、全程可观测（TraceContext）」的混合检索引擎，对内分五层装配，对外以独立的 MCP stdio 子进程暴露少量工具。

五大分区职责：

| 分区 | 路径 | 职责 |
|---|---|---|
| `core/` | `rag/core/` | 查询引擎（`query_engine/`）、响应组装（`response/`）、会话/追踪（`session/`、`trace/`）、类型与配置（`types.py`、`settings.py`）——纯算法编排层，全部依赖注入 |
| `ingestion/` | `rag/ingestion/` | 摄入管道：parse → chunk → transform → embed → store + 知识图谱持久化 |
| `mcp_server/` | `rag/mcp_server/` | MCP 协议外壳：JSON-RPC 握手、工具注册、异步摄入任务 |
| `internal_loop/` | `rag/internal_loop/` | 复杂查询的多轮内部检索循环（plan→search→fuse→verify→finalize） |
| `libs/` | `rag/libs/` | 可插拔后端：`loader/`（PDF/MD 解析）、`embedding/`、`vector_store/`（Chroma）、`reranker/`、`splitter/`、`llm/`——全部走 Factory |
| `observability/` | `rag/observability/` | 日志（stderr-only，避免污染 stdio 协议流） |

关键编排类：`IngestionPipeline`（`backend/nanoresearch/rag/ingestion/pipeline.py:156`）、`HybridSearch`（`backend/nanoresearch/rag/core/query_engine/hybrid_search.py:106`）、`RAGLoopRunner`（`backend/nanoresearch/rag/internal_loop/runner.py:91`）、MCP `create_mcp_server`（`backend/nanoresearch/rag/mcp_server/protocol_handler.py:229`）。

RAG 内部架构（ASCII）：

```
                       ┌──────────────────── MCP stdio 子进程（外壳）────────────────────┐
                       │  server.py → protocol_handler.py(JSON-RPC 2.0)                  │
   外层 Agent ─stdio─► │  暴露工具: kb_search / list_collections / list_documents        │
                       │            ingest_document / delete_document / get_task_status   │
                       └───┬───────────────────────────────────────────────┬────────────┘
                           │ kb_search                                      │ ingest_document
              ┌────────────▼─────────────┐                     ┌────────────▼──────────────┐
   simple ───►│  ConcurrentRetrievalEngine│◄── complex ──┐     │  AsyncTaskManager(线程池)  │
              │  batch_retrieval + round  │              │     └────────────┬──────────────┘
              └────────────┬─────────────┘   ┌───────────┴─────────┐        │
                           │                 │  internal_loop/      │        │
                           │                 │  runner(plan→search  │        │ unified.ingest_document
                           │                 │   →fuse→verify→final)│        │
                           ▼                 └───────────┬─────────┘        ▼
   ┌──────────────── core/query_engine ──────────────────┐        ┌──── ingestion/pipeline ────┐
   │ QueryProcessor → ┌ DenseRetriever ┐                 │        │ load→chunk→transform→embed  │
   │  (jieba 分词)     │ SparseRetriever├→ RRFFusion →    │        │      →store  + graph/persist │
   │                  └────────────────┘   CoreReranker  │        └──────────┬───────────┬──────┘
   └──────────┬─────────────────────────┬───────────────┘                   │           │
              │ 向量相似度               │ BM25 倒排                          ▼           ▼
        ┌─────▼──────┐            ┌──────▼───────┐                    ┌──────────┐ ┌──────────┐
        │ ChromaDB    │           │ BM25 JSON 索引│◄───── 写入 ──────│VectorUps.│ │BM25Index.│
        │(向量库,per- │           │ ~/.nanoresear │                    └──────────┘ └──────────┘
        │ uid 隔离)   │           │ ch/rag/bm25/  │   + ImageStorage / KG tables(PG)
        └─────────────┘           └──────────────┘
```

设计取舍：`core` 与 `libs` 严格分层（core 只持接口、libs 提供实现，经 `*_Factory` 创建），换嵌入/向量库/重排后端不动编排代码；MCP server 作为独立 stdio 子进程而非 in-process 调用——隔离重依赖（chromadb/onnxruntime）的导入与崩溃，代价是主进程的内存态（如 SessionManager）在子进程不可见（见 9.6 的坑）。

---

### 9.2 摄入管道

**一句话**：`IngestionPipeline.run()` 是一条 7 阶段、SHA256 幂等、LLM 失败不阻断的线性管道；`unified.ingest_document()` 在其外再包一层「去重 + PG 记录 + 失败回滚 + chunk 落库」事务语义。

阶段顺序（严格按 `backend/nanoresearch/rag/ingestion/pipeline.py:321` 的 `run()`，括号内为真实锚点）：

```
Stage1 完整性    pipeline.py:357   SQLiteIntegrityChecker.should_skip(sha256)  → 命中则直接返回
Stage2 解析      pipeline.py:378   _get_loader(suffix).load()  (Marker/MinerU/MarkItDown/Markdown)
Stage2.5 领域    pipeline.py:412   _infer_domain() + DocumentMetadataExtractor.extract() (title/authors/abstract)
Stage3 分块      pipeline.py:432   DocumentChunker.split_document()
Stage4 变换      pipeline.py:482   4a ChunkRefiner.transform()       (pipeline.py:490)
                                   4b MetadataEnricher.transform()   (pipeline.py:497)
                                   4c ImageCaptioner.transform()     (pipeline.py:504)
Stage5 编码      pipeline.py:541   BatchProcessor.process() → (dense_vectors, sparse_stats)
Stage6 存储      pipeline.py:593   6a VectorUpserter.upsert()        (pipeline.py:599)
                                   6b BM25Indexer.add_documents()    (pipeline.py:609)
                                   6c ImageStorage.register_image()  (pipeline.py:619)
                          → integrity_checker.mark_success()         (pipeline.py:689)
```

关键组件与算法：

- **分块** `DocumentChunker.split_document`（`backend/nanoresearch/rag/ingestion/chunking/document_chunker.py:194`）。策略三选一（`document_chunker.py:217-231`）：`semantic`（`SemanticChunker.split_text`，`chunking/semantic_chunker.py:195`）、`structured`（按 Markdown 标题切，抽取 `section_level`/`title`，并链接 `prev/next_chunk_id`，`document_chunker.py:238-262`）、默认 `fixed`（`StructuredChunker`/递归 splitter）。KB 级 `chunk_strategy_override` 优先于 `auto_detect`。chunk_id 确定性生成 `{doc_id}_{index:04d}_{hash8}`（`document_chunker.py:289`）。
- **变换**（均继承 `BaseTransform`，`transform/base_transform.py:24`）：`ChunkRefiner`（`transform/chunk_refiner.py:78`，rule+LLM 双轨，`refined_by` 标记）、`MetadataEnricher`（`transform/metadata_enricher.py:88`，生成 title/tags/summary）、`ImageCaptioner`（`transform/image_captioner.py:215`，Vision LLM）。注意：实体抽取 `EntityExtractor`（`transform/entity_extractor.py:71`）**不在主管道内**，由 `graph/build` 旁路调用（见下）。
- **编码**：`DenseEncoder.encode`（`embedding/dense_encoder.py:66`，走 `EmbeddingFactory` 嵌入后端）+ `SparseEncoder.encode`（`embedding/sparse_encoder.py:72`，`jieba.lcut` 分词 `sparse_encoder.py:150` 后小写化 `:164`，产出 term_frequencies/doc_length），二者由 `BatchProcessor.process`（`embedding/batch_processor.py:103`）批量合流。
- **存储**：`VectorUpserter.upsert`（`storage/vector_upserter.py:73`）写 ChromaDB 并回吐 `ChunkPayload`（`core/types.py:225`，含 `chroma_id`、`token_count=len(text)//4`、`char_start/end`、metadata），供上层落 PG；`BM25Indexer.add_documents`（`storage/bm25_indexer.py:324`）增量并入磁盘 JSON 倒排（`~/.nanoresearch/rag/bm25/{collection}/`）；关键对齐：管道把 `sparse_stats[i]["chunk_id"]` 改写为 Chroma 返回的 `vector_ids[i]`（`pipeline.py:604`），使 BM25 命中能回 Chroma 取正文。
- **知识图谱旁路**：`persist_chunk_entities`（`ingestion/graph/persist.py:19`）从 chunk metadata 的 `_kg_entities/_kg_relations` 抽取并写 PG 的 KG 表，"errors logged but never raised"——KG 失败绝不阻断摄入。

**unified 外层事务**（`backend/nanoresearch/rag/ingestion/unified.py:48`）：单一摄入入口（Web worker / Agent MCP / CLI 三通道共用）。流程：路径校验（拒临时目录，`unified.py:259`）→ 按 `content_hash` 去重（`unified.py:116`，已 indexed 且非 force 直接 skip）→ PG 建/复用 document 记录（区分 `uploaded/parsing` 预创建态与 `processing` 在跑态，`unified.py:137`）→ 线程池跑 `pipeline.run`（`unified.py:198`）→ 失败时 `delete_by_metadata({source_path})` 回滚 Chroma 并标 `failed`（`unified.py:201-216`）→ 成功则把 `chunk_payloads` 批量写 `KbChunk` 并 `update_document_status("indexed")`（`unified.py:218-244`）。

设计取舍/坑：管道层 collection 名由调用方（`unified`）从 `kb.chroma_collection` 注入，管道自身不感知租户；幂等有两层——管道内 SQLite SHA256 跳过 + unified 层 PG content_hash 去重，二者口径不同（前者按文件 hash，后者按 KB 内 content_hash），reprocess 走 `force=True` 时管道层会被强制重跑但 Chroma 旧向量靠 `delete_by_metadata(source_path)` 清理。

---

### 9.3 查询引擎

**一句话**：`HybridSearch.search()` 并行跑 Dense（向量）+ Sparse（BM25/jieba）两路召回，用 RRF 做无量纲秩融合，再过结构扩展与可选 Cross-Encoder 重排，任一路失败自动降级到另一路。

组件与锚点：

- 入口 `HybridSearch.search`（`backend/nanoresearch/rag/core/query_engine/hybrid_search.py:209`）；并行召回 `_run_parallel_retrievals` 用 `ThreadPoolExecutor(max_workers=2)`（`hybrid_search.py:460`，单路 30s 超时 `:483`）。
- `QueryProcessor.process`（`core/query_engine/query_processor.py:96`）：`jieba.lcut` 分词（`query_processor.py:230`，与索引侧 `SparseEncoder` 同分词器保证 BM25 可匹配）+ 中英停用词过滤 + `key:value` 过滤语法解析。
- `DenseRetriever.retrieve`（`core/query_engine/dense_retriever.py:100`）：embed query → `vector_store.query()`；支持 `precomputed_query_embedding` 跳过嵌入 API。
- `SparseRetriever.retrieve`（`core/query_engine/sparse_retriever.py:103`）：`bm25_indexer.query(keywords)` 拿 `{chunk_id,score}` → 再 `vector_store.get_by_ids()` 补正文/metadata（`sparse_retriever.py:172`）；每次查询都重载磁盘索引（`_ensure_index_loaded`，`sparse_retriever.py:222`，因别的进程可能更新过）。
- BM25 评分 `BM25Indexer._calculate_bm25_score`（`ingestion/storage/bm25_indexer.py:459`），`k1=1.5`（`bm25_indexer.py:80/99`）、`b` 可配，公式 `score = IDF·tf·(k1+1) / (tf + k1·(1-b+b·doc_len/avg_len))`。
- 融合 `RRFFusion.fuse`（`core/query_engine/fusion.py:88`），`DEFAULT_K=60`（`fusion.py:64`）。
- 重排 `CoreReranker.rerank`（`core/query_engine/reranker.py:235`，经 `RerankerFactory` 选 LLM/CrossEncoder/None，失败回退原序 `reranker.py:328`）；另有 `StructureAwareReranker`（`reranker.py:388`）按 section_level/content_type 加权。
- 结构扩展 `_expand_with_structure`（`hybrid_search.py:774`，补 `prev/next_chunk_id` 邻居，邻居打 `is_neighbor` 标记并排到末位）。
- 异步入口 `async_search`（`hybrid_search.py:892`）：先查 Redis embedding 缓存（命中跳嵌入 API，`hybrid_search.py:919`），再可选 KG 跨文档扩展 `_expand_with_graph`（`hybrid_search.py:975`，邻居固定分 `graph_expansion_score=0.1` 始终垫底）。

默认参数：`HybridSearchConfig` dense_top_k=20 / sparse_top_k=20 / fusion_top_k=10（`hybrid_search.py:73-75`），实际从 `RetrievalSettings`（`core/settings.py:137`，含 `dense_top_k/sparse_top_k/fusion_top_k/rrf_k` 必填字段）覆盖。

**RRF 融合伪代码**（对照 `fusion.py:145-189`）：

```
RRF(ranking_lists, k=60, top_k):
    weights = {"research": 1.0, "conversation": 0.4}   # fusion.py:68 源权重
    rrf_scores = {}            # chunk_id -> 累计分
    chunk_data = {}            # chunk_id -> 首次出现的 RetrievalResult(保留 text/metadata)
    for ranking_list in ranking_lists:          # 通常 [dense_results, sparse_results]
        for rank, result in enumerate(ranking_list, start=1):
            contrib = 1.0 / (k + rank)                          # 核心：只用秩，不用原始分
            contrib *= weights.get(result.metadata["source"], 1.0)
            if result.chunk_id not in rrf_scores:
                rrf_scores[result.chunk_id] = 0.0
                chunk_data[result.chunk_id] = result          # 首次占位
            rrf_scores[result.chunk_id] += contrib            # 多路命中累加
    fused = [RetrievalResult(id, score=s, text/meta from chunk_data[id])
             for id, s in rrf_scores.items()]
    fused.sort(key=lambda r: (-r.score, r.chunk_id))          # 同分按 chunk_id 稳定排序
    return fused[:top_k]
```

查询时序（ASCII，简单路径）：

```
caller ─► HybridSearch.search(query, top_k)
            │
            ├─ QueryProcessor.process(query) ──────► ProcessedQuery{keywords, filters}
            │
            ├─ ThreadPoolExecutor(max_workers=2)
            │     ├─[t1] DenseRetriever  ─embed─► ChromaDB.query ─► dense_results(≤20)
            │     └─[t2] SparseRetriever ─BM25──► bm25.query ─► get_by_ids ─► sparse_results(≤20)
            │           (任一路抛错 → 标 error，另一路结果直接作为 fused，used_fallback=True)
            │
            ├─ RRFFusion.fuse([dense, sparse], k=60, top_k) ─► fused(≤10)
            │     ├─ 记录 r.dense_score / r.sparse_score 交叉引用 (hybrid_search.py:655)
            ├─ _apply_metadata_filters (post-fusion 兜底过滤)
            ├─ _expand_with_structure (补 prev/next 邻居)
            └─► final_results[:top_k]    (CoreReranker 由 internal_loop / 上层按需调用)
```

设计取舍：RRF 选「秩融合」而非分数归一化——dense 的余弦相似度与 BM25 分数量纲完全不同，秩对异质打分天然鲁棒，且确定性（同分 chunk_id 兜底排序）。重排默认按 settings 可关（`RerankConfig.enabled`，`reranker.py:47`），关时直接截 top_k 返回，保证 reranker 后端不可用时检索不挂。

---

### 9.4 RAG MCP Server

**一句话**：MCP server 是个独立 stdio 子进程，stdout 只走 JSON-RPC、所有日志强制改道 stderr；启动时预热重依赖防 import 死锁；对外只暴露一小撮工具，向量库按 `{uid}_{kb_id}` 命名实现 per-uid 隔离。

握手与外壳：

- 入口 `server.run_stdio_server_async`（`backend/nanoresearch/rag/mcp_server/server.py:82`）。两个启动保护：`_redirect_all_loggers_to_stderr`（`server.py:25`，stdout 保留给协议流，任何 stdout 日志都会损坏 JSON-RPC）、`_preload_heavy_imports`（`server.py:47`，主线程预导 chromadb/onnxruntime/查询引擎模块，避免 `asyncio.to_thread` 工作线程与 stdin-reader 抢 Python import lock 死锁——明确写在注释里的坑）。
- 协议层 `ProtocolHandler`（`mcp_server/protocol_handler.py:43`）：`register_tool`（`:65`）/ `get_tool_schemas`（`:94`，tools/list）/ `execute_tool`（`:109`，tools/call，TypeError→INVALID_PARAMS、其它异常吞栈不外泄）。JSON-RPC 错误码集中在 `JSONRPCErrorCodes`（`protocol_handler.py:22`）。`create_mcp_server`（`protocol_handler.py:229`）用官方 `mcp.server.lowlevel.Server` 注册 `list_tools`/`call_tool` 两个 handler。

**实际暴露的工具集**（以 `register_tools` 实际执行为准）：

| 工具 | 注册处 | 用途 |
|---|---|---|
| `kb_search` | `tools/rag_search.py:332`（名 `:91`） | 智能多轮 KB 检索（按复杂度自动选简单/复杂路径） |
| `list_collections` | `tools/agentic/collections.py:1372` | 列集合 |
| `list_documents` | `collections.py:1380` | 列文档 |
| `ingest_document` | `collections.py:1388` | 摄入文档 |
| `delete_document` | `collections.py:1396` | 删文档 |
| `get_task_status` | `collections.py:1404` | 查异步摄入任务状态 |

> **坑/as-built 漂移**：`protocol_handler._register_default_tools`（`protocol_handler.py:198-226`）的文档与日志声称 `tools/agentic/retrieval.register_tools` 会注册 `kb_retrieve` 与 `memory_search`，但该函数当前是空实现（`tools/agentic/retrieval.py:529` 的 `register_tools` 仅 `pass`，因为 `FetchSectionTool/FetchNeighborsTool` 是内部循环工具、不对外）。故当前构建实际**不暴露** `kb_retrieve`/`memory_search`，与 `protocol_handler.py:226` 的 INFO 日志不符——属待修文档/实现漂移。

异步摄入：`AsyncTaskManager`（`mcp_server/async_tasks.py:78`，全局单例 `get_task_manager` `:323`）用 `ThreadPoolExecutor(max_workers=4)`，`submit`（`async_tasks.py:102`）让 `ingest_document` 立即返回 task_id、后台跑、`get_task_status` 工具轮询，含 TTL 清理（`cleanup` `:282`）。

**per-uid collection 隔离（落点确认）**：KB 创建时在 router 层把 Chroma collection 命名为 `f"{uid}_{kb.id}"`——`backend/nanoresearch/server/routers/knowledge_router.py:104`，写回 `kb.chroma_collection`。摄入/检索全程透传该 `chroma_collection`（`unified.py:108`、`worker.py:199/371`），KG 检索还反向从 collection 名解析 `{uid}_{kb_uuid}`（`agent/tools/graph_retrieval.py:61`）。因此**用户上传的知识库向量是按 uid 物理隔离**（不同租户落不同 collection，无跨租户读）。

> **残余隔离盲区（存疑点）**：研究记忆/claims 类向量走的是**固定共享 collection**——`research_chunks`（`cli/commands.py:486`）、`research_claims`（`collections.py:412`）、`research_insights`（`collections.py:461`）、`user_memory`——这些**不按 uid 隔离 collection**，靠查询时 metadata `filter by uid` 收口（`agent/context.py:68`）。这正是项目记忆里标记的「research_chunks 要按 uid 隔离避免跨租户读泄漏」的落点：KB 文档已隔离，研究记忆侧仍是「共享库 + uid 元数据过滤」，一旦过滤遗漏即跨租户泄漏，需人审。

---

### 9.5 internal_loop 数据飞轮

**一句话**：当 `kb_search` 判定查询「复杂」时进入内部循环——一个 plan→search→fuse→verify→finalize 的有界状态机，用「系统强制 verify + 置信度阈值」决定是否再检索一轮，并用结构/邻居扩展补盲。

角色分工：

- **runner**（`backend/nanoresearch/rag/internal_loop/runner.py:91`）：状态机驱动。`run()`（`runner.py:190`）流程——`classify_complexity`（`runner.py:26`，纯规则：含指代词且无上下文 / 对比词 / 多问号 → complex）在 `rag_search.py:169` 决定走简单还是复杂路；复杂路进 loop（`runner.py:240`，最多 `max_iterations` 轮）：Phase1 `_run_plan_phase`（`runner.py:393`）→ Phase2 `_run_search_phase`（`runner.py:432`）→ 对比查询前置 `expand_with_sections`（`runner.py:257`）→ Phase3 `fuse_results`(rrf,top_k=20) + `verify_results`（`runner.py:264-279`）→ 命中阈值（`verify.answered` 或 `confidence>=0.7`，`runner.py:285`）则 Phase4 `build_citations` 收尾；否则 `expand_with_neighbors`（`runner.py:300`）补邻居、把 `next_actions` 注入为新 sub_queries 续跑。
- **state**（`internal_loop/state.py`）：三级状态。`SessionState`（`state.py:73`，整个 kb_search 一次会话，含 `caller_session_key`）/`RoundState`（`state.py:125`，单轮，`results: query→chunks`，`fused_chunks`）/`SubQuery`（`state.py:28`，带 `strategy` 标注）。`SessionStateManager`（`state.py:165`）是线程安全单例，TTL 30 分钟。
- **tools**（`internal_loop/tools.py:54` `InternalTools`）：把 agentic 工具包成循环可用的统一接口——`plan_query`（`tools.py:219`）/`execute_batch`（`tools.py:268`）/`fuse_results`（`tools.py:416`）/`verify_results`（`tools.py:455`）/`build_citations`（`tools.py:539`）/`expand_with_sections`（`tools.py:319`，对比关键词当 section_path 过滤）/`expand_with_neighbors`（`tools.py:363`，对 top-N 取 prev/next 邻居，dedup 后只回新 chunk）。底层检索由 `ConcurrentRetrievalEngine`（`mcp_server/tools/agentic/batch_retrieval.py:140`）按每个 task 的 `strategy` 分派 dense/sparse/hybrid（`batch_retrieval.py:199/207/217`），写入 `RoundStateManager`（`tools/agentic/round_state.py:119`），最后 `round.fuse(strategy="rrf")`（`round_state.py:324`）做轮级 RRF。
- **cleanup**（`internal_loop/cleanup.py:20` `cleanup_messages_for_next_round`）：token 控制。每轮注入 next_actions 后裁剪消息——保留全部 system + 第一条 user（原始 query）+ 裁剪后的 tool/assistant 结果（常量 `MAX_TEXT_LENGTH=200`、`MAX_RESULTS_TO_KEEP=3`、`MAX_MESSAGES_TO_KEEP=10`，`cleanup.py:15-17`），防多轮 token 膨胀。

「知识内循环」做什么：循环本质是**自适应检索深度**——简单查询一跳直返（`rag_search.py:172` 简单路径，高分 chunk（score≥0.85）仍自动补邻居 `rag_search.py:229`）；复杂/对比/指代查询则反复「检索→系统强制自评（verify 给 confidence/missing_aspects/next_actions）→按缺口补检索」，直到自评满足或撞 `max_iterations`。verify 是系统强制而非 LLM 自由决定收尾，避免提前停。

```
   kb_search(complex) ─► RAGLoopRunner.run
        │
   Phase1 Plan ──► plan_query(query, context, session_key) ─► sub_queries[{query,strategy}]
        │                                                       （含指代消解，见 9.6）
   ┌────▼──── for iteration in range(max_iterations) ───────────────────────┐
   │ Phase2 Search ─► execute_batch(tasks) ─► RoundState.add_results        │
   │ Phase2.5      ─► expand_with_sections (仅对比查询)                       │
   │ Phase3 Fuse   ─► round.fuse(rrf, top_k=20)                              │
   │        Verify ─► verify_results → {confidence, missing, next_actions}   │
   │            confidence>=0.7 or answered? ──yes──► Phase4 build_citations ─► return
   │                    │ no                                                  │
   │            expand_with_neighbors + 注入 next_actions + cleanup ─► 续轮   │
   └────────────────────────────────────────────────────────────────────────┘
```

---

### 9.6 query rewrite / 指代消解

**一句话**：外层 Agent 的 `session_key` 沿 `kb_search → runner → plan_query` 透传到 MCP 子进程；子进程内用 **PG-backed SessionManager** 拉取多轮历史，LLM 改写当前 query 解析指代/省略，再做策略规划。

A 类透传链（每一跳真实锚点）：

```
kb_search.input_schema.session_key   (mcp_server/tools/rag_search.py:132)
  └► RAGSearchTool.execute(session_key)         (rag_search.py:146)  ──复杂路──►
       run_rag_loop(session_key)                (rag_search.py:287)
         └► RAGLoopRunner.run(session_key)       (internal_loop/runner.py:196)
              └► create_session(caller_session_key=session_key)  (runner.py:217 → state.py:217)
                   └► _run_plan_phase 用 session.caller_session_key  (runner.py:406)
                        └► InternalTools.plan_query(session_key)     (internal_loop/tools.py:219)
                             └► PlanQueryTool.execute(session_key)   (query_planning.py:233)
```

子进程 PG-backed SessionManager 取历史 + 指代消解（`backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py`）：

- `_get_subprocess_session_manager`（`query_planning.py:31`）：MCP server 是独立 stdio 子进程，主进程的模块级 SessionManager **不可见**；这里在子进程里用经 stdio env 传入的 `DATABASE_URL/REDIS_URL` 连**同一套 PG/Redis**，自建 `SessionManager`（`query_planning.py:52`）。注释明确：**禁止 JSONL fallback**（否则会建出与主会话不同步的孤儿存储），init 失败就降级返回空历史、跳过改写。
- `execute`（`query_planning.py:233`）顺序：① `_get_conversation_history(session_key)`（`query_planning.py:337`，取 `session.messages[-20:]`，故意不做 legal-start 过滤以保留 tool 消息）→ ② `_get_retrieval_titles`（`query_planning.py:403`，从最近 RAG tool 消息读 `_chunk_titles` sidecar，A 类期该字段不存在、返回 `[]`，是为 B 类预留）→ ③ `_rewrite_query`（`query_planning.py:421`，用 `REWRITE_PROMPT`（`query_planning.py:64`）+ 渲染后的近 6 轮 user/assistant（`_render_history_for_prompt` `:381`）让 LLM 把"它/那篇/上面那个"解析成独立完整 query；无历史且无 titles 时原样返回 `:430`）→ ④ 用改写后 query 做策略规划（`PLANNING_PROMPT`，标注每个子查询 dense/sparse/hybrid），结果回带 `original_query`/`rewritten_query`/`context_used`。
- 与 9.5 的咬合：`classify_complexity` 把「含指代词且无 context」的查询判为 complex（`runner.py:37`），正是为了把它送进复杂路、触发 plan_query 的历史改写——指代消解依赖多轮历史，简单路不拉历史。

设计取舍/坑：改写全链路**优雅降级**——SessionManager init 失败、历史为空、LLM 不可用、改写结果为空，任一环都回退原始 query（`query_planning.py:59/430/458/462`），即指代消解是「尽力而为」的增强而非硬依赖。当前是 A 类（透传 + 子进程 PG SessionManager 取历史改写）已落地；B 类（RAG tool 消息回写 `_chunk_titles` sidecar，让"上一轮检索到的那篇"也能消解）的读取侧已就位（`_get_retrieval_titles`），但写入侧（sidecar 落库）在 A 类期未产出该字段，故"上一轮检索标题"路径当前恒空——这是已知的下一步缺口。


## Ch10 Deep Research 编排

深度研究模块将用户提交的研究课题拆解为子问题，通过并行搜索→合成→评估→迭代的闭环，最终生成带引用溯源的 Markdown 报告。

### 10.1 总体架构

编排入口是 `ResearchRunner`，通过懒加载方式在首次调用时实例化五个子组件，避免循环导入：

```
backend/nanoresearch/research/runner.py:36–50
_lazy_imports()  →  _PLANNER=ResearchPlanner, _SEARCHER=SearchOrchestrator,
                    _SYNTHESIZER=InformationSynthesizer, _REFINER=ResearchRefiner,
                    _REPORTER=ReportGenerator
```

`ResearchRunner.__init__` (`runner.py:65–96`) 接收 `LLMProvider`、`web_search_tool`、`web_fetch_tool`、`ResearchConfig`、`knowledge_search`、`rag_store`、`settings`（RAG 配置）、`uid`、`workspace` 等参数，对应流水线的每个环节都注入了所需依赖。

### 10.2 编排流程 ASCII 图

```
用户 topic
    │
    ▼
[Phase 0] KnowledgeSearch（已有研究报告）
    + ChromaStore（用户上传文档）
    │ combined_context
    ▼
[Phase 1] ResearchPlanner.plan()
    │ ResearchPlan: sub_questions[]
    ▼
┌─────────────────────────────────────────────┐
│  for iteration in range(max_iterations):    │
│                                             │
│  [Phase 2] SearchOrchestrator.search()      │
│      └─ 并行搜 pending sub_questions        │
│         URL 去重(跨轮全局) + 评分 + rerank  │
│      → accumulated_results (cap=50)         │
│                                             │
│  [Phase 3] InformationSynthesizer.synthesize│
│      → SynthesisResult                     │
│        .coverage_score  ◄─── 驱动收敛      │
│        .findings                            │
│        .source_assignments                  │
│        .knowledge_gaps                      │
│                                             │
│  [Phase 4] ResearchRefiner.should_continue()│
│      coverage >= 0.7  ──► STOP             │
│      coverage declining  ─► STOP           │
│      max_iterations reached ─► STOP        │
│      else: refiner.refine() → new plan     │
│        (新增 sub_questions, 补充关键词)     │
└─────────────────────────────────────────────┘
    │ 最终 SynthesisResult
    ▼
[Phase 5] ReportGenerator.generate()
    │ 逐子问题写章节 + _integrate() 汇总
    ▼
[Phase 6] ReportGenerator.self_evaluate()
    │ overall < 6.0 → 重新生成一次
    ▼
[Phase 7] _save_report_md() + _auto_ingest()
    → research_notes/{rid}_{slug}.md
    → IngestionPipeline（fire-and-forget）
    ▼
ResearchResult（report, metrics, execution_log）
```

### 10.3 各阶段详解

#### Phase 0 — 知识预热

`runner.py:179–203`。调用 `_get_existing_knowledge()` 通过 `HybridSearch.async_search()` 检索已有研究报告（`filter: source=research, uid=<uid>`），再调用 `_get_document_context()` 从 `ChromaStore`（用户上传文档集）检索相关 chunk。两部分合并为 `combined_context` 传入规划阶段，指导规划器跳过已有结论、聚焦增量子问题。

#### Phase 1 — ResearchPlanner（规划器）

**职责**：将研究主题 LLM tool-call 拆解为 3–6 个 `SubQuestion`，每个子问题含中英文关键词和优先级。

- 类：`ResearchPlanner` (`planner.py:147`)
- 核心方法：`plan(topic, depth, existing_context)` (`planner.py:154`)
- LLM 工具：`_RESEARCH_PLAN_TOOL`（`research_plan` function）(`planner.py:14`)
- 深度映射 (`runner.py:121–129`)：
  - `quick` → max_iterations=1, max_sources=5，拆 2–3 子问题
  - `normal` → max_iterations=3, max_sources=10，拆 4–5 子问题
  - `deep` → max_iterations=5, max_sources=20，拆 5–6 子问题
- `existing_context` 非空时注入 `_EXISTING_CONTEXT_TEMPLATE` (`planner.py:136`)，提示 LLM 生成增量子问题
- 无法解析结构化输出时降级到 `_fallback_plan()`（3 个固定模板子问题）

#### Phase 2 — SearchOrchestrator（搜索编排器）

**职责**：仅搜 `status == "pending"` 的子问题，并行发起 web_search + web_fetch，评分去重。

- 类：`SearchOrchestrator` (`searcher.py:160`)
- 核心方法：`search(plan)` → `(list[SearchResult], rerank_details)` (`searcher.py:216`)
- 并发控制：`_MAX_CONCURRENT_SUBQ=2`（子问题）、`_MAX_CONCURRENT_FETCH=3`（URL 抓取） (`searcher.py:163–165`)
- 评分公式（`SearchResult.__post_init__`，`types.py:87`）：
  ```
  final_score = credibility * 0.4 + relevance * 0.4 + recency * 0.2
  ```
  - `credibility`：域名信誉分（`.gov/.edu` 等 → 0.9，`medium.com` 等 → 0.6，其他 → 0.4）
  - `relevance`：关键词命中率（`_score_result` `searcher.py:408`）
  - `recency`：内容时效性（发布日期或内容年份关键词，`_calculate_recency` `searcher.py:81`）
- 跨轮全局去重：`_global_seen_urls` 集合，迭代间不重复抓取同一 URL (`searcher.py:479–485`)
- 失败域名降权：`_failed_domains` 计数；SSL/连接错误 +3 立即触发阈值（≥3 跳过）(`searcher.py:399–404`)
- Rerank：可选 cross-encoder（`BAAI/bge-reranker-v2-m3`），启用时替换 `final_score`（`_rerank_results` `searcher.py:259`）

#### Phase 3 — InformationSynthesizer（信息合成器）

**职责**：将本轮累积搜索结果（跨迭代 top-50）合成为结构化 `SynthesisResult`，计算覆盖度分。

- 类：`InformationSynthesizer` (`synthesizer.py:195`)
- 核心方法：`synthesize(results, plan)` (`synthesizer.py:202`)
- LLM 工具：`_SYNTHESIZE_TOOL`（`synthesize` function）(`synthesizer.py:22`)
- 输出包含：`findings`（核心结论列表）、`source_assignments`（来源→子问题映射）、`contradictions`（矛盾观点）、`knowledge_gaps`、`coverage_score`
- 覆盖度计算（`_calc_coverage_score` `synthesizer.py:341`）：
  - LLM 对每个子问题输出 `sufficient/partial/insufficient`（分别映射 1.0/0.6/0.2）
  - 所有子问题分值均值即为 `coverage_score`
  - 无 LLM 输出时降级到 `total_results / (len(sub_questions) * 3)`

#### Phase 4 — ResearchRefiner（细化器）

**职责**：决定是否继续迭代，并生成补充子问题或关键词。

- 类：`ResearchRefiner` (`refiner.py:115`)
- 快速判断（无 LLM）：`should_continue(synthesis, iteration, config, prev_coverage)` (`refiner.py:122`)
- LLM 细化：`refine(plan, synthesis, config)` (`refiner.py:144`)

**收敛条件伪代码**：

```
# runner.py:257, refiner.py:122–141
should_continue(synthesis, iteration, config, prev_coverage):
    if iteration >= config.max_iterations:          # 达到最大轮次
        return False
    if synthesis.coverage_score >= 0.7:             # 覆盖度阈值（min_coverage_threshold）
        return False
    if prev_coverage is not None:
        if (prev_coverage - synthesis.coverage_score) >= 0.05:  # 覆盖度下降（coverage_decline_threshold）
            return False                            # 继续迭代只会稀释平均分
    return True

# refine() LLM 决策（refiner.py:144）
refine(plan, synthesis, config):
    if should_exit_early:
        return None                                # stop_reason = "no_gaps"
    new_sub_questions = llm_call(gaps, contradictions)[:max_new_sub_questions_per_iteration]  # cap=3
    for new_sq in new_sub_questions:
        plan.sub_questions.append(SubQuestion(status="pending"))  # 新增 pending
    # 已有子问题的 status="completed" 不重新搜索（Phase 2 过滤）
    return updated_plan
```

停止原因（`stop_reason`）：`coverage_threshold` / `max_iterations` / `no_gaps` / `coverage_declining`
跨迭代结果池（`accumulated_results_cap=50`）：每轮新结果追加，超限时按 `final_score` 保留 top-50 (`runner.py:235–238`)。

#### Phase 5 — ReportGenerator（报告生成器）

**职责**：按子问题逐节写作 + 整合 + 引用溯源。

- 类：`ReportGenerator` (`reporter.py:196`)
- 核心方法：`generate(topic, synthesis, plan)` (`reporter.py:203`)
- 分节写作：`_write_section()` 每个子问题调用一次 LLM，传入 `source_assignments` 选出的 top-3 相关来源原文（`MAX_CHARS_PER_SOURCE=8000` `reporter.py:193`）
- 引用规范（系统提示 `reporter.py:67`）：`[citation:Title](URL)` 格式，每个事实声明后必须标注
- 整合：`_integrate()` 汇总所有章节草稿，LLM 输出执行摘要 + 润色 + 矛盾与争议 + 知识空白章节 (`reporter.py:340`)
- 最终 Markdown 由 `_build_markdown()` 组装 (`reporter.py:402`)

#### Phase 6 — 自评估

`reporter.self_evaluate()` (`reporter.py:492`)，LLM 对报告打分（completeness、accuracy、readability、overall，0–10）。若 `overall < evaluation_threshold`（默认 6.0）则重新调用 `generate()` 一次，写回 `result.metrics` 和 `result.quality_score` (`runner.py:310–318`)。

#### Phase 7 — 报告落盘与自动入库

`_save_report_md()` (`runner.py:414`) 在 `research_notes/` 目录生成带 YAML front-matter 的 `.md` 文件（`source: research`、`uid`、`quality_score`），随后 `asyncio.create_task(_auto_ingest())` 异步调用 `IngestionPipeline` 将其注入 RAG 向量库（fire-and-forget，失败只记 warning）(`runner.py:327–335`)。

### 10.4 knowledge_search / knowledge_lint 的角色

**KnowledgeSearch** (`knowledge_search.py:29`) 在研究流水线中扮演两个角色：

1. **Phase 0 知识预热**：`dense_encoder.embed()` 生成查询向量，通过 `HybridSearch`（BM25+vector→RRF→Rerank→时间衰减）检索已有研究报告片段 (`runner.py:368–408`)
2. **用户上传文档检索**：`_get_document_context()` 直接调用 `knowledge_search.dense_encoder.embed([topic])[0]` 生成向量后查询 `rag_store`（用户文档 Chroma collection）(`runner.py:490`)

**KnowledgeLint** (`knowledge_lint.py:95`) 是独立的质量巡检工具，不在研究流水线运行时调用。它对 user_memory_store 中的 claim 执行：
- **结构检查**（无 LLM）：`lint_structural()` — `orphan_claims`、`broken_refs`、`missing_evidence`、`empty_domains`、`invalid_confidence` (`knowledge_lint.py:132`)
- **语义检查**（LLM）：`lint_semantic()` — `self_contained` 并发 LLM 判定，三级裁决 KEEP/DEMOTE/DELETE (`knowledge_lint.py:325`)

### 10.5 关键数据类型（types.py）

| 类型 | 文件:行 | 说明 |
|------|---------|------|
| `ResearchConfig` | `types.py:294` | 收敛参数：`min_coverage_threshold=0.7`、`evaluation_threshold=6.0`、`coverage_decline_threshold=0.05`、`accumulated_results_cap=50` |
| `ResearchPlan` | `types.py:53` | topic + sub_questions[] + iteration 计数 |
| `SubQuestion` | `types.py:32` | id/question/keywords/priority/status（pending→searching→completed） |
| `SearchResult` | `types.py:71` | url/title/content/credibility/relevance/recency/final_score |
| `SynthesisResult` | `types.py:151` | findings/contradictions/knowledge_gaps/coverage_score/source_assignments/sources |
| `ResearchResult` | `types.py:259` | 最终输出：report(str)/plan/synthesis/metrics/quality_score/execution_log |
| `ExecutionLog` | `types.py:233` | 白盒化：每轮 SearchIterationLog + stop_reason + final_coverage_score |

---

## Ch11 持久化层

持久化层由三个子系统组成：PostgreSQL（结构化数据）、Redis（短期缓存与事件总线）、Chroma（向量存储）。此外有一套轻量手动 SQL 迁移机制。

### 11.1 PostgreSQL — 结构化数据

#### Engine 与 Session 工厂

`database.py` 维护模块级单例：

```
backend/nanoresearch/storage/database.py
init_engine(url)                       # 行 30–34
  create_async_engine(url, echo=False, pool_pre_ping=True)
  async_sessionmaker(engine, expire_on_commit=False)

get_db() → AsyncGenerator[AsyncSession]  # 行 129–133（FastAPI 依赖注入）
init_db()                               # 行 43–49（建表，dev 用）
check_schema_migrations()               # 行 52–126（启动检查，列缺失则 SystemExit(1)）
```

`DATABASE_URL` 从环境变量读取（格式：`postgresql+asyncpg://...`）。`pool_pre_ping=True` 自动探活连接。`expire_on_commit=False` 使 ORM 对象在 `commit()` 后仍可访问字段。

`check_schema_migrations()`（`database.py:52`）在启动时探测 30+ 个已知列是否存在，如有缺失打印 warning 并调用 `SystemExit(1)` 阻止服务启动，防止运行时 cryptic 500。

#### 主要 ORM 表（models.py）

所有模型继承 `Base`（`database.py:12`，`DeclarativeBase`）。

| ORM 类 | 表名 | 一句话说明 | 文件:行 |
|--------|------|-----------|---------|
| `User` | `users` | 用户账户，uid PK + 密码哈希 | `models.py:19` |
| `UserSettings` | `user_settings` | uid PK + model/max_iterations + extra JSONB | `models.py:30` |
| `Agent` | `agents` | Agent 定义：skills_config/tools_config/harness/persona JSONB | `models.py:40` |
| `AgentKnowledgeBinding` | `agent_knowledge_bindings` | Agent ↔ KnowledgeBase M2M | `models.py:61` |
| `Conversation` | `conversations` | 会话：session_key 唯一索引、last_consolidated 整数 | `models.py:73` |
| `Message` | `messages` | 消息：content JSONB、seq 整数序号、CASCADE 删 | `models.py:90` |
| `AgentRun` | `agent_runs` | 单次 Agent 执行：tool_calls/tokens_used JSONB | `models.py:103` |
| `KnowledgeBase` | `knowledge_bases` | 知识库元信息：chroma_collection/enable_graph_expansion | `models.py:130` |
| `KbDocument` | `kb_documents` | 文档：filename/content_hash/status/pdf_parser | `models.py:150` |
| `KbChunk` | `kb_chunks` | 分块：chroma_id/content/char_start/char_end | `models.py:167` |
| `KgEntity` | `kg_entities` | 知识图谱实体：name+label+kb_id 唯一约束 | `models.py:187` |
| `KgEntityMention` | `kg_entity_mentions` | 实体出现在哪个 chunk | `models.py:199` |
| `KgTriple` | `kg_triples` | 实体关系三元组：source_id/target_id/label | `models.py:210` |
| `KgTripleMention` | `kg_triple_mentions` | 三元组出现在哪个 chunk | `models.py:222` |
| `EvalDataset` | `eval_datasets` | RAG 评估数据集 | `models.py:233` |
| `EvalDatasetItem` | `eval_dataset_items` | 评估问答对：query/gold_chunk_ids/gold_answer | `models.py:243` |
| `EvalRun` | `eval_runs` | RAG 评估批次：metrics JSONB/overall_score | `models.py:255` |
| `EvalRunItem` | `eval_run_items` | 单条评估结果：retrieved_contexts/item_metrics JSONB | `models.py:275` |
| `AgentEvalRun` | `agent_eval_runs` | Agent 批评估批次：baseline_eval_run_id/has_regression | `models.py:292` |
| `AgentRunSnapshot` | `agent_run_snapshots` | Agent 运行快照：badcase/scores/context_trace/tool_recordings JSONB | `models.py:313` |
| `AgentTestCase` | `agent_test_cases` | 测试用例：set_kind/tool_recordings/origin_badcase_id | `models.py:368` |
| `JudgeCalibrationLog` | `judge_calibration_logs` | Judge 校准日志：MAD 值/passed | `models.py:409` |
| `OptimizationProposal` | `optimization_proposals` | 优化提案：proposals JSONB/baseline_score/score_sample | `models.py:421` |
| `TunableObjectVersion` | `tunable_object_versions` | 可调对象版本注册表：kind/target_id/content/active | `models.py:439` |

#### 9 个 Repository 与对应表

| Repository 类 | 文件 | 管辖的主要表 |
|--------------|------|------------|
| `AgentEvalRepository` | `agent_eval_repo.py:48` | `agent_run_snapshots`、`agent_eval_runs`、`agent_test_cases`、`judge_calibration_logs`、`optimization_proposals`、`tunable_object_versions` |
| `AgentRepository` | `agent_repo.py:54` | `agents`、`agent_knowledge_bindings` |
| `ConversationRepository` | `conversation_repo.py:14` | `conversations`、`messages` |
| `EvalRepository` | `eval_repo.py:14` | `eval_datasets`、`eval_dataset_items`、`eval_runs`、`eval_run_items` |
| `GraphRepository` | `graph_repo.py:37` | `kg_entities`、`kg_entity_mentions`、`kg_triples`、`kg_triple_mentions` |
| `KnowledgeRepository` | `knowledge_repo.py:58` | `knowledge_bases`、`kb_documents`、`kb_chunks` |
| `RunRepository` | `run_repo.py:14` | `agent_runs` |
| `UserRepository` | `user_repo.py:11` | `users` |
| `UserSettingsRepository` | `user_settings_repo.py:120` | `user_settings` |

**设计取舍**：每个 Repository 在方法内独立开关 `async with self._factory() as session`，无长事务跨 repo 调用。`AgentEvalRepository` 兼管六张表（快照/评估/测试用例/校准/提案/版本），因为这些表均属 SDD 飞轮闭环，操作往往需要跨表原子读写（如 `delete_eval_run()` 先删 snapshots 再删 run）。

### 11.2 Redis — 缓存与事件总线

#### 接入

`redis_client.py` 维护进程级单例：

```
backend/nanoresearch/bus/redis_client.py:15–18
get_redis() -> aioredis.Redis
  _client = aioredis.from_url(REDIS_URL, decode_responses=True)
```

`decode_responses=True` 全局强制字符串解码，避免调用方 `b"..."` 处理。`REDIS_URL` 默认 `redis://localhost:6379`，由环境变量覆盖。

#### 键命名空间（redis_keys.py）

```
backend/nanoresearch/bus/redis_keys.py
RedisKeys.session_msg(uid, ch, chat_id)  → "session:msg:{uid}:{ch}:{chat_id}"   (List)
RedisKeys.session_meta(uid, ch, chat_id) → "session:meta:{uid}:{ch}:{chat_id}"  (Hash)
RedisKeys.agent(agent_id)               → "agent:{agent_id}"                    (Hash)
RedisKeys.user_settings(uid)            → "user_settings:{uid}"                 (Hash)
RedisKeys.kb_meta(kb_id)               → "kb:meta:{kb_id}"                     (Hash)
RedisKeys.chunk(ns, chunk_id)           → "chunk:{ns}:{chunk_id}"               (String)
RedisKeys.embedding(text_hash)          → "embedding:{text_hash}"               (String)
RedisKeys.run_events(run_id)            → "run_events:{run_id}"                 (Stream)
RedisKeys.chat_events(chat_id)          → "chat_events:{chat_id}"               (Stream)
RedisKeys.cancel(session_key)           → "cancel:{session_key}"                (String,无 TTL)
RedisKeys.pending(session_key)          → "pending:{session_key}"               (String,无 TTL)
```

TTL 常量（`redis_keys.py`）：`SESSION_TTL=7200`s、`AGENT_TTL=1800`s、`USER_SETTINGS_TTL=1800`s、`KB_META_TTL=600`s、`CHUNK_TTL=21600`s、`EMBEDDING_TTL=3600`s、`RUN_EVENTS_TTL=86400`s。

#### 缓存写策略：DEL + 全量 RPUSH（非增量）

会话消息列表的写入采用**原子替换**策略，落点在 `session/manager.py:_redis_save()`：

```python
# backend/nanoresearch/session/manager.py:189–201
async with redis.pipeline(transaction=True) as pipe:
    pipe.delete(msg_key)                              # 先清除旧 List
    if session.messages:
        pipe.rpush(msg_key, *[json.dumps(m) ...])    # 全量重写
    pipe.hset(meta_key, mapping={...})
    pipe.expire(msg_key, RedisKeys.SESSION_TTL)
    pipe.expire(meta_key, RedisKeys.SESSION_TTL)
    await pipe.execute()                              # MULTI/EXEC 原子执行
```

**为何不能增量 RPUSH**：Redis 配置为 `volatile-lru` 淘汰策略，带 TTL 的 key 在内存压力下随时可能被部分淘汰。若采用增量 RPUSH，List 可能已被淘汰一部分，追加写入后得到截断+新增的乱序列表，`last_consolidated` 偏移量计算错误，导致历史消息静默丢失（2026-06-28 观测到生产 bug，见 `manager.py:172–181` docstring）。DEL+全量 RPUSH 保证任何时刻 Redis 中要么是空 key（淘汰后），要么是完整列表，不会出现中间态。

**配置热缓存**（`agent_repo.py`、`knowledge_repo.py`、`user_settings_repo.py`）：用 Hash（`hset/hgetall`）缓存 Agent/KnowledgeBase/UserSettings 对象；写入/更新后调用 `get_redis().delete(cache_key)` 主动失效，下次读取时回源 Postgres 并回填缓存。

#### 事件总线

- **Redis Stream**（`bus/stream.py`）：`xadd_event()` 写 run_events Stream；超 8KB 时切片多条消息，带 `chunk_group_id` 字段；`xread_next()` 按游标读取并重组分片。SSE 推流（`chat_events:{chat_id}`）给前端。
- **控制信号**（`cancel:`/`pending:`）：无 TTL，手动 DEL，用 `volatile-lru` 也不会误淘汰（因为无 TTL 的 key 不参与 lru 淘汰）。

### 11.3 Chroma — 向量存储

Chroma 是 RAG 和用户记忆的向量后端。

**Per-uid 隔离**：`KnowledgeSearch.from_settings()` 接收 `collection_suffix`，构造 `user_memory{collection_suffix}` collection name（`knowledge_search.py:54`）。不同用户对应不同 collection，避免跨租户向量泄漏（与 Ch9 `research_chunks` per-uid 隔离策略一致）。

**研究报告入库**：Phase 7 中 `_auto_ingest()` 调用 `IngestionPipeline`，将研究报告 MD 切块后写入 `settings.vector_store.collection_name`（`runner.py:456–458`），元数据带 `source=research` 标签，供 Phase 0 知识预热时过滤检索。

**用户上传文档**：单独的 `rag_store`（`ChromaStore`，默认 collection），通过 `_get_document_context()` 查询（`runner.py:473`）。

**ChromaStore 接口**（`rag/libs/vector_store/chroma_store.py`）：`query(vector, top_k)`、`insert_batch(rows)`、`query_batch(vectors, top_k, threshold)`、`delete(ids)`、`get_all_documents()`。

### 11.4 Schema 迁移

迁移文件存于 `backend/migrations/`，**不使用 Alembic**，纯手动 SQL 脚本，`psql` 直接执行。

**约定**（自 A1 Phase 1，2026-06-26 起）：

- 每次迁移配对两个文件：`<name>.sql`（up，含 `IF NOT EXISTS`）+ `<name>_down.sql`（down，含 `IF EXISTS`）
- 文件名字母序决定执行顺序；需要有序时加数字前缀 `001_`
- NOT NULL 列分两步：先加 nullable 列 → 执行回填脚本（`scripts/backfill_*.py`）→ 再加约束（见 `add_case_metadata.sql` + `add_case_metadata_enforce.sql`）

已有迁移文件：`add_root_cause_auto.sql`、`add_pending_cases_fields.sql`、`add_case_metadata{_enforce,_down}.sql`、`add_proposal_score_sample.sql`、`add_proposal_signal_unreliable.sql`。

`database.py:check_schema_migrations()` 在服务启动时探测所有关键列，缺失则 `SystemExit(1)`，是迁移漏跑的最后一道防线。初始建表由 `init_db()` 调用 `Base.metadata.create_all` 完成，仅用于 dev/test。

### 11.5 设计取舍与已知坑

**Postgres 异步驱动**：使用 `asyncpg`，所有 I/O 非阻塞，但 ORM `expire_on_commit=False` 意味着对象字段在 session 关闭后仍然有效但可能 stale——调用方需在必要时 `refresh`（如 `agent_repo.py:update()` 行 125 显式调用 `refresh`）。

**Redis 降级**：所有 Redis 操作均在 `try/except Exception: pass` 内执行，Redis 不可用时降级为直接 Postgres 读取，避免 Redis 单点故障影响核心服务。

**AgentEvalRepository 宽表**：`AgentRunSnapshot` 含 30+ 列（badcase 状态、Phase 0–2 字段、v2 评估字段），历史演进中字段只增不减，通过 `check_schema_migrations()` 的列清单维护兼容性，而非版本化 schema。

**GraphRepository Upsert**：使用 `pg_insert(...).on_conflict_do_nothing()` + 主动回查冲突行，在高并发写入时保证幂等（`graph_repo.py:72–89`）。实体 ID 为 `sha256(kb_id:name:label)[:16]` 确定性生成，相同内容多次写入不产生重复。


## Ch12 Web 服务与前端

---

### 12.1 后端 Web 服务

**职责**：将 Agent 执行引擎、RAG 管道、评测系统暴露为 HTTP/SSE API，并在同一进程中托管 Vue 前端静态产物。

#### 12.1.1 FastAPI 应用工厂

入口为 `backend/nanoresearch/server/main.py:32` 的 `create_app(channel_loop, session_factory, ...)` 工厂函数，返回配置完毕的 `FastAPI` 实例。

**Lifespan 初始化序列**（`main.py:33-91`）：

```
startup
  ├── migrate_llm_keys()          — 一次性 api_keys 迁移
  ├── _engine.dispose()           — 清理旧 asyncpg 连接（Windows ProactorLoop 问题）
  ├── get_redis() → app.state.redis
  ├── PendingReaper.start()       — 孤儿 run 清理（后台）
  ├── RedisMonitor.start()        — 驱逐告警 + 内存采样
  ├── create_pool(ArqRedisSettings) → app.state.arq_pool
  └── channel_loop.run() + channel_manager.start_all()  （可选）

shutdown
  ├── channel_loop.stop() / channel_manager.stop_all()
  ├── web_loops 全部 stop()
  ├── pending_reaper.stop() / redis_monitor.stop()
  └── arq_pool.aclose() / redis.aclose()
```

**App 实例**（`main.py:92-93`）：
- `FastAPI(title="Nanoresearch API", version="2.0.0", lifespan=lifespan)`
- 全局异常处理器：`ModelResolutionError` → 422 JSON（`missing_provider` + `role` + `message`）

**Auth 内联端点**（`main.py:106-120`）：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/auth/token` | 接收 `OAuth2PasswordRequestForm`（form-data），验证密码哈希，返回 `{ access_token, token_type: "bearer" }` |
| `GET` | `/api/auth/me` | 返回 `{ uid }` |

**静态文件挂载**（`main.py:138-149`）：路由注册完成后按顺序挂载。必须在前端 `"/"` 之前挂载 RAG 图片路由，否则图片路径被前端 SPA 捕获。

```
/rag-images  →  NANORESEARCH_HOME/rag/images  （主机文件系统，静态伺服）
/            →  web/dist                       （Vue 构建产物，html=True 兜 SPA）
```

---

#### 12.1.2 七个 Router 端点总览

| Router | 文件 | 前缀 | 核心端点（方法 路径） | 用途 |
|--------|------|------|----------------------|------|
| `chat_router` | `server/routers/chat_router.py` | 无 | `GET/POST /api/conversations`、`GET/DELETE /api/conversations/{id}`、`GET /api/conversations/{id}/messages`、`PUT /api/conversations/{id}/agent-override`、`GET /api/conversations/{id}/runs`、`POST /api/runs`（201）、`GET /api/runs/{id}`、**`GET /api/runs/{id}/events`**（SSE） | 对话全生命周期 + 实时 run 事件流 |
| `agent_router` | `server/routers/agent_router.py` | 无 | `GET /api/skills`、`POST/GET /api/agents`、`GET/PUT/DELETE /api/agents/{id}`、`GET /api/agents/{id}/prompt-preview`、`GET /api/agents/{id}/tool-stats`、`GET /api/agents/{id}/runs`、`GET/POST /api/agents/{id}/knowledge`、`DELETE /api/agents/{id}/knowledge/{kb_id}` | Agent 卡片 CRUD + 知识库绑定 + skill 查询 |
| `agent_eval_router` | `server/routers/agent_eval_router.py` | `/api/eval/agent` | snapshots、badcases、testcases、eval-runs、trends、replay、classify-batch、optimize、tunable apply/rollback、diagnosis、pending-cases、data-flywheel | Agent 评测闭环（快照→badcase→优化提案→应用回滚） |
| `knowledge_router` | `server/routers/knowledge_router.py` | 无 | `GET/POST /api/knowledge`、`GET/PUT/DELETE /api/knowledge/{id}`、`GET/POST /api/knowledge/{id}/documents`、document file download/delete、chunks、`POST /api/knowledge/{id}/query/test`、graph build/stats、`POST /api/rag/image-ticket`、`GET /api/rag/images` | 知识库 CRUD + 文档摄取 + RAG 图片代理 |
| `eval_router` | `server/routers/eval_router.py` | 无 | `GET/POST /api/eval/{kb_id}/datasets`、dataset upload/generate/delete、`POST /api/eval/{kb_id}/runs`（quick）、`POST /api/eval/{kb_id}/runs/ragas`、`POST /api/eval/{kb_id}/runs/agent`、`GET/DELETE /api/eval/{kb_id}/runs/{id}` | RAG 评测（Quick Recall@K、RAGAS、Agent-RAG） |
| `settings_router` | `server/routers/settings_router.py` | 无 | `GET/PUT /api/settings/me`、`GET /api/settings/available-models` | 用户模型 / 供应商设置 |
| `workspace_router` | `server/routers/workspace_router.py` | 无 | `GET /api/workspace/files`、`GET /api/workspace/files/{path:path}`（下载）、`PUT /api/workspace/files/{path:path}`（写入，仅限白名单文件） | 用户工作区文件浏览 / 编辑引导文件 |

**白名单编辑文件**（`workspace_router.py:16`）：`SOUL.md`、`AGENTS.md`、`USER.md`、`TOOLS.md`。

---

#### 12.1.3 认证中间件

**实现**：`backend/nanoresearch/server/middleware/auth.py`

```
OAuth2PasswordBearer(tokenUrl="/api/auth/token")   # auth.py:8
                                ↓
get_current_user(token: str = Depends(oauth2_scheme))  # auth.py:11
    └── verify_token(token) → uid: str              # 调 nanoresearch.auth.jwt
```

- 所有受保护端点注入 `Depends(get_current_user)` 获取 `uid`。
- FastAPI 通过 `oauth2_scheme` 自动从请求 `Authorization: Bearer <token>` 头提取令牌。
- `verify_token` 校验 JWT 签名 + 过期，失败抛 `HTTPException(401)`。
- `POST /api/auth/token` 和 `GET /api/rag/images`（ticket 验证）不经过此 dependency。

---

#### 12.1.4 SSE run 事件流

**端点**：`GET /api/runs/{run_id}/events`（`chat_router.py:326-373`）

职责：将 worker 进程写入 Redis Stream 的执行事件实时推送到客户端。

```
客户端                    API 服务器                       Redis
  ├─ GET /events?last_id=0-0  ──────────────────────────────────
  │                             while True:
  │                               xread_next(run_events:{id}, cursor, timeout=5s)
  │  ← data: {"type":"message_delta","chunk":"..."}\n\n
  │  ← data: {"type":"tool_hint","content":"..."}\n\n
  │  ← data: {"type":"tool_call",...}\n\n
  │  ← data: {"type":"message_complete"}\n\n
  │  ← data: {"type":"run_end","status":"success"}\n\n
  │         _normal_exit=True; return
  │
  │  （断开连接）
  │                             finally: Redis SET cancel:{session_key} "1"
```

**关键设计取舍**：
- 使用 Redis Stream（`XREAD BLOCK`）而非 WebSocket，天然支持 `?last_id` 游标断点续传（24h 回放窗口）。
- `_normal_exit` 标志区分"正常结束"与"客户端提前断开"，避免对完成的 run 误设 cancel flag（`chat_router.py:356-367`）。
- `StreamingResponse` 头加 `X-Accel-Buffering: no` 禁用 nginx 缓冲，确保 delta 实时到达（`chat_router.py:369-373`）。
- 事件类型：`message_delta`、`tool_hint`、`tool_call`、`message_complete`、`subagent_result`、`run_end`。

---

### 12.2 前端

**职责**：提供 NanoResearch 的 Web 操作界面，覆盖对话、Agent 管理、知识库、RAG 评测、Agent 评测五大功能域。

#### 12.2.1 技术栈

| 库 | 版本 | 用途 |
|----|------|------|
| Vue 3 | 3.4.x | 渐进式框架，Composition API + `<script setup>` |
| Vite | 5.2.x | 构建工具，`dev` / `build` / `preview` |
| Pinia | 2.1.x | 状态管理 |
| vue-router | 4.3.x | 客户端路由（History 模式） |
| Ant Design Vue | 4.2.x | UI 组件库 |
| marked | 18.0.x | Markdown 渲染（消息内容） |
| vue-pdf-embed | 2.1.x | PDF 文档预览 |

入口：`web/src/main.js:1-13`，`createApp(App)` → `use(createPinia())` → `use(router)` → `use(Antd)` → `mount('#app')`。

---

#### 12.2.2 路由表

`web/src/router/index.js`（History 模式，`beforeEach` 守卫验证 `userStore.isLoggedIn`）：

| 路径 | 视图 | 认证 |
|------|------|------|
| `/login` | `LoginView.vue` | 不需要 |
| `/chat` | `ChatView.vue` | ✓ |
| `/chat/:id` | `ChatView.vue` | ✓ |
| `/agents` | `AgentsView.vue` | ✓ |
| `/agents/:id` | `AgentDetailView.vue` | ✓ |
| `/runs/:id` | `RunDetailView.vue` | ✓ |
| `/knowledge` | `KnowledgeView.vue` | ✓ |
| `/knowledge/:id` | `KnowledgeDetailView.vue` | ✓ |
| `/knowledge/:id/eval` | redirect → `/knowledge/:id` | — |
| `/eval/agent` | `AgentEvalView.vue` | ✓ |
| `/` | redirect → `/chat` | — |

守卫逻辑（`index.js:23-31`）：未登录访问受保护路由 → 跳 `/login?redirect=...`；已登录访问 `/login` → 跳 `/chat`。

---

#### 12.2.3 布局：AppLayout

`web/src/layouts/AppLayout.vue` 是所有受保护视图的外壳，用 `<slot />` 注入页面内容。

```
┌─────────────────────────────────────────────────────────┐
│ a-layout-sider (dark, width=220, collapsible)           │
│  ├─ logo "Nano Research" / "NR"                         │
│  ├─ a-menu                                              │
│  │   ├─ /chat      对话                                 │
│  │   ├─ /agents    Agent                                │
│  │   ├─ /knowledge 知识库                               │
│  │   └─ /eval/agent 评测                                │
│  └─ sider-footer                                        │
│      ├─ [系统设置] → a-drawer（供应商 / 引导文件）       │
│      └─ [退出]    → userStore.logout() + /login         │
├─────────────────────────────────────────────────────────┤
│ a-layout-content                                        │
│  └─ <slot />   ← 各页面主体                            │
└─────────────────────────────────────────────────────────┘
```

系统设置 Drawer（`AppLayout.vue:47-198`）含两个 Tab：
- **API 供应商**：provider 列表 CRUD + 模型用途分配（chat / ingestion_llm / embedding / vision / eval_generator / eval_evaluator）+ Base 模型。
- **引导文件**：读写 SOUL.md / AGENTS.md / USER.md / TOOLS.md，通过 `apis/workspace.js` 调 `PUT /api/workspace/files/{name}`。

---

#### 12.2.4 Pinia Stores

| Store | 文件 | 主要状态 | 锚点 |
|-------|------|---------|------|
| `useUserStore` | `stores/user.js:4` | `token`（持久化 `nr_token`）、`uid`（`nr_uid`）、`isLoggedIn`（computed） | `getAuthHeaders()` 返回 `{ Authorization: "Bearer <token>" }`（`user.js:10`） |
| `useChatStore` | `stores/chat.js:6` | `conversations`、`messages`、`currentConvId`、`streaming`（bool）、`streamingText`（增量文本） | `sendMessage` → `createRun`（`chat.js:84`）；`appendDelta` 增量追加（`chat.js:92`）；`finalizeStream` 写入 messages 列表（`chat.js:96`） |
| `useAgentStore` | `stores/agent.js:5` | `agents`、`current`、`skills`、`loading` | `fetchSkills` 调 `GET /api/skills`（`agent.js:22`） |
| `useKnowledgeStore` | `stores/knowledge.js:11` | `kbs`、`current`、`documents`、`chunks`、`evalRuns`、`loading` | 直接透传 `apis/knowledge.js` 调用（`knowledge.js:83-88`） |
| `useSettingsStore` | `stores/settings.js:18` | `providers`、`roles`（6 roles）、`baseModel`、ragas 三模型、`loading` | `allModelOptions` computed 汇总所有 provider 模型（`settings.js:28`）；`coverage` 判断 chat / embedding 是否配置（`settings.js:34`） |

---

#### 12.2.5 apis 层

`web/src/apis/base.js` 是所有 API 调用的底层。

**核心设计**（`base.js:4-60`）：
- 使用原生 `fetch`，无 axios，无 `baseURL` 配置——所有 URL 为相对路径（同源，由 Vite 或 nginx 反代）。
- `requiresAuth=true` 时自动注入 `userStore.getAuthHeaders()` → `Authorization: Bearer <token>`。
- `FormData` body 不自动加 `Content-Type: application/json`（`base.js:12`），避免 multipart 边界丢失。
- 错误处理：422 `missing_provider` → `message.error()` toast + 继续抛出（`base.js:35-39`）；401 → `userStore.logout()` + 1s 后跳转 `/login`（`base.js:42-48`）。
- 导出 `apiGet` / `apiPost` / `apiPut` / `apiDelete` 快捷封装。

**APIs 模块与后端 Router 对应**：

| apis 模块 | 后端 Router | 主要调用 |
|-----------|------------|---------|
| `apis/auth.js` | `main.py` 内联 | `POST /api/auth/token`、`GET /api/auth/me` |
| `apis/conversations.js` | `chat_router` | conversations CRUD + messages + agent-override |
| `apis/runs.js` | `chat_router` | `POST /api/runs`、`GET /api/runs/{id}` |
| `apis/agents.js` | `agent_router` | agents CRUD + skills + KB 绑定 |
| `apis/knowledge.js` | `knowledge_router` + `eval_router` | KB CRUD + 文档 + 检索测试 + RAG 评测 |
| `apis/agentEval.js` | `agent_eval_router` | 快照 / badcase / 评测运行 / 优化提案 |
| `apis/settings.js` | `settings_router` | `GET/PUT /api/settings/me` |
| `apis/workspace.js` | `workspace_router` | 引导文件读写 |

---

#### 12.2.6 useRunStream SSE 时序

`web/src/composables/useRunStream.js:3`，用原生 `ReadableStream` 订阅 SSE 事件。

**ASCII 时序图**：

```
ChatView                useRunStream             后端 SSE 端点              Redis Stream
   │                        │                         │                          │
   │  runStream.start(runId)│                         │                          │
   │───────────────────────►│                         │                          │
   │                        │  fetch GET /api/runs    │                          │
   │                        │  /{id}/events           │                          │
   │                        │ +Bearer token           │                          │
   │                        │────────────────────────►│                          │
   │                        │                         │  XREAD BLOCK 5s          │
   │                        │                         │─────────────────────────►│
   │                        │ ◄── data: message_delta │◄─────────────────────────│
   │  onDelta(chunk)        │                         │   worker 写入事件        │
   │◄───────────────────────│                         │                          │
   │  chatStore.appendDelta │                         │                          │
   │                        │ ◄── data: tool_hint     │                          │
   │  onToolHint(hint)      │                         │                          │
   │◄───────────────────────│                         │                          │
   │  toolHint.value = hint │                         │                          │
   │                        │ ◄── data: tool_call     │                          │
   │  onToolCall(tc)        │                         │                          │
   │◄───────────────────────│                         │                          │
   │  pendingToolCalls.push │                         │                          │
   │                        │ ◄── data: message_complete                         │
   │  onMessageComplete()   │                         │                          │
   │◄───────────────────────│                         │                          │
   │  finalizeStream(tcs)   │                         │                          │
   │                        │ ◄── data: run_end       │                          │
   │  onEnd("success")      │                         │  _normal_exit=True       │
   │◄───────────────────────│                         │  return (不设 cancel)    │
   │  await selectConv(id)  │                         │                          │
   │  (从 DB 重载为正典)     │                         │                          │
```

**SSE 解析器**（`useRunStream.js:26-62`）：
- `buffer` 积累字节块，`lines = buffer.split('\n')`，末尾不完整行留入 buffer。
- 空行 → `dispatch()`：将 `dataLines` 拼接为 JSON 字符串 → `JSON.parse` → 事件分发。
- `data:` 前缀剥离 `line.slice(5).trimStart()`。
- AbortController 控制取消（`stop()` 调用或组件卸载）。

**设计取舍**：选用原生 `fetch` + `ReadableStream` 而非 `EventSource`，原因是 `EventSource` 不支持自定义请求头（无法携带 Bearer token）。

---

#### 12.2.7 关键视图数据流

**ChatView**（`web/src/views/ChatView.vue`）

组件树：
```
AppLayout
  └── ChatView
        ├── sidebar: 会话列表（chatStore.conversations）
        ├── a-modal: 新建对话选 Agent（agentStore.agents）
        ├── chat-main
        │     ├── agent-bar: 显示当前 Agent + 模型 + 能力标签
        │     ├── MessageList: 渲染 chatStore.messages + streamingText
        │     └── input-area: ragMode 切换（simple/agentic）+ 输入框 + 发送
        ├── detail-panel（可选）: ConversationDetailPanel
        └── workspace-panel（可选）: WorkspaceFiles
```

**发送消息数据流**（`ChatView.vue:433-506`）：

```
用户点击"发送"
  → handleSend()                       # ChatView.vue:433
  → chatStore.messages.push(userMsg)   # 乐观更新，立即显示
  → chatStore.sendMessage(text, mode)  # chat.js:84
      → createRun(convId, text, ...)   # POST /api/runs → 201 { run_id }
  → connectStream(run_id, convId)      # ChatView.vue:463
      → runStream.start(run_id, {
          onDelta   → chatStore.appendDelta(chunk)        # 增量追加流式文本
          onToolHint→ toolHint.value = hint               # 工具进度提示
          onToolCall→ pendingToolCalls.push(tc)           # 收集工具调用
          onMessageComplete → chatStore.finalizeStream(pendingToolCalls)  # 写入消息列表
          onEnd     → chatStore.selectConversation(convId)# 从 DB 重载为正典
        })
```

**中断重连**：切换会话时 `pendingRuns[convId]` 记录中断状态；切回时（`handleSelect` `ChatView.vue:408`）若 DB 消息数未增长则重新调用 `connectStream`，后端 Redis Stream 支持 `last_id` 断点续传。

---

**AgentEvalView**（`web/src/views/AgentEvalView.vue`）

职责：Agent 评测闭环操作界面。

组件结构：
```
AppLayout
  └── AgentEvalView
        ├── page-header: 统计卡片（total_snapshots / total_badcases / pending_review）
        │     ← GET /api/eval/agent/stats
        └── a-tabs
              ├── 运行快照: 分页表格，筛选 run_status / is_badcase
              │     ← GET /api/eval/agent/snapshots
              ├── Badcase: 分页表格 + 标注 / promote / classify-batch
              ├── 测试集: 增删 test case
              ├── 评测运行: 触发 eval-run + 进度 + 回归分析
              └── 优化提案: optimize 列表 + 审批 / 应用 / 回滚
```

所有数据调用通过 `apis/agentEval.js` → `agent_eval_router`。

---

**RunDetailView + RunTimeline**（`web/src/views/RunDetailView.vue:1-32`、`web/src/components/RunTimeline.vue:1-170`）

`RunDetailView` 是薄壳——只从 `route.params.id` 提取 `run-id` 并渲染 `<run-timeline>`。

`RunTimeline` 数据流：
```
onMounted / watch(runId)
  → getRun(runId)           # GET /api/runs/{id}
  → getAgent(run.agent_id)  # GET /api/agents/{id}（异步获取 agent 名）

渲染：
  ├── run-header: status tag + agent link + model + 耗时
  ├── token-card: input/output/cache 比例条（tokens_used）
  ├── a-collapse: 工具调用列表（run.tool_calls），每项展示 input/output JSON
  ├── artifact-list: 产出文件列表
  └── a-alert: error_message（如有）
```

`RunTimeline` 是纯展示组件，读取已持久化的 run 数据，不订阅 SSE。实时进度在 `ChatView` 的 `MessageList` 中展示；run 结束后从 DB 读取的静态视图是 `RunTimeline`。

---

### 12.3 前后端契约

#### 12.3.1 REST 端点清单（前端 apis → 后端 router）

**认证**

| 前端调用 | 方法 | 路径 | 后端位置 |
|---------|------|------|---------|
| `loginApi()` | `POST` | `/api/auth/token` | `main.py:106`（form-data） |
| `getMeApi()` | `GET` | `/api/auth/me` | `main.py:118` |

**对话与 Run**

| 前端调用 | 方法 | 路径 | 后端位置 |
|---------|------|------|---------|
| `listConversations()` | `GET` | `/api/conversations` | `chat_router.py:52` |
| `createConversation()` | `POST` | `/api/conversations` | `chat_router.py:84` |
| `getConversation(id)` | `GET` | `/api/conversations/{id}` | `chat_router.py:113` |
| `getMessages(id)` | `GET` | `/api/conversations/{id}/messages` | `chat_router.py:131` |
| `deleteConversation(id)` | `DELETE` | `/api/conversations/{id}` | `chat_router.py:155` |
| `updateAgentOverride(id)` | `PUT` | `/api/conversations/{id}/agent-override` | `chat_router.py:167` |
| `createRun()` | `POST` | `/api/runs` | `chat_router.py:217` |
| `getRun(id)` | `GET` | `/api/runs/{id}` | `chat_router.py:316` |
| **SSE** `GET /api/runs/{id}/events` | `GET` | `/api/runs/{id}/events` | `chat_router.py:326` |

**Agent**

| 前端调用 | 方法 | 路径 | 后端位置 |
|---------|------|------|---------|
| `listSkills()` | `GET` | `/api/skills` | `agent_router.py:61` |
| `listAgents()` | `GET` | `/api/agents` | `agent_router.py:101` |
| `createAgent()` | `POST` | `/api/agents` | `agent_router.py:79` |
| `getAgent(id)` | `GET` | `/api/agents/{id}` | `agent_router.py:116` |
| `updateAgent(id)` | `PUT` | `/api/agents/{id}` | `agent_router.py:128` |
| `deleteAgent(id)` | `DELETE` | `/api/agents/{id}` | `agent_router.py:186` |

**知识库**

| 前端调用 | 方法 | 路径 | 后端位置 |
|---------|------|------|---------|
| `listKnowledge()` | `GET` | `/api/knowledge` | `knowledge_router.py:95` |
| `createKnowledge()` | `POST` | `/api/knowledge` | `knowledge_router.py:101` |
| `uploadDocument(kbId, file)` | `POST` | `/api/knowledge/{kb_id}/documents` | `knowledge_router.py:145` |
| `testQuery(kbId)` | `POST` | `/api/knowledge/{kb_id}/query/test` | `knowledge_router.py:401` |
| `createEvalRun(kbId, ...)` | `POST` | `/api/eval/{kb_id}/runs` | `eval_router.py:261` |

**设置 / 工作区**

| 前端调用 | 方法 | 路径 | 后端位置 |
|---------|------|------|---------|
| `getMySettings()` | `GET` | `/api/settings/me` | `settings_router.py:124` |
| `updateMySettings()` | `PUT` | `/api/settings/me` | `settings_router.py:130` |
| `getWorkspaceFile(name)` | `GET` | `/api/workspace/files/{name}` | `workspace_router.py:84` |
| `updateWorkspaceFile(name)` | `PUT` | `/api/workspace/files/{name}` | `workspace_router.py:68` |

---

#### 12.3.2 鉴权流

```
1. 登录
   前端 LoginView
   → POST /api/auth/token  (Content-Type: application/x-www-form-urlencoded)
     body: username=<uid>&password=<pwd>
   ← 200 { access_token: "eyJ...", token_type: "bearer" }
   → userStore.setToken(access_token, uid)
     localStorage.nr_token = access_token
     localStorage.nr_uid   = uid

2. 后续请求
   apiRequest(url, ..., requiresAuth=true)        # base.js:15
   → headers["Authorization"] = "Bearer " + token

3. 后端验证
   OAuth2PasswordBearer 提取 Bearer token         # auth.py:8
   → verify_token(token) → uid                   # auth.py:12
   （失败抛 HTTPException 401）

4. Token 过期
   后端 401 → base.js:42
   → userStore.logout()                          # 清空 localStorage
   → message.error("登录已过期")
   → setTimeout 1s → window.location.href="/login"

5. SSE 鉴权（useRunStream）
   fetch("/api/runs/{id}/events", {
     headers: userStore.getAuthHeaders()          # useRunStream.js:14
   })
   （SSE 不支持 EventSource 自定义头，故用 fetch ReadableStream）
```

**RAG 图片鉴权**（短令牌 ticket 机制，`knowledge_router.py:260-293`）：
- 前端先调 `POST /api/rag/image-ticket`（Bearer 认证）获取 60 秒有效 ticket。
- 图片 `<img>` src 携带 `?ticket=<ticket>`，不需要在 URL 放 Bearer token。
- `GET /api/rag/images?path=...&ticket=...` 校验 ticket → 验证 KB 归属 → 返回文件。


## Ch13 评测与优化闭环 Eval

本章描述 NanoResearch 的评测与自优化闭环。系统以**事件驱动**方式工作：每条生产 Agent 运行结束时自动采样快照，
检出坏案例，进而触发语义分类、沙箱验证、优化提案和数据飞轮生成。全流程无 cron 无定时器，只有钩子和后台任务。

---

### 13.1 闭环全景数据流

```
生产运行结束
      │
      ▼
_maybe_save_snapshot()          ← loop.py:579  [事件钩子，每条消息处理后触发]
      │  采样（20% 成功 + 全量失败 + 高危词全量）
      │  构建 RunSnapshotData（snapshot.py:RunSnapshotCollector.build）
      ▼
RunSnapshotCollector.build()    ← eval/snapshot.py:92
      │  工具链、LLM 调用、token 统计、ttft_ms 均在此聚合
      ▼
BadcaseDetector.detect()        ← eval/badcase_detector.py:73
      │  硬规则：run_failure / token_spike / excessive_retries / low_quality / tool_skip
      │  → mark_badcase() 写入 agent_run_snapshots
      ▼
[异步 API 调用 POST /badcases/classify-batch]
      │
      ▼
BadcaseClassifier.classify()    ← eval/badcase_classifier.py:120
      │  规则快捷路径 → LLM CoT → ClassifyResult
      │  输出 (layer, target_kind, target_id) 结构指针
      ▼
ContextDiagnoser.diagnose()     ← eval/context_diagnoser.py:20  [可选，context 层专项]
      │  重放检索，区分 kb_gap vs transient
      ▼
[异步 API 调用 POST /optimize]
      │
      ▼
OptimizationAgent.generate_proposals()  ← eval/optimizer.py:121
      │  1. 读取基线 baseline
      │  2. LLM 生成 3-5 候选
      │  3. 每候选在 fix_set + health_set 各评分 N=3 次（ScoreSample）
      │  4. σ-weighted gate 决策
      │  5. 持久化 OptimizationProposal
      ▼
SandboxedToolRegistry（replay / side_effect_only）  ← eval/sandbox.py:47
  +  RuleEvaluator.evaluate()            ← eval/evaluator.py:99
      │  工具调用回放 + keyword/tool_skip/contextual_recall 评分
      ▼
LLMJudge.score_with_consistency()       ← eval/judge.py:81  [可选，use_judge=True]
      │  3 次并发打分取中位数（G-Eval 格式，1-5 → 0-1 归一化）
      ▼
DataFlywheel.check_trigger() + generate_cases_from_badcases()  ← eval/data_flywheel.py:54,80
      │  失败率超阈值时 LLM 生成新用例，status="pending_review"
      ▼
RegressionDetector.compare()    ← eval/regression_detector.py:27
      │  对比当前 run 与 baseline run，逐维 delta 检查
      ▼
agent_run_snapshots / optimization_proposals / agent_test_cases（DB）
```

---

### 13.2 各组件职责

#### 13.2.1 RunSnapshotCollector — 运行采集器

**职责**：无 await 点地内联采集工具链、LLM 调用、token、ttft，组装 `RunSnapshotData`。

主要类/方法：
- `RunSnapshotData`（dataclass）`backend/nanoresearch/eval/snapshot.py:13`
- `RunSnapshotCollector.on_tool_start/on_tool_end/on_llm_end/build` `snapshot.py:59–120`

`context_trace` 字段在 `build()` 可选注入（`snapshot.py:97`），存储 Phase 0 上下文装配决策
（`memory_budget_tokens`、`fragment_ids`、`history_actual_chars` 等），供 `BadcaseClassifier` 使用。

工具结果截断至 `_MAX_RESULT_CHARS=2000`（`snapshot.py:5`），防大结果撑爆 DB。

#### 13.2.2 BadcaseDetector — 规则检出

**职责**：基于快照启发式检出坏案例，不调 LLM。

主类：`BadcaseDetector` `backend/nanoresearch/eval/badcase_detector.py:54`
核心方法：`detect(snapshot, scores, passed, tc)` `badcase_detector.py:73`

检出逻辑（优先级从高到低）：

| 检查项 | 触发条件 | 类别 |
|--------|----------|------|
| 运行失败 | `run_status in (failed/timeout/max_iterations)` | `run_failure` |
| Token 尖刺 | `total_tokens > p95_tokens`（p95 为 None 时禁用） | `token_spike` |
| 重试过多 | `retry_count > max_retries`（默认 3） | `excessive_retries` |
| 质量失败 | `passed=False` 且有维度 < 0.6 | `low_quality`（可被细化） |
| 工具跳过 | `tool_skip == 0.0` 且有 `expected_tools` | `tool_skip` |

质量失败通过 `_refine_failure_category()` 细化（`badcase_detector.py:26`）：
- `contextual_recall < 0.3` → `retrieval_failure`
- `faithfulness_score < 0.4` → `hallucination`
- `contextual_recall ≥ 0.5 且 task_completion < 0.5` → `reasoning_failure`

`detect()` 返回 `list[tuple[str, str]]`（trigger_source, category），可同时返回质量失败和工具跳过两项；
质量失败在前，`mark_badcase` 以第一项为主分类（`test_runner.py:281`）。

#### 13.2.3 BadcaseClassifier — 语义根因分类

**职责**：输出结构化根因指针 `(layer, target_kind, target_id)`，区分可修复层（Context/Tool）与
诊断专用层（Memory/Recovery）。

主类：`BadcaseClassifier` `backend/nanoresearch/eval/badcase_classifier.py:115`
核心方法：`classify(snapshot)` `badcase_classifier.py:120`

分类路径：
1. **规则快捷路径** `_rule_based_root_cause()`（`badcase_classifier.py:151`）——4 条高置信规则：
   - `error=True` 工具 → `Tool/tool_impl`
   - 运行失败 + 无工具调用 → `Context/system_prompt`
   - `contextual_recall ≥ 0.5` 且 `task_completion < 0.5` → `Context/system_prompt`
   - 输入 token ≥ 85% 上限（`_HIGH_TOKEN_RATIO=0.85`，`_MODEL_TOKEN_LIMIT=128_000`）→ `layer=None`，人审
2. **LLM CoT 路径**：规则无命中则 LLM 先 CoT 推理再输出 JSON，`temperature=0.0`，最多 512 token
3. **降级**：解析失败返回 `confidence="low"`，`layer=None`

`FIXABLE_LAYERS = {"Context", "Tool"}`，只有这两层可以进入后续自动优化链（`badcase_classifier.py:55`）。
`DIAGNOSIS_ONLY_LAYERS = {"Memory", "Recovery"}` 只产出指针，无自动修复（`badcase_classifier.py:56`）。

LLM 响应解析从末尾往前扫描 `\{[^{}]*\}` 避免 CoT 中的 JSON 片段干扰（`badcase_classifier.py:250`）。

**新用户安全设计**：`fragment_ids=[]` 既可能是新用户无记忆（正常），也可能是检索策略失败。
规则层故意不处理此模式，委托给 LLM 结合完整 `context_trace` 判断（`badcase_classifier.py:10–17`）。

#### 13.2.4 LLMJudge — LLM 作裁判

**职责**：多维度 G-Eval 打分，为优化候选提供质量信号，并支持人工标注校准。

主类：`LLMJudge` `backend/nanoresearch/eval/judge.py:44`
打分维度：`tool_rationality`、`task_completion`、`response_logic`、`faithfulness_score`
（多轮时加 `multi_turn_coherence`）

**打分流程**（`judge.py:55–79`）：
1. 组装 Prompt（历史对话 + 用户输入 + 工具链 + 最终回复 + 期望关键词）
2. 调 LLM，`temperature=0.0`，`max_tokens=4096`
3. 解析 `{"dimensions": {"dim": {"score": 1-5, "reason": "..."}}}` G-Eval 格式
4. 归一化：`score_normalized = (raw_score - 1) / 4`，映射到 `[0.0, 1.0]`（`judge.py:218`）

**一致性评分** `score_with_consistency()`（`judge.py:81–117`）：
- 并发运行 `runs=3` 次（`EvalRunConfig.judge_consistency_runs=3`，`test_runner.py:29`）
- 取各维度**中位数**作为共识分
- 当任意维度 `max-min > 0.25`（即原始 1 分差）时 `low_confidence=True`（`judge.py:113`）

**校准** `calibrate()`（`judge.py:119–146`）：
- 对有 `human_score` 的用例计算 `MAD`（Mean Absolute Deviation）
- `MAD ≤ 0.15` 则校准通过（`judge.py:130`）；失败时记日志但不阻断评分

工具链和回复均截断（`_MAX_TOOL_CHAIN_CHARS=1500`，`_MAX_RESPONSE_CHARS=1000`）防 token 超限（`judge.py:40–41`）。

#### 13.2.5 RuleEvaluator — 规则评估器

**职责**：确定性规则评分，不调 LLM；是优化候选打分和批量测试的核心评分器。

主类：`RuleEvaluator` `backend/nanoresearch/eval/evaluator.py:82`
核心方法：`evaluate(snapshot, test_case)` → `dict[str, float]`（`evaluator.py:99`）

评分维度：

| 维度 | 评分逻辑 |
|------|----------|
| `token_budget` | `total ≤ budget → 1.0`，否则 `0.0`（仅告知，不影响 pass） |
| `tool_skip` | 全命中=1.0，部分=0.7，全漏=0.0；参数错降至 0.5 |
| `keyword_coverage` | 字符串匹配 + 双语词典 `BILINGUAL_MAP` + 可选语义 fallback（cosine ≥ 0.65） |
| `contextual_recall` | 工具返回内容中含关键词的比例（字符串匹配） |

**pass/fail gate**（`evaluator.py:267`）：仅看 `keyword_coverage`，动态阈值：
- ≤4 关键词 → 0.8；≤7 → 0.6；>7 → 0.5

`tool_skip` 和 `contextual_recall` 记录为失败维度但不决定 pass/fail。

语义匹配：当提供 `embedding_fn` 时，对字符串未命中关键词执行批量语义相似度检索
（`_EMBED_BATCH_SIZE=10`，`_SEMANTIC_KW_THRESHOLD=0.65`，`evaluator.py:39`）；
`web_search` 参数也用语义匹配验证 query 与 user_input 相关性（阈值 0.6，`evaluator.py:40`）。

#### 13.2.6 ContextDiagnoser — 上下文根因诊断

**职责**：对 `retrieval_failure` 类坏案例重放检索，区分「知识库确无此内容（kb_gap）」
与「检索策略失败但内容存在（transient）」。

主类：`ContextDiagnoser` `backend/nanoresearch/eval/context_diagnoser.py:13`
核心方法：`diagnose(snapshot, kb_list, session_factory, top_k=5)` `context_diagnoser.py:20`

工作流：
1. 从工具调用链找空结果条目（`_is_empty_output(e.get("result"))`）
2. 提取 query 参数（`_QUERY_KEYS = ["query", "q", "search_query", "text", "keyword"]`，`diagnoser.py:10`）
3. 对每个 KB 重建完整 `HybridSearch` 栈（dense + sparse + RRF fusion）重放搜索
4. 当前 KB 能检出 `now_count > 0` → `verdict = "transient"`；全 KB 都空 → `verdict = "kb_gap"`

`transient` 案例（知识存在但检索未命中）应进优化链；`kb_gap` 案例（内容缺失）需人工补充知识库。

通过 REST API 触发：`POST /api/eval/agent/badcases/diagnose`（`agent_eval_router.py:1067`），非自动。

#### 13.2.7 OptimizationAgent — 优化提案生成

**职责**：以双集合（fix_set + health_set）对比评分驱动，生成并 gate 过滤优化提案。

主类：`OptimizationAgent` `backend/nanoresearch/eval/optimizer.py:107`
核心方法：`generate_proposals(target, representative_snapshots, fix_test_cases, health_test_cases)` `optimizer.py:121`

**目标对象（TunableTextObject）**：
- `PersonaObject`（`tunable.py:191`）：操作 `agents.persona` 字段，`kind="system_prompt"`
- `ToolDescriptionObject`（`tunable.py:277`）：操作 `agents.tools_config[name].description`，`kind="tool_description"`

两种 kind 均实现 `read/apply/generate_candidates/get_current_version/rollback` 接口（`tunable.py:67`）。
`apply()` 写版本记录 + 更新 DB，`rollback()` 写新版本行（历史行不可变，`tunable.py:254`）。

**双集合设计**（Phase 2 / Phase 5 约束）：
- `fix_test_cases`：由触发本次优化的坏案例衍生（运行时动态）
- `health_test_cases`：独立构建的健康集（`set_kind="health"`，≥50 用例）
- 两集合均必须非空，否则 `ValueError`（`optimizer.py:142–151`）
- 基线与所有候选共享**同一** Python 对象列表，保证 delta 可比（`optimizer.py:139`）

**评分流程**（`_score_candidate_set()`，`optimizer.py:431`）：
每个（候选, 测试用例）对 `_SCORE_REPEAT_N=3` 次重复评分（`optimizer.py:99`），
收集 `observations: list[float]`，`ScoreSample.from_observations()` 计算 (mean, std, n)。

`system_prompt` 候选：`mode="replay"` 沙箱（严格回放），`SandboxReplayError` 表示 cache miss。
`tool_description` 候选：`mode="side_effect_only"`，description_overrides 注入候选描述，
query 工具 cache miss → passthrough live call，side_effect 工具 cache miss → 拦截并抛出（`sandbox.py:182–196`）。

#### 13.2.8 ScoreSample — σ 采样结构

**职责**：N 次重复评分的 (mean, std, n) 三元组，供 σ-weighted gate 使用。

`ScoreSample`（frozen dataclass）`backend/nanoresearch/eval/score_sample.py:13`
`from_observations()`：Bessel 校正（`n-1`）样本标准差（`score_sample.py:25`）

#### 13.2.9 SandboxedToolRegistry — 沙箱工具注册表

**职责**：为评分提供确定性工具回放，防止副作用，并跟踪 fuzzy match 比率。

主类：`SandboxedToolRegistry` `backend/nanoresearch/eval/sandbox.py:47`

四种模式：
- `passthrough`：直通真实工具，不录制
- `record`：直通并录制结果（key=`{name}:{normalized_params_json}`，上限 `_MAX_ENTRIES=200`，`sandbox.py:15–16`）
- `replay`：严格按 key 返回录制结果；cache miss → `SandboxReplayError`（`sandbox.py:43`）
- `side_effect_only`：录制优先；miss 时 query 工具 passthrough，side_effect 工具拦截

**fuzzy match**（`side_effect_only` 专用）：key 精确匹配失败后，对所有录制 key 进行
字符串 strip + dict 排序后的 normalized 比较（`_normalize_params_for_fuzzy()`，`sandbox.py:28`）；
命中计入 `_fuzzy_hits`，最终 `fuzzy_match_ratio = _fuzzy_hits / _total_executions`（`sandbox.py:222`）。

`fuzzy_match_ratio > _FUZZY_UNRELIABLE_THRESHOLD=0.30` 时提案状态置为 `signal_unreliable`（`optimizer.py:59,283`）。
`description_overrides` 在 `get_definitions()` 时替换描述，模型看到候选描述（`sandbox.py:94–105`）。

#### 13.2.10 σ-weighted Gate（B2）

**核心算法**，位于 `optimizer.py:57–97`。

目标：区分真实改进与基线噪声包络内的随机波动。

常量：
- `_GATE_SIGMA_K = 1.96`（95% 单侧置信度 z 值，`optimizer.py:58`）
- `_SCORE_REPEAT_N = 3`（每用例评分次数，`optimizer.py:99`）
- `_FUZZY_UNRELIABLE_THRESHOLD = 0.30`（`optimizer.py:59`）

**伪代码**：

```
对候选 C，fix_set 各用例 i：
  fix_delta_i = candidate_scores[i].mean - baseline_scores[i].mean
  fix_sigma_i = sqrt(σ²_base/n_base + σ²_cand/n_cand)  # 两独立均值差的标准误

fix_delta_mean = mean(fix_delta_i for i in cases)
fix_sigma_combined = sqrt(sum(σ_i²)) / len(cases)  # 保守聚合

# gate 决策（health 同理）：
if fix_delta_mean < _GATE_SIGMA_K * fix_sigma_combined:
    gate_decision = "rejected"  # within_noise_envelope
elif health_delta_mean < -_GATE_SIGMA_K * health_sigma_combined:
    gate_decision = "rejected"  # health_regression
else:
    gate_decision = "approved"  # passes_sigma_gate
```

`gate_status` 字段向后兼容（`"pending_approval"` = approved，`"rejected_by_gate"` = rejected）；
新增 `gate_decision / gate_reason / sigma_combined / delta_mean / threshold` JSONB 字段供分析。

所有候选均被 gate 拒绝 → `proposal_status = "gate_all_rejected"`（`optimizer.py:281`）。

#### 13.2.11 Snapshot — 版本快照

**职责**：管理 TunableTextObject 的版本历史，支持回滚。

快照写入通过 `TunableTextObject.apply()` 调用 `eval_repo.create_tunable_version()`；
`rollback(version_id)` 读取历史版本内容，调 `apply()` 写新行（历史行不可变）（`tunable.py:254–258`）。

`get_current_version()` 返回当前 active 版本 UUID，在 `generate_proposals()` 调用前读取作为
`baseline_version_id` 存入 `OptimizationProposal`（`optimizer.py:155–157`）。

#### 13.2.12 DataFlywheel — 数据飞轮

**职责**：批量测试结束后，从高频失败模式自动生成新测试用例，回流训练集。

主类：`DataFlywheel` `backend/nanoresearch/eval/data_flywheel.py:47`
后处理函数：`run_flywheel()` `data_flywheel.py:142`（在 `TestRunner.run_all()` 末尾调用，`test_runner.py:414`）

**触发逻辑**（`check_trigger()`，`data_flywheel.py:54`）：

```
triggered = [category
             for category, threshold in config.flywheel_thresholds.items()
             if failure_stats[category] / total > threshold]
```

默认阈值（`EvalRunConfig`，`test_runner.py:35`）：
- `retrieval_failure` > 20%
- `hallucination` > 15%
- `reasoning_failure` > 25%
- `tool_skip` > 30%

超阈值时：`generate_cases_from_badcases()` 每类生成 3 个新用例（LLM，`temperature=0.7`，`data_flywheel.py:80`），
另可生成 `flywheel_adversarial_per_run`（默认 0）个红队对抗用例（`temperature=0.9`，`data_flywheel.py:123`）。
新用例 `status="pending_review"` 入库等待人审（`data_flywheel.py:178`）。

对抗类型包括：意图模糊、边界输入、指令注入、前提错误、幻觉诱导（`data_flywheel.py:32`）。

#### 13.2.13 RegressionDetector — 回归检测

**职责**：对比当前 eval run 与指定基线，逐维度检查 delta 是否低于阈值。

主类：`RegressionDetector` `backend/nanoresearch/eval/regression_detector.py:19`
核心方法：`compare(baseline_scores, current_scores)` → `(has_regression, diffs)` `regression_detector.py:27`

默认阈值（`DEFAULT_THRESHOLDS`，`regression_detector.py:5`）：

| 维度 | 回归阈值（delta < 触发） |
|------|--------------------------|
| `keyword_coverage` | -0.08 |
| `tool_rationality` | -0.10 |
| `task_completion` | -0.10 |
| `response_logic` | -0.10 |
| `tool_hit_rate` | -0.05 |

`token_budget` 为 0/1 值，在 `SKIP_DIMS` 中跳过（`regression_detector.py:16`）。
在 `TestRunner.run_all()` 中，当 `config.baseline_run_id` 非空时，run 结束后自动比对（`test_runner.py:390`）。
回归结果写入 `eval_run.has_regression` 和 `regression_diffs` JSONB。
REST API `GET /eval-runs/{run_id}/regression` 和 `GET /eval-runs/comparison` 供前端展示（`agent_eval_router.py:786,332`）。

#### 13.2.14 TestRunner — 批量测试执行器

**职责**：批量执行测试用例，整合评分、Judge、坏案例检测、回归、飞轮。

主类：`TestRunner` `backend/nanoresearch/eval/test_runner.py:54`
核心方法：`run_all(config, eval_run_id, uid)` `test_runner.py:83`

`EvalRunConfig` 关键参数（`test_runner.py:24`）：
- `use_judge: bool`（默认 False）
- `judge_consistency_runs: int = 3`
- `sandbox_mode: str = "record"`
- `concurrency: int = 5`
- `enable_flywheel: bool = False`
- `baseline_run_id`：触发回归检测

执行顺序：校准 Judge → 并发运行用例（信号量 `concurrency=5`）→ 聚合分数 → 回归检测 → 飞轮。
每 5 个用例刷新一次 DB 进度（`test_runner.py:329`）。

---

### 13.3 触发机制：事件驱动，无 cron

Eval 全系统没有任何 cron 或定时器，所有触发点均为事件响应：

**触发点 1（最核心）：生产消息处理后钩子**
`AgentLoop._maybe_save_snapshot()` `backend/nanoresearch/agent/loop.py:579`
- 每条 `InboundMessage` 处理后自动触发
- 失败运行**全量**保存；成功运行按 `EVAL_SAMPLING_RATE=20%` 采样
- 含高危词（`high_risk_keywords.json`：["退款","投诉","删除","注销","赔偿","紧急"]）的成功运行**绕过采样全量保存**
- `BadcaseDetector.detect()` 在同一 `_save()` 协程中同步执行（`loop.py:664`）
- 用户行为信号（重复问题 → Jaccard > 0.85，abandoned → 前轮未 success）在此合并（`loop.py:639–661`）
- 通过 `_schedule_background()` fire-and-forget，不阻塞主消息处理（`loop.py:677`）

**触发点 2：用户 dislike 按钮**
`POST /api/eval/agent/badcases/{snapshot_id}/flag` `agent_eval_router.py:103`
- 前端用户负反馈直接写 `mark_badcase(trigger="user:dislike", category="user_feedback")`

**触发点 3：按需分类批处理**
`POST /api/eval/agent/badcases/classify-batch` `agent_eval_router.py:736`
- 对未分类坏案例批量调 `BadcaseClassifier`（每次最多 50 个）
- `asyncio.create_task()` 异步执行，不阻塞请求

**触发点 4：按需上下文诊断**
`POST /api/eval/agent/badcases/diagnose` `agent_eval_router.py:1067`
- 对 `root_cause_auto="context"` 的坏案例重放检索，区分 kb_gap vs transient
- 需前端/运维手动调用

**触发点 5：按需优化**
`POST /api/eval/agent/optimize` `agent_eval_router.py:907`
- 从前端/运维触发，传入 `snapshot_ids`（坏案例集合）
- `asyncio.create_task()` 后台执行 `OptimizationAgent.generate_proposals()`

**触发点 6：批量测试（含回归+飞轮）**
`POST /api/eval/agent/eval-runs` `agent_eval_router.py:243`
- 前端/CI 触发，`asyncio.create_task(runner.run_all(...))` 后台执行
- 单次 run 结束时自动触发回归检测和数据飞轮（若配置开启）

---

### 13.4 高危词采样豁免

`high_risk_keywords.json`（`backend/nanoresearch/eval/high_risk_keywords.json:1`）：

```json
["退款", "投诉", "删除", "注销", "赔偿", "紧急"]
```

含以上词汇的成功运行绕过 `EVAL_SAMPLING_RATE` 全量录制（`loop.py:585–593`），
确保高风险对话 100% 进入 badcase 检测链路。

---

### 13.5 设计取舍与已知局限

**ScoreSample N=3 的局限**：3 次重复在高方差用例上 σ 估计不稳定，
`n=1` 时 `std=0.0`（`score_sample.py:28`），会使 σ_combined=0 → gate 退化为纯 delta 比较。
实测中 health_set ≥50 用例 + N=3 能提供足够信号，但对于单一高方差用例需人审。

**PersonaObject 范围边界**：`personas.agents.persona` 是唯一 DB 存储的可调文本段；
其余 system prompt 段（SOUL.md 结构、skills 摘要、KB 绑定、动态后缀）由 `ContextBuilder` 代码拼接，
不在此优化范围内（`tunable.py:14`）。

**ToolDescriptionObject Phase 1 约束**：评分使用 `side_effect_only` 沙箱，
`system_prompt` 通过 `ContextBuilder` 构建（无用户历史注入）确保可复现（`optimizer.py:456`）；
但 Phase 1 省略了 ContextBuilder 与 production 完全对齐（Phase 6 补全）。

**飞轮用例人审强制**：生成用例 `status="pending_review"` 不直接进入活跃测试集，
避免劣质生成用例污染 health_set——健康集质量是 gate 有效性的基础。

**回归 delta 硬编码**：`DEFAULT_THRESHOLDS` 为经验值，尚无动态阈值（optimizer 注释 `Phase 5` 中已明确
「Gate thresholds are hardcoded — no dynamic threshold」，`optimizer.py:26`）。

---

### 13.6 A1 SDD 相关设计点

代码中可见以下与 SDD 相关的明确设计约束（可在注释中核对）：

**双集合不变量（strict_replay 对应）**：
`generate_proposals()` 中 Phase 2 约束注释明确：`fix_test_cases` 和 `health_test_cases` 均必须非空
（`optimizer.py:141–151`），baseline 与候选共享同一 Python 对象（`optimizer.py:133–140`）。
这防止了「两集合对象不同导致 delta 计算无意义」的逻辑洞。

**sandbox mode 按 target.kind 分叉（execution_sanity 对应）**：
`system_prompt` 候选用 `replay` 模式（cache miss → `SandboxReplayError`，严格拒绝）；
`tool_description` 候选用 `side_effect_only` 模式（query tool passthrough，side_effect tool 拦截）。
两种模式确保候选不会因 sandbox miss 引发真实副作用（`optimizer.py:451–458`）。

**高 divergence / 放宽类候选需人审**：
- `signal_unreliable`（fuzzy_match_ratio > 0.30）的提案 `proposal_status` 置为 `signal_unreliable`，
  不自动推进（`optimizer.py:283`）
- `gate_all_rejected` 时所有候选被拒，需人工决策
- `Memory/Recovery` 层的根因分类结果只产出诊断指针，无自动修复链路，必须人审（`badcase_classifier.py:56`）
- 输入 token ≥ 85% 模型上限的案例 `layer=None`，不进入自动修复（`badcase_classifier.py:211–232`）


## Ch14 定时与心跳

### 职责概述

定时子系统由两个独立服务组成：`CronService` 提供用户可编程的任务调度能力，`HeartbeatService` 提供代理自检/任务驱动的周期性唤醒。两者均由 asyncio 驱动，不依赖系统 cron 或外部调度守护进程。

**重要区分**：这里的"cron"特指用户主动创建的定时任务（例如"每天 9:00 发送报告"）。系统内部的 eval/optimization 触发（记忆整合、评估、压缩等）完全是**事件驱动 hooks**，由消息到达或上下文状态变化触发，从不走定时器——见 memory consolidation 相关章节。

---

### 14.1 数据类型层

**文件**：`backend/nanoresearch/cron/types.py`

```
CronSchedule          # 调度定义
CronPayload           # 执行内容
CronRunRecord         # 单次执行记录
CronJobState          # 运行时状态
CronJob               # 完整任务对象
CronStore             # 持久化容器（version=1）
```

`CronSchedule`（第8行）支持三种调度模式：
- `kind="at"`：一次性绝对时间戳（`at_ms`，毫秒）
- `kind="every"`：固定间隔（`every_ms`，毫秒）
- `kind="cron"`：标准 cron 表达式（`expr`） + 可选 IANA 时区（`tz`）；时区只对 `kind="cron"` 有效（`backend/nanoresearch/cron/types.py:11-16`）

`CronPayload`（第22行）的 `kind` 字段区分 `"system_event"` 和 `"agent_turn"`（默认），`deliver=True` 时将执行结果推送到指定 channel/to（`backend/nanoresearch/cron/types.py:22-29`）。

`CronJob.delete_after_run`（第62行）标记一次性任务执行后是否自动删除，否则仅置 `enabled=False`（`backend/nanoresearch/cron/types.py:62`）。

---

### 14.2 CronService

**文件**：`backend/nanoresearch/cron/service.py`

**职责**：管理任务生命周期；通过 croniter 计算下次触发时刻；以 asyncio 单任务定时器串联执行。

#### 关键组件

| 函数/属性 | 行号 | 作用 |
|---|---|---|
| `_compute_next_run(schedule, now_ms)` | 20 | 计算下次触发时间；cron 模式调用 croniter.get_next() |
| `_validate_schedule_for_add(schedule)` | 49 | 拒绝 tz+非 cron 组合；校验 IANA 时区有效性 |
| `CronService.__init__` | 68 | 注入 `store_path`（JSON 文件）和 `on_job` 回调 |
| `_load_store()` | 80 | 读取并解析 jobs.json；检测文件 mtime 变化自动重载 |
| `start()` | 195 | 加载持久化 → 重算 next_run → 首次设置定时器 |
| `_arm_timer()` | 228 | 计算最早到期时间 → asyncio.sleep → `_on_timer()` |
| `_execute_job(job)` | 265 | 调用 `on_job` 回调；写 run_history（最多保留 `_MAX_RUN_HISTORY=20` 条） |

#### 数据流

```
start()
  └─ _load_store() → jobs.json (JSON, UTF-8)
  └─ _recompute_next_runs()
  └─ _arm_timer()
        └─ asyncio.sleep(delay)
              └─ _on_timer()
                    └─ 筛选 due_jobs
                    └─ _execute_job(job) → on_job(job) 回调
                    └─ _save_store() → jobs.json
                    └─ _arm_timer()  (重新挂载)
```

jobs.json 存储路径由 `config/paths.py:get_cron_dir()` 提供（`backend/nanoresearch/config/paths.py:27`），落在实例数据目录下 `cron/jobs.json`。

#### 设计取舍

- 单 asyncio Task + 精确 sleep，避免轮询循环，对低频任务（分钟/小时级）延迟完全可接受。
- mtime 检测允许外部直接编辑 jobs.json（适合开发场景），生产环境推荐通过 API 写入。
- `on_job` 回调签名为 `Callable[[CronJob], Coroutine[Any, Any, str | None]]`，解耦具体执行逻辑与调度器（`backend/nanoresearch/cron/service.py:71`）。

---

### 14.3 命令路由（CommandRouter / builtin）

**文件**：`backend/nanoresearch/command/router.py`、`backend/nanoresearch/command/builtin.py`

`CommandRouter`（`backend/nanoresearch/command/router.py:27`）实现三层优先级分发：

1. **priority**：在 dispatch lock 之外最先匹配（`/stop`、`/restart`、`/status`）
2. **exact**：精确匹配（`/new`、`/status`、`/help`、`/research`）
3. **prefix**：最长前缀匹配（`/research `，含尾空格），排序在 `__init__` 时按长度降序（`backend/nanoresearch/command/router.py:52`）
4. **interceptors**：兜底谓词列表

`register_builtin_commands(router)`（`backend/nanoresearch/command/builtin.py:158`）注册默认命令集：

| 命令 | 层级 | 处理函数 | 功能 |
|---|---|---|---|
| `/stop` | priority | `cmd_stop` | 取消 session 所有 tasks + 子 agent（`builtin.py:15`） |
| `/restart` | priority | `cmd_restart` | 1s 后 `os.execv` 原地重启（`builtin.py:32`） |
| `/status` | priority+exact | `cmd_status` | 返回版本/模型/token 用量（`builtin.py:44`） |
| `/new` | exact | `cmd_new` | 清空 session，归档快照（`builtin.py:69`） |
| `/help` | exact | `cmd_help` | 返回命令列表（`builtin.py:85`） |
| `/research` | exact+prefix | `cmd_research` | 触发研究任务，支持 `--depth=quick\|normal\|deep`（`builtin.py:104`） |

`CommandContext`（`backend/nanoresearch/command/router.py:15`）封装 `msg`、`session`、`key`、`raw`、`args`、`loop`，handler 无需直接依赖外部状态。

---

### 14.4 HeartbeatService

**文件**：`backend/nanoresearch/heartbeat/service.py`

**职责**：定期（默认 30 分钟）读取工作区的 `HEARTBEAT.md`，通过 LLM 虚拟工具调用（tool use）决策是否有主动任务需要执行，有则驱动 `on_execute` 回调并按 `evaluate_response` 结论决定是否推送结果。

#### 两阶段设计

```
阶段1（决策）：读 HEARTBEAT.md → _decide()
  → provider.chat_with_retry(tools=[_HEARTBEAT_TOOL])
  → 解析 tool_call.arguments  → action: "skip" | "run"

阶段2（执行，仅 action=="run"）：
  → on_execute(tasks)  → 完整 agent loop
  → evaluate_response(response, tasks, provider, model)
  → 结果推送 on_notify(response)  (若 evaluate 判定 should_notify)
```

`_HEARTBEAT_TOOL`（`backend/nanoresearch/heartbeat/service.py:14`）定义了仅有两个枚举值 `skip`/`run` 的 JSON Schema 工具，避免自由文本解析的不确定性。

`HeartbeatConfig`（`backend/nanoresearch/config/schema.py:92`）提供配置：`enabled=True`、`interval_s=1800`（30 分钟）、`keep_recent_messages=8`。

`trigger_now()`（`backend/nanoresearch/heartbeat/service.py:179`）支持外部强制触发，用于测试或手动唤醒。

**与 cron 的关系**：HeartbeatService 是固定间隔的内部自检循环，不接受用户配置的调度表达式；CronService 管理用户定义的任意调度任务。两者都不承担 eval/optimization 触发——那些均由消息事件 hook 驱动。

---

## Ch15 安全与鉴权

### 职责概述

安全层涵盖四个关注点：JWT 令牌签发/校验、bcrypt 密码哈希、FastAPI 请求鉴权中间件、出站请求的 SSRF 防护。

---

### 15.1 JWT（auth/jwt.py）

**文件**：`backend/nanoresearch/auth/jwt.py`

算法固定 `HS256`（`backend/nanoresearch/auth/jwt.py:11`），令牌有效期 7 天（`EXPIRE_DAYS=7`，第12行），依赖 `python-jose` 库。

密钥从 `JWT_SECRET_KEY` 环境变量读取（`_get_secret()`，第15行），未设置则启动时抛出 `RuntimeError`，提示生成命令（`secrets.token_hex(32)`）。

```
create_token(uid: str) -> str      # 签发：payload={sub, iat, exp}
verify_token(token: str) -> str    # 校验：返回 uid；失败抛 HTTP 401
```

`verify_token`（`backend/nanoresearch/auth/jwt.py:35`）在 `JWTError`（含过期、签名错误）时统一返回 HTTP 401 + `WWW-Authenticate: Bearer`，不区分具体失败原因，避免信息泄露。

---

### 15.2 密码哈希（auth/password.py）

**文件**：`backend/nanoresearch/auth/password.py`

两个函数，直接封装 `bcrypt`：

```python
hash_password(plain: str) -> str       # bcrypt.hashpw + gensalt，返回 str（行6）
verify_password(plain: str, hashed: str) -> bool  # bcrypt.checkpw（行10）
```

salt 由 `bcrypt.gensalt()` 自动生成并内嵌于 hash 字符串，无需外部存储（`backend/nanoresearch/auth/password.py:7`）。

---

### 15.3 请求鉴权中间件（server/middleware/auth.py）

**文件**：`backend/nanoresearch/server/middleware/auth.py`

使用 FastAPI 的 `OAuth2PasswordBearer`，`tokenUrl="/api/auth/token"`（`backend/nanoresearch/server/middleware/auth.py:8`）。

```python
async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    return verify_token(token)   # 返回 uid（行11）
```

`get_current_user` 作为 FastAPI Dependency 注入到需要鉴权的路由，校验失败由 `verify_token` 直接抛出 HTTP 401，FastAPI 自动转化为响应。

鉴权流：

```
HTTP 请求
  → Authorization: Bearer <token>
  → OAuth2PasswordBearer 提取 token
  → get_current_user → verify_token(token)
  → 成功：uid 注入 handler
  → 失败：HTTP 401 {"detail": "无效或已过期的 token"}
```

---

### 15.4 网络隔离与 SSRF 防护（security/network.py）

**文件**：`backend/nanoresearch/security/network.py`

防止代理将出站请求引向内网地址（SSRF），阻断列表（`_BLOCKED_NETWORKS`，第10行）覆盖：

| 网段 | 用途 |
|---|---|
| 0.0.0.0/8 | 通配符 |
| 10.0.0.0/8 | 私有 A 类 |
| 100.64.0.0/10 | 运营商级 NAT |
| 127.0.0.0/8 | 本地回环 |
| 169.254.0.0/16 | link-local / 云元数据（AWS 169.254.169.254） |
| 172.16.0.0/12 | 私有 B 类 |
| 192.168.0.0/16 | 私有 C 类 |
| ::1/128, fc00::/7, fe80::/10 | IPv6 回环/私有/link-local |

三个公开函数：

- `validate_url_target(url)` → `(bool, str)`（`backend/nanoresearch/security/network.py:30`）：校验 scheme（仅 http/https）、hostname 存在性、DNS 解析结果是否落入阻断网段。
- `validate_resolved_url(url)` → `(bool, str)`（第65行）：仅检查 IP（跳过 DNS），用于重定向后二次校验。
- `contains_internal_url(command)` → `bool`（第97行）：正则扫描命令字符串中所有 URL，任一命中私有地址则返回 True，用于 exec 工具的前置安全检查。

设计选择：DNS 解析阶段即阻断（而非等到 connect），可防止 DNS rebinding 攻击。重定向后通过 `validate_resolved_url` 再次校验，双重检查覆盖绕过。

---

### 15.5 SECURITY.md 摘要

SECURITY.md 列出的关键安全要点（`SECURITY.md`）：

- config.json 推荐权限 `0600`，不能提交 API key
- channel `allowFrom` 自 v0.1.4.post4 起空列表默认**拒绝所有**（而非允许所有），须设 `["*"]` 才开放
- exec 工具阻断 `rm -rf /`、fork bomb、`mkfs.*`、裸磁盘写等破坏性模式
- WhatsApp bridge 仅绑定 `127.0.0.1:3001`，可配 `bridgeToken` 做本地共享密钥验证
- 已知局限：无内置速率限制、配置文件明文存储、无自动 session 过期

---

## Ch16 配置系统

### 职责概述

配置系统负责将默认值、JSON 文件、环境变量三层叠加后构造出一个类型安全的 Pydantic `Config` 对象，并提供路径解析、迁移和向后兼容的 env 双读。

---

### 16.1 schema.py — Pydantic 配置模型

**文件**：`backend/nanoresearch/config/schema.py`

所有模型继承 `Base(BaseModel)`（第11行），`alias_generator=to_camel`，支持 camelCase（JSON 文件）和 snake_case（Python 代码）双向解析（`backend/nanoresearch/config/schema.py:14`）。

顶层根模型为 `Config(BaseSettings)`（第174行），`env_prefix="NANORESEARCH_"`，`env_nested_delimiter="__"`（第301行）——即 `NANORESEARCH_AGENTS__DEFAULTS__MODEL` 可覆盖 `agents.defaults.model`。

配置层级结构：

```
Config
├── agents: AgentsConfig
│   ├── defaults: AgentDefaults           # model, workspace, max_tokens, temperature...
│   └── allowed_models: list[str]
├── channels: ChannelsConfig              # extra="allow"，各 channel 自解析
│   ├── send_progress, send_tool_hints
│   └── send_max_retries: int [0..10]
├── providers: ProvidersConfig            # 20+ 个 ProviderConfig（api_key, api_base, extra_headers）
├── gateway: GatewayConfig
│   ├── host, port
│   └── heartbeat: HeartbeatConfig
└── tools: ToolsConfig
    ├── web: WebToolsConfig → search: WebSearchConfig
    ├── exec: ExecToolConfig
    ├── research: ResearchConfig
    ├── restrict_to_workspace: bool
    └── mcp_servers: dict[str, MCPServerConfig]
```

`Config._match_provider(model)`（第206行）实现多级 provider 自动匹配：强制 provider → 精确前缀 → 关键字匹配 → 本地 provider detect_by_base_keyword → gateway fallback。

`Config._warn_providers_in_server_mode()`（`model_validator`，第183行）：`NANORESEARCH_MODE=server` 时 config.json 中带 api_key 的 provider 不生效并发出警告，凭证须走 `user_settings.extra.providers`。

---

### 16.2 loader.py — 加载顺序与模式控制

**文件**：`backend/nanoresearch/config/loader.py`

```
load_config(config_path?) -> Config
  1. 调用 apply_legacy_env_compat()  （NANOBOT_* 双读，见 16.4）
  2. 读取 config_path（默认 ~/.nanoresearch/config.json）
  3. _migrate_config(data)           （inline 迁移，见 16.3）
  4. Config.model_validate(data)     （Pydantic 解析 + env 叠加）
  失败 → 降级 Config()（全默认值），记录 warning
```

`get_nanoresearch_home()`（`backend/nanoresearch/config/loader.py:18`）：先调 `apply_legacy_env_compat()`，再读 `NANORESEARCH_HOME`，fallback `~/.nanoresearch`。

`get_mode()`（第35行）读 `NANORESEARCH_MODE`（`"server"` | `"local"`，默认 `"local"`）；server 模式下 `env_key_or_raise()`（第54行）阻止使用主机 env var API key，强制凭证来自 user_settings。

`set_config_path(path)` / `get_config_path()`（第74、80行）允许多实例（multi-instance）场景下每个实例有独立配置目录，数据目录由配置文件所在 parent 决定（`paths.py:get_data_dir()`）。

---

### 16.3 migration.py — 配置迁移

**文件**：`backend/nanoresearch/config/loader.py`（内联迁移）和 `backend/nanoresearch/config/migration.py`（一次性 API key 迁移）

**inline 迁移**（`_migrate_config(data)`，`loader.py:129`）：
- 将旧路径 `tools.exec.restrictToWorkspace` 移至 `tools.restrictToWorkspace`；每次加载时静默执行。

**settings.yaml → config.json API key 迁移**（`migrate_llm_keys()`，`migration.py:58`）：
- 扫描 `settings.yaml` 中 `llm`、`embedding`、`vision_llm` 三个 section 的 `api_key`
- 规范化 provider 名称（`openai_compat` → `custom` 等，`_PROVIDER_FIELD_OVERRIDES`，第33行）
- 目标 provider 在 config.json 已有 key 则跳过（skip_already_exists）
- 支持 `dry_run=True` 预览，不写磁盘

---

### 16.4 env_compat.py — NANOBOT_* → NANORESEARCH_* 双读

**文件**：`backend/nanoresearch/utils/env_compat.py`

`apply_legacy_env_compat()`（`backend/nanoresearch/utils/env_compat.py:24`）：

- 幂等：`_applied` 全局标志，第二次调用直接返回 `[]`（第32行）
- 扫描所有 `NANOBOT_*` env var → 计算 `NANORESEARCH_*` 新名称
- 若新名称**已存在**：发出 DeprecationWarning（提示 conflict），不覆盖（第46行）
- 若新名称**不存在**：复制值到新名称 + 发出 DeprecationWarning（第50行）
- 返回实际复制的 `(old_name, new_name)` 列表，供调用者审计

调用时机：进程启动最早期——`get_nanoresearch_home()` 内部调用，确保 Pydantic Settings 初始化前已完成（`loader.py:25`）。移除时间线：v0.3.0（`backend/nanoresearch/utils/env_compat.py:9`）。

测试覆盖：`tests/utils/test_env_compat.py`，5 个测试验证：复制+警告、新名称优先、幂等性、忽略无关变量、历史五个 NANOBOT_* 变量全部 roundtrip。

---

### 16.5 paths.py — 路径解析

**文件**：`backend/nanoresearch/config/paths.py`

所有路径基于 `get_data_dir()` = `get_config_path().parent`，确保多实例隔离：

| 函数 | 返回路径 |
|---|---|
| `get_data_dir()` | `<config_parent>/` |
| `get_runtime_subdir(name)` | `<data_dir>/<name>/` |
| `get_cron_dir()` | `<data_dir>/cron/` |
| `get_logs_dir()` | `<data_dir>/logs/` |
| `get_media_dir(channel?)` | `<data_dir>/media/[<channel>/]` |
| `get_workspace_path(workspace?)` | 展开 `~` 后确保目录存在 |
| `get_cli_history_path()` | `~/.nanoresearch/history/cli_history` |
| `get_bridge_install_dir()` | `~/.nanoresearch/bridge/` |
| `get_legacy_sessions_dir()` | `~/.nanoresearch/sessions/`（迁移兜底）|

所有返回值均通过 `ensure_dir()` 保证目录存在。

---

## Ch17 横切关注点

### 职责概述

横切关注点涵盖可观测性（结构化日志 + RAG trace + 缓存指标）、统一错误处理模式、测试体系和 CI 流水线。

---

### 17.1 可观测性

#### 结构化日志（RAG Observability Logger）

**文件**：`backend/nanoresearch/rag/observability/logger.py`

两条日志输出路径并存：

1. **人类可读日志**：`get_logger(name, log_level?)`（第29行）配置标准 `logging` 到 stderr，格式 `%(asctime)s %(levelname)s %(name)s %(message)s`；`httpx` 日志静默到 WARNING（防止 URL 泄露敏感 endpoint，第52行）。

2. **JSON Lines trace 日志**：`get_trace_logger(traces_path, name?)`（第107行）返回附加了 `JSONFormatter` 的 FileHandler，写入 `.jsonl` 文件，`propagate=False` 防止重复输出到 console。`write_trace(trace_dict, traces_path?)`（第145行）不经 logging 框架，直接 append JSON 行，适合高频写入。

`JSONFormatter`（第60行）序列化 `logging.LogRecord`，保留 `extra=` 附带的任意字段（跳过 Python 内部属性），异常时附加 `exception` 字段。

全系统其他地方（非 RAG 路径）使用 loguru 的 `logger`（如 cron、heartbeat、auth），两套日志库并存，互不干扰。

#### TraceContext + TraceCollector

**文件**：`backend/nanoresearch/rag/core/trace/trace_context.py`、`backend/nanoresearch/rag/core/trace/trace_collector.py`

`TraceContext`（`backend/nanoresearch/rag/core/trace/trace_context.py:15`）是请求级 dataclass，携带：
- `trace_id`（UUID，第28行）
- `trace_type`：`"query"` | `"ingestion"`（第27行）
- `stages`：有序列表，每项记录 `stage_name`、`timestamp`、`elapsed_ms`、`data`（第44行 `record_stage`）
- `metadata`：任意 KV（第31行）
- 单调时钟（`_start_mono`）确保 `elapsed_ms()` 不受系统时钟回拨影响（第35行）

`finish()`（第68行）记录 wall-clock 结束时间；`to_dict()`（第100行）序列化为 JSON 可写的 plain dict，包含 `total_elapsed_ms`。

`TraceCollector`（`backend/nanoresearch/rag/core/trace/trace_collector.py:23`）接收已完成（或调用时自动 `finish()`）的 `TraceContext`，append 到 `traces.jsonl`（默认 `logs/traces.jsonl`）；写入失败记录 exception 但不传播错误，保证可观测性代码不阻塞业务（第47行）。

数据流：

```
pipeline stage
  → trace.record_stage("dense_retrieval", {...}, elapsed_ms)
  → trace.finish()
  → collector.collect(trace)
  → traces.jsonl (JSON Lines, UTF-8)
```

#### Cache Metrics

**文件**：`backend/nanoresearch/utils/cache_metrics.py`

`CacheMetrics`（第9行）统计提示词缓存效果：`hit_rate`、`savings_ratio`（read tokens / total tokens）、`avg_creation_tokens`、`avg_read_tokens`；每 10 次请求自动 `log_summary()`（`_log_interval=20`，实际代码第19行，初始值 10 对应 `_log_interval: int = 10`）。

`record_cache_stats(usage: dict)`（第111行）从 LLM 响应 usage 中提取 `cache_read_input_tokens`、`cache_creation_input_tokens`，写入全局单例 `get_cache_metrics()`（第103行）。

---

### 17.2 统一错误处理模式

系统未定义集中式 error handler 基类，但有三条一致约定：

1. **鉴权失败**：`verify_token` 直接 raise `HTTPException(401)`，FastAPI 转响应（`backend/nanoresearch/auth/jwt.py:44`）。
2. **可观测性代码**：写入失败只记录 exception，不重新抛出（`trace_collector.py:51`；`observability/logger.py` write_trace 亦同）。
3. **配置加载失败**：`load_config` 降级为默认 `Config()`，记录 warning（`backend/nanoresearch/config/loader.py:105`）。

cron 和 heartbeat 内的错误均被 catch 后记录 loguru error，任务状态标记为 `"error"` 继续调度下一次（`cron/service.py:278`；`heartbeat/service.py:176`）。

---

### 17.3 测试策略

**根目录**：`tests/`，共约 70 个测试文件。

| 子域 | 测试文件数 | 代表性内容 |
|---|---|---|
| `tests/agent/` | ~15（含 evaluation/） | loop save/consolidation、heartbeat service、cron timezone、task cancel、memory types |
| `tests/channels/` | ~15 | 各 channel 发送/接收/streaming/markdown 渲染 |
| `tests/cli/` | 3 | CLI 输入、slash 命令、restart 命令 |
| `tests/config/` | 3 | config migration（inline + API key）、paths |
| `tests/cron/` | 2 | CronService 调度逻辑（timezone 校验、run history）、cron tool list |
| `tests/providers/` | 7 | 各 provider 初始化、litellm kwargs、重试逻辑 |
| `tests/rag/` | 8（unit+integration+stress） | 文档分块、混合搜索、融合、reranker；pipeline 集成；并发/大文档压力 |
| `tests/security/` | 1 | `security/network.py` SSRF 阻断规则 |
| `tests/tools/` | 8 | exec 安全、filesystem、MCP、web fetch/search SSRF、tool validation |
| `tests/utils/` | 1 | `env_compat` 双读（5 个用例） |
| `tests/research/` | 1 | frontmatter 解析 |

多数单元测试不需要网络或数据库，使用 `tmp_path` fixture 隔离文件系统；`tests/rag/stress/` 下有并发和大文档测试，可能运行较慢。

---

### 17.4 CI 流水线

**文件**：`.github/workflows/ci.yml`

触发：push / PR 到 `main` 或 `nightly` 分支。

矩阵：Python 3.11 / 3.12 / 3.13，运行在 `ubuntu-latest`。

步骤：

1. `apt-get install libolm-dev build-essential`（Matrix channel 依赖 libolm）
2. `uv sync --all-extras`（含所有可选依赖）
3. **B4 gate 检查**（`.github/workflows/ci.yml:34`）：`uv run python scripts/ci/check_case_metadata_pr.py` — 校验 PR 新增 fixture 是否带 case metadata，防止无元数据的测试数据进入主干
4. `uv run pytest tests/`：全量测试，无额外 `--ignore` 或分组

CI 未拆分 lint/type-check 步骤（代码中无 mypy/ruff 专项 job），测试即为唯一质量门禁（基准 commit `3b590a8f`）。


---

## Ch18 附录

### 18.1 关键文件\:行号速查索引

下表汇总各层最常用的定位锚点（撰写时已核对，指向基线 `3b590a8f` 代码）。完整锚点散见各章「关键组件」小节。

| 子系统 | 锚点 | 含义 |
|---|---|---|
| 进程入口 | `cli/commands.py:576` / `:1339` / `:839` | `gateway` / `serve` / `agent` 子命令 |
| 进程入口 | `worker.py:558` / `:251` / `:457` | `WorkerSettings` / `run_agent_job` / `ingest_document_task` |
| 路径 | `config/loader.py:18` / `:29` | `get_nanoresearch_home()` / 唯一 fallback `~/.nanoresearch` |
| Agent | `agent/loop.py:54` / `:232` / `:427` | `AgentLoop` / `_ensure_mcp_connected` / `run` |
| Agent | `agent/runner.py:61` / `:94` / `:198` | `AgentRunner` / `run` / 逃生通道（连续失败=3） |
| Agent | `agent/context.py:26` / `:516` / `:20-22` | `ContextBuilder` / `build_messages` / 预算常量（3000，0.6/0.4） |
| 记忆 | `agent/memory.py:33` / `:34` / `:35` | `TAIL_PROTECT=8` / `TARGET_RATIO=0.5` / `SUMMARY_CONFIDENCE=0.7` |
| 记忆 | `agent/memory.py:491` / `:553` / `:649` | `pick_consolidation_boundary` / `maybe_consolidate_by_tokens` / `plan_startup_consolidation` |
| Channels | `channels/base.py:15` / `manager.py:167` / `registry.py:54` | `BaseChannel` / `_coalesce_stream_deltas` / `discover_all` |
| Bus | `bus/events.py:9` / `:28` / `pending_reaper.py:32` | `InboundMessage` / `OutboundMessage` / `PendingReaper`(300/7200) |
| Session | `session/manager.py:190,192` | Redis `DEL + RPUSH` 全量写（volatile-lru 兼容） |
| Providers | `providers/base.py:73` / `model_factory.py:35` / `:116` | `LLMProvider` / `ModelRole`(6 角色) / `resolve()` |
| RAG | `rag/core/query_engine/fusion.py:64` / `storage/bm25_indexer.py:80` | RRF `DEFAULT_K=60` / BM25 `k1=1.5` |
| RAG | `server/routers/knowledge_router.py:104` | per-uid collection `f"{uid}_{kb.id}"` |
| Research | `research/runner.py:99` / `types.py:303` | `ResearchRunner.run` / `evaluation_threshold=6.0` |
| Server | `server/main.py:32` / `:92` / `chat_router.py:326` | `create_app` / API v2.0.0 / `GET /api/runs/{id}/events`(SSE) |
| 前端 | `web/src/composables/useRunStream.js:3` / `web/src/stores/chat.js:6` | `useRunStream` / `useChatStore` |
| Eval | `eval/optimizer.py:58` / `judge.py:81` / `loop.py:579` | σ-gate `K=1.96` / `score_with_consistency` / `_maybe_save_snapshot` 钩子 |
| Auth | `auth/jwt.py:11` / `auth/password.py:7` / `security/network.py:10` | `HS256` / `bcrypt.gensalt` / `_BLOCKED_NETWORKS`(8 网段) |
| Config | `utils/env_compat.py:24` / `:32` | `apply_legacy_env_compat`（NANOBOT_*→NANORESEARCH_* 双读）/ 幂等守卫 |
| Cron | `cron/service.py:20` / `:228` / `:265` | `_compute_next_run` / `_arm_timer` / `_execute_job` |

### 18.2 术语表

| 术语 | 含义 |
|---|---|
| **AgentLoop** | `agent/loop.py:AgentLoop`，单实例消息处理主引擎。 |
| **gateway** | `nr gateway` 进程：频道机器人 + 仪表盘（:8765），**不含** REST API。 |
| **serve** | `nr serve` 进程：FastAPI REST API（:18790），多租户 Web 后端。 |
| **worker** | arq 后台 worker，执行 `run_agent_job` / `ingest_document_task`。 |
| **RAG MCP Server** | AgentLoop 的 stdio 子进程，JSON-RPC 2.0 暴露 RAG 工具。 |
| **consolidation / compaction** | token 感知的会话历史压缩：超阈值时摘要旧消息、保护尾部（tail-protect）。 |
| **per-uid 隔离** | 知识库向量按 `{uid}_{kb_id}` 独立 Chroma collection 物理隔离。 |
| **ReAct** | AgentRunner 的 think→tool_call→observe 循环范式。 |
| **RRF** | Reciprocal Rank Fusion，多路召回结果融合（`1/(k+rank)`，k=60）。 |
| **飞轮 / flywheel** | badcase → 优化 → 回放验证 → 回灌的评测闭环。 |

### 18.3 相关设计决策文档

本文为整体蓝图；以下文档记录单点子系统的设计与演进，深读时可对照：

- 共识压缩：`docs/superpowers/specs/2026-06-28-consolidation-compaction-design.md`、`...-consolidation-anchor-retention-design.md`
- 多租户 LLM 配置 / provider 角色：`...-multitenant-llm-config-design.md`、`...-provider-role-explicit-design.md`
- RAG 查询改写（A 类）：`...-rag-query-rewrite-fix.md`、`docs/superpowers/plans/2026-06-28-rag-query-rewrite-a-class.md`
- 工具层 Harness（A1）：`...-tool-layer-harness-a1-design.md`
- Redis 演进：`docs/redis-sdd.md`、`docs/redis-phase{0-4}-*.md`
- RAG 设计与知识闭环：`docs/RAG_DESIGN.md`、`docs/KNOWLEDGE_LOOP_DESIGN.md`、`docs/sdd/knowledge_loop_refactor_v1.md`
- 运行时与渠道插件：`docs/RUNTIME_DESIGN.md`、`docs/CHANNEL_PLUGIN_GUIDE.md`

### 18.4 已知 as-built 漂移与残余盲区

撰写过程中各层读码发现的、与文档/直觉不一致或值得后续处理的点（均为当前代码事实）：

1. **Dockerfile 过时**：`Dockerfile` 仍用旧包名（`ENTRYPOINT ["nanobot"]`、`COPY backend/nanobot/`），与 Phase 3 改名（已完成）和 `pyproject.toml` 仅注册 `nr`/`nanoresearch` 脚本不一致——按当前镜像构建会失效。
2. **gateway vs serve 端口**：docker-compose `nanoresearch-gateway` 服务暴露 18790+8765 并以 `command: gateway` 启动，但 `nr gateway` 只绑定 8765（仪表盘）；18790 需 `nr serve` 才绑定。单容器同时提供频道与 REST 需额外起 `serve`。
3. **RAG 工具暴露漂移**：`rag/mcp_server/protocol_handler.py:226` 文档/日志声称暴露 `kb_retrieve` + `memory_search`，但 `tools/agentic/retrieval.py:529` 的 `register_tools` 为空实现（`pass`），当前构建实际未暴露这两个工具。
4. **研究记忆跨租户盲区**：`research_chunks` / `research_claims` / `research_insights` / `user_memory` 为固定共享 collection，仅靠查询期 uid 元数据过滤（`agent/context.py:68`），未做 collection 级隔离——与 KB 文档的物理隔离不同，存在跨租户读泄漏风险，需人审。
5. **B 类指代消解写入侧缺失**：`query_planning.py:403` 的 `_chunk_titles` sidecar 读取侧已就位，但写入侧未产出，「上一轮检索标题」消解路径当前恒空。
6. **会话时间戳 naive**：`bus/events.py:16` 的 `InboundMessage.timestamp` 用 `datetime.now()`（naive），与会话层 aware-UTC 规范不一致（已在会话层归一化，但源头未改）。
7. **eval 在线/离线分离**：生产流仅触发硬规则 badcase（run_failure / token_spike / excessive_retries）；质量评分（LLM judge）只在 TestRunner 批量评测中生效，二者刻意分离。
8. **knowledge_lint 废弃项**：`knowledge_lint.py` 检查的 `claim_store` / `insight_store` 在当前 `KnowledgeSearch` 已废弃（注释明示 deprecated），对现实例无效。

> 说明：本文记录的是「代码现状」，上述漂移点不代表缺陷判定，仅供维护者优先级排序参考。


