# nanobot 项目结构

## 概述

**nanobot** 是一个超轻量级个人 AI 助手框架，支持多渠道（Telegram、Discord、Slack 等）和多 LLM 提供商。

## 根目录结构

```
D:/Code/nanobot/
├── nanobot/                 # 源代码主目录
├── config/                 # RAG 配置文件
│   └── rag/settings.yaml   # RAG 配置
├── workspace/              # 工作目录和数据
│   └── rag_data/          # RAG 向量数据
├── docs/                   # 文档
├── tests/                  # 测试
├── pyproject.toml          # 项目配置
├── README.md               # 项目文档
└── bridge/                # WhatsApp 桥接（Node.js）
```

## nanobot/ 源代码结构

```
nanobot/
├── agent/                  # Agent 核心
│   ├── loop.py             # AgentLoop - 主处理引擎
│   ├── runner.py           # AgentRunner - LLM 循环执行
│   ├── context.py         # ContextBuilder - 提示词构建
│   ├── memory.py          # MemoryStore - 记忆存储
│   ├── skills.py          # SkillsLoader - Skill 加载
│   ├── hook.py            # Agent hooks
│   ├── subagent.py        # 子 Agent 管理
│   └── tools/             # 内置工具
│       ├── base.py        # Tool 基类
│       ├── registry.py     # Tool 注册表
│       ├── filesystem.py  # 文件操作工具
│       ├── shell.py       # Shell 执行
│       ├── spawn.py       # 子进程
│       ├── cron.py        # 定时任务
│       ├── message.py     # 消息发送
│       ├── mcp.py          # MCP 客户端
│       ├── web.py         # 网页搜索/抓取
│       └── research.py    # 研究工具
│
├── providers/              # LLM 提供商
│   ├── base.py            # LLMProvider 抽象基类
│   ├── registry.py        # 提供商注册表
│   ├── anthropic_provider.py
│   ├── openai_compat_provider.py
│   ├── azure_openai_provider.py
│   ├── deepseek_provider.py
│   ├── zhipu_provider.py
│   ├── dashscope_provider.py
│   └── ...
│
├── channels/               # 聊天渠道
│   ├── base.py           # BaseChannel 抽象基类
│   ├── manager.py        # ChannelManager
│   ├── registry.py       # 渠道插件发现
│   ├── telegram.py
│   ├── discord.py
│   ├── slack.py
│   ├── feishu.py         # 飞书
│   ├── dingtalk.py       # 钉钉
│   ├── wechat.py         # 微信
│   ├── whatsapp.py
│   └── ...
│
├── session/               # 会话管理
│   └── manager.py         # SessionManager
│
├── bus/                   # 消息总线
│   ├── events.py         # InboundMessage, OutboundMessage
│   └── queue.py          # MessageBus
│
├── config/                # 配置系统
│   ├── schema.py         # Pydantic 模型
│   ├── loader.py         # 配置加载
│   └── paths.py          # 路径工具
│
├── cli/                   # CLI 命令
│   ├── commands.py       # 主 CLI 入口
│   ├── onboard.py        # 交互式引导
│   └── stream.py         # 流式渲染
│
├── command/              # 命令路由
│   └── router.py        # CommandRouter
│
├── cron/                 # 定时任务
│   ├── service.py
│   └── types.py
│
├── heartbeat/            # 心跳任务
│
├── providers/            # LLM 提供商
│
├── rag/                  # RAG 模块（新增）
│   ├── core/            # 核心模块
│   │   ├── settings.py  # 配置加载
│   │   ├── types.py    # 数据类型
│   │   ├── query_engine/   # 查询引擎
│   │   │   ├── hybrid_search.py
│   │   │   ├── dense_retriever.py
│   │   │   ├── sparse_retriever.py
│   │   │   ├── reranker.py
│   │   │   └── fusion.py
│   │   ├── response/    # 响应构建
│   │   └── trace/       # 追踪
│   │
│   ├── ingestion/      # 摄入管道
│   │   ├── pipeline.py  # 主管道
│   │   ├── chunking/    # 文本分割
│   │   ├── embedding/   # 向量编码
│   │   ├── storage/     # 存储
│   │   └── transform/   # 转换
│   │
│   ├── libs/           # 第三方库封装
│   │   ├── llm/         # LLM 提供商
│   │   ├── embedding/   # Embedding 提供商
│   │   ├── vector_store/ # 向量存储
│   │   ├── reranker/    # 重排序
│   │   ├── splitter/     # 文本分割器
│   │   └── loader/       # 文档加载器
│   │
│   ├── observability/   # 可观测性
│   │   └── logger.py    # 日志
│   │
│   └── mcp_server/      # RAG MCP Server（新增）
│       ├── __main__.py  # 入口
│       ├── server.py    # MCP Server
│       ├── protocol_handler.py
│       └── tools/       # MCP 工具
│           ├── query_knowledge_hub.py
│           ├── ingest_documents.py
│           ├── list_documents.py
│           └── ...
│
└── tools/               # nanobot 原生工具（已废弃，改为 MCP）
    └── rag_tool.py     # RAG 工具（保留但不再使用）
```

## 全局配置

```
~/.nanobot/
├── config.json          # MCP Server 配置
├── rag/settings.yaml    # RAG 配置（复制到项目内）
└── workspace/          # 工作空间
    └── rag_data/        # RAG 数据
```

## 启动流程

```
nanobot agent
  └── AgentLoop._connect_mcp()  # 连接 MCP Servers
  │     └── connect_mcp_servers()
  │           └── 启动子进程: python -m nanobot.rag.mcp_server
  └── AgentLoop.run()
        └── MCP Client ←──stdio──► RAG MCP Server (子进程)
```

## 关键配置

### ~/.nanobot/config.json
```json
{
  "tools": {
    "mcpServers": {
      "rag": {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "nanobot.rag.mcp_server"],
        "env": {
          "PYTHONPATH": "D:/Code/nanobot"
        },
        "enabledTools": ["*"]
      }
    }
  }
}
```

### config/rag/settings.yaml
```yaml
llm:
  provider: "openai"
  model: "qwen-turbo"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "your-key"

embedding:
  provider: "openai"
  model: "text-embedding-v3"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

vector_store:
  provider: "chroma"
  persist_directory: "./workspace/rag_data/chroma"
```

## MCP 工具列表

| 工具名 | 功能 | MCP Server |
|--------|------|------------|
| `mcp_rag_ingest_documents` | 摄入 PDF | rag |
| `mcp_rag_query_knowledge_hub` | 搜索 | rag |
| `mcp_rag_list_documents` | 列文档 | rag |
| `mcp_rag_list_collections` | 列集合 | rag |
| `mcp_rag_get_document_summary` | 文档摘要 | rag |

## Skills

```
nanobot/skills/
├── rag/SKILL.md           # RAG Skill
├── github/
├── weather/
├── summarize/
├── tmux/
├── cron/
├── memory/
├── clawhub/
└── skill-creator/
```

---

*最后更新: 2026-03-31*
