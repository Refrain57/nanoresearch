# Phase 5: 多租户 LLM API 配置 — Design Spec

日期：2026-06-28
状态：待用户审阅，未进入实施
前置：Phase 3（nanoresearch 包名）+ Phase 4（NANORESEARCH_HOME）已完成
brainstorm 草稿：`2026-06-28-multitenant-llm-config-brainstorm.md`

---

## 1. 背景与目标

### 1.1 现状一句话总结
`user_settings.extra.providers`（JSONB）+ `ModelFactory.resolve()` 这条 per-uid 链路已经全量打通（chat / ingestion / embedding / vision / eval 五个角色都走它）。Phase 5 不是"从无到有加多租户"，是**修这条链路上的三个 leak 漏洞**，让它在真多用户部署下站得住。

### 1.2 三个 leak
- **F1**：`providers/openai_compat_provider.py:140 _setup_env()` 把 user 的 api_key 写到 `os.environ[spec.env_key]`。worker 并发处理多 uid 时跨用户串号。
- **F2**：9 个文件 fallback 到 `os.environ.get("OPENAI_API_KEY"/"DASHSCOPE_API_KEY"/"AZURE_OPENAI_API_KEY")`。SaaS 部署下用户没填 key 自动走主机环境变量，平台账户付费。
- **F3**：`settings.yaml` 的 `llm.api_key` / `embedding.api_key` / `vision_llm.api_key` 字段和 `config.json` 顶层 `providers` 块在 server 模式下都不该读，但当前代码无差别 fallback。

### 1.3 范围
- ✅ 修 F1 / F2 / F3
- ✅ 凭证唯一真相收口到 DB
- ✅ 加 `NANORESEARCH_MODE=server|local` 开关
- ✅ 前端引导用户填 chat key + embedding key
- ✅ 验证 dashscope qwen-thinking / deepseek-r1 的 `reasoning_content` 正确落字段
- ❌ 不做团队 / org 层
- ❌ 不做 pooled key / 按 token 计费
- ❌ 不做 key 加密（DB / Redis 仍明文）
- ❌ 不做前端"测试连接"按钮
- ❌ 不删 anthropic / azure / codex backend 文件（留着，但不主动测）

---

## 2. 决策汇总

| 决策 | 选项 |
|---|---|
| Phase 5 范围 | 个人 BYOK（修漏洞，不做团队/计费） |
| Key 适用角色 | 全覆盖（chat + ingestion + embedding + vision + eval 都走用户 key） |
| Key 存储 | DB `user_settings.extra.providers` 明文，Redis cache 沿用现状 |
| 加密 | 无（明文） |
| CLI 兼容 | `NANORESEARCH_MODE=server` 严（DB only）/ `=local` 宽（config.json + env 沿用），默认 `local` |
| Backend 适配 | 主验 openai_compat（dashscope/deepseek/openai/openrouter 等共用此 backend），anthropic native / azure / codex backend 留着不动 |
| 响应格式适配 | 验证 `reasoning_content` / `thinking_blocks` 在 dashscope qwen-thinking、deepseek-r1 上正确 |
| 前端连接测试 | 不做 |

---

## 3. 配置层级（最终样子）

### 3.1 server 模式（`NANORESEARCH_MODE=server`）
```
凭证查找：user_settings.extra.providers   ← 唯一来源
模型 / 行为参数：rag_settings (settings.yaml)  ← 仅非凭证字段（model 名、base_url、chunk_size、rerank 等）
                config.json                  ← 仅 agents / tools / channels 等非凭证段
缺 key 时：raise ModelResolutionError，不 fallback
```

### 3.2 local 模式（`NANORESEARCH_MODE=local`，默认）
```
凭证查找：user_settings.extra.providers
        > config.json.providers           ← 沿用，本地 dev 友好
        > rag_settings.{llm,embedding,vision_llm}.api_key
        > os.environ.{OPENAI,DASHSCOPE,...}_API_KEY
缺 key 时：raise ModelResolutionError
```

---

## 4. 代码改动范围

### 4.1 providers/openai_compat_provider.py
- **改 `_setup_env()`**：删除 `os.environ[spec.env_key] = api_key`（gateway 分支）和 `os.environ.setdefault(spec.env_key, api_key)`（非 gateway 分支）。env_extras 同理删。
- **client 构造**：`api_key` 改成只通过 `AsyncOpenAI(api_key=...)` 显式注入，不依赖环境变量。该构造点已经在传 `api_key=`，所以删除 `_setup_env` 不影响功能。
- **副作用清查**：grep 项目里所有 `os.environ[spec.env_key]` 的读侧（应只在 `rag/libs/*` 里），见 4.3。

### 4.2 providers/model_factory.py
- **加 `mode` 参数**：`resolve(role, *, config, rag_settings, user_model, user_providers, mode: Literal["server","local"]="local", **overrides)`。
- **server 模式**：跳过 config / rag_settings 的凭证 fallback，user_providers 缺 key 直接 raise `ModelResolutionError(missing_role=role)`。
- **local 模式**：保持现状 fallback 链。
- **mode 来源**：默认从 `os.environ.get("NANORESEARCH_MODE", "local")` 读，所有调用点不强制传 mode 时自动取环境变量。

### 4.3 9 个 env var fallback 文件
对每个文件：
- 用 `nanoresearch.config.loader.get_mode()`（新加的小 helper，读 `NANORESEARCH_MODE`）判断
- server 模式下 `os.environ.get("OPENAI_API_KEY"...)` 改成 `raise RuntimeError("API key required from user_settings in server mode")`
- local 模式保持现状

文件清单：
```
worker.py:215                                   simple-RAG 流式
server/routers/eval_router.py:162,222,310       ragas / agent eval
rag/libs/embedding/openai_embedding.py:72       
rag/libs/embedding/dashscope_embedding.py:49    
rag/libs/embedding/azure_embedding.py:76        
rag/libs/llm/openai_llm.py:72                   
rag/libs/llm/openai_vision_llm.py:102           
rag/libs/llm/azure_llm.py:79                    
rag/libs/llm/azure_vision_llm.py:120            
```

### 4.4 config/schema.py
- `ProvidersConfig` **保留**（local 模式仍需读）。
- 加 Pydantic root validator：load 时若检测到 `NANORESEARCH_MODE=server` 且 `providers` 块非空，发 warning（不报错，运维信号）。

### 4.5 config/loader.py
- 加 `get_mode() -> Literal["server","local"]`：`os.environ.get("NANORESEARCH_MODE", "local")`。
- 验证只接受这两个值，其他值 raise ValueError。

### 4.6 调用点改动（5 处 ModelFactory 调用全部传 mode）
```
worker.py:104, 487
server/routers/knowledge_router.py:52
server/routers/eval_router.py:55, 1019
```
每处加 `mode=get_mode()`。

### 4.7 ModelResolutionError 处理
- 现有 `ModelResolutionError` 已经有 `sources_checked` 字段。
- server 模式下 raise 时附 `missing_role` 字段（chat / embedding / etc），前端可据此精准提示。
- API 层（settings_router、knowledge_router、eval_router）捕获后返回 422 + 结构化 body：`{"error": "missing_provider", "role": "embedding"}`。

---

## 5. 前端改动（web/）

### 5.1 Settings 页（providers 编辑区）
- 在 provider 列表上方加一行说明：**"至少需要填两组 key：一组用于 Chat（如 deepseek/openai）、一组用于 Embedding（如 dashscope/openai）。"**
- 每个 provider 行加 `models[]` 输入框（已有数据结构，UI 暴露），说明 "标注此 key 能跑的模型；多个 provider 共存时，按 model 精确匹配优先，匹配不到走第一个有 key 的 provider 兜底"（与 `ModelFactory._match_user_provider_by_model` 实际行为一致）。

### 5.2 缺 key 时的功能 gate
- 知识库"上传文档"按钮：缺 embedding key 时灰掉，hover 提示 "请在 Settings 添加 embedding provider"。
- 知识库"查询"输入框：缺 embedding key 时禁用。
- 聊天输入框：缺 chat key 时禁用 + 横幅提示。
- Gate 触发条件：调 `/api/settings/me` 返回的 `providers[]` 列表为空，或没有任何 provider 标注覆盖 chat / embedding 模型。

### 5.3 后端 422 错误展示
对 `{"error": "missing_provider", "role": ...}` 返回，前端弹一个 toast：`缺少 {role} 模型的 API key，请到 Settings 添加。`

---

## 6. 响应格式适配（thinking / reasoning_content）

**目标**：保证 dashscope qwen-thinking、deepseek-r1 的推理内容（"思考过程"）正确落到 `LLMResponse.reasoning_content` 字段，前端能渲染。

### 6.1 现状核查
- `LLMResponse.reasoning_content`（base.py:49）已存在，注释写 "Kimi, DeepSeek-R1 etc."
- `openai_compat_provider.py` 在非流式和流式两条路径都已提取 `reasoning_content`（见 `_parse_response` line 373/461、流式 line 522/540）
- `_ALLOWED_MSG_KEYS` 包含 `reasoning_content`，回填到 messages 不会被过滤

### 6.2 需验证
- **dashscope qwen3-thinking-latest**：response 是否在 `choices[0].message.reasoning_content`？是否 streaming delta 也带？
- **deepseek-r1 / deepseek-v3.1-thinking**：同上
- **若字段名不同**（如 dashscope 用 `thinking` 而非 `reasoning_content`）：在 `openai_compat_provider._parse_response()` 加 normalisation，兼容读 `reasoning_content` / `thinking` / `reasoning`，统一写到 `reasoning_content`。

### 6.3 工具调用格式
- `_extract_tc_extras()` 已经做了 SDK 对象 vs dict 的兼容，dashscope 的 `extra_content` 也已覆盖。
- 验证范围：确认 dashscope qwen-tool-use、deepseek tool calling 在并发场景下 `id` 字段稳定（`_short_tool_id` 已给 9 位 alphanumeric 防 Mistral 兼容问题）。

### 6.4 测试用例（验收用）
- 跑一次 dashscope qwen3 + 一个工具调用，断言 `LLMResponse.reasoning_content is not None and len > 0`
- 跑一次 deepseek-r1，断言同上
- 流式 + 非流式各跑一遍

---

## 7. 验收准则

### 7.1 功能
- [ ] `NANORESEARCH_MODE=server` 下，新用户没填 provider 时调聊天 → 422 + `missing_provider:chat`
- [ ] 用户填了 deepseek 的 chat key 但没填 embedding key，上传文档 → 422 + `missing_provider:embedding`
- [ ] 用户填了 dashscope 同时覆盖 chat + embedding，全功能可用
- [ ] `NANORESEARCH_MODE=local` 下（默认），现有 `config.json` 配置不动也能跑（向后兼容）

### 7.2 leak 验证
- [ ] worker 并发跑两个 uid（A 用 dashscope key1、B 用 dashscope key2）20 轮，每轮 assert 实际请求 header 用的是各自 key（用 monkeypatch + 断言）
- [ ] `os.environ` 在 worker 处理完一个用户请求后不被污染（断言 `OPENAI_API_KEY` 不出现在 env，除非启动前就有）

### 7.3 响应格式
- [ ] dashscope qwen3-thinking 推理内容到达前端
- [ ] deepseek-r1 同上
- [ ] 流式 / 非流式各覆盖

### 7.4 集成
- [ ] `python -m nanoresearch chat`（CLI 本地模式）跑通，沿用 config.json
- [ ] `pnpm dev` 起 server + frontend，按 5.1 / 5.2 手动验

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 删除 `_setup_env` 后，依赖 `os.environ[spec.env_key]` 隐式读 key 的第三方库失效 | 全文 grep 所有 env_key 读侧，确认只在 rag/libs/* 9 个文件；这 9 个已经在 4.3 范围内 |
| local 模式下行为与之前完全一致，但仍有用户期待 server 行为 | README + .env.example 写清 mode 默认值；server 部署的 systemd / docker compose 模板里强制 `NANORESEARCH_MODE=server` |
| dashscope thinking 字段名实际不叫 `reasoning_content` | 6.2 已留 normalisation 逻辑，实施时先跑探测脚本看响应原貌再写 |
| `user_settings.extra` JSONB 继续膨胀 | 不在本 Phase 范围，标到 Phase 6 拆 column |
| 加密缺失，DB 脱裤即 key 全泄露 | 已知风险，用户接受；future Phase 可加 fernet wrapper |

---

## 9. 不在范围

- 团队 / org / workspace 共享 key
- Pooled key + 按 token 计费 + quota
- key 加密（DB / Redis）
- key audit log（谁 / 何时改了 key）
- key 自动 rotation
- 前端"测试连接"按钮
- 删除 anthropic_provider / azure_openai_provider / openai_codex_provider backend 文件
- 拆 `user_settings.extra` JSONB 成独立 column / 表
- `providers/registry.py` 的 28 条 spec 改动

---

## 10. 工程量预估
- 后端：2-3 天（9 文件改 fallback + ModelFactory.mode 参数 + 错误结构化）
- 前端：1-2 天（settings 页提示 + 三个 gate）
- 响应格式验证 + fix：1 天（含跑探测 + 补 normalisation）
- 测试 + 文档：1 天
- 合计：5-7 天

---

**下一步**：用户审阅本 spec，没问题进 plan 阶段（`writing-plans` skill），把改动拆成 task 序列。
