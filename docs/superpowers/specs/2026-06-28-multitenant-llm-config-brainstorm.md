# Phase 5 多租户 LLM API 配置 — Brainstorm 草稿

状态：brainstorm（**不是** 最终 spec），写给用户做方向决策用。
日期：2026-06-28
前置：Phase 3（nanoresearch 包名）+ Phase 4（NANORESEARCH_HOME）已完成。

---

## 0. 现状摘要：provider 配置链路

### 0.1 三层数据源（按优先级）

读取链路在 `backend/nanoresearch/providers/model_factory.py:ModelFactory.resolve()`，按角色（CHAT / INGESTION_LLM / EMBEDDING / VISION / EVAL_GENERATOR / EVAL_EVALUATOR）派发，每个角色的逻辑略不同，但优先级一致：

1. **per-user**：`user_settings.extra.providers`（JSONB）→ `[{id, name, api_key, api_base, models[]}]`
2. **system**：`config.json` 顶层 `providers.{name}.{api_key, api_base, extra_headers}`（pydantic `ProvidersConfig`，28 个 provider）
3. **RAG fallback**：`settings.yaml` 的 `llm` / `embedding` / `vision_llm` 段（仅 RAG 相关角色）

### 0.2 provider registry

`providers/registry.py:PROVIDERS` 是 28 条 `ProviderSpec` 的元数据表（identity、keywords、env_key、default_api_base、gateway/local flags、prompt caching 支持）。

`Config._match_provider(model)` 用它做匹配：
- 显式 `agents.defaults.provider != "auto"` → 强制走指定 provider
- `auto` → 按 model name 前缀（"openai/..."）→ keyword 命中（"claude" → anthropic）→ 本地 fallback（base_url 含 "11434" → ollama）→ gateway/standard fallback

### 0.3 ModelFactory 已经在哪些 seam 调用

| 调用点 | 角色 | 现状 |
|---|---|---|
| `worker.py:104` | CHAT | 取 uid → user_providers → resolve → 构造 provider for AgentLoop |
| `worker.py:487` | INGESTION_LLM | KB 文档 ingest job，per-uid 已通 |
| `server/routers/knowledge_router.py:52` `_resolve_rag_settings()` | INGESTION_LLM 默认 | KB 查询/管理 endpoint，per-uid |
| `server/routers/eval_router.py:55` `_resolve_eval_spec()` | EVAL_GENERATOR/EVALUATOR | per-uid |
| `server/routers/eval_router.py:1019` | EVAL_* | per-uid |

**重要观察**：per-uid 路径已经全量打通。Phase 5 不是"加 tenant scope"，是"修当前 tenant scope 的漏洞和边界"。

### 0.4 settings/API 层

- `GET/PUT /api/settings/me`（`settings_router.py`）已支持 provider CRUD：`ProviderIn{id, name, api_key, api_base, models[]}`
- 返回时 `_mask_providers()` 做掩码（露后 4 位）
- `_merge_providers()` 语义：api_key=`None` 保留旧值、`""` 清空、非空更新

### 0.5 持久化与缓存

- DB：`user_settings.extra` JSONB，**api_key 明文存储**
- Redis：`UserSettingsRepository` 用 hash 缓存（`RedisKeys.user_settings(uid)`），api_key **明文 mirror** 到 Redis
- upsert 后 DEL Redis key（写策略一致）

### 0.6 已识别的多租户漏洞（critical findings）

#### F1. 进程级 env var 污染（最严重）
`providers/openai_compat_provider.py:140 _setup_env()`：构造 provider 时把 `api_key` 写入 `os.environ[spec.env_key]`：
- gateway：直接 `os.environ[spec.env_key] = api_key`（覆盖）
- 非 gateway：`setdefault`（先写入者占坑）
- `env_extras` 也用 `setdefault`

**后果**：worker 进程并发处理多 uid 任务时，user A 的 key 会污染 user B 的 env；后续依赖 env 兜底的代码（见 F2）读到错的 key。

#### F2. 大量 env var 兜底，绕过 ModelFactory
8 个文件 fallback 到 `os.environ.get("OPENAI_API_KEY"/"DASHSCOPE_API_KEY"/"AZURE_OPENAI_API_KEY")`：

```
worker.py:215                                   simple-RAG 流式
server/routers/eval_router.py:162,222,310       ragas / agent eval
rag/libs/embedding/openai_embedding.py:72       openai embedding 客户端
rag/libs/embedding/dashscope_embedding.py:49    dashscope embedding
rag/libs/embedding/azure_embedding.py:76        azure embedding
rag/libs/llm/openai_llm.py:72                   openai LLM
rag/libs/llm/openai_vision_llm.py:102           openai vision
rag/libs/llm/azure_llm.py:79                    azure LLM
rag/libs/llm/azure_vision_llm.py:120            azure vision
```

这些 fallback 在 SaaS 部署下都是 leak vector：用户 token 用完 → 自动用主机 env 的 key → 计费走主机账户。

#### F3. api_key 明文 at rest
- Postgres JSONB 明文
- Redis hash 明文
- 没有 KMS / Vault / fernet / sealed secret 包装
- 没有 audit log（谁/何时改了 key）

#### F4. 无 org / workspace 层
"tenant" 现在等于 "user"。多人团队共用一组 key 的场景需要每人各自填一份，没有"team owner 设 key，team 成员共用"的路径。

#### F5. 无 quota / 计费分账
没有 per-uid token 计数器、没有月度上限。SaaS 化必须的元数据缺失。

#### F6. config.json `providers` 块语义模糊
单租户时是"系统默认 key"；多租户 SaaS 时它代表什么？
- 选项 a：系统级 fallback（用户没填就用系统的）— 但这就是 F2 的 leak
- 选项 b：admin / 系统任务专用（用户路径完全不读）
- 选项 c：开发模式残留，生产部署强制移除

---

## 1. 用户故事

按部署形态分两类：

### A. 单机 / 本地开发（现状）
- A1. 我是开发者，本地跑 `nanoresearch`，config.json 填 OpenAI key，所有功能直接能用。
- A2. 我是本地用户，不想动 config.json，在 Web UI `/api/settings/me` 填 key 也能用。

### B. 多租户 SaaS（目标）
- B1. 我是 SaaS 平台运营者，想让多个独立用户共用一份部署，**每人自带 key（BYOK）**，互不可见、互不计费。
- B2. 我是 SaaS 用户，怀疑 key 泄露 → 想知道何时被读过、轮换 key 的成本是什么。
- B3. 我是企业 admin，想给团队 5 人共用一组公司账号 key（org-level 共享）。
- B4. 我是 SaaS 平台运营者，**想自己持有池化 key**，按 token 向用户计费（Pooled），用户不见 key。
- B5. 我是合规审计员，要求 api_key 静态加密，DB dump 出来直接读不出明文。

**Phase 5 范围决策点**：B1 + B5 是底线（多租户 SaaS 必须）。B3（org）、B4（pooled+计费）是要不要这一期做的判断题。

---

## 2. 隔离边界选项

四个选项可以独立 / 组合：

### O1. 修 F1+F2：去 env-var 污染 + 关 env 兜底
**做法**：
- `openai_compat_provider._setup_env()` 改成 per-instance（不写 `os.environ`），把 key 注入 client 构造（OpenAI SDK 已经支持 `api_key=` 参数，不需要 env）
- 9 个文件的 `os.environ.get(...)` fallback 删掉或加 flag `NANORESEARCH_ALLOW_ENV_FALLBACK=1`，默认关闭
- env_extras 改成 per-client header / config 注入

**收益**：根除 cross-uid 串号、终止主机 key 隐性兜底
**风险**：现有本地用户依赖 env 跑的会断
**估算**：~2-3 天（9 文件 + 测试）

### O2. 加 secret-at-rest 加密层
**做法**：
- `extra.providers[].api_key` 改存 envelope-encrypted blob（fernet / age / KMS）
- `NANORESEARCH_SECRET_KEY` 或 KMS endpoint env var
- `UserSettingsRepository` 加 encrypt-on-write / decrypt-on-read
- Redis cache 存密文（或干脆 cache miss 时不缓存 key 字段）

**收益**：DB dump 不再 = 凭证泄露
**风险**：key rotation / 升级路径要设计；丢 master key 等于全员 key 丢
**估算**：~3-5 天

### O3. 引入 org / workspace 层（B3）
**做法**：
- 新表 `orgs` + `org_members(org_id, uid, role)`
- `org_settings.providers` 同结构 JSONB
- `ModelFactory.resolve()` 加一层 `org_providers`：user > org > config.json > settings.yaml
- API：`/api/orgs/{id}/settings` 仅 owner/admin 可写

**收益**：覆盖 B3
**风险**：DB migration、auth middleware 改、UI 改，工程量大
**估算**：~1-2 周

### O4. Pooled key + 计费分账（B4）
**做法**：
- 新增 `usage_ledger(uid, ts, model, prompt_tokens, completion_tokens, cost_cents)` 表
- 每次 LLM 调用结束写一行（token 已经在 `LLMResponse.usage` 里）
- 月度 quota 表 + 强制 gate
- Pooled key 标记 `is_platform_key=True`，用户不可见

**收益**：覆盖 B4
**风险**：定价表维护、追溯老数据、压测元数据写入开销
**估算**：~2 周

### 我的推荐
**O1 + O2 必做，O3/O4 单独 Phase**。

理由：
- O1 是 bug fix 性质，不修就不是真多租户，工程量小
- O2 是合规底线，做完才能说"key 隔离"，工程量中
- O3 是 product 决策（要不要做团队产品），不是基建。先 ship 个人 BYOK，O3 看商业化节奏
- O4 同 O3，且 quota/计费往往牵涉前端、运营后台，不是单 spec 能装下

---

## 3. 配置层级方案

### L1. 现状（保持）
```
user_providers > config.providers > rag_settings.{llm,embedding,vision_llm} > env var fallback
```

### L2. 推荐方案
```
user_providers (per-uid, encrypted)
    > rag_settings (system fallback for RAG-only roles)
    > config.providers (DEV mode only, 生产部署不读)
    > env var (默认关闭，NANORESEARCH_ALLOW_ENV_FALLBACK=1 才开)
```

变化：
- 把 `config.json` 的 `providers` 块语义降为"开发模式默认"；引入 `NANORESEARCH_MODE=server|local` 开关，server 模式下硬禁
- env fallback 默认 off，本地开发可开 flag
- 给 ModelFactory 加 `tenant_id` 参数（先用 uid，未来可换 org_id）

### L3. 完整租户路径
```
user_providers > org_providers > rag_settings > config.providers > env
```
等 O3 上时再加。

---

## 4. 主要权衡

| 决策 | 选 X 的代价 | 选 Y 的代价 |
|---|---|---|
| 范围：只修 O1+O2 vs 全做 | 多租户不完整（无 org/pooled） | 工期 4 周+、交付风险大、UI 重做 |
| env fallback：删 vs 留 flag | 本地 dev 跑不通 | 多了一个生产部署 footgun |
| 加密：fernet（应用层）vs KMS | 简单，master key 自管理 | 依赖外部服务，部署门槛↑ |
| config.json providers：保留 vs 移除 | 单机用户友好 | server 模式下又是 leak vector |
| Pydantic schema：扩字段 vs 拆库 | 兼容期短，迁移代码少 | UserSettings.extra 越塞越大，未来一定要拆 |
| user_settings cache：缓存密文 vs 不缓存 key 字段 | 实现简单 | 每次 chat 都查 DB（小开销） |

---

## 5. 待用户决策的问题

按重要度排序，逐题确认（不要一次答完）：

1. **Phase 5 范围**：选 O1+O2（个人 BYOK 完成）还是 O1+O2+O3（含 org）还是全做？
2. **加密**：应用层 fernet（master key 自管）还是外部 KMS？还是先不加密，留 P6？
3. **env var fallback**：默认关 + flag 开 vs 完全删除？
4. **config.json providers**：保留作 dev 默认 vs `NANORESEARCH_MODE` 切换 vs 移除？
5. **api_key Redis 缓存**：缓存密文 vs 不缓存（每次查 DB）？
6. **pool key / 计费分账**：放 Phase 5 内 vs Phase 7（独立 spec）？

---

## 6. 不做什么（YAGNI）

- 不做前端 UI 改造（除非影响 API 契约）
- 不做用户 quota gate（除非选 O4）
- 不做 provider 元数据 admin 后台
- 不动 `providers/registry.py` 的 28 条 spec
- 不动 `ModelFactory` 的 6 个角色分发逻辑（只在边界注入新参数）

---

**下一步**：等用户答 §5 的 6 个问题（一次一个），然后我把 brainstorm 收敛成正式 design spec 文档。
