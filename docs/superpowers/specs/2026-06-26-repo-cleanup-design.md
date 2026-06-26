# 仓库清理设计

**日期**：2026-06-26
**范围**：nanobot 仓库的存量噪音清理 + 命名/路径/多租户重构的路线图
**状态**：草稿，待用户审

---

## 1. 目标

- 让 `git status` 干净：消除"长期被追踪但应该忽略"的运行时数据和依赖
- 让 git 仓库不再积累"半临时"散落文件（一次性脚本、eval dump、草稿、个人文件）
- 让命名一致：完成已经开了头的 `nanobot → nanoresearch` 迁移
- 让用户配置位置可配：为多租户/服务端部署铺路
- 修复已识别的规范性信号（拆超大文件、双 tests 进 CI、归档老迁移脚本）

**约束**：所有动作不破坏当前工作区未暂存的源码修改（用户正在改 60+ 文件）。

---

## 2. 审计结论（实测事实）

### 2.1 规模

| 板块 | 文件 | 行数 |
|---|---|---|
| backend Python（含 channels） | ~286 | ~80,676 |
| 前端 web/src | 33 | 8,729 |
| 测试（根 tests + backend/tests） | 97 | 25,736 |
| 服务接入代码（providers / llm / embedding / agent tools / loop） | 45 | 8,936（占 11%） |

超过 1000 行的单文件 8 个：`agent_eval_router.py` (1685)、`cli/commands.py` (1559)、`mcp_server/tools/agentic/collections.py` (1406)、`channels/feishu.py` (1405)、`eval_router.py` (1233)、`hybrid_search.py` (1081)、`channels/weixin.py` (1033)、`cli/onboard.py` (1023)。

### 2.2 当前 git 状态

- 本地与 `origin/main` 对齐
- 暂存区空
- 工作区有 60+ 未暂存修改：含正在改的 backend 源码、69 个 web/node_modules 自动产物（vite/pnpm 缓存）、3 个 chroma 数据文件被运行时 touch

### 2.3 被追踪但应忽略的运行时数据（精确清单）

- `web/node_modules/`（58,609 文件；`.gitignore` 缺 `node_modules/` 规则）
- `backend/nanobot/nanobot/workspace/`（5 个 chroma 文件）
- `backend/nanobot/data/db/image_index.db`
- `backend/nanobot/data/db/ingestion_history.db`

**两个 .db 是历史死数据（实测）**：

| 文件 | 仓库内 | `~/.nanoresearch/rag/` |
|---|---|---|
| `image_index.db` | 20 KB，mtime May 11 | **659 KB，mtime Jun 18** |
| `ingestion_history.db` | 16 KB，mtime May 11 | **28 KB，mtime Jun 25** |

home 下版本当前还在写入；仓库内版本 5/11 后冻结。叠加 `image_storage` 和 `SQLiteIntegrityChecker` 全部 caller 都 `resolve_path("~/.nanoresearch/rag/...")` 显式传参 → 当前运行时不会再写到 `backend/nanobot/data/db/`。

**`.env`、`token.json`、`server.log` 未被追踪** ✅

**token.json 安全说明**：`token.json` 是 188 字节 untracked 文件，仅 .gitignore 加规则防未来。若内容是**真实有效凭证**，**用户自行检查**——一旦确认是真凭证，建议单独处理（作废、重发、密钥轮换），不要只靠 .gitignore 了事。本 spec 不主动查 git 历史确认它是否曾暴露（违反"不做考古"规矩）；这是用户决策项。

### 2.4 根目录散落文件分类

**无引用可删（19 个）**：
- 一次性脚本：`_clean_all.py`、`_fix_session_history.py`、`_print_badcases.py`、`_update_testcases.py`、`extract_chunks.py`、`core_agent_lines.sh`
- eval dump：`badcase_flows.json`、`snapshots_tmp.json`、`tc_list.json`、`extracted_chunks.json`、`eval_data_real.json`、`eval_output.txt`、`eval_qwen_max.log`、`eval_results_qwen_max.json`、`eval_results_real.json`、`test_intermediate_output.txt`、`test_retrieval_output.txt`
- 草稿：`health_set_draft.yaml` v1/v2/v3
- 个人：`resume_part1_agent.txt`、`resume_star.txt`
- 失效产物：`server.log`、`token.json`、`todolist.md`

**保留**：
- `seed_testcases.py`（被 `/api/eval/agent/testcases` 调用）
- `testcases.json`（被 seed 用）
- `loadtest.py`（Locust 压测）
- `README_old.md`（`docs/sdd/PHASE_STATUS.md` 还引用）
- `case/`（README 演示素材）

**待用户拍板**：`COMMUNICATION.md`、`SECURITY.md`、`bridge/`、根 `~/` 目录

### 2.5 双层路径 `backend/nanobot/nanobot/workspace/` 与 chroma 兜底

**实测**：本地 `~/.nanoresearch/settings.yaml` 配 `vector_store.persist_directory: ~/.nanoresearch/rag/chroma`，运行时数据写到 `C:\Users\Augix\.nanoresearch\rag\chroma`，**不会再写到仓库内的双层路径**。

**潜在隐患**：若用户没 settings.yaml，`chroma_store.py:105` 兜底 `./data/db/chroma` 会被 `resolve_path()` 解析为 `D:\Code\nanobot\backend\data\db\chroma`——还是写在源码树里。

**处理**：本次纯 `git rm --cached` 清历史数据 + 加 `.gitignore`，不改代码；兜底隐患放进 Phase 3 改名时一并改。

### 2.6 双 tests/ 职责

- **根 `tests/`**（81 文件 / 22.7K 行）：agent + channels + CLI + providers 单元/集成；**CI 在跑** (`pytest tests/`)
- **`backend/tests/`**（16 文件 / 3K 行）：DB + API + 评估 E2E；**不在 CI**；`conftest.py` 用 psycopg2 绕开 Windows asyncpg 兼容问题

**结论**：不是重复，是职责分离，都不能删。规范性问题（backend/tests/ 缺 CI）放 Phase 6。

### 2.7 改名 nanobot → nanoresearch（已半完成）

**已迁移**：
- pyproject.toml PyPI 名 `nanoresearch-ai`
- CLI 双命令 `nr` + `nanoresearch`（都指向 `nanobot.cli.commands:app`）
- 部分硬编码路径（agent_router.py: `~/.nanoresearch/workspace`）

**未迁移**：
- 461 条 `import nanobot` / `from nanobot.`
- 3 处动态导入：`importlib.import_module(f"nanobot.channels.{name}")`（registry.py:32、onboard.py:751、tests 1 处）
- entry_points group：`group="nanobot.channels"`（registry.py:42-50）
- 5 个 `NANOBOT_` 环境变量：`NANOBOT_MAX_CONCURRENT_REQUESTS`、`NANOBOT_PYTHON`、`NANOBOT_WORKSPACE`、`NANOBOT_TMUX_SOCKET_DIR`、`NANOBOT_` 前缀（config/schema.py Pydantic）
- `__main__.py` 入口（`python -m nanobot`）
- 文档 / Dockerfile / docker-compose 引用

### 2.8 用户配置路径

集中点：`backend/nanobot/config/paths.py`（`get_data_dir / get_logs_dir / get_workspace_path / get_cli_history_path / get_bridge_install_dir / get_legacy_sessions_dir`），base = `Path.home() / ".nanoresearch"`。

**绕过点**：
- `channels/qq.py:183,185`：硬编码 `Path.home() / ".nanoresearch" / "media" / "qq"`
- `channels/weixin.py:91`：硬编码 `~/.nanoresearch/weixin/`
- `chroma_store.py:105`：相对路径兜底 `./data/db/chroma`
- `image_storage.py:73-74`：默认值是 `~/...` 但 caller 都已显式 resolve_path（无功能 bug，仅潜在缺陷）

**多租户**：base 固定 `Path.home()`、无 tenant_id 维度。

### 2.9 channels/ 目录

16 文件 / 8,774 行的活跃多渠道接入：feishu (1405)、weixin (1033)、telegram (950)、mochat (947)、matrix (739)、qq (639)、dingtalk (580)、email (552)、discord (395)、wecom (371)、slack (344)、whatsapp (301) + base/registry/manager。

通过 `entry_points(group="nanobot.channels")` + 动态导入做插件发现。`qq.py`、`weixin.py` 有硬编码路径（详见 2.8）。

### 2.10 服务代码（清理时一律不动）

- `providers/`：9 文件 / 2.9K 行（OpenAI / Anthropic / Azure / Qwen）
- `rag/libs/llm/`：10 文件 / 2.3K 行
- `rag/libs/embedding/`：7 文件 / 1.0K 行
- `agent/tools/`：11 文件 / 2.6K 行（MCP / Web / FS / Shell）
- `agent/loop.py`：agent 主循环

合计 45 文件 / 8.9K 行。

---

## 3. 已对齐的决策

| 决策 | 结论 | 依据 |
|---|---|---|
| A：取消追踪 web/node_modules | **做** | 本地与 origin 对齐；69 个 node_modules 工作区改动 100% 是 pnpm/vite 自动产物；`.gitignore` 真没有 `node_modules/` 规则 |
| 双层路径是否改代码 | **不改**（Phase 1）；Phase 3 改 chroma 兜底默认值 | settings.yaml 实配 `~/.nanoresearch/rag/chroma`，当前代码不复发；兜底是独立隐患 |
| NANOBOT_ env var 改名 | **带兼容期改**（Phase 3） | 双读：新优先、旧 fallback + deprecation warning；定下线版本 |
| Phase 顺序 | 清理 → 改名 → 用户配置可配 → 多租户 → 规范性 | 避免改名时碰到清理债，避免多租户改架构时碰到命名债 |

---

## 4. 清理路线图

### Phase 1：最小动作（本 spec 唯一详细实施部分）

**目标**：`git status` 不再显示运行时数据；未来不会再误追踪 node_modules / workspace / db。

**动作**：
1. `.gitignore` 追加 5 条规则：
   ```
   node_modules/
   backend/nanobot/nanobot/workspace/
   backend/nanobot/data/db/
   *.log
   token.json
   ```
2. `git rm --cached` 已追踪的运行时数据：
   - `git rm --cached -r web/node_modules`
   - `git rm --cached -r backend/nanobot/nanobot/workspace`
   - `git rm --cached backend/nanobot/data/db/image_index.db`
   - `git rm --cached backend/nanobot/data/db/ingestion_history.db`
3. commit + push：**只 `git add .gitignore`**，不用 `git add .`（避免误带工作区源码修改）

**不在 Phase 1 范围**：
- 删根目录散落文件（Phase 2）
- 改任何源码
- 改名
- 修 `.gitignore` 里的 `*.claude` 疑似 bug（不阻塞 Phase 1）

**验证**：
- `git status` 不再出现任何 node_modules / workspace / data/db 路径
- `git push` 成功
- 本地 `pnpm dev`、`python -m nanobot` 等命令仍正常（磁盘文件未动）

### Phase 2：根目录散落文件清理（独立 spec）

删 §2.4 列出的 19 个无引用文件；保留 5 个有引用；用户拍板 4 个待确认项。

### Phase 3：nanobot → nanoresearch 全量改名（独立 spec）

- 包名 `backend/nanobot/` → `backend/nanoresearch/`
- 461 import 全量替换
- 3 处动态导入字符串
- entry_points group
- 5 个 `NANOBOT_` env var 带兼容期改名（双读、warning、下线版本）
- chroma_store.py 兜底默认值改 `~/.nanoresearch/rag/chroma`
- qq/weixin 硬编码路径改走 `get_media_dir()`
- 文档全量同步

### Phase 4：用户配置 base 路径可配（独立 spec）

`paths.py` 的 `Path.home()` 改成读环境变量 `NANORESEARCH_HOME`（默认 `~/.nanoresearch`），为多租户/服务端部署铺路。

### Phase 5：多租户 LLM API 配置（独立大 spec）

独立架构议题：providers 加 tenant scope、API key 隔离、配置加载机制。

### Phase 6（长期）：规范性整改

不是单次 spec，每次改动顺手做：
- 8 个 1000+ 行单文件按职责拆
- backend/tests/ 进 CI（解决 psycopg2/asyncpg 问题）
- RAG 目录层级减层
- 老迁移脚本 `backend/scripts/migrate_*.py` 归档

---

## 5. 风险与缓解（仅 Phase 1）

| 风险 | 缓解 |
|---|---|
| commit 误带工作区未暂存源码修改 | 只 `git add .gitignore`，不用 `git add .`；commit 前跑 `git status` 复核只有 .gitignore + 一堆 deleted |
| chroma / .db 数据文件被 `rm --cached` 后丢真数据 | 真数据全在 `~/.nanoresearch/rag/`（实测仓库内版本 5/11 后冻结、home 版本当前在用）；`--cached` 不动磁盘 |
| `.gitignore` 新 `*.log` 规则与 `botpy.log` 重复 | 重复无冲突，可后续清理 |
| node_modules 删了 pnpm 失效 | `--cached` 只动 index 不动磁盘；即便用户重装 pnpm 也能完整恢复 |
| push 失败 | 失败就停，排查后再重试 |
| token.json 是真凭证 | Phase 1 仅 .gitignore 防未来；用户单独确认是否需要作废重发 |

---

## 6. 验收（Phase 1）

执行完后：
- `git status` 输出不含 `node_modules` / `workspace/rag_data` / `data/db/*.db` 任一路径
- `git log -1 --stat` 显示一次 commit，含 `.gitignore` 1 个修改 + 大量 `deleted:` 行
- `git push origin main` exit 0
- 本地工作区其余 60+ 未暂存修改保持原状
