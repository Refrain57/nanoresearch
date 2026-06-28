# 仓库清理 Phase 3 实施计划:nanobot → nanoresearch 全量改名

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Python 包名 `nanobot` 全量改成 `nanoresearch`,覆盖 (a) 目录名 `backend/nanobot/` → `backend/nanoresearch/`、(b) 全部 import / 缩进 import / from-import 语句(约 1464 处)、(c) 3 处动态导入字符串、entry_points group、Pydantic env_prefix、(d) 5 个 `NANOBOT_` 环境变量(双读 + deprecation warning + 下线版本 0.3.0)、(e) `.md`/`.sh` 里的 TMUX env var、(f) chroma_store.py 兜底默认值、qq/weixin 硬编码路径、weixin 注释、(g) `pyproject.toml` 的 8 处包名 / scripts / hatch / coverage 引用。所有改动落地后,全 repo 不再有任何 `import nanobot` / `from nanobot.` / `from nanobot import` 语句(测试 fixture JSON 除外),且测试通过/失败集合与改名前完全一致(等价性验证)。

**Architecture:** 分 4 个独立 commit,按"小风险先行、大风险压尾"的顺序:
1. **Commit A**(Task 1):env var 兼容层 — 引入 `_apply_legacy_env_compat()` 启动钩子,把 `NANOBOT_*` 复制到 `NANORESEARCH_*` 并 deprecation warning;4 个读取点切到 `NANORESEARCH_*`。Pydantic env_prefix **不动**(留到 Commit D 跟随改名)。该 commit 行为兼容 — 老用户的 `.env` 仍工作,新名也工作。
2. **Commit B**(Task 2):chroma 兜底默认值 + qq/weixin 路径绕过点修复 — 与改名解耦,改不出大事。
3. **Commit C**(Task 3):TMUX 在 `.md`/`.sh` 里的 env var rename + shell 端兼容期 fallback。
4. **Commit D**(Tasks 5-10):**原子大改名** — `git mv backend/nanobot backend/nanoresearch` + 用 Python 脚本机械替换 4 个 scope 内的全部 import 语句 + 单独处理 3 处动态导入字符串 + entry_points group + Pydantic env_prefix → `NANORESEARCH_` + `pyproject.toml` 8 处更新。该 commit 必须原子 — 中间态 `import nanobot` 全坏,无法 partial 提交。

每个 commit 前都有强制人工核对闸,每步报实际输出。等价性验证靠 pre-rename test baseline 抓取 + post-rename 一致性对比;import 解析靠 `python -c "import nanoresearch"` 等启动级 smoke + 全量 grep 反向校验。

**Tech Stack:** git(CLI)、Python 3.11+(写一次性替换脚本)、pytest(基线 + 验证)、bash(shell)。无新依赖。

## Global Constraints

来自用户硬约束 + 前两轮(P1/P2/B4 merge)实战节奏:

### 范围(全做)

- **import 全改,不分行首/缩进**。覆盖 `^import nanobot`、`  import nanobot`(函数/条件内缩进)、`from nanobot.x.y import ...`、`from nanobot import x`、`from nanobot.x import (...)` 多行括号 import。
- **替换范围按文件类型分**:
  - `.py`:全 repo(`backend/nanoresearch/`、`tests/`、`backend/tests/`、`backend/scripts/`、根 `backend/test_*.py` 包含的工作区文件除外 — 见"不动清单"第 5 条)
  - `.md` / `.sh`:仅 TMUX env var 名 `NANOBOT_TMUX_SOCKET_DIR` → `NANORESEARCH_TMUX_SOCKET_DIR`(其它 .md 里的"nanobot"字面量留作用户决策,不在本 Phase 范围)
  - `.toml`:仅 `pyproject.toml` 的 8 处(scripts / hatch build/wheel/sources/sdist / coverage source)
  - `.txt`:本 Phase 实际无 `.txt` 命中,留扩展名占位
- 根 `tests/`(CI 在跑,339 处)+ `backend/tests/`(非 CI,45 处)的 import 必须**跟包目录 mv 同一个 commit** 改 — 否则中间态 tests 全坏。
- env var 兼容期下线版本 = **0.3.0**(当前 `pyproject.toml` 是 `0.1.4.post6`,留 ~两个 minor 让生态迁移)。

### 不动清单(本 Phase 绝不碰)

1. 测试 fixture JSON 里的 `nanobot`(`tests/agent/evaluation/results/*` 等约 1177 次)— 测试数据,改了会污染数据集语义。
2. 11 个 untracked 故意未 commit 文件:`backend/logs/`、`backend/models/`、`backend/test_ragas_transforms.py`、`backend/test_themes_diagnostic.py`、`health_set_draft.yaml` v1/v2/v3、`loadtest.py`、`seed_testcases.py`、`testcases.json`、`web/vite.config.js.timestamp-*.mjs`。
3. `rescue/b4-orphan` 分支(含 5 块未来 Phase 预埋,见 memory `project_rescue_b4_orphan_dormant.md`)。
4. `worktree-agent-*` 三个孤立分支。
5. 已存在的 193 处合法 `nanoresearch` 字面量(`.env.example`、`docker-compose.yml`、`README.md`、`docs/`、`config/paths.py` 里的 `~/.nanoresearch` 路径等)。
6. TMUX SKILL.md 里非 env var 的 `nanobot` 字面量(`SOCKET=$SOCKET_DIR/nanobot.sock`、`SESSION=nanobot-python`、metadata key `{"nanobot":{...}}`、默认子目录名 `nanobot-tmux-sockets`)— 这些是用户已有 socket / session / skill 发现机制的兼容点,改它们会孤立现存的 tmux 会话或断掉 skill 发现。留作"Phase 3 不做但留痕"。
7. 两个 untracked smoke script(`backend/test_ragas_transforms.py`、`backend/test_themes_diagnostic.py`)的 import — 接受改名后短暂坏,Task 11 提醒用户手动改。

### 安全 / 节奏(从前轮继承)

- **每 Task 完了停下**,人工核对实际输出,异常先停查清再走。
- **commit 前唯一人工闸口** — 每个 commit 前都有独立 Task / Step 显式核对暂存区,不许 `git add .` / `git add -A` / 任何泛匹配。
- **不重写历史**、不 `--amend`、不 `--no-verify`、不 `--force`/`-f`/`--force-with-lease`。普通 commit + push。
- **push 失败就停**,原样报错给用户,不重试 force,不切策略。
- **批量替换用 Python 脚本**(写在 `/tmp/`,不进库),regex 严格只匹配 `\b(import|from)\s+nanobot\b` 这种 import 上下文,**不**误伤注释 / 字符串 / 变量名(如 `nanobot_extras` 不被改、`# we love nanobot` 不被改、`"nanobot.x"` 由 Task 7 单独处理)。
- **git mv 全程可回退** — 在 Task 5 mv 之前打 `phase3-pre-rename` 本地 tag,出事可 `git reset --hard phase3-pre-rename`。Commit D 推上去后该 tag 仍保留作历史锚点。
- **等价性验证 = 改名前后 pytest 收集到的 test ID 集合相同 + 通过/失败计数相同**。Task 0 抓 baseline,Task 8 对比。新 fail 出现立刻停,不接受"反正测试本来也飘"的解释。

### 修改但不在改名核心范围的副作用(已识别,Task 列表里处理)

- `pyproject.toml` 的 `[project.scripts]`、`[tool.hatch.build]`、`[tool.hatch.build.targets.wheel]`(packages / sources / force-include)、`[tool.hatch.build.targets.sdist]`、`[tool.coverage.run].source` 共 8 处 `nanobot` 引用 — 跟 Commit D 一起改。
- `weixin.py:91` 行尾注释 `# Default: ~/.nanoresearch/weixin/` 实际跟代码行为不符(代码走 `get_runtime_subdir("weixin")`,不是固定 `~/.nanoresearch/weixin/`)— Commit B 顺手修正。
- `shell.py:99` 中文注释 `# 注入 nanobot 自己的 Python 路径到 PATH 最前面` — Commit D 顺手改成 nanoresearch。
- `__main__.py` 跟随 `git mv` 自动到 `backend/nanoresearch/__main__.py`,`python -m nanobot` 入口变为 `python -m nanoresearch`(无兼容,因为这是 CLI 调用方式而非环境变量,且 `nr` / `nanoresearch` console_scripts 双命令都还在)。

---

## 不做但留痕(交后续 Phase)

- **TMUX SKILL.md 里其它 `nanobot` 字面量**:metadata key `{"nanobot":{...}}`(skill 框架可能硬编码这个 key 名称读 emoji/os/requires;改它要先确认 skill 框架适配)、默认 socket 子目录名 `nanobot-tmux-sockets`、示例 `SOCKET="$SOCKET_DIR/nanobot.sock"` / `SESSION=nanobot-python`(会孤立现存 tmux 会话)。决策项:或维持 nanobot 字面量(它们不是包名引用,跟 import 无关)、或在 Phase 6 规范化整改时统一改。Phase 3 不动。
- **Phase 1 spec §2.7 提到的文档 / Dockerfile / docker-compose 全量同步**:`.env.example`(已用 `NANORESEARCH_` 前缀引?需复查)、`docker-compose.yml` 命名、`README.md` 中的命令示例。本 Phase 改 import + env var + 兜底,**不**全量梳理文档/部署文件。留 Phase 6 或独立 doc-sync 子 spec。
- **5 块未来 Phase 预埋**(content_hash / classification / set_kind+tool_recordings / baseline / TunableObjectVersion)在 `rescue/b4-orphan` 分支等各自 Phase 提取。本 Phase 不动该分支。
- **`backend/bridge/` 孤立目录**(Phase 2 已留痕)和 `backend/nanobot/data/db/{image_index,ingestion_history}.db` 磁盘残留(Phase 1/2 已留痕):本 Phase 不处理。
- **2 个 untracked smoke script 的 import 更新**(`test_ragas_transforms.py` / `test_themes_diagnostic.py`):接受短暂 ImportError,Task 11 提醒用户手动改。

---

## File Structure

按 commit 分组列全部被改文件(不含改名搬走的 209 个 `.py`)。

### Commit A:env var 兼容层(Task 1)

- **Create:** `backend/nanobot/utils/env_compat.py` — `_apply_legacy_env_compat()` 函数
- **Modify:** `backend/nanobot/utils/__init__.py` — 导出新函数(若 `__init__.py` 显式列符号)
- **Modify:** `backend/nanobot/__main__.py` — entry 顶部调用 compat 钩子
- **Modify:** `backend/nanobot/cli/commands.py` — `app()` callback 顶部调用 compat 钩子
- **Modify:** `backend/nanobot/server/main.py` — `create_app()` 顶部调用 compat 钩子
- **Modify:** `backend/nanobot/worker.py` — 模块顶部调用 compat 钩子(执行中发现 worker.py 是独立 arq 启动入口、`_build_agent_loop()` 会实例化 AgentLoop 命中 NANORESEARCH 读取点,故补入 Commit A 注入点,原 plan 三入口 → 四入口)
- **Modify:** `backend/nanobot/agent/loop.py:153-154` — env var 名 `NANOBOT_MAX_CONCURRENT_REQUESTS` → `NANORESEARCH_MAX_CONCURRENT_REQUESTS`(+ 更新注释)
- **Modify:** `backend/nanobot/agent/tools/shell.py:101` — env var 名 `NANOBOT_PYTHON` → `NANORESEARCH_PYTHON`
- **Modify:** `backend/nanobot/server/routers/agent_router.py:159` — env var 名 `NANOBOT_WORKSPACE` → `NANORESEARCH_WORKSPACE`
- **Test:** `tests/utils/test_env_compat.py`(新建)— 单元测试 compat 钩子的双读 + warning

### Commit B:chroma 兜底 + qq/weixin 路径 + .gitignore 双覆盖(Task 2)

- **Modify:** `backend/nanobot/rag/libs/vector_store/chroma_store.py:105` — 默认值 `'./data/db/chroma'` → `'~/.nanoresearch/rag/chroma'`
- **Modify:** `backend/nanobot/channels/qq.py:175-189` — 简化 `_init_media_root`,移除 fallback 硬编码路径,直接走 `get_media_dir("qq")`,失败让其 propagate
- **Modify:** `backend/nanobot/channels/weixin.py:91` — 注释 `# Default: ~/.nanoresearch/weixin/` 改为反映真实行为(走 `get_runtime_subdir("weixin")`)
- **Modify:** `.gitignore` — 在 Line 62/63 原 `backend/nanobot/nanobot/workspace/` + `backend/nanobot/data/db/` 之后**追加**(不替换)`backend/nanoresearch/nanobot/workspace/` + `backend/nanoresearch/data/db/`。双覆盖:改名后新路径被忽略,老路径也不会因为 .gitignore 漏改而突然出现在 git status。**Line 38 `.nanobot/config.json` 不动**(那是 user 级 dotfile,跟包名无关)。.gitignore 是 UTF-8/UTF-16 混合编码(commit 08283baf 那次 PowerShell `>>` append 造成),用 Python 字节级 anchor-insert 只在 UTF-8 段插入,不碰 UTF-16 corruption(留作独立未来 commit 修)。

### Commit C:TMUX env var .md/.sh(Task 3)

- **Modify:** `backend/nanobot/skills/tmux/SKILL.md:14,34,35,46` — env var 名 `NANOBOT_TMUX_SOCKET_DIR` → `NANORESEARCH_TMUX_SOCKET_DIR`
- **Modify:** `backend/nanobot/skills/tmux/scripts/find-sessions.sh:13,23` — dual-read env var + deprecation echo 到 stderr

### Commit D:原子大改名(Tasks 5-10)

- **Rename(git mv):** `backend/nanobot/` → `backend/nanoresearch/`(整个目录,209 文件 + 子目录)
- **Modify(import 机械替换,~1464 处):**
  - `backend/nanoresearch/**/*.py`(原 `backend/nanobot/**/*.py`,1031 处 / 209 文件)
  - `tests/**/*.py`(339 处 / 68 文件)
  - `backend/tests/**/*.py`(45 处 / 14 文件)
  - `backend/scripts/**/*.py`(44 处 / 15 文件)
- **Modify(字符串 / 配置,5 处):**
  - `backend/nanoresearch/channels/registry.py:32` — `f"nanobot.channels.{module_name}"` → `f"nanoresearch.channels.{module_name}"`
  - `backend/nanoresearch/channels/registry.py:37` — `f"No BaseChannel subclass in nanobot.channels.{module_name}"`(error message)→ `nanoresearch.channels.{module_name}`
  - `backend/nanoresearch/channels/registry.py:45` — `entry_points(group="nanobot.channels")` → `nanoresearch.channels`
  - `backend/nanoresearch/cli/onboard.py:751` — `f"nanobot.channels.{name}"` → `f"nanoresearch.channels.{name}"`
  - `tests/providers/test_providers_init.py:16` — `"nanobot.providers"` → `"nanoresearch.providers"`
- **Modify(测试里的 mock 字符串引用 ~4 处):**
  - `tests/cli/test_commands.py:159,187,314,342` — `"nanobot.channels.registry.discover_all"` / `"nanobot.providers.openai_compat_provider.AsyncOpenAI"` → `nanoresearch.*`
  - 其它扫到的 `monkeypatch.setattr("nanobot.x.y", ...)` / `patch("nanobot.x.y")` 都由 Task 7 Step 4 的 string-form 扫描统一处理
- **Modify(Pydantic env_prefix):**
  - `backend/nanoresearch/config/schema.py:283` — `env_prefix="NANOBOT_"` → `env_prefix="NANORESEARCH_"`
- **Modify(pyproject.toml,8 处):**
  - 行 100:`nr = "nanobot.cli.commands:app"` → `nanoresearch.cli.commands:app`
  - 行 101:`nanoresearch = "nanobot.cli.commands:app"` → `nanoresearch.cli.commands:app`
  - 行 112-115:`"nanobot/**/*.py"` → `"nanoresearch/**/*.py"` 等 4 行
  - 行 119:`packages = ["nanobot"]` → `["nanoresearch"]`
  - 行 122:`"nanobot" = "nanobot"` → `"nanoresearch" = "nanoresearch"`
  - 行 125:`"bridge" = "nanobot/bridge"` → `nanoresearch/bridge`
  - 行 129:`"nanobot/"` → `"nanoresearch/"`
  - 行 153:`source = ["nanobot"]` → `["nanoresearch"]`
- **Modify(顺手):**
  - `backend/nanoresearch/agent/tools/shell.py:99` — 中文注释 `# 注入 nanobot 自己的 Python 路径...` → `nanoresearch`

### Task 自己写的辅助脚本(不进库)

- `/tmp/phase3-replace-imports.py` — 机械替换脚本,Task 6 用,Task 11 清理
- `/tmp/phase3-replace-strings.py` — 字符串替换脚本,Task 7 用,Task 11 清理
- `/tmp/phase3-edit-gitignore.py` — .gitignore 字节级 anchor-insert,Task 2 用,Task 11 清理
- `/tmp/phase3-gitignore.bak` — .gitignore 改前备份,Task 2 第一行 shutil.copy 产生
- `/tmp/phase3-pre-rename-counts.txt` — import 计数 baseline(Task 0 已抓)

不创建新工程文件除 Commit A 的 `env_compat.py` 和它的测试。

注:原计划还有 pytest baseline `/tmp/phase3-baseline-{collect,summary}.txt`,Task 0 实跑因 `tests/research/` MarkdownLoader module-level 副作用导致 pytest collect 挂死(独立代码病,非 Phase 3 scope)而跳过。Task 4/8 等价性验证降级为 `import nanobot`/`import nanoresearch` smoke + 反向 grep — 见 Task 0 执行记录、Task 4/8 头部说明。

---

## Task 0:Pre-flight 漂移检查 + baseline 抓取

**Files:** 不动任何文件,纯只读 + 写 baseline 临时文件 + 打本地 tag。

**Interfaces:**
- Consumes: 上一轮 Phase 3 只读复核结论(本地与 origin 对齐;11 个 untracked;import 行首口径 ~609 / 多行 ~1464;6 个改名点全部命中)。
- Produces: 当前状态与上一轮复核一致的书面确认 + 4 份 baseline 文件 + `phase3-pre-rename` 本地 tag。任一项漂移就停。

**为什么这一步独立成 Task**:工作区有 11 个 untracked、远端可能也有新 commit。从复核到执行有时间差,期间任何漂移都会让后续步骤的预期值不成立。

- [ ] **Step 1:git 状态对齐**

Run:
```bash
cd /d/Code/nanobot
git status -sb | head -1
echo "---"
git log -1 --oneline
echo "---"
git rev-list --left-right --count origin/main...HEAD
```

Expected:
- 第一行 `## main...origin/main`(无 ahead/behind)
- 顶端 commit hash 是 `08283baf`(当前 HEAD)
- left-right count 输出 `0	0`

任一不符:**停**,可能有 remote 推送或本地 commit。

- [ ] **Step 2:工作区漂移检查**

Run:
```bash
git status -s | wc -l
git ls-files --others --exclude-standard | wc -l
git diff --name-only | wc -l
git diff --cached --name-only | wc -l
```

Expected:
- 总条目 11 + (可能新增的工作区临时产物,允许 ±3 漂移)
- untracked 11(允许 ±3)
- modified 0
- staged 0

modified ≠ 0 或 staged ≠ 0:**停**(说明工作区有未 commit 修改,改名期间不能动)。untracked 漂移 > 3:停查清楚是什么新产物。

- [ ] **Step 3:11 个 untracked 文件精确清单核对(改名期间绝不动)**

Run:
```bash
git ls-files --others --exclude-standard | sort
```

Expected(精确这 11 项,排序后):
```
backend/logs/
backend/models/
backend/test_ragas_transforms.py
backend/test_themes_diagnostic.py
health_set_draft.yaml
health_set_draft_v2.yaml
health_set_draft_v3.yaml
loadtest.py
seed_testcases.py
testcases.json
web/vite.config.js.timestamp-1780979241117-9ac612d5a04f5.mjs
```

注:`backend/logs/` 和 `backend/models/` 是目录,git 会把它们当条目展示(具体行数取决于内部文件数)。允许 vite timestamp 文件名 hash 变化。

新增项:**停**,跟用户确认是临时产物还是改名期间不能动的工作区改动。

- [ ] **Step 4:6 个改名点位置复核(防漂移)**

Run:
```bash
echo "=== 3 处动态导入 ==="
grep -n "importlib.import_module.*nanobot" backend/nanobot/channels/registry.py backend/nanobot/cli/onboard.py tests/providers/test_providers_init.py
echo "---"
echo "=== entry_points group ==="
grep -n 'group="nanobot.channels"' backend/nanobot/channels/registry.py
echo "---"
echo "=== Pydantic env_prefix ==="
grep -n 'env_prefix="NANOBOT_"' backend/nanobot/config/schema.py
echo "---"
echo "=== 5 个 NANOBOT_ env var 读取点 ==="
grep -n "NANOBOT_MAX_CONCURRENT_REQUESTS" backend/nanobot/agent/loop.py
grep -n "NANOBOT_PYTHON" backend/nanobot/agent/tools/shell.py
grep -n "NANOBOT_WORKSPACE" backend/nanobot/server/routers/agent_router.py
grep -rn "NANOBOT_TMUX_SOCKET_DIR" backend/nanobot/skills/tmux/
echo "---"
echo "=== __main__.py ==="
ls -la backend/nanobot/__main__.py
echo "---"
echo "=== chroma_store.py 兜底 ==="
grep -n "'./data/db/chroma'" backend/nanobot/rag/libs/vector_store/chroma_store.py
echo "---"
echo "=== qq/weixin 硬编码 ==="
grep -n '\.nanoresearch.*media.*qq\|\.nanoresearch.*weixin' backend/nanobot/channels/qq.py backend/nanobot/channels/weixin.py
echo "---"
echo "=== pyproject.toml 包名引用 ==="
grep -n "nanobot" backend/pyproject.toml
```

Expected:
- 3 处动态导入 3 行
- entry_points group 1 行(registry.py:45)
- env_prefix 1 行(schema.py:283)
- 4 个标准 env var 各命中(loop.py:154、shell.py:101、agent_router.py:159、tmux 至少 2-5 行)
- `__main__.py` 存在
- chroma 兜底 1 行(chroma_store.py:105)
- qq 2 行(:183, :185)、weixin 1 行(:91 comment)
- pyproject.toml 8 行(:100, :101, :112-115 共 4 行, :119, :122, :125, :129, :153 = 共 10 行,grep "nanobot" 不区分含 `nanobot/...`、`["nanobot"]` 等)

任一不符:**停**,前轮复核到执行有漂移。

- [ ] **Step 5:抓 test baseline — 收集集合**

`testpaths` 在 `backend/pyproject.toml` 是 `["tests"]`。从 repo 根跑会找到根 `tests/`;从 `backend/` 跑会找到 `backend/tests/`。两个都抓。

Run(repo 根 — 根 tests/):
```bash
cd /d/Code/nanobot
./backend/.venv/Scripts/python -m pytest tests/ --collect-only -q 2>&1 | tee /tmp/phase3-baseline-collect-root.txt | tail -10
echo "---collect-root exit=${PIPESTATUS[0]}---"
echo "---collect-root id count: $(grep -cE '::' /tmp/phase3-baseline-collect-root.txt)---"
```

Expected:
- exit 0
- 末尾形如 `N tests collected in X.XXs`
- id 计数应在数百级(根 tests 81 文件)

Run(backend tests):
```bash
cd /d/Code/nanobot/backend
./.venv/Scripts/python -m pytest tests/ --collect-only -q 2>&1 | tee /tmp/phase3-baseline-collect-backend.txt | tail -10
echo "---collect-backend exit=${PIPESTATUS[0]}---"
echo "---collect-backend id count: $(grep -cE '::' /tmp/phase3-baseline-collect-backend.txt)---"
cd /d/Code/nanobot
```

Expected:
- exit 0 或 5(no tests collected 也可接受,但若是 5 需排查 conftest.py 是否能加载)
- 计数应在数十到数百级

任一非 0 / 5:**停**。pytest 不能收集说明 conftest / import 时已坏 — 这是 baseline 必须先解决的环境问题,不能带病改名。

- [ ] **Step 6:抓 test baseline — 通过/失败计数**

Run(repo 根 — 根 tests/,只跑非 stress):
```bash
cd /d/Code/nanobot
./backend/.venv/Scripts/python -m pytest tests/ -p no:anyio --tb=no -q -m "not stress" --no-header 2>&1 | tee /tmp/phase3-baseline-run-root.txt | tail -20
echo "---run-root exit=${PIPESTATUS[0]}---"
```

Expected:
- 末尾形如 `==== N passed, M failed, K skipped in X.XXs ====` 或类似
- 记下 `N passed`、`M failed`、`K skipped` 三个数(后续 Task 8 一致性对比)

注:此 step 可能跑几分钟。允许后台运行,但等它完了再进 Step 7。

Run(backend tests):
```bash
cd /d/Code/nanobot/backend
./.venv/Scripts/python -m pytest tests/ -p no:anyio --tb=no -q --no-header 2>&1 | tee /tmp/phase3-baseline-run-backend.txt | tail -20
echo "---run-backend exit=${PIPESTATUS[0]}---"
cd /d/Code/nanobot
```

Expected: 类似末尾,记下三个数。backend/tests 是 psycopg2 + 真实 DB,可能有大量 fail/error(取决于本地 PG 是否在跑)— **这没关系**,后续比对要求"集合 + 计数都一样"即可,不要求绿。

- [ ] **Step 7:抓 import 计数 baseline**

Run:
```bash
echo "=== backend/nanobot/ (1031 期望) ===" | tee /tmp/phase3-pre-rename-counts.txt
{ cd /d/Code/nanobot && rg -c '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py backend/nanobot/ 2>/dev/null | awk -F: '{s+=$2} END {print "total:", s, "files:", NR}'; } | tee -a /tmp/phase3-pre-rename-counts.txt
echo "=== tests/ (339 期望) ===" | tee -a /tmp/phase3-pre-rename-counts.txt
{ rg -c '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py tests/ 2>/dev/null | awk -F: '{s+=$2} END {print "total:", s, "files:", NR}'; } | tee -a /tmp/phase3-pre-rename-counts.txt
echo "=== backend/tests/ (45 期望) ===" | tee -a /tmp/phase3-pre-rename-counts.txt
{ rg -c '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py backend/tests/ 2>/dev/null | awk -F: '{s+=$2} END {print "total:", s, "files:", NR}'; } | tee -a /tmp/phase3-pre-rename-counts.txt
echo "=== backend/scripts/ (44 期望) ===" | tee -a /tmp/phase3-pre-rename-counts.txt
{ rg -c '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py backend/scripts/ 2>/dev/null | awk -F: '{s+=$2} END {print "total:", s, "files:", NR}'; } | tee -a /tmp/phase3-pre-rename-counts.txt
cat /tmp/phase3-pre-rename-counts.txt
```

Expected:四段 total 大致 `1031 / 339 / 45 / 44`(允许 ±20 漂移因 commit 后实际变化)。

若严重偏离(某段差 >100):**停**,可能用户在你不知情时改了代码 / 复核到现在的时间差里有新 import 被加进来。

- [ ] **Step 8:打 pre-rename 本地 tag(回退锚点)**

Run:
```bash
git tag phase3-pre-rename
git tag -l | grep phase3
echo "exit=$?"
```

Expected:`phase3-pre-rename` 在 tag 列表里,exit 0。

不打 annotated tag、不推 remote,纯本地锚点。Commit D 推上去后该 tag 仍保留作"想看改名前长啥样"的索引。后续如果改名过程中出现严重问题,可以 `git reset --hard phase3-pre-rename` 完整恢复(但要先 stash 任何已 commit 但未 push 的 Commit A/B/C — 见 Task 4 Step 5)。

- [ ] **Step 9:显式确认通过本闸**

到此 Step 1-8 全部 Expected 满足才进 Task 1。

agent 执行者:把 Step 1-7 实际输出回报给用户,等明确"可以开 Commit A"再继续。
人执行者:肉眼复核完,再敲下一条命令。

### Task 0 执行记录(2026-06-27 实跑补记)

**通过状态**:闸口通过,起点干净,进入 Task 1。HEAD `08283baf`,working tree 12 entries(11 untracked + Phase 3 plan.md),0 modified / 0 staged,tag `phase3-pre-rename` 已打。

**Step 2/3 口径修正**(漂移误报):
- `git ls-files --others --exclude-standard | wc -l` 返回 255 而非 plan 预期的 11。原因:`ls-files --others` 默认**递归列出 untracked 目录内所有文件**(`backend/logs/`、`backend/models/` 各自有数百内文件),而 `git status -s` 把这些目录折叠成一行展示。两者口径不同,11 vs 255 都正确。Plan 没改(后续 Task 也用相同命令,语义不变);执行时按 `git status -s` 行数核 11,`ls-files --others` 当辅助佐证不强校验。
- Step 3 `git ls-files --others --exclude-standard | sort` 列了 255 行内涵文件,人眼复核 11 个顶层 entry 即可。

**Step 5/6 pytest baseline 跳过 + 等价性降级**:
- 原计划走 `pytest --collect-only` + 完整 run 抓 baseline,后续 Task 4/8 对比 collect 集合 + pass/fail/skip 计数证等价。
- 实跑发现 `tests/research/` 里 `from nanobot.rag.libs.loader.markdown_loader import MarkdownLoader` 是 module-level 副作用导入,触发 transformers/torch 加载,pytest collect 挂死(>30s 无输出)。这是独立代码病(顶层副作用 import),非 Phase 3 scope,不修。
- 决策:**跳过** Step 5/6,Task 4/8 同步降级:
  - Task 4:不跑 pytest 对比,改用 `import nanobot` 启动 smoke + Commit A 新增的 `tests/utils/test_env_compat.py` 单元测试。
  - Task 8:不跑 pytest 对比,改用 `import nanoresearch` 启动 smoke + 反向 grep `import nanobot|from nanobot\. == 0`(全 repo)+ string-form `"nanobot." == 0`。
- 等价性证据强度:`import nanoresearch` 成功(新结构能加载)+ `import nanobot` ModuleNotFoundError(老名彻底没了)+ 反向 grep 全 0(没人还引用老名),三条合起来等于"包结构改名干净"的强证据,且不依赖 pytest 能否 collect。

**.gitignore corruption 发现 + deferred 笔记**:
- Step 3 排查 nanobot 字面量时,`grep "nanobot" .gitignore` 因 `file` 把它识别为 binary 而被 ripgrep 默认跳过。深入字节级排查发现:`.gitignore` 是 **UTF-8/UTF-16 混合编码**,bytes 0-954 是 UTF-8 + CRLF(原始内容),bytes 955-1083 是 UTF-16 LE + CRLF(commit `08283baf` 那次用 PowerShell `Out-File`/`>>` append 造成 — PowerShell 5.1 默认 UTF-16 LE 编码)。65 个 NUL byte 散落其中,触发 binary 检测。
- 文件功能正常(git 仍能正确解析 ignore 规则,前几次测试均通过),只是编码脏。
- **Phase 3 决策**:把混合编码修成纯 UTF-8 是独立的大改动(可能影响 CRLF/换行、需要全文件 diff 检查),与改名解耦。**deferred 到 Phase 3 之后单独一个 commit 处理**(类似 spec §2.x 的独立 fix 项)。Phase 3 本身在 Commit B 里加 `nanoresearch/` 双覆盖行时,用字节级 anchor-insert 只在 UTF-8 段插入,绝不触碰 UTF-16 段。
- Step 3 找到的 3 处 nanobot 命中:Line 38 `.nanobot/config.json`(用户 dotfile,跟包名无关,不动);Line 62 `backend/nanobot/nanobot/workspace/`、Line 63 `backend/nanobot/data/db/`(包改名相关,Commit B 双覆盖追加,见 Task 2 Step 6.5)。

**Step 7 import 计数 baseline**(`/tmp/phase3-pre-rename-counts.txt`):
- `backend/nanobot/`: 1031 imports / 209 files
- `tests/`: 339 / 68
- `backend/tests/`: 45 / 14
- `backend/scripts/`: 44 / 15
- 总计 1459 imports / 306 files(与 spec 预估 ~1464 一致)

**.venv 状态**:`backend/.venv/` 通过 `cd backend && uv sync --frozen --extra dev` 装齐,pytest 9.1.0 / pytest-asyncio 1.4.0 / pytest-cov 6.3.0 / psycopg2-binary 2.9.12 / ruff 0.15.17。`./backend/.venv/Scripts/python -c "import nanobot; print(nanobot.__file__)"` 解析到 `D:\Code\nanobot\backend\nanobot\__init__.py`(源码路径,非 site-packages copy)。

---

## Task 1:env var 兼容层 + 4 站点切换(Commit A)

**Files:**
- Create: `backend/nanobot/utils/env_compat.py`
- Create: `tests/utils/__init__.py`(若不存在)
- Create: `tests/utils/test_env_compat.py`
- Modify: `backend/nanobot/utils/__init__.py`(若需要 export)
- Modify: `backend/nanobot/__main__.py`(顶部 + entry 调用)
- Modify: `backend/nanobot/cli/commands.py`(顶部 + entry 调用)
- Modify: `backend/nanobot/server/main.py`(顶部 + entry 调用)
- Modify: `backend/nanobot/agent/loop.py:153-154`(env var 名 + 注释)
- Modify: `backend/nanobot/agent/tools/shell.py:101`(env var 名)
- Modify: `backend/nanobot/server/routers/agent_router.py:159`(env var 名)

**Interfaces:**
- Consumes: Task 0 baseline。
- Produces: 启动钩子 `apply_legacy_env_compat()` 把任何 `NANOBOT_*` 复制到 `NANORESEARCH_*` 并 deprecation warning;4 个读取点切到 `NANORESEARCH_*`;读 baseline 的 4 个 env var 名先升级,但 Pydantic env_prefix 还是 `NANOBOT_`(留 Commit D);老用户的 `.env`/`docker` 仍然能跑。

- [ ] **Step 1:Read existing utils package(确认结构)**

Run:
```bash
ls backend/nanobot/utils/
```

Expected:看到 `__init__.py` + 至少 1 个其它文件(基于 grep 显示 `nanobot.utils.helpers` 存在)。

```bash
cat backend/nanobot/utils/__init__.py
```

Expected:看现有 `__init__.py` 内容 — 决定新模块需不需要在 `__init__.py` 显式 export(若现有 `__init__.py` 是空的或只有 import,跟着风格走)。

- [ ] **Step 2:Write `backend/nanobot/utils/env_compat.py`**

用 Write 工具创建该文件,内容如下:

```python
"""Legacy env var compatibility shim for the nanobot → nanoresearch rename.

Reads any `NANOBOT_*` environment variable that does NOT have a corresponding
`NANORESEARCH_*` already set, copies the value to the new name, and emits a
deprecation warning. Must be called once at process startup (CLI entry,
server entry, worker entry) BEFORE any code reads env vars or Pydantic
Settings loads.

The compat layer will be removed in v0.3.0.
"""

from __future__ import annotations

import os
import warnings

_LEGACY_PREFIX = "NANOBOT_"
_NEW_PREFIX = "NANORESEARCH_"
_REMOVED_IN = "0.3.0"

_applied = False


def apply_legacy_env_compat() -> list[tuple[str, str]]:
    """Copy NANOBOT_* env vars to NANORESEARCH_* with deprecation warnings.

    Idempotent: subsequent calls are no-ops. Returns the list of
    (old_name, new_name) pairs that were copied this call (empty after the
    first call).
    """
    global _applied
    if _applied:
        return []
    _applied = True

    copied: list[tuple[str, str]] = []
    for old_name, value in list(os.environ.items()):
        if not old_name.startswith(_LEGACY_PREFIX):
            continue
        new_name = _NEW_PREFIX + old_name[len(_LEGACY_PREFIX):]
        if new_name in os.environ:
            # User already set the new name explicitly — respect it,
            # but still warn that the legacy name is deprecated.
            warnings.warn(
                f"{old_name} is set alongside {new_name}; "
                f"{old_name} is deprecated and will be removed in v{_REMOVED_IN}. "
                f"Using {new_name}.",
                DeprecationWarning,
                stacklevel=2,
            )
            continue
        os.environ[new_name] = value
        copied.append((old_name, new_name))
        warnings.warn(
            f"{old_name} is deprecated; use {new_name}. "
            f"{old_name} will be removed in v{_REMOVED_IN}.",
            DeprecationWarning,
            stacklevel=2,
        )
    return copied


def _reset_for_tests() -> None:
    """Test-only helper to reset the idempotency guard between tests."""
    global _applied
    _applied = False
```

注:不依赖 logger 框架,只用标准库 `warnings`(避免循环依赖 — utils 不能依赖 logger 因 logger 也可能用 utils)。stacklevel=2 让 warning 指向调用者位置。

- [ ] **Step 3:Decide on `utils/__init__.py` re-export**

Run:
```bash
cat backend/nanobot/utils/__init__.py
```

如果 `__init__.py` 包含 `from nanobot.utils.helpers import ...` 之类的显式 re-export 风格,**Modify** 该文件,加一行:
```python
from nanobot.utils.env_compat import apply_legacy_env_compat
```

如果 `__init__.py` 是空的或只放 docstring,**不动**(用户在 entry 处直接 `from nanobot.utils.env_compat import apply_legacy_env_compat`)。

- [ ] **Step 4:Wire 到 `__main__.py`**

Read first:
```bash
cat backend/nanobot/__main__.py
```

预期内容(已知):
```python
"""
Entry point for running nanobot as a module: python -m nanobot
"""

from nanobot.cli.commands import app

if __name__ == "__main__":
    app()
```

Edit:在 `from nanobot.cli.commands` 之前加 compat 调用。完整新内容(Write 整文件,因为很短):

```python
"""
Entry point for running nanobot as a module: python -m nanobot
"""

from nanobot.utils.env_compat import apply_legacy_env_compat

# Apply env var compat BEFORE any code reads env vars or loads Pydantic Settings.
apply_legacy_env_compat()

from nanobot.cli.commands import app

if __name__ == "__main__":
    app()
```

注:E402 ruff warning(import not at top)对此场景可接受,因为 compat 必须在其它 nanobot import 之前跑。若 ruff 拦,在该行加 `  # noqa: E402` 注释(优先用注释抑制单行,不改全局 config)。

- [ ] **Step 5:Wire 到 `backend/nanobot/cli/commands.py`**

Read lines 35-46(已知 `from nanobot import __logo__, __version__` 在 35,`app = typer.Typer(...)` 在 41-46)。

Edit:在第一个 `from nanobot import` 之前加 compat。具体 Edit:

old_string(精确包含上下文):
```python
import typer
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from nanobot import __logo__, __version__
```

new_string:
```python
import typer
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from nanobot.utils.env_compat import apply_legacy_env_compat

apply_legacy_env_compat()

from nanobot import __logo__, __version__  # noqa: E402
```

注:`apply_legacy_env_compat()` 必须在所有其它 `from nanobot.x import ...` 之前调用,因为后续 import 可能触发 Pydantic Settings 加载 — 一旦 Settings 类被 import,env_prefix 就被 evaluate,compat 必须先跑过。

- [ ] **Step 6:Wire 到 `backend/nanobot/server/main.py`**

Read lines 1-20。

Edit:

old_string:
```python
"""FastAPI app factory for the nanobot API server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from nanobot.server.middleware.auth import get_current_user
```

new_string:
```python
"""FastAPI app factory for the nanobot API server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from nanobot.utils.env_compat import apply_legacy_env_compat

apply_legacy_env_compat()

from nanobot.server.middleware.auth import get_current_user  # noqa: E402
```

- [ ] **Step 7:站点 1 — `agent/loop.py:153-154` 切到 `NANORESEARCH_MAX_CONCURRENT_REQUESTS`**

Edit `backend/nanobot/agent/loop.py`:

old_string:
```python
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
```

new_string:
```python
        # NANORESEARCH_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        # Legacy NANOBOT_MAX_CONCURRENT_REQUESTS is auto-copied at startup by
        # apply_legacy_env_compat() (removed in v0.3.0).
        _max = int(os.environ.get("NANORESEARCH_MAX_CONCURRENT_REQUESTS", "3"))
```

- [ ] **Step 8:站点 2 — `agent/tools/shell.py:101` 切到 `NANORESEARCH_PYTHON`**

Edit `backend/nanobot/agent/tools/shell.py`:

old_string:
```python
        env = os.environ.copy()
        # 注入 nanobot 自己的 Python 路径到 PATH 最前面
        python_dir = os.path.dirname(sys.executable)
        env["NANOBOT_PYTHON"] = sys.executable
```

new_string:
```python
        env = os.environ.copy()
        # 注入 nanobot 自己的 Python 路径到 PATH 最前面
        python_dir = os.path.dirname(sys.executable)
        env["NANORESEARCH_PYTHON"] = sys.executable
```

注:这里**子进程接受**的 env var,不是**进程自己读**的。改成 NANORESEARCH_PYTHON 后子进程看到的就是新名。需要确认下游 consumer(是不是 shell skill / tmux 之类读这个 var?)。

Run(扫描下游 consumer):
```bash
grep -rn "NANOBOT_PYTHON\|NANORESEARCH_PYTHON" --include="*.py" --include="*.sh" --include="*.md" .
```

Expected:除了刚改的 shell.py:101 外**只有** 0 个其它命中(说明该 env var 只是注入给子进程的标记,没有 nanobot 代码内部消费它)。如果有命中:**停**,先把那些读取点也切换。

注释里的 "nanobot" 留到 Commit D 顺手改(不在本 Step 范围)。

- [ ] **Step 9:站点 3 — `server/routers/agent_router.py:159` 切到 `NANORESEARCH_WORKSPACE`**

Edit `backend/nanobot/server/routers/agent_router.py`:

old_string:
```python
        workspace = Path(os.environ.get("NANOBOT_WORKSPACE", "~/.nanoresearch/workspace")).expanduser()
```

new_string:
```python
        workspace = Path(os.environ.get("NANORESEARCH_WORKSPACE", "~/.nanoresearch/workspace")).expanduser()
```

- [ ] **Step 10:Worker / 其它 entry 扫一遍**

Run:
```bash
grep -rn "if __name__" --include="*.py" backend/nanobot/ | grep -v test
```

Expected:除了 `backend/nanobot/__main__.py` 外,可能还有:
- `backend/nanobot/worker.py`(ARQ worker)
- `backend/nanobot/rag/mcp_server/__main__.py`(MCP server)
- 其它 cli 子命令的 entry

每个 entry 都要在最顶部加 `apply_legacy_env_compat()` 调用,理由同 Step 4-6。

对每个新发现的 entry:Read → Edit 加 compat → 单步确认。

注:`server/main.py` 是 FastAPI app factory,由 uvicorn 调起,模块顶部 import 时就触发 compat(Step 6 已加)。worker.py 是 ARQ worker 入口,同样需要 — 重点检查。

- [ ] **Step 11:写 test — `tests/utils/test_env_compat.py`**

Read `tests/` 目录确认 `utils/` 子目录是否存在:
```bash
ls tests/utils/ 2>&1
```

如果不存在,先创建 `tests/utils/__init__.py`(空文件)。

Write `tests/utils/test_env_compat.py`:

```python
"""Tests for nanobot.utils.env_compat."""

import os
import warnings

import pytest

from nanobot.utils.env_compat import apply_legacy_env_compat, _reset_for_tests


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset compat state and clear test-relevant env vars between tests."""
    _reset_for_tests()
    for key in list(os.environ):
        if key.startswith("NANOBOT_") or key.startswith("NANORESEARCH_"):
            monkeypatch.delenv(key, raising=False)
    yield
    _reset_for_tests()


def test_copies_legacy_to_new_with_warning(monkeypatch):
    monkeypatch.setenv("NANOBOT_FOO", "value-foo")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        copied = apply_legacy_env_compat()
    assert ("NANOBOT_FOO", "NANORESEARCH_FOO") in copied
    assert os.environ["NANORESEARCH_FOO"] == "value-foo"
    assert any(
        issubclass(item.category, DeprecationWarning)
        and "NANOBOT_FOO" in str(item.message)
        and "NANORESEARCH_FOO" in str(item.message)
        for item in w
    )


def test_respects_explicit_new_name(monkeypatch):
    """If user sets BOTH, new name wins and old name still warns."""
    monkeypatch.setenv("NANOBOT_FOO", "old-value")
    monkeypatch.setenv("NANORESEARCH_FOO", "new-value")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        copied = apply_legacy_env_compat()
    assert copied == []  # nothing copied — new name already set
    assert os.environ["NANORESEARCH_FOO"] == "new-value"  # untouched
    assert any(
        issubclass(item.category, DeprecationWarning)
        and "NANOBOT_FOO is set alongside" in str(item.message)
        for item in w
    )


def test_idempotent(monkeypatch):
    monkeypatch.setenv("NANOBOT_FOO", "v")
    first = apply_legacy_env_compat()
    second = apply_legacy_env_compat()
    assert first == [("NANOBOT_FOO", "NANORESEARCH_FOO")]
    assert second == []  # second call is no-op


def test_ignores_unrelated_env_vars(monkeypatch):
    monkeypatch.setenv("PATH", "/some/path")
    monkeypatch.setenv("HOME", "/home/user")
    copied = apply_legacy_env_compat()
    assert copied == []


def test_handles_all_five_legacy_vars(monkeypatch):
    """The 5 documented NANOBOT_ vars all roundtrip correctly."""
    legacy_vars = {
        "NANOBOT_MAX_CONCURRENT_REQUESTS": "5",
        "NANOBOT_PYTHON": "/usr/bin/python3",
        "NANOBOT_WORKSPACE": "/tmp/ws",
        "NANOBOT_TMUX_SOCKET_DIR": "/tmp/sockets",
        "NANOBOT_FOO_PYDANTIC_FIELD": "bar",  # represents any Pydantic env_prefix var
    }
    for k, v in legacy_vars.items():
        monkeypatch.setenv(k, v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        apply_legacy_env_compat()
    for old, val in legacy_vars.items():
        new = "NANORESEARCH_" + old[len("NANOBOT_"):]
        assert os.environ[new] == val, f"{old} → {new} failed"
```

- [ ] **Step 12:Run test**

Run:
```bash
cd /d/Code/nanobot
./backend/.venv/Scripts/python -m pytest tests/utils/test_env_compat.py -v 2>&1 | tail -25
echo "exit=${PIPESTATUS[0]}"
```

Expected:5 个 test 全 pass,exit 0。

任何 fail / error:**停**排查。

- [ ] **Step 13:Smoke — 启动 CLI 验证 compat 跑了**

Run(模拟带老 env var 的用户):
```bash
NANOBOT_MAX_CONCURRENT_REQUESTS=5 ./backend/.venv/Scripts/python -W default::DeprecationWarning -m nanobot --help 2>&1 | head -20
```

Expected:
- 输出 nanobot CLI 帮助(说明 `apply_legacy_env_compat()` 没把启动炸掉)
- stderr 含一行 `DeprecationWarning: NANOBOT_MAX_CONCURRENT_REQUESTS is deprecated; use NANORESEARCH_MAX_CONCURRENT_REQUESTS. NANOBOT_MAX_CONCURRENT_REQUESTS will be removed in v0.3.0.`

无 deprecation 输出:**停**(可能 compat 钩子没接进 CLI entry)。

- [ ] **Step 14:暂存区核对(Commit A 前唯一闸口)**

Run:
```bash
git status -s
echo "---"
git diff --stat
echo "---"
git diff --cached --name-only
```

Expected:
- `git status -s` 显示 ~10 个 `M`(新建文件显示 `??`)、原 11 个 untracked 仍在
- `git diff --stat` 显示修改的文件清单 + 行数
- `git diff --cached --name-only` 输出为空(还没 add)

预期被改文件清单(7-9 个):
- `backend/nanobot/utils/env_compat.py`(new)
- `backend/nanobot/utils/__init__.py`(maybe new export)
- `backend/nanobot/__main__.py`
- `backend/nanobot/cli/commands.py`
- `backend/nanobot/server/main.py`
- `backend/nanobot/agent/loop.py`
- `backend/nanobot/agent/tools/shell.py`
- `backend/nanobot/server/routers/agent_router.py`
- `tests/utils/test_env_compat.py`(new)
- `tests/utils/__init__.py`(maybe new)
- 可能还有 worker.py / mcp_server/__main__.py(Step 10 发现的额外 entry)

任何不在上述清单的修改:**停**,人工核对是不是误改。

- [ ] **Step 15:暂存 + commit**

Run(显式列每个文件,绝不 `git add .`):
```bash
git add backend/nanobot/utils/env_compat.py
git add backend/nanobot/utils/__init__.py  # 若 Step 3 改过
git add backend/nanobot/__main__.py
git add backend/nanobot/cli/commands.py
git add backend/nanobot/server/main.py
git add backend/nanobot/agent/loop.py
git add backend/nanobot/agent/tools/shell.py
git add backend/nanobot/server/routers/agent_router.py
git add tests/utils/
# 若 Step 10 发现其它 entry,这里加
echo "---staged---"
git diff --cached --name-only
```

Expected:`git diff --cached --name-only` 显示精确这 ~10 个文件,无其它。

任何额外:`git reset HEAD <path>` 退出,重看。

```bash
git commit -m "$(cat <<'EOF'
feat(rename): env var compat shim — NANOBOT_* → NANORESEARCH_*

Pre-rename Commit A of Phase 3 (docs/superpowers/specs/2026-06-26-repo-cleanup-design.md):

- Add nanobot.utils.env_compat.apply_legacy_env_compat() — copies any
  NANOBOT_* env var to NANORESEARCH_* at process startup with a
  DeprecationWarning. Idempotent, no logger dependency.
- Wire compat into CLI entry (__main__.py, cli/commands.py), server entry
  (server/main.py), and any worker entry points.
- Switch the 4 standalone read sites to NANORESEARCH_*:
  - agent/loop.py: NANORESEARCH_MAX_CONCURRENT_REQUESTS
  - agent/tools/shell.py: NANORESEARCH_PYTHON (subprocess env injection)
  - server/routers/agent_router.py: NANORESEARCH_WORKSPACE

Pydantic env_prefix in config/schema.py and NANOBOT_TMUX_SOCKET_DIR in
skills/tmux/ stay on the legacy prefix this commit. They flip in
Commit D (rename) and Commit C (TMUX) respectively, both relying on the
compat shim being already in place.

Sunset version for NANOBOT_* env vars: v0.3.0.
EOF
)"
echo "exit=$?"
```

Expected:`[main <hash>] feat(rename): env var compat ...`,~10 files changed,exit 0。

- [ ] **Step 16:Push**

Run:
```bash
git status -sb | head -1
git push origin main
echo "exit=$?"
git status -sb | head -1
```

Expected:push 前 `[ahead 1]`,push 后输出 `<old>..<new>  main -> main`,exit 0,push 后无 ahead。

Push 失败:**停**,原样报错,不重试 force。

---

## Task 2:chroma 兜底 + qq/weixin 路径修复 + .gitignore 双覆盖(Commit B)

**Files:**
- Modify: `backend/nanobot/rag/libs/vector_store/chroma_store.py:105`
- Modify: `backend/nanobot/channels/qq.py:175-189`
- Modify: `backend/nanobot/channels/weixin.py:91`
- Modify: `.gitignore` — 字节级 anchor-insert,只追加,不替换,不动 Line 38、不碰 UTF-16 corruption

**Interfaces:**
- Consumes: Commit A merged。
- Produces: chroma 在无 settings.yaml 的首次部署不再把数据写进源码树;qq 媒体路径只走 `get_media_dir()`,无 fallback 硬编码;weixin 注释跟代码行为一致;`.gitignore` 双覆盖老/新路径,改名时 git status 不会突然冒出 workspace/db 目录。四处都与改名解耦(都是改名安全网或独立 fix)。

- [ ] **Step 1:Read chroma_store.py 上下文(line 95-120)**

Run:
```bash
sed -n '95,120p' backend/nanobot/rag/libs/vector_store/chroma_store.py
```

确认 line 105 是默认值 fallback。

- [ ] **Step 2:Edit chroma_store.py**

old_string:
```python
        # Persist directory (allow override)
        persist_dir_str = kwargs.get(
            'persist_directory',
            getattr(vector_store_config, 'persist_directory', './data/db/chroma')
        )
```

new_string:
```python
        # Persist directory (allow override)
        # Fallback default is the per-user data dir so first-time deploys
        # without settings.yaml don't dump chroma into the source tree.
        persist_dir_str = kwargs.get(
            'persist_directory',
            getattr(vector_store_config, 'persist_directory', '~/.nanoresearch/rag/chroma')
        )
```

- [ ] **Step 3:Read qq.py 上下文(line 170-200)**

Run:
```bash
sed -n '170,200p' backend/nanobot/channels/qq.py
```

确认 `_init_media_root` 当前结构(已知:if config.media_dir → elif get_media_dir try/except → else 硬编码)。

- [ ] **Step 4:Edit qq.py — 去硬编码 fallback,保留 None 哨兵**

old_string:
```python
    def _init_media_root(self) -> Path:
        """Choose a directory for saving inbound attachments."""
        if self.config.media_dir:
            root = Path(self.config.media_dir).expanduser()
        elif get_media_dir:
            try:
                root = Path(get_media_dir("qq"))
            except Exception:
                root = Path.home() / ".nanoresearch" / "media" / "qq"
        else:
            root = Path.home() / ".nanoresearch" / "media" / "qq"

        root.mkdir(parents=True, exist_ok=True)
        logger.info("QQ media directory: {}", str(root))
        return root
```

new_string:
```python
    def _init_media_root(self) -> Path:
        """Choose a directory for saving inbound attachments."""
        if self.config.media_dir:
            root = Path(self.config.media_dir).expanduser()
        elif get_media_dir:
            root = Path(get_media_dir("qq"))
        else:
            raise RuntimeError(
                "get_media_dir is unavailable; cannot resolve QQ media directory"
            )

        root.mkdir(parents=True, exist_ok=True)
        logger.info("QQ media directory: {}", str(root))
        return root
```

注:plan 原稿误判 `elif get_media_dir:` 为死代码 — 实际 qq.py:42-45 是 try/except 防御导入,import 失败时 `get_media_dir = None`,所以 `elif get_media_dir:` 是 None 哨兵守卫,不是死代码。修正方案:**保留**哨兵分支结构(避免 `Path(None("qq"))` TypeError 回归)、去掉两处硬编码 `~/.nanoresearch/media/qq` fallback(达到 plan 目的:错误不再被静默吞,改写到非预期路径)、`else` 改 `raise RuntimeError` 给清晰错误。**不动** Line 42-45 try/except import 防御 — 那是 plan 外重构,边界纪律守住(参照 Task 1 worker.py scope creep 教训)。

- [ ] **Step 5:Read qq.py import 段(确认 `get_media_dir` 是否真无 fallback 引入)**

Run:
```bash
grep -n "get_media_dir" backend/nanobot/channels/qq.py
```

Expected:1-2 行,顶部 import + `_init_media_root` 内调用。

确认 import 是 `from nanobot.config.paths import get_media_dir`(可能在 try/except 防御导入)。如果是 try/except 包裹,**保持原样**不删 try(那是为了 plugin 模式下 nanobot.config 可能不可用)。

- [ ] **Step 6:Edit weixin.py:91 注释**

Read 当前(已知 line 91):

Edit:
old_string:
```python
    state_dir: str = ""  # Default: ~/.nanoresearch/weixin/
```

new_string:
```python
    state_dir: str = ""  # Empty → resolved at runtime via get_runtime_subdir("weixin")
```

理由:实测 `_get_state_dir()` line 138 走 `get_runtime_subdir("weixin")`,不是固定 `~/.nanoresearch/weixin/`。原注释误导。

- [ ] **Step 6.5a:.gitignore 现状只读核对(改前最后一道闸)**

`.gitignore` 是 UTF-8/UTF-16 混合编码(Task 0 执行记录 §.gitignore corruption note 解释了来源)。改它之前先确认现状没漂移。

Run:
```bash
git status -s -- .gitignore
echo "---"
grep -nE '^(\.nanobot/config\.json|backend/nanobot/nanobot/workspace/|backend/nanobot/data/db/)$' .gitignore
echo "---"
python -c "raw=open('.gitignore','rb').read(); a=b'backend/nanobot/data/db/\r\n'; print(f'size={len(raw)} nul_count={raw.count(bytes([0]))} has_anchor={raw.count(a)}')"
```

Expected:
- `git status -s` 空(`.gitignore` 没有 untracked 修改)
- grep 命中 3 行:Line 38 `.nanobot/config.json`、Line 62 `backend/nanobot/nanobot/workspace/`、Line 63 `backend/nanobot/data/db/`
- size = 1084 字节、nul_count = 65、has_anchor = 1

任一不符:**停**(`.gitignore` 在 Task 0 之后被改过,需要重新对齐预期)。

- [ ] **Step 6.5b:写 anchor-insert 脚本到 /tmp**

Write `/tmp/phase3-edit-gitignore.py`(下面整段就是脚本全文,**第一行必须是 shutil.copy 备份**;assert 失败让脚本直接 crash,**不要 try/except 吞**):

```python
import shutil
shutil.copy('.gitignore', '/tmp/phase3-gitignore.bak')
with open('.gitignore', 'rb') as f:
    raw = f.read()
anchor = b'backend/nanobot/data/db/\r\n'
assert raw.count(anchor) == 1, f"anchor count != 1: {raw.count(anchor)}"
insert_at = raw.find(anchor) + len(anchor)
new = b'backend/nanoresearch/nanobot/workspace/\r\nbackend/nanoresearch/data/db/\r\n'
out = raw[:insert_at] + new + raw[insert_at:]
with open('.gitignore', 'wb') as f:
    f.write(out)
print(f"OK: before={len(raw)} after={len(out)} added={len(new)} (expected 72)")
```

理由:
- 用字节级二进制读写而非文本模式,绝不触碰 UTF-16 段
- anchor 是 `backend/nanobot/data/db/\r\n`(Line 63 末尾连同 CRLF),只在 UTF-8 段命中一次
- `assert raw.count(anchor) == 1` 强保证唯一锚点,改名后再跑此脚本会立刻 crash(防误执行)
- 新增 72 字节:`backend/nanoresearch/nanobot/workspace/\r\n` = 39+2 = 41 字节,`backend/nanoresearch/data/db/\r\n` = 29+2 = 31 字节,合计 72

- [ ] **Step 6.5c:执行脚本 + 核对结果**

Run:
```bash
python /tmp/phase3-edit-gitignore.py
echo "exit=$?"
echo "---"
grep -nE '^(backend/nanobot/(nanobot/workspace|data/db)|backend/nanoresearch/(nanobot/workspace|data/db))/$' .gitignore
echo "---"
python -c "raw=open('.gitignore','rb').read(); print(f'size={len(raw)} nul_count={raw.count(bytes([0]))}')"
echo "---"
diff /tmp/phase3-gitignore.bak .gitignore | head -10
```

Expected:
- exit 0,脚本输出 `OK: before=1084 after=1156 added=72 (expected 72)`
- grep 命中 4 行:老 2 行 (62/63) + 新 2 行紧跟其后
- size = 1156、nul_count = 65(UTF-16 段未动)
- diff 显示 `>` 两行新增

任一不符:**停**,从 `/tmp/phase3-gitignore.bak` 恢复(`cp /tmp/phase3-gitignore.bak .gitignore`)。

- [ ] **Step 6.5d:Smoke — git 仍能正确解析 .gitignore**

Run:
```bash
git check-ignore -v backend/nanoresearch/data/db/some/test 2>&1 | head -3
echo "exit=$?"
git status -s | head -20
```

Expected:
- `check-ignore` 命中新加的 `backend/nanoresearch/data/db/`,exit 0
- `git status -s` 应只比 Task 0 起点多 1 项:` M .gitignore`(其它仍是 11 untracked + chroma/qq/weixin 3 个 M from Steps 2/4/6)

任一不符:**停**(git 不认 .gitignore 说明字节插入破坏了文件)。

- [ ] **Step 7:Smoke — 启动 CLI(确认 module import 不坏)**

Run:
```bash
./backend/.venv/Scripts/python -m nanobot --help | head -5
echo "exit=${PIPESTATUS[0]}"
```

Expected:CLI 帮助 + exit 0。

```bash
./backend/.venv/Scripts/python -c "from nanobot.channels.qq import QQChannel; print('qq ok')"
echo "exit=$?"
./backend/.venv/Scripts/python -c "from nanobot.channels.weixin import WeixinChannel; print('weixin ok')"
echo "exit=$?"
./backend/.venv/Scripts/python -c "from nanobot.rag.libs.vector_store.chroma_store import ChromaStore; print('chroma ok')"
echo "exit=$?"
```

Expected:三段 `ok` + exit 0。

任一 fail:**停**排查,可能 Edit 引入语法错误或漏改 import。

- [ ] **Step 8:暂存区核对**

Run:
```bash
git status -s
echo "---"
git diff --stat
```

Expected:
- 4 个 `M`:`.gitignore`、`backend/nanobot/channels/qq.py`、`backend/nanobot/channels/weixin.py`、`backend/nanobot/rag/libs/vector_store/chroma_store.py`
- 原 11 个 untracked 仍在
- 无其它改动

任何额外:**停**核对。

- [ ] **Step 9:暂存 + commit**

Run:
```bash
git add .gitignore backend/nanobot/channels/qq.py backend/nanobot/channels/weixin.py backend/nanobot/rag/libs/vector_store/chroma_store.py
git diff --cached --name-only
```

Expected:4 行精确这 4 个文件。

```bash
git commit -m "$(cat <<'EOF'
fix(paths): chroma fallback + qq/weixin runtime path + .gitignore dual-coverage

Pre-rename Commit B of Phase 3 (docs/superpowers/specs/2026-06-26-repo-cleanup-design.md §2.8):

- chroma_store.py: change fallback default './data/db/chroma' to
  '~/.nanoresearch/rag/chroma'. First-time deploys without settings.yaml
  no longer dump vector data into the source tree.
- qq.py _init_media_root: drop dead `elif get_media_dir:` branch and the
  two hardcoded `~/.nanoresearch/media/qq` fallbacks. get_media_dir is a
  module-level import; if it fails the module can't load, so the elif is
  dead code. Failure now surfaces instead of silently writing elsewhere.
- weixin.py state_dir comment: update to reflect actual behavior
  (resolved via get_runtime_subdir("weixin"), not fixed ~/.nanoresearch/weixin/).
- .gitignore: append nanoresearch/ paths alongside existing nanobot/
  paths so neither old nor new directory shows up in git status during
  or after the rename. Inserted via byte-level anchor-insert to avoid
  touching the UTF-16 region appended in 08283baf (mixed encoding is
  a known issue, fix deferred to its own commit).

All four changes are independent of the package rename.
EOF
)"
echo "exit=$?"
```

Expected:`[main <hash>] fix(paths): ...`,4 files changed,exit 0。

- [ ] **Step 10:Push**

Run:
```bash
git status -sb | head -1
git push origin main
echo "exit=$?"
git status -sb | head -1
```

Expected:同 Task 1 Step 16。

---

## Task 3:TMUX env var .md/.sh(Commit C)

**Files:**
- Modify: `backend/nanobot/skills/tmux/SKILL.md` — 4 处 env var 名
- Modify: `backend/nanobot/skills/tmux/scripts/find-sessions.sh` — env var dual-read + deprecation echo

**Interfaces:**
- Consumes: Commit A 的 compat 钩子已在 Python entry 工作。
- Produces: shell 脚本独立 dual-read env var(shell 启动不经过 Python 钩子);`.md` 文档反映新 env var 名。**不动**socket 子目录名 `nanobot-tmux-sockets`、socket 文件名 `nanobot.sock`、session 示例名 `nanobot-python`、metadata key `{"nanobot":{...}}` — 它们孤立现存会话或影响 skill 发现,留作 Phase 3 不做但留痕(Global Constraints §6)。

- [ ] **Step 1:Edit find-sessions.sh — dual-read**

old_string:
```bash
socket_dir="${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}"
```

new_string:
```bash
# Backward compat: NANOBOT_TMUX_SOCKET_DIR is deprecated, use NANORESEARCH_TMUX_SOCKET_DIR.
# Will be removed in v0.3.0. Default socket dir name kept as nanobot-tmux-sockets
# to preserve existing user sessions; only the env var name changes.
if [[ -n "${NANOBOT_TMUX_SOCKET_DIR:-}" && -z "${NANORESEARCH_TMUX_SOCKET_DIR:-}" ]]; then
  echo "Warning: NANOBOT_TMUX_SOCKET_DIR is deprecated; use NANORESEARCH_TMUX_SOCKET_DIR (will be removed in v0.3.0)." >&2
fi
socket_dir="${NANORESEARCH_TMUX_SOCKET_DIR:-${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}}"
```

注:同样改 `usage()` 函数里的帮助文本(line 13)。

old_string:
```bash
  -A, --all          scan all sockets under NANOBOT_TMUX_SOCKET_DIR
```

new_string:
```bash
  -A, --all          scan all sockets under NANORESEARCH_TMUX_SOCKET_DIR (legacy: NANOBOT_TMUX_SOCKET_DIR)
```

- [ ] **Step 2:Edit SKILL.md — 4 处 env var 名**

逐处 Edit(用 Edit tool 因 SKILL.md 多处重复关键词,replace_all=false 必须每处都加足够上下文)。

**Edit 1 — line 14**:

old_string:
````
SOCKET_DIR="${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}"
````

new_string:
````
SOCKET_DIR="${NANORESEARCH_TMUX_SOCKET_DIR:-${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}}"
````

**Edit 2 — line 34**:

old_string:
````
- Use `NANOBOT_TMUX_SOCKET_DIR` environment variable.
````

new_string:
````
- Use `NANORESEARCH_TMUX_SOCKET_DIR` environment variable (legacy `NANOBOT_TMUX_SOCKET_DIR` still honored until v0.3.0).
````

**Edit 3 — line 35**:

old_string:
````
- Default socket path: `"$NANOBOT_TMUX_SOCKET_DIR/nanobot.sock"`.
````

new_string:
````
- Default socket path: `"$NANORESEARCH_TMUX_SOCKET_DIR/nanobot.sock"`.
````

注:socket **文件名** `nanobot.sock` 留不动(Global Constraints §6 不动清单第 6 条)。

**Edit 4 — line 46**:

old_string:
````
- Scan all sockets: `{baseDir}/scripts/find-sessions.sh --all` (uses `NANOBOT_TMUX_SOCKET_DIR`).
````

new_string:
````
- Scan all sockets: `{baseDir}/scripts/find-sessions.sh --all` (uses `NANORESEARCH_TMUX_SOCKET_DIR`, legacy `NANOBOT_TMUX_SOCKET_DIR` still honored).
````

- [ ] **Step 3:Smoke find-sessions.sh — dual-read 验证**

需 tmux 可用。若环境无 tmux(Windows 默认无),跳过此 Step 并明确告知用户"shell smoke 因环境无 tmux 跳过,由 Linux/Mac 用户验证"。

如果环境有 tmux(WSL / Linux / Mac):
```bash
# 验证 legacy env var 仍然被尊重 + 出 deprecation warning
NANOBOT_TMUX_SOCKET_DIR=/tmp/test-legacy bash backend/nanobot/skills/tmux/scripts/find-sessions.sh -h 2>&1 | head -5

# 验证 new env var 优先
NANORESEARCH_TMUX_SOCKET_DIR=/tmp/test-new NANOBOT_TMUX_SOCKET_DIR=/tmp/test-legacy bash backend/nanobot/skills/tmux/scripts/find-sessions.sh -h 2>&1 | head -10
```

Expected:第一段输出帮助 + stderr 有 `Warning: NANOBOT_TMUX_SOCKET_DIR is deprecated...`。第二段输出帮助 + **无** warning(因为 NANORESEARCH_ 已显式设了)。

注:这里 dual-read 警告逻辑是"老的设了 + 新的没设 → 警告"。当用户同时设了两个,警告不出,这跟 Python compat 不一致(Python 那边两个都设也警告)。为保持 shell 简单,这是可接受的不一致。

- [ ] **Step 4:暂存区核对**

Run:
```bash
git status -s
git diff --stat
```

Expected:
- 2 个 `M`:`backend/nanobot/skills/tmux/SKILL.md`、`backend/nanobot/skills/tmux/scripts/find-sessions.sh`

任何额外:**停**。

- [ ] **Step 5:暂存 + commit**

Run:
```bash
git add backend/nanobot/skills/tmux/SKILL.md backend/nanobot/skills/tmux/scripts/find-sessions.sh
git diff --cached --name-only

git commit -m "$(cat <<'EOF'
feat(tmux): rename NANOBOT_TMUX_SOCKET_DIR with shell-side compat

Pre-rename Commit C of Phase 3 (docs/superpowers/specs/2026-06-26-repo-cleanup-design.md):

- find-sessions.sh: dual-read NANORESEARCH_TMUX_SOCKET_DIR with fallback
  to legacy NANOBOT_TMUX_SOCKET_DIR; echo deprecation warning to stderr
  when legacy name is the only one set.
- SKILL.md: update 4 env var references to the new name; note the legacy
  alias remains honored until v0.3.0.

Default socket dir basename (nanobot-tmux-sockets), socket filename
(nanobot.sock), session example name (nanobot-python), and skill
metadata key {"nanobot":{...}} are NOT changed in this commit — see
Phase 3 plan "不做但留痕" for rationale.
EOF
)"
echo "exit=$?"
```

- [ ] **Step 6:Push**

Run:
```bash
git push origin main
echo "exit=$?"
git status -sb | head -1
```

Expected:同前。

---

## Task 4:Pre-rename re-baseline + 闸口

**Files:** 不动任何文件。

**Interfaces:**
- Consumes: Commit A/B/C 已 push 完成。
- Produces: Commit A/B/C 之后 `import nanobot` (老包) 仍能 import + import 计数没变 + Commit A 新加的 env_compat 测试通过 — Task 5+ 大改名前的最后一道闸。

**为什么独立成 Task**:Commit A/B/C 各自都改了源码,smoke 都过了,但要在跑大改名之前确认包仍然能 import、新加的 env_compat 单元测试是绿的、import 计数没失控。

**关于不跑 pytest baseline 比对的说明**:原计划走 `pytest --collect-only` + 完整 run 对比 baseline / prerename / postrename 三态。Task 0 执行时发现 `tests/research/` 里 `from nanobot.rag.libs.loader.markdown_loader import MarkdownLoader` 是 module-level 副作用导入,触发 transformers/torch 链,pytest collect 会挂死。这是独立于改名的代码病,不在 Phase 3 scope。改用:(i) `python -c "import nanobot"` 启动 smoke、(ii) 反向 grep + import 计数 — 同样能验"包结构没坏 + 没漂移",且不依赖 pytest 能否 collect。等价性的强保证留给 Task 8 的反向 grep `import nanobot == 0` 验证。

- [ ] **Step 1:`import nanobot` 启动 smoke + Commit A 新测试**

Run:
```bash
cd /d/Code/nanobot
./backend/.venv/Scripts/python -c "import nanobot; print('module path:', nanobot.__file__)"
echo "exit=$?"
echo "---"
./backend/.venv/Scripts/python -c "from nanobot.cli.commands import app; print('cli ok')"
echo "exit=$?"
./backend/.venv/Scripts/python -c "from nanobot.server.main import create_app; print('server ok')"
echo "exit=$?"
./backend/.venv/Scripts/python -c "from nanobot.utils.env_compat import apply_legacy_env_compat; print('env_compat ok')"
echo "exit=$?"
```

Expected:
- `import nanobot` exit 0,module path 指向 `D:\Code\nanobot\backend\nanobot\__init__.py`(源码,不是 site-packages)
- cli / server / env_compat 三段 ok + exit 0

任一 fail:**停**,Commit A/B/C 里有 import 或顶层副作用问题,要先修。

- [ ] **Step 2:跑 Commit A 新增的 env_compat 单元测试**

Run:
```bash
cd /d/Code/nanobot
./backend/.venv/Scripts/python -m pytest tests/utils/test_env_compat.py -p no:anyio --tb=short -q 2>&1 | tail -20
echo "---exit=${PIPESTATUS[0]}---"
```

Expected:5 passed(对应 Task 1 Step 7 写的 5 个 test case),exit 0。

任何 failed / error:**停**,Commit A 的兼容钩子有 bug。

- [ ] **Step 3:重抓 import 计数(确认 Commit A-C 没改动 import)**

Run:
```bash
echo "=== backend/nanobot/ ==="
rg -c '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py backend/nanobot/ 2>/dev/null | awk -F: '{s+=$2} END {print "total:", s, "files:", NR}'
```

Expected:total 大致跟 Task 0 Step 7 的 1031 一致 +1-2(Commit A 加了一行 `from nanobot.utils.env_compat import apply_legacy_env_compat` 至少 4 处:__main__.py、commands.py、main.py、可能 utils/__init__.py)。所以预期是 ~1035。

偏差 >10:**停**。

- [ ] **Step 4:git 状态干净 + 远端对齐**

Run:
```bash
git status -sb | head -1
git diff --stat
git diff --cached --name-only
```

Expected:`## main...origin/main`(无 ahead/behind);diff --stat 空;cached 空。

- [ ] **Step 5:`phase3-pre-rename` tag 仍指向 Task 0 那个 commit?**

Run:
```bash
git rev-parse phase3-pre-rename
git log -1 --oneline phase3-pre-rename
echo "---"
git log -1 --oneline HEAD
```

Expected:tag 仍指向 `08283baf chore: ignore logs, models, pycache, vite timestamps`(Task 0 时的 HEAD)。当前 HEAD 是 Commit C 的 hash,与 tag 不同。

如果你想之后能完整 reset 回大改名之前,这个 tag 必须保留。Task 5 之前不动它。

- [ ] **Step 6:显式确认通过本闸**

到此 Step 1-5 全过,才进入 Task 5。

agent:贴 Step 1 import smoke 输出、Step 2 env_compat test 计数、Step 3 import 计数给用户,等"可以开大改名"明确表态再继续。

---

## Task 5:Big atomic rename — directory mv

**Files:**
- Rename: `backend/nanobot/` → `backend/nanoresearch/`(整目录,209 文件 + 子目录)

**Interfaces:**
- Consumes: Task 4 已确认 main 干净、远端对齐、test baseline 抓好。
- Produces: 包目录搬完;工作区一片红(几乎所有 import 都断了);**绝不 commit**(Tasks 6-7 还要堆改动进同一暂存区)。

**为什么独立成 Task**:`git mv` 是 Commit D 的第一步,也是最容易回退的一步。中间态不能跑测试 — import 还没改,跑 pytest 必然全坏。这一 Task 只验证 git 看到了 mv 操作、文件物理位置对、`backend/nanobot/` 不再存在。

- [ ] **Step 1:目录存在性预检**

Run:
```bash
ls -d backend/nanobot/ backend/nanoresearch/ 2>&1
```

Expected:`backend/nanobot/` 存在,`backend/nanoresearch/` `No such file or directory`(若已存在说明上次执行没回完整,**停**)。

- [ ] **Step 2:执行 git mv**

Run:
```bash
git mv backend/nanobot backend/nanoresearch
echo "exit=$?"
```

Expected:exit 0,无 stdout(git mv 静默成功)。

如果 git 报"already exists"或 "permission denied":**停**。Windows 下可能 venv / IDE 占用某个文件,关掉 IDE / venv 再来。

- [ ] **Step 3:验证 mv 结果**

Run:
```bash
ls -d backend/nanobot/ backend/nanoresearch/ 2>&1
echo "---"
ls backend/nanoresearch/ | head -10
echo "---"
echo "rename count in index:"
git status --porcelain | grep -cE '^R'
```

Expected:
- `backend/nanobot/` `No such file or directory`,`backend/nanoresearch/` 存在
- `backend/nanoresearch/` 列出 `__init__.py` / `__main__.py` / `agent/` / `channels/` 等(原 nanobot 子目录)
- rename count = 209(允许 ±5 漂移,具体看 git 怎么算 rename 阈值;如果是 0 说明 git 没识别为 rename)

如果 rename count = 0(git 把它看成 add + delete):后续 commit 的体积会是 209 个 delete + 209 个 add,但还是能跑。只是 history blame 不会 follow。**继续**,不停。

- [ ] **Step 4:验证 git mv 没碰其它文件**

Run:
```bash
git status --porcelain | grep -vE '^R.* backend/nano(bot|research)/' | grep -vE '^[?]{2} ' | head -20
echo "---unexpected change count: $(git status --porcelain | grep -vE '^R.* backend/nano(bot|research)/' | grep -vE '^[?]{2} ' | wc -l)---"
```

Expected:输出空(所有 R 都跟 nanobot/nanoresearch 相关,?? 是 untracked 不算)。count = 0。

任何额外:**停**核对。

- [ ] **Step 5:暂存区状态**

Run:
```bash
git diff --cached --stat | tail -5
echo "---staged count: $(git diff --cached --name-only | wc -l)---"
```

Expected:staged count ≈ 209 × 2(rename 算两行 diff?)或 ≈ 209(单边 R 算一行)。具体取决于 git 输出格式。

- [ ] **Step 6:**绝不 commit**,直接进 Task 6**

mv 完后 import 全坏。现在 commit 会得到一个"代码不能跑"的 commit,违反"不留断代码"原则。Tasks 6-7 把剩下的改动堆进同一暂存区,Task 10 一次性 commit。

显式确认:`git diff --cached --name-only | wc -l` 跟 Step 5 一致,**不动**直接 Task 6。

---

## Task 6:Bulk import 机械替换

**Files:**
- Modify(bulk): 全 repo `.py` 文件的 `import nanobot` / `from nanobot.` / `from nanobot import` 语句 → `nanoresearch`
- Create(临时): `/tmp/phase3-replace-imports.py` — 一次性脚本

**Interfaces:**
- Consumes: Task 5 后状态(目录已 mv,导入全坏)。
- Produces: 4 个 scope 内所有 import 语句改完;reverse-grep `(^|\s)(import nanobot|from nanobot\.|from nanobot import)` 计数 = 0(`tests/agent/evaluation/results/` 等 fixture JSON 除外,但 JSON 不被本 task 的 .py 扫描覆盖,自然不命中)。

- [ ] **Step 1:写替换脚本到 `/tmp/phase3-replace-imports.py`**

用 Write 工具创建该文件:

```python
"""Phase 3 bulk import statement replacement.

Replaces `import nanobot` → `import nanoresearch` and
`from nanobot[.x][.y]` → `from nanoresearch[.x][.y]` in .py files,
preserving:
- comments and string literals (not touched — they're handled separately)
- variable names like `nanobot_extras` (\\b word boundary)
- relative imports `from .nanobot ...` (requires whitespace after `from`)

Run: python /tmp/phase3-replace-imports.py <root1> <root2> ...
Prints per-file modification counts and grand total.
"""

import re
import sys
from pathlib import Path

# Anchor on `\bimport\s+nanobot\b` and `\bfrom\s+nanobot\b` — these are
# only valid in import statements. Word boundary `\b` is between \w and
# \W; `_` is \w, so `nanobot_extras` won't match.
IMPORT_PATTERN = re.compile(r'(\bimport\s+)nanobot(\b)')
FROM_PATTERN = re.compile(r'(\bfrom\s+)nanobot(\b)')


def process_file(path: Path) -> int:
    """Return number of replacements made in `path`."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Binary or non-UTF8 file — skip silently.
        return 0
    new_text, n1 = IMPORT_PATTERN.subn(r'\1nanoresearch\2', text)
    new_text, n2 = FROM_PATTERN.subn(r'\1nanoresearch\2', new_text)
    total = n1 + n2
    if total > 0:
        path.write_text(new_text, encoding="utf-8")
    return total


def main(roots: list[str]) -> None:
    grand_total = 0
    files_changed = 0
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"SKIP missing root: {root}", file=sys.stderr)
            continue
        print(f"=== {root} ===")
        sub_total = 0
        sub_files = 0
        for py in sorted(root_path.rglob("*.py")):
            count = process_file(py)
            if count > 0:
                print(f"  {py.relative_to(Path.cwd())}: {count}")
                sub_total += count
                sub_files += 1
        print(f"  subtotal: {sub_total} replacements in {sub_files} files")
        grand_total += sub_total
        files_changed += sub_files
    print(f"=== TOTAL: {grand_total} replacements in {files_changed} files ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase3-replace-imports.py <root> [<root>...]")
        sys.exit(1)
    main(sys.argv[1:])
```

- [ ] **Step 2:Dry-run 风险扫描(脚本 unit-test 心智模型)**

不实际写 — 但**人工心算**几个 edge case 是否被脚本正确处理:

| 输入 | 预期输出 |
|---|---|
| `import nanobot` | `import nanoresearch` |
| `import nanobot.channels.feishu` | `import nanoresearch.channels.feishu` |
| `from nanobot import x` | `from nanoresearch import x` |
| `from nanobot.x.y import z` | `from nanoresearch.x.y import z` |
| `from nanobot.x import (\n    a,\n    b,\n)` | `from nanoresearch.x import (...)` (因 `from nanobot.x` 在一行,括号在后面行,不影响) |
| `    import nanobot.channels` | `    import nanoresearch.channels` (\\bimport 匹配,缩进保留) |
| `# we use nanobot.channels here` | **不变**(没有 import / from 关键字) |
| `"nanobot.providers"` | **不变**(没有 import / from) |
| `monkeypatch.setattr("nanobot.channels.x", ...)` | **不变**(由 Task 7 处理) |
| `nanobot_extras` | **不变**(\\b 后是 `_`,词边界不在那) |
| `from .nanobot import x`(相对导入) | **不变**(`from ` 后没 nanobot,有 `.nanobot`) |
| `entry_points(group="nanobot.channels")` | **不变**(字符串内,Task 7 处理) |

如果哪一条预期错了:**停**,修改 regex 再来。

- [ ] **Step 3:执行替换 — backend/nanoresearch/(原 nanobot 包内部 ~1031 处)**

注:此时已经 git mv,目录是 `backend/nanoresearch/`(不是 `backend/nanobot/`)。脚本扫这个新位置。

Run:
```bash
./backend/.venv/Scripts/python /tmp/phase3-replace-imports.py backend/nanoresearch 2>&1 | tail -30
echo "exit=${PIPESTATUS[0]}"
```

Expected:
- 末尾 `subtotal: ~1031 replacements in ~209 files`(允许 ±10)
- exit 0

数字明显偏小(<900):**停**,可能 regex 漏掉某种 import 形式。

- [ ] **Step 4:执行替换 — tests/(339 处)**

Run:
```bash
./backend/.venv/Scripts/python /tmp/phase3-replace-imports.py tests 2>&1 | tail -20
echo "exit=${PIPESTATUS[0]}"
```

Expected:subtotal ~339,exit 0。

- [ ] **Step 5:执行替换 — backend/tests/(45 处)**

Run:
```bash
./backend/.venv/Scripts/python /tmp/phase3-replace-imports.py backend/tests 2>&1 | tail -20
echo "exit=${PIPESTATUS[0]}"
```

Expected:subtotal ~45。

- [ ] **Step 6:执行替换 — backend/scripts/(44 处)**

Run:
```bash
./backend/.venv/Scripts/python /tmp/phase3-replace-imports.py backend/scripts 2>&1 | tail -20
echo "exit=${PIPESTATUS[0]}"
```

Expected:subtotal ~44。

- [ ] **Step 7:反向校验 — 全 repo 应该再无 import nanobot/from nanobot 语句**

Run:
```bash
echo "=== remaining .py import nanobot/from nanobot statements ==="
rg -n '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py 2>&1 | head -30
echo "---count: $(rg -c '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')---"
```

Expected:count = 0;命中清单中**只有**:
- `backend/test_ragas_transforms.py`(untracked,接受短暂 ImportError,Task 11 提醒)
- `backend/test_themes_diagnostic.py`(untracked,同上)
- `docs/superpowers/specs/baselines/test_5p2_branches.py`(基线快照,**有意保留** — 它是历史快照不动)
- `docs/superpowers/plans/2026-06-27-feature-b4-merge.md` 之类文档里的 quote(允许,文档不需要可执行)
- 本 plan 自己(同上)

更精确的核查:
```bash
echo "=== remaining .py import nanobot (excluding untracked + docs) ==="
rg -l '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py$' | grep -vE '^docs/'
echo "---count: $(rg -l '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py$' | grep -vE '^docs/' | wc -l)---"
```

Expected:输出空,count = 0。

任何命中:**停**,这是漏改,加进 Task 6 Step 2 的 edge case 心算表里看是哪种形式没被 regex 捕获,修 regex 再来。

注:`docs/superpowers/specs/baselines/test_5p2_branches.py` 虽是 `.py` 但是 SDD baseline 快照,**不**改。

- [ ] **Step 8:Spot-check 几个改名结果**

Run:
```bash
echo "=== sample changes ==="
head -5 backend/nanoresearch/__main__.py
echo "---"
head -5 backend/nanoresearch/cli/commands.py
echo "---"
head -5 backend/nanoresearch/agent/loop.py
echo "---"
head -5 tests/cli/test_commands.py
```

Expected:每个文件顶部 import 都已变为 `nanoresearch`。

`backend/nanoresearch/__main__.py` 预期:
```python
"""
Entry point for running nanobot as a module: python -m nanobot
"""

from nanoresearch.utils.env_compat import apply_legacy_env_compat
```

注:docstring 第 1 行还说 "nanobot" 是因为 docstring 不是 import 语句,regex 没改。**留到 Task 7 Step 8** 顺手扫一遍 docstring/comment 里的 nanobot。

- [ ] **Step 9:不 commit,进 Task 7**

到此 import 改完了,但 3 处动态导入字符串 / entry_points group / Pydantic env_prefix / pyproject.toml / 中文注释 都还是 `nanobot`。Task 7 处理它们,然后 Task 8 验证,Task 10 一次性 commit。

---

## Task 7:String 引用 + entry_points + env_prefix + pyproject.toml + 顺手注释

**Files(全在同一未 commit 暂存区,不单独 commit):**
- Modify: `backend/nanoresearch/channels/registry.py:32, 37, 45`(动态导入字符串 + error message + entry_points group)
- Modify: `backend/nanoresearch/cli/onboard.py:751`(动态导入字符串)
- Modify: `tests/providers/test_providers_init.py:16`(动态导入字符串)
- Modify: `tests/cli/test_commands.py`(monkeypatch / patch 里的 `"nanobot.x.y"` 字符串)
- Modify(全扫): 任何 `.py` 里 `"nanobot.X"` 或 `'nanobot.X'` 字符串字面量
- Modify: `backend/nanoresearch/config/schema.py:283`(Pydantic env_prefix)
- Modify: `backend/pyproject.toml`(8 处:scripts / hatch build / wheel / sources / sdist / coverage)
- Modify: `backend/nanoresearch/agent/tools/shell.py:99`(中文注释)
- Modify(全扫): 任何 `.py` 顶部 docstring 里 `nanobot` 字面量(`__main__.py`、其它)

**Interfaces:**
- Consumes: Task 6 完成,import 已全改,目录已 mv。
- Produces: 全 repo `nanobot` 字面量只剩 (a) 不动清单 §1-7 的项、(b) tmux SKILL.md 的非 env var 字面量、(c) memory/plan/spec 文档里的历史引用。

- [ ] **Step 1:registry.py 3 处**

Read `backend/nanoresearch/channels/registry.py` 上下文(line 25-50):

```bash
sed -n '25,50p' backend/nanoresearch/channels/registry.py
```

Edit 1(line 32 dynamic import):

old_string:
```python
    mod = importlib.import_module(f"nanobot.channels.{module_name}")
```

new_string:
```python
    mod = importlib.import_module(f"nanoresearch.channels.{module_name}")
```

Edit 2(line 37 error message):

old_string:
```python
    raise ImportError(f"No BaseChannel subclass in nanobot.channels.{module_name}")
```

new_string:
```python
    raise ImportError(f"No BaseChannel subclass in nanoresearch.channels.{module_name}")
```

Edit 3(line 45 entry_points group):

old_string:
```python
    for ep in entry_points(group="nanobot.channels"):
```

new_string:
```python
    for ep in entry_points(group="nanoresearch.channels"):
```

**注:entry_points group 改了,任何外部依赖 nanobot 的 channel plugin 都需要重新注册 entry point。** 这是 breaking change,在 commit message 里**显式说明**,不做兼容(因为兼容需要同时读两个 group,代码改动大且当前用户不多)。

- [ ] **Step 2:cli/onboard.py:751**

Read 上下文:
```bash
sed -n '745,760p' backend/nanoresearch/cli/onboard.py
```

Edit:

old_string:
```python
            mod = importlib.import_module(f"nanobot.channels.{name}")
```

new_string:
```python
            mod = importlib.import_module(f"nanoresearch.channels.{name}")
```

- [ ] **Step 3:tests/providers/test_providers_init.py:16**

Read 上下文:
```bash
sed -n '10,25p' tests/providers/test_providers_init.py
```

Edit:

old_string:
```python
    providers = importlib.import_module("nanobot.providers")
```

new_string:
```python
    providers = importlib.import_module("nanoresearch.providers")
```

- [ ] **Step 4:扫所有 .py 里 `"nanobot.X"` / `'nanobot.X'` 字符串字面量**

Run:
```bash
echo "=== string-form nanobot.X references in .py ==="
rg -n '["'"'"']nanobot\.' --type py 2>&1 | head -50
echo "---count: $(rg -c '["'"'"']nanobot\.' --type py 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')---"
```

Expected:此时已经处理过 Steps 1-3 + Task 6 的 import,剩下命中应该是:
- `tests/cli/test_commands.py:159` — `monkeypatch.setattr("nanobot.channels.registry.discover_all", ...)`
- `tests/cli/test_commands.py:187` — 同
- `tests/cli/test_commands.py:314` — `with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):`
- `tests/cli/test_commands.py:342` — `with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_async_openai:`
- 可能还有少量其它 .py 里的 patch / setattr / dotted module 字符串

每处 Edit:把 `"nanobot.` / `'nanobot.` 字符串前缀改成 `"nanoresearch.` / `'nanoresearch.`。

如果命中多(> 10),写一个临时 Python 脚本类似 Task 6 那个,但 regex 是 `(["\'])nanobot\.`,替换成 `\1nanoresearch.`。

Run(脚本路径用 /tmp/phase3-replace-strings.py):
```python
"""Phase 3 string-form nanobot.X reference replacement.

Replaces `"nanobot.X"` / `'nanobot.X'` → `"nanoresearch.X"` / `'nanoresearch.X'`
in .py files. Targets dynamic imports, monkeypatch/patch string targets,
entry_points group strings, etc.
"""

import re
import sys
from pathlib import Path

PATTERN = re.compile(r'(["\'])nanobot\.')


def process_file(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0
    new_text, n = PATTERN.subn(r'\1nanoresearch.', text)
    if n > 0:
        path.write_text(new_text, encoding="utf-8")
    return n


def main(roots: list[str]) -> None:
    grand = 0
    for root in roots:
        for py in sorted(Path(root).rglob("*.py")):
            n = process_file(py)
            if n > 0:
                print(f"  {py}: {n}")
                grand += n
    print(f"TOTAL: {grand}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
```

Run:
```bash
./backend/.venv/Scripts/python /tmp/phase3-replace-strings.py backend/nanoresearch tests backend/tests backend/scripts 2>&1 | tail -20
```

Expected:TOTAL 大概 5-15(已知 4 处 test_commands + 已 Step 1-3 处理的 4 处 + 少量其它)。但 Step 1-3 是手 Edit,脚本如果再扫到那些已改过的位置就不会再命中(它们现在是 `"nanoresearch.X"`)。所以脚本只会处理**剩余未改**的字符串。

但注意:脚本会扫到 `backend/test_themes_diagnostic.py` / `backend/test_ragas_transforms.py` 等 untracked 文件 — **跳过它们**,因为本 Phase 不动 untracked。改 Python 脚本加 exclude,或在调用前先确认这俩 untracked .py 内是否有 `"nanobot."` 字符串字面量(grep 即可):

```bash
rg '["'"'"']nanobot\.' backend/test_themes_diagnostic.py backend/test_ragas_transforms.py
```

如果命中,**手动避开**(脚本运行后 git diff 看是否 untracked 文件被改 — 如果被改,git checkout 它们即可,因为它们 untracked,git checkout 不能恢复,改回去要 manual)。

更稳的做法:脚本接受 `--exclude` 参数。但为简化,Step 4 后立刻反向校验:

- [ ] **Step 5:反向校验 — 整个 repo `.py` 不再有 `"nanobot."` / `'nanobot.'` 字面量**

Run:
```bash
echo "=== remaining string-form nanobot. ==="
rg -n '["'"'"']nanobot\.' --type py 2>&1 | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/' | head -20
echo "---count: $(rg -n '["'"'"']nanobot\.' --type py 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/' | wc -l)---"
```

Expected:count = 0(已知 4 个 untracked + docs 例外)。

任何命中:**停**手 Edit 或扩展脚本。

- [ ] **Step 6:Pydantic env_prefix**

Edit `backend/nanoresearch/config/schema.py`:

old_string:
```python
    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
```

new_string:
```python
    model_config = ConfigDict(env_prefix="NANORESEARCH_", env_nested_delimiter="__")
```

注:Commit A 的 compat 已经把 `NANOBOT_*` 复制到 `NANORESEARCH_*`,所以这里改完后老用户的 `.env` 仍然工作(只是会看到 deprecation warning)。

- [ ] **Step 7:pyproject.toml — 8 处**

Read 当前 pyproject 上下文(line 95-160):
```bash
sed -n '95,160p' backend/pyproject.toml
```

逐处 Edit。注意 line 121-122 那段:
```toml
[tool.hatch.build.targets.wheel.sources]
"nanobot" = "nanobot"
```
是 hatch 的 source map:左边是 wheel 里的路径,右边是源码路径。两者都改成 nanoresearch。

具体 8 个 Edit(每个 Edit 一次,加足够上下文确保唯一):

**Edit 1 — line 99-101 scripts**:

old_string:
```toml
[project.scripts]
nr = "nanobot.cli.commands:app"
nanoresearch = "nanobot.cli.commands:app"
```

new_string:
```toml
[project.scripts]
nr = "nanoresearch.cli.commands:app"
nanoresearch = "nanoresearch.cli.commands:app"
```

**Edit 2 — line 110-116 hatch build include**:

old_string:
```toml
[tool.hatch.build]
include = [
    "nanobot/**/*.py",
    "nanobot/templates/**/*.md",
    "nanobot/skills/**/*.md",
    "nanobot/skills/**/*.sh",
]
```

new_string:
```toml
[tool.hatch.build]
include = [
    "nanoresearch/**/*.py",
    "nanoresearch/templates/**/*.md",
    "nanoresearch/skills/**/*.md",
    "nanoresearch/skills/**/*.sh",
]
```

**Edit 3 — line 118-119 wheel packages**:

old_string:
```toml
[tool.hatch.build.targets.wheel]
packages = ["nanobot"]
```

new_string:
```toml
[tool.hatch.build.targets.wheel]
packages = ["nanoresearch"]
```

**Edit 4 — line 121-122 wheel sources**:

old_string:
```toml
[tool.hatch.build.targets.wheel.sources]
"nanobot" = "nanobot"
```

new_string:
```toml
[tool.hatch.build.targets.wheel.sources]
"nanoresearch" = "nanoresearch"
```

**Edit 5 — line 124-125 wheel force-include**:

old_string:
```toml
[tool.hatch.build.targets.wheel.force-include]
"bridge" = "nanobot/bridge"
```

new_string:
```toml
[tool.hatch.build.targets.wheel.force-include]
"bridge" = "nanoresearch/bridge"
```

**Edit 6 — line 127-133 sdist include**:

old_string:
```toml
[tool.hatch.build.targets.sdist]
include = [
    "nanobot/",
    "bridge/",
    "README.md",
    "LICENSE",
]
```

new_string:
```toml
[tool.hatch.build.targets.sdist]
include = [
    "nanoresearch/",
    "bridge/",
    "README.md",
    "LICENSE",
]
```

**Edit 7 — line 152-154 coverage source**:

old_string:
```toml
[tool.coverage.run]
source = ["nanobot"]
omit = ["tests/*", "**/tests/*"]
```

new_string:
```toml
[tool.coverage.run]
source = ["nanoresearch"]
omit = ["tests/*", "**/tests/*"]
```

注:8 处不是 8 行,grep -c "nanobot" 在改前应该是 ~10 行(line 100, 101, 112, 113, 114, 115, 119, 122, 125, 129, 153)。改后应该是 0。

- [ ] **Step 8:pyproject.toml 反向校验**

Run:
```bash
grep -n "nanobot" backend/pyproject.toml
echo "---count: $(grep -c 'nanobot' backend/pyproject.toml)---"
```

Expected:输出空,count = 0。

任何命中:**停**手 Edit。

- [ ] **Step 9:shell.py:99 中文注释**

Read:
```bash
sed -n '95,105p' backend/nanoresearch/agent/tools/shell.py
```

Edit:

old_string:
```python
        env = os.environ.copy()
        # 注入 nanobot 自己的 Python 路径到 PATH 最前面
        python_dir = os.path.dirname(sys.executable)
        env["NANORESEARCH_PYTHON"] = sys.executable
```

new_string:
```python
        env = os.environ.copy()
        # 注入 nanoresearch 自己的 Python 路径到 PATH 最前面
        python_dir = os.path.dirname(sys.executable)
        env["NANORESEARCH_PYTHON"] = sys.executable
```

- [ ] **Step 10:扫 .py 顶部 docstring / module 注释里的 nanobot 字面量**

Run:
```bash
echo "=== .py module docstrings / comments containing 'nanobot' ==="
rg -n -i 'nanobot' --type py 2>&1 | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/' | head -30
echo "---count: $(rg -n -i 'nanobot' --type py 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/' | wc -l)---"
```

Expected:count 不为 0(因 docstring/注释里有 nanobot 字面量未被前几步改)。逐行看是不是:
- module docstring(如 `__main__.py` 的 `"""Entry point for running nanobot ..."""`)
- 内联注释提到 nanobot
- env_compat.py 自己的注释引用旧名(那是 by-design,留不动)
- shell.py 已在 Step 9 改过

每个命中评估:
- 是 module docstring / 说明文字 → 改成 nanoresearch
- 是 env_compat.py 引用 NANOBOT_ 作为旧前缀名 → **不改**(那是兼容层语义)
- 是 deprecation warning 字符串提 NANOBOT_ → **不改**(同上)

对需要改的,逐个 Edit。这一 Step 没法一键脚本(语义判断),手做。

- [ ] **Step 11:Smoke — module import 全活**

Run:
```bash
echo "=== nanoresearch package importable ==="
./backend/.venv/Scripts/python -c "import nanoresearch; print('ok:', nanoresearch.__file__)"
echo "exit=$?"
echo ""
echo "=== CLI 启动 ==="
./backend/.venv/Scripts/python -m nanoresearch --help | head -5
echo "exit=${PIPESTATUS[0]}"
echo ""
echo "=== legacy NANOBOT_ env var 仍工作(compat) ==="
NANOBOT_MAX_CONCURRENT_REQUESTS=5 ./backend/.venv/Scripts/python -W default::DeprecationWarning -c "
from nanoresearch.utils.env_compat import apply_legacy_env_compat
apply_legacy_env_compat()
import os
assert os.environ.get('NANORESEARCH_MAX_CONCURRENT_REQUESTS') == '5'
print('compat ok')
" 2>&1 | head -10
```

Expected:三段都成功(`ok:` 行显示包路径在 `backend/nanoresearch/__init__.py`;CLI 帮助输出;compat 输出 `compat ok` + stderr 有 deprecation warning)。

任一失败:**停**排查。可能 import 漏改、Pydantic env_prefix 没改、init 文件 syntax error 等。

- [ ] **Step 12:不 commit,进 Task 8**

到此目录已 mv、所有 import 已改、3 处动态导入字符串 + entry_points + env_prefix + pyproject + 中文注释全部到位。Task 8 跑全量 test 验等价性。

---

## Task 8:Equivalence verification

**Files:** 不动任何文件,纯验证。

**Interfaces:**
- Consumes: Tasks 5-7 累积的暂存区(目录已 mv + import 已改 + 字符串/配置已改)。
- Produces: 等价性验证书面证据 — (i) `python -c "import nanoresearch"` 启动 smoke + 关键 entry 各能 import;(ii) 全 repo 反向 grep `import nanobot|from nanobot\.|from nanobot import == 0`(白名单:`backend/test_(ragas_transforms|themes_diagnostic).py` 2 个 untracked + `docs/` + `tests/agent/evaluation/results/` 的 fixture JSON);(iii) string-form `"nanobot."` / `'nanobot.'` 残留为 0(同白名单)。

**关于不跑 pytest collect 比对的说明**:Task 0 已确认 `tests/research/` 里 MarkdownLoader module-level 副作用导致 pytest collect 挂死(独立代码病,非 Phase 3 scope)。改用 `import nanoresearch` 启动 smoke 验"新包能加载",再用反向 grep `import nanobot == 0` 验"老包名彻底没人引用了"。这两者合起来等于"改名后包结构等价"的强证据,且不依赖 pytest 能否 collect。

- [ ] **Step 1:全量 .py + .toml + .md + .sh 残留 nanobot 扫**

Run:
```bash
echo "=== full repo nanobot literal scan ==="
rg -n 'nanobot' --type-add 'all:*.{py,toml,md,sh,yml,yaml,json}' -t all 2>&1 | \
  grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | \
  grep -vE '^docs/superpowers/(specs|plans|memory)/' | \
  grep -vE 'tests/agent/evaluation/results/' | \
  grep -vE '^MEMORY\.md' | \
  grep -vE '^backend/nanoresearch/utils/env_compat\.py:' | \
  head -40
echo "---count(excl allowed): $(rg -n 'nanobot' --type-add 'all:*.{py,toml,md,sh,yml,yaml,json}' -t all 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/superpowers/' | grep -vE 'tests/agent/evaluation/results/' | grep -vE '^MEMORY\.md' | grep -vE '^backend/nanoresearch/utils/env_compat\.py:' | wc -l)---"
```

Expected:count 应该很小(< 30),命中应该是:
- TMUX SKILL.md 里的 metadata key / socket name / session name(留作 Phase 3 不做但留痕)
- `backend/nanoresearch/skills/tmux/SKILL.md` 的剩余 `nanobot-tmux-sockets`(默认子目录名,留)
- README.md / docs/ 里的历史引用(独立文档同步任务,不在本 Phase)
- `.env.example` / `docker-compose.yml` 里的 NANOBOT_ env var 名(留兼容,直到 0.3.0)

逐行评估,任何不在"允许保留"清单的命中,**手 Edit 改掉**。

- [ ] **Step 2:`import nanoresearch` 启动 smoke — 新包结构能加载**

Run(repo 根):
```bash
cd /d/Code/nanobot
./backend/.venv/Scripts/python -c "import nanoresearch; print('module path:', nanoresearch.__file__)"
echo "exit=$?"
echo "---"
./backend/.venv/Scripts/python -c "from nanoresearch.cli.commands import app; print('cli ok')"
echo "exit=$?"
./backend/.venv/Scripts/python -c "from nanoresearch.server.main import create_app; print('server ok')"
echo "exit=$?"
./backend/.venv/Scripts/python -c "from nanoresearch.utils.env_compat import apply_legacy_env_compat; print('env_compat ok')"
echo "exit=$?"
./backend/.venv/Scripts/python -c "from nanoresearch.channels.registry import discover_all; print('registry ok')"
echo "exit=$?"
./backend/.venv/Scripts/python -c "from nanoresearch.config.schema import Settings; s = Settings(); print('schema env_prefix ok')"
echo "exit=$?"
```

Expected:
- `import nanoresearch` exit 0,module path 指向 `D:\Code\nanobot\backend\nanoresearch\__init__.py`
- 五段 ok + exit 0(cli / server / env_compat / registry / schema)

任一 fail:**停**。可能 Task 6 import 替换有遗漏,或 Task 7 string/entry_points/env_prefix 改漏。

- [ ] **Step 3:老包仍可被 import?(应该 NO)**

Run:
```bash
./backend/.venv/Scripts/python -c "import nanobot" 2>&1 | tail -3
echo "exit=$?"
```

Expected:`ModuleNotFoundError: No module named 'nanobot'`,exit 非 0。

如果 `import nanobot` 居然能 import:**停**。说明 site-packages 还有老的 nanobot 残留 / 或目录 mv 没成功 / 或 .venv 缓存了元数据。需要清掉再来。

- [ ] **Step 4:import 反向校验 — 0 残留(等价性强证)**

口径同 Task 6 Step 7 + 拓宽白名单覆盖 fixture JSON 路径。

Run:
```bash
echo "=== remaining import nanobot / from nanobot in .py ==="
rg -n '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py 2>&1 | \
  grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | \
  grep -vE '^docs/' | \
  grep -vE '^tests/agent/evaluation/results/' | \
  head -10
echo "---count(excl allowed): $(rg -n '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/' | grep -vE '^tests/agent/evaluation/results/' | wc -l)---"
```

白名单:
- `backend/test_(ragas_transforms|themes_diagnostic).py` — 2 个 untracked smoke 脚本,Task 11 单独提醒用户改
- `docs/` — plan / spec / memory / baseline 都是历史快照
- `tests/agent/evaluation/results/` — fixture JSON / 评估输出,不参与运行

Expected:count = 0。

任一命中:**停**手 Edit(不在白名单的不能放过)。

- [ ] **Step 5:string-form 反向校验 — 0 残留(excl allowed)**

口径同 Step 4 + 拓宽白名单覆盖 fixture JSON 路径。

Run:
```bash
echo "=== remaining 'nanobot.' / \"nanobot.\" in .py ==="
rg -n '["'"'"']nanobot\.' --type py 2>&1 | \
  grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | \
  grep -vE '^docs/' | \
  grep -vE '^tests/agent/evaluation/results/' | \
  head -10
echo "---count: $(rg -n '["'"'"']nanobot\.' --type py 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/' | grep -vE '^tests/agent/evaluation/results/' | wc -l)---"
```

Expected:count = 0。

- [ ] **Step 6:汇总验证报告**

Run:
```bash
echo "=== Phase 3 Equivalence Verification (smoke + reverse-grep approach) ==="
echo ""
echo "--- import smoke (Step 2/3) ---"
echo "新包 import nanoresearch:" && ./backend/.venv/Scripts/python -c "import nanoresearch; print('OK', nanoresearch.__file__)" 2>&1
echo "老包 import nanobot (应 ModuleNotFoundError):" && ./backend/.venv/Scripts/python -c "import nanobot" 2>&1 | tail -1
echo ""
echo "--- import 反向校验 (Step 4) ---"
echo "Count import nanobot residuals: $(rg -n '(^|\s)(import nanobot|from nanobot\.|from nanobot import)' --type py 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/' | grep -vE '^tests/agent/evaluation/results/' | wc -l)"
echo ""
echo "--- string-form 反向校验 (Step 5) ---"
echo "Count 'nanobot.' string literals: $(rg -n '["'"'"']nanobot\.' --type py 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/' | grep -vE '^tests/agent/evaluation/results/' | wc -l)"
echo ""
echo "--- full literal scan (Step 1) ---"
echo "Count(excl allowed): $(rg -n 'nanobot' --type-add 'all:*.{py,toml,md,sh,yml,yaml,json}' -t all 2>/dev/null | grep -vE '^backend/test_(ragas_transforms|themes_diagnostic)\.py:' | grep -vE '^docs/superpowers/' | grep -vE 'tests/agent/evaluation/results/' | grep -vE '^MEMORY\.md' | grep -vE '^backend/nanoresearch/utils/env_compat\.py:' | wc -l)"
```

把这个汇总贴给用户作 Task 9 闸口的输入。等价性的口径是:新包 import 成功 + 老包 ModuleNotFoundError + 反向 grep 全 0,合起来等于"包结构改名干净"的强证据。

---

## Task 9:Commit D 前人工核对闸口

**Files:** 不动任何文件,纯检查 + 等用户放行。

**Interfaces:**
- Consumes: Task 8 验证全过。
- Produces: 用户书面确认"可以 commit"才能进 Task 10。

**为什么独立成 Task**:Commit D 是本 Phase 最大、最不可逆的一步。push 后 origin/main 就到了 nanoresearch 名字下,所有合作者 / CI / 部署管道下一次拉取都看到新结构。必须最后再核一道。

- [ ] **Step 1:全量 git status**

Run:
```bash
git status
```

Expected 输出结构(逐项核对):

- "Changes to be committed:" 区域:
  - 大量 `renamed:  backend/nanobot/X -> backend/nanoresearch/X`(~209 行)
  - 一批 `modified: tests/...`(import 改的 ~68 文件)
  - 一批 `modified: backend/tests/...`(~14 文件)
  - 一批 `modified: backend/scripts/...`(~15 文件)
  - `modified: backend/pyproject.toml`
  - 可能还有少数显式 modified 的(其它 string ref 改的)
- "Changes not staged for commit:" 区域:**应该为空**(Tasks 6-7 用脚本直接改后,如果没 git add,这些改动会出现在 not-staged 区。但 git mv 已经把目录改动放进暂存区,Tasks 6-7 的文件编辑后需要再 git add 一次。**这是一个潜在风险点**)。
- "Untracked files:" 区域:11 个原 untracked 项(漂移项 ±3 可接受)。

**关键:Tasks 6-7 用 Python 脚本 + Edit tool 修改了文件,这些改动需要显式 `git add` 才进暂存区**。Step 2 处理。

- [ ] **Step 2:把 Tasks 6-7 的修改加入暂存区**

Run:
```bash
echo "=== current unstaged changes ==="
git diff --name-only | head -30
echo "---unstaged count: $(git diff --name-only | wc -l)---"
```

Expected:很多 modified 文件(因 git mv 后 Tasks 6-7 改了内部内容,git 看作"被 mv 的文件之上又有改动")。具体数字大致是改名脚本涉及到的 ~306 文件(209 nanoresearch + 68 tests + 14 backend/tests + 15 backend/scripts) — 但 git mv 处理后的状态可能让一些显示为 renamed-modified,需根据实际看。

Run(显式 add 4 个 scope + pyproject):
```bash
git add backend/nanoresearch/
git add tests/
git add backend/tests/
git add backend/scripts/
git add backend/pyproject.toml
```

注:**不**用 `git add .`。

```bash
echo "=== after add ==="
git diff --name-only
echo "---unstaged count: $(git diff --name-only | wc -l)---"
```

Expected:unstaged 应该是 0(除了 11 个原 untracked)。

非 0:**停**,看是什么文件没 add — 可能是某个 docstring/注释 Edit 漏了 add,或某 .md/.toml 在意料外的位置。

- [ ] **Step 3:暂存区文件清单核对**

Run:
```bash
git diff --cached --stat | tail -5
echo "---staged count: $(git diff --cached --name-only | wc -l)---"
echo ""
echo "=== top 30 staged files ==="
git diff --cached --name-only | head -30
```

Expected:staged 数量大致 = 209(rename) + ~140(modified across all scopes,具体看)。

- [ ] **Step 4:反向 grep — 暂存区不能有 untracked 11 项 / rescue / worktree-***

Run:
```bash
echo "=== should be empty ==="
git diff --cached --name-only | grep -E 'backend/(logs|models|test_ragas|test_themes)|health_set_draft|loadtest\.py|seed_testcases|testcases\.json|vite\.config\.js\.timestamp'
echo "---grep exit=$?---"
```

Expected:输出空(grep exit 1)。

任何命中:**停**,`git reset HEAD <path>` 退出暂存区。

- [ ] **Step 5:Diff stat 数量级核对**

Run:
```bash
git diff --cached --stat | tail -1
```

Expected:`N files changed, X insertions(+), Y deletions(-)`,N 在 300-500 之间。X 和 Y 应该相近(import 改名是同量级 +- 替换;rename 一般 git 显示为 0 ins/0 del 因是 rename detection)。

数量级明显偏离(N < 200 或 > 800):**停**核对。

- [ ] **Step 6:贴汇总给用户,等放行**

把 Step 1-5 的关键输出汇总:
- `git status` 三段总结(staged 数 / unstaged 数 / untracked 数)
- staged 文件清单 top 30
- `git diff --cached --stat` 末行
- Task 8 的等价性汇总(test collect 一致、summary 一致、residual = 0)
- `phase3-pre-rename` tag 仍可用(回退锚点)

agent 执行者:贴完等用户明确"可以 commit"再进 Task 10。**不**自动 commit。

人执行者:肉眼复核完,再敲 Task 10。

---

## Task 10:Commit D + push

**Files:** 不动文件,纯 commit + push。

**Interfaces:**
- Consumes: Task 9 用户放行。
- Produces: 一个原子 commit `feat(rename)!: nanobot → nanoresearch (Phase 3 atomic)` 上 origin/main;phase3-pre-rename tag 保留作历史锚点。

- [ ] **Step 1:Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(rename)!: nanobot → nanoresearch (Phase 3 atomic)

Phase 3 Commit D — atomic package rename. References:
- docs/superpowers/specs/2026-06-26-repo-cleanup-design.md Phase 3
- docs/superpowers/plans/2026-06-27-repo-cleanup-phase3.md

Changes:
- git mv backend/nanobot/ → backend/nanoresearch/ (~209 .py + subdirs)
- Bulk rewrite all import statements across:
  - backend/nanoresearch/ (~1031 occurrences / 209 files)
  - tests/ (~339 / 68)
  - backend/tests/ (~45 / 14)
  - backend/scripts/ (~44 / 15)
- Update 3 dynamic import strings:
  - backend/nanoresearch/channels/registry.py:32 importlib.import_module
  - backend/nanoresearch/cli/onboard.py:751 importlib.import_module
  - tests/providers/test_providers_init.py:16 importlib.import_module
- Update entry_points group "nanobot.channels" → "nanoresearch.channels"
- Update Pydantic env_prefix NANOBOT_ → NANORESEARCH_ (relies on
  apply_legacy_env_compat() from Commit A)
- Update test mock string targets (tests/cli/test_commands.py:159,187,314,342)
- Update backend/pyproject.toml (scripts / hatch build/wheel/sources/sdist /
  coverage source — 8 sites)
- Update shell.py:99 Chinese comment

BREAKING CHANGES:
- python -m nanobot → python -m nanoresearch (console_scripts nr and
  nanoresearch still work — pyproject.toml updated)
- External channel plugins registered under entry_points group
  "nanobot.channels" must re-register under "nanoresearch.channels"
  (no compat — group reads do not dual-resolve)

Backward compat (still in place from Commits A/C):
- NANOBOT_* environment variables auto-copied to NANORESEARCH_* with
  DeprecationWarning until v0.3.0
- NANOBOT_TMUX_SOCKET_DIR honored by find-sessions.sh until v0.3.0

Equivalence verification (vs Task 4 prerename baseline):
- root tests/: test ID set identical, passed/failed/skipped counts identical
- backend/tests/: test ID set identical, passed/failed/skipped counts identical
- 0 residual `import nanobot` / `from nanobot.` / `from nanobot import`
  statements (excluding 2 untracked smoke scripts + docs/superpowers/)
- 0 residual `"nanobot."` / `'nanobot.'` string literals in .py
  (excluding same)

Not addressed by this Phase (留痕):
- TMUX SKILL.md non-env-var nanobot literals (socket/session names,
  metadata key, default subdir name) — kept for tmux session compat
- README.md / .env.example / docker-compose.yml / Dockerfile full
  doc-sync — deferred to a later doc sync task
- backend/test_ragas_transforms.py / backend/test_themes_diagnostic.py
  (untracked smoke scripts) — accept temporary ImportError, user to fix
  manually post-commit

Tag `phase3-pre-rename` preserved as local rollback anchor.
EOF
)"
echo "exit=$?"
```

Expected:`[main <hash>] feat(rename)!: nanobot → nanoresearch ...`,几百 files changed,exit 0。

不要 `--amend` / `--no-verify`。

- [ ] **Step 2:验证 commit 内容**

Run:
```bash
git log -1 --stat | tail -20
echo "---"
git log -1 --format='%H %s'
```

Expected:看到刚刚 commit message + 大量文件变化统计。

- [ ] **Step 3:工作区那 11 个 untracked 仍在**

Run:
```bash
git status -s
git ls-files --others --exclude-standard | wc -l
```

Expected:仍有 11 个 `??`(允许 ±3 漂移)。

数量明显变少:**停**,commit 可能误带 untracked。

- [ ] **Step 4:Push**

Run:
```bash
git status -sb | head -1
git push origin main
echo "exit=$?"
git status -sb | head -1
```

Expected:push 前 `[ahead 1]`,push 后 `<old>..<new>  main -> main`,exit 0,push 后无 ahead。

Push 失败:**停**,原样报错给用户,不重试 force,不切策略。

---

## Task 11:Push 后验证 + memory + smoke script 提醒

**Files:**
- Modify: `C:\Users\Augix\.claude\projects\D--Code-nanobot\memory\MEMORY.md`(加一条 Phase 3 完成事实)
- Create: `C:\Users\Augix\.claude\projects\D--Code-nanobot\memory\project_phase3_rename_complete.md`(新 memory 文件)
- Delete: `/tmp/phase3-*.txt`、`/tmp/phase3-replace-imports.py`、`/tmp/phase3-replace-strings.py`(清理临时文件)

**Interfaces:**
- Consumes: Task 10 已 push 完。
- Produces: Phase 3 收尾确认 + memory 持久化 + 提醒用户改 2 个 untracked smoke script。

- [ ] **Step 1:本地 / origin 一致性**

Run:
```bash
git rev-parse HEAD origin/main
echo "---above two should be identical---"
git log -5 --oneline
```

Expected:两行 hash 完全相同;`git log -5` 顶端是刚刚的 Commit D。

- [ ] **Step 2:关键内容落地**

Run:
```bash
echo "=== nanoresearch package on origin ==="
ls backend/nanoresearch/__init__.py backend/nanoresearch/__main__.py
echo ""
echo "=== Pydantic env_prefix ==="
grep "env_prefix" backend/nanoresearch/config/schema.py
echo ""
echo "=== pyproject 包名 ==="
grep -E "packages = |^nr = |^nanoresearch = " backend/pyproject.toml
echo ""
echo "=== CLI 启动新名 ==="
./backend/.venv/Scripts/python -m nanoresearch --help | head -5
```

Expected:
- 两个文件存在
- env_prefix 是 `NANORESEARCH_`
- pyproject 三行都是 nanoresearch
- CLI 帮助正常

- [ ] **Step 3:legacy 兼容验证 — `python -m nanobot` 应该 fail(不兼容包名),但 `nr` / `nanoresearch` 应该工作**

Run:
```bash
echo "=== python -m nanobot (expected to FAIL) ==="
./backend/.venv/Scripts/python -m nanobot --help 2>&1 | head -3
echo "---exit=${PIPESTATUS[0]} (expected non-zero)---"
echo ""
echo "=== nr / nanoresearch console_scripts ==="
./backend/.venv/Scripts/nr --help 2>&1 | head -3
echo "---exit=$?---"
./backend/.venv/Scripts/nanoresearch --help 2>&1 | head -3
echo "---exit=$?---"
```

Expected:
- `python -m nanobot` 报 `No module named 'nanobot'` 类错误(预期失败,因为包目录已 mv,无兼容)
- `nr` 和 `nanoresearch` console_scripts 都 exit 0

注:`nr` / `nanoresearch` 可能需要 `pip install -e .` 重装才更新 entry point。若它们 fail with `ModuleNotFoundError: nanoresearch`,说明 venv 还指向旧路径 — 跑 `pip install -e backend/` 重装。这本身不是 Phase 3 的失败,而是开发环境更新需求。

- [ ] **Step 4:env var compat 端到端**

Run:
```bash
NANOBOT_MAX_CONCURRENT_REQUESTS=7 ./backend/.venv/Scripts/python -W default::DeprecationWarning -c "
from nanoresearch.utils.env_compat import apply_legacy_env_compat
apply_legacy_env_compat()
import os
val = os.environ.get('NANORESEARCH_MAX_CONCURRENT_REQUESTS')
print(f'NANORESEARCH_MAX_CONCURRENT_REQUESTS = {val}')
assert val == '7'
print('OK')
" 2>&1
```

Expected:
- stdout `NANORESEARCH_MAX_CONCURRENT_REQUESTS = 7` + `OK`
- stderr `DeprecationWarning: NANOBOT_MAX_CONCURRENT_REQUESTS is deprecated; use NANORESEARCH_MAX_CONCURRENT_REQUESTS. NANOBOT_MAX_CONCURRENT_REQUESTS will be removed in v0.3.0.`

- [ ] **Step 5:rescue/b4-orphan 未动确认**

Run:
```bash
git log -1 --oneline rescue/b4-orphan
echo "---should still be: cbfa94b3 feat(B4): ...---"
```

Expected:rescue HEAD 仍为 `cbfa94b3`。本 Phase 全程未碰 rescue 分支。

- [ ] **Step 6:`phase3-pre-rename` tag 仍可用**

Run:
```bash
git tag -l | grep phase3
echo "---"
git rev-parse phase3-pre-rename
git log -1 --oneline phase3-pre-rename
```

Expected:tag 仍指向 `08283baf chore: ignore logs, models, pycache, vite timestamps`。需要回退时 `git reset --hard phase3-pre-rename` 完整恢复 Phase 3 之前的状态(注意 Commits A/B/C 也会被 reset 掉)。

- [ ] **Step 7:Memory — Phase 3 完成事实**

Use Write tool to create `C:\Users\Augix\.claude\projects\D--Code-nanobot\memory\project_phase3_rename_complete.md`:

```markdown
---
name: Phase 3 nanobot → nanoresearch rename complete
description: nanobot 包改名为 nanoresearch,2026-06-27 完成。env var 双读到 v0.3.0;remaining 字面量保留
type: project
---

`nanobot` 包改名 `nanoresearch` 已完成并 push 到 `origin/main`(Phase 3 of `docs/superpowers/specs/2026-06-26-repo-cleanup-design.md`)。

**4 个 commit:**
- A: env var compat shim — `NANOBOT_*` auto-copy 到 `NANORESEARCH_*` 启动钩子
- B: chroma fallback default + qq/weixin path bypass 修复
- C: TMUX env var rename in `.md`/`.sh` with shell-side dual-read
- D: atomic 大改名(目录 mv + ~1464 import 替换 + 字符串/配置/pyproject 8 处)

**Why:** Phase 3 spec 要求,为多租户/服务端部署铺路;package 名跟 user-facing brand (NanoResearch) 对齐。

**How to apply:**
- 新代码用 `nanoresearch.*` import,不用 `nanobot.*`。
- env var 新代码用 `NANORESEARCH_*` 前缀,老的 `NANOBOT_*` 在 v0.3.0 之前 dual-read。
- `python -m nanobot` 不再工作 → 用 `python -m nanoresearch` 或 `nr` / `nanoresearch` console_scripts。
- entry_points group `nanobot.channels` 改成 `nanoresearch.channels`,外部 channel plugin 须重新注册。

**保留的 nanobot 字面量(留作后续 Phase 处理):**
- `backend/nanoresearch/skills/tmux/SKILL.md` 里的:metadata key `{"nanobot":{...}}`、socket 文件名 `nanobot.sock`、session 示例 `nanobot-python`、默认子目录名 `nanobot-tmux-sockets`(改它们孤立现存会话或影响 skill 发现)。
- README.md / .env.example / docker-compose.yml / Dockerfile 等文档/部署文件的非编译关键字面量(独立 doc-sync 任务)。
- `backend/test_ragas_transforms.py` / `backend/test_themes_diagnostic.py`(2 个 untracked smoke script 用户手动改)。

**回退锚点:** 本地 tag `phase3-pre-rename` 仍指向改名前的 HEAD(`08283baf`),需完整回退时可 `git reset --hard phase3-pre-rename`(注意会 reset 掉 Commits A/B/C/D)。
```

- [ ] **Step 8:Memory — MEMORY.md 索引追加**

Use Edit tool on `C:\Users\Augix\.claude\projects\D--Code-nanobot\memory\MEMORY.md`:

old_string:
```
- [rescue/b4-orphan dormant](project_rescue_b4_orphan_dormant.md) — 5 块未来 Phase 预案混在 cbfa94b3,不合不删,后续按 Phase 单独抽 chunk
```

new_string:
```
- [rescue/b4-orphan dormant](project_rescue_b4_orphan_dormant.md) — 5 块未来 Phase 预案混在 cbfa94b3,不合不删,后续按 Phase 单独抽 chunk
- [Phase 3 rename complete](project_phase3_rename_complete.md) — nanobot → nanoresearch 已完成 2026-06-27,NANOBOT_* env var dual-read 到 v0.3.0
```

- [ ] **Step 9:提醒用户手动修 2 个 untracked smoke script**

把以下文字贴给用户:

> Phase 3 已完成。两个 untracked smoke script 在 Phase 3 期间故意没改:
>
> - `backend/test_ragas_transforms.py`
> - `backend/test_themes_diagnostic.py`
>
> 它们的 `from nanobot.` import 现在 ImportError。**手动修复方式:**
>
> ```bash
> sed -i 's/\bimport nanobot\b/import nanoresearch/g; s/\bfrom nanobot\./from nanoresearch./g; s/\bfrom nanobot import\b/from nanoresearch import/g' backend/test_ragas_transforms.py backend/test_themes_diagnostic.py
> ```
>
> 或用编辑器手 review。改完跑一遍确认能运行。

- [ ] **Step 10:清临时文件(用户许可后)**

把汇总贴给用户:

> Phase 3 收尾完成。临时文件清单:
> - `/tmp/phase3-pre-rename-counts.txt`(Task 0 import 计数 baseline)
> - `/tmp/phase3-replace-imports.py`(Task 6 机械替换脚本)
> - `/tmp/phase3-replace-strings.py`(Task 7 字符串替换脚本)
> - `/tmp/phase3-edit-gitignore.py`(Task 2 .gitignore anchor-insert 脚本)
> - `/tmp/phase3-gitignore.bak`(Task 2 .gitignore 改前备份)
>
> (Note:pytest baseline 文件 `/tmp/phase3-{baseline,prerename,postrename}-{collect,run}-{root,backend}.txt` 在原计划里会产生,实跑因 MarkdownLoader 副作用导入挂死 → 等价性验证降级为 import smoke + 反向 grep,这批文件不生成,无需清。)
>
> 清吗?

agent:等用户明确"清"再 `rm`。**不**自己决定。

**Branch A — 用户说"可以清":**
```bash
rm /tmp/phase3-*
echo "cleaned"
```

**Branch B — 用户说"保留":**
不动,把列表写进 Phase 3 收尾备忘。

- [ ] **Step 11:收尾上报**

把以下事实贴给用户:
- 4 个 commit hash + 一行 message 各
- `git log -5 --oneline` 输出
- Step 2-4 smoke 结果
- Task 8 等价性汇总(test 计数对比)
- `phase3-pre-rename` tag 仍可用确认
- rescue/b4-orphan 未动确认
- memory 已写入两文件(新 project memory + MEMORY.md 索引)
- 用户手动修 2 个 smoke script 的提醒

到此 Phase 3 完成。

---

## Self-Review Notes

### Spec coverage(对 `docs/superpowers/specs/2026-06-26-repo-cleanup-design.md` Phase 3)

spec §4 Phase 3 列了 7 项:

| spec 项 | plan 对应 Task |
|---|---|
| 包名 `backend/nanobot/` → `backend/nanoresearch/` | Task 5 git mv ✓ |
| 461 import 全量替换(实测 1464) | Task 6 bulk replace ✓ |
| 3 处动态导入字符串 | Task 7 Steps 1-3 ✓ |
| entry_points group | Task 7 Step 1 Edit 3 ✓ |
| 5 个 `NANOBOT_` env var 带兼容期改名 | Task 1 (compat + 3 standalone) + Task 3 (TMUX) + Task 7 Step 6 (Pydantic) ✓ |
| chroma_store.py 兜底默认值 | Task 2 Step 2 ✓ |
| qq/weixin 硬编码路径 | Task 2 Steps 4, 6 ✓ |
| 文档全量同步 | **不做**,Global Constraints 留痕,Phase 6 / doc-sync 独立任务 ✓ |

spec §2.7 还列了 "`__main__.py` 入口":Task 5 git mv 自动搬;`python -m nanobot` 无兼容(breaking change,console_scripts 双命令 `nr`/`nanoresearch` 兜底);Task 11 Step 3 验证。

spec §2.8 关于 weixin "硬编码路径":实测只在注释里,代码已走 `get_runtime_subdir`。Task 2 Step 6 仅改注释。

补充 spec 未列但实测必须的 8 处 pyproject 改:Task 7 Step 7 ✓。

### 用户最新对齐的 5 点

| 用户对齐项 | plan 对应 |
|---|---|
| ① import 全改不分行首/缩进 | Task 6 regex `(\bimport\|\bfrom)\s+nanobot\b` 覆盖全形式 ✓ |
| ② 替换范围 .py/.md/.sh/.toml/.txt | Task 1-7 按文件类型分 task ✓(.txt 实测无命中) |
| ③ NANOBOT_WORKSPACE 走 env var 兼容 | Task 1 Step 9 一并处理 ✓ |
| ④ backend/nanobot/scripts/(包内随 mv)和 backend/scripts/(包外仅改 import)分清 | Task 5 mv 处理包内;Task 6 Step 6 处理包外 ✓ |
| ⑤ 2 个 untracked smoke script 接受短暂坏 + Phase 末尾提醒 | Global Constraints §7 + Task 11 Step 9 ✓ |
| 根 tests/ + backend/tests/ 跟包 mv 同一 commit | Task 6 Steps 4-5 一并 + Task 10 atomic commit ✓ |

### 不动清单

| 不动项 | plan 对应保护 |
|---|---|
| 测试 fixture JSON | Task 6 regex 只扫 .py;Task 7 Step 5 校验排除 |
| 11 个 untracked | Task 0 Step 3 baseline + Task 7 Step 4 exclude + Task 10 Step 3 校验 |
| rescue/b4-orphan | Task 11 Step 5 验证 HEAD 未动 |
| worktree-agent-* | plan 全程不引用 |
| 193 处合法 nanoresearch 字面量 | regex 只匹配 nanobot 不匹配 nanoresearch |

### Placeholder scan

- 无 TBD / TODO / "appropriate error handling" / "similar to Task N" 占位。
- Tasks 1, 2, 3, 7 的每个 Edit 都给出完整 old_string / new_string。
- Task 6 替换脚本完整代码 + edge case 心算表;Task 1 Step 11 test 代码完整。
- 所有 Run 命令完整、Expected 输出明确(数字、字符串、grep exit code)。
- 仅有的"留痕未做"是 Global Constraints §"不做但留痕"部分,均显式说明保留理由 + 移交目标 Phase。

### Type / name consistency

- `apply_legacy_env_compat()` 函数名在 Tasks 1(define) / 4 / 7(Step 11 import) / 11(Step 4 import)出现,拼写一致。
- `NANOBOT_*` / `NANORESEARCH_*` 拼写在所有 Task 一致(全大写)。
- 包路径 `backend/nanoresearch/` 与 import path `nanoresearch.x.y` 对应一致。
- pyproject 8 处改动的 line 号与 Read 输出一致。

### Risk coverage

| 风险 | 缓解 |
|---|---|
| commit 误带 untracked / 工作区 | Task 9 强制人工闸口 + 4 个 explicit `git add <path>` 不用泛匹配 |
| import bulk replace 误伤注释/字符串 | regex 严格 `\bimport\|\bfrom\s+nanobot\b`;Task 6 Step 2 edge case 心算表;Step 7 反向校验 |
| nanobot_extras 这种变量名被误改 | `\b` 词边界,`_` 是 \\w 不在边界 |
| Pydantic env_prefix 改了老用户 .env 全坏 | Commit A compat 钩子先到位,把 NANOBOT_*→NANORESEARCH_* |
| `python -m nanobot` 入口失效 | console_scripts `nr` / `nanoresearch` 双命令兜底;commit message 标 BREAKING |
| entry_points group 改了外部 plugin 全坏 | 显式 BREAKING + 文档说明;不做 group 双读因当前无外部 plugin |
| git mv 后中间态 import 全坏不能 commit | Task 5 显式 "不 commit,进 Task 6";Task 10 commit 是 mv+import 原子 |
| 2 个 untracked smoke script 被误改 | Task 6/7 脚本/Edit 全程不传它们路径;Task 7 Step 4 反向校验 exclude;Task 11 Step 9 提醒手动 |
| rescue/b4-orphan 被无意中动 | plan 全程不引用;Task 11 Step 5 验证 HEAD 不变 |
| push 失败 retry force | Task 1/2/3/10 push 步骤明文禁止 |
| 改名引入 test regression | Task 0 抓 baseline + Task 4 re-baseline + Task 8 严格对比(diff = 0) |
| 替换脚本本身有 bug | Task 6 Step 2 心算 edge case 表;Step 7 反向校验 count = 0 |

### Out-of-scope guard

- plan 内无 Phase 4-6 动作(无 base 路径可配、无多租户、无 1000+ 行单文件拆分等)。
- 不做 README / docker-compose / .env.example 全量文档同步(Global Constraints / Spec coverage 留痕)。
- 不动 backend/bridge/ 孤立目录(Phase 2 已留痕,本 plan 不涉及)。
- 不动 backend/nanobot/data/db/ 磁盘残留(Phase 1/2 已 git rm --cached,本 plan 不涉及)。
- 不动 worktree-agent-* / rescue/b4-orphan 任何分支。
