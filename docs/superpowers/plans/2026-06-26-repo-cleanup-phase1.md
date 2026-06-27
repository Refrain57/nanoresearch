# 仓库清理 Phase 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `.gitignore` 覆盖运行时数据 + 把已误追踪的 4 批运行时数据从 git index 移除，commit & push，不动磁盘文件、不动工作区那 60+ 个未暂存源码修改、不动历史。

**Architecture:** 纯 git 元数据操作 + 一行 `.gitignore` 修改。`git rm --cached` 只动 index 不动磁盘，所以本地 `python -m nanobot`、`pnpm dev` 等命令在 push 后仍然可跑（磁盘上的 `node_modules/`、chroma 数据、`.db` 文件原样保留）。唯一真有可能出事的环节是 commit 时误把工作区 60+ 个未暂存源码修改带进 commit —— 用一个独立的人工核对步骤把它拦在 commit 之前。

**Tech Stack:** git（CLI）。无其他依赖。

## Global Constraints

来自 spec `docs/superpowers/specs/2026-06-26-repo-cleanup-design.md` Phase 1：

- 本轮只做 Phase 1，**不**碰 Phase 2-6 的任何动作：不删散落文件、不改任何源码、不做改名、不修 `.gitignore` 里 `*.claude` 疑似 bug。
- 不破坏工作区当前 60+ 个未暂存的源码修改。
- 不重写历史、不 force push。普通 commit + push。
- commit 暂存策略：**只 `git add .gitignore`**，绝不用 `git add .` 或 `git add -A`。`git rm --cached` 产生的 deletion 已自动入暂存区，不需要再 add。
- commit 之前必须有一次**显式人工核对**，肉眼确认暂存区只含 `.gitignore` 一处 modified + 一批 `deleted:`，没有掺入工作区源码修改。
- `git rm --cached` 不动磁盘文件，无需"先备份"。
- push 后必须验证：`git status` 不再出现 `node_modules` / `workspace` / `data/db` 路径，且本地 `python -m nanobot`、前端 dev 命令仍能跑。

---

## File Structure

本 Phase 唯一被修改/创建的工程文件：

- Modify: `.gitignore` —— 末尾追加 5 条规则块（详见 Task 1）。

被 `git rm --cached` 从 index 移除（磁盘文件保留原样）：

- `web/node_modules/`（整目录，58,609 文件）
- `backend/nanobot/nanobot/workspace/`（整目录，5 个 chroma 文件）
- `backend/nanobot/data/db/image_index.db`
- `backend/nanobot/data/db/ingestion_history.db`

不创建测试文件、不创建新源码文件。Phase 1 的"测试"是 git 状态与本地启动命令的人工验证，不写 pytest。

---

## Task 1: 追加 .gitignore 规则

**Files:**
- Modify: `.gitignore`（末尾追加 5 条规则块）

**Interfaces:**
- Consumes: 无
- Produces: 经过本任务后，未来运行时再生成 `node_modules/`、`backend/nanobot/nanobot/workspace/`、`backend/nanobot/data/db/`、任意 `*.log`、`token.json` 都不会再被 git 看见。Task 2 的 `git rm --cached` 配合本任务规则，才能让被移除的路径不立刻重新冒回 untracked 列表。

- [ ] **Step 1: 检查当前 .gitignore 末尾**

Run: `tail -5 .gitignore`
Expected: 输出现有最后几行，确认末尾不是空行也没未提交 trailing 内容。本 Step 只为确定追加位置，不修改。

- [ ] **Step 2: 在 .gitignore 末尾追加 5 条规则**

在文件末尾追加（用 Edit 或编辑器，**不要**用 `>>` 重定向以免编码问题），新增以下块：

```
# Phase 1 cleanup: runtime data must never be tracked
node_modules/
backend/nanobot/nanobot/workspace/
backend/nanobot/data/db/
*.log
token.json
```

注：`.gitignore` 第 44 行已有 `*.db` 全局规则、第 27 行已有 `botpy.log`。追加规则与现有规则有重叠 —— 重叠无害（spec §5 已确认），目的是把 Phase 1 关心的 4 条路径**显式锁住**，避免后续误改 `.gitignore` 时漏一个。

- [ ] **Step 3: 验证 .gitignore 修改正确**

Run: `git diff .gitignore`
Expected: 只看到末尾追加了上面 6 行（含注释行）。没有任何其他改动、没有删除行。

- [ ] **Step 4: 验证 .gitignore 改动只影响工作区，未污染暂存区**

Run: `git status --short .gitignore`
Expected: 输出仅 ` M .gitignore`（前导空格 + M，表示工作区已改、暂存区未动）。**不**能出现 `M  .gitignore`（暂存区已改）。

- [ ] **Step 5: 不 commit，进入 Task 2**

本 task 不单独 commit。`.gitignore` 修改和 Task 2 的 `git rm --cached` 一起进入同一个 commit（spec §6 验收要求"一次 commit"）。

---

## Task 2: 从 index 移除 4 批运行时数据（git rm --cached）

**Files:**
- Modify (git index only, disk untouched):
  - `web/node_modules/` 整目录（递归）
  - `backend/nanobot/nanobot/workspace/` 整目录（递归）
  - `backend/nanobot/data/db/image_index.db`
  - `backend/nanobot/data/db/ingestion_history.db`

**Interfaces:**
- Consumes: Task 1 的 `.gitignore` 改动（工作区已改但未暂存）
- Produces: 暂存区会出现一批 `deleted: ...` 条目；磁盘文件不动。

- [ ] **Step 1: 确认 4 条目标路径当前确实被 git 追踪**

Run: `git ls-files --error-unmatch backend/nanobot/data/db/image_index.db backend/nanobot/data/db/ingestion_history.db`
Expected: 两行路径回显，exit 0。若任一报错 `did not match any file(s) known to git`，说明 spec 假设与现状不符，**停止**并回头确认。

Run: `git ls-files web/node_modules | head -3 && git ls-files backend/nanobot/nanobot/workspace | head -3`
Expected: 每条命令至少回显 1 行被追踪文件。若任一命令输出为空，**停止**并确认。

- [ ] **Step 2: 移除 web/node_modules（最大量，58K+ 文件）**

Run: `git rm --cached -r web/node_modules`
Expected: 标准输出大量 `rm 'web/node_modules/.../...'`。可能输出极长（5万+ 行）。exit 0。

- [ ] **Step 3: 移除 backend/nanobot/nanobot/workspace**

Run: `git rm --cached -r backend/nanobot/nanobot/workspace`
Expected: 标准输出 5 行 `rm 'backend/nanobot/nanobot/workspace/rag_data/chroma/...'`。exit 0。

- [ ] **Step 4: 移除两个 .db 文件**

Run: `git rm --cached backend/nanobot/data/db/image_index.db backend/nanobot/data/db/ingestion_history.db`
Expected: 输出两行 `rm '...'`。exit 0。

- [ ] **Step 5: 确认磁盘文件原样保留**

Run: `ls web/node_modules | head -3 && ls backend/nanobot/nanobot/workspace/rag_data/chroma | head -3 && ls -la backend/nanobot/data/db/`
Expected: 三条命令都能正常列出文件 —— `--cached` 只动 index，磁盘文件原样在。

- [ ] **Step 6: 把 .gitignore 改动加入暂存区**

Run: `git add .gitignore`
Expected: 静默成功。**注意**：只 add 这一个文件，绝对**不**用 `git add .` / `git add -A` / `git add backend/` 等任何泛匹配（spec §5 风险表第一条）。

---

## Task 3: 暂存区人工核对（commit 前唯一拦截点）

**Files:** 不动任何文件，纯检查。

**Interfaces:**
- Consumes: Task 2 完成后的暂存区状态。
- Produces: 用户/执行者书面确认"暂存区干净"，才能进入 Task 4。

**为什么这一步独立成 Task**：Phase 1 唯一真有可能出事的环节是 commit 误带工作区 60+ 未暂存源码修改。spec §5 风险表把它列为头号风险。绝不能默认假设、绝不能跟 commit 合并成一步。

- [ ] **Step 1: 全量 git status**

Run: `git status`
Expected 输出结构（**逐项核对，全对才能继续**）：

  - "Changes to be committed:" 区域：
    - 恰好 1 行 `modified:   .gitignore`
    - 一批 `deleted:    web/node_modules/...`（数量极大，5 万+）
    - 一批 `deleted:    backend/nanobot/nanobot/workspace/...`（5 条）
    - 2 行 `deleted:    backend/nanobot/data/db/{image_index,ingestion_history}.db`
    - **不**能出现任何非上述路径的 `modified:` / `new file:`
  - "Changes not staged for commit:" 区域：
    - 60+ 个工作区未暂存修改（backend 源码、web/node_modules 自动产物、chroma 运行时 touch 等）—— 这些**保持未暂存**就是对的，**不**要 add 它们。

- [ ] **Step 2: 用 grep 反向确认暂存区没有掺入源码**

Run: `git diff --cached --name-only | grep -vE '^\.gitignore$|^web/node_modules/|^backend/nanobot/nanobot/workspace/|^backend/nanobot/data/db/'`
Expected: 输出为空（exit code 可能是 1，正常）。**任何一行输出都意味着暂存区掺入了不该进 commit 的文件 —— 停止**，跑 `git reset HEAD <那一行路径>` 把它退出暂存区，再重新核对。

- [ ] **Step 3: 统计暂存区改动数量是否符合预期**

Run: `git diff --cached --name-only | wc -l`
Expected: 数量大致符合「1（.gitignore）+ ~58,609（node_modules）+ 5（workspace）+ 2（db）≈ 58,617」。允许小幅偏差（node_modules 实际文件数可能因 pnpm 版本略差），但**数量级**必须对。如果是 1、几十、几百，明显错了 —— 停止排查。

- [ ] **Step 4: 显式确认通过本核对**

到此为止全部 Expected 都满足，才进入 Task 4 commit。

如果执行者是 agent：把 Step 1-3 的实际输出（或关键摘要）回报给用户/审阅人，等明确"可以 commit"再继续。
如果执行者是人：自己肉眼复核完，再敲下一条命令。

---

## Task 4: Commit

**Files:** 不修改文件，只产生 commit object。

**Interfaces:**
- Consumes: Task 3 已通过的暂存区。
- Produces: 一个新 commit，HEAD 前进 1 步，未 push。

- [ ] **Step 1: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore(cleanup): Phase 1 — stop tracking runtime data

- .gitignore: add node_modules/, workspace/, data/db/, *.log, token.json
- git rm --cached: web/node_modules, workspace chroma, 2 stale .db files

Disk files untouched. Refs docs/superpowers/specs/2026-06-26-repo-cleanup-design.md Phase 1.
EOF
)"
```

Expected: 输出 `[main <hash>] chore(cleanup): Phase 1 ...`，紧跟一行 `N files changed, ...`，N ≈ 58,617。exit 0。

不要用 `--amend`、不要用 `--no-verify`（spec §5 / 默认硬规则）。

- [ ] **Step 2: 验证 commit 内容**

Run: `git log -1 --stat | head -20`
Expected: 看到刚刚的 commit message + `.gitignore | ... +/-` + 一片 `web/node_modules/... | bin ...` / `... | <n> --` 删除行。

Run: `git log -1 --name-only | grep -vE '^(commit|Author|Date|chore|$|    )' | grep -vE '^\.gitignore$|^web/node_modules/|^backend/nanobot/nanobot/workspace/|^backend/nanobot/data/db/'`
Expected: 输出为空 —— 再次反向核对 commit 只含预期 4 类路径 + .gitignore。

- [ ] **Step 3: 验证工作区那 60+ 个未暂存修改没被吃掉**

Run: `git status --short | wc -l`
Expected: 数量与 commit 前的"Changes not staged" 区域基本一致（应该 60+，含 backend 源码、node_modules 工作区改动、chroma 运行时改动等）。如果显著变少，说明 commit 误带了它们，**停止**排查。

Run: `git diff --stat | tail -5`
Expected: 仍能看到那 60+ 个文件的工作区 diff。

---

## Task 5: Push

**Files:** 不动文件，纯远端推送。

**Interfaces:**
- Consumes: Task 4 产生的新 commit。
- Produces: `origin/main` 前进 1 步。

- [ ] **Step 1: 确认 remote / branch 关系**

Run: `git status -sb | head -1`
Expected: `## main...origin/main [ahead 1]`。

- [ ] **Step 2: Push**

Run: `git push origin main`
Expected: 标准输出 `To <remote-url>` + `<old>..<new>  main -> main`。exit 0。

**不**用 `--force` / `-f` / `--force-with-lease`。普通 push（spec §5 风险表 + 用户硬约束）。

如果 push 失败（被 reject、网络错误等）：停止，不要重试 force 或重写历史，先把错误信息发回给用户排查。

- [ ] **Step 3: 验证已与 origin 对齐**

Run: `git status -sb | head -1`
Expected: `## main...origin/main`（不再有 `[ahead N]`）。

Run: `git log origin/main -1 --oneline`
Expected: 顶端就是刚刚那条 cleanup commit。

---

## Task 6: Push 后验证（用户硬约束）

**Files:** 不修改文件，纯验证。

**Interfaces:**
- Consumes: Task 5 已推送状态。
- Produces: 验证 Phase 1 达到 spec §6 验收标准。

- [ ] **Step 1: git status 不再出现目标路径**

Run: `git status | grep -E 'node_modules|workspace/rag_data|data/db/.*\.db'`
Expected: **输出为空**（grep exit code 1 正常）。如果还能看到这些路径，说明 .gitignore 规则没生效或还有别的追踪点 —— 停止排查。

注：`git status` 仍然会有 60+ 个未暂存修改（backend 源码等），这部分是用户在改的代码，**不**该消失。

- [ ] **Step 2: 后端可启动**

Run: `python -m nanobot --help`
Expected: 看到 CLI 帮助文本（任何 nanobot 子命令列表），exit 0。如果报 `ModuleNotFoundError: No module named 'nanobot'`，说明 Phase 1 之外的环境问题（与本次清理无关），但仍要回报给用户。

注：spec 已确认 `__main__.py` 入口仍在 `nanobot` 包名下。不跑实际服务，只验证模块可加载。

- [ ] **Step 3: 前端 dev 可启动（启动后立即停）**

Run（在 `web/` 目录下，后台跑）:
```bash
cd web && pnpm dev
```
Expected: 在 5-10 秒内看到 vite 的本地端口监听提示（如 `Local: http://localhost:5173/`），无 fatal error。看到端口提示后立刻 Ctrl-C 或杀进程。如果 pnpm 报 `ENOENT` / `Cannot find module`，说明本地 `node_modules` 被误删 —— 排查（但 `git rm --cached` 不动磁盘，理论上不应该发生）。

如果当前环境没装 pnpm 或前端依赖未装好，跳过此 Step 并明确告知用户"前端启动验证因环境问题未跑"。

- [ ] **Step 4: 上报 Phase 1 完成**

把以下事实回报给用户：
- commit hash & message 一行
- `git status` 中目标路径已消失（截图或粘贴 Step 1 输出）
- `python -m nanobot --help` 成功 / 失败
- `pnpm dev` 成功 / 失败 / 因环境跳过
- 工作区那 60+ 未暂存修改的数量是否与 Phase 1 开工前一致

到此 Phase 1 结束。Phase 2-6 是独立 spec，不在本 plan 范围。

---

## Self-Review Notes

- **Spec coverage**：spec §4 Phase 1 三个动作（.gitignore 追加 / git rm --cached / commit+push）已分别对应 Task 1 / Task 2 / Task 4+5；用户硬约束的人工核对独立成 Task 3、push 后验证独立成 Task 6；spec §5 风险表 6 条风险全部在 plan 内有对应防护（add .gitignore only / disk untouched / 重叠规则注释 / 工作区未暂存数量核对 / push 失败不 force / token.json 仅 .gitignore 防未来）；spec §6 验收 4 条对应 Task 6 Step 1-4。
- **Placeholder scan**：无 TBD/TODO/"appropriate error handling"/"similar to Task N" 等占位。所有 git 命令完整写出，所有期望输出明确。
- **Type/name consistency**：Phase 1 不写代码，无类型一致性问题。路径名前后一致（`backend/nanobot/nanobot/workspace`、`backend/nanobot/data/db/{image_index,ingestion_history}.db`、`web/node_modules`），与 spec §2.3、§4 Phase 1 完全对齐。
- **Out-of-scope guard**：plan 内无 Phase 2-6 动作（不删散落文件、不改源码、不改名、不动 `*.claude` 疑似 bug）。
