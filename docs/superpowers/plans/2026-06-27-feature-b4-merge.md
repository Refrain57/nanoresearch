# B4 feature → main 合并实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `feature/a1-phase1-b4-b2` 的 2 个 commit（`d019c3fd` .gitignore /scripts/ 锚定 + `52818413` B4 case_metadata schema）通过显式 `--no-ff` merge commit 合回 `main`，push 到 `origin/main`。不动磁盘工作区那 ~131 个未暂存改动、不动 `rescue/b4-orphan` 分支、不动任何 `worktree-agent-*` 分支、不动历史。

**Architecture:** 一次 `git merge --no-ff -m "..."`，在 main 上产生显式 merge commit；合并 d019c3fd + 52818413 共 10 个文件（.gitignore + 9 B4 文件），零冲突预期（诊断已验证 9 B4 文件 ∩ 工作区未暂存 = 0；.gitignore 自 Phase 1 后 main 没再动，3-way merge 直接采用 feature 那版无 conflict）。唯一真有可能 abort 的点是工作区 `.gitignore` 也有未暂存改动 → merge refuse；Task 0 Step 5 提前显式检测。

**Tech Stack:** git（CLI）。无其他依赖。

## Global Constraints

来自用户硬约束 + P1/P2 实战节奏：

- 本轮范围：**仅** merge feature → main + push。不动 `rescue/b4-orphan`、不动 `worktree-agent-*` 三个分支、不动 spec Phase 3 改名。
- 不重写历史、不 rebase、不 `--amend`、不 `--no-verify`、不 `--force` / `-f` / `--force-with-lease`。普通 merge + push。
- 用 `git merge --no-ff -m "<heredoc>"` 显式合并，绝不让 git 调起编辑器。
- 不破坏工作区那 ~131 个未暂存源码改动。
- Task 0 preflight 必须显式检查 `.gitignore` 是否在工作区未暂存改动里——它在的话 merge 会 abort，需先停下来跟用户对齐如何处理，**不**许 agent 自动 stash / commit / reset。
- merge 后、push 前必须独立 task 做人工核对闸口（对齐 P1/P2 Task 3）。
- 任何 conflict、refuse、unexpected 状态：**停**，原样报错给用户，不重试，不切策略。
- 不动 `rescue/b4-orphan`——保留其 HEAD `cbfa94b3`（含 5 块未来 Phase 预埋改动）作为预埋仓库，后续 Phase 1/2/5 单独提取。
- Task 5 收尾：更新 `MEMORY.md`，记两条事实：(a) rescue/b4-orphan 含 5 块预埋未合入；(b) feature/a1-phase1-b4-b2 已合并。

---

## File Structure

merge 引入的所有路径（plan 自己不创建任何工程文件）：

**d019c3fd 改：**
- `.gitignore`（`scripts/` → `/scripts/`，1 行修改 + 注释）

**52818413 改/创建：**
- `backend/migrations/README.md`
- `backend/migrations/add_case_metadata.sql`
- `backend/migrations/add_case_metadata_down.sql`
- `backend/migrations/add_case_metadata_enforce.sql`
- `backend/migrations/add_case_metadata_enforce_down.sql`
- `backend/nanobot/storage/models.py`
- `backend/scripts/backfill_case_metadata.py`
- `backend/tests/storage/__init__.py`
- `backend/tests/storage/test_case_metadata_migration.py`

merge commit 自身不引入新文件。

plan 自己唯一写的文件：Task 5 的 2 个 memory 文件 + `MEMORY.md` 索引修改。

不写测试、不改源码。

---

## Task 0: Preflight 漂移检查 + baseline 抓取

**Files:** 不动任何文件，纯只读 + 写一个 baseline 临时文件。

**Interfaces:**
- Consumes: 上一轮诊断结论（main @ `676ef8c7`、feature @ `52818413`、rescue @ `cbfa94b3`、9 B4 文件 ∩ 工作区 = 0）。
- Produces: 当前状态与上一轮诊断一致的书面确认 + 工作区 baseline。任一项漂移就停。

**为什么独立成 Task**：用户工作区 ~131 个未暂存改动持续在变。从诊断到执行有时间差，期间 (a) main / feature HEAD 可能动了、(b) 用户可能改了 `.gitignore` 或某个 9 B4 文件，任一都会让 merge 行为偏离 plan 预期。这是合并前最后的状态快照 + 唯一的 abort-or-continue 决策点。

- [ ] **Step 1: 当前分支与 HEAD 对齐确认**

Run:
```bash
git status -sb | head -1
git log -1 --oneline main
git log -1 --oneline feature/a1-phase1-b4-b2
git log -1 --oneline rescue/b4-orphan
```

Expected:
- 第一行 `## main...origin/main`（无 ahead/behind）
- main HEAD = `676ef8c7 chore(cleanup): Phase 2 — remove scattered orphan files`
- feature HEAD = `52818413 feat(B4): add case metadata schema with backfill and down-migration convention`
- rescue HEAD = `cbfa94b3 feat(B4): add case metadata schema with backfill and down-migration convention`

任一不符：**停**，报回用户，不进 Step 2。

- [ ] **Step 2: 共同基底确认**

Run:
```bash
git merge-base main feature/a1-phase1-b4-b2
echo "---expected: d9d420a8...---"
git log -1 --oneline d9d420a8
```

Expected:
- merge-base 输出以 `d9d420a8` 开头的 hash。
- `d9d420a8 chore(cleanup): Phase 1 — stop tracking runtime data`

不符：**停**，可能有人把 main / feature reset 过。

- [ ] **Step 3: feature..main 差集复验（仍是 2 commit）**

Run:
```bash
git log --oneline main..feature/a1-phase1-b4-b2
echo "---count: $(git log --oneline main..feature/a1-phase1-b4-b2 | wc -l)---"
```

Expected：
```
52818413 feat(B4): add case metadata schema with backfill and down-migration convention
d019c3fd build: anchor /scripts/ ignore to repo root so backend/scripts/ tracks normally
```
count = 2。

不符：**停**。

- [ ] **Step 4: 9 B4 文件不在工作区未暂存改动里（关键防冲突）**

Run:
```bash
echo "=== 9 B4 文件 ∩ 工作区改动 ==="
comm -12 <(git show --name-only --format= 52818413 | grep -v '^$' | sort -u) <(git status --porcelain | sed 's/^...//' | sort -u)
echo "---intersection count: $(comm -12 <(git show --name-only --format= 52818413 | grep -v '^$' | sort -u) <(git status --porcelain | sed 's/^...//' | sort -u) | wc -l)---"
```

Expected：输出空，count = 0。

count ≥ 1：**停**。意味着用户在诊断后又改了某个 B4 文件，merge 会 abort 或被工作区改动污染——必须先跟用户对齐（让其手动 stash / commit / 撤销那个改动，**不**许 agent 自动决定）。

- [ ] **Step 5: .gitignore 状态（d019c3fd 要改它）**

Run:
```bash
echo "=== .gitignore 工作区状态 ==="
git status --porcelain .gitignore
echo "---above should be empty---"
echo
echo "=== d019c3fd 对 .gitignore 的改动预览（诊断目的） ==="
git show d019c3fd -- .gitignore
```

Expected：
- 第一段**为空**（`.gitignore` 工作区干净）。
- 第二段显示 d019c3fd 把 `scripts/` 改成 `/scripts/`。

第一段有输出（比如 ` M .gitignore`）：**停**。merge 会 refuse-overwrite。需让用户决定是 commit、stash、还是放弃那一行改动——**不**许 agent 自动 stash / reset / commit。

注：诊断已确认 Phase 2 commit (676ef8c7) 没动 `.gitignore`，共同基底 d9d420a8 之后 main 一直没动 `.gitignore`，所以 3-way merge 直接拿 d019c3fd 那版无冲突（前提是工作区干净）。

- [ ] **Step 6: rescue/b4-orphan 完整性（确认它还在，但不动）**

Run:
```bash
git log -1 --oneline rescue/b4-orphan
echo "---"
git show --stat cbfa94b3 | head -3
```

Expected：rescue HEAD 仍是 `cbfa94b3`，标题 `feat(B4): add case metadata schema...`。本 plan 全程**不动**该分支。

- [ ] **Step 7: 抓工作区 baseline**

Run:
```bash
M_BASE=$(git status --porcelain | grep -c '^ M ')
D_BASE=$(git status --porcelain | grep -c '^ D ')
QQ_BASE=$(git status --porcelain | grep -c '^?? ')
TOTAL_BASE=$(git status --porcelain | wc -l)
{
  echo "M_BASE=$M_BASE"
  echo "D_BASE=$D_BASE"
  echo "QQ_BASE=$QQ_BASE"
  echo "TOTAL_BASE=$TOTAL_BASE"
} | tee /tmp/b4-merge-baseline.txt
```

Expected：4 行落地 `/tmp/b4-merge-baseline.txt`。`TOTAL_BASE` 大约 131（诊断时数），允许 ±10 漂移。

Task 2 / Task 4 用此基线对比。merge 完后预期：`M_NOW == M_BASE`、`D_NOW == D_BASE`、`TOTAL_NOW == TOTAL_BASE`。

- [ ] **Step 8: 显式确认通过本闸**

Step 1-7 全部 Expected 满足才进 Task 1。

agent：把 Step 1-7 实际输出回报给用户，等"可以 merge"再继续。
人：肉眼复核完再敲下一条命令。

---

## Task 1: 执行 merge --no-ff

**Files:**
- Modify: HEAD（main 上产生新 merge commit）
- Files brought in by merge: 10 个（`.gitignore` + 9 B4）

**Interfaces:**
- Consumes: Task 0 已确认无漂移、无冲突隐患。
- Produces: main 前进 1 个 merge commit，HEAD 包含 d019c3fd + 52818413 的完整变更，未 push。

- [ ] **Step 1: 执行 merge**

Run:
```bash
git merge --no-ff feature/a1-phase1-b4-b2 -m "$(cat <<'EOF'
merge(B4): pull feature/a1-phase1-b4-b2 into main

Brings 2 commits:
- d019c3fd  build: anchor /scripts/ ignore to repo root
- 52818413  feat(B4): add case_metadata schema with backfill and down-migration convention

Pre-Phase-3 merge to avoid models.py conflicts when the upcoming
nanobot → nanoresearch rename relocates backend/nanobot/storage/models.py.

rescue/b4-orphan branch retained unmerged; it carries 5 future-Phase
preliminaries (KbDocument.content_hash, AgentRunSnapshot classification_*,
AgentTestCase set_kind/tool_recordings, OptimizationProposal baseline_*,
TunableObjectVersion table) that will be extracted into their own commits
when their respective Phases land.
EOF
)"
echo "exit=$?"
```

Expected：
- 输出含 `Merge made by the 'ort' strategy.`（或类似 strategy 名）。
- 输出含 10 文件改动统计（`.gitignore | ... +/-` + 9 B4 文件 `... | N ++++` 等）。
- exit 0。

如果 git 报 `CONFLICT` 或 `error: Your local changes to ...`：**停**。原样报错给用户，**不**自动 abort / reset / stash。

如果 git 调起编辑器（本步骤已用 `-m` 应不会）：**停**，排查 git 配置后再决定。

- [ ] **Step 2: 验证 HEAD 是 merge commit、含两个 parent**

Run:
```bash
git log -1 --format='%H %P %s'
echo "---"
git log --oneline -3
```

Expected：
- 第一行格式 `<merge-hash> <parent1> <parent2> merge(B4): pull ...`，parent 列**有两个 hash**：parent1 = `676ef8c7...`（Phase 2），parent2 = `52818413...`。
- `git log --oneline -3` 顶端是新 merge commit。

只有 1 个 parent（说明 ff 发生，本不该）：**停**。

- [ ] **Step 3: 验证 merge 引入的两个新 commit 在 HEAD 的历史里**

Run:
```bash
echo "=== HEAD 现在包含的两个新进来的 commit（相对 Phase 2） ==="
git log --oneline 676ef8c7..HEAD
echo "---above should show 3 lines: merge + 52818413 + d019c3fd---"
```

Expected：3 行 —— merge commit + `52818413` + `d019c3fd`。

不是 3 行：**停**。

- [ ] **Step 4: 验证 10 个 merge-in 文件落地**

Run:
```bash
ls -la .gitignore backend/migrations/README.md backend/migrations/add_case_metadata.sql backend/migrations/add_case_metadata_down.sql backend/migrations/add_case_metadata_enforce.sql backend/migrations/add_case_metadata_enforce_down.sql backend/nanobot/storage/models.py backend/scripts/backfill_case_metadata.py backend/tests/storage/__init__.py backend/tests/storage/test_case_metadata_migration.py 2>&1
echo "---"
echo "=== models.py 含 B4 字段 ==="
grep -n "origin_badcase_id\|target_dimension\|coverage_tags" backend/nanobot/storage/models.py
echo "---should see 3+ lines---"
echo
echo "=== .gitignore 含 /scripts/ 锚定 ==="
grep -n "^/scripts/\|^scripts/" .gitignore
echo "---should see /scripts/ (anchored), NOT bare scripts/---"
```

Expected：
- 10 个文件全部存在。
- models.py 至少 3 处命中（三个字段名）。
- .gitignore 含一行 `/scripts/`，**不**应有裸 `scripts/`。

任一不符：**停**。

---

## Task 2: 工作区核对闸口（push 前唯一拦截点）

**Files:** 不动任何文件，纯只读。

**Interfaces:**
- Consumes: Task 1 完成后状态。
- Produces: 用户书面确认"工作区无误"才能进 Task 3 push。

**为什么独立成 Task**：与 P1/P2 Task 3 同款强制闸口。push 是远端动作，必须最后再核一道。

- [ ] **Step 1: 工作区状态对齐 baseline**

Run:
```bash
source /tmp/b4-merge-baseline.txt
M_NOW=$(git status --porcelain | grep -c '^ M ')
D_NOW=$(git status --porcelain | grep -c '^ D ')
QQ_NOW=$(git status --porcelain | grep -c '^?? ')
TOTAL_NOW=$(git status --porcelain | wc -l)
echo "M     : base=$M_BASE  now=$M_NOW  (expect equal)"
echo "D     : base=$D_BASE  now=$D_NOW  (expect equal)"
echo "??    : base=$QQ_BASE now=$QQ_NOW (expect equal)"
echo "TOTAL : base=$TOTAL_BASE now=$TOTAL_NOW (expect equal)"
```

Expected：4 行，now 与 base 全部相等（merge 不动工作区任何文件）。

任一不等：**停**。可能 merge 被工作区污染或漂移过大。

- [ ] **Step 2: 暂存区应为空**

Run:
```bash
git diff --cached --name-only
echo "---staged count: $(git diff --cached --name-only | wc -l)---"
```

Expected：输出空，count = 0。merge --no-ff 完成后暂存区不应残留任何文件。

非 0：**停**，跑 `git diff --cached` 看是什么，再决定。

- [ ] **Step 3: 远端关系应 ahead 1**

Run:
```bash
git status -sb | head -1
```

Expected：`## main...origin/main [ahead 1]`（merge commit 是新增的 1 commit）。

不含 `[ahead 1]`：**停**（如果是 ahead 2/3 等，远端比预期少；如果已 aligned，merge commit 没产生）。

- [ ] **Step 4: 显式确认通过本闸**

Step 1-3 全部 Expected 满足才进 Task 3。

agent：贴 Step 1-3 输出给用户，等"可以 push"才往下。
人：肉眼复核完再敲。

---

## Task 3: Push

**Files:** 不动文件，纯远端推送。

**Interfaces:**
- Consumes: Task 2 已通过的本地状态。
- Produces: `origin/main` 前进 1 步（merge commit）。

- [ ] **Step 1: Push**

Run:
```bash
git push origin main
echo "exit=$?"
```

Expected：输出 `<old>..<new>  main -> main`，exit 0。`<old>` 应是 `676ef8c7...`，`<new>` 是 Task 1 产生的 merge commit hash。

**不**用 `--force` / `-f` / `--force-with-lease`。

push 失败（认证、reject、网络等）：**停**，把原始错误发回用户，不重试 force / 切策略 / 重写历史（P2 push 也失败过一次是 SSH key 问题，用户手动 push 即可——本 Task 同样按此节奏）。

- [ ] **Step 2: 验证已对齐**

Run:
```bash
git status -sb | head -1
git log origin/main -1 --oneline
```

Expected：
- `## main...origin/main`（无 ahead）。
- `origin/main` 顶端是 Task 1 的 merge commit。

如果 fetch 失败但 push 是用户手动跑的（P2 同款 SSH 情境）：缓存的 `origin/main` ref 已被 push 操作更新，仍可信。

---

## Task 4: Push 后验证

**Files:** 不修改文件，纯只读。

**Interfaces:**
- Consumes: Task 3 已推送状态。
- Produces: B4 已在 origin/main 上的最终书面确认。

- [ ] **Step 1: origin/main 与本地 HEAD 一致**

Run:
```bash
git rev-parse HEAD origin/main
echo "---above two should be identical---"
```

Expected：两行 hash 完全相同。

- [ ] **Step 2: 关键内容落地**

Run:
```bash
echo "=== models.py B4 字段 ==="
grep -n "origin_badcase_id\|target_dimension\|coverage_tags\|added_at\|added_by" backend/nanobot/storage/models.py
echo "---should see 5 lines---"
echo
echo "=== backfill 脚本 I2 安全闸 ==="
grep -n "I2: safety guard\|is_nullable" backend/scripts/backfill_case_metadata.py
echo "---should see 2+ lines (注释 + SQL)---"
echo
echo "=== .gitignore /scripts/ 锚定 ==="
grep -n "^/scripts/" .gitignore
echo "---should see 1 line---"
```

Expected：
- models.py 5 行命中。
- backfill 脚本含 I2 安全闸 2+ 行命中。
- `.gitignore` 含 1 行 `/scripts/`。

任一缺失：**停**，可能 merge 引入不完整。

- [ ] **Step 3: 工作区 ~131 个未暂存改动仍在**

Run:
```bash
source /tmp/b4-merge-baseline.txt
TOTAL_NOW=$(git status --porcelain | wc -l)
echo "TOTAL : base=$TOTAL_BASE  now=$TOTAL_NOW"
echo "---should be equal---"
git diff --stat | tail -3
```

Expected：`TOTAL_NOW == TOTAL_BASE`；`git diff --stat` 仍显示工作区源码改动汇总。

- [ ] **Step 4: rescue/b4-orphan 未动确认**

Run:
```bash
git log -1 --oneline rescue/b4-orphan
echo "---should still be: cbfa94b3 feat(B4): ...---"
```

Expected：rescue HEAD 仍为 `cbfa94b3`。

不等：**停**，意味着 rescue 被无意中动了。

---

## Task 5: 更新 memory + 收尾报告

**Files:**
- Create: `C:\Users\Augix\.claude\projects\D--Code-nanobot\memory\project_rescue_b4_orphan_dormant.md`
- Modify: `C:\Users\Augix\.claude\projects\D--Code-nanobot\memory\project_feature_branch_unmerged.md`（旧的"未合并"内容改写成"已合并"）
- Modify: `C:\Users\Augix\.claude\projects\D--Code-nanobot\memory\MEMORY.md`（替换 feature 那行 + 追加 rescue 那行）

**Interfaces:**
- Consumes: Task 4 已验证 B4 在 origin/main 上。
- Produces: 持久化两条事实供未来会话使用 + 收尾报告。

- [ ] **Step 1: 用 Write tool 创建 `project_rescue_b4_orphan_dormant.md`**

文件内容（逐字写入，frontmatter + body）：

```markdown
---
name: rescue/b4-orphan branch dormant content
description: rescue/b4-orphan @ cbfa94b3 含 5 块未合入 main 的未来 Phase 预埋改动，留待对应 Phase 单独提取
type: project
---

`rescue/b4-orphan` HEAD `cbfa94b3` 与已合并的 `52818413`（feature/a1-phase1-b4-b2 的 B4）同标题不同 hash，B4 本体字节级相同，但 rescue 那版的 `models.py` 多塞了 5 块 commit message 未提的未来 Phase 改动：

1. `KbDocument.content_hash` 列
2. `AgentRunSnapshot` 加 `classification_layer` / `classification_target_kind` / `classification_target_id` 3 字段（"Phase 1: structured root-cause pointer"）
3. `AgentTestCase` 加 `set_kind` + `tool_recordings` 2 字段（"Phase 2: regression set separation" + sandbox replay）
4. `OptimizationProposal` status 加 `gate_all_rejected` + 加 `baseline_score` / `baseline_version_id` 2 字段（"Phase 5: baseline anchor + deployment gate"）
5. 新增类 `TunableObjectVersion(Base)`（immutable version history for system_prompt / tool_description，带 apply/rollback）

外加 `test_case_metadata_migration.py` 一段 `if False else \` 死代码、`add_case_metadata.sql` 末行一句与事实不符的注释。

**Why:** 选 feature（纯 B4 + I2 backfill 安全闸）合入 main，rescue 的 5 块未来工作不当作废弃——它们是被误塞进 B4 的预埋，以后做对应 Phase 时单独走干净 commit。

**How to apply:** 启动 Phase 1 root-cause 改造 / Phase 2 regression set / Phase 5 baseline gate / TunableObjectVersion 任一时，先 `git show cbfa94b3 -- <相关文件>` 翻出对应那块预埋作为起点参考（不是直接 cherry-pick——cbfa94b3 与那时的 main 已经偏离 Phase 3 改名等改动）。`rescue/b4-orphan` 分支保留不删，直到 5 块全部各自落地。
```

- [ ] **Step 2: 用 Read + Write 改写 `project_feature_branch_unmerged.md`**

先 Read 该文件（确认旧内容、走 Write 前置约束），然后 Write 覆盖成：

```markdown
---
name: feature/a1-phase1-b4-b2 merged
description: feature/a1-phase1-b4-b2 (B4 case_metadata + .gitignore /scripts/) 已通过 --no-ff merge 合入 main，2026-06-27
type: project
---

`feature/a1-phase1-b4-b2` 的 2 commit（`d019c3fd` .gitignore + `52818413` B4）已通过 `git merge --no-ff` 合入 `main` 并 push 到 `origin/main`，Phase 3 改名前置债已清。

**Why:** Phase 3 改名（`nanobot → nanoresearch`）会移动 `backend/nanobot/storage/models.py`，而 B4 commit 改过此文件；先合再改名可避免改名后再合的大量冲突。

**How to apply:** Phase 3 改名 plan 可直接基于当前 main 起手，无需再考虑 feature 分支。`feature/a1-phase1-b4-b2` 本地分支可由用户自行决定删或留（已并入 main，删它不丢代码）。`rescue/b4-orphan` 另议——见 `project_rescue_b4_orphan_dormant.md`。
```

- [ ] **Step 3: 用 Edit tool 改 `MEMORY.md` 索引**

替换 `project_feature_branch_unmerged.md` 的旧索引行 + 追加 `project_rescue_b4_orphan_dormant.md`。

旧行（precise `old_string`）：
```
- [feature/a1-phase1-b4-b2 has 2 unmerged commits](project_feature_branch_unmerged.md) — B4 case_metadata + .gitignore /scripts/，Phase 3 起手前必须先定去向
```

新行（`new_string`，两行）：
```
- [feature/a1-phase1-b4-b2 merged](project_feature_branch_unmerged.md) — 2 commit 已 --no-ff merge 入 main 2026-06-27；Phase 3 改名前置债已清
- [rescue/b4-orphan dormant](project_rescue_b4_orphan_dormant.md) — 含 5 块未合入 main 的未来 Phase 预埋（content_hash / classification / set_kind+tool_recordings / baseline / TunableObjectVersion），各自 Phase 落地时再提取
```

- [ ] **Step 4: 上报收尾**

把以下事实贴给用户：
- merge commit hash + 一行 message
- `git log -3 --oneline` 输出
- Task 4 Step 2 三段 grep 结果摘要
- 工作区 TOTAL 数（应 == baseline）
- rescue/b4-orphan HEAD 未动确认（仍 cbfa94b3）
- memory 已更新两文件 + `MEMORY.md` 索引

收尾选项（不替用户决定，只列）：
- `feature/a1-phase1-b4-b2` 本地分支：可保留（无害）或 `git branch -d feature/a1-phase1-b4-b2`（已并入 main，git 会用 ff-check 阻止真有 unmerged 内容时的删除）。
- `rescue/b4-orphan` 本地分支：**保留**（memory 已记录其 dormant 内容）。
- `worktree-agent-*` 三个分支：本轮不动（P2 也只记一笔）。
- 下一步：写 Phase 3 改名 plan（独立 plan 文件，本轮不写）。

到此本 plan 结束。

---

## Self-Review Notes

- **Spec coverage**：本 plan 是 spec 之外的前置债清理（spec 当初没预料 feature 分支问题）。Plan 覆盖 (a) merge feature → main、(b) 不动 rescue/b4-orphan、(c) 不破工作区、(d) memory 收尾 4 件用户硬约束。Phase 3 改名留独立 plan ✓。
- **Placeholder scan**：无 TBD / TODO / "appropriate error handling" / "similar to Task N"。所有 git 命令完整写出，所有期望输出明确。memory 文件内容逐字写出。
- **Risk coverage**：
  - "merge 被工作区 .gitignore 改动 refuse" → Task 0 Step 5 提前检测 ✓
  - "merge 撞工作区 9 B4 文件" → Task 0 Step 4 提前检测 ✓
  - "ff 偷偷发生丢了 merge commit" → Task 1 用 --no-ff + Step 2 parent 数核对 ✓
  - "push 失败 retry force" → Task 3 Step 1 明文禁止 ✓
  - "rescue/b4-orphan 被无意中动" → Task 0 Step 6 + Task 4 Step 4 双确认 ✓
  - "git 调起编辑器" → Task 1 Step 1 用 -m heredoc ✓
  - "main / feature HEAD 漂移" → Task 0 Step 1-3 三段对齐核对 ✓
- **Out-of-scope guard**：plan 内无 Phase 3 改名动作，无 worktree-* 分支处理，无 5 块预埋提取——全部留作后续 ✓。
