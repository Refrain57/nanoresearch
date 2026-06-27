# 仓库清理 Phase 2 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删 spec §2.4 列出的 21 个无引用根目录散落文件（3 tracked 走 `git rm`，18 untracked 走 `mv` 兜底）+ 字面意义的根 `~/` 死数据目录（`mv` 兜底）。所有不可逆删除前都有兜底副本和强制人工核对。

注：原 spec 列 25 项，Task 0 Step 3-4 引用复验时发现 4 项被 `scripts/` 下脚本作为默认参数引用（`eval_data_real.json` + 3 个 `health_set_draft*.yaml`），从删除清单移除，留给后续 Phase 跟 `scripts/` 整体处理。最终清单 21（见下文 "Phase 2 不做、但留痕"）。

**Architecture:** 分两类执行：(1) tracked 文件用 `git rm` 一并消磁盘+消 index，产生 commit 内容；(2) untracked 文件和 `~/` 用 `mv` 移到 `/tmp/phase2-backup-<timestamp>/`，仅动磁盘，不进 commit（因为 git 一开始就不知道它们存在）。Commit 只含 3 个 tracked 删除条目，干净最小。验证全通过 + 用户点头后才清备份。

**Tech Stack:** git（CLI）、bash（mv / rm / ls）。无其他依赖。

## Global Constraints

来自 spec `docs/superpowers/specs/2026-06-26-repo-cleanup-design.md` Phase 2 + 用户硬约束：

- 本 Phase 范围：spec §2.4 列出的根目录散落文件 + 字面意义 `~/` 目录。**不**碰 Phase 3-6 任何内容（不改名、不改源码、不动 `backend/bridge/`、不动 `backend/data/db/*.db` 磁盘残留、不动 chroma 目录磁盘残留、不动 `loadtest.py:16` 的硬编码 JWT）。
- **删除策略**：3 个 tracked 文件走 `git rm`（git 历史可恢复）；18 个 untracked 文件走 `mv` 兜底到 `/tmp/phase2-backup-<timestamp>/`；根 `~/` 同样 `mv` 兜底。**绝不**对 untracked 文件或 `~/` 直接 `rm -rf`。
- **resume 两份特殊**：`resume_part1_agent.txt`、`resume_star.txt` 是用户个人简历，Task 2 单独最后一组 mv 并显式确认。
- **`~/` mv 命令必须用户亲眼确认**：执行者必须把完整命令打出来等用户明确说"可以"，**不**许 agent 自动跑。命令必须用引号 `mv "~" ...` 防 shell 把 `~` 展开成真 home。
- **commit 策略**：commit 前必须人工核对暂存区只含 3 个 `git rm` 的 `D` 条目，不含工作区任何源码改动。`git add` 不许用（`git rm` 自动入暂存区，无需 add）。
- **commit / push**：普通 commit + push，不 `--amend`、不 `--no-verify`、不 `--force` / `-f` / `--force-with-lease`、不重写历史。push 失败就停下，不重试 force / reset。
- **清备份的前提**：用户在 Task 7 明确说"可以清"才 `rm -rf` 备份目录。否则备份永久保留。

---

## Phase 2 不做、但留痕（在 plan 里留一笔，移交后续 Phase）

- `loadtest.py:16` 硬编码 JWT (`exp: 1782227115` ≈ 2026-06-28，即将过期) —— 与 `token.json` 同类凭证泄露问题。**留给 Phase 5（多租户 LLM API 配置）或 Phase 6 规范化整改**。Phase 2 不改 loadtest.py。
- `backend/bridge/` 孤立目录 —— `_get_bridge_dir()` 查的是 `backend/nanobot/bridge/`，**不是** `backend/bridge/`，无任何代码引用这个孤立目录。**留给 Phase 3 改名时一起判**（是命名差一档、还是改名半截留下的残骸）。Phase 2 不删它。
- `backend/nanobot/data/db/{image_index,ingestion_history}.db` 磁盘残留 —— Phase 1 已 `git rm --cached`，但磁盘文件仍在（spec §2.3 说本地版本 5/11 后冻结、home 版本当前在写）。**留给 Phase 6**（规范性整改 / 老迁移产物清理）。Phase 2 不删。
- `backend/nanobot/nanobot/workspace/rag_data/chroma/*` 磁盘残留 —— 同上。Phase 1 已 `git rm --cached`，磁盘残留留给 Phase 6 或 Phase 3 改名时一起判。
- spec §2.4 写"19 个无引用可删"，实测原本是 25，Task 0 Step 3-4 引用复验剔除 4 项后最终 21。本 plan 按 21 走。
- **`eval_data_real.json`** —— Task 0 Step 4 重扫发现被 `scripts/collect_ragloop_data.py:490`、`scripts/collect_rag_data.py:401`、`scripts/evaluate_agentic_rag_ragas.py:257-258`、`scripts/evaluate_rag_ragas_v2.py:545` 引用为默认 `--samples` / `default_path`。**留给后续 Phase 跟 `scripts/` 整体处理**（`scripts/` 当前被 `.gitignore:58` 排除，本身就是开发期本地脚本，需统一定盘）。Phase 2 不删。
- **`health_set_draft.yaml` / `_v2.yaml` / `_v3.yaml`** —— 同上 Step 补扫（多扩展名）发现被 `scripts/seed_health_set.py:4,636` 作为 `--out` 默认值引用；`docs/sdd/PHASE_STATUS.md:326,404` 把 `health_set_draft.yaml` 定位为 SDD 流程"待人工审核"的活跃草稿。`_v2`/`_v3` 看名是手动迭代版本，最新版可能是用户在用的当前草稿。**全 3 个留给后续 Phase 跟 `scripts/` 整体处理**。Phase 2 不删。

---

## File Structure

本 Phase 涉及的所有路径（全部相对于 repo root `D:\Code\nanobot\`）：

**3 个 tracked 文件（Task 1 删 — git rm，进 commit）**：

- `core_agent_lines.sh`
- `extracted_chunks.json`
- `todolist.md`

**18 个 untracked 文件（Task 2 删 — mv 兜底，不进 commit）**：

一次性脚本（5）：
- `_clean_all.py`
- `_fix_session_history.py`
- `_print_badcases.py`
- `_update_testcases.py`
- `extract_chunks.py`

eval dump（9 — 原 10，剔除 `eval_data_real.json` 留给 scripts/ 整体处理）：
- `badcase_flows.json`
- `snapshots_tmp.json`
- `tc_list.json`
- `eval_output.txt`
- `eval_qwen_max.log`
- `eval_results_qwen_max.json`
- `eval_results_real.json`
- `test_intermediate_output.txt`
- `test_retrieval_output.txt`

失效产物（2）：
- `server.log`
- `token.json`

个人（2 — Task 2 最后一组，单独确认）：
- `resume_part1_agent.txt`
- `resume_star.txt`

（**剔除项 4 个，留给后续 Phase**：`eval_data_real.json`、`health_set_draft.yaml`、`health_set_draft_v2.yaml`、`health_set_draft_v3.yaml`，理由见上"Phase 2 不做、但留痕"。）

**1 个 untracked 目录（Task 3 删 — mv 兜底，须用户亲眼确认）**：

- `~/`（字面意义，包含 `~/.nanobot/rag/{chroma,images}`，99 文件 / 15MB，mtime May 11，已被 `.gitignore` 第 41 行覆盖）

**备份目录（Task 2/3 创建，Task 7 用户许可后清）**：

- `/tmp/phase2-backup-<YYYYMMDD-HHMM>/`

不创建测试文件、不改任何源码。

---

## Task 0: 只读复核门（防漂移）

**Files:** 不动任何文件，纯只读。

**Interfaces:**
- Consumes: 上一次核查结果（21 文件分类、引用检查 0 风险；4 项剔除已在 plan 顶层留痕段说明）。
- Produces: 当前状态与上次核查一致的书面确认。任一项漂移就停。

**为什么这一步独立成 Task**：用户工作区有 80+ 未暂存修改且持续在改。从核查到执行有时间差，期间可能新增了对待删文件的引用、或文件本身被改动/重命名。这是删除前最后的状态快照。

- [ ] **Step 1: 21 个目标文件的存在性 + 追踪状态**

Run:
```bash
cd /d/Code/nanobot
files=(
  "_clean_all.py" "_fix_session_history.py" "_print_badcases.py" "_update_testcases.py"
  "extract_chunks.py" "core_agent_lines.sh"
  "badcase_flows.json" "snapshots_tmp.json" "tc_list.json" "extracted_chunks.json"
  "eval_output.txt" "eval_qwen_max.log" "eval_results_qwen_max.json"
  "eval_results_real.json" "test_intermediate_output.txt" "test_retrieval_output.txt"
  "resume_part1_agent.txt" "resume_star.txt"
  "server.log" "token.json" "todolist.md"
)
echo "=== existence + tracking ==="
tracked_count=0
untracked_count=0
missing_count=0
for f in "${files[@]}"; do
  if [ ! -e "$f" ]; then
    echo "MISSING: $f"
    missing_count=$((missing_count+1))
    continue
  fi
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    tracked_count=$((tracked_count+1))
    echo "TRACKED:   $f"
  else
    untracked_count=$((untracked_count+1))
  fi
done
echo "---"
echo "tracked=$tracked_count   untracked=$untracked_count   missing=$missing_count   total=${#files[@]}"
```
Expected:
- `tracked=3`、`untracked=18`、`missing=0`、`total=21`
- 3 个 TRACKED 行恰好是：`core_agent_lines.sh`、`extracted_chunks.json`、`todolist.md`

如果任何数字偏离（哪怕 1 个 missing 或 tracked 变了 4 个）：**停**。停下报回用户，等明确指示再继续 —— 漂移可能是用户在你不知情时改了文件。

- [ ] **Step 2: 根 `~/` 目录存在性 + 追踪状态**

Run:
```bash
ls -d "~/" 2>&1
echo "tracked count: $(git ls-files "~" 2>/dev/null | wc -l)"
echo "file count: $(find "~/" -type f 2>/dev/null | wc -l)"
du -sh "~/" 2>&1
```
Expected:
- `~/` 存在
- tracked count = 0
- file count 应在 99 上下（允许 ±10 漂移，运行时可能产生小量临时文件）
- size 约 15M

任何偏离（特别是 tracked count > 0）：**停**。

- [ ] **Step 3: 引用复验 —— 6 个一次性脚本**

Run:
```bash
echo "=== _update_testcases / _fix_session_history ==="
grep -rn "_update_testcases\|_fix_session_history" --include="*.py" 2>/dev/null
echo "---"
echo "=== _clean_all / _print_badcases / extract_chunks / core_agent_lines ==="
grep -rn "_clean_all\|_print_badcases\|extract_chunks\|core_agent_lines" --include="*.py" 2>/dev/null
```
Expected:
- 第一组 0 命中
- 第二组：可能命中 `extract_chunks.py` 自己（line 133 `output_file = ... / "extracted_chunks.json"`），其余 0 命中。`extract_chunks.py` 自引不算外部依赖（它是删除清单的成员，自己写自己的输出）。

任何新出现的活跃代码引用（特别是 `seed_testcases.py` / `eval_router.py` / `agent_eval_router.py` / `cli/` 下任何文件命中）：**停**。

- [ ] **Step 4: 引用复验 —— 11 个 eval dump 文件名**

本 Step 包含两轮 grep：(a) 原 spec §2.4 的 `.py` 单扩展名扫描；(b) **补强扫描** —— 多扩展名（`.py/.yaml/.yml/.json/.sh/.toml/.ini/.md/.txt`）覆盖 fixture / argparse default / config 等所有可能引用形式。补强扫描是为防漏扫（实测过去就漏掉了 `scripts/seed_health_set.py` 把 yaml 当 `--out` 默认值，以及 `scripts/collect_*.py` 把 json 当 `--samples` 默认值，那些命中导致 `eval_data_real.json` + 3 个 `health_set_draft*.yaml` 都从清单剔除留给后续 Phase）。

Run（轮 a：单扩展名）:
```bash
grep -rn "badcase_flows\.json\|snapshots_tmp\.json\|tc_list\.json\|extracted_chunks\.json\|eval_output\.txt\|eval_qwen_max\.log\|eval_results_qwen_max\.json\|eval_results_real\.json\|test_intermediate_output\.txt\|test_retrieval_output\.txt" --include="*.py" 2>/dev/null
```
Expected（轮 a）:
- 仅 1 处删除清单**内部互引**：
  - `_print_badcases.py:111` 写 `badcase_flows.json`
- （`extract_chunks.py:133` 的字符串是 `extracted_chunks.json`，而 grep 模式里的 `extract_chunks` 不是 `extracted_chunks` 的子串，所以不会命中；这是 grep 字符串子串匹配的正常行为。）
- 无其他活跃代码命中（特别是 `eval_router.py` / `agent_eval_router.py` / fixture / 默认 `--input` 值）。

Run（轮 b：多扩展名 + 全 repo）:
```bash
grep -rn --include="*.py" --include="*.yaml" --include="*.yml" --include="*.json" --include="*.sh" --include="*.toml" --include="*.ini" --include="*.md" --include="*.txt" "badcase_flows\.json\|snapshots_tmp\.json\|tc_list\.json\|eval_output\.txt\|eval_qwen_max\.log\|eval_results_qwen_max\.json\|eval_results_real\.json\|test_intermediate_output\.txt\|test_retrieval_output\.txt" . 2>/dev/null | grep -v "^\./\(_print_badcases\)\.py:" | grep -v "^\./docs/superpowers/" | grep -v "^\./docs/sdd/" 2>&1
```
Expected（轮 b）:
- 唯一允许命中：`./eval_output.txt:1378:Results saved to: eval_results_real.json` —— 这是 `eval_output.txt` 自己日志里印的输出路径字串，非代码引用，无害（且整个 `eval_output.txt` 本身就在删除清单内，删了一并消失）。
- 其他任何命中都是新增活跃引用：**停**，等用户决定是否再剔除一项。

新增命中（任何一轮）：**停**。

- [ ] **Step 5: 抓工作区基线（供后续步骤动态核对）**

用户工作区有 80+ 持续在改的源码。从核查到执行有时间差，期间 `M` / `D` / `??` 计数都会变。本 Step 抓当前快照作为基线，后续 Task 4 / Task 5 不再用写死数字（如 80、37）核对，而是跟此基线比相对变化。

Run:
```bash
M_BASE=$(git status --porcelain | grep -c '^ M ')
D_BASE=$(git status --porcelain | grep -c '^ D ')
QQ_BASE=$(git status --porcelain | grep -c '^?? ')
{
  echo "M_BASE=$M_BASE"
  echo "D_BASE=$D_BASE"
  echo "QQ_BASE=$QQ_BASE"
} | tee /tmp/phase2-baseline.txt
```
Expected: 3 行，三个数字落地到 `/tmp/phase2-baseline.txt`。`M_BASE` 通常在 80 上下，`D_BASE` 通常 0 或 1，`QQ_BASE` 通常 50+。具体数字不预判，**抓到的就是基线**。

后续 Task 4 / Task 5 的预期：
- `M_NOW == M_BASE`（commit 不动工作区源码改动）
- `D_NOW == D_BASE`（同上）
- `QQ_NOW == QQ_BASE - 18`（Task 2 mv 走了 18 个 untracked；允许 ±5 漂移，因 vite 临时文件等运行时产物）

- [ ] **Step 6: 显式确认通过本核对门**

到此 Step 1-5 全部 Expected 满足，才进入 Task 1。

如果执行者是 agent：把 Step 1-5 实际输出回报给用户，等明确"可以删"再继续。
如果执行者是人：肉眼复核完，再进 Task 1。

---

## Task 1: 删 3 个 tracked 文件（git rm）

**Files:**
- Modify (git index + disk):
  - `core_agent_lines.sh`
  - `extracted_chunks.json`
  - `todolist.md`

**Interfaces:**
- Consumes: Task 0 已确认这 3 个文件 tracked、无外部引用。
- Produces: 暂存区出现 3 个 `D ` 条目；磁盘文件被删；这 3 个内容是最终 commit 的全部内容。

- [ ] **Step 1: 执行 git rm**

Run:
```bash
cd /d/Code/nanobot
git rm core_agent_lines.sh extracted_chunks.json todolist.md
echo "exit=$?"
```
Expected: 输出 3 行 `rm '...'`，exit 0。

- [ ] **Step 2: 验证 index 状态**

Run:
```bash
git status --porcelain | grep -E '^D  (core_agent_lines\.sh|extracted_chunks\.json|todolist\.md)$'
echo "---match count: $(git status --porcelain | grep -cE '^D  (core_agent_lines\.sh|extracted_chunks\.json|todolist\.md)$')---"
```
Expected: 3 行 `D  ...`，match count = 3。

- [ ] **Step 3: 验证磁盘文件已删**

Run:
```bash
ls core_agent_lines.sh extracted_chunks.json todolist.md 2>&1
```
Expected: 3 个 `No such file or directory` 错误。

- [ ] **Step 4: 没碰别的文件**

Run:
```bash
git diff --cached --name-only
```
Expected: 恰好 3 行 —— `core_agent_lines.sh`、`extracted_chunks.json`、`todolist.md`。无其他文件。

任何额外文件：**停**。

---

## Task 2: 删 18 个 untracked 文件（mv 兜底）

**Files:**
- Create: `/tmp/phase2-backup-<timestamp>/` 备份目录
- Modify (disk only, git unaware): 18 个 untracked 文件从 repo root 移到备份目录

**Interfaces:**
- Consumes: Task 1 完成（暂存区已含 3 个 D）。
- Produces: 磁盘上 18 个文件消失；备份目录有 18 个文件副本；git status 中这 18 个 `??` 条目消失；暂存区**不变**（仍然只有 Task 1 的 3 个 D）。

注：18 个文件 untracked，git 一开始就不感知，mv 不影响 index 也不入 commit。

- [ ] **Step 1: 创建备份目录**

Run:
```bash
BACKUP_DIR="/tmp/phase2-backup-$(date +%Y%m%d-%H%M)"
mkdir -p "$BACKUP_DIR"
echo "$BACKUP_DIR" > /tmp/phase2-backup-path.txt
echo "backup dir: $BACKUP_DIR"
ls -la "$BACKUP_DIR"
```
Expected: 备份目录创建成功，空目录。`/tmp/phase2-backup-path.txt` 记下路径供后续 step 引用。

- [ ] **Step 2: mv 16 个非简历文件（5 脚本 + 9 dump + 2 失效产物）**

Run:
```bash
BACKUP_DIR=$(cat /tmp/phase2-backup-path.txt)
non_resume=(
  "_clean_all.py" "_fix_session_history.py" "_print_badcases.py" "_update_testcases.py"
  "extract_chunks.py"
  "badcase_flows.json" "snapshots_tmp.json" "tc_list.json"
  "eval_output.txt" "eval_qwen_max.log"
  "eval_results_qwen_max.json" "eval_results_real.json"
  "test_intermediate_output.txt" "test_retrieval_output.txt"
  "server.log" "token.json"
)
moved=0
for f in "${non_resume[@]}"; do
  if [ -e "$f" ]; then
    mv "$f" "$BACKUP_DIR/"
    moved=$((moved+1))
  else
    echo "MISSING (skipped): $f"
  fi
done
echo "moved=$moved   expected=16"
ls "$BACKUP_DIR" | wc -l
```
Expected: `moved=16`，备份目录有 16 个文件，无 MISSING。

- [ ] **Step 3: 验证 16 个磁盘文件已消失**

Run:
```bash
BACKUP_DIR=$(cat /tmp/phase2-backup-path.txt)
non_resume=(
  "_clean_all.py" "_fix_session_history.py" "_print_badcases.py" "_update_testcases.py"
  "extract_chunks.py"
  "badcase_flows.json" "snapshots_tmp.json" "tc_list.json"
  "eval_output.txt" "eval_qwen_max.log"
  "eval_results_qwen_max.json" "eval_results_real.json"
  "test_intermediate_output.txt" "test_retrieval_output.txt"
  "server.log" "token.json"
)
still_on_disk=0
for f in "${non_resume[@]}"; do
  [ -e "$f" ] && { echo "STILL_ON_DISK: $f"; still_on_disk=$((still_on_disk+1)); }
done
echo "still_on_disk=$still_on_disk   expected=0"
echo "backup file count: $(ls "$BACKUP_DIR" | wc -l)"

echo
echo "=== 反向校验：留痕项必须仍在磁盘（**绝不**能被这一步误 mv 走）==="
preserved=(
  "eval_data_real.json"
  "health_set_draft.yaml" "health_set_draft_v2.yaml" "health_set_draft_v3.yaml"
)
gone=0
for f in "${preserved[@]}"; do
  if [ -e "$f" ]; then
    echo "PRESERVED-OK: $f still on disk"
  else
    echo "PRESERVED-GONE: $f MISSING — this is a BUG, stop"
    gone=$((gone+1))
  fi
done
echo "preserved_missing=$gone   expected=0"
```
Expected: `still_on_disk=0`，备份计数 16；`preserved_missing=0`，4 个留痕项全部仍在磁盘上。

如果 `preserved_missing > 0`：**停**，意味着 Step 2 的 non_resume 数组被错误地包含了某个留痕项，可能误删了用户在用的文件，需要立刻从备份恢复。

- [ ] **Step 4: 单独 mv 2 个简历文件（最后一组，最高谨慎）**

Run:
```bash
BACKUP_DIR=$(cat /tmp/phase2-backup-path.txt)
echo "=== about to mv resume files ==="
ls -la resume_part1_agent.txt resume_star.txt 2>&1
echo "---"
mv resume_part1_agent.txt "$BACKUP_DIR/"
mv resume_star.txt "$BACKUP_DIR/"
echo "exit-after-resume-mv=$?"
echo "=== verify ==="
ls -la resume_part1_agent.txt resume_star.txt 2>&1
ls -la "$BACKUP_DIR"/resume_part1_agent.txt "$BACKUP_DIR"/resume_star.txt
echo "backup total: $(ls "$BACKUP_DIR" | wc -l)"
```
Expected:
- 第一段 ls 显示两文件原始信息（大小、mtime）
- 两条 mv 成功
- 第二段 ls 显示原位置 `No such file or directory`、备份目录两文件存在
- backup total = 18

- [ ] **Step 5: 复验 —— 18 个文件磁盘上全部消失 + 4 个留痕项仍在磁盘 + 暂存区未被污染**

文件名里含 `.` 等正则元字符，不用 porcelain 正则比对（容易误判）；直接用磁盘存在性检查（同 Step 3 风格），最稳。

Run:
```bash
files=(
  "_clean_all.py" "_fix_session_history.py" "_print_badcases.py" "_update_testcases.py"
  "extract_chunks.py"
  "badcase_flows.json" "snapshots_tmp.json" "tc_list.json"
  "eval_output.txt" "eval_qwen_max.log"
  "eval_results_qwen_max.json" "eval_results_real.json"
  "test_intermediate_output.txt" "test_retrieval_output.txt"
  "server.log" "token.json"
  "resume_part1_agent.txt" "resume_star.txt"
)
still_on_disk=0
for f in "${files[@]}"; do
  [ -e "$f" ] && { echo "STILL_ON_DISK: $f"; still_on_disk=$((still_on_disk+1)); }
done
echo "still_on_disk=$still_on_disk   expected=0"

echo
echo "=== 反向校验：4 个留痕项必须仍在磁盘 ==="
preserved=(
  "eval_data_real.json"
  "health_set_draft.yaml" "health_set_draft_v2.yaml" "health_set_draft_v3.yaml"
)
gone=0
for f in "${preserved[@]}"; do
  if [ -e "$f" ]; then
    echo "PRESERVED-OK: $f still on disk"
  else
    echo "PRESERVED-GONE: $f MISSING — BUG, stop"
    gone=$((gone+1))
  fi
done
echo "preserved_missing=$gone   expected=0"

echo
echo "---暂存区核对（应该仍只有 3 个 D）---"
git diff --cached --name-only
echo "---staged count: $(git diff --cached --name-only | wc -l)（应为 3）---"
```
Expected: `still_on_disk=0`，`preserved_missing=0`，暂存区仍然只有 Task 1 的 3 个 `D` 条目，staged count = 3。

---

## Task 3: 删根 `~/` 死数据目录（mv 兜底，**须用户亲眼确认**）

**Files:**
- Modify (disk only, git unaware): 字面意义根 `~/` 整目录移到备份。

**Interfaces:**
- Consumes: Task 2 完成（18 个 untracked 文件已 mv，暂存区只剩 3 个 D）。
- Produces: 磁盘上 `~/` 消失；备份目录多一个 `tilde-dir/` 子目录含 99 文件 / 15MB。

**为什么这一步独立成 Task 且需用户亲眼确认**：
- 命令里的 `~` 字符在 shell 里默认会展开成真 home 目录。`"~"`（带双引号）是字面 `~`，但前提是磁盘上那个目录**真的就是单字符 `~`** —— 不是 `~tmp`、不是 `~ `（带尾空格）、不是字节序列里夹了不可见字符的怪名字。Windows + Git Bash 环境里 `~` 误展开的产物可能不是规范 `~`。盲用 `mv "~"` 假设它精确叫 `~`，万一不是，要么命令失败、要么 `mv` 把别的东西挪走。这是 Phase 2 唯一一条**真有可能误伤用户实际 home** 的命令。
- 因此 Step 1 必须先用 `find` + 字节级显示**精确捕获**那个字面目录的真实文件系统名字，存到 shell 变量里。后续 mv 用变量（带 `./` 前缀）操作，**绝不**用裸 `~` 或带引号 `"~"` 假设它叫什么。

- [ ] **Step 1: 精确捕获字面 tilde 目录的真实名字**

Run:
```bash
cd /d/Code/nanobot
echo "=== find all repo-root entries whose name starts with ~ ==="
find . -maxdepth 1 -name '~*' -print
echo
echo "=== count of such entries ==="
TILDE_COUNT=$(find . -maxdepth 1 -name '~*' -print | wc -l)
echo "tilde-prefix entry count = $TILDE_COUNT"
echo
echo "=== if exactly 1 match, capture path + show byte sequence ==="
if [ "$TILDE_COUNT" -eq 1 ]; then
  TILDE_PATH=$(find . -maxdepth 1 -name '~*' -print)
  echo "captured: '$TILDE_PATH'"
  printf '%s' "$TILDE_PATH" | xxd
  echo "$TILDE_PATH" > /tmp/phase2-tilde-path.txt
  echo "(saved to /tmp/phase2-tilde-path.txt for later steps)"
else
  echo "WARNING: not exactly 1 match — STOP and report to user"
fi
echo
echo "=== ls -la --quoting-style=c on captured target (escape unusual bytes) ==="
[ -n "$TILDE_PATH" ] && ls -ld --quoting-style=c "$TILDE_PATH" 2>&1
echo
echo "=== target contents + size + file count ==="
[ -n "$TILDE_PATH" ] && ls -la "$TILDE_PATH" 2>&1
[ -n "$TILDE_PATH" ] && du -sh "$TILDE_PATH" 2>&1
[ -n "$TILDE_PATH" ] && find "$TILDE_PATH" -type f | wc -l
echo
echo "=== 真 home 状态（确认不动）==="
ls -ld ~/
echo "real home expanded as: $(printf '%s\n' ~/)"
```
Expected:
- `find` 输出**恰好 1 行**，形如 `./~`（带 `./` 前缀，因为 find 从 `.` 开始）。`TILDE_COUNT=1`。
- `xxd` 输出该字面名的精确字节序列，常规情况下是 `00000000: 2e 2f 7e` (= `./~`，3 字节)。任何额外字节（如 `2e 2f 7e 20` = 尾随空格、`2e 2f 7e xx ...` = 多字符）**都是异常信号**。
- `ls -ld --quoting-style=c` 应输出 `"./~"` 这种带 C 转义的引用形式；任何 `"./~\xxx"` 或 `"./~ "` 表明有不可见字节。
- target 内容约 99 文件 / 15M，目录里只有 `.nanobot/`。
- 真 home 路径展开形如 `/c/Users/Augix/`，完全独立于 `./~`，**两者不重叠**。

`TILDE_COUNT ≠ 1` 或 xxd 显示非预期字节：**停**，原样报回用户，不进 Step 2。

- [ ] **Step 2: 停下等用户明确放行**

执行者（agent）：把 Step 1 的全部输出（特别是 `find` 行、`xxd` 行、`ls --quoting-style=c` 行、真 home 路径）原样发给用户。**不**自动跑 Step 3。等用户明确说"可以 mv"或同等表态，才进 Step 3。

执行者（人）：肉眼复核 Step 1 输出，特别确认 (a) `TILDE_PATH` 字节序列就是 `./~`（3 字节）；(b) 真 home 路径展开为 `/c/Users/Augix/` 或同类，与 `./~` 不重叠；再敲 Step 3。

- [ ] **Step 3: 执行 mv（仅在用户放行后，用捕获的字面路径）**

Run（**绝不**用 `mv "~"`，用变量里捕获的字面路径 + `./` 前缀防 shell 误展开）:
```bash
BACKUP_DIR=$(cat /tmp/phase2-backup-path.txt)
TILDE_PATH=$(cat /tmp/phase2-tilde-path.txt)
echo "about to: mv '$TILDE_PATH' -> '$BACKUP_DIR/tilde-dir'"
mv "$TILDE_PATH" "$BACKUP_DIR/tilde-dir"
echo "exit=$?"
```
Expected: exit 0。命令文本里 `'$TILDE_PATH'` 应展开成 `./~`（或 Step 1 捕获到的任何精确字节）。

如果 exit ≠ 0：**停**，原样报错给用户，不重试，不切换命令格式。

- [ ] **Step 4: 验证字面 tilde 目录消失、备份齐全、真 home 完好**

Run:
```bash
BACKUP_DIR=$(cat /tmp/phase2-backup-path.txt)
TILDE_PATH=$(cat /tmp/phase2-tilde-path.txt)
echo "=== '$TILDE_PATH' 应不存在 ==="
ls -ld "$TILDE_PATH" 2>&1
echo
echo "=== find 复验：repo root 不再有 ~* 条目 ==="
find . -maxdepth 1 -name '~*' -print
echo "(empty above = good)"
echo
echo "=== 备份位置应有 tilde-dir ==="
ls -ld "$BACKUP_DIR/tilde-dir"
du -sh "$BACKUP_DIR/tilde-dir"
find "$BACKUP_DIR/tilde-dir" -type f | wc -l
echo
echo "=== 真 home 仍完整 ==="
ls -ld ~/
ls ~/ | head -5
```
Expected:
- 第一段：`No such file or directory`
- `find` 复验：空输出
- 第三段：备份 tilde-dir 存在，约 15M / 99 文件
- 第四段：真 home 正常（看到你 home 下常规文件如 `.bashrc` / `Documents` / `AppData` 等）

任何异常（比如真 home 内容变少、备份目录文件数远小于 99）：**停**报警，可能命令被误展开。

---

## Task 4: 暂存区核对门（commit 前唯一拦截点）

**Files:** 不动任何文件，纯检查。

**Interfaces:**
- Consumes: Task 3 完成后状态（暂存区应仍只含 Task 1 的 3 个 D）。
- Produces: 用户书面确认"暂存区干净"才能进 Task 5。

**为什么这一步独立成 Task**：与 Phase 1 Task 3 同款强制闸口。Phase 2 期间用户工作区那 80+ 未暂存源码改动持续在改，commit 误带它们的风险与 Phase 1 同级。Task 2/3 的 mv 不影响 index，所以理论上暂存区只应该有 Task 1 的 3 个 D —— 但必须 **眼睛见过** 才能往下走。

- [ ] **Step 1: 全量 git status**

Run:
```bash
git status
```
Expected 输出结构（**逐项核对，全对才能继续**）：

- "Changes to be committed:" 区域恰好 3 行：
  - `deleted:    core_agent_lines.sh`
  - `deleted:    extracted_chunks.json`
  - `deleted:    todolist.md`
- "Changes not staged for commit:" 区域：用户正在改的源码（`M`/`D`），**与 Task 0 Step 5 抓的 baseline 一致**，不能出现额外修改。
- "Untracked files:" 区域：相比 baseline 少 18 个（Task 2 mv 走的），但**不**能出现 Task 2/3 已 mv 走的具体 18 文件或字面 tilde 目录。同时 4 个留痕项（`eval_data_real.json`、`health_set_draft.yaml/_v2/_v3`）应**仍**作为 `??` 出现（它们是 untracked，没被 Task 2 动）。

- [ ] **Step 2: 反向 grep —— 暂存区只有 3 个预期文件**

Run:
```bash
git diff --cached --name-only | grep -vE '^(core_agent_lines\.sh|extracted_chunks\.json|todolist\.md)$'
echo "---grep exit=$?---"
echo "outside count = $(git diff --cached --name-only | grep -cvE '^(core_agent_lines\.sh|extracted_chunks\.json|todolist\.md)$')"
```
Expected: 输出空（grep exit 1 正常），count = 0。任何输出 → **停**，跑 `git reset HEAD <那一行>` 把额外文件退出暂存区，再重核。

- [ ] **Step 3: 暂存区总数 = 3**

Run:
```bash
git diff --cached --name-only | wc -l
```
Expected: 精确 `3`。

- [ ] **Step 4: 工作区相对 baseline 的变化符合预期**

Run:
```bash
source /tmp/phase2-baseline.txt
M_NOW=$(git status --porcelain | grep -c '^ M ')
D_NOW=$(git status --porcelain | grep -c '^ D ')
QQ_NOW=$(git status --porcelain | grep -c '^?? ')
QQ_EXPECTED=$((QQ_BASE - 18))
echo "M : base=$M_BASE  now=$M_NOW  (expect equal)"
echo "D : base=$D_BASE  now=$D_NOW  (expect equal)"
echo "?? : base=$QQ_BASE  now=$QQ_NOW  expect≈$QQ_EXPECTED  drift=$((QQ_NOW - QQ_EXPECTED))"
```
Expected:
- `M_NOW == M_BASE`
- `D_NOW == D_BASE`
- `QQ_NOW ≈ QQ_BASE - 18`，drift 绝对值 ≤ 5（vite timestamp 临时文件等运行时产物可造成小漂移）

`M_NOW ≠ M_BASE` 或 `D_NOW ≠ D_BASE` 或 `|drift| > 5`：**停**，可能误碰了工作区。

- [ ] **Step 5: 显式确认通过本核对门**

到此 Step 1-4 全部 Expected 满足，才进入 Task 5 commit。

agent 执行者：把 Step 1-4 实际输出贴给用户，等"可以 commit"才往下。
人执行者：肉眼复核完后再进 Task 5。

---

## Task 5: Commit

**Files:** 不修改文件，只产生 commit object。

**Interfaces:**
- Consumes: Task 4 已通过的暂存区（3 个 D）。
- Produces: 新 commit，HEAD 前进 1 步，未 push。

- [ ] **Step 1: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore(cleanup): Phase 2 — remove scattered orphan files

Drop 3 tracked root-level orphans:
- core_agent_lines.sh (one-off script)
- extracted_chunks.json (eval dump)
- todolist.md (stale planning note)

18 untracked orphans + literal "~/" stale RAG dir moved to /tmp backup
(disk-only, not in this commit). 4 items deferred to a later phase:
eval_data_real.json and health_set_draft{,_v2,_v3}.yaml are referenced by
scripts/ as argparse defaults and will be handled when scripts/ is cleaned up.
Refs docs/superpowers/specs/2026-06-26-repo-cleanup-design.md §2.4 Phase 2.
EOF
)"
```
Expected: `[main <hash>] chore(cleanup): Phase 2 ...`，`3 files changed, 0 insertions(+), N deletions(-)`，exit 0。

不要 `--amend` / `--no-verify`。

- [ ] **Step 2: 验证 commit 内容**

Run:
```bash
git log -1 --stat
echo "---"
git log -1 --name-only --format= | grep -v '^$'
echo "---count: $(git log -1 --name-only --format= | grep -cv '^$')---"
```
Expected: stat 显示 3 文件全是 `deleted`；name-only 输出恰好 3 行 `core_agent_lines.sh` / `extracted_chunks.json` / `todolist.md`；count = 3。

- [ ] **Step 3: 工作区未被吃掉（与 Task 4 baseline 同款对比）**

Run:
```bash
source /tmp/phase2-baseline.txt
M_NOW=$(git status --porcelain | grep -c '^ M ')
D_NOW=$(git status --porcelain | grep -c '^ D ')
QQ_NOW=$(git status --porcelain | grep -c '^?? ')
QQ_EXPECTED=$((QQ_BASE - 18))
echo "M : base=$M_BASE  now=$M_NOW  (expect equal)"
echo "D : base=$D_BASE  now=$D_NOW  (expect equal)"
echo "?? : base=$QQ_BASE  now=$QQ_NOW  expect≈$QQ_EXPECTED  drift=$((QQ_NOW - QQ_EXPECTED))"
git diff --stat | tail -1
```
Expected:
- `M_NOW == M_BASE`、`D_NOW == D_BASE`、`|drift| ≤ 5`（commit 不动工作区任何状态）
- diff --stat tail 仍显示工作区源码改动总行数（与 Task 4 Step 4 / Phase 1 Task 4 Step 3 同量级）

---

## Task 6: Push

**Files:** 不动文件，纯远端推送。

**Interfaces:**
- Consumes: Task 5 的新 commit。
- Produces: `origin/main` 前进 1 步。

- [ ] **Step 1: 状态确认**

Run:
```bash
git status -sb | head -1
```
Expected: `## main...origin/main [ahead N]`，`N ≥ 1`（Task 5 至少前进 1 步；若上一次 push 后又积累了别的 commit，N 可以更大）。

如果 `N == 0`：Task 5 commit 没成或 HEAD 不对，**停**。
如果不含 `[ahead`：本地已与 origin 对齐，Task 5 没产生 commit，**停**。

- [ ] **Step 2: Push**

Run:
```bash
git push origin main
```
Expected: `<old>..<new>  main -> main`，exit 0。

**不**用 `--force` / `-f` / `--force-with-lease`。

如果 push 失败（认证、reject、网络）：**停**，把原始错误发回用户。不重试 force / 重写历史 / 切 HTTPS。等用户决定（与 Phase 1 一样可能是 SSH key 问题，用户手动 push 即可）。

- [ ] **Step 3: 验证已对齐**

Run:
```bash
git status -sb | head -1
git log origin/main -1 --oneline
```
Expected: `## main...origin/main`（无 ahead），origin/main 顶端就是刚刚的 Phase 2 commit。

如果 fetch 失败（与 Phase 1 同款 SSH 问题但 push 是用户手动跑的），缓存的 `origin/main` ref 在 push 后已被更新，仍可信。

---

## Task 7: Push 后验证 + 清备份

**Files:** Push 后只读验证；备份目录用户许可后 `rm -rf`。

**Interfaces:**
- Consumes: Task 6 已推送状态 + Task 2/3 的备份目录。
- Produces: Phase 2 收尾确认；备份目录清除（若用户许可）或保留。

- [ ] **Step 1: 21 个目标文件磁盘上已消失、字面 tilde 目录也消失、4 个留痕项仍在磁盘**

用磁盘存在性检查代替 porcelain 正则（文件名含 `.` 等元字符）。

Run:
```bash
TILDE_PATH=$(cat /tmp/phase2-tilde-path.txt 2>/dev/null)
files=(
  "_clean_all.py" "_fix_session_history.py" "_print_badcases.py" "_update_testcases.py"
  "extract_chunks.py" "core_agent_lines.sh"
  "badcase_flows.json" "snapshots_tmp.json" "tc_list.json" "extracted_chunks.json"
  "eval_output.txt" "eval_qwen_max.log" "eval_results_qwen_max.json"
  "eval_results_real.json" "test_intermediate_output.txt" "test_retrieval_output.txt"
  "resume_part1_agent.txt" "resume_star.txt"
  "server.log" "token.json" "todolist.md"
)
residual=0
for f in "${files[@]}"; do
  [ -e "$f" ] && { echo "RESIDUAL: $f"; residual=$((residual+1)); }
done
echo "residual_count=$residual   expected=0"
echo
echo "=== 字面 tilde 目录 ==="
[ -n "$TILDE_PATH" ] && ls -ld "$TILDE_PATH" 2>&1
echo "find result:"
find . -maxdepth 1 -name '~*' -print
echo "(both above should say 'No such file' / empty)"
echo
echo "=== 4 个留痕项必须仍在磁盘（未被本 Phase 误删）==="
preserved=(
  "eval_data_real.json"
  "health_set_draft.yaml" "health_set_draft_v2.yaml" "health_set_draft_v3.yaml"
)
gone=0
for f in "${preserved[@]}"; do
  if [ -e "$f" ]; then
    echo "PRESERVED-OK: $f still on disk"
  else
    echo "PRESERVED-GONE: $f MISSING — BUG, stop"
    gone=$((gone+1))
  fi
done
echo "preserved_missing=$gone   expected=0"
```
Expected: `residual_count=0`；字面 tilde 目录 `No such file or directory`，`find` 输出空；`preserved_missing=0`。

- [ ] **Step 2: 后端可加载**

Run:
```bash
./backend/.venv/Scripts/python -m nanobot --help | head -20
echo "exit=${PIPESTATUS[0]}"
```
Expected: 看到 9 个子命令（onboard / gateway / agent / status / migrate / serve / channels / plugins / provider），exit 0。

注：用项目 venv，不用系统 python（Phase 1 Task 6 已确认 Windows Store python 占位符问题）。

- [ ] **Step 3: 备份完整性**

Run:
```bash
BACKUP_DIR=$(cat /tmp/phase2-backup-path.txt)
echo "=== backup root listing ==="
ls -la "$BACKUP_DIR"
echo
echo "=== backup file count (top-level, not recursive) ==="
ls "$BACKUP_DIR" | wc -l
echo
echo "=== tilde-dir backup ==="
ls -ld "$BACKUP_DIR/tilde-dir"
find "$BACKUP_DIR/tilde-dir" -type f | wc -l
du -sh "$BACKUP_DIR/tilde-dir"
```
Expected:
- backup 根有 19 个条目（18 文件 + 1 tilde-dir）
- tilde-dir 存在，约 99 文件 / 15M
- resume_part1_agent.txt、resume_star.txt 在 backup 根列表里可见

- [ ] **Step 4: 上报 + 等用户对清备份的明确指示**

把以下汇总贴给用户：
- Phase 2 commit hash & message 一行
- Step 1 结果（无 RESIDUAL）
- Step 2 后端启动 OK
- Step 3 备份目录路径 + 文件数 + 大小
- 工作区 ` M` count、` D` count、`??` count 三项数字

然后问用户：**"备份目录 `<path>` 现在清吗？还是先保留几天？"**

**不**自己决定。

- [ ] **Step 5: 清备份 —— 两条分支，等用户对话明确表态后选一**

**绝不**在 Run 块里写 `read -p` / `read -r` / 任何等输入的交互命令——agent 非交互环境会卡死。等用户在对话里给出明确表态（"清"或"保留"），再走对应分支。

**Branch A — 用户说"可以清" / "清" / 同等表态：**

Run:
```bash
BACKUP_DIR=$(cat /tmp/phase2-backup-path.txt)
TILDE_PATH=$(cat /tmp/phase2-tilde-path.txt)
echo "removing $BACKUP_DIR"
rm -rf "$BACKUP_DIR"
ec1=$?
rm /tmp/phase2-backup-path.txt /tmp/phase2-tilde-path.txt /tmp/phase2-baseline.txt
ec2=$?
echo "exit: rm-backup=$ec1  rm-paths=$ec2"
ls -ld "$BACKUP_DIR" 2>&1
```
Expected: 两个 exit 都 0；`ls -ld` 输出 `No such file or directory`。

**Branch B — 用户说"保留" / "先放着" / 同等表态：**

不执行任何 rm。把以下内容写进 Phase 2 收尾报告：
- 备份路径：`cat /tmp/phase2-backup-path.txt` 的内容
- 备份大小：`du -sh "$(cat /tmp/phase2-backup-path.txt)"`
- 备份文件清单：`ls "$(cat /tmp/phase2-backup-path.txt)"`
- 提示用户日后手动清理命令（不执行）：
  ```
  rm -rf "$(cat /tmp/phase2-backup-path.txt)"
  rm /tmp/phase2-backup-path.txt /tmp/phase2-tilde-path.txt /tmp/phase2-baseline.txt
  ```

`/tmp/phase2-backup-path.txt` / `/tmp/phase2-tilde-path.txt` / `/tmp/phase2-baseline.txt` 保留供用户日后引用。

---

## Self-Review Notes

- **Spec coverage**：spec §4 Phase 2 = "删 §2.4 列出的 19 个无引用文件；保留 5 个有引用；用户拍板 4 个待确认项"。
  - 19 → 实测 25 → Task 0 Step 3-4 引用复验剔除 4 项 → 最终 21（Task 1 + Task 2 覆盖全部 21）✓
  - 5 个保留项（seed_testcases.py / testcases.json / loadtest.py / README_old.md / case/）—— plan 不动它们，符合保留 ✓
  - 4 个待确认项 →
    - COMMUNICATION.md：保留（已拍板）→ 不动 ✓
    - SECURITY.md：保留（已拍板）→ 不动 ✓
    - 根 `bridge/`：保留（已拍板）→ 不动 ✓
    - 根 `~/`：删（已拍板）→ Task 3 ✓
  - 新发现 `backend/bridge/` 孤立目录 → 留 Phase 3（已在"不做但留痕"记一笔）✓
  - Task 0 引用复验新发现 4 项被 scripts/ 引用 → eval_data_real.json + 3 yaml 全部留给后续 Phase（已在"不做但留痕"记一笔）✓
- **Placeholder scan**：无 TBD / TODO / "appropriate error handling" / "similar to Task N"。所有命令完整写出，所有期望输出明确。
- **Type/name consistency**：路径名前后一致（21 个文件名在 Task 0 / Task 2 / Task 4 / Task 7 用同一份数组；4 个留痕项在 Task 2 Step 3/5 和 Task 7 Step 1 的反向校验数组里）；BACKUP_DIR 通过 `/tmp/phase2-backup-path.txt` 在 Task 2-7 之间稳定传递。
- **Risk coverage**:
  - "误删用户简历" → Task 2 Step 4 单独成步 + mv 兜底 ✓
  - "误展开 `~`" → Task 3 用户亲眼确认 + 引号 + Step 4 校验真 home 完好 ✓
  - "commit 误带工作区源码" → Task 4 强制闸口（与 Phase 1 Task 3 同款）✓
  - "push 失败 retry force" → Task 6 Step 2 明文禁止 ✓
  - "提前清备份导致不可恢复" → Task 7 Step 5 等用户明确许可 ✓
  - "误删 4 个留痕项（eval_data_real.json + 3 yaml）" → Task 2 Step 3 / Step 5 / Task 7 Step 1 反向校验 + Task 4 Step 1 备注留痕项应仍作为 `??` 出现 ✓
- **Out-of-scope guard**：plan 内无 Phase 3-6 动作；4 项"不做但留痕" 集中列出 ✓。
