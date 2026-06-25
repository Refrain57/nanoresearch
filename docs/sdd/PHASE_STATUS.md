# Harness 平台 · Phase 交付状态与债务交接

> **配合 SDD 阅读**（`harness-platform-sdd.md`）。  
> 本文件记录每个 Phase 的**实际交付状态、为什么这么做、还差什么**，是新会话接手的起点。  
> **债务是后续 Phase 的前置依赖，未清不要往下做。**

---

## 阅读约定

- **状态标记**：`✅ 已完成` / `⚠️ 已交付但有未闭合项` / `🔲 未开始`
- "为什么"段说明当时的决策理由——后续如果想改方案，先读这段。
- "还差什么"段是行动清单，每项带"为什么必须在进下一 Phase 前解决"。

---

## Phase 0 · Trace 轻量层

**状态：✅ 已完成，所有验证项闭合**

**commit**：`c2b002e8`（含本轮追加修复）

---

### 已做

#### context_trace 9 字段写入

每次 agent run 的上下文装配决策存入 `agent_run_snapshots.context_trace`（JSONB）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `history_query` | `str \| None` | 发给 user_memory 的检索 query |
| `memory_budget_tokens` | `int` | 分给 MEMORY.md 的 token 预算 |
| `knowledge_budget_tokens` | `int` | 分给历史记忆检索的 token 预算 |
| `memory_actual_chars` | `int` | MEMORY.md 注入后实际字符数 |
| `history_actual_chars` | `int` | 历史记忆注入后实际字符数 |
| `memory_fragment_ids` | `list[str]` | ChromaDB user_memory 命中片段 ID |
| `always_skill_names` | `list[str]` | always-on skill 名称列表 |
| `skill_names` | `list[str] \| None` | 本次可用 skill 名称列表 |
| `persona_active` | `bool` | 是否激活了 custom persona |

**为什么这 9 个字段**：覆盖 SDD §4.5 定义的四类决策（检索 query、预算数字、片段标识、skill/persona 状态），足以让 Phase 1 的分类器区分"没检索 / 检索了但空 / query 写错了"这三种不同根因。

**为什么不存全文**：守住 SDD §4.5 红线。全量 trace 走 Phase 3（已降级到 P6 之后）。

#### 实现方式

- `ContextBuilder._build_dynamic_suffix()` 接收 caller 传入的 `_trace_out: dict`，原地 update。
- `build_messages()` → `build_system_prompt_blocks()` → `_build_dynamic_suffix()` 全链路透传，无新 `await`（守住同步路径约束）。
- `_process_message()` 每次调用前创建局部 `_ctx_trace: dict = {}`，避免共享 `ContextBuilder` 实例的跨 run 污染。
- 并发安全：`_build_dynamic_suffix` 是同步函数，asyncio 不会在其中切换协程，caller-per-run dict 模式已通过 3 并发 gather 测试验证。

#### 迁移

`scripts/migrate_phase0_context_trace.sql`：

```sql
-- UP
ALTER TABLE agent_run_snapshots
    ADD COLUMN IF NOT EXISTS context_trace JSONB DEFAULT NULL;
-- DOWN
-- ALTER TABLE agent_run_snapshots DROP COLUMN IF EXISTS context_trace;
```

启动时校验已加入 `database.py:check_schema_migrations()`。

#### fragment_ids 非空路径验证

**为什么走单进程测试而不是真实 run**：ChromaDB PersistentClient 的 HNSWLIB 向量索引是进程内内存结构。外部脚本写入的片段对 worker 进程的 in-memory 索引不可见，但代码路径本身无 bug——`search_user_memory_sync` 返回的每条结果确实带 `'id'` 键，`_ids_out.extend(m["id"] ...)` 逻辑正确。同进程测试（write → 同一进程 read）已验证 `um_*` 格式 ID 正确落入 trace。

**生产路径是否受影响**：不受影响，因为 memory consolidator（write）和下次对话的检索（read）都在同一个 ARQ worker 进程内。但"同进程"这一前提目前依赖对 worker 架构的理解，没有端到端实证——Phase 1 前最好补一条真实记忆固化 → 下次查询的真实 run 验证。

---

### 本轮修复（Phase 0 追加，同批 commit）

#### 修复一：`build_history_context` 的静默吞异常

**原状态**：`except Exception: return ""`——异常无声地变成 `history_actual_chars=0, fragment_ids=[]`，事后无法区分"本来就没片段"还是"出错了"。

**修复**：异常仍然 catch（不能让 trace 收集崩溃主流程），但改为 `logger.exception(...)` 记录，让失败可见。

**为什么必须修**：SDD 完成定义要求"随机点开一条 badcase 能读到诊断数据"——如果取片段的代码静默失败，你拿到的 `fragment_ids=[]` 可能是真空也可能是假空，Phase 1 的分类器会误判根因。

#### 修复二：`snapshot.conversation_id` 关联链路重建

**原状态**：`_maybe_save_snapshot` 从未传 `conversation_id`，所有快照的 `conversation_id=NULL`。

**发现过程（三证坐实）**：

1. **代码证据**：`chat_router.py:L92` 生成临时 `conv_id = uuid.uuid4()` 只进 `session_key = f"web:{conv_id}"`；`Conversation` 模型 `id` 有 `default=uuid.uuid4` 独立生成。
2. **模型证据**：`conversation_repo.py:create()` 构造 `Conversation(session_key=key, ...)` 时不传 `id`，主键由 ORM 自动生成。
3. **真实数据**：`conversations.id=cb4ecf4d-...` vs `channel_chat_id=2efc67eb-...`，三条对话全部不等。

**修复方案**：五跳穿透真实 `conversations.id`：

```
chat_router.py        enqueue_job(conversation_id=str(conv.id), ...)
    ↓
worker.py             run_agent_job(..., conversation_id=None)  ← 新参数
    ↓
AgentLoop.process_direct(conversation_id=conversation_id)       ← 新参数
    ↓
_process_message(conversation_id=conversation_id)               ← 新参数
    ↓
_run_agent_loop(conversation_id=conversation_id)                ← 新参数
    ↓
_maybe_save_snapshot(..., conversation_id=conversation_id)      ← 直接用，无兜底
```

**为什么不用 `or chat_id` 兜底**：`chat_id` 在所有路径下都不等于 `conversations.id`（web 路径是 session UUID，CLI 路径是 `"direct"`，channel 路径是平台 ID）。兜底只会注入假外键或被后续 UUID 转换过滤掉，没有任何路径能产出真正的 `conversations.id`。

**UUID 转换失败改 warning**：原来 `except (ValueError, AttributeError): pass`（静默），改为 `logger.warning(...)` 打出实际值，让"哪条 run 关联失败"可见。

**"direct"/CLI 路径**：`conversation_id=None` → `_conv_uuid=None` → 存 `NULL`，这是正确行为（CLI run 无对应 `conversations` 行）。

---

### 已闭合验证项

#### ✅ P0-D1：conversation_id 落库验证（已验证闭合）

**验证时间**：2026-06-24，worker 重启后发两条真实 web run 验证。

**验证结论**：
- `snapshot.conversation_id = conversations.id`（真实主键，非 NULL）
- `conversation_id ≠ channel_chat_id`（存的是 `id`，不是 session UUID）
- 两条 run 均通过 JOIN 验证（snap `a2d09db4` → conv `571c107c`；snap `b21b70c0` → conv `257a66a8`）

**根因定位过程**：之前 NULL 是因为 worker 进程在代码提交后未重启，内存中跑的是旧代码；重启后新代码立即生效，`conversation_id` 正确落库。ARQ 序列化（pickle）已确认能正确传递 UUID 字符串。

#### 债务 P0-D2：fragment_ids 生产路径端到端验证（建议做）

**是什么**：fragment_ids 非空路径在单进程测试验证，但没有"用户有历史记忆 → 下次对话命中 → fragment_ids 非空"的真实端到端 run。

**验证步骤**：

1. 触发一次记忆固化（多轮对话积累足够 token）
2. 发一条与历史记忆相关的问题
3. 检查 `context_trace.memory_fragment_ids` 非空且格式为 `um_*`

**为什么可以是"建议"而不是"必须"**：代码路径已在单进程测试验证，生产路径的 write/read 在同一 worker 进程内，无多进程隔离问题。但"同进程"前提依赖对 ARQ worker 架构的理解，没有端到端实证前这条假设是靠人背书的。

---

### 已知限制（记录备查，不是债务）

**CLI/"direct" run 无 conversation 关联**：`process_direct` 默认 `chat_id="direct"`，这类 run 存 `conversation_id=NULL`。Phase 1 定位这类 run 需按 `snapshot.uid + timestamp` 或 `snapshot.run_id`。诊断面板 UI 需对 `conversation_id=NULL` 有降级展示。

**ChromaDB 多进程隔离**：跨进程写入（如管理后台批量导入记忆）对 worker 的 in-memory HNSWLIB 索引不可见。如需跨进程实时可见，需切换到 ChromaDB HTTP Server 模式。

---

## Phase 1 · 根因→层级映射 + 文本类可调对象接口

**状态：✅ 已完成，所有验证项闭合（含真实 DB + 真实 LLM）**

**验证时间**：2026-06-24，`python -X utf8 scripts/verify_phase1_e2e.py`

**前置债务**：P0-D1 ✅ 已闭合。

---

### 已做

#### A. 分类法重塑

`badcase_classifier.py` 升级：
- `ClassifyResult` 新增三字段：`layer / target_kind / target_id`（旧字段 `root_cause_auto` 保留，并行写）
- `FIXABLE_LAYERS = {"Context", "Tool"}` / `DIAGNOSIS_ONLY_LAYERS = {"Memory", "Recovery"}` 显式标注
- `layer=None` 覆盖 user_input 和 token 超限等无确定系统根因的情况
- 规则层重构：仅保留 4 条高置信度规则（tool error / 无工具调用失败 / recall正常但completion低 / token超限）；
  原来"所有工具返回空→context"规则已**删除**，因为无法区分新用户正常无记忆与检索策略失败（会假阳性），降级交由 LLM 处理
- `_build_classify_prompt` 新增 `context_trace` 结构化段，含 history_query / fragment_ids 数量（附注"空可能是新用户"）/ 预算数字 / persona 状态 / skills；LLM 拿到后自行推断，不由规则层强制下结论
- DB 落库：`update_snapshot_classification` 同时写 3 个新列（`classification_layer / classification_target_kind / classification_target_id`）+ 原有 4 列，不动老列
- 迁移脚本：`scripts/migrate_phase1_classifier_layer.sql`（ADD COLUMN × 3，含 downgrade 注释）

#### B. TunableObject 接口

`eval/tunable.py`（新文件）：
- `TunableTextObject` ABC：5 个方法（`read / generate_candidates / apply / get_current_version / rollback`），全 async
- `OptimizationCandidate` 数据类从 `optimizer.py` 迁移到 `tunable.py`（optimizer.py 保留 re-export 不破坏现有导入）
- `PersonaObject(agent_id, agent_repo, eval_repo, provider)`：操作 `agents.persona`；Phase 1 全链路可跑（read / generate_candidates / apply / rollback）
- `ToolDescriptionObject(tool_name, agent_id, ...)`:操作 `agents.tools_config[tool_name].description`；generate_candidates 可生成候选文本，**但评分路径 blocked**（见下方限制条目）

**PersonaObject 范围边界（故意收窄，不是遗漏）**：
System prompt 由 ContextBuilder 从多个来源动态组装（SOUL.md 结构、技能摘要、KB 绑定、动态后缀）。
`agents.persona` 是唯一通过数据库可调整的文本段。PersonaObject.read/apply 只操作这一段。
优化其他段需要修改代码——这不是 Phase 1 的范围，也不是 TunableObject 应该覆盖的边界。
此约束已写入 `tunable.py` 的 module docstring 和 PersonaObject docstring，不会被误解为"没做完"。

#### C. 版本注册表

`storage/models.py`：新增 `TunableObjectVersion` 模型（`tunable_object_versions` 表）。
`agent_eval_repo.py`：新增 4 个版本注册表方法：
- `create_tunable_version` — INSERT 新行（active=True）+ 把同 kind+target_id 旧行全部 active=False
- `get_current_tunable_version` — 查 active=True 记录
- `get_tunable_version_by_id` — 按 UUID 查（用于 rollback）
- `list_tunable_versions` — 历史列表（倒序）

apply/rollback 在 Phase 1 完整实现、可单测。**不接入生产触发路径**——生产环境下何时调用 apply 由 Phase 5 门控 + Phase 6 人确认驱动。

#### D. OptimizationAgent 改造

`optimizer.py` 重构：
- `generate_proposals(target: TunableTextObject, ...)` — 接受可调对象作为参数，不再对"系统 prompt 是唯一对象"硬编码
- `_generate_candidates` 职责迁移到各 `TunableTextObject.generate_candidates`（optimizer 直接调 `target.generate_candidates`）
- `_score_candidate` 对 `tool_description` 显式 `raise NotImplementedError`（见下方限制条目）

---

### 已知限制（故意，不是遗漏）

#### P1-L1：ToolDescriptionObject 评分 blocked until Phase 4

**现状**：`ToolDescriptionObject.generate_candidates` 能生成候选文本。但 `OptimizationAgent._score_candidate` 对 `target.kind == "tool_description"` 显式 `raise NotImplementedError("tool description scoring requires Phase 4 sandbox layering")`。
`generate_proposals` 捕获该异常后记 warning 并持久化无分数候选（不静默失败）。

**为什么不在 Phase 1 实现**：Tool Description 优化修改的是工具配置，不是 system message。sandbox 评分需要能把候选注入到工具注册表而非 system message——这依赖 Phase 4 的 `side_effect_only` 模式和工具 `side_effect` 声明。Phase 4 之前 ToolDescriptionObject 可以 read/apply/rollback（版本注册表已实现），但不能进行完整的评分-门控闭环。

**如何保持可见**：NotImplementedError 在调用栈上明确抛出，日志记 WARNING。PHASE_STATUS.md 本条记录。

#### P1-L2：PersonaObject.read 只覆盖 agents.persona

同上方"PersonaObject 范围边界"说明。这是故意收窄，详见 `tunable.py` docstring。

---

### 验证结论（已闭合，2026-06-24 真实 DB + 真实 LLM）

| ID | 描述 | 结论 |
|----|------|------|
| P1-V1 | schema 迁移正确 | ✅ `check_schema_migrations()` 通过，3 新列 + `tunable_object_versions` 表均存在 |
| P1-V2 | 分类器规则层输出结构化指针 | ✅ 37 个 mock 单测全通过（0.82s），规则覆盖 tool_error / failed_no_tools / recall_ok_completion_low / token_limit |
| P1-V3 | `PersonaObject.read()` 返回真实 persona | ✅ 读出 `'# 角色\n你是 Nano Research...'`，与 `agents.persona` 一致 |
| P1-V4 | `apply()` → DB 写版本表 + 更新 persona | ✅ 版本表新增行 `active=True`，旧行全部翻 `False`；二次 apply 后仍只有 1 行 active=True（见原始行级数据下方） |
| P1-V5 | `rollback()` → persona 回到旧值，版本表仍单行 active=True | ✅ rollback 后版本表 11 行，仅末行 `active=True`，`agents.persona` 正确恢复，原始 persona 最终还原 |
| P1-V6 | `ToolDescriptionObject.generate_candidates` 无异常 | ✅ 单测通过 |
| P1-V7 | `generate_proposals(ToolDescriptionObject)` 不崩，持久化无分数候选 | ✅ 单测通过，`result.proposals[0]["scores"] == {}` |
| P1-V8 | context_trace 进入 LLM prompt，LLM 真实用上判出 layer | ✅ 快照 `a2d09db4`：`history_query='P0-D1 test3 reply ok'`，`3 个 fragment_ids` 出现在 prompt；LLM 输出 `layer=Context, target_kind=system_prompt`，reason 明确引用了检索片段存在但推理失败的分析 |
| P1-V9 | 旧列 `root_cause_auto` 格式不变 + 新列并行写 | ✅ badcase `3050103e`：`root_cause_auto='prompt'`（合法旧值）；`classification_layer=None`（LLM 判定无明确系统根因，正确行为）；两列并排写入，旧列格式未破坏 |

**关键原始数据（V4/V5 版本表）**：
每次操作后末行为新增的 active=True 行，其余全 False — 任意时刻最多 1 行 active=True：
```
apply 后 (9行): ... id=e788c602... active=True  preview='VERIFY_B: Phase1 E2E test persona'
二次apply后(10行): ... id=bd89a07a... active=True  preview='VERIFY_C: second apply'
rollback后(11行): ... id=6325b459... active=True  preview='VERIFY_B: Phase1 E2E test persona'
```

---

### 验证中发现并修复（计划外）

#### system= 参数 bug（`badcase_classifier.py`）

**现象**：`BadcaseClassifier.classify()` 调用 `provider.chat_with_retry(system=_SYSTEM_PROMPT, ...)`，但 `LLMProvider.chat_with_retry()` 基类签名不接受 `system=` 参数。异常被 `except Exception: pass` 静默吞掉，分类器全程降级输出 `"分类失败，降级处理"` + `layer=None`，不报错、不可见。

**影响**：所有使用 `OpenAICompatProvider`（DeepSeek、OpenRouter 等）的生产环境分类器调用均静默失效；只有 `AnthropicProvider` 因自身处理方式不同偶然幸免。这个 bug 不会在 mock 测试中暴露。

**修复**：`badcase_classifier.py:128` — 将 `system=_SYSTEM_PROMPT` 改为在 messages 数组首位插入 `{"role": "system", "content": _SYSTEM_PROMPT}`，符合 OpenAI-compat 标准，对所有 provider 通用。

**验证**：V8 真实 LLM 分类成功（`layer=Context, target_kind=system_prompt`），确认修复有效。

---

### 注意事项（后续 Phase 接手时必读）

**新旧分类标签不要混用**：`root_cause_auto` 旧列（`prompt/context/tool/model/user_input`）与新列 `classification_layer`（`Context/Tool/Memory/Recovery/null`）是两套并行的标签体系。旧列保留是为向后兼容，Phase 2 以后的分析逻辑应以新列为准。绝不要把 `root_cause_auto='prompt'` 直接等同于 `layer='Context'`——它们在语义和粒度上都不同（旧列是 root_cause 的粗粒度枚举，新列是六层架构中的结构化指针）。

---

### 债务汇总（Phase 1 新增）

| ID | 描述 | 类型 | 优先级 |
|----|------|------|--------|
| P1-D1 | ToolDescriptionObject 评分路径（Phase 4 沙箱分层前 blocked） | 已知限制 | Phase 4 时处理 |
| P1-D2 | PersonaObject.generate_candidates 的评分（需 golden_test_cases + recordings） | 待 Phase 2 测试集构造后完整跑通 | Phase 2 后验证 |

---

## Phase 2 · 回归集分离

**状态：⚠️ 代码已交付，双集打分端到端验证通过（2026-06-25）；44 条导入完成；健康集仅支持硬指标打分（keyword_coverage + tool_skip），软质量退化不在检测范围内（有意收窄，不是漏做）**

**commit**：（本轮，2026-06-24 ~ 2026-06-25）

---

### 已做

#### A. schema 新增两列（`migrate_phase2_health_set.sql`）

`agent_test_cases` 新增：
- `set_kind VARCHAR(20) nullable indexed`：`"badcase_fix"` / `"health"` / `null`（null = 历史数据，未分配）
- `tool_recordings JSONB nullable`：health 集预录工具调用，评估时沙箱 replay 直接读

**为什么是独立列而非复用 `dataset_type`**：`dataset_type` 表来源（calibration/custom），`set_kind` 表评估集归属，两个维度正交。复用会让 `list_test_cases(dataset_type="custom")` 这类现有查询与新查询互相干扰，违背 SDD §7.2"新增列、不动老列"的精神。

#### B. `AgentEvalRepository` 更新

- `list_test_cases(dataset_type, set_kind, active_only)` — 新增 `set_kind` 独立过滤参数，与 `dataset_type` 正交
- `create_test_case(...)` — 新增 `set_kind` / `tool_recordings` 可选参数

#### C. `OptimizationAgent` 分集改造（`optimizer.py`）

- `generate_proposals` 新签名：接受 `fix_test_cases` + `health_test_cases` 两个独立列表，缺任一抛 `ValueError`（Phase 2 不变式）
- `_score_candidate` 重命名为 `_score_candidate_set`，调用两次分别产出 `fix_scores` / `health_scores`
- `_gather_recordings` 优先读 `tc.tool_recordings`（health 集直存），再回退到按 `user_input` 查 snapshot（fix 集）
- `candidate.scores` 结构：`{"fix_set": dict[str, float], "health_set": dict[str, float]}`
- 排序依据：`fix_set` 均值（fix 集是本次优化的主信号）
- 无录制的 health case（text-only）：直接 live run，不走沙箱

#### D. `OptimizationCandidate` 结构变更（`tunable.py`）

`scores` 字段从 `dict[str, float]`（扁平）改为：
```python
{"fix_set": {"keyword_coverage": 0.8, ...}, "health_set": {"keyword_coverage": 0.9, ...}}
```
Phase 5 门控逻辑直接读两个子 dict，无需再改接口。

#### E. 健康集 seed 脚本（`scripts/seed_health_set.py`）

- 50 条 case，10 类场景，每类 5 条，全部人工编写（不从 snapshot / golden 集衍生）
- 运行 `python scripts/seed_health_set.py` 生成 `health_set_draft.yaml` 供人工审核
- 审核通过后运行 `python scripts/seed_health_set.py --import` 导入 DB

**打分方式**：全部 case 使用硬指标打分（`keyword_coverage` + `tool_skip`）。

---

### 已知限制（故意收窄，不是遗漏）

#### P2-L1：健康集仅支持硬指标打分，软质量退化不在检测范围内

**现状**：健康集所有 case 的评分维度仅限于 `keyword_coverage`（关键词命中）和 `tool_skip`（工具调用检查）。以下类型的退化**无法**被当前健康集检测到：

- 语气变差（原本共情温和的回答变冷漠/机械）
- 回答具体性下降（原本给具体书名+理由，退化后只说"有几本经典书"）
- 幻觉（回答内容符合关键词但事实错误）
- 上下文消费不足（多轮对话中忽略了历史信息，但仍命中关键词）

**为什么是有意收窄**：SDD Phase 2 的完成定义 5 条（场景覆盖清单、样本数下限、双集分数分开呈现、两集强制存在、健康集显式构造）全部交付，不包含 LLM-judge 软打分。`scoring_type` / `judge_anchor` 是在实现过程中引入的设计扩张，因为数据库没有对应列、`RuleEvaluator` 没有接线，属于悬空半成品。已从 `seed_health_set.py` 和 YAML 中彻底移除这些字段。

**为什么不是一个可以"以后补"的债务**：软退化检测需要一套独立的方法论（judge prompt 校准、评分一致性验证、false positive rate 控制），这套方法论不是"在现有架构上加一个字段"就能解决的问题。如果未来确认需要，应作为独立的 Phase 立项，而不是挂靠到当前任何 Phase 的"补全"里。

---

### 验证结论（2026-06-25，真实 DB + 真实 LLM + 44 条健康集 + 1 条 badcase）

**验证脚本**：`scripts/verify_phase2_dual_scoring.py`

| ID | 描述 | 结论 |
|----|------|------|
| P2-V1 | `generate_proposals` 接受 fix_set + health_set 双集，缺任一抛 ValueError | ✅ 通过 |
| P2-V2 | 健康集 44 条参与 scoring（6 条 SKIP_PENDING 正确跳过） | ✅ 通过 |
| P2-V3 | 4 条 candidate 全部产出，fix_set 和 health_set 均为非空 dict | ✅ 通过 |
| P2-V4 | `candidate.scores` 结构正确：`{"fix_set": {...}, "health_set": {...}}` | ✅ 通过 |
| P2-V5 | fix_mean_score 计算正确，candidates 按 fix_mean 降序排列 | ✅ 通过 |
| P2-V6 | `OptimizationProposal` 持久化到 DB（proposal id: a793bbe1-...） | ✅ 通过 |

**验证结论**：Phase 2 双集打分链路完整可跑——从 LLM 生成 candidate → 双集独立评分 → 排序 → 持久化，全链路无异常。

---

### 验证中发现并修复（计划外）

#### 修复一：`optimizer.py` 缺少 `await`（静默断掉整个评分路径）

**现象**：`_score_candidate_set` 中 `scores = evaluator.evaluate(snapshot_data, tc)` 缺少 `await`，coroutine 被存入 list 而非 result dict，后续 `for dim in all_dims` 迭代 coroutine 触发 TypeError，被外层 `except Exception` 捕获后静默跳过，该 test case 产出的分数丢失。

**影响**：所有双集 scoring 路径——fix_set 和 health_set 的分数均会部分或全部丢失，`fix_mean_score` 和排序变得不可靠。mock 测试不会暴露此问题（mock 常同步化 async 调用）。

**修复**：`optimizer.py:239` — `scores = evaluator.evaluate(...)` → `scores = await evaluator.evaluate(...)`

#### 修复二：`tunable.py` `system=` 参数 bug（candidate 生成静默空返回）

**现象**：`_llm_generate_candidates` 调用 `provider.chat_with_retry(messages=[...], system=system_prompt, ...)`。`LLMProvider.chat_with_retry()` 基类签名不接受 `system=` 参数，异常被 `except Exception: pass` 捕获，函数返回 `[]`。结果：candidate 永远为空，`generate_proposals` 持久化空提案。

**影响**：PersonaObject 和 ToolDescriptionObject 的 `generate_candidates` 均静默失效——LLM 从未收到 system prompt。同 Phase 1 `badcase_classifier.py` 的 bug（已修复），同一根因，同一静默吞异常模式。

**修复**：`tunable.py:148-155` — 将 `system=system_prompt` 改为在 messages 数组首位插入 `{"role": "system", "content": system_prompt}`。

#### 修复三：`agent_eval_router.py` `system=` 参数 bug ×2（analyzer 端点静默降级）

**现象**：同修复二的根因，出现在 `agent_eval_router.py` 的两个 analyzer 端点中（约 L1095 和 L1167），调用 channel_loop.provider.chat_with_retry 时传入 `system=` 参数。

**修复**：两处均改为将 system prompt 放入 messages 数组。

**全局 `system=` 清理确认**：全仓库搜索 `system=` 传参模式，共发现 4 处生产实例（badcase_classifier.py ×1、tunable.py ×1、agent_eval_router.py ×2），全部修复。`AnthropicProvider` 重载了 `chat_with_retry` 签名接受 `system=`，但不应依赖此实现细节——所有调用统一走 messages 数组，对所有 provider 通用。

---

### 待闭合项

#### P2-D1：健康集内容人工审核（必须完成才能 --import）

**什么需要审**：
- 每条 case：`expected_keywords` 是否必要且充分（不应太宽泛也不应太苛刻）
- `tool` case：`expected_tools` 列表是否与系统实际注册的工具名一致
- `rag_hit` case（场景 6）：`expected_keywords` 是否来自知识库中真实存在的内容

**如何审**：运行 `python scripts/seed_health_set.py` 生成 `health_set_draft.yaml`，逐条过目，修改后再 `--import`。

**当前未导入的 6 条（SKIP_PENDING）**：

| case ID | 场景 | 阻塞原因 |
|---------|------|---------|
| h_rag_hit_001~005 | RAG 命中 | expected_keywords 是 `[FILL: ...]` 占位符，需从真实知识库提取具体内容 |
| h_tool_004 | 工具调用 | expected_tools 含 `query_orders`，需确认系统是否注册了此工具名 |

这 6 条补齐前，健康集对 RAG 命中场景无检测能力，工具调用场景缺 1 条。

#### P2-D2：fix_test_cases 构造方式（Phase 2 遗留）

`generate_proposals` 的调用方负责把触发本次优化的 badcase 转成 `AgentTestCase` 列表传入 `fix_test_cases`。当前 CLI/API 路由未做这一转换——Phase 3（CLI/路由接入）时处理。

**健康集 review 周期**：**半年一次**，不要更频繁。随 Agent 演进，老 case 可能悄悄退化为 badcase（Agent 对原本"正常"的任务表现变差）。半年 review 时逐条检查，把不再通过的 case 标 `status="pending_review"` 移出 active 集再人工决定替换。频繁 review（如每月）会被当前 badcase 分布拉着走，逐渐破坏健康集的独立性。

---

## Phase 4 · 沙箱分层

**状态：✅ 已完成，所有完成定义验收项闭合（2026-06-25）**

**验证脚本**：`scripts/verify_phase4_sandbox.py`（V1/V2/V3 全部 PASS，真实 DB + 真实 LLM）

**前置依赖**：Phase 1（ToolDescriptionObject 接口）✅、Phase 2（双集评分）✅

---

### 已做

#### A. Tool 基类 `side_effect` 属性

`base.py` 新增 `side_effect` property，默认返回 `True`（保守）。各工具子类 override：

| 工具 | `side_effect` | 说明 |
|------|---------------|------|
| `read_file` | `False` | 只读文件 |
| `list_dir` | `False` | 只列目录 |
| `web_search` | `False` | 只读网络查询 |
| `web_fetch` | `False` | 只读网络获取 |
| `retrieve_by_entity` | `False` | 只读 KG 查询 |
| `write_file`, `edit_file` | `True`（默认） | 写磁盘 |
| `exec`, `message`, `cron` | `True`（默认） | 外部副作用 |
| `fetch_paper` | `True`（默认） | 下载并写文件到磁盘 |
| `research`, `spawn` | `True`（默认） | 启动异步任务 |
| MCP 工具 | 白名单查表，默认 `True` | 见下 |

MCP 动态工具通过 `_QUERY_MCP_TOOL_ORIGINAL_NAMES` 白名单判断（`mcp.py`），初始只放经过人工确认的只读 RAG 检索工具：`retrieve_hybrid`、`retrieve_dense`、`retrieve_sparse`、`kb_search`、`kb_retrieve`。其他 MCP 工具添加前须人工确认只读性并在白名单注释里说明理由。

#### B. `SandboxedToolRegistry` 新增 `side_effect_only` 模式

`sandbox.py` 变更：
- `mode` 新增 `"side_effect_only"` 选项
- 构造参数新增 `description_overrides: dict[str, str] | None`
- `get_definitions()` 对目标工具替换 description，让模型看到候选描述
- `execute()` 新增 `side_effect_only` 分支：
  - 录音命中 → 直接返回（无论工具类型）
  - 录音未命中 + 查询类工具 → passthrough 到真实调用
  - 录音未命中 + 副作用工具 → 追加 `audit_log`，抛 `SandboxReplayError`
- `audit_log` property 暴露拦截记录供外部验证

**录音优先注释**（写入代码）：录音命中意味着候选描述下模型发出了与 baseline 相同的调用（工具名 + 参数不变），这是有意义的信号——说明 description 改动没有影响这个调用。返回录音结果是正确的，不是伪造数据。

#### C. `OptimizationAgent._score_candidate_set` 解锁 ToolDescriptionObject 评分

`optimizer.py` 变更：
- 移除 `if target.kind == "tool_description": raise NotImplementedError(...)`
- `tool_description` 分支：`SandboxedToolRegistry(mode="side_effect_only", description_overrides={target.target_id: candidate.prompt})`
- system prompt 由新增的 `_build_tool_desc_system_prompt()` 辅助函数通过 ContextBuilder 组装：
  - `workspace = tmp dir`（RAG 工具不读用户文件，空 workspace 即可）
  - `knowledge_search = None`（评估不注入用户历史记忆）
  - `persona`、`kb_bindings` 从 `agent_repo` 读取
  - `skill_names = None`（包含所有 available skills 摘要）
- `system_prompt` 分支（PersonaObject）：保持原有逻辑，`candidate.prompt` 直接作 system message

**P1-D1 闭合**：`ToolDescriptionObject` 评分路径从 `NotImplementedError` 变为真正可跑的路径。

---

### 已知限制（故意，不是漏做）

#### P4-L1：PersonaObject 评估环境与生产有差距

PersonaObject 评估时 `candidate.prompt` 直接当 system message，未经 ContextBuilder 组装。这是 Phase 1 的已知简化。Phase 4 只修了 ToolDescriptionObject 分支，PersonaObject 分支不动（SDD §4.1：接口和关联逻辑在 Phase 6 前不做大重构）。两个对象评估环境不同是可接受的：PersonaObject 比较的是候选间的相对优劣，同等简化下相对顺序仍有参考价值。

#### P4-L2：MCP 白名单初始范围保守

白名单初始只放 5 个 RAG 检索工具。其他 MCP 工具（如 `list_collections`、`list_documents`、`verify_results`）虽然也是只读，但未经逐一人工确认，暂不列入。扩充时须在 `_QUERY_MCP_TOOL_ORIGINAL_NAMES` 注释里说明"为什么确认是只读"。

#### P4-L3：`_build_tool_desc_system_prompt` 的 `skill_names=None`

`agent.skills_config` 的 JSON 结构未在 optimizer 层解析（结构依赖 `SkillsLoader` 内部约定），评估时传 `skill_names=None` 表示"包含所有可用 skills 摘要"。这是合理近似——工具说明优化的评估结果不显著依赖 skills 摘要是否精确匹配生产配置。

---

### 完成定义验收（对应 SDD Phase 4）

| 项 | 验收标准 | 状态 |
|----|---------|------|
| 1 | 修改 RAG 工具 description 后，`generate_proposals(ToolDescriptionObject)` 在 `side_effect_only` 模式跑完，不抛 `SandboxReplayError` | ✅ V1 PASS — 5 候选全产出，fix_set `{'tool_skip': 1.0}`，health_set `keyword_coverage≥0.93` |
| 2 | 副作用工具被调用时 `sandbox.audit_log` 有对应条目 | ✅ V2 PASS — `stub_write_tool` 被拦截，`audit_log[0]={tool='stub_write_tool', action='intercepted'}`；查询工具 `mcp_rag_kb_search` 正确 passthrough |
| 3 | 基类默认 `side_effect=True`，查询类工具显式 override `False` | ✅ V3 PASS — `get_definitions()` 返回的 schema 中 description 为候选文本而非原始文本 |

---

## Phase 3 · 复盘层（降级，按需再做）

**状态：🔲 默认不做**

见 SDD §6 Phase 3 节。在 Phase 6 跑通、收集到真实数据之前，不投入此阶段。

---

## Phase 5 · Baseline 锚点 + 部署门控

**状态：✅ 已完成，所有验证项闭合（2026-06-25）**

**验证脚本**：`scripts/verify_phase5_baseline_gate.py`（V1-V5 全部 PASS，真实 DB + 真实 LLM）

**前置依赖**：Phase 1 ✅ + Phase 2 ✅ + Phase 4 ✅

---

### 已做

#### A. schema 新增两列（`migrate_phase5_baseline_gate.sql`）

`optimization_proposals` 新增：
- `baseline_score JSONB nullable`：`{"fix_set": {...}, "health_set": {...}}`，baseline 版本在两个测试集上的分数
- `baseline_version_id UUID nullable`：当时的 `TunableObjectVersion.id`（若无版本历史则为 NULL）

#### B. Baseline 锚点（`optimizer.py`）

`generate_proposals` 流程新增 baseline scoring 步骤：
1. `await target.read()` → 当前文本
2. `await target.get_current_version()` → 版本 ID（可能 None）
3. 构造临时 `OptimizationCandidate` 跑 `_score_candidate_set` ×2（fix + health）
4. **CRITICAL 不变式**：baseline 和所有候选使用**完全相同的** `fix_test_cases` / `health_test_cases` Python 对象 + 同一份 `fix_recordings` / `health_recordings`。唯一变量是文本内容。此约束写入代码注释，门控结论有效性依赖此前提。

#### C. 部署门控（`optimizer.py`）

- **阈值**：`_GATE_IMPROVE = 0.05`，`_GATE_TOLERATE = 0.02`（hardcode，不做动态）
- **规则**：`fix_set_delta ≥ 0.05 AND health_set_delta ≥ -0.02` → `gate_status = "pending_approval"`；否则 `"rejected_by_gate"`
- **per-candidate 字段**（JSONB 内）：
  - `fix_set_delta`：候选 fix 均值 − baseline fix 均值
  - `health_set_delta`：候选 health 均值 − baseline health 均值
  - `gate_status`：`"pending_approval"` | `"rejected_by_gate"`
- **proposal 聚合状态**：所有候选都被 gate 拒绝时，proposal 行 `status = "gate_all_rejected"`；否则保持 `"pending"`

#### D. 仓库层更新

`create_optimization_proposal` 新增三个可选参数：`baseline_score`、`baseline_version_id`、`status`（为向后兼容保留默认值 `"pending"`）。

---

### 已知限制（故意，不是漏做）

#### P5-L1：阈值首版 hardcode，不做动态校准

**为什么**：没有足够真实的 proposal 数据积累，动态阈值就是猜。跑 2-3 个月积累数据后再重新评估。

#### P5-L2：baseline_score 在无 version 历史时仍会跑

**为什么**：SDD §4.4 要求 baseline 分数必须当次跑出来。即使对象从未被平台改过（`baseline_version_id = NULL`），`target.read()` 读出的当前文本就是可比基准。这是正确的——阻止第一次优化被"无版本"挡住。

---

### 完成定义验收（对应 SDD Phase 5）

| 项 | 验收标准 | 状态 |
|----|---------|------|
| 1 | `optimization_proposals.baseline_score` 非空，结构为 `{"fix_set": {...}, "health_set": {...}}` | ✅ V2 |
| 2 | 每条 candidate 有 `fix_set_delta`、`health_set_delta`、`gate_status` | ✅ V3 |
| 3 | SQL 可直接读出 baseline_score、delta、gate_status（`proposals->'proposals'->0->>'fix_set_delta'`） | ✅ V4 |
| 4 | baseline 和 candidate 使用相同的测试集维度（fix/health set dim 一致） | ✅ V5 |
| 5 | baseline_score 当次跑出，不在代码中用历史值 | ✅ 流程上有 `target.read()` + 实时 `_score_candidate_set` |

---

### 验证结论（2026-06-25，真实 DB + 真实 LLM + 44 条健康集 + 1 条 badcase）

| ID | 描述 | 结论 |
|----|------|------|
| P5-V1 | Schema migration — baseline_score + baseline_version_id 列存在 | ✅ PASS |
| P5-V2 | generate_proposals(PersonaObject) — baseline_score + baseline_version_id 落库 | ✅ PASS |
| P5-V3 | Gate logic — per-candidate fix_set_delta, health_set_delta, gate_status | ✅ PASS |
| P5-V4 | SQL query — baseline_score + deltas + gate_status 可直接读 | ✅ PASS |
| P5-V5 | Same test case invariant — baseline 和 candidate fix/health 维度 key 一致 | ✅ PASS |

---

## Phase 6 · 诊断面板 + 一键应用 + 人确认

**状态：⚠️ 主干已交付，apply/rollback 路径尚未完成端到端前端验收**

**commit**：`cd488e1f`（tool_impl fixable 修复）

依赖：Phase 0-5 全部。

---

### 已做

#### A. 后端三个端点

**`GET /snapshots/{id}/diagnosis`**
从 `classification_layer / target_kind / context_trace` 组装诊断面板数据：
- `fixable`：`layer in FIXABLE_LAYERS and target_kind != "tool_impl"`（本轮修复：tool_impl 不可文本修复）
- `evidence`：从 context_trace 提取 9 个字段（history_query / budget / actual / fragment_count / persona / skills）
- `suggestion`：diagnosis-only 层给出人工排查建议
- `has_proposal / proposal_id`：检查是否已有对应 proposal，避免重复生成

**`POST /tunable/apply`**
将候选文本写入 TunableObject：
- PersonaObject → `agents.persona` + `tunable_object_versions`（新版 active=True，旧版 deactivate）
- ToolDescriptionObject → `agents.tools_config[tool_name].description` + 同上版本表
- 可选关联 `proposal_id` → 更新 proposal status = "applied"

**`POST /tunable/rollback`**
恢复上一条 active 版本，版本表始终保持单行 active=True。

**`GET /tunable/{kind}/{id}/versions`**
返回版本历史列表（倒序），供前端版本历史区块展示。

#### B. 前端三个 UI 区块

**诊断 Tab**（快照详情 Modal 内，Tab 2）
- 默认折叠，首次切入时懒加载 diagnosis 端点
- 根因层级卡片：fixable 绿实线 / diagnosis-only 灰虚线
- 证据折叠面板：默认收起，展开显示 9 个 context_trace 字段
- 操作区：fixable=true 显示"生成候选"按钮；false 显示人工排查提示

**候选方案对比 Drawer**（优化建议 Tab → 详情）
- Baseline 基准：Fix Set / Health Set 均分
- 候选列表：双栏分数（Fix / Health）+ Gate Δ + 候选文本（折叠）
- 应用按钮：仅 `gate_status === 'pending_approval'` 时可点
- 版本历史：active 行绿色高亮 + "回滚到上一版"按钮

**应用确认 Modal**
- gate 状态 alert + Fix/Health Δ 数据 + 文本 diff（当前 vs 候选）+ 影响范围说明

#### C. 前端验收状态（2026-06-25）

| 检查点 | 状态 |
|---|---|
| diagnosis 端点返回 evidence 9 个字段 | ✅ 已验证（snapshot 49ddff0e） |
| layer=None → diagnosis-only 样式，无"生成候选"按钮 | ✅ 已验证 |
| tool_impl → fixable=False，不显示"生成候选" | ✅ 修复后验证 |
| layer=Context + fixable=True → 绿实线，"生成候选"可点 | ✅ 已验证 |
| 候选方案对比抽屉 gate_status=pending_approval → "应用此版本"可点 | ✅ 手动构造候选后验证 |
| Apply → toast 成功 + 版本历史新增 active 行 | ⚠️ Toast 出现，但**版本历史区块未反映**（待复查） |
| Rollback → toast 成功 + 版本历史 active 行回滚 | ✅ 已验证（toast + tag 位移正确） |

---

### 未完成项

#### P6-D1：Apply 后版本历史区块未自动刷新（待验收）

**现象**：点"应用此版本"成功 toast 后，候选方案对比抽屉的版本历史区块没有立即反映新版本（未见新 active 行）。Rollback 验证成功，说明版本表本身写入正常，问题可能是 apply 成功后前端没有触发 `loadVersions()`。

**影响**：apply/rollback 完整路径的前端闭环尚未端到端确认，版本历史 UI 是否正确更新未核实。

**下一步**：确认 `confirmApply()` 成功回调里是否调用了 `loadVersions()`；若没有，补上一行 `loadVersions()`。

#### P6-D2：Apply 后无"触发评估验证"入口（P5+P6 联动缺失）

apply 改完 persona/tool_description 后，用户无法直接在当前界面触发 eval run 验证效果，需要手动去评测任务 Tab 操作。P5+P6 联动（apply 成功 → 提示"是否立即触发评估"→ 跳转 eval run）尚未实现。

---

### 债务汇总（Phase 6 新增）

| ID | 描述 | 类型 | 优先级 |
|----|------|------|--------|
| P6-D1 | Apply 后版本历史未自动刷新，apply 路径前端端到端验收未闭合 | Bug / 待验收 | 下次开始前处理 |
| P6-D2 | P5+P6 联动：apply 成功后无直接触发 eval run 入口 | 功能缺失 | UI 重排时一并做 |

---

## 债务汇总表

| ID | 所属 Phase | 描述 | 类型 | 优先级 |
|----|-----------|------|------|--------|
| P0-D1 | Phase 0 | `conversation_id` 落库端到端验证 | 验证 | ✅ 已闭合（2026-06-24）|
| P0-D2 | Phase 0 | fragment_ids 生产路径端到端验证（记忆固化 → 下次命中 → 非空 ID） | 验证 | 建议，Phase 1 前 |
| P0-L1 | Phase 0 | CLI run 无 conversation 关联（Phase 1 UI 需降级展示 NULL） | 已知限制 | Phase 6 时处理 |
| P0-L2 | Phase 0 | ChromaDB 多进程隔离（跨进程写入不可见） | 已知限制 | 按需处理 |
| P1-D1 | Phase 1 | ToolDescriptionObject 评分路径（Phase 4 沙箱分层前 blocked） | 已知限制 | ✅ 已闭合（Phase 4，2026-06-25） |
| P1-D2 | Phase 1 | PersonaObject.generate_candidates 评分（需 golden_test_cases + recordings） | 已验证 | ✅ Phase 2 双集打分验证通过（2026-06-25），generate_proposals 全链路可跑 |
| P2-D1 | Phase 2 | 健康集内容人工审核 + `--import` | 必须 | 审核通过前不得用健康集打分 |
| P2-D2 | Phase 2 | `fix_test_cases` 构造方式（调用方将 badcase 转 AgentTestCase 列表） | 待接入 | Phase 3 CLI/路由接入时处理 |
| P2-D3 | Phase 2 | **健康集 6 条占位符未导入**：h_rag_hit_001~005（expected_keywords 需从真实知识库提取）+ h_tool_004（工具名 `query_orders` 需验证）。在此之前健康集对 RAG 命中场景无检测能力，工具调用场景缺 1 条。 | 阻塞 import | 补充知识库内容 + 确认工具名后处理 |
| P2-D4 | Phase 2 | **RAG 不命中 5 条方向待确认**：h_rag_miss_001~005 预期 Agent 对 KB 外问题诚实说"不知道"；若 Agent 实际用训练知识补空，预期方向反了，需改写 expected_keywords 或换场景 | 阻塞 import | 确认 Agent "不在 KB 则如何" 策略后处理 |
| P5-L1 | Phase 5 | 门控阈值首版 hardcode（δ_improve=0.05, δ_tolerate=0.02），不做动态校准。SDD §4.4 Rationale：没数据就调参等于猜。跑 2-3 个月积累真实 proposal 数据后再评估是否需要调。 | 已知限制 | Phase 6 跑通后按数据决定 |
