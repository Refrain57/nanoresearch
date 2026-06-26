# A1 — 工具层 Harness 深化 Design

**日期**: 2026-06-26
**状态**: Brainstorm 输出,待 writing-plans 转 implementation plan
**范围标签**: A1(工具层做透);A2 / D1 留待后续轮次

---

## 1. 背景与主轴

项目目标(用户原始措辞):

> 把 Agent 的错误从"模型问题"转化成"系统问题",让 Agent 的能力沉淀在系统里,而不是沉淀在 Prompt 里。通过上下文、工具、编排、记忆、评估、恢复六个层面持续优化。

调研后的主轴(重新表述,本 spec 据此推进):

> 系统的"闭环没合上"只是表象;真正的病灶是两根正交的轴同时弱:**闭合度**(detect→fix→verify→**monitor/rollback**)在 monitor/rollback 端断开;**深度**(每个 axis 自身是不是真 harness)在每个轴上都只做到了"表面"。仅补闭合度,得到的是"能可靠验证浅层改动的闭环"——浅层改动本就改不动什么。

A1 不是"做完工具层就结束",而是**在工具层这一格上把两根轴同时做到位**,作为后续五个轴的样板间和方法论来源。

系统的职责是**产出值得人信任的判断和监控**;触发 / apply / rollback 的最终动作由人执行。自动触发机制(无论何种形式)不在本 spec 范围内。

---

## 2. 范围

### 2.1 In-Scope(A1)

- 工具层从"文字 harness"升级到"行为级 harness"
- 满足 §3 判据清单中除 D1 之外的全部 11 条
- 产出 §9 附录:"造一个轴底座的方法论清单",供 A2 复用

### 2.2 Out-of-Scope

- **A2**(其他五个轴 — 上下文 / 编排 / 记忆 / 恢复 / 评估自身 — 的底座):A1 完成、§9 方法论清单成形之后启动
- **D1**(候选生成本身从历史中学习):涉及对历史候选聚类,工程量大;A1 接受"候选仍由 LLM 生成"的现状
- **cron 自动触发**:不做
- **评估层自身的 meta-eval**:递归问题,留给单独讨论 — 但 §3 的 C3 判据其实也应该最终递归适用于评估层

---

## 3. C 阶段产出 — Harness 判据清单

每条判据标注**来源诉求**(从用户原始 framing 推导而来,不是从 sandbox 反推)和**验证方法**(一句话)。

### Layer A — 可观察可重现(实验的最低门槛)

**A1. 事件可快照**
轴在一次 run 中"做了什么"能被序列化成结构化对象(不是日志字符串),粒度细到能区分"换了策略 vs 没换"。
- 来源:"不要表面工作" — 没结构化对象就无法做差分对比
- 验证:能 dump 一份 JSON 描述该轴这一 run 中所有决策的输入/输出

**A2. 同输入可重放**
给同一份输入快照,该轴的内部行为可被确定性复现(或所有不确定因素被显式固定)。
- 来源:detect→fix→verify 链条 — 没重放就无法做对照实验
- 验证:对同一份输入快照跑两次,产出相同的事件快照

**A3. 隔离性**
评测某轴时,其他五个轴的行为通过快照/录像固定,不串扰。
- 来源:六个 axis 互不污染
- 验证:在 candidate 评测中,其他轴的事件序列与 baseline 完全一致

### Layer B — 实验有意义(候选 + 信号)

**B1. 候选有非文字形态**
候选改动可以是行为级(routing / schema / 阈值 / 策略),不只是描述文字。
- 来源:"沉淀在系统里不在 Prompt 里"
- 验证:该轴至少能生成 2 类非文字候选

**B2. 信号可测量且噪声可刻画**
有一个跟该轴**内在能力**相关的度量,且方差/置信区间可估计。
- 来源:"不要表面工作" + 测试地基
- 验证:对同一候选重复评测 N 次得到 σ;比较候选差异时使用 σ 加权

**B3. 反事实对照成立**
baseline 与 candidate 在同一组 cases 上做 paired 比较,差异主要归因于该轴的改动。
- 来源:dual-set 不变量真正被执行
- 验证:fix_set 与 health_set 都非空、都跑通、deltas 真有统计意义

**B4. case 集自身可被审计**
fix_set / health_set 不只是 fixture,要能回答"覆盖了什么 / 漏了什么 / 谁加的 / 为什么加"。
- 来源:`test_optimizer.py:143` 传空 golden_test_cases 绕过 `optimizer.py:103-107` 不变量这个教训
- 验证:每条 case 元数据中含 origin_badcase_id / target_dimension / added_at / added_by / coverage_tags

**B5. 行为级候选必须通过 execution 健全性 pre-gate**
任何动 schema / retry_policy / routing 的候选,在进入 trial / 接生产流量之前,先在 witness 历史录音上重放"会不会让工具调用失败"(参数 JSON Schema 校验、required/类型/enum 兼容性、retry 不变量等)。pre-gate 不通过的候选直接 reject,不进 dual-set scoring,不进 trial。
- 来源:用户硬约束 — 行为级坏配置(尤其 schema 收紧、retry 把 timeout 调没)能让功能失败但语义分数抓不到,SDD 反对自动 apply 的核心理由之一
- 验证:存在 `execution_sanity_check(candidate, witness_recordings)`,对 schema 类候选能 deterministic 判出"参数不再符合"的录音条目数与比例;比例超阈值 → reject

### Layer C — 实验可以闭合成生产改动

**C1. Apply 粒度 = 单一轴**
候选应用时只动这个轴,其他轴不被波及。
- 来源:六个 axis 互不污染
- 验证:apply 前后,其他轴的 hash/version 不变

**C2. Rollback 原子且不需要解读**
回滚不是"再 apply 一次旧文本",是"系统级别的撤销",不留半应用状态。
- 来源:detect→fix→verify→monitor 闭环
- 验证:rollback 后系统状态 = apply 前状态(逐字段比对);异常中断留下的半状态可被 reconcile 任务检测和清理

**C3. 应用后监控(meta-eval)**
apply 之后,系统在 health_set 之外的**真实流量样本**(witness_set)上观察该候选,检测未被打分维度的退化。
- 来源:"警惕自我修改的副作用" + "谁评估飞轮本身"
- 验证:存在 production_witness_set,每个生产 run 按比例采样进入;trial 状态机存在;trial 期间持续比较 candidate vs baseline 在 witness 上的分数

**C4. 监控信号能反向触发回滚 propose**
C3 的信号必须能 mark 候选为 regressing,接到 rollback。
- 来源:同 C3
- 验证:trial status 自动转 regressing 时,系统主动创建 rollback_proposal(含证据),人 click 一下即可走 C2 路径回滚

### Layer D — 真正"沉淀在系统里"(A1 显式排除)

**D1. 候选生成本身从历史中学习** —— 留给后续轮次。

---

## 4. 现状诊断 — 工具层逐条对照

| 判据 | 状态 | 证据 / 缺口 |
|---|---|---|
| A1 | ✓ | `sandbox.py:97-122` tool call → result 的 (key, value) 记录;`_normalize_params` 保证 key 稳定 |
| A2 | ✓ | `sandbox.py:100-105` replay 模式;`from_recordings_json` 反序列化 |
| A3 | ✓ | `description_overrides` (`sandbox.py:51-88`) 只动工具层视图 |
| B1 | **✗** | `TunableTextObject` (`tunable.py:69-115`) 显式 text-only(`tunable.py:1-9` 注释);`_llm_generate_candidates` 产出仅有 prompt 文本 |
| B2 | **△** | `_execute_side_effect_only` (`sandbox.py:130-131`) "key 命中即等效"假设脆 — 候选参数微调即 key miss;噪声未刻画 |
| B3 | ✓ | `optimizer.py:140-146` 双 baseline 重新打分;`optimizer.py:152-170` dual-set scoring 路径完整 |
| B4 | **✗** | `test_optimizer.py:140-144` 用**过时参数名** `golden_test_cases=[]` 调用 `generate_proposals`;当前签名(`optimizer.py:77-83`)是 `fix_test_cases` + `health_test_cases`。该测试要么 silently broken,要么从未在新签名下运行 — 无论哪种,`optimizer.py:98-107` 的非空不变量都未被 CI 真正验证过。另:`test_score_candidate_raises_for_tool_description` (`:184-194`) 仍 pin 旧 `NotImplementedError` 行为,与 Phase 4 应有的解锁状态矛盾。case 集亦无元数据治理 |
| B5 | **✗** | 无 execution 健全性 pre-gate。当前流程:候选生成 → dual-set scoring → gate → apply。中间没有一道"这个 schema 改完工具还跑不跑得起来"的离线校验。B1 引入 schema/retry 候选后,gate 的语义分数对"调用直接失败"的退化基本盲(失败的 run 在评测里要么被 retry 救回要么得低分,但跟"分布上略差"区分不开) |
| C1 | ✓ | `ToolDescriptionObject.apply` (`tunable.py:338-361`) 只改 `tools_config` 中目标工具 |
| C2 | **△** | `rollback` (`tunable.py:367-371`) 等于"再 apply 一次旧 content";对 schema/routing/retry 这类多字段联动改动不够 |
| C3 | **✗** | 无 production_witness_set,无 trial 监控 |
| C4 | **✗** | 无自动 mark regressing 路径 |
| D1 | ✗(显式排除) | 候选全靠 `_llm_generate_candidates` (`tunable.py:142-173`) |

**重要旁注**: `tunable.py:23` 和 `tunable.py:286-288` 还在写"Phase 4 blocked / ToolDescriptionObject scoring blocked",但 `sandbox.py` 的 `side_effect_only` 机制实际上已就位 — **注释/文档与代码事实不一致**。A1 验收要求清理。这本身就是"表面工作"的活样本。

---

## 5. A1 交付物 — 逐判据要做的事

**实施顺序**(每一节是后一节的前置):

> **B4** → **B2** → **B1** → **B5** → **C2** → **C3** → **C4**

理由:B4 没修(测试 mock 还在绕过不变量),其他验证都是假的;B2 不就位,B1 引入的非文字候选评测出的差异分不清是改进还是噪声;B5 紧跟 B1 — B1 一引入行为级候选就必须立刻把 execution 健全性 pre-gate 接上,否则坏 schema 直接进 dual-set scoring → trial → 生产,语义分数对调用失败基本盲;C2 不升级,B1 引入的多字段候选 rollback 不可靠;C3/C4 是最上层,必须建在前五层之上。

### 5.1 B4 — case 集治理(地基)

**动机**: 修测试 mock 绕过不变量的腐烂;给 case 集装元数据,让后续验证有可信底座。

**改什么**:
- `storage/models/`:扩展 `test_case` 表(或现有等价表)字段:
  - `origin_badcase_id UUID NULL`(case 是否来源于真实 badcase)
  - `target_dimension TEXT NOT NULL`(覆盖维度,如 "tool_schema_correctness" / "param_validation")
  - `added_at TIMESTAMP NOT NULL`
  - `added_by TEXT NOT NULL`(可以是 "system" 或具体用户)
  - `coverage_tags TEXT[]`(标签数组)
- 新 endpoint `GET /cases/audit`:返回每个 case 的元数据 + 维度覆盖直方图 + 孤立 case(无 origin_badcase 引用)列表
- `tests/eval/test_optimizer.py`:
  - **重写**(不是补丁)`test_generate_proposals_*` 一组测试:使用当前签名 `fix_test_cases` + `health_test_cases`(非 `golden_test_cases`);传入非空真实 case
  - 检视 `test_score_candidate_raises_for_tool_description` (`:184-194`) 等仍 pin 旧 `NotImplementedError` 行为的测试 — Phase 4 若已解锁则应改为期望正常返回;若未解锁则需要补完 Phase 4 工作(这部分若发现属于另一个 PR 范围,标注为前置依赖)
  - 新增 integration test:不靠 mock,用 in-memory repo + fake LLM 跑完整 `generate_proposals` 路径,验证非空不变量真正生效
- CI gate:case 集变更 PR 必须同步元数据(校验 origin_badcase_id / target_dimension 非空)

**验收**:
- 现有 case 集 metadata 100% 回填
- `/cases/audit` 能列出孤立 case
- `test_optimizer.py::test_generate_proposals_full_path` 在 CI 中跑通,不依赖 mock 跳过 `optimizer.py:103-107` 的非空校验

### 5.2 B2 — 信号噪声刻画

**动机**: 当前 gate 用硬阈值 `_GATE_IMPROVE = 0.05` (`optimizer.py:54`);非文字候选引入后,5% delta 可能在噪声内,也可能是真改进,无法区分。

**改什么**:
- `optimizer.py::_score_candidate_set` 增加 repeat 路径:同一候选在同一 case 上跑 N=3 次,产出 `ScoreSample(mean, std, n)`
- gate 判定改为:`fix_set_delta_mean ≥ k · σ_combined` 且 `health_set_delta_mean ≥ -k · σ_combined`(k 取 1.96 给 95% 单侧)
- `sandbox.py::_execute_side_effect_only`:key miss 时尝试"参数标准化二次匹配"(只忽略字符串空格、引号转义、键序);命中算 `fuzzy_match`,记入 `audit_log` 的 `fuzzy_match_ratio` 字段
- proposal 持久化中增加 `score_sample`(含 mean/std/n)和 `fuzzy_match_ratio` 字段

**接口**:
```python
@dataclass
class ScoreSample:
    mean: float
    std: float
    n: int

async def _score_candidate_set(...) -> dict[str, ScoreSample]: ...
```

**验收**:
- 同一候选跑 3 次,σ < 0.05(在当前金 case 集上)
- gate 在 σ 加权下能区分"显著改进"与"在噪声内"(在合成测试中:故意构造一个"分数在 baseline 噪声内"的候选,gate 应该拒)
- `audit_log` 中 `fuzzy_match_ratio` 可查;若 ratio > 30%,proposal 状态应 mark `signal_unreliable`,需人审

### 5.3 B1 — 非文字候选(核心去 prompt 化)

**动机**: 工具层"优化"目前 ≡ 改 description 文字,本质仍是 prompt engineering。要让能力沉淀在系统里,候选必须能改变行为。

**改什么**:

- **接口**: `tunable.py`
  - `TunableTextObject` 重命名 `TunableObject`,`read/apply` 签名从 `str` 改为 `Any`
  - `OptimizationCandidate.prompt` 重命名 `payload: Any`,旧字段 `prompt` 保留为 property 兼容现有 PersonaObject(渐进迁移)
  - PersonaObject 仍存在,kind 仍为 `"system_prompt"`。它的 payload 仍是文字,但**合理性边界值得说清楚**——

`agents.persona` 是 DB 列,存的是 "这个 agent 的角色 / 行为基调"。`ContextBuilder` 在每次装配 system prompt 时,把它跟 SOUL.md 结构、技能摘要、KB 绑定、动态后缀**拼成最终 prompt**(见 `tunable.py:11-17` 的 SCOPE BOUNDARY 注释)。换句话说,**persona 不是 prompt 本身,它是 prompt 的一个**有意暴露给运营**的可配置插槽**。

本 spec 反对的 "prompt engineering" 指 "**编辑某段文字以哄模型听话**"——典型代表是反复改 tool description 的字眼,让模型更愿意调用某个工具。这是把改进沉淀在**对当前模型权重的怪癖**上,模型一换就废。

而 persona 是 "**人为决定这个 agent 应该是什么角色**"——它是 Agent A 与 Agent B 的合法差异化点,改 persona 是在改产品定位,不是在哄模型。

两者用同一个 `TunableObject` 抽象,但**语义不同**。所以保留 PersonaObject 为文字类不矛盾于本 spec "去 prompt 化" 的诉求
  - 新增两个 concrete class:
    - `ToolSchemaObject(TunableObject)`:kind = `"tool_schema"`;payload 是 `dict`(JSON Schema 片段);apply 到 `tools_config[i].parameters`
    - `ToolRetryPolicyObject(TunableObject)`:kind = `"tool_retry_policy"`;payload 是 `dict`(timeout / max_retries / backoff);apply 到 `tools_config[i].retry_policy`(新字段)
- **sandbox**: 新增 override 维度
  - `schema_overrides: dict[tool_name, dict]`:在 `get_definitions()` 应用,与 `description_overrides` 并列
  - `retry_policy_overrides: dict[tool_name, dict]`:`execute()` 包装层应用
- **optimizer**: `_score_candidate_set` 中删除 `NotImplementedError` catch(`optimizer.py:161-168`),替换为正常 dispatch — 对 `tool_schema` / `tool_retry_policy` 真正跑 dual-set scoring,不再 persist 空 scores
- **candidate generation**: 为新 kind 写 prompt 模板,产出 JSON-schema 候选(LLM 仍是来源,但产出物是结构化 schema 而非自由文本)
- **lint guard**: `ToolSchemaObject.generate_candidates` 后置一个 lint:
  - 不允许改参数类型(string → int 之类)
  - enum 只能增不能删
  - required 字段不能转 optional(反过来可以)
  - 违反 lint 的候选直接丢弃,记 warn

**验收**:
- `ToolSchemaObject` 能为一个真实工具生成 ≥2 个候选(如:某参数从 optional 改 required;某参数加 enum 约束)
- 候选通过 dual-set scoring 产出 score sample(B2 路径),gate 判定可接受
- 接受的候选 apply 后,生产 agent 调该工具时 LLM 看到的 schema 是新的,实际调用按新 schema 校验
- 已有 PersonaObject / ToolDescriptionObject 仍正常工作(回归测试覆盖)

### 5.4 C2 — 状态级 rollback

**动机**: 文字 rollback 等价于"再 apply 一次旧 content"对单字段够用;但 B1 引入 schema/retry 这类**多字段联动**的改动后,"再 apply 旧字段"不能保证整体状态回到 apply 前(中间可能有其他变更穿插)。

**改什么**:
- `TunableObject` 增加两个抽象方法:
  ```python
  async def snapshot_state(self) -> dict: ...
  async def restore_state(self, snapshot: dict) -> None: ...
  ```
- `apply` 流程改为:`pre_snapshot = snapshot_state()` → 写 version 时存 `state_snapshot` 字段 → 写新内容
- `rollback` 流程改为:从 version 读 `state_snapshot` → `restore_state(snapshot)`(整体写回,而非 `apply(old_content)`)
- `agent_eval_repo`:`tunable_version` 表新增 `state_snapshot JSONB`
- reconcile 任务:扫描 "apply_in_progress" 标记,检测半应用状态(apply 中途崩溃),自动 restore 到 pre_snapshot

**验收**:
- `ToolSchemaObject.apply` 修改 `tools_config[i].parameters`,中间穿插一次其他字段变更,rollback 后 `tools_config[i].parameters` 恢复到 apply 前的精确 JSON
- 模拟 apply 中途异常(在写 DB 前抛错),reconcile 任务能检测并清理半状态
- PersonaObject / ToolDescriptionObject 的 rollback 行为向后兼容(它们的 snapshot 等于"旧 content",restore 等于"写回旧 content")

### 5.5 C3 — 应用后监控(meta-eval)+ strict_replay 模式

**动机**: gate 通过的候选可能在 health_set **没有覆盖到的维度**上让系统变糟,目前没人在看。这是用户"警惕自我修改副作用"的核心担忧。

**改什么**:

- **production_witness_set 采样**:
  - 每个生产 agent run 按 `WITNESS_SAMPLE_RATE`(默认 0.05)概率被采样,采样的 run 元数据(input、tool_call_chain、final_response、score)进入 `production_witness` 表
  - 采样不阻塞生产,异步入库
  - PII 处理见 §8 未决问题 3
- **trial 状态机**:
  - 候选 apply 后,version 的 `lifecycle_status` 字段从 `"applied"` 改为 `"trialing"`
  - **触发机制:只走 hook,不引入定时器 / 后台扫描 / cron**。具体两条入口:
    - (a) **生产 hook**:`production_witness` 表插入新样本时,业务事务内只做一件事 — `enqueue ARQ task: trial_check(version_id, witness_id)`;真正的 trial_check 在 worker 里异步执行(fire-and-forget),不阻塞生产 run。若 agent 无 `trialing` 候选,enqueue 前过滤
    - (b) **admin 端点**:`POST /trial/{version_id}/check` 作为人工入口(用于 oncall debug 或 hook 失效兜底),内部也走同一条 enqueue 路径
  - `trial_check` 执行(ARQ worker 内):
    - 拉最近 `WITNESS_TRIAL_WINDOW`(默认 24h 或 50 samples,取最严)的 witness 样本
    - 对每条样本,用同一份输入分别在 baseline 和 candidate 上重放,**强制走 `strict_replay` 模式**(见下),产出对比分数和每次 replay 的 `divergence_rate`
    - 计算 `witness_delta_mean ± σ` 和 `divergence_rate_mean`
    - 若 `divergence_rate_mean > _DIVERGENCE_REJECT`(默认 0.3) → trial 终止判为 `signal_unreliable`,不允许转 `stable`,自动 propose rollback 走人工 click(理由:对比基础已经被环境漂移污染,看似"持平"或"略好"都不可信)
    - 若 `witness_delta_mean ≥ -k·σ` 且 divergence 可接受 → `stable`
    - 若 `witness_delta_mean < -k·σ` 且 divergence 可接受 → `regressing`
    - 样本不足且未超时 → 继续 `trialing`
- **strict_replay 模式**(`sandbox.py` 新增):
  - 现有 `side_effect_only` 模式在 query 工具录音 miss 时 passthrough 真调用(`sandbox.py:156-161`);用于 trial 对比时,这等于"baseline 和 candidate 看到的世界不一样",对比结论不可信
  - 新增 `strict_replay`:任何工具调用 — query 或 side-effect — 录音 miss 都不 passthrough,而是返回结构化"录音缺失"标记并把这次 miss 计入 `divergence_log`
  - `divergence_rate` = miss 次数 / 总调用次数,在 trial_check 的 evidence 中持久化
  - 强制规则:trial 对比(baseline replay / candidate replay)只能用 `strict_replay`,不允许用 `side_effect_only`
  - **诚实登记的残余盲区**:`divergence_rate > _DIVERGENCE_REJECT` 触发的"signal_unreliable"对**文本类**(persona 大改、tool_description 颠覆性改写)和**行为级**(schema/routing 大改)候选**都适用** — 候选改动越大,divergence 越高,这道 gate 越容易把它判成 signal_unreliable。换句话说,**A1 的自动监测对"激进改动"的候选无能为力,这类候选不能依赖 C3 / C4,必须由人在 apply 前评估或事后查 audit 决策**。这是有意为之的保守边界,不是 bug
- **接口**:
  ```python
  @dataclass
  class TrialState:
      candidate_version_id: str
      trial_started_at: datetime
      witness_samples_seen: int
      witness_delta: ScoreSample | None
      status: Literal["trialing", "stable", "regressing"]
  ```

**验收**:
- 生产 run 按比例进入 witness_set,可通过 `GET /witness/recent` 查询
- 对一个 trialing 候选触发 `trial_check`,能产出 trial state
- 合成测试:故意构造一个会让 witness_delta 跌破阈值的 candidate,`trial_check` 后 status = `regressing`

### 5.6 C4 — 监控信号 → 自动 propose rollback

**动机**: C3 的检测信号不能只是 dashboard;必须能闭合到 rollback 路径。但**不自动执行** rollback —— 这是用户"警惕自我修改"边界的硬约束。

**改什么**:
- 当 trial status 转 `regressing` 时,自动创建 OptimizationProposal,`category` = `"rollback:{candidate_version_id}"`,内容包含:
  - 原 candidate version reference
  - witness_delta 证据(mean / σ / N / 退化最严重的维度 top-3)
  - 建议动作:"rollback to baseline"
- 这个 proposal 仍要求人 click(走现有 `/optimize/{id}` patch 路径)
- 通知:`regressing` 状态推送给人。实现优先级:(a) 复用现有通知通道(如有);(b) 兜底:在 admin dashboard 暴露 `regressing_proposals_count` 指标 + stderr 日志(且日志按 ERROR 级别打),保证 oncall 能看到;(c) 邮件/告警渠道留待 implementation plan 决策。**不允许只 log INFO 级别就算交付**(否则等同于 C4 未达成)

**接口**:
```python
class TrialMonitor:
    async def check_regressing(self, candidate_version_id: str) -> bool: ...
    async def propose_rollback(
        self, candidate_version_id: str, evidence: dict
    ) -> str:  # 返回 proposal_id
        ...
```

**验收**:
- 模拟 witness_delta 跌破阈值,系统自动创建 `rollback:*` 类型 proposal
- proposal payload 包含 evidence(对比数据 + 退化维度)
- 人 click 后走 C2 路径执行 rollback,执行完毕 candidate version 的 `lifecycle_status` 转 `rolled_back`

### 5.7 B5 — Execution 健全性 pre-gate

> 实施次序上 B5 在 B1 之后、C2 之前(见 §5 顺序);章节编号靠后是因为这是整个 A1 安全地基的下半段(上半段是 §5.5 的 strict_replay),放一起方便审计两道防线的措辞。

**动机**: B1 引入 schema/retry 候选后,gate 评测的是**语义分数**,对"工具调用直接失败"的退化基本盲(失败的 run 在 dual-set 里要么被 retry 救回,要么得低分跟"分布上略差一点"混在一起,gate 阈值很难区分)。这是 SDD 反对自动 apply 的核心理由之一,必须正面堵。

**改什么**:
- 新增 `eval/execution_sanity.py`:
  ```python
  @dataclass
  class ExecutionSanityResult:
      total_recordings_checked: int
      failures: list[dict]   # {recording_key, reason, severity}
      failure_rate: float
      has_loosening: bool
      loosening_dimensions: list[str]
  ```
- 检查内容(按候选 kind 分发):
  - `tool_schema`:对 witness_recordings 里每条该工具的调用,把录音参数对**新 schema** 做 JSON Schema validate;不通过 → 计一次 failure(severity = `param_validation_failed`);required 字段从 optional 改 required → failure(severity = `required_tightened`);enum 删值且历史调用用了该值 → failure(severity = `enum_value_removed`)
  - `tool_retry_policy`:timeout 改短于历史调用 P99 时长 → failure(severity = `timeout_too_short`);max_retries 改 0 而历史有 retry 救回的调用 → failure(severity = `retry_disabled_with_history`)
  - `tool_description` / `persona`:不适用(跳过,直接返回 `failure_rate = 0`,但写明 "not applicable")
- gate 接线:`optimizer.generate_proposals` 在 dual-set scoring **之前**先跑 `execution_sanity_check`。`failure_rate > _SANITY_REJECT`(默认 0.05)或出现任何 `severity = required_tightened / enum_value_removed` → 直接拒,proposal 状态 `rejected_by_sanity_gate`,evidence 含失败录音列表(脱敏后),**不进 dual-set scoring**
- 持久化:proposal 表新增 `sanity_check_result JSONB`,即使 pass 也存(供后续审计 / B2 噪声分析回看)

**接口**: 上文 dataclass

**验收**:
- 对"把某 required string 参数改成 number"的合成候选,sanity_check 失败率 = 100%,proposal 状态 `rejected_by_sanity_gate`,不进 scoring
- 对"删一个 enum 值但该值在 witness 历史中出现过"的候选,直接 reject
- 对"timeout 5s → 1s 而历史 P99 = 3s"的候选,reject
- 对 description/persona 候选,sanity_check 返回 "not applicable",proposal 仍正常走 scoring
- 对放宽类候选(`has_loosening = true`),sanity_check 全 pass 但 `has_loosening / loosening_dimensions` 字段被填充并在 proposal 卡片中显眼展示

**显式登记的盲区(B5 放宽方向)**:
B5 的实现机制是"用 witness 历史录音校验新 schema/policy",这只能抓**收紧类**退化(新 schema 比旧 schema 严,历史调用不再通过)。**放宽类**改动 — 比如 required → optional、enum 加值、timeout 调大、新增可选参数 — sanity_check 必然全部 pass,因为旧录音对新 schema 永远兼容。

但放宽是优化里**最常见**的方向(LLM 会主动提议"把这个参数改成可选,模型更容易调用"之类),放宽引入的运行时失败(比如:把 required 字段改 optional 后,下游消费方仍假设字段存在;timeout 调大让上游级联 timeout)B5 一律抓不到。

**这道盲区不在 A1 范围内堵**,显式记录在此并在 §6 风险登记中登记。可能的后续防御方向(留给 A2 或后续轮次决策):
- 在 sanity_check 里加"反向校验"——用 sandbox/staging 跑 candidate schema 下的合成调用,但需要可信的请求生成器,这本身是另一个 harness
- 让放宽类候选**强制经过更长的 trial 窗口**,放弃 sanity pre-gate 的硬性约束,改靠 C3 witness 监控兜底(代价:C3 又被高 divergence 盲区限制,见 §5.5 — **"既放宽又大改"的候选落在两道盲区的交集上,B5 / C3 全部无效,只能纯人工 pre-apply 评估,见 §6 风险登记中标 🔴 的双盲候选类条目**)
- 完全禁止 LLM 提出放宽类候选(过严,放弃合法优化空间)

短期内,A1 接受这个盲区,做法是:**B5 的 evidence 字段显式标注 candidate 是否含放宽类改动**(`has_loosening: bool` + `loosening_dimensions: list[str]`),让人在 click apply 时看得到。这不是技术防御,是知情边界。**绝不在 spec / PR description / proposal 卡片里把 B5 包装成"行为级候选的安全网" — 它只是"收紧方向的安全网"**。

---

## 6. 风险登记

| 风险 | A1 是否引入 | 堵法 |
|---|---|---|
| 非文字候选改动幅度大,gate 阈值不够保守,gate 通过的"改进"实际是噪声 | A1 引入 | B2 把 gate 改 σ 加权;非文字候选首次 apply 强制进 trial(C3),trial 通过才转 stable |
| trial 期信号不足判 regressing,但 candidate 实际没问题(假阳性) | A1 引入 | trial_gate k=1.96 取较宽;regressing 不自动回滚,只 propose;N ≥ `WITNESS_TRIAL_MIN_N`(默认 50)才允许判定 |
| schema 收紧让正常调用失败 | A1 引入 | §5.7 B5 execution_sanity_check 在 witness 历史录音上跑兼容性校验,`failure_rate > _SANITY_REJECT` 或出现 required/enum 收紧直接 reject |
| **行为级"坏配置"(schema/retry)进 trial,语义分数抓不到但工具调用直接失败** | A1 引入 | §5.7 B5 在 dual-set scoring **之前**强制跑 execution_sanity_check;`tool_schema` / `tool_retry_policy` 类候选不通过 pre-gate 直接 `rejected_by_sanity_gate`,不进 scoring、不进 trial |
| **trial 期 baseline / candidate 重放在 query 工具上看到不一样的世界(环境漂移污染对比结论)** | A1 引入 | §5.5 strict_replay 模式:trial 对比禁止使用 `side_effect_only`(query miss passthrough)。所有 replay 走 strict_replay,录音 miss 计入 `divergence_log`;`divergence_rate > _DIVERGENCE_REJECT` 时 trial 判 `signal_unreliable`,不允许转 stable,自动 propose rollback |
| **高 divergence_rate 让 C3 对"激进改动"候选的判断失效(文本类 + 行为级 都中招)** | A1 引入,但**只能登记不能消除** | divergence_rate 超阈值 → `signal_unreliable` + propose rollback;evidence 中存证;**这类候选不能依赖 C3/C4 自动监测,必须由人在 click apply 前评估**;在 proposal 卡片中显眼展示候选的 "改动幅度估计"(diff 行数 / schema 字段变化数 / persona 文本相似度) |
| **B5 对放宽方向退化天然盲(required→optional、enum 加值、timeout 调大都 pass)** | A1 引入,显式不堵 | §5.7 末段"显式登记的盲区"详述。短期措施:`has_loosening / loosening_dimensions` 字段强制在 proposal 卡片中展示;不在任何文档里把 B5 包装成行为级候选的安全网;长期方向(反向 fuzz / 更长 trial 窗口)留 A2 |
| **🔴 双盲候选类 — "既放宽又大改"(放宽 schema/policy + 改动幅度大,如 required→optional 且 diff 行数多 / persona 文本相似度低)** | A1 引入,**系统无任何一层防御** | B5 因为放宽方向必然 pass、C3 因为高 divergence 必然判 `signal_unreliable`,gate / B5 / C3 trial **三道机制全部对此类候选无效**。**唯一防线是 apply 前的纯人工评估**:proposal 卡片必须把 `has_loosening = true` 与"改动幅度估计"两个信号同时显眼展示,平台必须强制人审通道(不允许走"系统已通过 → 直接 click apply"心智);本风险登记在面向使用者(oncall / 决策者)的位置,目的是杜绝"B5 通过 + trial 在跑 = 系统兜着"的虚假信心 |
| C3 witness 采样的离线 dual-set 评测耗 LLM 成本 | A1 引入 | 采样率默认 5% 可配置;witness 离线评测有日预算上限;超出预算暂停 trial(不影响生产) |
| LLM 生成的 schema 候选离谱(改类型 / 删 enum) | A1 引入 | B1 的 lint guard 直接丢弃违规候选,不进 scoring |
| **A1 做完团队认为"系统已经在自我改进了"放下 A2** | 流程风险 | §9 附录"造下一个轴底座的方法论清单"是 A1 验收硬要求;不交付即 A1 未完成 |
| 注释/文档继续腐烂(重蹈 Phase 4 覆辙) | 流程风险 | A1 验收包含:`tunable.py:1-25` 和 `:280-289` 关于 "Phase 4 blocked" 的描述全部清理;PHASE_STATUS.md 同步 |
| ToolSchemaObject 改 schema 影响 LLM 已缓存的旧 schema(prompt cache 失效) | A1 引入 | 见 §8 未决问题 1;暂定 apply 后由 provider 层主动 invalidate session-scoped cache |
| trial 期间人忘 click,regressing 候选长期承载生产流量 | A1 引入 | 见 §8 未决问题 3 |

---

## 7. 验收 — 整个 A1 完成的判据

A1 算交付,当且仅当以下五条全部满足:

1. **§3 的 12 条判据(排除 D1)对工具层每条都能用代码证明满足** — 每条对应一个 acceptance test 或 inspection record
2. **端到端 demo 三条路径全部跑通**:
   - **正向**:从一个真实 badcase 开始,检测 → 分类到 `tool_schema` → 生成至少 2 个非文字候选 → execution_sanity pre-gate 通过(§5.7) → dual-set scoring(含 σ) → gate 通过 → apply → 进入 trial → witness 监控走 strict_replay 模式(§5.5) → trial 通过 → 转 stable。全程**无人工干预** *除*: 最后一次 click apply,以及(若发生) click rollback
   - **被 B5 拦截**:故意构造一个收紧类候选(如 enum 删值、required 收紧),证明在进入 dual-set scoring 之前被 `rejected_by_sanity_gate`,evidence 含失败录音条目
   - **被 strict_replay 拦截**:构造一个改动幅度大的候选(如 persona 颠覆性重写或 schema 大幅 routing 变化),进入 trial 后 `divergence_rate > _DIVERGENCE_REJECT`,trial 判 `signal_unreliable` 而非 stable,自动 propose rollback
3. **CI 中 `test_optimizer.py::test_generate_proposals_full_path` 跑通**,不靠 mock 绕过 `optimizer.py:103-107` 的非空校验
4. **`PHASE_STATUS.md` 与 `tunable.py` 注释更新到当前真实状态** — 关于 "Phase 4 blocked" / "scoring not available for tool_description" 的描述全部清理
5. **§9 附录的"造下一个轴底座的方法论清单"完成** — A2 启动时不需要重新 brainstorm,清单是它的输入

---

## 8. 未决问题

1. **ToolSchemaObject.apply 与 prompt cache 的交互**:apply 修改 schema 后,LLM provider 层(如 OpenAI 兼容)可能仍缓存旧 schema 的 prompt 前缀。需要在 `apply` 完成后主动 invalidate 受影响 agent 的 session-scoped prompt cache。具体实现取决于 provider 层的能力,留待 implementation plan 决策。
2. **witness_set 采样的 PII 顾虑**:5% 真实用户 query 进入评测管道。可选方案:(a) 全采样但脱敏(需要脱敏规则) / (b) 部分采样(限内部账号 / 限非敏感场景) / (c) 全采样不脱敏(若产品定位允许)。需用户在 implementation plan 阶段拍板。
3. **trial 超时硬上限**:若 candidate 进入 trial 后,witness 样本积累慢 + 人长期不 click,新流量持续按 candidate 走是否可接受?候选方案:(a) trial 状态有硬上限(如 7 天),超过自动 propose rollback(仍人 click) / (b) 不设上限,完全依赖人 / (c) 设上限且自动回滚(触碰"自我修改"边界,需用户明确授权)。
4. **retry_policy 的语义粒度**:是 per-tool(每个工具一份)还是 per-call-site(每个调用点一份)?per-tool 简单但粒度粗;per-call-site 精细但需要 call_site_id 概念,工具注册接口要扩。倾向 per-tool 起步,但需用户确认。
5. **高 divergence_rate 让 C3 失效的应对(文本类 + 行为级)**:strict_replay 把"对比基础被污染"的情况识别出来后,trial 判 `signal_unreliable` 并 propose rollback,但这等同于承认 A1 自动监测对"激进改动"候选(persona 大改、tool_description 颠覆性重写、schema/routing 大改 — 文本类与行为级都中招)无能为力。可选方案:(a) 现状 — 这类候选必须人在 click apply 前评估幅度,trial 仅作事后审计 / (b) 引入"改动幅度"前置度量(文本相似度阈值 / schema 字段变化数 / diff 行数),超过阈值的候选**禁止自动 apply 流程**,强制走人审通道 / (c) 接受现状但在 proposal 卡片中显眼展示幅度估计。倾向 (c) + 部分 (b),但需用户在 implementation plan 阶段拍板。
6. **B5 放宽方向盲区的可接受性**:§5.7 末段已显式登记 B5 抓不到 required→optional / enum 加值 / timeout 调大 这类放宽改动引发的运行时退化。短期靠 `has_loosening` 在卡片中提示 + C3 trial 兜底,但 C3 又被高 divergence 限制。是否需要在 A1 内做轻量反向校验(比如:对放宽类 schema 候选,要求至少 K 个合成调用在 staging 跑通)?倾向暂不在 A1 内做,但需用户明确知情。

---

## 9. 附录 — 造一个轴底座的方法论清单(供 A2 复用)

**说明**: 本章节在 A1 起草时是骨架;A1 实施完成后,每一步填入"做工具层时学到的具体经验"。这一节的完成是 A1 验收的硬要求(§7.5)。

### 步骤(骨架)

1. **定义该轴的"事件"** — 它的 (key, payload) 形式是什么?事件粒度怎么取(太粗 → 区分不出策略变化,太细 → 噪声多)?
2. **写 record/replay 子层** — 对照 `sandbox.py` 的 record/replay/passthrough/side_effect_only 四模式,类比构造该轴的等价模式
3. **定义 override 维度** — 该轴有哪些可被"per-candidate 替换"的部分?(工具层:description / schema / retry_policy;上下文层?记忆层?)
4. **定义 side-effect 防护规则** — 该轴的哪些操作"动了真世界"?如何在评测时拦截?
5. **扩 TunableObject 接口(如适用)** — 该轴的 candidate payload 是什么类型?需要新的 concrete class?
6. **设计 case 集元数据治理** — 该轴的 fix_set / health_set 怎么构造?怎么标注覆盖维度?
7. **接入 dual-set + trial + witness** — 该轴的 witness 信号怎么从生产 run 中抽取?
8. **接入 rollback 状态机** — 该轴的 state_snapshot 是什么?restore_state 怎么写?
9. **写端到端 demo** — 完整跑通一次 detect → fix → verify → trial → stable

### 工具层经验填实(待 A1 实施时填)

(留空,A1 完成时补全。)

---

**End of A1 design.**
