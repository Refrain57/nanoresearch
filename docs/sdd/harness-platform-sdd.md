# Harness 诊断平台 SDD

> 本文档锁定的是**边界、关键决策、验收标准**，不锁定字段名、表结构、函数签名。实现细节在每个 Phase 启动时再定。
>
> 凡是带 **Rationale** 段的章节，下次有人想"改进"它之前请先读完 Rationale——很多看起来像漏掉的东西是**故意没做**。

---

## 1. 这个平台是什么 / 不是什么

### 是什么

一个 **Agent 错误诊断与结构化修复平台**。当 Agent 出错时，平台能告诉你：
- 错在六层（Context / Tool / Orchestration / Memory / Eval / Recovery）的哪一层
- 指向那一层里具体的可调对象
- 提供从 trace 抽出的证据
- 对文本类可调对象，自动生成候选修复方案、在两个独立测试集上对比打分
- 把 before/after 摆给人看，由**人点确认**是否部署

### 不是什么

- **不是会自我修改的黑盒**。所有修复动作的最后一步是"人点确认"，不做无人值守。
- **不是 Prompt 优化器**。Prompt 优化只是平台支持的一类修复对象，不是平台的定义。
- **不是性能监控**。延迟、QPS、错误率这些已有的 APM 工具能干，本平台不做。
- **不是把 Agent 错误都自动修掉**。平台的核心价值是"指向正确的层让人做结构化修复"，自动化修复是其中一部分场景，不是全部。

### 北极星

> **让 Agent 的能力沉淀在系统里，而不是越堆越长的 Prompt 里。**

任何 Phase 在实现时如果开始往"全自动闭环"滑，要拿这条对照——平台价值的 90% 来自"指向正确的层 + 结构化证据"，不来自"无人值守"。

### 上游假设：badcase 标记的可信性

本平台所有 Phase 都建立在一个假设上：上游 badcase 检测的标记是可信的。Phase 0 给 badcase 存 trace、Phase 1 给 badcase 分类、Phase 2 用 badcase 生成候选——整条链路的输入都是"已被标记的 badcase"。

badcase 检测自身的质量（会不会系统性漏标某类错误、会不会把正常 case 误标成 badcase）不在本平台范围内，沿用现有 badcase_detector 的能力。但这里留一个路标：**如果将来整条修复闭环跑起来效果系统性地不对，排查顺序应该是先怀疑 badcase 标记是否有偏，而不是一头扎进去调 optimizer、调阈值、调健康集。** badcase 集的偏差会污染整条链路，比健康集偏差（§4.2）影响更大，但它更隐蔽，因为没有任何下游机制会暴露它。

---

## 2. Non-Goals（明确不做）

以下事项**在本平台范围内明确不做**。如果将来要做，需要重新立项，不能作为某个 Phase 的"自然延伸"。

### 2.1 Task Orchestration 的 Plan-and-Execute / 显式 Plan 阶段
- **Rationale**：对话型 Agent + RAG 的场景不需要 coding agent 那种 hierarchical planning。原文里 Harness 的 Orchestration 层在我们的语境下退化为"trace / 可观测性"，已经被纳入 Phase 0 和 Phase 3 的职责，不需要独立的 plan 编排。

### 2.2 数值类 / 规则类可调对象的自动修复
- 例：budget 分配比例、检索 top-k、记忆衰减权重、Circuit Breaker 阈值。
- **Rationale**：这些是 hyperparameter search 问题，需要 Bayesian / Grid search 之类完全不同的引擎。强行塞进文本类对象的 `generate_candidates` 接口会把接口稀释成 `dict[str, Any]`，丧失类型表达力，最后变成另一种形态的 if-else 地狱。
- **保留的能力**：诊断层**可以指向**这类对象（"该调 retrieval top-k"），但只产出指针 + 建议，不产候选、不自动应用。

### 2.3 全自动部署 / 回滚
- **Rationale**：语言质量的退化是渐进且隐蔽的，不像编译错误有硬约束。全自动部署 Prompt / 工具描述的风险收益比很差。"一键应用 + 人确认"已经实现 Harness 价值的 90%，且可靠得多。
- 这不只是"暂时不做"——这是**设计上的终点形态**。要变成全自动需要重新立项。

### 2.4 复盘层（Phase 3）默认不做
- 见 §6 Phase 3 节。它从主干降级为"闭环跑通后视情况再决定"的增强件。

### 2.5 把六层都做成"可自动修复的对象"
- **Rationale**：六层不平权。Context / Tool 里有文本类对象天然适合自动生成候选；Memory / Recovery 主要是规则与逻辑，不适合自动修。强求统一会导致接口抽象稀释。详见 §3 和 §4.1。

---

## 3. 六层模型与修复边界

### 3.1 六层

| 层 | 说明 | 平台角色 |
|----|------|---------|
| Context | 上下文装配（检索、记忆注入、预算分配） | 诊断 + 自动修复（文本类对象） |
| Tool | 工具描述、工具选择、工具调用 | 诊断 + 自动修复（文本类对象） |
| Orchestration | 执行轨迹、步骤可观测性 | **不作为修复目标**，仅作为 trace 数据来源 |
| Memory | 记忆写入规则、衰减、检索 | 诊断指向，**不自动修复** |
| Eval | 评估方法本身 | 平台**就是** eval 自身，不作为被自动修复的对象——但其质量靠人工维护（健康集腐烂 review、δ 阈值校准），见 §4.2 / §4.4 |
| Recovery | 超时、Circuit Breaker、回滚阈值 | 诊断指向，**不自动修复** |

### 3.2 fixable vs diagnosis_only

- **fixable_layers = { Context, Tool }**：有文本类 TunableObject 实例，完整支持"生成候选 → 双集打分 → 部署门控 → 一键应用 → 回滚"。
- **diagnosis_only_layers = { Memory, Recovery }**：分类器**可以指向**这一层的某个对象，诊断面板展示证据 + 一段"建议手动调整 X"的文字，但**没有"生成候选"按钮**，没有自动修复链路。

#### Rationale：为什么 Recovery 进 layer 枚举但不进自动修复

> 这是**故意收窄**，不是漏掉。
>
> Recovery 层的可调对象主要是数值阈值（超时秒数、连续失败次数）和逻辑规则（Circuit Breaker 触发条件），这些是 §2.2 明确不做的。但 Recovery 仍然进 layer 枚举的原因是：诊断层需要能告诉人"这次 badcase 是因为 escape hatch 太迟触发"，让人去手工调阈值。如果 Recovery 不进枚举，这类 badcase 就无处归类，会被错分到其他层造成噪音。
>
> **预期心理影响**：进度感上，本平台真正能跑通"自动修复闭环"的，只有 Context 和 Tool 里的文本类对象。Memory / Recovery 在很长一段时间里都停在"诊断指向、人工修复"。看到 layer 枚举有四层不要误以为四层都自动了。

#### Rationale：为什么 Orchestration 不在表里独立成行

> 它的两个职责（trace 数据采集 + 步骤可观测性）已经合并到 Phase 0 和 Phase 3 里。独立列一层会暗示它有自己的修复对象，反而误导。

---

## 4. 关键设计决策

### 4.1 可调对象抽象：只覆盖文本类

**决策**：定义一个统一的 TunableObject 接口（read / generate_candidates / apply / get_current_version / rollback），但**只为文本类对象实现**——System Prompt、Tool Description、Persona、Skill Instruction 片段。

#### Rationale

- 文本类对象共享同一种生成范式：LLM 看 badcase → 输出文本候选 → 用测试集打分。一套接口够用。
- 数值/规则类强行纳入会让 `generate_candidates` 退化成无类型 `Any`。
- **先窄后宽**：等文本类两个实例（System Prompt + Tool Description）跑通验证抽象正确，再考虑下一类对象。三点拟合之前不抽象。
- 数值/规则类对象的"被诊断指向"能力不依赖这个接口——诊断输出是个指针 `(layer, target_kind, target_id)`，文本类有自动链路接管，数值/规则类没接管方但指针仍然有效。

#### 反约束

- 接口在 Phase 6 完成前**不允许加新方法**。新需求一律先在调用方塞补丁，攒到下一轮接口 review 再统一处理。
- 不要为了"统一性"把数值/规则类对象塞进同一个接口。这就是接口稀释的开始。

### 4.2 双集回归：badcase_fix 集 + 独立 health 集

**决策**：所有候选方案的打分都同时在两个集上跑，部署门控用"且"逻辑——fix 集要提升、health 集不能退化。

#### Rationale

- **Optimizer 用 badcase 生成候选、再用相关测试集打分** 是经典过拟合陷阱。候选可能"专治这几个 case、伤害其他正常 case"，单看 fix 集分数会得到虚假的部署许可。
- 健康集**不能从生产流量随机切**——同时期、同类用户、同类 query 的随机切分布耦合，过拟合方案在伪独立集上也表现不差。完成定义必须是**显式构造**：覆盖广度（场景清单） + 每格深度（样本量下限）。
- 单集打分给 baseline 锚点会埋一个"看起来安全实际有偏"的雷，比没有门控更危险。

#### 反约束

- 健康集**只切不构造** 是这一步最容易出的错。Phase 2 的验收必须卡住这一点。
- 健康集会随 Agent 演进慢慢"变臭"（原本健康的 case 变成 badcase）。**不要每月 review**——会被 badcase 拉着走。建议半年一次。

### 4.3 沙箱分层：副作用类 vs 查询类

**决策**：工具分两类——副作用类（write_file、exec、message、cron）必须 mock/replay；查询类（read_file、web_fetch、rag_search）在 Tool Description 优化场景下可 passthrough。

#### Rationale

- 当前沙箱以 `(tool_name, normalized_params)` 为 replay key，System Prompt 改了 key 还能命中，但 Tool Description 改了模型可能调**不同的工具**，replay 直接报错。
- 一刀切的"纯 replay"和"纯真实执行"都不对——前者让 Tool 优化跑不起来，后者让评估真的发邮件/改文件。
- 正确的切分是按**工具本身的语义**：副作用类不能放、查询类可以放。
- **保守默认**：工具基类未声明 `side_effect` 字段时一律视为副作用。误标成查询类会真的写线上，反过来误标成副作用只是评估贵一点。

### 4.4 部署门控：不是自动部署

**决策**：候选满足"fix 集提升 ≥ δ_improve 且 health 集退化 ≤ δ_tolerate"才**进入待批准状态**，部署动作仍需人点确认。

#### Rationale

- 门控只是把"明显变差的候选"挡掉，不是替代人判断。
- 阈值首版用经验值 hardcode（improve > 0.05、tolerate > -0.02），跑一段时间收数据再校准。**不要一开始就做"动态阈值"**——还没数据就调参，等于猜。
- baseline 分数用**当次评估同时跑出来的**，不用历史存档分。模型升级、数据变化会让历史 baseline 漂移，门控会失真。

### 4.5 Trace 分层：常态轻量 + 复盘按需

**决策**：常态 trace 存结构化的装配决策（检索 query 文本、预算分配数字、片段标识、技能/persona 状态），**不存全文**。badcase 标记后才考虑全量重建（即 Phase 3，已降级到 P6 之后）。

#### Rationale

- 全量 trace 体积大、含 PII、容易被关掉，最后等于没做。
- 结构化决策已经能回答 80% 的诊断问题（"为什么没检索到" = 看 query 和 budget 分配）。剩 20% 需要全量的难案例**恰恰最可能因为内容漂移而无法重建**——见 §6 Phase 3。
- 所以"实时全量抓取"和"事后重建"是两种不同的设计，**不要把它们当成连续频谱**。当前选事后重建（轻量 + 按需），如果实践证明需要实时全量，是一个**独立的新提案**，不复用 Phase 3 的设计假设。

---

## 5. 阶段依赖图

```
Phase 0  trace 轻量层
   ↓
Phase 1  根因→层级映射 + 文本类可调对象接口
   ↓
Phase 2  回归集分离
   ↓
Phase 4  沙箱分层
   ↓
Phase 5  baseline 锚点 + 部署门控
   ↓
Phase 6  诊断面板 + 一键应用 + 人确认
─────────  闭环到此打通  ─────────
Phase 3  复盘层（按 P6 真实使用情况再决定是否做、做成什么样）
```

**关键路径全串行，无并行岔路。** 编号保留原顺序（不重排为 0-5），是为了和早期讨论文档保持可追溯。

> 注：Phase 4 仅 Tool Description 优化线依赖。System Prompt 优化线用现有 strict replay 即可，在 Phase 2 之后即可接 Phase 5，不必等 Phase 4。如果需要让 System Prompt 闭环先跑通，这是一个可用的加速路径。

---

## 6. 各阶段

### Phase 0 · Trace 轻量层

#### 做什么
让每一次 Agent run 在 snapshot 里留下结构化的**上下文装配决策**记录：
- 检索时实际使用的 query 文本（user_memory 查什么、knowledge 查什么）
- 预算分配数字（memory / knowledge 各占多少 token、实际用了多少）
- 被注入的 memory / knowledge 片段的标识
- 加载的 skills 和激活的 persona

不存被注入内容的全文。

#### 为什么这个顺序
当前 `badcase_classifier` 喂给 LLM 的只有 `tool_call_chain[:500]` 和 `final_response[:800]`，它**没法区分**"没检索 / 检索了但空 / query 写错了"这三种——而这三种是三个不同层的根因。映射层（Phase 1）的输入数据不补，分类就是浮沙上盖楼。这是逻辑前置依赖。

#### 依赖
无。可立即开工。

#### 完成定义
随机点开任意一条 badcase，能从存储里读到"这次检索用的 query 文本是 X、memory 占了 N tokens、注入了片段 [id 列表]"，**不需要 SSH 进服务复现**。

> **实现注记（Phase 0 已交付，2026-06-23）**：`context_trace` 字段已写入 `agent_run_snapshots`（JSONB），9 个字段覆盖上述四类决策。完成定义中"随机点开"隐含的**快照需能按 badcase/conversation 定位**这一前提已修复（snapshot.conversation_id 现在存真实 conversations.id，见下方风险项）。详见 `PHASE_STATUS.md`。

#### 风险
- **不要存全文**。只存 id + query 文本 + 数字。全文留给 Phase 3 按需重建（或如 §4.5 所说，可能根本不做）。
- ContextBuilder 当前是同步路径，记录决策时**不要为了 trace 引入 await**，避免热路径异步化。
- 字段命名、存储形态（新 JSONB 列 vs 扩展现有列）由实现时定，本文档不锁。
- **snapshot ↔ conversation 关联**（实现时发现的实际风险）：`agent_run_snapshots.conversation_id` 原来存的是 session UUID（`conversations.channel_chat_id`），而非 `conversations.id` 外键。已通过五跳穿透链修复（chat_router → worker → process_direct → _process_message → _run_agent_loop），代码走查通过，**待重启服务后真实 run 验证**。Phase 1 开工前必须验证通过，否则"给定一条 badcase 找到其 context_trace"这一前提不成立。
- **CLI/"direct" run 的 conversation_id 为 NULL**：`process_direct` 默认 `chat_id="direct"` 的路径（`nanobot chat` 命令等）不经过 conversations 表，snapshot 存 `conversation_id=NULL` 是正确行为。Phase 1 和 Phase 6 需为 `conversation_id=NULL` 的快照提供按 `run_id` 或 `uid+timestamp` 的降级定位方式。

---

### Phase 1 · 根因→层级映射 + 文本类可调对象接口

#### 做什么

A. **重塑分类法**：把 `badcase_classifier` 当前的 `root_cause_auto ∈ {prompt, context, tool, model, user_input}` 升级为结构化二元组：
   - `layer ∈ { Context, Tool, Memory, Recovery }`（明确标注 fixable / diagnosis_only 两组，UI 上视觉区分）
   - `target_kind`：进一步指向 layer 内的具体对象类型（如 `system_prompt`、`tool_description`、`retrieval_strategy`、`memory_write_rule`）
   - `target_id`：具体对象标识

B. **TunableObject 接口**：定义文本类对象的统一行为——读取当前内容、生成候选、应用版本、查询当前版本、回滚到指定版本。**首批实现 2 个**：System Prompt、Tool Description。

C. **版本注册表**：所有文本类对象的修改通过版本注册表落地，每次 apply 写新版本并激活，旧版本保留以备回滚。具体表结构由实现时定。

D. **数值/规则类对象**：分类器的 `target_kind` 可以指向它们（命名空间不限于文本类），但**不实现 TunableObject 接口**。诊断面板对这类对象只展示证据 + 建议。

#### 为什么这个顺序
这一步定义平台的"语言"——后面所有阶段都围绕"对一个 TunableObject 做评估/部署/回滚"展开。映射层（诊断结果）和接口（修复对象）必须同时立起来，否则诊断结果没有接收方。

#### 依赖
- Phase 0 的装配决策数据。
- **`snapshot.conversation_id` 能稳定定位到 `conversations.id`**（Phase 0 遗留债务 P0-D1，未验证前 Phase 1 无法从 badcase 可靠地找到其 context_trace）。

#### 完成定义
1. 给定任意一条 badcase，分类器输出结构化指针，指向一个具体对象。
2. 现有 `OptimizationAgent` 改造为接受 `TunableTextObject` 参数，不再对"系统 prompt 是唯一被优化对象"做硬编码假设。
3. 至少 2 个 TunableTextObject 实例（System Prompt、Tool Description）能完整跑通 read / generate_candidates。apply / rollback 在本阶段完整实现并可单测（写版本注册表、激活、回退都要真做），但不接入生产触发路径——生产环境下何时调用 apply 由 Phase 5 的门控 + Phase 6 的人确认驱动。

#### 风险
- **抽象漂移**：每来一个需求就想给接口加一个方法。接口在 Phase 6 完成前**冻结**新增方法。
- **被 layer 枚举误导**：layer 列出四层不等于四层都有自动修复链路。UI 和文档都要明确 fixable / diagnosis_only 两组的视觉区分。
- **历史数据迁移**：`root_cause_auto` 列已有数据。新增列**并行写入**，旧字段读老数据，稳定后再切。不动老列。
- **CLI/"direct" run 定位**：`conversation_id=NULL` 的快照（CLI 路径）无法按 conversation 索引。分类器在处理 badcase 时需要能按 `run_id` 定位，不能假设 `conversation_id` 一定非 NULL。

---

### Phase 2 · 回归集分离

#### 做什么
- 测试集形式化分两类：`badcase_fix`（触发优化的 badcase 集）+ `health`（独立健康集）。
- 健康集**显式构造**，不从生产流量随机切：
  - 场景广度：列出场景分类清单（事实问答 / 总结 / 多轮 / 工具调用 / 闲聊 / RAG 命中 / RAG 不命中 / 新用户 / 老用户……）
  - 每格深度：每类场景的样本量满足"在预期效应量下能区分噪声"，**经验下限不低于每类 5 个 case，总规模不低于 50 个 case**
- 评估接口同时返回两个分数，分别报告。

#### 为什么这个顺序
没有这一步，后面 baseline 锚点和门控会基于被污染的评分，给出虚假的部署许可。先把评估方法论修对，再扩展沙箱和门控。

#### 依赖
Phase 1 的 TunableObject 接口（评分要按对象分别给出）。

#### 完成定义
1. 健康集有一份**显式的场景覆盖清单**（markdown 或表格），review 人能逐条核对。
2. 清单包含每类场景的样本数，符合下限要求。
3. Optimizer 报告里 fix / health 两个分数**分开呈现**，不是合并平均。
4. 候选打分的代码路径上**强制要求两个集都存在**，缺一不允许打分。

#### 风险
- **只切不构造**：最容易犯的错。验收时**明确拒绝**"我从 golden 集里随机切了一半"这种做法。
- **样本量不够的虚假门控**：每类只有 1-2 个样本时，分数波动多半是噪声。完成定义里的"每类下限 5 + 总下限 50"必须卡。
- **健康集腐烂**：随着 agent 演进，老 case 可能变 badcase。建议半年 review 一次，**不要更频繁**——会被 badcase 拉着走。

---

### Phase 4 · 沙箱分层

#### 做什么
- 工具元数据增加"是否有副作用"声明，**未声明视为副作用**（保守默认）。
- 沙箱增加 `side_effect_only` 模式：副作用工具仍走 mock/replay，查询工具未命中录音时 passthrough 到真实调用。
- Tool Description 优化的回归评估**默认走** `side_effect_only` 模式。

#### 为什么这个顺序
只有 Tool Description 优化场景下沙箱才会因为 replay key 失效而崩。System Prompt 优化用现有 `strict` replay 就够。所以排在 Phase 2（评分方法论）之后、Phase 5（门控）之前——前者影响所有优化，后者只影响 Tool 层，但 Tool 层的门控必须依赖沙箱能跑起来。

#### 依赖
- Phase 1 的 TunableObject（有 tool_description 实例触发需求）
- Phase 2 的两套测试集（评分要在两个集上跑）

#### 完成定义
1. 修改一个 RAG 工具的 description 后，能在 `side_effect_only` 模式下跑完一次回归，**不抛 SandboxReplayError**。
2. 副作用工具被调用时被拦截，记入 audit log。
3. 工具基类强制声明 side_effect 字段（或基类默认为副作用），未声明时无法注册到 registry。

#### 风险
- **passthrough 的真实成本**：评估 Tool Description 时会真发 web_fetch / rag_search，token 和 RPS 都是钱。要预估并设上限（候选数 ≤ 3、测试集规模适中）。
- **工具分类错漏**：把副作用工具误标成查询类，评估时真写线上。基类强制声明 + 保守默认是为了防这个。
- **passthrough 引入不确定性**：真实调用结果不稳定（web 内容会变），评估打分会有噪声。这是必然代价，没法消除，只能用更大的测试集稀释。

---

### Phase 5 · Baseline 锚点 + 部署门控

#### 做什么
- 优化提案数据结构补齐 baseline 锚点：target 对象、基准版本、基准在 fix/health 两集的分数、候选在 fix/health 两集的分数。
- 部署门控规则：`fix 提升 ≥ δ_improve` **且** `health 退化 ≤ δ_tolerate`。两条**与**关系，缺一不可。
- 门控通过的候选进入"待批准"状态，不自动 apply。
- baseline 分数**用当次评估同时跑出来的**，不用历史存档。

#### 为什么这个顺序
数据结构和评分流（双集 + 沙箱分层）都到位才能锚 baseline。提前锚没意义。

#### 依赖
Phase 1（apply / rollback 接口）+ Phase 2（双集分数） + Phase 4（沙箱能跑工具优化）。

#### 完成定义
一条优化提案从生成到批准，可以通过查询数据库直接看到"基准版本 X、新候选 Y、两集分数分别如何、改善差值多少"——**不需要看代码或日志**。

#### 风险
- **δ 阈值怎么定**：首版 hardcode 经验值，跑一段时间后再调。**不要一开始就做动态阈值**。
- **baseline 漂移**：用当次同时跑出来的 baseline 分数防漂移，不要图省事用历史值。
- **门控通过 ≠ 该部署**：门控只是兜底防御，最终决定权在 Phase 6 的人。门控通过的候选 UI 上不要默认高亮"建议部署"——防止橡皮图章。

---

### Phase 6 · 诊断面板 + 一键应用 + 人确认

#### 做什么
- Badcase 列表（已有）→ 点开进入**诊断面板**：
  - 现象（semantic_category）
  - 根因层级 + 可调对象指针（Phase 1 产出）
  - 证据（从 Phase 0 的 trace 抽出关键行）
- 对 fixable 层的对象：有"生成候选"按钮 → 调 Optimizer。
- 对 diagnosis_only 层的对象：**只展示证据和文字建议**，没有"生成候选"按钮。
- 候选对比页：左右栏列 baseline / candidates，**默认同时呈现 fix / health 两集分数**，不折叠不二级 tab。
- "应用此版本"按钮 → 人点 → 走 TunableObject.apply（版本注册表写新版本并激活）。
- "回滚到上一版"按钮 → 走 TunableObject.rollback。

#### 为什么这个顺序
前面所有数据流到位才能渲染有意义的 UI。这是闭环最后一环。

#### 依赖
Phase 0-5 全部。

#### 完成定义
一个工程师从 badcase 列表点进去，**5 分钟内**能完成"看诊断 → 看证据 → 看候选 → 看对比 → 决定是否应用"全流程，且整个过程不需要看代码。

#### 风险
- **UI 默认呈现**：双集分数必须默认同时显示，不能藏在二级 tab。否则人只看一个分数，Phase 2 等于白做。
- **橡皮图章式确认**：人看到高分就盲点。建议**证据默认收起、需要手动展开**——制造一点摩擦，防止"机器说 OK 我就点"。
- **fixable / diagnosis_only 视觉区分**：UI 上必须明显不同。不能让 diagnosis_only 的对象看起来"差一个按钮没做完"，要让人一看就知道"这层故意没有自动修复"。
- **CLI run 的诊断面板降级展示**：`conversation_id=NULL` 的快照（CLI/"direct" 路径）无法从 conversation 维度导航进来。诊断面板需为这类快照提供按 `run_id` 的直接链接入口，或在 badcase 列表里标注"CLI run"。

#### 闭环到此打通

到 Phase 6 完成，从"badcase 检测 → 根因定位 → 候选生成 → 双集打分 → 人点确认 → 部署/回滚"的完整闭环就通了。

---

### Phase 3 · 复盘层（降级，按需再做）

> **当前状态：默认不做**。在 Phase 6 跑通、收集到真实数据之前，不投入此阶段。

#### 为什么从主干降级
1. **职责依赖**：复盘层是给"已经能跑通的闭环"提供更强证据的增强件，本身不在闭环关键路径上。闭环都没通就先建增强件，顺序反了。
2. **消费者依赖**：复盘层重建出的全量 context 给谁看？给 Phase 6 的诊断面板。面板不存在时，复盘层是悬空的。
3. **价值上限低**：复盘层服务于最难的 20% 案例，但这些案例最常见的形态是"当时检索到的文档现在变了"——而**这恰恰是复盘层无法重建的**。复盘层真正能还原的只是装配决策层面（budget 怎么分、persona 状态），而装配决策 Phase 0 已经存了。
4. **风险高**：PII、对象存储、签名 URL、审计——一整套合规与基础设施。

#### 真要做时的边界

如果 Phase 6 跑通后发现轻量 trace 确实不够、需要复盘层，做之前先问：
- 是不是要重建装配决策层面的问题？→ Phase 0 应该已经够了，去补 Phase 0 的字段。
- 是不是要看"当时模型实际收到的完整 messages"？→ Phase 3 重建可以做，但要标注"重建时刻可能与运行时有差异"。
- 是不是要看"当时检索到的文档内容"？→ Phase 3 重建做不到。**如果决定要做实时全量抓取，那是独立的新提案，不复用 Phase 3 的设计假设**——它有完全不同的写路径开销、存储模型和隐私边界。

#### 何时重新评估
Phase 6 完成后跑 1-2 个月真实流量，统计：
- 轻量 trace 不够用的 badcase 占比
- 这些 case 中"装配决策类" vs "内容漂移类"的分布

如果装配决策类占多数 → 补 Phase 0 字段即可。
如果内容漂移类占多数 → Phase 3 价值有限，考虑做实时全量（独立提案）。
如果两类都不多 → Phase 3 可以不做。

---

## 7. 跨阶段风险

### 7.1 抽象漂移
每个 Phase 都有诱惑把接口"再通用一点"。规矩：**TunableObject 接口在 Phase 6 完成前冻结新增方法**。新需求一律在调用方塞补丁。

### 7.2 数据迁移
现有列（`root_cause_auto`、`OptimizationProposal.proposals` 等）都有历史数据。每个 Phase 一律新增列而非改老列，迁移期双写、读老字段，稳定后再切。**不要直接 ALTER 老字段**。

### 7.3 评估成本
Phase 4 之后每次优化要跑 fix + health 两集 × 候选数。需要预算和上限：
- 单次优化候选数 ≤ 3
- 单次优化总耗时上限（建议 ≤ 30 分钟）
- 单次优化 token 成本上限（具体阈值待校准）
超限警告 + 熔断。

### 7.4 文档与代码漂移
SDD 不应该和代码反复脱节。**每个 Phase 完成后**，回头读这份文档，把"决策已经变了"的地方更新——尤其是 Rationale 段。如果发现某条 Rationale 已经不成立但决策还在，说明决策本身可能要重审，不是文档要修。

---

## 8. 阅读指引

- 只想知道**做什么** → 看 §6 各 Phase 的"做什么"段。
- 想知道**为什么这么决定** → 看 §4 关键设计决策、§2 Non-Goals、每个 Phase 的"为什么这个顺序"和"Rationale"段。
- 想验收某个 Phase → 看那个 Phase 的"完成定义"段。
- 想加新需求 / 改方案 → 先读 §2 Non-Goals 和 §4 关键设计决策，**确认你要改的不是被故意排除的**。

---

## 9. 修订记录

| 日期 | 修订内容 | 备注 |
|------|----------|------|
| 2026-06-23 | 初稿 | Phase 3 从主干降级到 P6 之后 |
| 2026-06-23 | Phase 0 交付；回填实现发现的两处风险到 Phase 0/1/6 风险段 | 见下 |

### Phase 0 交付说明（2026-06-23）

**已完成并验收：**
- 9 个 trace 字段全部写入 `agent_run_snapshots.context_trace`（JSONB，迁移脚本 `scripts/migrate_phase0_context_trace.sql`，含 downgrade）
- DB round-trip 验证通过；fragment_ids 非空路径同进程单元测试验证通过（`um_*` 格式）
- 3 并发隔离测试通过（caller-per-run dict，无跨 run 污染）
- commit `c2b002e8`

**本轮修复（随 Phase 0 一并提交）：**
- `build_history_context` 静默吞异常改为 `logger.exception`
- `snapshot.conversation_id` 关联：发现原实现存的是 session UUID 而非 `conversations.id`（三证坐实），改为五跳穿透真实 `conv.id`；UUID 转换失败改为 warning 而非静默 pass
- 上述代码走查通过，**尚待重启服务后真实 run 验证**

**需要你确认的不确定项（本文档不自行修改这些内容）：**

1. **Phase 0 完成定义的"随机点开"是否隐含 conversation 关联**：目前 Phase 0 完成定义写的是"能从存储里读到诊断数据"，但 Phase 1 开工依赖"从 badcase 稳定找到其 context_trace"。如果你认为 conversation_id 落库验证属于 Phase 0 完成定义的必要条件，请明确写入；否则它作为 Phase 1 前置债务（P0-D1）处理即可。

2. **Phase 1 依赖段是否需要明确写出 P0-D1**：当前已在 Phase 1 依赖段加了"snapshot.conversation_id 能稳定定位"这一条。如果你认为这个表述过重（Phase 1 可以先开始不依赖此），请告诉我删掉。

3. **fragment_ids 生产路径验证的位置**：目前作为"建议"放在 PHASE_STATUS.md 债务表里，没有写入 SDD。如果你认为它应该进 Phase 0 完成定义，请告诉我。
