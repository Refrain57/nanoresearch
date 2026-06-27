# Eval Framework 系统性改进方案

## Context

当前 eval 框架已有 rule eval + LLM Judge + badcase 检测的基础结构，但存在系统性短板：
keyword_coverage 纯字符串匹配导致中英文跨语言漏判；Judge 单次打分方差大；badcase 分类粗糙；
没有 faithfulness 检测；工具评估停留在调没调而非调得对；测试集静态无法自增长。
本方案在不引入外部框架的前提下，沿现有代码结构改进，分评估指标层和测试集运营层两个层次推进。

---

## 关键文件清单

| 文件 | 当前职责 |
|------|---------|
| `backend/nanobot/eval/evaluator.py` | `RuleEvaluator` — keyword_coverage / tool_skip / token_budget |
| `backend/nanobot/eval/judge.py` | `LLMJudge` — 4-5 维度打分 + calibrate |
| `backend/nanobot/eval/badcase_detector.py` | `BadcaseDetector` — 5 类 badcase 检测 |
| `backend/nanobot/eval/badcase_classifier.py` | `BadcaseClassifier` — 语义分类 + 根因 |
| `backend/nanobot/eval/test_runner.py` | `TestRunner.run_all()` — 批量执行 + 打分编排 |
| `backend/nanobot/eval/snapshot.py` | `RunSnapshotData` / `RunSnapshotCollector` |
| `backend/nanobot/storage/models.py` | `AgentRunSnapshot` / `AgentTestCase` ORM |
| `backend/nanobot/storage/repositories/agent_eval_repo.py` | `AgentEvalRepository` — DB 读写 |
| `backend/nanobot/server/routers/agent_eval_router.py` | API 路由 |
| `backend/migrations/` | 原始 SQL 迁移文件 |

---

## 方向一：语义匹配替换字符串匹配

### 改什么

`evaluator.py` 中 `RuleEvaluator.evaluate()` 的 `_kw_hit()` 辅助函数，改为双层判断：
1. 先做当前的 `kw.lower() in resp_lower` 字符串匹配（零成本）
2. 未命中时，调用 embedding 服务计算关键词与回答文本的语义相似度，阈值 0.85

### 怎么改

- `RuleEvaluator.__init__` 新增可选参数 `embedding_fn: Callable[[str], list[float]] | None = None`
- `evaluate()` 改为 `async`；当 `embedding_fn` 不为 None 时，对未命中的关键词做向量相似度判断
- 向量策略：对回答按句子分块，取关键词 embedding 与各句 embedding 的最高余弦相似度
- 复用 `backend/nanobot/rag/libs/embedding/embedding_factory.py` 的现有 embedding 服务
- `BILINGUAL_MAP` 在语义匹配稳定后删除（可分两个 PR）

### 影响文件

- `evaluator.py` — `__init__`、`evaluate()`（改 async）、`_kw_hit()`
- `test_runner.py` — `run_one()` 中 `evaluator.evaluate()` 改为 `await`，构造 `RuleEvaluator` 时注入 embedding_fn

### 预期效果

消除中文回答 + 英文 keyword 导致的系统性误判；`BILINGUAL_MAP` 可退役；keyword_coverage 准确率提升。

---

## 方向二：G-Eval 式 Judge

### 改什么

`judge.py` 的 `_SYSTEM_PROMPT` 和 `_build_prompt()`，将单步"直接打分"改为两步：
1. 先让模型针对本题生成评分标准（chain-of-thought）
2. 再按标准打分

### 怎么改

- `_SYSTEM_PROMPT` 新增指令：先输出 `criteria`（针对本题的评分标准列表），再输出分数
- 输出 JSON 格式变更：
  ```json
  {
    "dimensions": {
      "task_completion": {"criteria": ["是否直接回答了问题", "是否引用了来源"], "score": 4, "reason": "..."},
      "tool_rationality": {"criteria": [...], "score": 3, "reason": "..."}
    }
  }
  ```
- `_parse_scores()` 更新解析逻辑以匹配新格式，向后兼容旧格式（如遇旧格式降级处理）
- `max_tokens` 从 512 提升到 1024（criteria 会增加输出长度）
- `reasoning` 字段保留，从各维度 `reason` 拼接而来

### 影响文件

- `judge.py` — `_SYSTEM_PROMPT`、`_build_prompt()`、`_parse_scores()`

### 预期效果

Judge 推理过程可追溯；`judge_metadata.raw_output` 存储 criteria 供人工审计；打分分差显著降低。

---

## 方向三：Self-consistency

### 改什么

`judge.py` 新增 `score_with_consistency()` 方法，对同一 case 运行 3 次 Judge，取众数得分；差异超过 1 分标记 `low_confidence`。

### 怎么改

- `LLMJudge.score_with_consistency()` 并发调用 `score()` 3 次
- 逐维度取众数（或中位数）作为最终分
- 若任意维度 3 次结果极差 > 1，在返回值中附带 `low_confidence: True`
- 结果写入 `judge_metadata` JSONB：`{"low_confidence": true, "score_runs": [[...], [...], [...]]}`
- `test_runner.py` 中 `config.use_judge=True` 时调用 `score_with_consistency()` 而非 `score()`
- `EvalRunConfig` 新增 `judge_consistency_runs: int = 3`，允许调为 1（跳过一致性检查）

### 影响文件

- `judge.py` — 新增 `score_with_consistency()`
- `test_runner.py` — 调用入口切换

### 预期效果

`judge_metadata.low_confidence=True` 的 case 自动进入人工复核队列（可在前端按此字段过滤）；打分稳定性量化可见。

---

## 方向四：Faithfulness 检测

### 改什么

`judge.py` 的 Judge prompt 新增 `faithfulness_score` 维度：对比 agent 最终回复与工具实际返回内容，判断关键声明是否有 source 支撑。

### 怎么改

- `_SYSTEM_PROMPT` 新增 `faithfulness` 维度说明：1=关键声明与工具返回严重矛盾或无任何 source，5=所有关键声明均有 source 支撑
- `_build_prompt()` 在"工具调用链"部分明确标注每次工具调用的返回内容（当前 `result` 字段已在 `tool_call_chain` 中），加标题提示 Judge 这是 ground truth
- `faithfulness_score` **不再像 `hallucination` 那样 pop 出来单独存**，直接保留在 `scores` JSONB 作为一级指标
- `test_runner.py` 中现有 `hallucination = judge_scores.pop("hallucination", None)` 逻辑替换为：`faithfulness_score` 直接保留在 `combined_scores`；Judge prompt 把 `hallucination` 维度重命名为 `faithfulness_score`
- **迁移兼容**：处理在 **repo 读层**，`agent_eval_repo.py` 所有读取 snapshot 的方法（`get_snapshot()`、`list_snapshots()` 等），在返回 `scores` JSONB 前执行一次字段映射：若 `scores` 中有 `hallucination` 但无 `faithfulness_score`，则 `scores["faithfulness_score"] = scores.pop("hallucination")`。此映射仅对历史数据生效（新快照写入时已是 `faithfulness_score`），对 API 和前端透明，不需要 DB migration
- **faithfulness_score 与 is_passed() 的关系**：`faithfulness_score` **不影响** `is_passed()` 的布尔返回值——`is_passed()` 继续只看 `keyword_coverage >= threshold`。`faithfulness_score` 的作用路径是：
  1. 若 `faithfulness_score < 0.6`，写入 `failed_dimensions` JSONB（与现有 judge 维度行为一致）
  2. `badcase_detector.detect()` 用 `faithfulness_score < 0.4` 来把 `low_quality` badcase 细化分类为 `hallucination`（方向七）
  3. 进入 trend / regression 统计作为独立观测指标
  这样不改变 passed 布尔语义，保持向后兼容。

### 影响文件

- `judge.py` — `_SYSTEM_PROMPT`、`_build_prompt()`
- `test_runner.py` — 移除 `hallucination` pop 逻辑
- `agent_eval_repo.py` — 读层字段映射

### 预期效果

`faithfulness_score` 成为一级 scores 指标，参与 trend、regression、badcase 检测。`hallucination` 字段退役。

---

## 方向五：Contextual Recall

### 改什么

`evaluator.py` 新增 `contextual_recall` 指标：检查检索到的 chunks（工具返回内容）是否覆盖了期望关键词。纯规则，无 LLM 调用。

### 怎么改

- `RuleEvaluator.evaluate()` 末尾新增逻辑：当 `test_case.expected_keywords` 非空**且** `snapshot.tool_call_chain` 非空时，遍历 `tool_call_chain[*].result`，合并为 retrieved_text，再执行 `_kw_hit()` 检查
- **边缘情况**：若 `tool_call_chain` 为空（agent 未调任何工具，直接凭预训练知识回答），则**不写入** `contextual_recall` 到 scores——与 `expected_keywords` 为 None 时跳过 `keyword_coverage` 的逻辑完全一致。不能默认给 0，因为"没检索"和"检索了但没命中"是不同的失败模式
- 输出：`contextual_recall = covered_in_chunks / total_keywords`
- `badcase_detector.py` 的 `_NON_QUALITY_DIMS` 加入 `contextual_recall`（仅观测，不纳入 pass/fail）

### 影响文件

- `evaluator.py` — `evaluate()` 尾部追加逻辑
- `badcase_detector.py` — `_NON_QUALITY_DIMS` 加入 `contextual_recall`

### 预期效果

可以区分"回答没提到关键词（keyword_coverage 低）"和"检索就没拿回来（contextual_recall 低）"这两种截然不同的失败原因。

---

## 方向六：工具参数级别验证

### 改什么

`evaluator.py` 的 `tool_skip` 评分升级：不只判断工具有没有被调用，还检查调用参数合理性。

### 怎么改

- embedding 调用成本：每条 web_search case 多 2 次 embedding 调用（query + user_input），50 条 case = 100 次/run，量级可接受。embedding_fn 全程在 async 路径中调用（`evaluate()` 已改 async），不阻塞并发 case。跨 case 复用：在 `run_one()` 里可对同一 user_input 的 embedding 做进程内缓存（dict keyed by text），避免重复计算
- `evaluate()` 中 tool_skip 计算改为：
  1. 工具未调用 → `tool_skip = 0.0`（不变）
  2. 工具已调用，参数校验通过 → `tool_skip = 1.0` / 部分命中 → `0.7`（不变）
  3. 工具已调用，参数校验失败 → `tool_skip = 0.5`（新增中间档）
- 参数校验规则（可扩展的 `_validate_tool_params()` 函数）：
  - `web_search`：`params.query` 与 `snapshot.user_input` embedding 余弦相似度 > 0.6（复用方向一的 embedding_fn）
  - `retrieve_by_entity`：`params.entity`（或 `params.query`）中至少有一个词出现在 `expected_keywords`
  - 其他工具：暂不校验（返回 True）
- 现有 `is_passed()` 和 `badcase_detector.py` 逻辑不变：`tool_skip < 0.5` 触发 failed_dimensions，`= 0.5` 新增 `tool_param_wrong` 标记到 failed_dimensions

### 影响文件

- `evaluator.py` — `evaluate()` 的 tool_skip 段，新增 `_validate_tool_params()`

### 预期效果

区分"没调工具"（0.0）和"调了但调错了"（0.5），让 tool_skip 指标真正反映工具调用质量。

---

## 方向七：失败模式分类细化

### 改什么

`badcase_detector.py` 的 `detect()` 对 `low_quality` 类型做二次分解，输出更细粒度的失败模式。同步更新 `badcase_classifier.py` 的语义分类法，加入 `reasoning_gap`。

### 怎么改

- `badcase_detector.py` 中，`low_quality` 触发后继续分析：
  1. `retrieval_failure`：`scores.get("contextual_recall", 1.0) < 0.3`（检索层就没找到）
  2. `hallucination`：`scores.get("faithfulness_score", 1.0) < 0.4`（方向四上线后生效）
  3. `reasoning_gap`：`scores.get("contextual_recall", None) is not None and scores["contextual_recall"] >= 0.5 and scores.get("task_completion", None) is not None and scores["task_completion"] < 0.5`——**task_completion 来自 Judge，只对跑了 Judge 的 case 存在**；任一分项为 None 时直接跳过 reasoning_gap 检测，fallback 到 `low_quality`
  4. 以上都不满足（含 Judge 未运行的 case）→ 保留 `low_quality`
- `badcase_classifier.py` 的 `SEMANTIC_TAXONOMY` 新增 `reasoning_gap`（当前 `reasoning_error` 侧重 prompt，`reasoning_gap` 侧重"有 context 但断链"），同步更新 `TAXONOMY_LABELS_ZH`
- `_rule_based_root_cause()` 增加 reasoning_gap 的规则快捷路径：contextual_recall >= 0.5 + task_completion < 0.5 → `reasoning_gap`

### 影响文件

- `badcase_detector.py` — `detect()` 的 low_quality 分支
- `badcase_classifier.py` — `SEMANTIC_TAXONOMY`、`TAXONOMY_LABELS_ZH`、`_rule_based_root_cause()`

### 预期效果

badcase dashboard 可按 `retrieval_failure / hallucination / reasoning_gap / tool_skip / low_quality` 五类分布查看，优化方向更明确。

---

## 方向八：多轮对话一致性评估

### 改什么

增强 `judge.py` 中 `multi_turn_coherence` 维度的 prompt 指令，从宽泛的"连贯性"升级为显式的"矛盾检测"。

### 怎么改

- `_SYSTEM_PROMPT` 中 `multi_turn_coherence` 的评分说明细化：
  - 1 = 当前回答与历史 assistant 消息中的事实陈述存在明显矛盾
  - 3 = 未主动利用历史上下文但无矛盾
  - 5 = 正确引用历史信息且无矛盾
  - 特别说明：**先检查矛盾，再评连贯性**（矛盾 → 低分，连贯性差 → 中分）
- `_build_prompt()` 中历史对话部分增加提示："请重点检查 assistant 历史回复中的事实性陈述是否与当前回复矛盾"
- `consistency_score` 和 `multi_turn_coherence` 指向同一维度，不新增字段（保持现有结构不变）
- 仍为观测指标，不纳入 pass/fail

### 影响文件

- `judge.py` — `_SYSTEM_PROMPT`、`_build_prompt()` 的历史对话部分

### 预期效果

多轮场景下的前后矛盾问题（如"第一轮说论文发表于2020年，第二轮说2021年"）能被 Judge 捕获并反映在低分上。

---

## 方向九：数据飞轮

### 改什么

建立 badcase → 新 case 的自动生成机制，包括触发条件检测、LLM 生成 case、待审核队列管理、Red Teaming 集成。

### 怎么改

**1. DB 扩展 — `AgentTestCase` 新增字段**（新增 SQL migration `add_pending_cases_fields.sql`）：
```sql
ALTER TABLE agent_test_cases
  ADD COLUMN status VARCHAR(20) DEFAULT 'active',        -- active | pending_review | rejected
  ADD COLUMN generated_from_snapshot_id UUID REFERENCES agent_run_snapshots(id),
  ADD COLUMN generation_reason TEXT;
```
同步更新 `models.py` 的 `AgentTestCase` ORM 类。

**2. 新文件 `backend/nanobot/eval/data_flywheel.py`**，核心类 `DataFlywheel`：
- `check_trigger(run_summary, failure_stats, thresholds) -> list[str]`：检查各失败模式占比是否超阈值，返回触发的失败模式列表
- `generate_cases_from_badcases(snapshots, failure_mode, count=3) -> list[dict]`：用 LLM 基于失败快照生成同类型新 case，返回 `{user_input, expected_keywords, expected_tools, generation_reason, source_snapshot_id}`
- `generate_adversarial_cases(count=5) -> list[dict]`：Red Teaming — 用 LLM 生成注入攻击、意图模糊、边界输入等对抗性 case

**3. `EvalRunConfig` 新增飞轮配置字段**（阈值不硬编码）：
```python
enable_flywheel: bool = False
flywheel_thresholds: dict[str, float] = field(default_factory=lambda: {
    "retrieval_failure": 0.20,
    "hallucination": 0.15,
    "reasoning_gap": 0.25,
    "tool_skip": 0.30,
})
flywheel_adversarial_per_run: int = 0  # 0=不自动生成；正整数=每次 run 结束生成对应数量对抗 case 进待审核队列
```
`generate_adversarial_cases()` 的触发逻辑：`run_all()` 完成后，若 `config.flywheel_adversarial_per_run > 0`，调用一次生成并写入待审核队列（独立于失败模式阈值触发，每次 run 都执行）。

**4. `test_runner.py` 尾部调用飞轮**（`enable_flywheel` 门控）：
- `run_all()` 完成后，统计本次 run 的失败模式分布
- 调用 `DataFlywheel.check_trigger()`
- 触发后调用 `generate_cases_from_badcases()`，写入 DB（`status="pending_review"`）

**5. `agent_eval_repo.py` 新增方法**：
- `list_pending_cases()` — 查 `status="pending_review"` 的 case
- `approve_pending_case(case_id)` — 改 `status="active"`
- `reject_pending_case(case_id)` — 改 `status="rejected"`

**6. `agent_eval_router.py` 新增端点**：
- `GET /api/eval/agent/pending-cases` — 待审核 case 列表
- `POST /api/eval/agent/pending-cases/{id}/approve`
- `POST /api/eval/agent/pending-cases/{id}/reject`
- `POST /api/eval/agent/data-flywheel/trigger` — 手动触发飞轮（指定 eval_run_id）

### 影响文件

- 新建：`backend/nanobot/eval/data_flywheel.py`
- 新建：`backend/migrations/add_pending_cases_fields.sql`
- `backend/nanobot/storage/models.py` — `AgentTestCase` 新增三个字段
- `backend/nanobot/storage/repositories/agent_eval_repo.py` — 新增三个 repo 方法
- `backend/nanobot/server/routers/agent_eval_router.py` — 新增四个端点
- `backend/nanobot/eval/test_runner.py` — 尾部追加飞轮调用（门控）

### 预期效果

每次 eval run 后自动输出待审核 case，人工 approve 后加入正式测试集；`retrieval_failure` 类 case 积累越多，下一轮 eval 的检索评估越全面，形成正向飞轮。

---

## 执行顺序

强依赖链：
- 方向一（semantic matching）→ 方向六（tool param，复用 embedding_fn）
- 方向四（faithfulness）→ 方向七（失败模式分类使用 faithfulness_score）
- 方向七（失败模式）→ 方向九（飞轮触发基于失败模式统计）
- 方向二（G-Eval）→ 方向三（self-consistency，基于同一 score() 接口）

建议批次：
1. **批次 A**（独立，先上）：方向五（contextual_recall）、方向八（多轮矛盾检测）
2. **批次 B**（依赖 embedding）：方向一 → 方向六
3. **批次 C**（依赖 Judge 结构）：方向二 → 方向三 → 方向四
4. **批次 D**（依赖 B+C）：方向七
5. **批次 E**（依赖 D）：方向九

---

## 验证方式

- 方向一：跑现有测试集，对比 `keyword_coverage` 前后分布；人工抽查 10 条中文回答 + 英文 keyword 的 case
- 方向二/三：对比同一 case 多次 Judge 分数方差；检查 `judge_metadata.raw_output` 里 criteria 字段是否存在
- 方向四：在已知有幻觉的 badcase 上验证 `faithfulness_score` 是否低于 0.4
- 方向五：对 retrieval_failure 类 badcase 验证 `contextual_recall` 是否低于 `keyword_coverage`
- 方向六：构造一条 `web_search.query` 和 user_input 完全无关的 case，验证 `tool_skip=0.5`
- 方向七：跑一批历史 badcase，验证 `retrieval_failure / hallucination / reasoning_gap` 的分布合理性
- 方向八：构造两轮对话，第二轮 assistant 回答与第一轮矛盾，验证 `multi_turn_coherence < 0.5`
- 方向九：手动调用 `/data-flywheel/trigger`，验证 pending_cases 写入 DB；approve 后验证可进入 eval run
