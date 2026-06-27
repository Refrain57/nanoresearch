# RootCauseClassifier 实现计划

## 背景

badcase 优化流水线当前问题：所有 badcase 默认归因为 prompt 问题，非 prompt 类根因（tool 报错、context 召回不足等）被错误送入 optimizer，sandbox replay 因工具返回值是录制好的会误判"通过"，但上线后生产环境仍然失败。

**目标**：在 badcase 进入 optimizer 之前，增加归因分类层，只有 `root_cause_auto == "prompt"` 的 badcase 才进入 optimizer。

---

## 架构决策

### 合并进 BadcaseClassifier，不新建文件

现有 `badcase_classifier.py` 已做语义分类（`semantic_category`），新增根因分类（`root_cause_auto`）。

两者输入材料完全相同（tool_call_chain + final_response + judge 信息），合并进一次 LLM 调用，同时输出两个字段，避免重复调用和职责分散。

### 分层分类策略

1. **规则层**（先跑，成本零，可靠）：命中直接返回，跳过 LLM
2. **LLM 层**（处理规则覆盖不到的模糊案例）：一次调用同时输出 `semantic_category + root_cause_auto + confidence + reason`
3. **人工抽检**：定期复核 LLM 归因结果（前两周目标 100 条），评估分类器 F1，发现系统性偏差

### 保守门控策略

`confidence == "low"` 时默认不进 optimizer，降低因分类错误导致无效优化的风险。

---

## 根因枚举值

| root_cause_auto | 含义 |
|---|---|
| `prompt` | Agent 推理方式或行为模式有问题，改 system prompt 有效 |
| `context` | ContextBuilder 召回的上下文不完整或有误 |
| `tool` | 工具返回错误数据、空结果、解析失败 |
| `model` | 模型能力边界，复杂推理或长上下文丢失信息 |
| `user_input` | 用户表达歧义，根因在输入侧 |

---

## 改动清单

### 1. `backend/nanobot/eval/badcase_classifier.py`（扩展）

- 新增 `ClassifyResult` dataclass，字段：`semantic_category: str`、`root_cause_auto: str`、`confidence: str`（high/medium/low）、`reason: str`
- 新增 `_rule_based_root_cause(snapshot) -> ClassifyResult | None` 规则层函数
- LLM call 改为输出 JSON，一次返回两个分类 + confidence + reason
- `classify()` 返回类型从 `str` 改为 `ClassifyResult`

**规则层逻辑**：
```
tool_call_chain 中任何 entry["error"] == True
  → root_cause_auto = "tool", confidence = "high"

tool_call_chain 非空 但所有工具返回值为空串/null/[]
  → root_cause_auto = "context", confidence = "high"

run_status == "failed" 且 tool_call_chain 为空
  → root_cause_auto = "prompt", confidence = "medium"

total_input_tokens >= 0.85 * 模型上限（128000）
  → root_cause_auto = "model", confidence = "high"
```

**LLM Prompt 设计要点**：
- Section 1：用户输入 + 工具调用链摘要（每条截至 500 字符）+ 最终回复 + judge 失分维度 + judge 原始评语
- Section 2：各根因的判断锚点定义，重点说明 context vs prompt 边界：
  > "如果工具确实返回了内容，但 Agent 忽略了或错误解读了这些内容，归 prompt；如果工具返回内容本身就不足以回答问题（例如检索结果为空、或完全无关），归 context。"
- 要求 LLM 先 chain-of-thought（2-3 句），再输出 JSON：
  ```json
  {
    "semantic_category": "...",
    "root_cause_auto": "...",
    "confidence": "high|medium|low",
    "reason": "一句话理由"
  }
  ```
- `confidence = "low"` 的情况：LLM 认为 context 和 prompt 难以区分时，归 context（保守策略）

### 2. `backend/nanobot/storage/models.py`（新增三列）

在 `AgentRunSnapshot` 中新增：
```python
root_cause_auto: Mapped[str | None] = mapped_column(String(32), nullable=True)
root_cause_auto_confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)
root_cause_auto_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
```

> 已有 `root_cause` 字段用于手工标注，`root_cause_auto` 独立存放，不覆盖人工数据，便于后续做一致性比对评估分类器准确率。

### 3. `backend/nanobot/storage/repositories/agent_eval_repo.py`（新增方法）

新增 `update_snapshot_classification(snapshot_id: UUID, result: ClassifyResult) -> None`，同时写：
- `semantic_category`（原逻辑）
- `root_cause_auto`
- `root_cause_auto_confidence`
- `root_cause_auto_reason`

### 4. `backend/nanobot/server/routers/agent_eval_router.py`（两处改动）

**改动 A：`classify_badcases_batch` 的 `_run()` 函数**（约 681 行）

```python
# 改前
category = await classifier.classify(snap)
await repo.update_snapshot_semantic_category(snap.id, category)

# 改后
result = await classifier.classify(snap)
await repo.update_snapshot_classification(snap.id, result)
```

**改动 B：`/optimize` 端点加门控**

在调用 `optimizer.generate_proposals()` 之前，过滤 snapshots：
```python
prompt_snaps = [s for s in snapshots if s.root_cause_auto == "prompt" or s.root_cause_auto is None]
# root_cause_auto 为 None 表示尚未分类，降级处理允许进入（保持现有行为）
if not prompt_snaps:
    raise HTTPException(400, "所选 badcase 均非 prompt 类根因，无需 prompt 优化")
```

### 5. 数据库迁移 SQL（手写，无 Alembic）

项目使用 `Base.metadata.create_all`，新列不会自动加到存量数据库，需手动执行：

```sql
ALTER TABLE agent_run_snapshots
  ADD COLUMN IF NOT EXISTS root_cause_auto VARCHAR(32),
  ADD COLUMN IF NOT EXISTS root_cause_auto_confidence VARCHAR(16),
  ADD COLUMN IF NOT EXISTS root_cause_auto_reason VARCHAR(500);
```

建议将此 SQL 存为 `backend/migrations/add_root_cause_auto.sql`，和 models.py 改动一起提交。

---

## 风险点

1. **context vs prompt 边界模糊**（最主要风险）：LLM 分辨不清时 `confidence = "low"`，默认归 context 不进 optimizer
2. **多因叠加**：tool 报错同时 prompt 也有问题，规则层只看第一个命中的根因。可后续扩展 `secondary_cause` 字段，路由逻辑只看 `root_cause_auto`
3. **sandbox 误判封住**：分类器建成后，非 prompt 类不进 optimizer，sandbox 录制回放误判问题从结构上解决
4. **冷启动无标注数据**：前两周对所有 `confidence != "high"` 的结果做人工复核，积累 100 条后统计 F1

---

## 唯一破坏性调用点

`agent_eval_router.py:681` — 已在改动 B 中一并修复，无其他调用点。

`mineru_test.py` 和 `mineru_loader.py` 中的 `.classify()` 是 `ds` 对象方法，与 `BadcaseClassifier` 无关。
