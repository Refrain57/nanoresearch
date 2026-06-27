# Agent 评测系统修复计划

## 背景

对照《Agent评测体系方案》文档与代码实际实现，发现 5 个关键差距。本计划按顺序修复，每个问题完成后验证再继续下一个。

修复顺序：低风险 → 核心路径 → 工具链。

---

## 问题 3：Badcase 检测默认关闭 + is_failure 未覆盖 max_iterations

### 改动内容

| 改项 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 默认值改为 true | `agent/loop.py` | 20 | `_EVAL_BADCASE_DETECTION` 默认值 `"false"` → `"true"` |
| 默认值改为 true | `agent/loop.py` | 555 | `_save()` 内第二处环境变量读取同步改为 `"true"` |
| is_failure 加 max_iterations | `agent/loop.py` | 529 | `is_failure` 集合加入 `"max_iterations"` |

### 影响

- 默认开启 Badcase 检测后，所有失败/超时/max_iterations 的 run 自动写入 `is_badcase=True`
- `max_iterations` 加入 `is_failure` 后不受采样率控制（全量记录），与方案"失败全量"一致

---

## 问题 1：生产路径接沙箱录制

### 改动内容

- `agent/loop.py` `_run_agent_loop`：生产路径的 `AgentRunSpec(tools=self.tools)` 改为用 `SandboxedToolRegistry(self.tools, mode="record")` 包裹
- `agent/loop.py` `_maybe_save_snapshot`：签名增加 `tool_recordings` 参数，传入 `save_snapshot`

### 影响

- 生产快照的 `tool_recordings` 不再为空，后续 replay 和 optimizer 功能可用

---

## 问题 2：sandbox.py 录制截断与容错

### 改动内容

- `eval/sandbox.py` `SandboxedToolRegistry`：record 模式下结果截断、条目上限、`export_recordings`/`from_recordings_json` 容错

### 影响

- 防止大工具返回值撑爆内存和数据库 JSONB 列

---

## 问题 4：optimizer.py 回放 registry 传 None

### 改动内容（方案 A）

- `eval/optimizer.py` `OptimizationAgent.__init__` 增加 `registry: ToolRegistry` 参数
- `_score_candidate` 传入真实 registry 给 `from_recordings_json`
- `server/routers/agent_eval_router.py` 调用时传入 `channel_loop.tools`

### 影响

- 优化 Agent 跑分时 LLM 能看到工具定义，不再静默跳过所有测试用例

---

## 问题 5：sandbox_mode 默认 + 字段缺失

### 改动内容

| 改项 | 文件 | 说明 |
|------|------|------|
| sandbox_mode 默认 record | `eval/test_runner.py:28` + `agent_eval_router.py:205` | `"passthrough"` → `"record"` |
| system_prompt_version 写入 | `agent/loop.py` `_maybe_save_snapshot` | 传入 `save_snapshot(system_prompt_version=...)` |
| judge_metadata 写入（方案 A） | `eval/judge.py` `LLMJudge.score()` 返回 `tuple[dict, str]` | 返回 (scores, raw_output) |

### 影响

- 评测任务默认录制工具返回值，replay 功能可用
- 快照记录 prompt 版本，支持版本追踪
- Judge 原始输出可查，支持人工审查

---

## 完成状态（2026-06-16）

### 验证结果

| 验证项 | 预期 | 实际 |
|--------|------|------|
| `tool_recordings` 录制 | JSON 对象，含工具调用记录 | ✅ 正常写入（工具调用用例录制成功） |
| `system_prompt_version` 生产路径 | `"production"` | ✅ 正常写入 |
| `system_prompt_version` 离线评测 | `"production"` | ✅ 正常写入（修复后） |
| `judge_metadata` 写入 | JSON 对象，含 raw_output | ✅ 正常写入 |
| `EVAL_BADCASE_DETECTION_ENABLED` 默认值 | `true` | ✅ 已改为 `true` |
| `is_failure` 含 `max_iterations` | 是 | ✅ |
| `sandbox_mode` 默认值 | `"record"` | ✅ |
| Judge 模型默认值 | 随 provider 默认 | ✅ 不再硬编码 `claude-sonnet-4-6` |
| `BadcaseDetector` 捕获 `max_iterations` | 是 | ✅ |

### 计划外补充修复

| 修复 | 文件 | 原因 |
|------|------|------|
| `test_runner.py` 传入 `system_prompt_version` | `eval/test_runner.py` | 评测快照该字段为 NULL |
| `LLMJudge` 默认模型使用 `provider.get_default_model()` | `eval/judge.py` | 硬编码 `claude-sonnet-4-6` 在部分环境不可用 |
| `BadcaseClassifier` 默认模型同上 | `eval/badcase_classifier.py` | 同上 |
| `OptimizationAgent` 移除 `_DEFAULT_MODEL` | `eval/optimizer.py` | 同上 |
| API 路由 judge_model 默认值 `None` | `server/routers/agent_eval_router.py` | 允许默认模型回退到 provider 默认 |
| 记录 judge 实际使用的模型名到 metadata | `server/routers/agent_eval_router.py` | metadata 不记录 null |
| 删除 "timeout" 死代码 | `agent/loop.py` `eval/test_runner.py` | AgentRunner 从不产生 timeout stop_reason |

### 数据库验证 SQL

```sql
-- 1. 查看 tool_recordings / system_prompt_version / judge_metadata
SELECT id, substr(user_input, 1, 30) AS input,
       system_prompt_version,
       jsonb_typeof(tool_recordings) AS rec_type,
       jsonb_typeof(judge_metadata) AS judge_type,
       (SELECT count(*) FROM jsonb_object_keys(tool_recordings)) AS rec_count
FROM agent_run_snapshots
ORDER BY timestamp DESC
LIMIT 5;

-- 2. 查看 Badcase
SELECT id, run_status, is_badcase, badcase_trigger, badcase_category
FROM agent_run_snapshots
WHERE is_badcase = true
ORDER BY timestamp DESC
LIMIT 10;

-- 3. 查看 Optimization Proposal 分数
SELECT id, category, status, created_at,
       proposals->0->>'mean_score' AS c1,
       proposals->1->>'mean_score' AS c2
FROM optimization_proposals
ORDER BY created_at DESC
LIMIT 5;
```

### 未覆盖的功能缺口（不在修复计划范围内）

| 功能 | 方案位置 | 说明 |
|------|----------|------|
| 用户行为触发 Badcase | 方案 2.3 | 用户踩踏/重复提问/中途放弃打标，当前只支持规则触发 |
| 高风险场景 100% 采样 | 方案 2.2 | 按输入内容类别提高采样率，当前只区分失败/成功 |
