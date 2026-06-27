# 四类根因闭环实现方案

## 背景

badcase 经 RootCauseClassifier 归因后，非 prompt 类根因（context / tool / user_input / model）
被 optimizer 门控拦住，但拦住之后没有后续处理——信息断路，无法产生决策依据。

本方案为这四类根因各增加一条后续处理路径。

---

## 改动总览

| 根因 | 新端点 | 说明 |
|---|---|---|
| context | `POST /agent-eval/badcases/context-diagnosis` | 重跑检索，区分 kb_gap vs transient |
| tool | `GET /agent-eval/tool-error-stats?days=N` | SQL 聚合报错率，附 sample_errors |
| user_input | `POST /agent-eval/badcases/user-input-analysis` | LLM 分析歧义模式，给 system prompt 建议 |
| model | `GET /agent-eval/badcases/model-analysis?limit=N` | LLM 分析 token/推理失败，给架构层建议 |

**不写回 DB 的说明**：`user_input` 和 `model` 两个分析接口是即时诊断报告，前端拿到展示即可，无需持久化。如后续需要趋势分析再加存储。`context-diagnosis` 同理。

---

## 端点 1：`POST /agent-eval/badcases/context-diagnosis`

### 新建文件：`backend/nanobot/eval/context_diagnoser.py`

```python
class ContextDiagnoser:
    async def diagnose(
        self,
        snapshot: AgentRunSnapshot,
        kb_list: list[KnowledgeBase],
        session_factory,
        top_k: int = 5,
    ) -> dict:
```

**流程：**

1. 从 `tool_call_chain` 筛出 `_is_empty_output(entry["result"])` 为 True 的条目
2. 对每条空结果工具调用，按候选 key 顺序提取 query：
   ```python
   _QUERY_KEYS = ["query", "q", "search_query", "text", "keyword"]
   query = next((entry["params"][k] for k in _QUERY_KEYS if k in entry.get("params", {})), None)
   ```
3. **query 为 None 时**：该工具调用记 `verdict="unknown"`，跳过检索，计入 `unknown_count`
4. query 不为 None 时：对每个 KB 调 `await hybrid.async_search(query, top_k=top_k)`
   - 任意 KB 返回 `now_count > 0` → `verdict="transient"`
   - 所有 KB 均返回 0 结果 → `verdict="kb_gap"`
5. 汇总：`kb_gap_count / transient_count / unknown_count`

**复用：**
- `_is_empty_output()` from `backend/nanobot/eval/badcase_classifier.py`
- HybridSearch 实例化模式 from `backend/nanobot/server/routers/knowledge_router.py:401-415`
  ```python
  # 关键：用 await hybrid.async_search()，不是 search()
  hybrid = HybridSearch(
      settings=settings, query_processor=QueryProcessor(),
      dense_retriever=dense, sparse_retriever=sparse,
      fusion=RRFFusion(), config=HybridSearchConfig(fusion_top_k=top_k),
      session_factory=session_factory, kb_id=kb.id,
  )
  results = await hybrid.async_search(query, top_k=top_k)
  ```
- `AgentRepository.list_bound_kbs(agent_id)` from `backend/nanobot/storage/repositories/agent_repo.py:157`

**端点实现位置：** `backend/nanobot/server/routers/agent_eval_router.py`

```python
class ContextDiagnosisRequest(BaseModel):
    snapshot_ids: list[uuid.UUID] | None = None  # None → 自动取最近 20 条 root_cause_auto=context
    top_k: int = 5
```

- `snapshot_ids` 为 None 时：`repo.list_badcases(root_cause_auto="context", page_size=20)`
- 依赖注入：`request.app.state.session_factory` 传给 ContextDiagnoser；`AgentRepository(get_session_factory())` 取 KB 列表

**返回：**
```json
{
  "summary": {"kb_gap": 3, "transient": 2, "unknown": 1},
  "items": [
    {
      "snapshot_id": "...",
      "user_input": "...(120 chars)",
      "queries": [
        {"tool_name": "retrieve_docs", "query": "什么是 RAG？",
         "kb_name": "产品文档", "now_count": 0, "verdict": "kb_gap"},
        {"tool_name": "retrieve_docs", "query": null,
         "kb_name": null, "now_count": null, "verdict": "unknown"}
      ]
    }
  ]
}
```

---

## 端点 2：`GET /agent-eval/tool-error-stats?days=7`

**实现位置：** `backend/nanobot/server/routers/agent_eval_router.py`

**两步查询（复用 `backend/nanobot/storage/repositories/run_repo.py:71-84` 的 jsonb_array_elements 模式）：**

步骤 A — 聚合统计（raw SQL via `get_session_factory()` + `text()`）：
```sql
SELECT
    entry->>'name'                                        AS tool_name,
    COUNT(*)                                              AS total_calls,
    SUM((entry->>'error' = 'true')::int)                  AS error_calls,
    ROUND(AVG((entry->>'error' = 'true')::int)::numeric * 100, 1) AS error_rate_pct
FROM agent_run_snapshots,
     jsonb_array_elements(tool_call_chain) AS entry
WHERE timestamp > NOW() - CAST(:interval AS INTERVAL)
  AND jsonb_typeof(tool_call_chain) = 'array'
  AND jsonb_array_length(tool_call_chain) > 0
GROUP BY tool_name
ORDER BY error_rate_pct DESC
```
参数：`{"interval": f"{days} days"}`

步骤 B — 采样错误：
```sql
SELECT entry->>'name' AS tool_name, entry->>'result' AS result_text, timestamp
FROM agent_run_snapshots,
     jsonb_array_elements(tool_call_chain) AS entry
WHERE timestamp > NOW() - CAST(:interval AS INTERVAL)
  AND entry->>'error' = 'true'
ORDER BY timestamp DESC
LIMIT 200
```
Python 按 tool_name 分组，每组取前 3 条 result（截断 200 字符）。

**severity 规则：**
- `error_rate_pct > 20` → `"red"`
- `error_rate_pct > 5` → `"orange"`
- 其余 → `"green"`

**返回：**
```json
{
  "window_days": 7,
  "tools": [
    {"tool_name": "read_file", "total_calls": 45, "error_calls": 12,
     "error_rate": 0.267, "severity": "red",
     "sample_errors": ["Error: file not found /tmp/abc", "Error: permission denied"]}
  ]
}
```

---

## 端点 3：`POST /agent-eval/badcases/user-input-analysis`

**实现位置：** `backend/nanobot/server/routers/agent_eval_router.py`（内联，无新文件）

```python
class UserInputAnalysisRequest(BaseModel):
    snapshot_ids: list[uuid.UUID] | None = None  # None → 自动取最近 20 条 root_cause_auto=user_input
```

**流程：**
1. 取快照，收集 `user_input`（去重，截 300 字符，最多 20 条）
2. 调 `request.app.state.channel_loop.provider.chat_with_retry()`
3. System prompt：
   ```
   你是 Agent 系统优化专家。以下是一批因用户表达歧义导致失败的对话输入，
   请先用 2-3 句分析共同模式，然后输出 JSON：
   {"patterns": ["..."], "prompt_suggestions": [{"issue": "...", "suggestion": "..."}]}
   suggestion 要具体，格式为"在 system prompt 加入：当用户 X 时，先追问 Y"
   ```
4. 解析：`re.search(r'\{.*\}', raw, re.DOTALL)` + `json.loads()`（参考 `backend/nanobot/eval/optimizer.py` 模式）
5. 解析失败 → `{"error": "parse_failed", "raw": raw[:500]}`

**不写回 DB**——即时报告，前端展示即可。

**返回：**
```json
{
  "analyzed_count": 18,
  "patterns": ["用户省略主语", "时态歧义导致工具选择错误"],
  "prompt_suggestions": [
    {"issue": "省略主语", "suggestion": "在 system prompt 加入：当用户未指定主体时，先追问"您是指哪个项目/产品？""}
  ]
}
```

---

## 端点 4：`GET /agent-eval/badcases/model-analysis?limit=20`

**实现位置：** `backend/nanobot/server/routers/agent_eval_router.py`（内联，无新文件）

**流程：**
1. `repo.list_badcases(root_cause_auto="model", page_size=limit)`
2. 构建 payload（每条快照提取）：
   - `user_input[:200]`, `total_input_tokens`, `tool_call_count`, `llm_call_count`, `run_status`
   - 从 `llm_calls` 取 `max(c["input_tokens"] for c in llm_calls)` → 哪次 LLM call token 最多
3. LLM call（同 provider）：
   ```
   以下是一批因模型能力边界失败的 Agent 运行记录（token 超限 / 复杂推理失败），
   请先 2-3 句分析失败模式，然后输出 JSON（止步于决策依据，不做自动化）：
   {"pattern_report": "...", "recommendations": [{"type": "...", "detail": "..."}]}
   type 枚举：summarization_layer | context_window_limit | model_upgrade | prompt_compression | other
   ```
4. 同时计算 `token_stats`：`mean(total_input_tokens)`, `max(total_input_tokens)`（Python 计算，不走 LLM）

**不写回 DB**——即时报告，前端展示即可。

**返回：**
```json
{
  "analyzed_count": 12,
  "token_stats": {"mean": 112000, "max": 127500},
  "pattern_report": "12 条记录中 8 条为长上下文对话（输入 > 100k token），4 条为多步推理任务...",
  "recommendations": [
    {"type": "summarization_layer", "detail": "对超 80k token 的对话历史先做摘要再送入模型"},
    {"type": "context_window_limit", "detail": "设置 max_context_tokens=100000，强制截断"}
  ]
}
```

---

## 文件清单

| 文件 | 改动 |
|---|---|
| `backend/nanobot/eval/context_diagnoser.py` | **新建** |
| `backend/nanobot/server/routers/agent_eval_router.py` | 新增 4 个端点 |
| `backend/nanobot/storage/repositories/agent_eval_repo.py` | 无需修改（`list_badcases` 已支持 `root_cause_auto` 过滤） |
| `backend/nanobot/storage/models.py` | 无需修改 |

---

## 复用清单

| 复用点 | 来源文件 |
|---|---|
| `_is_empty_output()` | `backend/nanobot/eval/badcase_classifier.py` |
| HybridSearch 实例化 + `async_search` | `backend/nanobot/server/routers/knowledge_router.py:401-415` |
| `AgentRepository.list_bound_kbs()` | `backend/nanobot/storage/repositories/agent_repo.py:157` |
| `jsonb_array_elements` SQL 模式 | `backend/nanobot/storage/repositories/run_repo.py:71-84` |
| LLM JSON 解析 (`re.search` + `json.loads`) | `backend/nanobot/eval/optimizer.py` |
| `channel_loop.provider` | `request.app.state.channel_loop`（同 classify-batch 端点） |
| `get_session_factory()` | `backend/nanobot/storage/database.py` |

---

## 验证方法

1. **context-diagnosis**：手动准备几条 `root_cause_auto=context` 的 badcase（其中一条对应的 KB 确实有内容，另一条 KB 无内容），调接口验证 transient vs kb_gap 分类正确
2. **tool-error-stats**：`?days=30`，确认有报错记录的工具出现在结果中，`sample_errors` 内容是真实错误信息，非 `error=False` 的工具调用不计入
3. **user-input-analysis / model-analysis**：验证 LLM JSON 被正确解析；故意让 LLM 输出格式错误，确认降级为 `{"error": "parse_failed", "raw": ...}` 而非 500
4. 所有文件语法检查：`python -c "import ast; ast.parse(open(f, encoding='utf-8').read())"`
