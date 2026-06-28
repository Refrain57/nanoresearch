# RAG 查询改写 — A 类透传链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pass `session_key` from main agent through MCPToolWrapper → `kb_search` MCP subprocess → `RAGSearchTool` → `RAGLoopRunner` → `PlanQueryTool`, then have `PlanQueryTool` fetch real PG-backed session history asynchronously and rewrite indirect-reference queries (e.g. "它" / "那个") before retrieval.

**Architecture:** 三段。 (1) **客户端注入**：主 Agent 已通过 `set_session_key()` 把 `channel:chat_id` 存到 wrapper 实例，`MCPToolWrapper.execute()` 在调用 `kb_search/kb_retrieve` 时把它写进 kwargs。 (2) **子进程透传 7 处**：`RAGSearchTool` input_schema / execute / handler / `_execute_complex` → `RAGLoopRunner.run` → `SessionState.caller_session_key` → `_run_plan_phase` → `InternalTools.plan_query` → `PlanQueryTool.execute`（最后一段已接好）。 (3) **子进程自建 PG-backed `SessionManager`**：子进程通过 `_stdio_env` 透传到的 `DATABASE_URL` 直连同一个 PG，**禁止 JSONL 孤儿**；`_get_conversation_history` 改 async，返回 `list[dict]`（含 user/assistant/tool 消息），渲染层只把 user+assistant 写进 prompt。

**Tech Stack:** Python 3.11+, MCP stdio subprocess, SQLAlchemy 2.x async (PG), pytest（`asyncio_mode=auto`），loguru。

## Global Constraints

- **路径基准**：`nanobot → nanoresearch` rename 已落地（2026-06-28），spec 写的 `backend/nanobot/...` 全部平移到 `backend/nanoresearch/...`。
- **工作分支**：`feature/rag-query-rewrite-a-class`，base on `origin/main`（本地 main HEAD `f5284059`）。
- **前置假设**：`fix/kb-search-prompt-typo`（独立 PR，HEAD `008e6b27`）已合或将合，已修 `agent/context.py:279`。本 plan T0 只清剩下 2/3 处 rename 残留。
- **前置已 ship**：§5.2 MCPToolResponse 裸 JSON 修复 (`3c82e50d`)，本 plan 依赖 role:"tool" content 是裸 JSON 这一事实，但 A 类自身不读 tool content。
- **测试运行**：`cd backend && pytest tests/...`（`asyncio_mode=auto` 已配，无需 `@pytest.mark.asyncio`）。
- **A 类不动 B 类范围**：不修 §4.2 `_chunk_titles` sidecar 写入路径，不动 `backend/nanoresearch/session/manager.py:67-87` `Session.get_history()` 白名单。`_get_retrieval_titles` 实现但在 A 类期间永远返回 `[]`，B 类才喂数据——这是预期行为。
- **A 类不动已接好的下游**：`PlanQueryTool.input_schema` / `PlanQueryTool.execute` / `plan_query_handler` 三处已含 session_key（`query_planning.py:172-190 / 192-271 / 891-902`），**不要改**。
- **rewrite 短路保留**：当 `history` 和 `retrieval_titles` 都为空时不调 LLM，直接返回原 query。
- **降级而不出错**：拿不到 SessionManager → warn + 返回 `[]`，rewrite 跑原 query。**禁止**回退到 JSONL。
- **commit message 中文为主，subject 用 conventional commits 前缀**：`feat(rag)`, `fix(rag)`, `refactor(rag)`, `test(rag)`。

---

## File Structure

| 文件 | 责任 | 任务 |
|---|---|---|
| `backend/nanoresearch/rag/mcp_server/tools/__init__.py` | 清理 `register_rag_search_tools` 别名 + 注释 | T0 |
| `backend/nanoresearch/rag/internal_loop/state.py` | `SessionState.caller_session_key` 字段 + `SessionStateManager.create_session` 透传；docstring rename | T0 + T1 |
| `backend/nanoresearch/rag/mcp_server/tools/rag_search.py` | `RAGSearchTool` 暴露 `session_key`，向下游透传到 `run_rag_loop` | T2 |
| `backend/nanoresearch/rag/internal_loop/runner.py` | `RAGLoopRunner.run()` 写 `session.caller_session_key`；`_run_plan_phase` 读出并 forward | T3 + T5 |
| `backend/nanoresearch/rag/internal_loop/tools.py` | `InternalTools.plan_query` 加 `session_key`；`get_plan_tool_schema` parameters 加字段 | T4 |
| `backend/nanoresearch/agent/tools/mcp.py` | `MCPToolWrapper.execute` 对 `kb_search`/`kb_retrieve` 注入 `session_key` | T6 |
| `backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py` | 模块级 PG-backed `SessionManager` 工厂；`_get_conversation_history` 异步 + 返回 `list[dict]`；新增 `_render_history_for_prompt` / `_get_retrieval_titles`；`REWRITE_PROMPT` 改造；`_rewrite_query` 新签名；`execute()` 调用点改 await | T7 + T8 |
| `backend/tests/unit/rag/test_query_rewrite_a_class.py`（新建）| T1-T8 各任务单元测试 | T1-T8 |
| `backend/tests/integration/test_query_rewrite_e2e.py`（新建）| 合成多轮 + 指代消解 smoke test | T9 |

---

## 创建工作分支

提示：实施开始前，先建分支：

```bash
git checkout main && git checkout -b feature/rag-query-rewrite-a-class
```

---

### Task 0: 清理 rename 残留（2/3）

**Files:**
- Modify: `backend/nanoresearch/rag/mcp_server/tools/__init__.py:52-97`
- Modify: `backend/nanoresearch/rag/internal_loop/state.py:74-77`

**Interfaces:**
- Consumes: 无（独立清理）
- Produces: 移除符号 `register_rag_search_tools`，新增同义 `register_kb_search_tools`。下游目前唯一引用方 `rag/mcp_server/protocol_handler.py:215` 用的是 `from ... .tools.rag_search import register_tools as register_kb_search_tools`（直连模块），不依赖 `tools/__init__.py` 的 re-export，不会被影响。

- [ ] **Step 1: 改 `tools/__init__.py` 的 re-export 别名 + 注释**

`backend/nanoresearch/rag/mcp_server/tools/__init__.py:52-56`：

```python
# kb_search - unified entry point
from nanoresearch.rag.mcp_server.tools.rag_search import (
    RAGSearchTool,
    register_tools as register_kb_search_tools,
)
```

同文件 `__all__` 末尾 L95-97：

```python
    # kb_search - unified entry point
    "RAGSearchTool",
    "register_kb_search_tools",
```

- [ ] **Step 2: 改 `state.py:76` docstring**

`backend/nanoresearch/rag/internal_loop/state.py:74-77`：

```python
    """State for an entire RAG search session.

    This holds all state for a single kb_search call,
    including the original query, context, and all rounds.
```

- [ ] **Step 3: 验证导入仍成立**

```bash
cd backend && python -c "from nanoresearch.rag.mcp_server.tools import register_kb_search_tools; print(register_kb_search_tools.__name__)"
```

Expected output: `register_tools`

- [ ] **Step 4: Commit**

```bash
git add backend/nanoresearch/rag/mcp_server/tools/__init__.py backend/nanoresearch/rag/internal_loop/state.py
git commit -m "refactor(rag): rag_search → kb_search rename 残留 2 处清理"
```

---

### Task 1: `SessionState.caller_session_key` 字段 + `create_session` 透传

**Files:**
- Modify: `backend/nanoresearch/rag/internal_loop/state.py:72-119,210-236`
- Test: `backend/tests/unit/rag/test_query_rewrite_a_class.py`（新建）

**Interfaces:**
- Consumes: 无
- Produces:
  - `SessionState(caller_session_key: Optional[str] = None)` — 新字段，位于 `context` 之后、`plan` 之前。
  - `SessionStateManager.create_session(query, context=None, max_iterations=5, caller_session_key=None)` — 新增 kwarg。

**说明**：字段名用 `caller_session_key` 而不是 `session_key`——`SessionState` 内部已有 `session_id`，外层 `session_key=channel:chat_id` 是另一个概念，名字必须显式区分（§8 #10）。

- [ ] **Step 1: 写失败测试**

新建 `backend/tests/unit/rag/__init__.py`（空文件）和 `backend/tests/unit/rag/test_query_rewrite_a_class.py`：

```python
"""A 类透传链 unit tests."""

from nanoresearch.rag.internal_loop.state import SessionState, SessionStateManager


def test_session_state_has_caller_session_key_field():
    """SessionState 必须含 caller_session_key 字段，默认 None。"""
    s = SessionState(session_id="abc", original_query="q")
    assert s.caller_session_key is None


def test_session_state_accepts_caller_session_key():
    s = SessionState(
        session_id="abc",
        original_query="q",
        caller_session_key="telegram:123",
    )
    assert s.caller_session_key == "telegram:123"


def test_create_session_propagates_caller_session_key():
    SessionStateManager.reset_instance()
    mgr = SessionStateManager.get_instance()
    s = mgr.create_session(query="q", caller_session_key="telegram:456")
    assert s.caller_session_key == "telegram:456"


def test_create_session_default_caller_session_key_is_none():
    SessionStateManager.reset_instance()
    mgr = SessionStateManager.get_instance()
    s = mgr.create_session(query="q")
    assert s.caller_session_key is None
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 4 tests FAIL，错误关于 `unexpected keyword argument 'caller_session_key'` 或 `no attribute 'caller_session_key'`。

- [ ] **Step 3: 实现**

`backend/nanoresearch/rag/internal_loop/state.py:88-97` 在 `context` 后新增字段：

```python
@dataclass
class SessionState:
    """State for an entire RAG search session.

    This holds all state for a single kb_search call,
    including the original query, context, and all rounds.

    Attributes:
        session_id: Unique identifier for this session
        original_query: The user's original query
        context: External context from main agent (optional)
        caller_session_key: Outer agent's session key (channel:chat_id) for multi-turn context
        plan: Result from plan_query
        rounds: List of round states
        current_phase: Current phase in the loop
        max_iterations: Maximum iterations allowed
    """
    session_id: str
    original_query: str
    context: Optional[str] = None
    caller_session_key: Optional[str] = None
    plan: Optional[PlanResult] = None
    rounds: List["RoundState"] = field(default_factory=list)
    current_phase: str = "plan"
    max_iterations: int = 5
    iteration: int = 0
    fused_chunks: List[Dict[str, Any]] = field(default_factory=list)
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
```

`backend/nanoresearch/rag/internal_loop/state.py:210-236` `create_session` 改：

```python
    def create_session(
        self,
        query: str,
        context: Optional[str] = None,
        max_iterations: int = 5,
        caller_session_key: Optional[str] = None,
    ) -> SessionState:
        """Create a new session.

        Args:
            query: The original query
            context: External context (optional)
            max_iterations: Maximum iterations
            caller_session_key: Outer agent session key (channel:chat_id), propagated for rewrite

        Returns:
            New SessionState
        """
        session_id = str(uuid.uuid4())[:12]
        session = SessionState(
            session_id=session_id,
            original_query=query,
            context=context,
            caller_session_key=caller_session_key,
            max_iterations=max_iterations,
        )
        self._sessions[session_id] = session

        logger.debug(f"Created session: {session_id}")
        return session
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/rag/internal_loop/state.py backend/tests/unit/rag/__init__.py backend/tests/unit/rag/test_query_rewrite_a_class.py
git commit -m "feat(rag): SessionState 新增 caller_session_key 字段透传外层 session_key"
```

---

### Task 2: `rag_search.py` — `kb_search` 工具暴露并透传 `session_key`

**Files:**
- Modify: `backend/nanoresearch/rag/mcp_server/tools/rag_search.py:111-134,136-174,255-295,299-317`
- Test: `backend/tests/unit/rag/test_query_rewrite_a_class.py`（追加）

**Interfaces:**
- Consumes: 无（用 Task 1 的 `caller_session_key` 通过 Task 3 间接）
- Produces:
  - `RAGSearchTool.input_schema` properties 含 `session_key`（不在 required）。
  - `RAGSearchTool.execute(query, collection, context, max_iterations, session_key)` 接 kwarg。
  - `RAGSearchTool._execute_complex(query, collection, context, max_iterations, session_key)` 接 kwarg 并传给 `run_rag_loop`。
  - `rag_search_handler(query, collection, context, max_iterations, kb_id, session_key)` 接 kwarg 并传给 `tool.execute`。

**说明**：简单路径 `_execute_simple` 不需要 session_key——简单路径走 batch retrieval、不调 plan_query。

- [ ] **Step 1: 写失败测试**

追加到 `backend/tests/unit/rag/test_query_rewrite_a_class.py`：

```python
from nanoresearch.rag.mcp_server.tools.rag_search import RAGSearchTool, rag_search_handler


def test_rag_search_input_schema_has_session_key():
    tool = RAGSearchTool()
    schema = tool.input_schema
    assert "session_key" in schema["properties"]
    assert "session_key" not in schema["required"]
    assert schema["properties"]["session_key"]["type"] == "string"


async def test_rag_search_handler_accepts_session_key(monkeypatch):
    captured = {}

    async def fake_execute(self, query, collection="default", context=None,
                           max_iterations=5, session_key=None):
        captured["session_key"] = session_key
        from nanoresearch.rag.core.response.response_builder import MCPToolResponse
        return MCPToolResponse(content='{"success": true, "chunks": []}')

    monkeypatch.setattr(RAGSearchTool, "execute", fake_execute)
    await rag_search_handler(
        query="q",
        collection="c",
        session_key="telegram:789",
    )
    assert captured["session_key"] == "telegram:789"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py::test_rag_search_input_schema_has_session_key tests/unit/rag/test_query_rewrite_a_class.py::test_rag_search_handler_accepts_session_key -v
```

Expected: FAIL — schema 缺 `session_key`，handler 缺 kwarg。

- [ ] **Step 3: 实现 — input_schema 加字段**

`backend/nanoresearch/rag/mcp_server/tools/rag_search.py:110-134` `input_schema`：

```python
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户查询",
                },
                "context": {
                    "type": "string",
                    "description": "外部上下文，用于解析指代词（如'它'、'这个'）",
                },
                "max_iterations": {
                    "type": "integer",
                    "default": 5,
                    "description": "最大迭代次数（复杂查询时使用）",
                },
                "kb_id": {
                    "type": "string",
                    "description": "知识库 ID，不传时自动搜索默认知识库",
                },
                "session_key": {
                    "type": "string",
                    "description": "Main agent session key (channel:chat_id) for multi-turn context",
                },
            },
            "required": ["query"],
        }
```

- [ ] **Step 4: 实现 — execute() 加 session_key 并传给 _execute_complex**

`backend/nanoresearch/rag/mcp_server/tools/rag_search.py:136-174` `execute`：

```python
    async def execute(
        self,
        query: str,
        collection: str = "default",
        context: Optional[str] = None,
        max_iterations: int = 5,
        session_key: Optional[str] = None,
    ) -> "MCPToolResponse":
        """Execute RAG search.

        Args:
            query: User query
            collection: Collection to search
            context: External context for query rewriting
            max_iterations: Maximum iterations for complex queries
            session_key: Outer agent session key (channel:chat_id), forwarded to
                internal loop / plan_query for history-based query rewrite

        Returns:
            MCPToolResponse with retrieval results
        """
        self._ensure_initialized()

        _, _, classify_complexity, _ = _get_rag_loop_components()

        _log(f"Query: {query[:50]}... (context={context is not None})")

        complexity = classify_complexity(query, context)
        _log(f"Complexity: {complexity}")

        if complexity == "simple":
            _log("Taking simple path (direct retrieval)")
            return await self._execute_simple(query, collection)
        else:
            _log("Taking complex path (internal loop)")
            return await self._execute_complex(
                query, collection, context, max_iterations, session_key
            )
```

- [ ] **Step 5: 实现 — _execute_complex 透传到 run_rag_loop**

`backend/nanoresearch/rag/mcp_server/tools/rag_search.py:255-295` `_execute_complex`：

```python
    async def _execute_complex(
        self,
        query: str,
        collection: str,
        context: Optional[str],
        max_iterations: int,
        session_key: Optional[str] = None,
    ) -> "MCPToolResponse":
        """Execute complex query: Internal loop with verification."""
        try:
            _, _, _, run_rag_loop = _get_rag_loop_components()

            _log("Running internal loop...")
            result = await run_rag_loop(
                query=query,
                context=context,
                collection=collection,
                max_iterations=max_iterations,
                session_key=session_key,
            )

            _log(f"Loop completed: success={result.success}, chunks={len(result.chunks)}")
            return build_json_response(result.to_dict())

        except Exception as e:
            _log(f"Complex retrieval failed: {e}")
            traceback.print_exc(file=sys.stderr)
            return build_json_response({
                "success": False,
                "error": str(e),
                "chunks": [],
            })
```

- [ ] **Step 6: 实现 — rag_search_handler 加 session_key**

`backend/nanoresearch/rag/mcp_server/tools/rag_search.py:298-317` `rag_search_handler`：

```python
async def rag_search_handler(
    query: str,
    collection: str = "default",
    context: Optional[str] = None,
    max_iterations: int = 5,
    kb_id: Optional[str] = None,
    session_key: Optional[str] = None,
) -> "MCPToolResponse":
    """Handler for kb_search MCP tool.

    Note: 'collection' is injected by the main process (mcp.py) and
    should NOT be passed by the Agent. The Agent should use 'kb_id' instead.
    """
    tool = RAGSearchTool()
    return await tool.execute(
        query=query,
        collection=collection,
        context=context,
        max_iterations=max_iterations,
        session_key=session_key,
    )
```

- [ ] **Step 7: 运行测试确认通过**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 全部 PASS（Task 1 + Task 2 共 6 个测试）。

- [ ] **Step 8: Commit**

```bash
git add backend/nanoresearch/rag/mcp_server/tools/rag_search.py backend/tests/unit/rag/test_query_rewrite_a_class.py
git commit -m "feat(rag): kb_search MCP 工具暴露并透传 session_key 到内部 loop"
```

---

### Task 3: `RAGLoopRunner.run()` 接 `session_key` 写入 `SessionState`

**Files:**
- Modify: `backend/nanoresearch/rag/internal_loop/runner.py:190-218`
- Test: `backend/tests/unit/rag/test_query_rewrite_a_class.py`（追加）

**Interfaces:**
- Consumes: `SessionStateManager.create_session(caller_session_key=...)`（Task 1）
- Produces:
  - `RAGLoopRunner.run(query, context, collection, max_iterations, session_key)` — 新 kwarg。
  - `run_rag_loop` 模块级函数同步加 kwarg 并 forward（在 `runner.py` 末尾，需 grep 定位）。

- [ ] **Step 1: 定位 run_rag_loop wrapper**

```bash
cd backend && grep -n "^async def run_rag_loop\|^def run_rag_loop" nanoresearch/rag/internal_loop/runner.py
```

记录行号，后续 Step 4 用。

- [ ] **Step 2: 写失败测试**

追加到 `backend/tests/unit/rag/test_query_rewrite_a_class.py`：

```python
import inspect
from nanoresearch.rag.internal_loop.runner import RAGLoopRunner, run_rag_loop


def test_runner_run_signature_has_session_key():
    sig = inspect.signature(RAGLoopRunner.run)
    assert "session_key" in sig.parameters
    assert sig.parameters["session_key"].default is None


def test_run_rag_loop_signature_has_session_key():
    sig = inspect.signature(run_rag_loop)
    assert "session_key" in sig.parameters
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py::test_runner_run_signature_has_session_key tests/unit/rag/test_query_rewrite_a_class.py::test_run_rag_loop_signature_has_session_key -v
```

Expected: FAIL — `'session_key' not in parameters`。

- [ ] **Step 4: 实现 — `RAGLoopRunner.run` 加 session_key + 写入 session**

`backend/nanoresearch/rag/internal_loop/runner.py:190-218`：

```python
    async def run(
        self,
        query: str,
        context: Optional[str] = None,
        collection: str = "default",
        max_iterations: int = 5,
        session_key: Optional[str] = None,
    ) -> RAGLoopResult:
        """Run the RAG loop.

        Args:
            query: User query
            context: External context from main agent
            collection: Collection to search
            max_iterations: Maximum iterations
            session_key: Outer agent session key (channel:chat_id); propagated to
                plan_query for history-based rewrite. None disables rewrite.

        Returns:
            RAGLoopResult with chunks and citations
        """
        self._ensure_initialized()

        from nanoresearch.rag.internal_loop.state import SessionState

        session = self.session_manager.create_session(
            query=query,
            context=context,
            max_iterations=max_iterations,
            caller_session_key=session_key,
        )
```

- [ ] **Step 5: 实现 — `run_rag_loop` wrapper 加 session_key**

按 Step 1 的行号定位，在该 wrapper 加 `session_key: Optional[str] = None` 参数并 forward 给 `runner.run(...)`。

实施时机：执行者读到该位置（典型形如）：

```python
async def run_rag_loop(query, context=None, collection="default", max_iterations=5):
    runner = RAGLoopRunner()
    return await runner.run(query, context, collection, max_iterations)
```

改为：

```python
async def run_rag_loop(query, context=None, collection="default", max_iterations=5, session_key=None):
    runner = RAGLoopRunner()
    return await runner.run(query, context, collection, max_iterations, session_key=session_key)
```

- [ ] **Step 6: 运行测试确认通过**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add backend/nanoresearch/rag/internal_loop/runner.py backend/tests/unit/rag/test_query_rewrite_a_class.py
git commit -m "feat(rag): RAGLoopRunner.run 接 session_key 并写入 SessionState"
```

---

### Task 4: `InternalTools.plan_query` 透传 + plan_tool schema 暴露字段

**Files:**
- Modify: `backend/nanoresearch/rag/internal_loop/tools.py:100-122,215-259`
- Test: `backend/tests/unit/rag/test_query_rewrite_a_class.py`（追加）

**Interfaces:**
- Consumes: 无（下游 `PlanQueryTool.execute` 已含 session_key）
- Produces:
  - `InternalTools.plan_query(query, context, session_key)` — 新 kwarg。
  - `InternalTools.get_plan_tool_schema()` 返回的 schema parameters 含 `session_key`。

- [ ] **Step 1: 写失败测试**

追加到 `backend/tests/unit/rag/test_query_rewrite_a_class.py`：

```python
from nanoresearch.rag.internal_loop.tools import InternalTools


def test_plan_tool_schema_has_session_key():
    tools = InternalTools()
    schema = tools.get_plan_tool_schema()
    params = schema["function"]["parameters"]
    assert "session_key" in params["properties"]


def test_plan_query_signature_has_session_key():
    sig = inspect.signature(InternalTools.plan_query)
    assert "session_key" in sig.parameters
    assert sig.parameters["session_key"].default is None
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py::test_plan_tool_schema_has_session_key tests/unit/rag/test_query_rewrite_a_class.py::test_plan_query_signature_has_session_key -v
```

Expected: FAIL。

- [ ] **Step 3: 实现 — get_plan_tool_schema 加字段**

`backend/nanoresearch/rag/internal_loop/tools.py:100-122`：

```python
    def get_plan_tool_schema(self) -> Dict[str, Any]:
        """Get schema for plan_query tool."""
        return {
            "type": "function",
            "function": {
                "name": "plan_query",
                "description": "分析查询并分解为子查询，标注每个子查询的检索策略。必须在搜索开始前调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用户查询",
                        },
                        "context": {
                            "type": "string",
                            "description": "外部上下文（可选）",
                        },
                        "session_key": {
                            "type": "string",
                            "description": "外层 agent session key (channel:chat_id)，用于多轮指代消解（可选）",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
```

- [ ] **Step 4: 实现 — plan_query 加 session_key 并 forward**

`backend/nanoresearch/rag/internal_loop/tools.py:215-259`：

```python
    async def plan_query(
        self,
        query: str,
        context: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute plan_query tool.

        Args:
            query: The user query
            context: External context (optional)
            session_key: Outer agent session key for history-based rewrite (optional)

        Returns:
            Plan result with sub_queries and strategy annotations
        """
        self._ensure_initialized()

        try:
            result = await self._plan_tool.execute(
                query=query,
                context=context,
                session_key=session_key,
            )

            if hasattr(result, "content"):
                content = result.content
                if isinstance(content, list):
                    text = ""
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            text += block["text"]
                    return json.loads(text) if text else {}
                elif isinstance(content, str):
                    return json.loads(content)
            return {}

        except Exception as e:
            logger.error(f"plan_query failed: {e}")
            return {
                "complexity": "complex",
                "sub_queries": [
                    {"query": query, "strategy": "hybrid", "reason": "fallback"}
                ],
            }
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/rag/internal_loop/tools.py backend/tests/unit/rag/test_query_rewrite_a_class.py
git commit -m "feat(rag): InternalTools.plan_query 与 plan_tool schema 透传 session_key"
```

---

### Task 5: `_run_plan_phase` 从 SessionState 读取并 forward 到 plan_query

**Files:**
- Modify: `backend/nanoresearch/rag/internal_loop/runner.py:389-405`
- Test: `backend/tests/unit/rag/test_query_rewrite_a_class.py`（追加）

**Interfaces:**
- Consumes: `SessionState.caller_session_key`（Task 1）、`InternalTools.plan_query(session_key=...)`（Task 4）
- Produces: `_run_plan_phase` 通过 `session.caller_session_key` 把 session_key forward 给 plan_query。

- [ ] **Step 1: 写失败测试（用 AsyncMock 拦 plan_query）**

追加到 `backend/tests/unit/rag/test_query_rewrite_a_class.py`：

```python
from unittest.mock import AsyncMock
from nanoresearch.rag.internal_loop.state import SessionState


async def test_run_plan_phase_forwards_session_key():
    """_run_plan_phase 必须从 session.caller_session_key 读出并传给 plan_query。"""
    runner = RAGLoopRunner()
    # avoid heavy init
    runner._initialized = True
    runner.tools = AsyncMock()
    runner.tools.plan_query = AsyncMock(return_value={"sub_queries": [], "complexity": "simple"})

    session = SessionState(
        session_id="abc",
        original_query="它是什么",
        caller_session_key="telegram:42",
    )

    await runner._run_plan_phase(session, messages=[])

    call_kwargs = runner.tools.plan_query.call_args.kwargs
    assert call_kwargs.get("session_key") == "telegram:42"
    assert call_kwargs.get("query") == "它是什么"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py::test_run_plan_phase_forwards_session_key -v
```

Expected: FAIL — `call_kwargs.get("session_key")` 是 `None`。

- [ ] **Step 3: 实现**

`backend/nanoresearch/rag/internal_loop/runner.py:389-405`：

```python
    async def _run_plan_phase(
        self,
        session: "SessionState",
        messages: List[Dict[str, Any]],
    ) -> "PlanResult":
        """Run Phase 1: Plan."""
        from nanoresearch.rag.internal_loop.state import PlanResult, SubQuery

        plan_dict = await self.tools.plan_query(
            query=session.original_query,
            context=session.context,
            session_key=session.caller_session_key,
        )
```

（后续 sub_queries 转换逻辑不变）

- [ ] **Step 4: 运行测试确认通过**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/rag/internal_loop/runner.py backend/tests/unit/rag/test_query_rewrite_a_class.py
git commit -m "feat(rag): _run_plan_phase 从 SessionState 取 caller_session_key 透给 plan_query"
```

---

### Task 6: `MCPToolWrapper.execute` 客户端注入 `session_key`

**Files:**
- Modify: `backend/nanoresearch/agent/tools/mcp.py:128-216`
- Test: `backend/tests/unit/rag/test_query_rewrite_a_class.py`（追加）

**Interfaces:**
- Consumes: `MCPToolWrapper.set_session_key()`（已存在）
- Produces: 调用 `kb_search` / `kb_retrieve` 时，`self._session.call_tool(name, arguments=kwargs)` 的 `kwargs` 含 `session_key`（仅当 wrapper 的 `_session_key` 不为 None）。**其他 MCP 工具不注入**——会被子进程拒绝。

**说明**：spec §3.1 #1 写"对 `kb_search` / `kb_retrieve` / `rag_search` 三个 original_name"——`rag_search` 在 rename 后已不存在 MCP 工具，丢掉。注入位置必须在所有 collection 校验之后、`self._session.call_tool(...)` 之前，避免污染早期分支的 dict-error 返回。

- [ ] **Step 1: 写失败测试（mock self._session.call_tool）**

追加到 `backend/tests/unit/rag/test_query_rewrite_a_class.py`：

```python
from unittest.mock import AsyncMock, MagicMock
import pytest

from nanoresearch.agent.tools.mcp import MCPToolWrapper


def _make_wrapper(original_name: str, session_key: str | None = None):
    """Helper: build a MCPToolWrapper bypassing __init__ heavy paths."""
    w = MCPToolWrapper.__new__(MCPToolWrapper)
    w._session = AsyncMock()
    fake_result = MagicMock()
    fake_result.content = []
    w._session.call_tool = AsyncMock(return_value=fake_result)
    w._original_name = original_name
    w._name = f"mcp_test_{original_name}"
    w._description = ""
    w._parameters = {}
    w._tool_timeout = 5
    w._session_key = session_key
    w._kb_map = {"kb1": "col_user_kb1"}
    return w


async def test_mcp_wrapper_injects_session_key_for_kb_search():
    w = _make_wrapper("kb_search", session_key="telegram:99")
    await w.execute(query="x", kb_id="kb1")
    args, kwargs = w._session.call_tool.call_args
    forwarded = kwargs["arguments"]
    assert forwarded.get("session_key") == "telegram:99"


async def test_mcp_wrapper_injects_session_key_for_kb_retrieve():
    w = _make_wrapper("kb_retrieve", session_key="telegram:99")
    await w.execute(query="x", kb_id="kb1")
    forwarded = w._session.call_tool.call_args.kwargs["arguments"]
    assert forwarded.get("session_key") == "telegram:99"


async def test_mcp_wrapper_no_inject_when_session_key_none():
    w = _make_wrapper("kb_search", session_key=None)
    await w.execute(query="x", kb_id="kb1")
    forwarded = w._session.call_tool.call_args.kwargs["arguments"]
    assert "session_key" not in forwarded


async def test_mcp_wrapper_no_inject_for_other_tools():
    """session_key 不能注入到 memory_search / web_search 等——子进程会拒绝。"""
    w = _make_wrapper("memory_search", session_key="telegram:99")
    await w.execute(query="x")
    forwarded = w._session.call_tool.call_args.kwargs["arguments"]
    assert "session_key" not in forwarded


async def test_mcp_wrapper_user_provided_session_key_wins():
    """若 kwargs 已含 session_key（理论上不会发生），使用 setdefault 不覆盖。"""
    w = _make_wrapper("kb_search", session_key="auto")
    await w.execute(query="x", kb_id="kb1", session_key="explicit")
    forwarded = w._session.call_tool.call_args.kwargs["arguments"]
    assert forwarded["session_key"] == "explicit"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -k "mcp_wrapper" -v
```

Expected: 4 个 inject 测试 FAIL（`'session_key' not in forwarded` 反例不通过），no_inject 测试可能 PASS（未注入是当前行为）。

- [ ] **Step 3: 实现 — 在 call_tool 前插入注入**

`backend/nanoresearch/agent/tools/mcp.py:210` 行的 `effective_timeout = self._tool_timeout` 之后、`try:` 之前，新增：

```python
        # Auto-inject session_key for RAG retrieval tools to enable subprocess
        # history-based query rewrite. Only kb_search / kb_retrieve accept
        # session_key — other tools (memory_search, web_search, ...) would reject it.
        if self._original_name in ("kb_search", "kb_retrieve") and self._session_key:
            kwargs.setdefault("session_key", self._session_key)

        effective_timeout = self._tool_timeout
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/agent/tools/mcp.py backend/tests/unit/rag/test_query_rewrite_a_class.py
git commit -m "feat(agent): MCPToolWrapper 对 kb_search/kb_retrieve 注入 session_key"
```

---

### Task 7: 子进程 PG-backed SessionManager + `_get_conversation_history` 异步

**Files:**
- Modify: `backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py:1-19,293-353`（含顶层新增工厂和改写 `_get_conversation_history`）
- Test: `backend/tests/unit/rag/test_query_rewrite_a_class.py`（追加）

**Interfaces:**
- Consumes: `nanoresearch.storage.database.get_session_factory()`、`nanoresearch.session.manager.SessionManager(workspace, session_factory)`、`nanoresearch.config.paths.get_workspace()`。
- Produces:
  - 模块级 `_get_subprocess_session_manager() -> SessionManager | None`（lazy 单例，PG-backed，DB 失败返回 None）。
  - `PlanQueryTool._get_conversation_history(session_key: str) -> list[dict]`（**async**，返回 `list[dict]`，不做 role 过滤）。

**说明**：spec §3.3 明确禁止 `SessionManager(get_workspace())` 无 factory 构造——那是 JSONL 孤儿，与父进程数据隔离。**必须**走 `get_session_factory()`。失败时返回 `[]`，rewrite 降级到原 query，不抛异常。

- [ ] **Step 1: 写失败测试**

追加到 `backend/tests/unit/rag/test_query_rewrite_a_class.py`：

```python
import inspect as _inspect
from nanoresearch.rag.mcp_server.tools.agentic import query_planning as qp


def test_subprocess_session_manager_factory_exists():
    assert hasattr(qp, "_get_subprocess_session_manager")


def test_get_conversation_history_is_async():
    sig = _inspect.signature(qp.PlanQueryTool._get_conversation_history)
    assert _inspect.iscoroutinefunction(qp.PlanQueryTool._get_conversation_history)


async def test_get_conversation_history_returns_list_of_dicts(monkeypatch):
    """SessionManager 返回 user/assistant/tool 三种 role，_get_conversation_history 原样返回（不过滤）。"""
    from nanoresearch.session.manager import Session

    fake_session = Session(key="telegram:1")
    fake_session.messages = [
        {"role": "user", "content": "介绍 NeRF 先驱"},
        {"role": "assistant", "content": "Ben Mildenhall, Jonathan Barron..."},
        {"role": "tool", "tool_call_id": "t1", "name": "kb_search", "content": "{}"},
        {"role": "user", "content": "你提到那两个做渲染的"},
    ]

    fake_mgr = AsyncMock()
    fake_mgr.get_or_create = AsyncMock(return_value=fake_session)
    monkeypatch.setattr(qp, "_get_subprocess_session_manager", lambda: fake_mgr)

    tool = qp.PlanQueryTool()
    history = await tool._get_conversation_history("telegram:1")

    roles = [m["role"] for m in history]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles, "tool 消息必须保留，B 类要用"
    assert isinstance(history, list)
    assert all(isinstance(m, dict) for m in history)


async def test_get_conversation_history_returns_empty_when_no_manager(monkeypatch):
    """SessionManager 拿不到 → 返回 []，不抛异常。"""
    monkeypatch.setattr(qp, "_get_subprocess_session_manager", lambda: None)
    tool = qp.PlanQueryTool()
    history = await tool._get_conversation_history("telegram:nope")
    assert history == []
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -k "subprocess_session_manager or get_conversation_history" -v
```

Expected: FAIL — 无工厂、`_get_conversation_history` 不是 async、返回类型不对。

- [ ] **Step 3: 实现 — 顶层新增工厂**

`backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py` 在 `logger = logging.getLogger(__name__)` 之后（约 L19 之后）新增：

```python
# Subprocess-side PG-backed SessionManager
# MCP server is an independent stdio subprocess (see §8 #1 of spec).
# Module-level globals in the main process are invisible here. We rely on
# DATABASE_URL/REDIS_URL transported via _stdio_env (§8 #6) to connect to the
# SAME PG/Redis the main process uses. JSONL fallback is forbidden — that
# would create an orphan store with no sync to the main session.

_subprocess_session_manager = None


def _get_subprocess_session_manager():
    """Lazy-init a PG-backed SessionManager in the MCP subprocess.

    Returns None on any init failure; callers must degrade (return empty
    history) rather than fall back to JSONL.
    """
    global _subprocess_session_manager
    if _subprocess_session_manager is not None:
        return _subprocess_session_manager
    try:
        from nanoresearch.storage.database import get_session_factory, init_engine
        from nanoresearch.session.manager import SessionManager
        from nanoresearch.config.paths import get_workspace

        try:
            factory = get_session_factory()
        except RuntimeError:
            # Engine not initialized yet in this subprocess — initialize it.
            init_engine()
            factory = get_session_factory()

        _subprocess_session_manager = SessionManager(
            workspace=get_workspace(),
            session_factory=factory,
        )
        logger.info("Subprocess PG-backed SessionManager initialized")
        return _subprocess_session_manager
    except Exception as e:
        logger.warning(f"Subprocess SessionManager init failed: {e}; query rewrite degraded")
        return None
```

- [ ] **Step 4: 实现 — 重写 `_get_conversation_history`**

`backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py:293-353`，**整段替换**：

```python
    async def _get_conversation_history(self, session_key: str) -> list[dict]:
        """Fetch conversation history from main agent's PG-backed session store.

        Returns list[dict] — raw messages with role/content/tool_calls/etc.
        No role filtering: tool messages are preserved for §3.4 A4 layering.
        Render layer (_render_history_for_prompt) skips tool when rendering
        to prompt; B class's _get_retrieval_titles reads them for chunk titles.

        Degrades to [] when subprocess SessionManager init fails — caller
        falls back to original query without rewrite.
        """
        manager = _get_subprocess_session_manager()
        if manager is None:
            logger.warning(
                f"No SessionManager available in subprocess for {session_key!r}; "
                "query rewrite degraded"
            )
            return []
        try:
            session = await manager.get_or_create(session_key)
            return session.get_history(max_messages=20)
        except Exception as e:
            logger.warning(f"Failed to fetch history for {session_key!r}: {e}")
            return []
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py backend/tests/unit/rag/test_query_rewrite_a_class.py
git commit -m "feat(rag): 子进程 PG-backed SessionManager + _get_conversation_history 异步化"
```

---

### Task 8: 渲染层 + `_get_retrieval_titles` + REWRITE_PROMPT 改造 + `_rewrite_query` 新签名

**Files:**
- Modify: `backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py:21-33,192-271,355-389`（REWRITE_PROMPT + execute 调用点 + _rewrite_query 签名 + 新增渲染辅助）
- Test: `backend/tests/unit/rag/test_query_rewrite_a_class.py`（追加）

**Interfaces:**
- Consumes: `_get_conversation_history(session_key) -> list[dict]`（Task 7）
- Produces:
  - `PlanQueryTool._render_history_for_prompt(history: list[dict]) -> list[str]` — 渲染 user+assistant 为 `[用户]/[助手] content` 字符串，跳过 tool，assistant 截断到 200 字，最近 6 条 user+assistant 消息为窗口。
  - `PlanQueryTool._get_retrieval_titles(history: list[dict]) -> list[str]` — 从尾向前找首条 `role=="tool" and name in {kb_search, rag_search, kb_retrieve}`，取 `_chunk_titles` 字段；找不到返回 `[]`。**A 类期间永远返回 `[]`（sidecar 由 B 类写）**——这是预期。
  - `PlanQueryTool._rewrite_query(query, history: list[dict], retrieval_titles: list[str]) -> str` — 新签名。
  - `PlanQueryTool.execute` 调用点改 `history = await self._get_conversation_history(...)` + `_rewrite_query(query, history, titles)`。
  - 新 `REWRITE_PROMPT`：`{history_section}{retrieval_section}当前问题：{query}` 结构。

**说明**：assistant content 可能是 list（多模态），渲染时拼接 `text` 块；多模态文本超 200 字截断。窗口"最近 6 条 user+assistant"指过滤掉 tool 后的最后 6 条。retrieval_section 在 A 类期间恒为空字符串，B 类 ship 后才有内容——prompt 模板和槽位结构 A 类一次到位，B 类不再动模板。

- [ ] **Step 1: 写失败测试**

追加到 `backend/tests/unit/rag/test_query_rewrite_a_class.py`：

```python
async def test_render_history_skips_tool():
    tool = qp.PlanQueryTool()
    history = [
        {"role": "user", "content": "介绍 NeRF 先驱"},
        {"role": "assistant", "content": "Ben Mildenhall（NeRF 一作）, Jonathan Barron（Mip-NeRF）"},
        {"role": "tool", "name": "kb_search", "content": '{"chunks":[]}'},
        {"role": "user", "content": "你提到那两个做渲染的"},
    ]
    rendered = tool._render_history_for_prompt(history)
    joined = "\n".join(rendered)
    assert "tool" not in joined.lower(), "tool 消息不能出现在 prompt 里"
    assert "[用户]" in joined and "[助手]" in joined
    assert "kb_search" not in joined


async def test_render_history_truncates_assistant_to_200():
    tool = qp.PlanQueryTool()
    long_text = "A" * 500
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": long_text},
    ]
    rendered = tool._render_history_for_prompt(history)
    assistant_line = [l for l in rendered if l.startswith("[助手]")][0]
    assert len(assistant_line) <= 220  # 200 字 + role 标签前缀余量


async def test_render_history_window_keeps_recent_6():
    tool = qp.PlanQueryTool()
    history = [
        {"role": "user", "content": f"u{i}"} for i in range(10)
    ]
    rendered = tool._render_history_for_prompt(history)
    assert len(rendered) == 6
    assert rendered[-1].endswith("u9")
    assert rendered[0].endswith("u4")


async def test_render_history_assistant_multimodal_list():
    tool = qp.PlanQueryTool()
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "first text"},
            {"type": "text", "text": "second text"},
        ]},
    ]
    rendered = tool._render_history_for_prompt(history)
    joined = "\n".join(rendered)
    assert "first text" in joined and "second text" in joined


async def test_get_retrieval_titles_empty_for_a_class():
    """A 类期间 sidecar 未写入，应返回 []。"""
    tool = qp.PlanQueryTool()
    history = [
        {"role": "user", "content": "q1"},
        {"role": "tool", "name": "kb_search", "content": '{"chunks":[]}'},
    ]
    assert tool._get_retrieval_titles(history) == []


async def test_get_retrieval_titles_reads_sidecar_when_present():
    """B 类 ship 后，sidecar 写入则可读出。"""
    tool = qp.PlanQueryTool()
    history = [
        {"role": "user", "content": "q1"},
        {
            "role": "tool",
            "name": "kb_search",
            "content": '{"chunks":[]}',
            "_chunk_titles": ["PGSR", "SuGaR"],
        },
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "上面那个"},
    ]
    titles = tool._get_retrieval_titles(history)
    assert titles == ["PGSR", "SuGaR"]


async def test_get_retrieval_titles_takes_most_recent_rag_tool():
    """多条 RAG tool 消息时，取最后一条。"""
    tool = qp.PlanQueryTool()
    history = [
        {"role": "tool", "name": "kb_search", "content": "{}", "_chunk_titles": ["OLD"]},
        {"role": "tool", "name": "kb_search", "content": "{}", "_chunk_titles": ["NEW"]},
    ]
    assert tool._get_retrieval_titles(history) == ["NEW"]


async def test_get_retrieval_titles_ignores_non_rag_tool():
    """web_search 等非 RAG tool 即使有 _chunk_titles 也忽略。"""
    tool = qp.PlanQueryTool()
    history = [
        {"role": "tool", "name": "web_search", "content": "{}", "_chunk_titles": ["bogus"]},
    ]
    assert tool._get_retrieval_titles(history) == []


async def test_rewrite_query_short_circuits_when_no_context():
    """history 和 titles 都为空时不调 LLM，原样返回。"""
    tool = qp.PlanQueryTool()
    tool._llm_client = object()  # truthy 但永远不应被调用——若调用会 AttributeError
    rewritten = await tool._rewrite_query("它是什么", history=[], retrieval_titles=[])
    assert rewritten == "它是什么"


async def test_rewrite_query_new_signature():
    sig = _inspect.signature(qp.PlanQueryTool._rewrite_query)
    params = list(sig.parameters.keys())
    assert "history" in params and "retrieval_titles" in params
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -k "render_history or retrieval_titles or rewrite_query" -v
```

Expected: 全部 FAIL — 新方法不存在或签名旧。

- [ ] **Step 3: 实现 — REWRITE_PROMPT 模板改造**

`backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py:21-33`：

```python
# Rewrite prompt for resolving pronouns and references
REWRITE_PROMPT = """{history_section}{retrieval_section}当前问题：{query}

将当前问题改写为独立完整的检索查询，解析所有指代和省略。

注意：
- 指代词（"它"、"那篇"、"上面那个"）优先用最近的上下文解析。
- 如果"上一轮检索到"列出了具体论文/文档，且当前问题用"那篇""上面"等指代，优先指向其中一篇。
- 如果话题已切换，忽略更早的话题锚点。
- 如果问题已经完整清晰，原样返回。

只输出改写后的查询，不要其他内容。"""
```

- [ ] **Step 4: 实现 — 新增渲染辅助 + 改写 _rewrite_query**

在 `_get_conversation_history` 之后（约 L390）新增并替换 `_rewrite_query`：

```python
    _RAG_TOOL_NAMES = ("kb_search", "rag_search", "kb_retrieve")

    @staticmethod
    def _extract_text_from_content(content) -> str:
        """Flatten str / multimodal list to text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(parts)
        return ""

    def _render_history_for_prompt(self, history: list[dict]) -> list[str]:
        """Render user + assistant turns as `[用户]/[助手] text` lines.

        - Skip tool messages (B class reads them via _get_retrieval_titles).
        - Truncate assistant content to 200 chars to bound prompt size.
        - Multimodal assistant content (list of blocks) is flattened to text.
        - Window: last 6 user+assistant messages (tool not counted).
        """
        rendered: list[str] = []
        for msg in history:
            role = msg.get("role")
            if role == "user":
                text = self._extract_text_from_content(msg.get("content", ""))
                rendered.append(f"[用户] {text}")
            elif role == "assistant":
                text = self._extract_text_from_content(msg.get("content", ""))
                if len(text) > 200:
                    text = text[:200] + "..."
                rendered.append(f"[助手] {text}")
            # tool 跳过：A4 数据保留 vs 渲染过滤分层
        return rendered[-6:]

    def _get_retrieval_titles(self, history: list[dict]) -> list[str]:
        """Read `_chunk_titles` sidecar from the most recent RAG tool message.

        Written by B class (§4.2 _save_turn sidecar). In A class period this
        field is never present — returns []. Caller renders retrieval_section
        as empty string in that case.
        """
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            if msg.get("name") not in self._RAG_TOOL_NAMES:
                continue
            titles = msg.get("_chunk_titles")
            if isinstance(titles, list) and titles:
                return [str(t) for t in titles]
            return []
        return []

    async def _rewrite_query(
        self,
        query: str,
        history: list[dict],
        retrieval_titles: list[str],
    ) -> str:
        """Rewrite query to resolve pronouns using history + last RAG titles."""
        rendered_history = self._render_history_for_prompt(history)

        if not rendered_history and not retrieval_titles:
            return query

        history_section = ""
        if rendered_history:
            history_section = "对话历史：\n" + "\n".join(
                f"{i+1}. {line}" for i, line in enumerate(rendered_history)
            ) + "\n\n"

        retrieval_section = ""
        if retrieval_titles:
            retrieval_section = "上一轮检索到：\n" + "\n".join(
                f"- {t}" for t in retrieval_titles
            ) + "\n\n"

        prompt = REWRITE_PROMPT.format(
            history_section=history_section,
            retrieval_section=retrieval_section,
            query=query,
        )

        import asyncio
        try:
            await asyncio.to_thread(self._ensure_initialized)
            if not self._llm_client:
                return query
            response = await asyncio.to_thread(self._call_llm, prompt)
            rewritten = response.strip()
            if not rewritten or rewritten in [".", "。"]:
                return query
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed, falling back to original: {e}")
            return query
```

- [ ] **Step 5: 实现 — `execute()` 调用点 await + 传 titles**

`backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py:216-220`：

```python
        # 1. Get conversation history for query rewriting (async)
        history = await self._get_conversation_history(session_key) if session_key else []

        # 2. Extract retrieval titles from last RAG tool message (B class data)
        retrieval_titles = self._get_retrieval_titles(history)

        # 3. Rewrite query to resolve pronouns
        rewritten_query = await self._rewrite_query(query, history, retrieval_titles)
```

同时检查同函数下面的 `len(history) > 0` 判定（约 L236, L259, L269）保持不变——`history` 现在是 `list[dict]`，`len()` 仍有效。

- [ ] **Step 6: 运行测试确认通过**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py -v
```

Expected: 全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add backend/nanoresearch/rag/mcp_server/tools/agentic/query_planning.py backend/tests/unit/rag/test_query_rewrite_a_class.py
git commit -m "feat(rag): A4 渲染层 + REWRITE_PROMPT 改造 + _rewrite_query 新签名"
```

---

### Task 9: 端到端 smoke test — 合成多轮 + 指代消解

**Files:**
- Create: `backend/tests/integration/__init__.py`（空）
- Create: `backend/tests/integration/test_query_rewrite_e2e.py`

**Interfaces:**
- Consumes: Task 1-8 全部已 ship
- Produces: 1 个 e2e 测试覆盖 §7.1 验证标准（合成 idx=37 式指代消解）

**说明**：本测试 mock `_get_subprocess_session_manager`（返回带合成 user+assistant 多轮的 Session）和 LLM client（断言 prompt 含 assistant 内容），**不**走真 LLM、不走真 DB。

- [ ] **Step 1: 写测试**

```python
"""A 类端到端 smoke test — §7.1 验证标准（合成多轮 + 指代消解）。

不依赖真实 LLM 和真实 DB。Mock 层：
- _get_subprocess_session_manager → 返回带合成 user/assistant 的 Session
- PlanQueryTool._call_llm → 返回固定字符串，断言 prompt 内容符合预期
"""

from unittest.mock import AsyncMock, MagicMock, patch

from nanoresearch.rag.mcp_server.tools.agentic import query_planning as qp
from nanoresearch.session.manager import Session


async def test_a_class_end_to_end_rewrite_resolves_pronoun(monkeypatch):
    """Synthetic idx=37：'你说 NeRF 很多，你具体指的是什么' 应被改写。"""
    # 1. mock subprocess SessionManager 返回合成历史
    fake_session = Session(key="test:1")
    fake_session.messages = [
        {"role": "user", "content": "介绍几个 3D 视觉先驱"},
        {"role": "assistant", "content": "Ben Mildenhall（NeRF 一作）, Jonathan Barron（Mip-NeRF）等"},
        {"role": "user", "content": "你说 NeRF 很多，你具体指的是什么"},
    ]
    fake_mgr = AsyncMock()
    fake_mgr.get_or_create = AsyncMock(return_value=fake_session)
    monkeypatch.setattr(qp, "_get_subprocess_session_manager", lambda: fake_mgr)

    # 2. mock LLM client：断言 prompt 含 assistant 内容 + 当前 query
    captured_prompts: list[str] = []

    def fake_call_llm(self, prompt):
        captured_prompts.append(prompt)
        # 模拟 LLM 改写：把"NeRF"改成"Ben Mildenhall / Jonathan Barron 的 NeRF 工作"
        return "Ben Mildenhall 和 Jonathan Barron 的 NeRF 相关工作具体是什么"

    monkeypatch.setattr(qp.PlanQueryTool, "_call_llm", fake_call_llm)

    # 3. mock _ensure_initialized 让 _llm_client 真的不为 None
    def fake_init(self):
        self._llm_client = MagicMock()
        self._llm_model = "test-model"
        self._initialized = True

    monkeypatch.setattr(qp.PlanQueryTool, "_ensure_initialized", fake_init)

    # 4. 跑 execute
    tool = qp.PlanQueryTool()
    response = await tool.execute(
        query="你说 NeRF 很多，你具体指的是什么",
        session_key="test:1",
    )

    # 5. 断言：rewritten_query 跟 original_query 不一样且语义更具体
    import json
    payload = json.loads(response.content)
    assert payload["original_query"] == "你说 NeRF 很多，你具体指的是什么"
    assert payload["rewritten_query"] != payload["original_query"]
    assert "Ben Mildenhall" in payload["rewritten_query"] or "Barron" in payload["rewritten_query"]

    # 6. 断言：prompt 实际含 assistant 内容（证明渲染对了）
    assert any("Ben Mildenhall" in p for p in captured_prompts), \
        "REWRITE_PROMPT 必须含 assistant 渲染内容"
    assert any("[助手]" in p for p in captured_prompts), \
        "渲染层必须给 assistant 加 [助手] 标签"


async def test_a_class_no_session_key_skips_rewrite(monkeypatch):
    """没有 session_key 时不调 SessionManager 也不 rewrite。"""

    fake_mgr = AsyncMock()
    monkeypatch.setattr(qp, "_get_subprocess_session_manager", lambda: fake_mgr)

    def fake_init(self):
        self._llm_client = MagicMock()
        self._llm_model = "test-model"
        self._initialized = True

    monkeypatch.setattr(qp.PlanQueryTool, "_ensure_initialized", fake_init)

    tool = qp.PlanQueryTool()
    response = await tool.execute(query="它是什么")

    import json
    payload = json.loads(response.content)
    assert payload["rewritten_query"] == payload["original_query"] == "它是什么"
    fake_mgr.get_or_create.assert_not_called()


async def test_a_class_subprocess_sm_failure_degrades(monkeypatch):
    """SessionManager 拿不到 → 不抛异常，rewritten = original。"""
    monkeypatch.setattr(qp, "_get_subprocess_session_manager", lambda: None)

    def fake_init(self):
        self._llm_client = MagicMock()
        self._llm_model = "test-model"
        self._initialized = True

    monkeypatch.setattr(qp.PlanQueryTool, "_ensure_initialized", fake_init)

    tool = qp.PlanQueryTool()
    response = await tool.execute(query="它是什么", session_key="test:dead")

    import json
    payload = json.loads(response.content)
    assert payload["rewritten_query"] == "它是什么"
```

- [ ] **Step 2: 运行测试**

```bash
cd backend && pytest tests/integration/test_query_rewrite_e2e.py -v
```

Expected: 3 PASS（如果之前 7 个 task 都通过，e2e 应直接通过；任何 FAIL 都指向某一前置 task 的回归）。

- [ ] **Step 3: 全量回归**

```bash
cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py tests/integration/test_query_rewrite_e2e.py -v
```

Expected: 全部 PASS。

- [ ] **Step 4: Commit**

```bash
git add backend/tests/integration/__init__.py backend/tests/integration/test_query_rewrite_e2e.py
git commit -m "test(rag): A 类 e2e smoke — 合成多轮 + 指代消解（§7.1 验证标准）"
```

---

## Self-Review Checklist（开发者实施完毕后跑一遍）

1. **Spec §3.1 7 处透传** — 对照下表：

| Spec # | Spec 引用 | 本 plan 任务 | 修改文件 |
|---|---|---|---|
| 1 | MCPToolWrapper.execute | T6 | `agent/tools/mcp.py` |
| 2 | RAGSearchTool.input_schema | T2 Step 3 | `rag_search.py` |
| 3 | RAGSearchTool.execute | T2 Step 4 | `rag_search.py` |
| 4 | rag_search_handler | T2 Step 6 | `rag_search.py` |
| 5 | _execute_complex → run_rag_loop | T2 Step 5 + T3 | `rag_search.py` + `runner.py` |
| 6 | RAGLoopRunner.run / SessionState | T1 + T3 | `state.py` + `runner.py` |
| 7 | _run_plan_phase → InternalTools.plan_query + schema | T4 + T5 | `tools.py` + `runner.py` |

2. **Spec §3.2 A1 await** — T7 step 4 改 `_get_conversation_history` 为 async + await。

3. **Spec §3.3 A3 子进程 PG-backed SessionManager** — T7 Step 3 工厂 + Step 4 用。

4. **Spec §3.4 A4 数据保留 + 渲染过滤** — T7 返回 list[dict] 不过滤 + T8 `_render_history_for_prompt` 跳 tool。

5. **Spec §7.1 验证标准** — T9 e2e smoke 覆盖。

6. **Spec §8 #14 rename 残留** — T0 清理 2/3，第 3 处（`agent/context.py:279`）由独立 PR `fix/kb-search-prompt-typo` 处理。

7. **未覆盖（确认是 B 类范围，不在本 plan）**：
   - §4.2 `_save_turn` 写 `_chunk_titles` sidecar
   - §4.2 `Session.get_history` 白名单加 `_chunk_titles`
   - §5.1 `plan_query_handler` 模块级单例（性能，独立 commit；可在 A 类合并前后任意时点单独处理）

---

## 实施完成后的合并步骤

1. 跑全量回归（unit + integration）：
   ```bash
   cd backend && pytest tests/unit/rag/test_query_rewrite_a_class.py tests/integration/test_query_rewrite_e2e.py -v
   ```
2. 自检 `git log --oneline main..feature/rag-query-rewrite-a-class` — 应有 9 个 commit（T0-T8 + T9）。
3. Push：`git push -u origin feature/rag-query-rewrite-a-class`
4. PR 标题：`feat(rag): A 类查询改写透传链 + 子进程 PG-backed SessionManager`
5. PR body：链接 spec `docs/superpowers/specs/2026-06-26-rag-query-rewrite-fix.md` §3 + §7.1 验证标准。
