# RAG 查询改写 / 指代消解修复 Design

**日期**: 2026-06-26
**状态**: Spec 定稿（含四项修订），实施待批准
**范围标签**: 仅查询改写 / 指代消解 — A 类（透传链）+ B 类（上一轮检索摘要）。**缓存（query cache）不在本 spec 范围**，见 §6。

---

## 1. 背景

### 1.1 病灶

主 RAG loop 路径上，**指代消解从未真正运行过**。表面上 `PlanQueryTool._rewrite_query()` 存在、`REWRITE_PROMPT` 写好，但实际运行时三层断点叠加 → `history` 永远空 → rewrite 第一行 `if not history: return query` 短路。结果：用户说"那篇论文""它"时，原始字符串直接喂检索，质量崩。

### 1.2 三层断点（已查证）

1. **透传链全程断开**：外层 Agent 持有 `session_key = channel:chat_id`，已通过 `MCPToolWrapper.set_session_key()` 注入到 wrapper 实例，**但 `MCPToolWrapper.execute()` 从来没把它写进 kwargs 转发给 MCP 子进程**（`backend/nanobot/agent/tools/mcp.py:128-246`）。下游全链路 `session_key=None` → `history=[]` → 短路。
2. **`_get_conversation_history` 没 await**：`backend/nanobot/rag/mcp_server/tools/agentic/query_planning.py:310-311` 同步函数里调异步的 `manager.get_or_create(session_key)`，拿到的是 coroutine。
3. **过滤逻辑只保留 user**：同文件 L317-333，`if msg.get("role") != "user": continue` 把 assistant 全过滤。指代常指向 assistant 上一轮的具体名词。

### 1.3 指代两类（决定 A/B 拆分）

- **A 类**：指代指向用户/assistant 前面**说过的话**。修上述三个断点足够。
- **B 类**：指代指向上一轮 **RAG 检索/推荐出来的论文**（"那篇论文""上面那个 paper"）。需要把上一轮检索结果（至少 title 列表）塞进 rewrite prompt。

---

## 2. 范围

### 2.1 In-Scope

- A 类：7 处透传链 + A1（await）+ A3（子进程自建 PG-backed SessionManager）+ A4（history 含 assistant，且**保留 tool 消息供 B 类用**）。
- B 类：title-only 抽取（写入时落 sidecar）+ REWRITE_PROMPT 槽位改造 + 容错降级。
- 独立修复（与主线解耦，单独 commit）：
  - §5.1 `plan_query_handler` 模块级单例（性能）
  - §5.2 `protocol_handler.execute_tool` MCPToolResponse 默认 repr → 裸 JSON（正确性，B 类前置）

### 2.2 Out-of-Scope

- **Query cache**：整条线搁置。缓存 key 依赖消解后的 query，必须等本 spec 落地、消解跑稳后再做。本 spec 落地之前**不要写任何缓存相关代码或 spec**。
- **更细粒度的 chunk 摘要**（80/200 字摘要）：等 title-only 上线、确认确有需求再加。
- **复杂指代**（指向多轮前/跨会话）：本 spec 只覆盖"最近一轮"。
- **改 PlanQueryTool 已经接好的那一层**（input_schema 已加 session_key、execute 已接入参、handler 已透传）：保持现状。

---

## 3. A 类 — 透传链 + 三个修复点

### 3.1 透传链 7 处

| # | 文件 | 当前状态 | 改动 |
|---|---|---|---|
| 1 | `backend/nanobot/agent/tools/mcp.py:128-246` `MCPToolWrapper.execute()` | setter 存在，execute 不用 | **仅对 `kb_search` / `kb_retrieve` / `rag_search` 三个 original_name**，在 `self._session.call_tool(...)` 之前 `if self._session_key: kwargs.setdefault("session_key", self._session_key)`。不要对所有 `mcp_*` 一刀切——其他工具（memory/list_collections/ingest 等）没有 session_key 参数，会被 MCP 子进程拒绝。 |
| 2 | `backend/nanobot/rag/mcp_server/tools/rag_search.py:111-134` `RAGSearchTool.input_schema` | 4 字段，缺 session_key | `properties` 加 `"session_key": {"type": "string", "description": "Main agent session key (channel:chat_id) for multi-turn context"}`。不加进 `required`。 |
| 3 | `backend/nanobot/rag/mcp_server/tools/rag_search.py:136-174` `RAGSearchTool.execute()` 签名 | 缺 session_key | 加 `session_key: Optional[str] = None`，透给 `_execute_complex`（简单路径不需要）。 |
| 4 | `backend/nanobot/rag/mcp_server/tools/rag_search.py:299-317` `rag_search_handler` | 缺 session_key | 加 `session_key: Optional[str] = None`，透给 `tool.execute(...)`。 |
| 5 | `backend/nanobot/rag/mcp_server/tools/rag_search.py:255-295` `_execute_complex` → `run_rag_loop` | 缺 session_key | `_execute_complex` 加参数 + `run_rag_loop` / `RAGLoopRunner.run` 加参数。 |
| 6 | `backend/nanobot/rag/internal_loop/runner.py:190-207` `RAGLoopRunner.run()` + `backend/nanobot/rag/internal_loop/state.py:72-119` `SessionState` | run 签名缺 session_key，SessionState 缺 `caller_session_key` 字段 | `SessionState` 加 `caller_session_key: Optional[str] = None`（在 `context` 后）；`session_manager.create_session(...)` 同步加该参数；`run()` 签名加 `session_key: Optional[str] = None` 并写入 session。**SessionState 是内部 loop 概念，跟外层 SessionManager 不是同一个东西，字段名用 `caller_session_key` 防混淆。** |
| 7 | `backend/nanobot/rag/internal_loop/runner.py:389-425` `_run_plan_phase` → `self.tools.plan_query(...)` + `backend/nanobot/rag/internal_loop/tools.py:215-259` `InternalTools.plan_query` → `self._plan_tool.execute(...)` | 两层都缺 session_key | `_run_plan_phase` 从 `session.caller_session_key` 取出 → 传给 `plan_query`；`InternalTools.plan_query` 加 `session_key: Optional[str] = None` 透给 `self._plan_tool.execute(...)`。**同时给 `get_plan_tool_schema()`（L100-122）的 parameters 加 session_key，否则 LLM 视图里没有这个字段。** |

下游已接好、**不要碰**：
- `PlanQueryTool.input_schema` 已含 session_key（`query_planning.py:172-190`）
- `PlanQueryTool.execute()` 已接 session_key 入参（`query_planning.py:192-271`）
- `plan_query_handler` 已透传 session_key（`query_planning.py:891-902`）

### 3.2 A1 — `_get_conversation_history` 改 async + await

文件：`backend/nanobot/rag/mcp_server/tools/agentic/query_planning.py:293-353`。

当前签名 `def _get_conversation_history(self, session_key: str) -> List[str]:`，L311 `session = manager.get_or_create(session_key)` 漏 await，manager 是异步 API。

改：
- 签名 → `async def _get_conversation_history(self, session_key: str) -> list[dict]:`（**注意返回类型从 `List[str]` 改 `list[dict]`**，见 §3.4 + §4.2）
- L311 → `session = await manager.get_or_create(session_key)`
- 调用点 `execute()` → `history = await self._get_conversation_history(session_key) if session_key else []`

### 3.3 A3 — 子进程自建 PG-backed SessionManager（修订 1，已重设计）

**为什么不能"主进程注入全局单例"**：MCP server 是独立 stdio 子进程，主进程的模块级全局变量子进程看不到（两份内存）。证据见 §8 台账 #1。

**前提已确认**：父进程通过 `dotenv` 把 `DATABASE_URL`/`REDIS_URL` 注入 `os.environ`；`MCPToolWrapper` 拉起子进程时 `_stdio_env = {**os.environ, **cfg.env}` 全量透传——子进程拿得到同一份连接串。证据见 §8 台账 #6。

**改动**：`query_planning.py` 模块顶层构造一个**模块级 PG-backed SessionManager**，禁止 JSONL 孤儿。

```python
# query_planning.py 顶层
from nanobot.config.paths import get_workspace
from nanobot.session.manager import SessionManager
from nanobot.storage.database import get_session_factory

_session_manager: SessionManager | None = None

def _get_subprocess_session_manager() -> SessionManager | None:
    """Lazy-init a SessionManager that connects to the SAME PG/Redis the main
    process uses. Subprocess can do this because _stdio_env transports
    DATABASE_URL/REDIS_URL from parent. PG is the single source of truth."""
    global _session_manager
    if _session_manager is None:
        try:
            factory = get_session_factory()  # reads DATABASE_URL from os.environ
            _session_manager = SessionManager(
                workspace=get_workspace(),
                session_factory=factory,
            )
        except Exception as e:
            logger.warning("Subprocess SessionManager init failed: {}", e)
            return None
    return _session_manager
```

`_get_conversation_history` 改：

```python
async def _get_conversation_history(self, session_key: str) -> list[dict]:
    manager = _get_subprocess_session_manager()
    if manager is None:
        logger.warning("No SessionManager available in subprocess; query rewrite degraded")
        return []
    session = await manager.get_or_create(session_key)
    return session.get_history(max_messages=20)  # 返回 list[dict]，不抽 user-only
```

**禁止**：
- 不要 `SessionManager(get_workspace())` 无 session_factory 构造——那是 JSONL 孤儿。
- 不要"拿不到就回退 JSONL"——拿不到就明确返回 `[]` 并 warn，rewrite 走原 query，降级但不出错。

**前提失效信号**：如果将来 `get_session_factory()` 改为不再读 `os.environ`（如改读 settings 文件路径而子进程不挂载该路径），本方案立刻失效。台账 §8 #6 必须随之更新。

### 3.4 A4 — history 保留 user + assistant + **tool 消息**（修订 2，重要补充）

**关键**：A 类 rewrite 自身**不读** tool 消息内容，但 `_get_conversation_history` 返回的 `list[dict]` **必须包含 tool 消息**——B 类的 `_get_retrieval_titles(history)` 要从这里面找最近的 `role==tool && name in {kb_search, rag_search, kb_retrieve}` 取其 `_chunk_titles`。

> ⚠️ **A4 是数据保留 + 渲染过滤的分层，不是数据过滤。** 渲染给 LLM 的字符串里不展示 tool，但 history 数据结构里 tool 消息原封不动保留。任何"为了 prompt 干净直接 filter 掉 tool"的写法会**让 B 类失能**，禁止。

实现要点：
- `_get_conversation_history` 返回 `list[dict]`（不做 role 过滤，原样返回 `session.get_history()`）。
- 新增 `_render_history_for_prompt(history: list[dict]) -> list[str]`：只把 user + assistant 渲染成字符串列表，**跳过 tool**。assistant content 截断到前 200 字（防 prompt 爆炸；多模态 list 拼接 text 块）。渲染顺序按时间序穿插，标注 role：
  ```
  1. [用户] 介绍几个 3D 视觉先驱
  2. [助手] Ben Mildenhall（NeRF 一作）...Jonathan Barron（Mip-NeRF）...
  3. [用户] 你提到那两个做渲染的，是谁？
  ```
- 新增 `_get_retrieval_titles(history: list[dict]) -> list[str]`：从尾部往前找第一条 `role==tool && name in {kb_search, rag_search, kb_retrieve}`，取其 `_chunk_titles` 字段（由 §4.2 写入路径预存的 sidecar）；找不到返回 `[]`。
- window：最近 6 轮消息（3 轮 user+assistant 来回），tool 消息不计入这个窗口（tool 只看"最近一条 RAG tool"）。

---

## 4. B 类 — 上一轮检索摘要喂进 rewrite

### 4.1 三个硬约束

1. **title 抽取路径是 `chunk["metadata"]["title"]`**，不是顶层 `doc_title`。
   - 兜底链：`metadata.title` → `Path(metadata.source_path).stem` → `chunk_id[:12]`。三层都拿不到就跳过该 chunk。
2. **优先方案**（写入时存 sidecar）+ 容错降级，不做读取时解析大 JSON（见 §4.2）。
3. **抽取限定"最近一条 `role==tool && name in {kb_search, rag_search, kb_retrieve}`"**。别的 tool（web_search/message/shell）不含 chunks，误抽会污染 prompt。

### 4.2 实现：写入时落 `_chunk_titles` sidecar

**前置依赖**：§5.2（MCPToolResponse repr → 裸 JSON 修复）必须先 merge。否则 `_extract_chunk_titles` 输入是 `MCPToolResponse(content='...', ...)` Python repr 而非裸 JSON，需要正则剥壳——我们选择修真 bug，不打补丁。

在 `AgentLoop._save_turn`（`backend/nanobot/agent/loop.py:873-897`）处理 `role == "tool"` 分支，对 `name in {kb_search, rag_search, kb_retrieve}` 的消息**额外**写一个伴生字段：

```python
if role == "tool" and entry.get("name") in {"kb_search", "rag_search", "kb_retrieve"}:
    titles = _extract_chunk_titles(entry.get("content", ""))  # 解析失败返回 []
    if titles:
        entry["_chunk_titles"] = titles[:10]  # 最多 10 条
```

`_extract_chunk_titles` 签名与契约：

```python
def _extract_chunk_titles(content: str) -> list[str]:
    """Parse a bare-JSON tool result string and extract chunk titles via
    metadata.title → Path(source_path).stem → chunk_id[:12] fallback chain.
    Returns [] on any parse failure (never raises).

    Precondition: §5.2 fix is merged so `content` is bare JSON, not
    MCPToolResponse(content='...') Python repr."""
```

- **大小**：10 条 title × 平均 60 字 ≈ 600 字节，远低于 16K 阈值，**不受截断影响**。
- **抽取时机**：写入时（JSON 还完整），不是读取时（被截断后救不回）。
- **存储位置**：role:"tool" 消息的扩展字段，跟 `tool_call_id` / `name` 同级。
- **副作用**：`Session.get_history()`（`backend/nanobot/session/manager.py:82-86`）的保留字段白名单必须加 `_chunk_titles`，否则被丢弃。

**容错降级**：`_chunk_titles` 不存在时（旧消息、非 RAG tool、解析失败时没写入），`_get_retrieval_titles` 安全返回 `[]`，rewrite 降级为只用 history（A 类），绝不抛异常。

### 4.3 完整签名（修订 2）

```python
class PlanQueryTool:
    async def execute(
        self,
        query: str,
        context: str = "",
        session_key: str | None = None,
        # ... 其他既有参数
    ) -> MCPToolResponse:
        history: list[dict] = (
            await self._get_conversation_history(session_key)
            if session_key else []
        )
        retrieval_titles: list[str] = self._get_retrieval_titles(history)
        rewritten: str = await self._rewrite_query(query, history, retrieval_titles)
        # ... 余下逻辑用 rewritten 作为后续检索 query

    async def _rewrite_query(
        self,
        query: str,
        history: list[dict],
        retrieval_titles: list[str],
    ) -> str:
        rendered_history: list[str] = self._render_history_for_prompt(history)
        # 见 §4.4 的渲染逻辑
        ...
```

### 4.4 REWRITE_PROMPT 改造

文件：`backend/nanobot/rag/mcp_server/tools/agentic/query_planning.py:22-33`。

```python
REWRITE_PROMPT = """{history_section}{retrieval_section}当前问题：{query}

将当前问题改写为独立完整的检索查询，解析所有指代和省略。

注意：
- 指代词（"它"、"那篇"、"上面那个"）优先用最近的上下文解析。
- 如果"上一轮检索到"列出了具体论文/文档，且当前问题用"那篇""上面"等指代，优先指向其中一篇。
- 如果话题已切换，忽略更早的话题锚点。
- 如果问题已经完整清晰，原样返回。

只输出改写后的查询，不要其他内容。"""
```

`_rewrite_query` 内渲染逻辑：

```python
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

if not history_section and not retrieval_section:
    return query  # 没任何上下文，不调 LLM 直接返回
```

---

## 5. 独立修复（与主线解耦，单独 commit）

### 5.1 顺手 bug — `plan_query_handler` 模块级单例（性能）

文件：`backend/nanobot/rag/mcp_server/tools/agentic/query_planning.py:891-902`。

```python
async def plan_query_handler(query, context=None, session_key=None):
    tool = PlanQueryTool()  # ← 每次都 new，LLM client 反复初始化
    return await tool.execute(...)
```

改：

```python
_plan_query_tool: Optional[PlanQueryTool] = None

def _get_plan_query_tool() -> PlanQueryTool:
    global _plan_query_tool
    if _plan_query_tool is None:
        _plan_query_tool = PlanQueryTool()
    return _plan_query_tool

async def plan_query_handler(query, context=None, session_key=None):
    tool = _get_plan_query_tool()
    return await tool.execute(query=query, context=context, session_key=session_key)
```

`PlanQueryTool._ensure_initialized` 已有幂等保护（`if self._initialized: return`），单例化无正确性风险，纯性能改善。

**独立 commit**，message 写清"performance only, no behavior change"。

### 5.2 前置独立修复 — `protocol_handler.execute_tool` MCPToolResponse 默认 repr（正确性 / B 类前置）

文件：`backend/nanobot/rag/mcp_server/protocol_handler.py:108-179`。

**问题**：`execute_tool` 只 isinstance 检查 `CallToolResult` / `str` / `list`，其余落进 L150-154 默认分支 `TextContent(text=str(result))`。`MCPToolResponse` 是 `@dataclass` 且无 `__str__` 覆盖（`response_builder.py:34-49`），`str()` 触发 dataclass repr，外层 Agent 拿到的是：

```
MCPToolResponse(content='{"success": true, "chunks": [...]}', citations=[], metadata={...}, is_empty=False, image_contents=[])
```

——**不是裸 JSON，是 Python repr 包裹的伪 JSON**。这一直在毒害所有走 `build_json_response` 的工具回答质量（不只 B 类），模型收到的是带类名前缀的 Python repr。

**改动**：`execute_tool` 在 `if isinstance(result, list):` 之前插入：

```python
from nanobot.rag.core.response.response_builder import MCPToolResponse
if isinstance(result, MCPToolResponse):
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=result.content)],
        isError=False,
    )
```

（或者从 `to_mcp_content()` 取所有 blocks，但那会把 References JSON 附加到 text 里，更激进。**保守做法是只返 `result.content` 裸 JSON**，跟现有 `str(result)` 的最小语义差就是去掉 dataclass 包装。）

**影响面**：所有走 `build_json_response` 的 MCP 工具输出格式都从"伪 JSON repr"变为"裸 JSON"。影响范围 > B 类。

**因此**：
- **独立 PR**，独立 commit，跟 A/B 都解耦。
- **独立验证**：跑一组现有的非指代检索（单轮 `kb_search`），确认主路径回答**没退化**（实际预期：质量改善，因为模型不再被 `MCPToolResponse(content='...', ...)` 这种 repr 噪声干扰）。
- **必须先 merge**，然后 B 类才能依赖"role:tool content 是裸 JSON"这个前提（§4.2 `_extract_chunk_titles` 直接 `json.loads`）。
- 与 §5.1 同级，**不埋进 B 类**。

---

## 6. 不在本 spec 范围（显式声明）

- **Query cache**：不写代码、不写 spec。理由：缓存 key 是消解后的 query，key 稳定性依赖本 spec 落地。本 spec 跑稳前任何缓存实现都会因 rewrite 行为变化导致 key 漂移、命中率失真。**前置依赖**：本 spec A 类全部 merge + B 类全部 merge + 真实多轮会话验证 rewrite 行为符合预期后，另起 spec。

---

## 7. 实施顺序

### 7.0 前置（必须最先 merge）

- **§5.2** `protocol_handler.execute_tool` MCPToolResponse 修复 + 独立验证主路径无退化。
- 这一项过了，B 类才能开做。

### 7.1 A 类（修订 3 前段：合成数据 OK）

- **前置阻断**：`rag_search` → `kb_search` rename 半成品，`agent/context.py:279` system prompt 仍指导模型用旧名 → 模型被指导一个不存在的工具名，A 类透传链接通也可能触发不到。**A 类动工时必须一并修全 3 处残留**（`tools/__init__.py:52`、`agent/context.py:279`、`internal_loop/state.py:76`）+ `protocol_handler.py:196-227` docstring。见 §8 #14。
- 7 处透传 + A1 + A3 + A4 是同一个完整修复，**作为一个 PR**（或紧密耦合的多个 commit，但同一 review 单元）merge。
- **验证标准**：跑一条带指代且指代指向 user/assistant 前文的多轮会话即可（A 类不依赖检索结果真实性）。例如 `_fix_session_history.py` 中的合成对话 idx=37 "你说 NeRF 很多，你具体指的是什么"——合成数据足够覆盖 A 类。日志里看到 `rewritten_query` 不等于 `original_query` 且语义合理即通过。
- A 类 merge 后，B 类未完成期间"那篇论文"这类指代仍失效——**预期**，不阻塞 A 类发布。

### 7.2 B 类（修订 3 前段：**必须真实两轮数据**）

- **前置阻断 (a)**：当前 KB 的 BM25 sparse 索引与向量库 chunk_id 不一致，真实 `kb_search` 返回 `chunks=[]`，B 类即使写对也无 title 可抽。**B 类动工前必须先修索引**。见 §8 #12。
- **前置阻断 (b)**：`_extract_chunk_titles` 不能假设输入恒为合法 JSON——`ExecuteRetrievalBatchTool / FuseAndFetchRoundTool / GetRoundStatusTool` 返回的是"JSON+Markdown"混排。B 类动工前先确认这三个工具的输出是否会落入顶层 role:"tool"；若会则需先分离 JSON 段或正则抽 title，并在 `json.loads` 失败时容错降级（§4.2 退路覆盖需显式包含此类，不只是 16K 截断）。见 §8 #13。
- 依赖 §5.2 已 merge + 7.1 A 类已 merge。
- 三块（`_save_turn` 写 sidecar / `Session.get_history` 白名单 / REWRITE_PROMPT retrieval_section）作为同一个 PR。
- **验证标准（硬要求）**：**必须用真实两轮 kb_search 对话验证，不接受合成数据**。
  - 第一轮：真实 KB 触发 `kb_search`（不是手工构造的 role:"tool" JSON），让 chunks 真实写入；
  - 第二轮：用"那篇论文"/"上面那个"指代；
  - 检查点：(a) 第一轮 session.messages 里 role:"tool" 消息**真实**带上 `_chunk_titles`；(b) 第二轮 `rewritten_query` 包含具体论文 title 之一。
- **`_fix_session_history.py` 是合成数据，不能作为 B 类 ground truth**。原因：合成数据无法验证 §4.2 的写入路径（`_save_turn` 写 sidecar）和 §5.2 的真实 JSON 形态在生产链路下是否真的连通。

### 7.3 独立 commit 时序

- §5.1（plan_query_handler 单例）：可塞进 A 类 PR 也可单独，commit 必须独立。
- §5.2（MCPToolResponse 修复）：**独立 PR，先于 B 类 merge**。

---

## 8. 已查证事实台账（重置记忆后读这一节可重建全部上下文）

每条都带 `文件:行号` 证据。任何条目失效（重构/重命名）需同步更新本节。

| # | 事实 | 证据 |
|---|---|---|
| 1 | **MCP server 是独立 stdio 子进程**——主进程模块级全局子进程拿不到，跨进程共享数据只能靠共享后端（PG/Redis） | `backend/nanobot/rag/mcp_server/__main__.py:3`（`Run with: python -m nanobot.rag.mcp_server`，独立入口）；`backend/nanobot/rag/mcp_server/server.py:28,82,89`（`MCP stdio transport reserves stdout for JSON-RPC` + `run_stdio_server_async` + `import mcp.server.stdio`）；`backend/nanobot/agent/tools/mcp.py:273-281`（Agent 端 `StdioServerParameters(command=cfg.command, args=cfg.args, env=_stdio_env)` 拉起 + `stdio_client(params)`） |
| 2 | **role:"tool" 内容是 `MCPToolResponse(content='...')` Python repr 包裹的伪 JSON**，非裸 JSON（§5.2 修复后变裸 JSON） | `backend/nanobot/rag/mcp_server/protocol_handler.py:108-179`（`execute_tool` 只 isinstance `CallToolResult/str/list`，默认分支 L150-154 `TextContent(text=str(result))`）；`backend/nanobot/rag/core/response/response_builder.py:34-49`（`MCPToolResponse` 是 `@dataclass`，无 `__str__` 覆盖，`str()` 触发 dataclass repr）；`backend/nanobot/agent/tools/mcp.py:237-244`（Agent 端 `raw_result = "\n".join(parts)`，parts 来自 `TextContent.text`，原样回传） |
| 3 | **16K 截断位置**——长检索结果尾部被切，事后 `json.loads` 必失败 → B 类只能走"写入时落 sidecar"路线 | `backend/nanobot/agent/loop.py:61`（`_TOOL_RESULT_MAX_CHARS = 16_000`）；截断点在 `_save_turn`（`backend/nanobot/agent/loop.py:881-883` 附近） |
| 4 | **`session_key` 在 `MCPToolWrapper.execute()` 中不被转发**——setter 存在但 execute 从未写入 kwargs | `backend/nanobot/agent/tools/mcp.py:101`（`self._session_key: str \| None = None`）；`backend/nanobot/agent/tools/mcp.py:124-126`（`set_session_key` 存在）；`backend/nanobot/agent/tools/mcp.py:128-246`（`execute()` 全文，L213-216 `self._session.call_tool(self._original_name, arguments=kwargs)` 前无 session_key 注入） |
| 5 | **chunk title 实际路径是 `chunk["metadata"]["title"]`**，无顶层 `doc_title`；source 路径在 `chunk["metadata"]["source_path"]` | `backend/nanobot/rag/mcp_server/tools/agentic/citations.py:195`（`"title": metadata.get("title", "")`）；`backend/nanobot/rag/mcp_server/tools/agentic/batch_retrieval.py:670-680`（chunk 字段访问模式 `chunk.get("metadata", {}).get("source_path")`） |
| 6 | **`_stdio_env` 透传 `DATABASE_URL`/`REDIS_URL`**——A3 子进程自建 PG-backed SessionManager 的前提成立 | `backend/nanobot/agent/tools/mcp.py:273-280`（`_stdio_env = {**_os.environ, **(cfg.env or {})}` 全量透传父进程 env）；`backend/nanobot/cli/commands.py:1357-1358`（父进程启动 `load_dotenv(.env)` 注入 `os.environ`）；`backend/nanobot/storage/database.py:21`（`get_database_url()` 读 `os.environ.get("DATABASE_URL")`，父进程依此连 PG，故 env 中必有此值）；`backend/nanobot/bus/redis_client.py:10`（`REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")`，同理） |
| 7 | **session_key 在主 Agent 调用点可得**——透传链 7 处的"源头"位置 | `backend/nanobot/agent/loop.py:247`（`session_key = f"{channel}:{chat_id}"`）；同文件 L245-267 `_set_tool_context` 已通过 `tool.set_session_key(session_key)` 注入到 wrapper 实例 |
| 8 | **工具返回值落 role:"tool" 的代码点**——sidecar 写入位置参考 | `backend/nanobot/agent/runner.py:169-175`（`{"role": "tool", "tool_call_id": ..., "name": ..., "content": result}`，content 为 `MCPToolWrapper.execute()` 返回的 raw_result 字符串） |
| 9 | **`Session.get_history` 保留字段白名单**——B 类 sidecar 字段 `_chunk_titles` 必须加入此白名单 | `backend/nanobot/session/manager.py:82-86`（`for key in ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks")`） |
| 10 | **内部 `SessionState` ≠ 外层 `SessionManager`**——两个完全不同概念，字段命名要防混淆（用 `caller_session_key`） | `backend/nanobot/rag/internal_loop/state.py:72-119`（内部 loop 的 SessionState）vs `backend/nanobot/session/manager.py:116-237`（外层会话 SessionManager） |
| 11 | **`PlanQueryTool` 下游已接好**——`input_schema` + `execute()` + `plan_query_handler` 三处已透传 session_key，不要重改 | `backend/nanobot/rag/mcp_server/tools/agentic/query_planning.py:172-190`（input_schema 含 session_key）；同文件 L192-271（execute 已接入参）；同文件 L891-902（handler 透传） |
| 12 | **BM25 sparse index 与向量库 chunk_id 不一致 → 真实 `kb_search` 检索 `chunks=[]` → B 类验收前置阻断**：B 类 §4.2 `_chunk_titles` sidecar 写入依赖 `_save_turn` 在真实 role:"tool" content 里能读到 `chunk["metadata"]["title"]`；当前 KB 实跑三条 query 全部 `chunks=[]`，sidecar 永远是空的，B 类即使代码写对也跑不出可验证结果。**B 类动工前必须先修索引（重建 BM25 或对齐 chunk_id），否则 §7.2 真实两轮验证不可达。** | `docs/superpowers/specs/baselines/pre-5p2-runlog.txt`（2026-06-27 跑,大量 `nanobot.rag.core.query_engine.sparse_retriever WARNING No record found in vector store for chunk_id='...'  Skipping this result`）+ `pre-5p2-q{1,2,3}-toolcontent.txt` 三条均 `"chunks": []` |
| 13 | **部分检索工具 content 是 "JSON + \n\n + Markdown" 混排，非裸 JSON → B 类 `_extract_chunk_titles` 不能假设输入恒为合法 JSON**：`ExecuteRetrievalBatchTool` / `FuseAndFetchRoundTool` / `GetRoundStatusTool` 三处返回的 `MCPToolResponse.content` 是 `json.dumps(...) + "\n\n" + markdown` 拼接。`§1.4` 复杂路径 `_execute_complex → RAGLoopRunner → ... → build_json_response` 走的是裸 JSON 路径，但上述三个工具在内部多轮检索中被调用——**B 类动工前必须先确认这三个工具的输出是否会落入顶层 role:"tool" 消息**（即外层 agent 看到的 tool message）。若会：`_extract_chunk_titles` 必须先按 `"\n\n"` 分离 JSON 段再 `json.loads`，或直接正则抽 `metadata.title`；且 `json.loads` 失败须容错降级（§4.2 已覆盖 16K 截断这一类，需显式补"混排"这一类失败模式）。若不会（仅作为内部 loop 子工具、不直返）：B 类按裸 JSON 处理即可。**这一条不影响 §5.2 merge**——剥壳对混排文本同样更优、不引入退化。 | `backend/nanobot/rag/mcp_server/tools/agentic/batch_retrieval.py:561,685,774`（三处 `content=json.dumps(response_data, ensure_ascii=False, indent=2) + "\n\n" + markdown` 拼接）；对比 `backend/nanobot/rag/mcp_server/tools/agentic/shared.py:44-61`（`build_json_response` 唯一 `json.dumps`、纯裸 JSON）。普查表：§5.2 PR 验证日志（claude session 2026-06-27）|
| 14 | **`rag_search` → `kb_search/kb_retrieve/memory_search` rename 是工作树半成品（从未 commit），且 system prompt 仍指导模型用旧名 → A 类前置阻断**：注册名层早切到 `kb_search`（`tools/rag_search.py:91` 的 `class.name` 返回 `"kb_search"`），agent 侧消费者也已硬编码新名（`agent/tools/mcp.py:85-180`），但 **system prompt `agent/context.py:279` 仍告诉模型"use rag_search"** —— 模型被指导一个不存在的工具名，A 类的"外层 agent 调 kb_search 并透传 session_key"前提靠模型自纠错才能触发。rename 半成品共 3 处残留需 PR 2 动 `agent/context.py` 时**一并修全**：(1) `tools/__init__.py:52-95` 注释 + import 路径，(2) `agent/context.py:279` system prompt，(3) `rag/internal_loop/state.py:76` docstring。**§5.2 不动这块**——`protocol_handler.py:196-227` 工作树里那段 docstring 跟随 hunk 也属于此 rename 残留，§5.2 commit 拒掉、留工作树，由 PR 2 统一处理。 | `git log -S "kb_search"` 返回 0（rename 从未 commit）；`backend/nanobot/rag/mcp_server/tools/rag_search.py:91`（`return "kb_search"`）；`backend/nanobot/agent/tools/mcp.py:85,86,131,144,164`（新名硬编码）；`backend/nanobot/agent/context.py:279`（旧名 system prompt 残留）；`backend/nanobot/rag/mcp_server/tools/__init__.py:52,53,95`（旧名 import）；`backend/nanobot/rag/internal_loop/state.py:76`（旧名 docstring）|

---

## 9. 验证清单（实施时按此勾选）

- [ ] §5.2 merge 后：跑 1 条普通 `kb_search`，确认 role:"tool" content 已是裸 JSON、主路径回答质量不降。
- [ ] A 类 merge 后：跑合成 idx=37（指代 assistant 前文），日志 `rewritten_query != original_query` 且语义对。
- [ ] B 类 merge 后：**真实**两轮 `kb_search`——
  - [ ] 第一轮 session.messages 检查 role:"tool" 含 `_chunk_titles` 字段；
  - [ ] 第二轮 prompt 渲染检查 retrieval_section 非空；
  - [ ] 第二轮 `rewritten_query` 包含具体论文 title 之一。
- [ ] 全部 merge 后才可起 query cache spec。
