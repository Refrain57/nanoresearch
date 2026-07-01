# Chat Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在聊天气泡里给 LLM 的回答展示结构化、可点击、可持久化的引用来源(provenance),复刻现有 `tool_call` 旁路机制。

**Architecture:** 引用在 `kb_search` 工具边界生成 → `loop.py` 的 `after_iteration` 钩子捕获/去重/累积 → 经新 `on_citations` 回调既发 `citations` SSE 事件(直播)又嵌入 assistant 消息 `content._citations`(持久化/重进)→ 回放给 LLM 前由 `_sanitize_empty_content` 剥离顶层 `_citations` 防污染 → 前端 `useRunStream` 收事件、`stores/chat.js` 挂到消息、`MessageList.vue` 渲染面板。

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy(async)/ Redis Streams / arq;Vue3 + Pinia。

**Spec:** `docs/superpowers/specs/2026-06-30-chat-citations-design.md`(Option A,2026-07-01 定)

## Global Constraints

- 仅覆盖 **agentic 路径**(走 `loop.py` 钩子);simple RAG(`worker.py` 简单路径)本计划不接,留第二步。
- 检索工具集:`kb_search` + `retrieve_by_entity`。
- 点击引用 = **展开看片段 + page**,不做跳转打开源文档。
- `source_path` 原样显示,不美化;正文不插 `[n]`,引用只在独立面板。
- **不加数据库列、不改迁移**:citations 嵌入 `Message.content._citations`,复刻 `content.tool_calls` 重进机制。
- Citation item 字段:`index / chunk_id / source / score / snippet / page / doc_id`。
- 去重:一轮内多次检索按 `chunk_id` 去重,`index` 在单条 assistant 消息内重排(1..N)。

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `backend/nanoresearch/providers/base.py` | LLM 请求前消息清洗 | 扩展 `_sanitize_empty_content` 剥离顶层 `_citations`(安全闸) |
| `backend/nanoresearch/rag/mcp_server/tools/rag_search.py` | 检索工具 | `_execute_simple`/`_execute_complex` 填 `citations` |
| `backend/nanoresearch/rag/mcp_server/tools/agentic/shared.py` | 引用构建复用 | 新增 `build_citations_from_chunks(chunks)` 纯函数 |
| `backend/nanoresearch/agent/loop.py` | Agent 主循环 | `on_citations` 回调 + `after_iteration` 捕获去重 + `_save_turn` 嵌入 `_citations` |
| `backend/nanoresearch/worker.py` | 后台执行 + 事件 | `citations_log` + `on_citations`→`xadd` + 传参 |
| `web/src/composables/useRunStream.js` | SSE 消费 | 新增 `citations` 事件分支 |
| `web/src/stores/chat.js` | 消息状态 | 直播挂载 + 重进读 `content._citations` |
| `web/src/components/MessageList.vue` | 消息渲染 | 新增引用折叠面板 |
| `backend/tests/...`、`web/...` | 测试 | 各任务自带 |

---

## Task 1: `_sanitize_empty_content` 剥离顶层 `_citations`(安全闸,必须最先)

**Files:**
- Modify: `backend/nanoresearch/providers/base.py`(`_sanitize_empty_content`,约 `:105-140`)
- Test: `backend/tests/providers/test_sanitize_citations.py`(Create)

**Interfaces:**
- Consumes: 现有 `_sanitize_empty_content(messages: list[dict]) -> list[dict]`(staticmethod)。
- Produces: 同签名;额外保证:返回的每条消息 dict **不含顶层 `_citations` 键**;`content`/`role`/`tool_calls` 不变。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/providers/test_sanitize_citations.py
from nanoresearch.providers.base import LLMProvider  # 持有 _sanitize_empty_content 的类

def test_sanitize_strips_top_level_citations():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "_citations": [{"index": 1, "source": "x.pdf"}]},
    ]
    out = LLMProvider._sanitize_empty_content(msgs)
    assert all("_citations" not in m for m in out)
    # 不误伤正常字段
    assert out[1]["content"] == "a"
    assert out[1]["role"] == "assistant"

def test_sanitize_keeps_tool_calls():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]}]
    out = LLMProvider._sanitize_empty_content(msgs)
    assert out[0].get("tool_calls") == [{"id": "t1"}]
```

> 注:`_sanitize_empty_content` 定义在 `providers/base.py` 的哪个类上,实现时按文件确认导入名(`:105` 上文的 class)。若是模块级函数则直接导入函数。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/providers/test_sanitize_citations.py -v`
Expected: FAIL（`_citations` still present）

- [ ] **Step 3: 实现 —— 在 sanitize 里剥离顶层 `_citations`**

在 `_sanitize_empty_content` 的循环里,构造返回 dict 时排除顶层 `_citations`。最稳妥:在函数**入口处先归一**,对每条 msg 去掉 `_citations` 再走原逻辑:

```python
@staticmethod
def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize message content: fix empty blocks, strip internal _meta / _citations fields."""
    # 先剥离顶层内部字段(_citations),不影响后续逻辑
    messages = [
        {k: v for k, v in m.items() if k != "_citations"} if isinstance(m, dict) else m
        for m in messages
    ]
    result: list[dict[str, Any]] = []
    for msg in messages:
        ...  # 原有逻辑不变
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/providers/test_sanitize_citations.py -v`
Expected: PASS

- [ ] **Step 5: 回归现有 provider 测试**

Run: `cd backend && python -m pytest tests/providers/ -q`
Expected: 原有用例不破。

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/providers/base.py backend/tests/providers/test_sanitize_citations.py
git commit -m "feat(providers): strip internal _citations from messages before LLM call"
```

---

## Task 2: 引用构建纯函数 + `kb_search` 填充 `citations`

**Files:**
- Modify: `backend/nanoresearch/rag/mcp_server/tools/agentic/shared.py`(新增纯函数)
- Modify: `backend/nanoresearch/rag/mcp_server/tools/rag_search.py`(`_execute_simple` `:245-251`、`_execute_complex` 返回处)
- Test: `backend/tests/rag/test_build_citations_from_chunks.py`(Create)

**Interfaces:**
- Produces: `build_citations_from_chunks(chunks: list[dict], start_index: int = 1) -> list[dict]`
  - 入参 chunk dict 形如 `{"chunk_id": str, "score": float, "text": str, "metadata": {...}}`。
  - 返回 citation item:`{"index": int, "chunk_id": str, "source": str, "score": float, "snippet": str, "page": int|None, "doc_id": str}`。
  - 按 `chunk_id` 去重(保留首次出现),`index` 从 `start_index` 连续编号,`snippet` 截断 200 字。
- `kb_search` 的 JSON 结果中 `citations` 字段从 `None` 变为该列表。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/rag/test_build_citations_from_chunks.py
from nanoresearch.rag.mcp_server.tools.agentic.shared import build_citations_from_chunks

def test_build_basic_and_dedup():
    chunks = [
        {"chunk_id": "c1", "score": 0.9, "text": "T" * 300,
         "metadata": {"source_path": "a.pdf", "page": 3, "doc_id": "d1"}},
        {"chunk_id": "c1", "score": 0.8, "text": "dup", "metadata": {"source_path": "a.pdf"}},
        {"chunk_id": "c2", "score": 0.7, "text": "hello",
         "metadata": {"source": "b.md"}},
    ]
    out = build_citations_from_chunks(chunks)
    assert [c["index"] for c in out] == [1, 2]          # 去重后两条，序号连续
    assert out[0]["chunk_id"] == "c1"
    assert out[0]["source"] == "a.pdf"
    assert out[0]["page"] == 3
    assert out[0]["doc_id"] == "d1"
    assert len(out[0]["snippet"]) <= 203                 # 200 + "..."
    assert out[1]["source"] == "b.md"                    # 回退到 metadata.source
    assert out[1]["page"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/rag/test_build_citations_from_chunks.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现纯函数**(`agentic/shared.py` 追加)

```python
def build_citations_from_chunks(chunks: list[dict], start_index: int = 1) -> list[dict]:
    """Build deduped, indexed citation items from retrieval chunks."""
    out: list[dict] = []
    seen: set[str] = set()
    idx = start_index
    for c in chunks or []:
        cid = c.get("chunk_id") or ""
        if not cid or cid in seen:
            continue
        seen.add(cid)
        md = c.get("metadata") or {}
        text = (c.get("text") or "").strip()
        snippet = text[:200] + "..." if len(text) > 200 else text
        page = md.get("page", md.get("page_num"))
        try:
            page = int(page) if page is not None else None
        except (ValueError, TypeError):
            page = None
        out.append({
            "index": idx,
            "chunk_id": cid,
            "source": md.get("source_path", md.get("source", "unknown")),
            "score": round(float(c.get("score", 0.0)), 4),
            "snippet": snippet,
            "page": page,
            "doc_id": md.get("doc_id", ""),
        })
        idx += 1
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/rag/test_build_citations_from_chunks.py -v`
Expected: PASS

- [ ] **Step 5: 接入 `kb_search`**

`rag_search.py` `_execute_simple`(`:245`)把 `"citations": None` 改为:

```python
from nanoresearch.rag.mcp_server.tools.agentic.shared import build_citations_from_chunks
return build_json_response({
    "success": True,
    "chunks": fused_chunks,
    "citations": build_citations_from_chunks(fused_chunks),
    "summary": f"Simple retrieval completed with {len(fused_chunks)} chunks",
    "iterations": 1,
})
```
`_execute_complex` 在其返回 chunks 处同样填 `citations`(用最终 fused/verified chunks 调同一函数)。

- [ ] **Step 6: Commit**

```bash
git add backend/nanoresearch/rag/mcp_server/tools/agentic/shared.py backend/nanoresearch/rag/mcp_server/tools/rag_search.py backend/tests/rag/test_build_citations_from_chunks.py
git commit -m "feat(rag): kb_search emits structured citations from retrieved chunks"
```

---

## Task 3: `loop.py` 捕获引用 + `on_citations` 回调 + 嵌入 `content._citations`

**Files:**
- Modify: `backend/nanoresearch/agent/loop.py`(`after_iteration` `:392-400`;`_run_agent_loop`/`process_direct` 签名 `:325-326,706-708,845-846`;`_save_turn` `~:875-899`)
- Test: `backend/tests/agent/test_loop_citations.py`(Create)

**Interfaces:**
- Consumes: `build_citations_from_chunks`(Task 2);工具结果字符串(RAG 工具返回的 JSON,含 `chunks`/`citations`)。
- Produces:
  - `process_direct(..., on_citations: Callable[[list[dict]], Awaitable[None]] | None = None)` 新参(与 `on_tool_call` 平行)。
  - 一轮内累积的去重引用列表,经 `on_citations(items)` 上抛,并写入该轮 assistant 消息的 `content["_citations"]`。
- 检索工具集常量:`_CITATION_TOOLS = {"kb_search", "retrieve_by_entity"}`。

- [ ] **Step 1: 写失败测试**(用脚本化 provider + 假工具结果,断言 on_citations 收到去重引用)

```python
# backend/tests/agent/test_loop_citations.py
import json, pytest
# 复用 tests/agent 现有 fixtures 风格(参见 test_runner.py 的脚本化 provider/工具)。
@pytest.mark.asyncio
async def test_after_iteration_forwards_deduped_citations(loop_with_scripted_rag):
    """两次 kb_search 命中同一 chunk_id，on_citations 收到的合并列表去重且序号连续。"""
    loop, captured = loop_with_scripted_rag  # fixture: 见下方实现说明
    got: list[dict] = []
    await loop.process_direct(
        "q", session_key="s1", on_citations=lambda items: got.extend(items) or _async_noop(),
    )
    cids = [c["chunk_id"] for c in got]
    assert len(cids) == len(set(cids))               # 去重
    assert [c["index"] for c in got] == list(range(1, len(got) + 1))  # 序号连续
```

> Fixture 说明(实现时落在 `tests/agent/conftest.py` 或本文件):脚本化 provider 第一轮返回一个 `kb_search` 工具调用、第二轮返回纯文本;脚本化工具层让 `kb_search` 返回 `{"chunks":[...], "citations":[{chunk_id:"c1",...}]}` 两次含重复 `c1`。参照 `tests/agent/test_runner.py:39-43` 的 `LLMResponse(tool_calls=[ToolCallRequest(...)])` 写法。`on_citations` 须为 async,可用一个 `async def collector(items): got.extend(items)`。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/agent/test_loop_citations.py -v`
Expected: FAIL（`process_direct` 无 `on_citations` 参数 / 未转发）

- [ ] **Step 3: 实现 —— 签名 + 累积 + 转发**

3a. 给 `process_direct` 与内部 `_run_agent_loop` 增加 `on_citations` 参数(平行于 `on_tool_call`,见 `:326,707,846`),一路透传到运行 hooks 的构造处。

3b. 在 hooks 的 `after_iteration`(`:392-400`)里,`on_tool_call` 之后追加引用提取(turn 级累积器 `loop_self._turn_citations: list[dict]`,在 turn 开始清空):

```python
_CITATION_TOOLS = {"kb_search", "retrieve_by_entity"}
...
async def after_iteration(self, context):
    if on_tool_call and context.tool_calls:
        for tc, result in zip(context.tool_calls, context.tool_results or []):
            ...  # 现有 on_tool_call 逻辑不变
            if tc.name in _CITATION_TOOLS:
                items = _extract_citations(result)  # 见 3c
                if items:
                    merged = _merge_citations(loop_self._turn_citations, items)  # 去重+重排
                    loop_self._turn_citations = merged
                    if on_citations:
                        await on_citations(merged)
```

3c. 辅助(放 `loop.py` 模块级或工具模块):

```python
def _extract_citations(result) -> list[dict]:
    """从工具结果(str JSON 或 dict)取 citations / 或由 chunks 现build。"""
    import json
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(data, dict):
        return []
    cites = data.get("citations")
    if cites:
        return cites
    chunks = data.get("chunks")
    if chunks:
        from nanoresearch.rag.mcp_server.tools.agentic.shared import build_citations_from_chunks
        return build_citations_from_chunks(chunks)
    return []

def _merge_citations(existing: list[dict], new: list[dict]) -> list[dict]:
    """按 chunk_id 去重合并，index 全列表重排 1..N。"""
    by_id = {c["chunk_id"]: c for c in existing}
    for c in new:
        by_id.setdefault(c["chunk_id"], c)
    merged = list(by_id.values())
    for i, c in enumerate(merged, 1):
        c["index"] = i
    return merged
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/agent/test_loop_citations.py -v`
Expected: PASS

- [ ] **Step 5: 嵌入 `content._citations`(持久化路径)**

打开 `_save_turn`(`loop.py` `~:875-899`,assistant 消息 dict 被 append 到 session 的地方)。在构造 assistant 消息 dict 处,若 `loop_self._turn_citations` 非空,加入键:

```python
assistant_msg = {"role": "assistant", "content": text}
if tool_calls_payload:
    assistant_msg["tool_calls"] = tool_calls_payload   # 现有逻辑
if self._turn_citations:
    assistant_msg["_citations"] = self._turn_citations  # 新增：随 content 入库
```
> 注:`_turn_citations` 须在每个 turn 起点(`_process_message`/`process_direct` 进入处)重置为 `[]`,避免跨 turn 串。

- [ ] **Step 6: 写持久化嵌入测试 + 跑**

```python
@pytest.mark.asyncio
async def test_save_turn_embeds_citations_in_content(loop_with_scripted_rag):
    loop, _ = loop_with_scripted_rag
    await loop.process_direct("q", session_key="s2", on_citations=_noop)
    saved = loop.sessions.get_or_create("s2").messages  # 或对应读取 API
    asst = [m for m in saved if m.get("role") == "assistant"][-1]
    assert asst.get("_citations") and asst["_citations"][0]["index"] == 1
```

Run: `cd backend && python -m pytest tests/agent/test_loop_citations.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/nanoresearch/agent/loop.py backend/tests/agent/test_loop_citations.py
git commit -m "feat(agent): capture/dedup RAG citations, forward via on_citations, embed in content"
```

---

## Task 4: worker 发 `citations` SSE 事件 + 传参

**Files:**
- Modify: `backend/nanoresearch/worker.py`(`:542` 邻近加 `citations_log`/`on_citations`;`:699` 等 `process_direct(...)` 调用处传 `on_citations`)
- Test: `backend/tests/test_worker_citations_event.py`(Create)

**Interfaces:**
- Consumes: Task 3 的 `process_direct(..., on_citations=...)`。
- Produces: Redis stream 上出现 `{"type": "citations", "items": [...]}` 事件(与 `tool_call` 同 `xadd_event` 通道,`:561`)。

- [ ] **Step 1: 写失败测试**(假 redis 捕获 xadd,断言出现 citations 事件)

```python
# backend/tests/test_worker_citations_event.py
import pytest
@pytest.mark.asyncio
async def test_on_citations_emits_sse_event(fake_redis, run_ctx):
    """worker 的 on_citations 回调应 xadd 一个 type=citations 的事件。"""
    on_citations = run_ctx.make_on_citations(fake_redis)   # 见实现说明
    await on_citations([{"index": 1, "chunk_id": "c1", "source": "a.pdf"}])
    events = fake_redis.xadds_for(run_ctx.run_stream_key)
    assert any(e.get("type") == "citations" for e in events)
    cite_ev = next(e for e in events if e["type"] == "citations")
    assert cite_ev["items"][0]["source"] == "a.pdf"
```
> 实现说明:把 worker 里内联的 `on_citations` 抽成可测的小工厂 `_make_on_citations(redis, run_stream_key)`(与现有 `on_tool_call` 同构),便于单测;`fake_redis.xadds_for` 收集 `xadd_event` 写入(参照本仓现有 worker/stream 测试的 fake redis 风格)。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/test_worker_citations_event.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 —— worker 内 on_citations + 传参**

在 `worker.py` `:542` 附近(`tool_calls_log` 旁)新增,并仿 `on_tool_call`(`:551-565`):

```python
async def on_citations(items: list[dict]) -> None:
    await xadd_event(redis, run_stream_key, {"type": "citations", "items": items})
```
在三处 `process_direct(...)` 调用(`:606-610`、`:652-656`、`:697-701`)的参数里加 `on_citations=on_citations`(与 `on_tool_call=on_tool_call` 并列)。

> citations 已随 assistant 消息 `content._citations` 持久化(Task 3),worker **无需**额外写库;`run_repo.update`(`:735`)不动。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/test_worker_citations_event.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/nanoresearch/worker.py backend/tests/test_worker_citations_event.py
git commit -m "feat(worker): emit citations SSE event and wire on_citations into agent runs"
```

---

## Task 5: 前端 —— SSE 分支 + store 挂载 + 引用面板

**Files:**
- Modify: `web/src/composables/useRunStream.js`(`dispatch` `:28-44`、`start` 回调签名 `:6`)
- Modify: `web/src/stores/chat.js`(直播 `:96` `finalizeStream`;重进 `:36-42`)
- Modify: `web/src/components/MessageList.vue`(模板 `:22-35` 工具面板旁加引用面板)
- Test: `web/tests/...`(若有前端测试设施则加;否则手动验证步骤见 Step 6)

**Interfaces:**
- Consumes: SSE `{"type":"citations","items":[...]}`;持久化消息 `content._citations`。
- Produces: 每条 assistant 消息上 `msg.citations: list`(直播与重进统一字段)。

- [ ] **Step 1: useRunStream 新增 citations 分支**

`useRunStream.js` `start(...)` 解构加 `onCitations`(`:6`);`dispatch`(`:34-39`)加:

```js
else if (event.type === 'citations') onCitations?.(event)
```

- [ ] **Step 2: chat store —— 直播挂载**

在调用 `useRunStream().start(...)` 的地方传入 `onCitations`,把 `event.items` 暂存到当前流式消息的 `citations`,并在 `finalizeStream`(`:96-103`)落到最终消息:

```js
// 流式期间
function onCitations(ev) { streamingCitations.value = ev.items }
// finalizeStream:
messages.value.push({
  id: `stream-${Date.now()}`,
  role: 'assistant',
  content: { text: streamingText.value },
  toolCalls: toolCalls.length ? [...toolCalls] : undefined,
  citations: streamingCitations.value?.length ? [...streamingCitations.value] : undefined,
  seq: messages.value.length,
})
// 之后清空 streamingCitations
```

- [ ] **Step 3: chat store —— 重进读 `content._citations`**

`selectConversation` 的映射(`:36-42`)里,与 `tool_calls` 并列读出:

```js
const tool_calls = m.tool_calls ?? stored?.tool_calls
const citations = m.content?._citations ?? stored?._citations
return { ...m, content: { text }, tool_calls, toolCalls: _normalizeToolCalls(tool_calls), citations }
```
> 合并"tool-only 消息到后续文本消息"的逻辑(`:51-56`)同样把 `citations` 一并带过去(仿 `toolCalls` 的 `pendingTc`)。

- [ ] **Step 4: MessageList 引用面板**

`MessageList.vue` 在工具面板(`:22-35`)之后加(仿其折叠结构):

```vue
<div v-if="msg.role === 'assistant' && msg.citations?.length" class="citations-panel">
  <a-collapse size="small" :bordered="false">
    <a-collapse-panel key="c" :header="`引用来源 (${msg.citations.length})`">
      <div v-for="c in msg.citations" :key="c.chunk_id" class="cite-item">
        <span class="cite-idx">[{{ c.index }}]</span>
        <span class="cite-src">{{ c.source }}</span>
        <span v-if="c.page" class="cite-page">p.{{ c.page }}</span>
        <span class="cite-score">{{ (c.score * 100).toFixed(0) }}%</span>
        <div class="cite-snippet">{{ c.snippet }}</div>
      </div>
    </a-collapse-panel>
  </a-collapse>
</div>
```

- [ ] **Step 5: 构建前端确认无语法错**

Run: `cd web && npm run build`
Expected: 构建通过。

- [ ] **Step 6: 手动 e2e 验证**

1. 起服务(agentic 模式),向已建库的 KB 提问。
2. 直播:回答出现后,气泡下出现"引用来源 (N)"面板,展开见 source/page/score/片段。
3. **刷新会话** → 该 assistant 消息的引用面板**仍在**(`content._citations` 重进生效)。
4. 验证未污染:同会话继续追问,LLM 正常(无 400/未知字段错误)→ 印证 Task 1 的 sanitize 生效。

- [ ] **Step 7: Commit**

```bash
git add web/src/composables/useRunStream.js web/src/stores/chat.js web/src/components/MessageList.vue
git commit -m "feat(web): render structured citation panel in chat (live + reload)"
```

---

## Self-Review

**Spec coverage:** §4.1 生成→Task2;§4.2 捕获/去重→Task3;§4.3 传输→Task4;§4.4 存储(content._citations + sanitize)→Task1(sanitize)+Task3(嵌入);§4.5 渲染→Task5。§6 范围决策(agentic only / 展开看片段 / kb_search+retrieve_by_entity)→Global Constraints + Task3 `_CITATION_TOOLS`。§7 测试→各任务 Step。全部有对应。

**Placeholder scan:** 无 TBD/TODO;两处"实现时确认"为定位指引(sanitize 所属类、`_save_turn` 精确行)非空洞,均给了锚点与做法。

**Type consistency:** citation item 字段(`index/chunk_id/source/score/snippet/page/doc_id`)在 Task2 定义,Task3/4/5 一致引用;`on_citations: Callable[[list[dict]], Awaitable[None]]` 在 Task3 定义、Task4 实现、Task5 消费同形;`_merge_citations` 的 index 重排与 `build_citations_from_chunks` 的连续编号一致。

**顺序:** C1(安全闸)必须最先;C2、C3 可并行;C4 依赖 C3;C5 依赖 C4(直播)与 C3(持久化字段)。
