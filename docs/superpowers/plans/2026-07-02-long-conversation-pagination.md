# 长对话消息分页 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 长对话打开时加载最近 N 条、向上滚动加载更早的消息，替换当前「只加载最旧 100 条、无分页」的行为。

**Architecture:** 后端 `get_messages_paged` 改为 before-seq 游标（`seq < before_seq`，倒序取 N 再反转为升序）；`/messages` 端点加 `before_seq` 参数、响应改为 `{ messages, has_more }`（`has_more` 按内部过滤前的原始行数判定）。前端 store 保留一份未合并的原始累积数组，初次加载最近 40、`loadOlder` prepend 更早页并对整表重跑 tool-call 合并（保留流式/重连追加的尾部消息）；`MessageList` 触顶触发 `loadOlder` 并保持滚动位置。

**Tech Stack:** 后端 FastAPI + SQLAlchemy(async) + asyncpg + pytest（真实 Postgres 测试库）；前端 Vue 3 + Pinia + Ant Design Vue + Vite（**无 JS 测试框架**）。

## Global Constraints

- 初始加载与每页大小 = **40** 条。
- `/messages` 端点响应形状 = `{ "messages": [...], "has_more": bool }`（不再是裸数组）。
- `has_more` 用 repo 返回的**原始行数**判定（`len(rows) == limit`），在 internal 消息过滤**之前**计算。
- 游标语义 = `Message.seq < before_seq`；`before_seq=None` 表示取最近 N 条。
- 无 DB schema 变更。
- 前端无测试运行器，**仅后端做 TDD**；前端改动在 Task 5 通过运行 app 手动验证。已核实：`get_messages_paged` 唯一调用方是 `chat_router.py`，`getMessages` 唯一调用方是 `chat.js` —— 移除 `offset`、改响应形状都是内聚的。

---

## Prerequisites（执行环境）

在开始前确认（不满足则先解决，否则后端测试无法运行）：

- **Postgres 测试库可达**：`conftest.py` 默认连 `postgresql+asyncpg://postgres:123456@localhost:5432/nanoresearch_test`（psycopg2 DSN 同库）。库需存在；表由 `setup_database` fixture 自动建。如本机配置不同，设 `TEST_DATABASE_URL` / `TEST_DATABASE_DSN` 环境变量。
- **后端依赖装在本工区**：在 `<worktree>/backend` 下创建/激活 venv 并可编辑安装，使 `import nanoresearch` 解析到**本工区**源码：
  ```bash
  cd backend
  python -m pip install -e .
  ```
  之后所有后端测试用 `cd backend && python -m pytest ...` 运行。
- **前端依赖**（仅 Task 5 需要）：`cd web && npm install`。

---

## Task 1: 后端 repo — `get_messages_paged` 游标分页

**Files:**
- Modify: `backend/nanoresearch/storage/repositories/conversation_repo.py:57-66`
- Test: `backend/tests/test_repositories.py`（在 Conversation 段末尾追加）

**Interfaces:**
- Produces: `ConversationRepository.get_messages_paged(conv_id: uuid.UUID, limit: int = 40, before_seq: int | None = None) -> list[Message]`，返回**升序**列表；`before_seq=None` 时为最近 `limit` 条；否则为 `seq < before_seq` 的最近 `limit` 条。

- [ ] **Step 1: 写失败测试**

在 `backend/tests/test_repositories.py` 末尾追加：

```python
def test_get_messages_paged_recent_first_and_cursor():
    async def _():
        user_repo = UserRepository(make_factory())
        await user_repo.create("grace", hash_password("pw"))
        conv_repo = ConversationRepository(make_factory())
        conv = await conv_repo.create(key="web:paged", uid="grace")
        await conv_repo.replace_messages(
            conv.id, [{"role": "user", "content": f"m{i}"} for i in range(5)]
        )
        # 最近 2 条，升序
        recent = await conv_repo.get_messages_paged(conv.id, limit=2)
        assert [m.seq for m in recent] == [3, 4]
        # seq < 3 的最近 2 条
        older = await conv_repo.get_messages_paged(conv.id, limit=2, before_seq=3)
        assert [m.seq for m in older] == [1, 2]
        # limit 超过总数 → 全部升序
        allm = await conv_repo.get_messages_paged(conv.id, limit=10)
        assert [m.seq for m in allm] == [0, 1, 2, 3, 4]
    run(_())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/test_repositories.py::test_get_messages_paged_recent_first_and_cursor -v`
Expected: FAIL（当前实现按升序 + offset，`recent` 会是 `[0, 1]` 而非 `[3, 4]`）。

- [ ] **Step 3: 实现**

将 `conversation_repo.py:57-66` 的 `get_messages_paged` 整体替换为：

```python
    async def get_messages_paged(
        self, conv_id: uuid.UUID, limit: int = 40, before_seq: int | None = None
    ) -> list[Message]:
        async with self._factory() as db:
            stmt = select(Message).where(Message.conversation_id == conv_id)
            if before_seq is not None:
                stmt = stmt.where(Message.seq < before_seq)
            stmt = stmt.order_by(Message.seq.desc()).limit(limit)
            rows = list((await db.execute(stmt)).scalars().all())
            rows.reverse()  # 升序返回，前端可直接 append/prepend
            return rows
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/test_repositories.py::test_get_messages_paged_recent_first_and_cursor -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/storage/repositories/conversation_repo.py backend/tests/test_repositories.py
git commit -m "feat(chat): before-seq cursor pagination in get_messages_paged (#2)"
```

---

## Task 2: 后端端点 — `before_seq` 参数 + `{messages, has_more}` 响应

**Files:**
- Modify: `backend/nanoresearch/server/routers/chat_router.py:132-156`
- Test: `backend/tests/test_chat_api.py`（更新 2 个既有测试 + 新增 4 个）

**Interfaces:**
- Consumes: `get_messages_paged(conv_id, limit, before_seq)` from Task 1。
- Produces: `GET /api/conversations/{conv_id}/messages?limit=40&before_seq=<int|None>` → `{ "messages": [ {id, role, content, seq, created_at} ], "has_more": bool }`。

- [ ] **Step 1: 更新既有 2 个测试为新响应形状**

在 `backend/tests/test_chat_api.py` 中，把 `test_get_messages_empty`（约 172 行）改为：

```python
def test_get_messages_empty(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        resp = client.get(f"/api/conversations/{conv_id}/messages", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == {"messages": [], "has_more": False}
```

删除既有的 `test_get_messages_pagination`（约 180-198 行，用了 `offset` 且断言裸数组），本任务 Step 2 用新测试替代。

- [ ] **Step 2: 写新失败测试**

在 `test_chat_api.py` 的 Messages 段追加：

```python
def _seed_msgs(conv_id_str, msgs):
    import uuid
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository

    async def _():
        repo = ConversationRepository(make_factory())
        await repo.replace_messages(uuid.UUID(conv_id_str), msgs)
    run(_())


def test_get_messages_recent_first(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        _seed_msgs(conv_id, [{"role": "user", "content": f"m{i}"} for i in range(5)])
        resp = client.get(f"/api/conversations/{conv_id}/messages?limit=2", headers=auth_headers)
    body = resp.json()
    assert [m["seq"] for m in body["messages"]] == [3, 4]
    assert body["has_more"] is True


def test_get_messages_before_seq(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        _seed_msgs(conv_id, [{"role": "user", "content": f"m{i}"} for i in range(5)])
        resp = client.get(
            f"/api/conversations/{conv_id}/messages?limit=2&before_seq=3", headers=auth_headers
        )
    body = resp.json()
    assert [m["seq"] for m in body["messages"]] == [1, 2]
    assert body["has_more"] is True


def test_get_messages_has_more_false(app, auth_headers):
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        _seed_msgs(conv_id, [{"role": "user", "content": f"m{i}"} for i in range(5)])
        resp = client.get(f"/api/conversations/{conv_id}/messages?limit=10", headers=auth_headers)
    body = resp.json()
    assert len(body["messages"]) == 5
    assert body["has_more"] is False


def test_get_messages_has_more_counts_internal(app, auth_headers):
    """has_more 用原始行数判定（internal 过滤之前）。"""
    with TestClient(app) as client:
        conv_id = client.post("/api/conversations", json={}, headers=auth_headers).json()["id"]
        _seed_msgs(conv_id, [
            {"role": "user", "content": "m0"},
            {"role": "assistant", "content": "m1"},
            {"role": "assistant", "content": "internal", "internal": True},
            {"role": "user", "content": "m3"},
        ])
        # 最近 2 条原始行 = seq[2(internal), 3]；过滤后只剩 seq3，但 has_more 仍为 True
        resp = client.get(f"/api/conversations/{conv_id}/messages?limit=2", headers=auth_headers)
    body = resp.json()
    assert [m["seq"] for m in body["messages"]] == [3]
    assert body["has_more"] is True
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/test_chat_api.py -k "get_messages" -v`
Expected: FAIL（端点仍返回裸数组、无 `before_seq`）。

- [ ] **Step 4: 实现端点**

将 `chat_router.py:132-156` 的 `get_messages` 整体替换为：

```python
@router.get("/api/conversations/{conv_id}/messages")
async def get_messages(
    conv_id: str,
    request: Request,
    limit: int = 40,
    before_seq: int | None = None,
    uid: str = Depends(get_current_user),
):
    conv = await _get_conv_or_404(conv_id, uid, request)
    factory = request.app.state.session_factory
    repo = ConversationRepository(factory)
    rows = await repo.get_messages_paged(conv.id, limit=limit, before_seq=before_seq)
    has_more = len(rows) == limit  # 原始行数判定，先于 internal 过滤
    messages = [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "seq": m.seq,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in rows
        # Hide internal orchestration turns (subagent results + continuation instruction).
        if not (isinstance(m.content, dict) and m.content.get("internal"))
    ]
    return {"messages": messages, "has_more": has_more}
```

- [ ] **Step 5: 跑全部相关测试确认通过**

Run: `cd backend && python -m pytest tests/test_chat_api.py -k "get_messages" -v`
Expected: PASS（5 个 `get_messages*` 测试全绿）。

- [ ] **Step 6: 提交**

```bash
git add backend/nanoresearch/server/routers/chat_router.py backend/tests/test_chat_api.py
git commit -m "feat(chat): /messages before_seq param + {messages,has_more} response (#2)"
```

---

## Task 3: 前端 api + store — 分页状态与 loadOlder

**Files:**
- Modify: `web/src/apis/conversations.js:9-12`
- Modify: `web/src/stores/chat.js`

**Interfaces:**
- Consumes: `GET /messages` 新响应 `{ messages, has_more }` from Task 2。
- Produces: store 暴露 `messages`（渲染用 ref）、`oldestSeq`、`hasMore`、`loadingOlder`、`loadOlder()`；`selectConversation(id)` 改为加载最近 40 条。

> 无前端测试运行器；本任务实现后不跑单测，端到端验证在 Task 5。

- [ ] **Step 1: `getMessages` 支持可选参数并返回对象**

将 `web/src/apis/conversations.js` 的 `getMessages` 替换为（`before_seq` 为空时不进 query）：

```javascript
export const getMessages = (id, params = {}) => {
  const clean = {}
  for (const [k, v] of Object.entries(params)) if (v != null) clean[k] = v
  const qs = new URLSearchParams(clean).toString()
  return apiGet(`/api/conversations/${id}/messages${qs ? '?' + qs : ''}`)
}
```

（返回体现在是 `{ messages, has_more }` 对象，由 store 消费。）

- [ ] **Step 2: 提取纯函数 + 重写 store 加载逻辑**

编辑 `web/src/stores/chat.js`：

(a) 在文件底部 `_normalizeToolCalls` 旁新增两个模块级纯函数：

```javascript
function _mapRawMessage(m) {
  const stored = m.content
  const text = typeof stored === 'string'
    ? stored
    : (stored?.text ?? stored?.content ?? '')
  const tool_calls = m.tool_calls ?? stored?.tool_calls
  const citations = typeof stored === 'string' ? null : (stored?._citations ?? null)
  return {
    ...m,
    content: { text },
    tool_calls,
    toolCalls: _normalizeToolCalls(tool_calls),
    citations: citations?.length ? citations : undefined,
  }
}

// 后端已过滤 internal，这里 defense-in-depth 再过滤一次；映射为渲染形状（未合并）。
function _toMapped(apiMessages) {
  return (apiMessages || [])
    .filter(m => !(m.content && typeof m.content === 'object' && m.content.internal))
    .map(_mapRawMessage)
}

// 把「仅 tool_calls 的 assistant 消息」并入紧随其后的文本 assistant 消息。
function _mergeToolCallMessages(mapped) {
  const merged = []
  let pendingTc = null
  let pendingCitations = null
  for (const m of mapped) {
    if (m.role === 'assistant' && !m.content.text && m.toolCalls?.length) {
      pendingTc = m.toolCalls
      pendingCitations = m.citations ?? null
    } else if (m.role === 'assistant' && m.content.text) {
      merged.push(pendingTc
        ? { ...m, toolCalls: pendingTc, citations: m.citations ?? pendingCitations ?? undefined }
        : m)
      pendingTc = null
      pendingCitations = null
    } else {
      merged.push(m)
    }
  }
  return merged
}
```

(b) 在 `useChatStore` 内新增状态（放在现有 `streamingText` ref 之后）：

```javascript
  const oldestSeq = ref(null)
  const hasMore = ref(false)
  const loadingOlder = ref(false)
  // 内部：已加载的原始（映射但未合并）消息，升序；以及当前 messages 中「合并派生」部分的长度
  let _rawLoaded = []
  let _mergedLen = 0
  const PAGE = 40
```

(c) 用下面的实现整体替换现有 `selectConversation`（`chat.js:17-78`）：

```javascript
  async function selectConversation(id) {
    streaming.value = false
    streamingText.value = ''
    currentConvId.value = id
    messages.value = []
    _rawLoaded = []
    _mergedLen = 0
    oldestSeq.value = null
    hasMore.value = false
    loadingOlder.value = false
    try {
      const resp = await getMessages(id, { limit: PAGE })
      // Guard against race: user switched conversations while fetching
      if (currentConvId.value !== id) return
      _rawLoaded = _toMapped(resp.messages)
      const merged = _mergeToolCallMessages(_rawLoaded)
      _mergedLen = merged.length
      messages.value = merged
      oldestSeq.value = _rawLoaded.length ? _rawLoaded[0].seq : null
      hasMore.value = !!resp.has_more
    } catch (e) {
      console.error('[chat] selectConversation failed:', e)
      if (currentConvId.value === id) messages.value = []
    }
  }

  async function loadOlder() {
    if (!currentConvId.value || !hasMore.value || loadingOlder.value || oldestSeq.value == null) return
    const id = currentConvId.value
    loadingOlder.value = true
    try {
      const resp = await getMessages(id, { limit: PAGE, before_seq: oldestSeq.value })
      if (currentConvId.value !== id) return
      const olderMapped = _toMapped(resp.messages)
      if (olderMapped.length) {
        _rawLoaded = [...olderMapped, ..._rawLoaded]
        const remerged = _mergeToolCallMessages(_rawLoaded)
        // 保留流式/重连追加的尾部消息（不属于 _rawLoaded 合并派生的部分）
        const tail = messages.value.slice(_mergedLen)
        messages.value = [...remerged, ...tail]
        _mergedLen = remerged.length
        oldestSeq.value = _rawLoaded[0].seq
      }
      hasMore.value = !!resp.has_more
    } catch (e) {
      console.error('[chat] loadOlder failed:', e)
    } finally {
      loadingOlder.value = false
    }
  }
```

(d) 更新 store 的 `return {...}`，加入新状态与方法：

```javascript
  return {
    conversations, messages, currentConvId, streaming, streamingText,
    oldestSeq, hasMore, loadingOlder,
    fetchConversations, selectConversation, loadOlder, newConversation, removeConversation,
    sendMessage, appendDelta, finalizeStream
  }
```

> `finalizeStream` 与 ChatView 的重连逻辑仍直接 `push` 到 `messages.value`，这些追加落在索引 `>= _mergedLen` 的尾部，`loadOlder` 的 `tail` 切片会保留它们。

- [ ] **Step 3: 静态检查（构建）**

Run: `cd web && npm run build`
Expected: 构建成功，无未定义引用报错。

- [ ] **Step 4: 提交**

```bash
git add web/src/apis/conversations.js web/src/stores/chat.js
git commit -m "feat(chat): store cursor pagination + loadOlder, preserve live tail (#2)"
```

---

## Task 4: 前端 MessageList 触顶加载 + 滚动位置保持 + ChatView 接线

**Files:**
- Modify: `web/src/components/MessageList.vue`
- Modify: `web/src/views/ChatView.vue:115-117`

**Interfaces:**
- Consumes: store 的 `hasMore`、`loadingOlder`、`loadOlder` from Task 3。
- `MessageList` 新增 props `hasMore`、`loadingOlder`；新增 emit `load-older`。

> 无前端测试运行器；端到端验证在 Task 5。

- [ ] **Step 1: MessageList 加 props / emit / 滚动逻辑**

在 `web/src/components/MessageList.vue` 的 `<script setup>`：

(a) `defineProps` 追加两项（放在现有 props 内）：

```javascript
  hasMore: { type: Boolean, default: false },
  loadingOlder: { type: Boolean, default: false },
```

(b) 在 `defineProps({...})` 之后新增：

```javascript
const emit = defineEmits(['load-older'])
```

(c) 在 `const listRef = ref(null)` 之后新增滚动保持状态与处理器：

```javascript
const _prevScrollHeight = ref(0)
const _prevScrollTop = ref(0)
const _prepending = ref(false)

function onScroll() {
  const el = listRef.value
  if (!el) return
  if (el.scrollTop < 80 && props.hasMore && !props.loadingOlder) {
    _prevScrollHeight.value = el.scrollHeight
    _prevScrollTop.value = el.scrollTop
    _prepending.value = true
    emit('load-older')
  }
}
```

(d) 用下面的实现替换现有的 `watch(...)`（`MessageList.vue:96-99`）：

```javascript
watch(() => [props.messages.length, props.streamingText], async () => {
  await nextTick()
  const el = listRef.value
  if (!el) return
  if (_prepending.value) {
    // prepend 了更早消息：保持视口锚定在原先可见的消息
    el.scrollTop = _prevScrollTop.value + (el.scrollHeight - _prevScrollHeight.value)
    _prepending.value = false
  } else {
    el.scrollTop = el.scrollHeight
  }
})
```

(e) 模板里给滚动容器绑定 `@scroll` 并在顶部加载入提示。将 `MessageList.vue:2` 改为：

```html
  <div class="message-list" ref="listRef" @scroll="onScroll">
    <div v-if="loadingOlder" class="loading-older">加载更早消息…</div>
```

(f) 在 `<style scoped>` 末尾（`}` 前最后一条规则后）追加：

```css
.loading-older { text-align: center; color: var(--nr-ink-3); font-size: 12px; padding: 4px 0; }
```

- [ ] **Step 2: ChatView 接线**

在 `web/src/views/ChatView.vue` 把 MessageList 的绑定（`:115-117`）扩展为：

```html
            :messages="chatStore.messages"
            :streaming-text="chatStore.streamingText"
            :streaming="chatStore.streaming"
            :has-more="chatStore.hasMore"
            :loading-older="chatStore.loadingOlder"
            @load-older="chatStore.loadOlder"
```

- [ ] **Step 3: 静态检查（构建）**

Run: `cd web && npm run build`
Expected: 构建成功。

- [ ] **Step 4: 提交**

```bash
git add web/src/components/MessageList.vue web/src/views/ChatView.vue
git commit -m "feat(chat): scroll-up loads older messages, preserve scroll position (#2)"
```

---

## Task 5: 端到端验证（运行 app）

**Files:** 无（仅验证）

- [ ] **Step 1: 起后端 + 前端**

按项目常规起服务（后端 API + `cd web && npm install && npm run dev`）。确保连的是有长对话（>40 条消息）的库；若无，先在一个对话里发够 40+ 条消息。

- [ ] **Step 2: 手动验证清单**

- 打开长对话：默认停在**底部**、显示的是**最新**消息（不再是最旧 100 条）。
- 向上滚动到顶：自动加载更早一页，出现「加载更早消息…」提示；加载后视口**不跳动**，停在原先那条消息。
- 反复上滚直到最早一条：到头后不再触发加载（`has_more=false`）。
- tool-call 折叠面板在跨页加载后仍正确挂在对应 assistant 消息上（不丢 `toolCalls`）。
- 发一条新消息 / 流式回复：仍自动滚到底；随后上滚加载更早，不影响刚发的消息。

- [ ] **Step 3: 回归 — 后端全测**

Run: `cd backend && python -m pytest tests/test_repositories.py tests/test_chat_api.py -v`
Expected: 全绿。

- [ ] **Step 4: 完成提交（如验证中有微调）**

```bash
git add -A
git commit -m "test(chat): verify long-conversation pagination end-to-end (#2)"
```

---

## Self-Review

- **Spec coverage**：后端游标查询→Task 1；端点 `before_seq`+`{messages,has_more}`+has_more 原始计数→Task 2；前端 api/store 分页+loadOlder+保留 live tail→Task 3；触顶加载+位置保持+底部自动滚区分→Task 4；短对话/internal/流式边界→Task 2(internal 测试)、Task 3(tail 保留)、Task 5(手动)。既有 2 个会被打破的测试→Task 2 Step 1 显式更新。
- **Placeholder scan**：无 TODO/TBD，每个代码步骤含完整代码。
- **Type consistency**：`get_messages_paged(conv_id, limit, before_seq)`、`{messages, has_more}`、store 的 `oldestSeq/hasMore/loadingOlder/loadOlder`、MessageList props `hasMore/loadingOlder` + emit `load-older` 在各任务间一致。
