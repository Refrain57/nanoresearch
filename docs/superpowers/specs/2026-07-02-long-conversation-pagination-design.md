# 长对话消息分页 — 设计文档

**日期**：2026-07-02
**问题来源**：待修清单 #2「长对话看不到全文 / 够不到最早消息」

## 问题

会话消息超过一屏加载量后，前端只拿到**最旧的 N 条**，且没有向上滚动加载更早的机制：

- 后端 `conversation_repo.py:get_messages_paged` 按 `Message.seq` **升序** + `limit/offset` 取数，端点 `chat_router.py` 默认 `limit=50, offset=0`。
- 前端 `chat.js:selectConversation` 调 `getMessages(id, { limit: 100 })`（offset 0）→ 得到**最旧 100 条**，随后 `MessageList.vue` 自动滚到底部。

结果：>100 条的对话里，最近的消息永远不会被拉取（「全文不全」），也没有滚动加载更早的分页。

## 方案：增量游标分页（before-seq cursor）

打开时加载**最近 N 条**（`seq` 倒序取 N，返回前反转为升序），滚动到顶部时以「当前最旧消息的 seq」为游标加载更早的一页，prepend 并保持滚动位置。选择游标而非 offset：`seq` 在会话内单调递增且稳定，会话中途来新消息时不会像 offset 那样漂移导致重复/跳条。

**参数**：初始加载 40 条，向上滚动每页 40 条。

## 后端

### `storage/repositories/conversation_repo.py`

`get_messages_paged` 增加可选 `before_seq` 参数，改为倒序取最近 N 再反转：

```python
async def get_messages_paged(
    self, conv_id, limit=50, before_seq: int | None = None
) -> list[Message]:
    stmt = select(Message).where(Message.conversation_id == conv_id)
    if before_seq is not None:
        stmt = stmt.where(Message.seq < before_seq)
    stmt = stmt.order_by(Message.seq.desc()).limit(limit)
    rows = list((await db.execute(stmt)).scalars().all())
    rows.reverse()          # 升序返回，前端直接 append/prepend
    return rows
```

保留旧签名兼容性由调用方（仅本端点）承担，`offset` 参数移除。

### `server/routers/chat_router.py` — `GET /api/conversations/{conv_id}/messages`

- 查询参数：`limit: int = 40`，`before_seq: int | None = None`（移除 `offset`）。
- **响应形状改为** `{ "messages": [...], "has_more": bool }`。
- `has_more` 用**原始行数**判定（`len(raw_rows) == limit`），在过滤 internal 编排消息**之前**计算。原因：internal 消息的过滤发生在取数之后，若用过滤后的条数判断，边界页被过滤掉几条就会误判「没有更早了」。
- internal 过滤逻辑不变（`content.internal` 的消息不进 `messages` 数组）。

该端点当前唯一调用方是 `chat.js`，改响应形状是内聚的。

## 前端

### `apis/conversations.js`

`getMessages(id, params)` 返回体从裸数组变为 `{ messages, has_more }`，调用方相应取 `.messages`。

### `stores/chat.js`

新增内部状态：`oldestSeq`（当前已加载的最旧 seq）、`hasMore`（是否还有更早）、`loadingOlder`（并发护栏）。

- **保留一份未合并的原始累积数组**（`_rawLoaded`，映射过但未做 tool-call 合并，按 seq 升序）。
- `selectConversation` → 改用 `loadMessages(id, { limit: 40 })`（不带 before_seq）：拿最近 40 → map → 存入 `_rawLoaded` → 整体重算合并 → 赋给 `messages`；记录 `oldestSeq`、`hasMore`。
- 新增 `loadOlder()`：`before_seq = oldestSeq` 拉一页 → map → **prepend 到 `_rawLoaded`** → 对整个 `_rawLoaded` **重跑 tool-call 合并** → 更新 `messages`、`oldestSeq`、`hasMore`。

**为什么每次加页都对整表重算合并**：`chat.js` 现有逻辑会把「只有 tool_calls 的助手消息」与**紧跟其后的文本助手消息**合并（`chat.js:52-72`），这对相邻顺序敏感。分页边界可能把这一对劈开，只有保留未合并的原始累积、每次整体重算，才不会丢 `tool_calls`。

### `components/MessageList.vue`

滚动容器（`listRef`）在此组件。

- **触顶加载**：`@scroll` 中当 `scrollTop < ~80px` 且 `hasMore` 且非 `loadingOlder` → 触发 `loadOlder`。
- **位置保持**：prepend 前记录 `prevScrollHeight`；`nextTick` 后 `scrollTop += (scrollHeight - prevScrollHeight)`，视口锚定在原消息，不跳动。
- **底部自动滚**：现有 `watch(messages.length)` 无条件滚到底（`MessageList.vue:96-98`）会和 prepend 打架。改为仅在「首次加载 / 新消息追加到底部 / 流式」时滚到底；prepend 期间用 `preservingScroll` 标志跳过底部滚动。

## 边界情况

- **internal 编排消息**：`seq` 全局连续、过滤在取数之后，游标 `before_seq = 最旧可见消息 seq` 不会跳段（被过滤的 internal 消息 seq 仍落在覆盖区间内）。`has_more` 用原始行数判定。
- **流式消息**：重算合并时保留尾部未落库的 `stream-*` 消息，避免 `loadOlder` 时把正在流式输出的消息从渲染列表挤掉。
- **短对话（≤40 条）**：首次加载即 `has_more=false`，不触发 `loadOlder`，行为与现状一致。

## 测试

- **后端**（`conversation_repo` + 端点）：
  - `before_seq=None` 返回最近 N 条且升序。
  - `before_seq=k` 只返回 `seq < k` 的最近 N 条。
  - 消息数恰为 limit 时 `has_more=true`；不足 limit 时 `false`。
  - 边界页含 internal 消息时，`has_more` 仍按原始行数正确判定。
- **前端 store**：`loadOlder` prepend 后 `messages` 顺序正确、tool-call 合并不丢、`oldestSeq` 更新。
- **滚动行为**：手动验证触顶加载 + 位置保持不跳动（`/verify` 或运行 app 目视）。

## 规模

后端小改（1 查询 + 1 端点参数/响应形状）；前端中改（store 分页状态 + MessageList 滚动逻辑）。无 DB schema 变更。
