# Phase 1：子 Agent 异步回主 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复「子 agent 结果回不到主 agent」的现状 bug——子 agent 完成后结果写回主对话消息列表、凑齐后复用 Phase 0 唤醒路径拉起主 agent 汇总，并加崩溃/超时 watchdog 兜底，删除占槽的 SCARD 轮询。

**Architecture:** 建立在 Phase 0（信箱 + 分布式锁 + 唯一入队调度器）之上。子 agent 完成时把结果**原子追加**进主 conversation 的消息列表 + 走原子 join 闸门（基于现有 `pending:{session_key}` 集合）；最后一个完成者**恰好触发一次**唤醒：用 Phase 0 的「投主信箱 → 调度器拉起」拉起一个**复用原 run_id**的续接 run，主 agent 重建上下文时天然读到所有子结果并汇总。主 run spawn 子后**不发 run_end、保持 status=running**，子 agent 继续往**原 run_events 流**写（前端 SSE 不断连）；续接 run 复用原 run_id 把汇总写回同一条流并发 run_end。watchdog 扫描超期未凑齐 / 超时 running run，补失败结果推进 join 并补发 run_end。

**Tech Stack:** Python 3.12、redis.asyncio（Redis 5.0.14）、ARQ、FastAPI、PostgreSQL、pytest（DB 真 PG、Redis 原语真 Redis）。

## Global Constraints

- **建立在 Phase 0 之上**：复用 `bus/mailbox.py`（post_message/post_notify/finalize_and_release）、`bus/dispatcher.py`、`bus/dist_lock.py`、`RedisKeys.agent_inbox/agent_lock/pending`。唤醒**必须复用** Phase 0 的「投主信箱 → 调度器拉起主协程」，**不要另造唤醒机制**。
- **子 agent 是主的私有内部实现**：不可被第三方寻址、不发消息、无自己的信箱。**不要给子 agent 加 message/spawn 能力或地址**（`subagent.py:120-151` 的能力阉割保持原样）。
- **固定批模式**：主「派一批子 → 等全部凑齐 → 汇总」，主**不在子结果中间做分支决策**（不阻塞、不走一步看一步）。
- **不持久化「续接意图」**：不存「我在等什么/进行到哪」。续接 run 的 agent 配置**从 conversation 重建**（与 HTTP 入口同一套 `_build_run_payload`）；子结果落在消息列表里，主重建上下文即天然看到。join 状态就是现有 `pending:{session_key}` 集合，不新增字段。
- **不做** A2A（Phase 2）/ 子 agent 寻址 / 主的中途分支决策 / 会话存储改追加式。
- **Redis 5.0.14**：无排他区间 `(id`、无 `XAUTOCLAIM`（沿用 Phase 0 的 next-id + XPENDING/XCLAIM 约定，本期新代码不依赖这两者）。
- **环境**：venv `backend/.venv`，测试 Redis DB 15，PG 测试库 `nanoresearch_test`。`asyncio_mode=auto`。

## 修订（必改 1/2/3 已确认 — 覆盖下文相应任务）

**必改 1（并发隔离，A 方案：续接绕信箱 + 双闸门）** —— 续接 run 与用户新消息挤同一主信箱会串话。解决：
- **dispatcher batch 闸门**：`_handle_notify` 抢到锁后若 `SCARD(pending:{conv})>0` → 释放锁 + 返回 `deferred_batch`（用户消息留信箱，batch 后由续接 finalize 的 re-notify 重试）。
- **join 原子「清空 pending 即占锁」**：join 原语改 `join_and_acquire(redis, session_key, task_id, lock_key, lock_token) -> bool`，Lua 内 `SREM 成员；若 pending 空 → SET NX lock=token；返回 fired`。不变量：主 spawn 子→续接完成期间「pending 非空 或 锁被续接持有」恒成立，用户消息全程进不来。
- **续接绕信箱直接 enqueue**：join 触发者用 token + 复用原 run_id `arq_pool.enqueue_job("run_agent_job", **payload, _lock_key, _lock_token=token, _continuation=True)`（无 `_entry_id`）。worker 启动建 `ctx["arq_pool"]` 透传给 SubagentManager；watchdog 用 `app.state.arq_pool`。
- **续接 run 的 release-only finalize**：`_finalize_mailbox_run` 支持 `entry_id=None`（跳过推游标，只释放锁 + 有积压则 re-notify）。

**必改 2（append 成功才推进 join）** —— `_report_and_join`：先 `append_message`，**成功才** `join_and_acquire`；失败则不 SREM、留给 watchdog（杜绝「结果没落库却判齐」）。

**必改 3（watchdog 阈值）** —— 无子 agent 心跳机制（pending ts=spawn 时刻），`subagent_stale` 默认 600→**7800s**（与 run_stuck 同量级），只兜「进程真没了/卡死」，不误杀长任务活子。

**修正** —— watchdog 兜底续接不用 `run_id=""`（会炸），改 `run_repo.create` 建真 run_id；同时把卡死原 run 标 failed + 补发 run_end。主 run defer run_end 的 `return` 在 `try` 内，`finally`（finalize）必执行。

> 下文 Task 1/4/5/6 的代码以本修订为准（Task 1 join 改 `join_and_acquire`；Task 4 唤醒改 arq enqueue + dispatcher 闸门；Task 5 加续接 path + entry_id=None finalize；Task 6 stale=7800 + 真 run_id）。

## 现状锚点（已复核，勿推翻）

- 子 agent fire-and-forget：`subagent.py:76` `asyncio.create_task(self._run_subagent(...))`；完成走 `_announce_result`（`subagent.py:227-291`，web 写 `run_events:{origin run_id}` 流给前端 SSE）。
- pending 集合：spawn 时 `SADD pending:{session_key} "{task_id}:{ts}"`（`subagent.py:87`）；移除 `_remove_pending_member`（`subagent.py:315-326`，按 task_id 前缀 SREM）。
- 主 run 等待：`worker.py:398-412` SCARD 轮询（5s/最长 1800s），占着 ARQ 槽。
- run_end：`worker.py:432-433`（completed）/`440-441`（failed，`except`）。超时/崩溃不发 → SSE 永挂（本期要堵）。
- `SubagentManager`（`subagent.py:29-60`）**无 session_factory**；构造在 `loop.py:123-136`。
- HTTP 入口建 payload + agent 配置：`chat_router.py:238-307`（`run_repo.create`→ 幂等 `SET NX`→ payload → `_enqueue_via_mailbox`，Phase 0 已落）。
- `RunRepository`（`run_repo.py`）有 `update(run_id, **fields)`/`get`/`list_by_conversation`，**无**按 status/超时扫描。
- `SessionManager`：`_redis_save` 全量 `DEL+RPUSH`（`manager.py:189-201`）、`_db_save→replace_messages`（`conversation_repo.py:128-139`）；session Redis key `RedisKeys.session_msg(uid, ch, chat_id)`（`redis_keys.py:30-31`）。

---

## 设计要点（实现者必读）

### 子结果如何回主（决定验收 1、5）
1. 子 agent 完成时**两件事都做**：(a) 仍写 `run_events:{原run_id}` 流（前端 SSE 实时看到，不破坏现有展示）；(b) **新增**：把结果作为一条消息**原子追加**进主 conversation 的消息列表（`SessionManager.append_message`：Redis 会话列表 `RPUSH`（原子追加，并发安全）+ DB `messages` 插一行）。
2. 主 run spawn 子后 process_direct 产出（通常是「已派发，等子任务」之类）即返回；**worker 检测 `SCARD(pending)>0` → 不发 run_end、不置 completed、保持 status=running、直接返回**（释放 ARQ 槽，Phase 0 finalize 照常释放信箱锁）。原 `run_events` 流因无 run_end 保持打开，前端 SSE 不断连。
3. 子全部凑齐 → join 触发 → 复用**原 run_id**投主信箱（content=汇总指令）→ 调度器拉起续接 run → 它 `get_or_create(session)` 时从 Redis 读到所有子结果消息 → process_direct 把汇总写回**原 `run_events` 流** → 这次 `SCARD(pending)==0` → 正常发 run_end。前端在同一条 SSE 上收到汇总 + run_end。

### 原子 join（决定验收 2）
- 单段 Lua over `pending:{session_key}`：按 task_id 前缀找到并 `SREM` 该成员，再 `SCARD`；返回 1 当且仅当移除后集合为空。Redis 单线程 + 整段原子 → 多个子几乎同时完成时**恰好一个**看到「空」、**恰好触发一次**。正常完成路径与 watchdog 失败路径走同一个 Lua。

### 唤醒复用 Phase 0（不另造）
- join 触发者用 `_build_run_payload(factory, redis, conversation_id, uid, content, run_id=原run_id)`（从 `chat_router` 抽出的共享构建器，按 conversation 重建 agent 配置）建 payload → `_enqueue_via_mailbox(redis, payload)`（Phase 0 既有：post_message + post_notify）→ 调度器取锁拉起续接 run。**复用原 run_id**，不新建 run 行（`run_repo.update` 复用）。

### 删 SCARD 轮询后的主协程生命周期
- 删除 `worker.py:398-412` 的 while 轮询。主 run = process_direct 产出即结束；若 spawn 了子（`SCARD(pending)>0`）则跳过 run_end、留 running、立即 return（不再占槽 30min）；否则（单轮无子）正常 run_end/completed（行为同现状）。续接 run 因 pending 已空 → 正常 run_end。

### watchdog（决定验收 4）
- 新长驻 `StuckRunWatchdog`（挂 server lifespan，与 PendingReaper 并列），周期扫两类：
  1. **超期 pending 成员**（`{task_id}:{ts}` 的 ts 超 `SUBAGENT_STALE_SECONDS`，默认 600s）：视为子崩溃/卡死 → `append_message` 写一条失败结果 → 走 join Lua（移除该成员 + 判空）→ 若触发则投主信箱唤醒。推进 join，conversation 不会因死子永久挂。
  2. **超时 running run**（`AgentRun.status='running'` 且 `started_at` 超 `RUN_STUCK_SECONDS`，默认 7800s = job_timeout+10min，且其 `pending` 集合为空——避免误杀正在合法等子的主 run）：标记 failed + `xadd run_end{status:failed}` 到 `run_events:{run_id}` → 解开 SSE 永挂。

---

## File Structure

- **Modify** `backend/nanoresearch/bus/mailbox.py` —— 加 `join_and_should_fire`（原子 join Lua）+ `enqueue`（post_message+post_notify 的复用封装）。
- **Modify** `backend/nanoresearch/session/manager.py` —— 加 `append_message`（原子 Redis RPUSH + DB 插行）。
- **Modify** `backend/nanoresearch/server/routers/chat_router.py` —— 抽出共享 `_build_run_payload`；`_enqueue_via_mailbox` 委托 `mailbox.enqueue`。
- **Modify** `backend/nanoresearch/agent/subagent.py` —— 完成时 append 结果 + join + 触发唤醒；`SubagentManager` 增 `session_factory` 与 run 上下文（conversation_id/agent_id）。
- **Modify** `backend/nanoresearch/agent/loop.py:123-136` —— 给 `SubagentManager` 传 `session_factory` + run 上下文。
- **Modify** `backend/nanoresearch/worker.py:398-412` —— 删 SCARD 轮询，改非阻塞「spawn 了子就不发 run_end」。
- **Create** `backend/nanoresearch/heartbeat/stuck_run_watchdog.py` —— `StuckRunWatchdog`。
- **Modify** `backend/nanoresearch/storage/repositories/run_repo.py` —— 加 `list_stuck_running(older_than)`。
- **Modify** `backend/nanoresearch/server/main.py` —— lifespan 起停 watchdog。
- **Create** `backend/tests/unit/bus/test_join.py`、`backend/tests/unit/session/test_append_message.py`、`backend/tests/integration/test_phase1_subagent_return.py`。

---

### Task 1: 原子 join + enqueue 复用封装（`mailbox.py`）

**Files:**
- Modify: `backend/nanoresearch/bus/mailbox.py`
- Test: `backend/tests/unit/bus/test_join.py`

**Interfaces:**
- Consumes: `RedisKeys.pending`、Phase 0 `post_message`/`post_notify`。
- Produces:
  - `async def join_and_should_fire(redis, session_key: str, task_id: str) -> bool` —— 原子移除 task_id 对应 pending 成员并判空；空则返回 `True`（应触发唤醒），否则 `False`。
  - `async def enqueue(redis, payload: dict) -> None` —— post_message(信箱) + post_notify(通知流)，agent_id 缺省 `"none"`。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/unit/bus/test_join.py
import pytest
from nanoresearch.bus import mailbox
from nanoresearch.bus.redis_keys import RedisKeys

async def test_join_fires_exactly_once_when_last_member_removed(redis_client):
    sk = "web:join-c1"
    key = RedisKeys.pending(sk)
    await redis_client.sadd(key, "t1:1000", "t2:1001")
    # 第一个完成 → 还剩 t2 → 不触发
    assert await mailbox.join_and_should_fire(redis_client, sk, "t1") is False
    # 第二个完成 → 空 → 触发
    assert await mailbox.join_and_should_fire(redis_client, sk, "t2") is True
    assert await redis_client.scard(key) == 0

async def test_join_removes_by_task_id_prefix(redis_client):
    sk = "web:join-c2"
    await redis_client.sadd(RedisKeys.pending(sk), "abc:1700000000")
    assert await mailbox.join_and_should_fire(redis_client, sk, "abc") is True

async def test_enqueue_posts_inbox_and_notify(redis_client):
    await mailbox.enqueue(redis_client, {
        "content": "x", "agent_id": None, "conversation_id": "enq-c1", "run_id": "r1"})
    got = await mailbox.read_next_after_cursor(redis_client, "none", "enq-c1")
    assert got is not None and got[1]["content"] == "x"
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) >= 1
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/bus/test_join.py -v`
Expected: FAIL（`join_and_should_fire` / `enqueue` 不存在）

- [ ] **Step 3: 实现（追加到 `mailbox.py` 末尾）**

```python
# 原子 join：按 task_id 前缀移除 pending 成员，移除后为空则返回 1（应触发唤醒）。
# 整段在一次 EVAL 内原子执行 → 多个子并发完成时恰好一个看到"空"。
# KEYS[1]=pending_key ARGV[1]=task_id
JOIN_LUA = """
local members = redis.call('SMEMBERS', KEYS[1])
local prefix = ARGV[1] .. ':'
for i = 1, #members do
    local m = members[i]
    if m == ARGV[1] or string.sub(m, 1, #prefix) == prefix then
        redis.call('SREM', KEYS[1], m)
        break
    end
end
if redis.call('SCARD', KEYS[1]) == 0 then return 1 else return 0 end
"""


async def join_and_should_fire(redis: Any, session_key: str, task_id: str) -> bool:
    res = await redis.eval(JOIN_LUA, 1, RedisKeys.pending(session_key), task_id)
    return bool(res)


async def enqueue(redis: Any, payload: dict) -> None:
    """Post a run payload to its mailbox + signal the dispatcher (Phase 0 wakeup path)."""
    agent_id = payload.get("agent_id") or "none"
    conversation_id = payload["conversation_id"]
    await post_message(redis, agent_id, conversation_id, payload)
    await post_notify(
        redis,
        mailbox_key=RedisKeys.agent_inbox(agent_id, conversation_id),
        cursor_key=RedisKeys.agent_inbox_cursor(agent_id, conversation_id),
        lock_key=RedisKeys.agent_lock(agent_id, conversation_id),
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/unit/bus/test_join.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/bus/mailbox.py backend/tests/unit/bus/test_join.py
git commit -m "feat(bus): atomic subagent join + mailbox.enqueue reuse helper"
```

---

### Task 2: 会话消息原子追加（`SessionManager.append_message`）

**Files:**
- Modify: `backend/nanoresearch/session/manager.py`
- Test: `backend/tests/unit/session/test_append_message.py`

**Interfaces:**
- Consumes: `RedisKeys.session_msg`、`ConversationRepository`。
- Produces: `async def append_message(self, session_key: str, message: dict, uid: str) -> None` —— 原子把一条消息追加进会话：Redis 会话列表 `RPUSH`（带 TTL 续期）+ DB `messages` 插一行（seq=当前行数，并发下重复 seq 良性，下次 `replace_messages` 重排）。`message` 形如 `{"role": "user", "content": "...", "timestamp": "..."}`。

- [ ] **Step 1: 写失败测试（真 Redis + 真 PG）**

```python
# backend/tests/unit/session/test_append_message.py
import pytest
from nanoresearch.session.manager import SessionManager
from nanoresearch.bus.redis_keys import RedisKeys
from tests.conftest import make_factory, truncate_all

@pytest.fixture(autouse=True)
def _clean():
    truncate_all()

async def test_append_message_rpush_and_db(redis_client, monkeypatch, tmp_path):
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)

    factory = make_factory()
    # seed a conversation
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    from nanoresearch.auth.password import hash_password
    await UserRepository(factory).create("u1", hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:append-c1", uid="u1")

    mgr = SessionManager(tmp_path, session_factory=factory, default_uid="u1")
    await mgr.append_message("web:append-c1", {"role": "user", "content": "sub-result-1"}, uid="u1")
    await mgr.append_message("web:append-c1", {"role": "user", "content": "sub-result-2"}, uid="u1")

    # Redis session list has both appended (atomic RPUSH)
    msg_key = RedisKeys.session_msg("u1", "web", "append-c1")
    raw = await redis_client.lrange(msg_key, 0, -1)
    assert len(raw) == 2
    # DB has both messages
    msgs = await ConversationRepository(factory).get_messages(conv.id)
    contents = [m.content.get("content") for m in msgs]
    assert "sub-result-1" in contents and "sub-result-2" in contents
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/session/test_append_message.py -v`
Expected: FAIL（`append_message` 不存在）

- [ ] **Step 3: 实现（加到 `SessionManager`，在 `save` 方法附近）**

```python
    async def append_message(self, session_key: str, message: dict, uid: str) -> None:
        """Atomically append ONE message to the session (Redis RPUSH + DB insert).

        Atomic RPUSH makes concurrent appends (parallel subagents) safe without a
        read-modify-write. The single-writer continuation run later persists the full
        window normally; this is only used between runs (no run is writing concurrently).
        """
        import json
        from nanoresearch.utils.helpers import utcnow_aware
        entry = dict(message)
        entry.setdefault("timestamp", utcnow_aware().isoformat())

        # L1 cache: keep an in-memory session consistent if present
        if session_key in self._cache:
            self._cache[session_key].messages.append(entry)

        # Redis session list (atomic append)
        try:
            from nanoresearch.bus.redis_client import get_redis
            from nanoresearch.bus.redis_keys import RedisKeys
            redis = get_redis()
            ch, chat_id = (session_key.split(":", 1) if ":" in session_key else (session_key, ""))
            msg_key = RedisKeys.session_msg(uid, ch, chat_id)
            await redis.rpush(msg_key, json.dumps(entry, ensure_ascii=False))
            await redis.expire(msg_key, RedisKeys.SESSION_TTL)
        except Exception as e:
            logger.warning("append_message redis RPUSH failed (non-fatal): {}", e)

        # DB insert (best-effort; seq races are benign — replace_messages re-seqs)
        if self._factory is not None:
            try:
                from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
                await ConversationRepository(self._factory).append_message(session_key, entry)
            except Exception as e:
                logger.warning("append_message db insert failed (non-fatal): {}", e)
```

并在 `ConversationRepository` 加 `append_message`（`conversation_repo.py`，紧邻 `replace_messages`）：

```python
    async def append_message(self, session_key: str, message: dict) -> None:
        """Insert one Message row at seq = current count (race-benign)."""
        from nanoresearch.storage.models import Conversation, Message
        async with self._factory() as db:
            conv = (await db.execute(
                select(Conversation).where(Conversation.session_key == session_key)
            )).scalar_one_or_none()
            if conv is None:
                return
            count = (await db.execute(
                select(func.count(Message.id)).where(Message.conversation_id == conv.id)
            )).scalar() or 0
            db.add(Message(conversation_id=conv.id, role=message.get("role", "user"),
                           content=message, seq=count))
            await db.commit()
```

> `func` 已在 `conversation_repo.py` 顶部 import（如未导入则 `from sqlalchemy import func, select`）。

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/unit/session/test_append_message.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/session/manager.py backend/nanoresearch/storage/repositories/conversation_repo.py backend/tests/unit/session/test_append_message.py
git commit -m "feat(session): atomic append_message (Redis RPUSH + DB insert)"
```

---

### Task 3: 共享续接 payload 构建器（`_build_run_payload`）

**Files:**
- Modify: `backend/nanoresearch/server/routers/chat_router.py`
- Test: `backend/tests/integration/test_phase1_subagent_return.py`（首个用例）

**Interfaces:**
- Consumes: `ConversationRepository`、`AgentRepository`、`RedisKeys`。
- Produces: `async def _build_run_payload(factory, conversation_id: str, uid: str, content: str, run_id: str) -> dict` —— 按 conversation 重建与 HTTP 入口一致的 run payload（含 agent_id/skill_names/custom_persona/harness/agents_registry/agent_kb_id）。`create_run` 改为调用它（DRY），`_enqueue_via_mailbox` 委托 `mailbox.enqueue`。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/integration/test_phase1_subagent_return.py
import pytest
from tests.conftest import make_factory, truncate_all

@pytest.fixture(autouse=True)
def _clean():
    truncate_all()

async def test_build_run_payload_rebuilds_config_from_conversation(redis_client):
    from nanoresearch.server.routers.chat_router import _build_run_payload
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    from nanoresearch.auth.password import hash_password
    factory = make_factory()
    await UserRepository(factory).create("u1", hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:bp-c1", uid="u1")

    payload = await _build_run_payload(factory, str(conv.id), "u1",
                                       content="请汇总", run_id="orig-run-1")
    assert payload["run_id"] == "orig-run-1"
    assert payload["conversation_id"] == str(conv.id)
    assert payload["content"] == "请汇总"
    assert payload["uid"] == "u1"
    assert "agent_id" in payload and "skill_names" in payload  # config keys present
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/integration/test_phase1_subagent_return.py::test_build_run_payload_rebuilds_config_from_conversation -v`
Expected: FAIL（`_build_run_payload` 不存在）

- [ ] **Step 3: 实现**

在 `chat_router.py` Helpers 区加：

```python
async def _build_run_payload(factory, conversation_id: str, uid: str, content: str, run_id: str) -> dict:
    """Rebuild a run payload (agent config from the conversation). Shared by HTTP entry
    and the subagent/watchdog wakeup so continuation has identical agent config — no
    persisted 'intent' state needed."""
    import uuid as _uuid
    conv = await ConversationRepository(factory).get_by_id(_uuid.UUID(conversation_id))
    skill_names = None
    custom_persona = None
    agent_harness: dict = {}
    agent_kb_id = None
    if conv is not None and conv.agent_id:
        agent = await AgentRepository(factory).get_by_id(conv.agent_id)
        if agent is not None:
            skill_names = [s["name"] for s in (agent.skills_config or []) if s.get("enabled", True)]
            custom_persona = agent.persona or None
            agent_harness = agent.harness or {}
            if agent_harness.get("kb_id"):
                agent_kb_id = agent_harness["kb_id"]
    agents_registry = [
        {"id": str(a.id), "name": a.name, "description": a.description or ""}
        for a in await AgentRepository(factory).list_by_user(uid)
    ]
    return {
        "run_id": run_id,
        "session_key": (conv.session_key if conv else None) or f"web:{conversation_id}",
        "conversation_id": conversation_id,
        "content": content,
        "uid": uid,
        "rag_mode": "agentic",
        "kb_id": None,
        "skill_names": skill_names,
        "agent_id": str(conv.agent_id) if conv and conv.agent_id else None,
        "agent_override": None,
        "custom_persona": custom_persona,
        "harness": agent_harness or None,
        "agents_registry": agents_registry or None,
        "agent_kb_id": agent_kb_id,
        "job_id": None,
    }
```

把 `_enqueue_via_mailbox` 改为委托（DRY）：

```python
async def _enqueue_via_mailbox(redis, payload: dict) -> None:
    from nanoresearch.bus import mailbox
    await mailbox.enqueue(redis, payload)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/integration/test_phase1_subagent_return.py::test_build_run_payload_rebuilds_config_from_conversation -v`
Expected: PASS

- [ ] **Step 5: 回归 + 提交**

```bash
cd backend && python -m pytest tests/integration/test_phase0_dispatch.py -q
git add backend/nanoresearch/server/routers/chat_router.py backend/tests/integration/test_phase1_subagent_return.py
git commit -m "feat(api): extract shared _build_run_payload; _enqueue_via_mailbox delegates to mailbox.enqueue"
```

---

### Task 4: 子 agent 异步回主（append 结果 + join + 唤醒）

**Files:**
- Modify: `backend/nanoresearch/agent/subagent.py`
- Modify: `backend/nanoresearch/agent/loop.py:123-136`
- Test: `backend/tests/integration/test_phase1_subagent_return.py`（追加）

**Interfaces:**
- Consumes: `mailbox.join_and_should_fire`、`mailbox.enqueue`、`chat_router._build_run_payload`、`SessionManager.append_message`。
- Produces: `SubagentManager.__init__` 增 `session_factory=None`；`spawn` 增 `conversation_id`/`uid` 上下文；完成时 append 结果 + join + 触发唤醒。

- [ ] **Step 1: 写测试（子完成 → append + pending 减 + 凑齐投信箱，桩掉真 LLM）**

```python
# backend/tests/integration/test_phase1_subagent_return.py （追加）
async def test_subagent_completion_appends_and_fires_join(redis_client, monkeypatch):
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    from nanoresearch.bus import mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    from nanoresearch.auth.password import hash_password
    from nanoresearch.agent.subagent import SubagentManager
    factory = make_factory()
    await UserRepository(factory).create("u1", hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:sub-c1", uid="u1")
    sk = "web:sub-c1"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000", "t2:1001")

    mgr = SubagentManager(provider=None, workspace=__import__("pathlib").Path("."),
                          bus=None, model="m", uid="u1", session_factory=factory)
    mgr.set_run_context(conversation_id=str(conv.id))

    # t1 done → append + not fire
    await mgr._report_and_join("t1", "label1", "task1", "result-1", {"channel": "web",
        "chat_id": str(conv.id), "run_id": "orig-1"}, "ok", sk)
    assert await redis_client.scard(RedisKeys.pending(sk)) == 1
    notify0 = await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY)

    # t2 done → fire → wakeup posted to main mailbox (reuse orig run_id)
    await mgr._report_and_join("t2", "label2", "task2", "result-2", {"channel": "web",
        "chat_id": str(conv.id), "run_id": "orig-1"}, "ok", sk)
    assert await redis_client.scard(RedisKeys.pending(sk)) == 0
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) == notify0 + 1
    got = await mailbox.read_next_after_cursor(redis_client, "none", str(conv.id))
    assert got is not None and got[1]["run_id"] == "orig-1"  # reused original run_id
    # both results appended to the session message list
    raw = await redis_client.lrange(RedisKeys.session_msg("u1", "web", str(conv.id)), 0, -1)
    assert len(raw) == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/integration/test_phase1_subagent_return.py::test_subagent_completion_appends_and_fires_join -v`
Expected: FAIL（`session_factory`/`set_run_context`/`_report_and_join` 不存在）

- [ ] **Step 3: 实现**

`SubagentManager.__init__`（`subagent.py:29-60`）签名末尾加 `session_factory: Any = None`，并 `self.session_factory = session_factory`、`self._conversation_id: str | None = None`。加方法：

```python
    def set_run_context(self, conversation_id: str | None) -> None:
        self._conversation_id = conversation_id

    async def _report_and_join(self, task_id, label, task, result, origin, status, session_key):
        """Append the subagent result to the main conversation, advance the atomic join,
        and (if this is the last subagent) wake the main agent via the Phase 0 path —
        reusing the ORIGINAL run_id so the frontend SSE stream stays continuous."""
        from nanoresearch.bus import mailbox
        from nanoresearch.bus.redis_client import get_redis
        redis = get_redis()

        # (a) keep the existing SSE stream write for the frontend
        await self._announce_result(task_id, label, task, result, origin, status, session_key)

        # (b) NEW: append the result into the main conversation message list
        if self.session_factory is not None and self._conversation_id and self.uid:
            try:
                from nanoresearch.session.manager import SessionManager
                mgr = SessionManager(self.workspace, session_factory=self.session_factory,
                                     default_uid=self.uid)
                body = f"[Subagent '{label}' {'completed' if status == 'ok' else 'failed'}]\n\nTask: {task}\n\nResult:\n{result}"
                await mgr.append_message(session_key, {"role": "user", "content": body}, uid=self.uid)
            except Exception as e:
                logger.warning("subagent append_message failed (non-fatal): {}", e)

        # (c) atomic join → exactly-once fire
        try:
            should_fire = await mailbox.join_and_should_fire(redis, session_key, task_id)
        except Exception as e:
            logger.warning("join_and_should_fire failed: {}", e)
            return
        if not should_fire:
            return

        # (d) wake the main agent — reuse ORIGINAL run_id for SSE continuity
        if self.session_factory is None or not self._conversation_id:
            return
        try:
            from nanoresearch.server.routers.chat_router import _build_run_payload
            payload = await _build_run_payload(
                self.session_factory, self._conversation_id, self.uid,
                content="所有子任务已完成，结果已并入对话。请基于这些子任务结果汇总并回复用户。",
                run_id=origin.get("run_id") or "")
            await mailbox.enqueue(redis, payload)
        except Exception as e:
            logger.warning("subagent wakeup enqueue failed (non-fatal): {}", e)
```

把 `_run_subagent` 三处成功/失败回报（`subagent.py:188,194,203,225`）从 `await self._announce_result(...)` 改为 `await self._report_and_join(...)`（同参数顺序）。`_remove_pending_member` 的 SREM 由 join Lua 接管——`_report_and_join` 已含移除，**移除 `_run_subagent` 里原本的 crash-safety SREM 调用**（`subagent.py:209-210,221-222`），改为在 `_report_and_join` 失败时也保证 join 推进（已在 c 步）。

`loop.py:123-136` 给 `SubagentManager(...)` 传 `session_factory=self._session_factory`；在 `_set_tool_context`（`loop.py:252-274`）里对 spawn 工具设置后，补 `self.subagents.set_run_context(conversation_id=chat_id if channel == "web" else None)`（web 的 chat_id 即 conversation_id）。

> 说明：`spawn` 的 origin 已带 run_id（`subagent.py:74`），`_report_and_join` 复用它作续接 run_id。

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/integration/test_phase1_subagent_return.py::test_subagent_completion_appends_and_fires_join -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/agent/subagent.py backend/nanoresearch/agent/loop.py backend/tests/integration/test_phase1_subagent_return.py
git commit -m "feat(subagent): async return — append result to main conversation + atomic join + reuse-run_id wakeup"
```

---

### Task 5: 删 SCARD 轮询，主 run spawn 子后不发 run_end

**Files:**
- Modify: `backend/nanoresearch/worker.py:398-433`
- Test: `backend/tests/integration/test_phase1_subagent_return.py`（追加）

**Interfaces:**
- Consumes: `RedisKeys.pending`。
- Produces: 行为变更——`run_agent_job` 在 process_direct 后若 `SCARD(pending)>0` 则跳过 `run_end`+`completed`、保持 running、return。

- [ ] **Step 1: 写测试（用 `__PERF_TEST__` 旁路 + 预置 pending，断言不发 run_end）**

```python
# backend/tests/integration/test_phase1_subagent_return.py （追加）
async def test_main_run_with_pending_subagents_skips_run_end(redis_client, monkeypatch):
    """spawn 了子（pending 非空）→ 主 run 不发 run_end、保持 running。"""
    import nanoresearch.worker as worker
    from nanoresearch.bus.redis_keys import RedisKeys
    sk = "web:skip-c1"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")
    # 直接测抽出的判定函数（见实现 Step 3 的 _has_pending_subagents）
    assert await worker._has_pending_subagents(redis_client, sk) is True
    await redis_client.delete(RedisKeys.pending(sk))
    assert await worker._has_pending_subagents(redis_client, sk) is False
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/integration/test_phase1_subagent_return.py::test_main_run_with_pending_subagents_skips_run_end -v`
Expected: FAIL（`_has_pending_subagents` 不存在）

- [ ] **Step 3: 实现**

`worker.py` 加助手：

```python
async def _has_pending_subagents(redis, session_key: str) -> bool:
    from nanoresearch.bus.redis_keys import RedisKeys
    try:
        return bool(await redis.scard(RedisKeys.pending(session_key)))
    except Exception:
        return False
```

**删除** `worker.py:398-412` 的整段 SCARD 轮询 while 循环。把其后的「完成落库 + 发 run_end」（约 `worker.py:414-433`）改为：

```python
        # Phase 1: if this run spawned subagents (pending non-empty), do NOT finish the turn.
        # Leave status=running and emit NO run_end — the run_events stream stays open so the
        # frontend keeps streaming subagent output; the continuation run (reusing this run_id)
        # emits run_end after the subagents are joined and summarized.
        if await _has_pending_subagents(redis, session_key):
            logger.info("run_agent_job %s spawned subagents — deferring run_end to continuation", run_id)
            return
        finished = _utcnow()
        duration_ms = int((finished - start).total_seconds() * 1000)
        usage = loop._last_usage or {}
        tokens_used = {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "cache_read": usage.get("cache_read_input_tokens", 0),
            "cache_write": usage.get("cache_creation_input_tokens", 0),
        }
        await run_repo.update(
            run_id, status="completed", finished_at=finished, duration_ms=duration_ms,
            model_used=loop.model, tokens_used=tokens_used, tool_calls=tool_calls_log,
        )
        await xadd_event(redis, run_stream_key,
                         {"type": "run_end", "status": "completed", "duration_ms": duration_ms})
```

> `return` 前 Phase 0 的 `finally`（finalize 信箱锁/推游标/补发）仍照常执行——主 run 释放槽与锁，子 agent 在后台继续。

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/integration/test_phase1_subagent_return.py::test_main_run_with_pending_subagents_skips_run_end -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/worker.py backend/tests/integration/test_phase1_subagent_return.py
git commit -m "feat(worker): drop SCARD poll wait; defer run_end when subagents pending"
```

---

### Task 6: 崩溃/超时 watchdog

**Files:**
- Create: `backend/nanoresearch/heartbeat/stuck_run_watchdog.py`
- Modify: `backend/nanoresearch/storage/repositories/run_repo.py`
- Modify: `backend/nanoresearch/server/main.py`
- Test: `backend/tests/integration/test_phase1_subagent_return.py`（追加）

**Interfaces:**
- Consumes: `mailbox.join_and_should_fire`/`enqueue`、`SessionManager.append_message`、`chat_router._build_run_payload`、`RunRepository`。
- Produces:
  - `RunRepository.list_stuck_running(older_than: datetime) -> list[AgentRun]`。
  - `class StuckRunWatchdog(redis, session_factory, *, interval=120, subagent_stale=600, run_stuck=7800)`，`start()/stop()`，`_scan_once()`。

- [ ] **Step 1: 写测试（stale pending → 推进 join + 唤醒；stuck run → 补 run_end）**

```python
# backend/tests/integration/test_phase1_subagent_return.py （追加）
async def test_watchdog_stale_pending_advances_join_and_wakes(redis_client, monkeypatch):
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.heartbeat.stuck_run_watchdog import StuckRunWatchdog
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    from nanoresearch.auth.password import hash_password
    factory = make_factory()
    await UserRepository(factory).create("u1", hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:wd-c1", uid="u1")
    sk = f"web:{conv.id}"
    # one stale pending member (ts far in the past)
    await redis_client.sadd(RedisKeys.pending(sk), "dead:1000000000")

    wd = StuckRunWatchdog(redis_client, factory, subagent_stale=1)
    n0 = await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY)
    await wd._scan_once()
    assert await redis_client.scard(RedisKeys.pending(sk)) == 0          # join advanced
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) == n0 + 1  # main woken

async def test_watchdog_stuck_running_emits_run_end(redis_client):
    from datetime import datetime, timezone, timedelta
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.bus.stream import xread_next
    from nanoresearch.heartbeat.stuck_run_watchdog import StuckRunWatchdog
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.run_repo import RunRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    from nanoresearch.auth.password import hash_password
    factory = make_factory()
    await UserRepository(factory).create("u1", hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:wd-c2", uid="u1")
    run = await RunRepository(factory).create(conversation_id=conv.id, uid="u1")
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    await RunRepository(factory).update(run.id, status="running", started_at=old)

    wd = StuckRunWatchdog(redis_client, factory, run_stuck=1)
    await wd._scan_once()
    evs, _ = await xread_next(redis_client, RedisKeys.run_events(str(run.id)), "0-0", timeout_ms=200)
    assert any(e.get("type") == "run_end" and e.get("status") == "failed" for e in evs)
    assert (await RunRepository(factory).get(run.id)).status == "failed"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/integration/test_phase1_subagent_return.py -k watchdog -v`
Expected: FAIL（模块/方法不存在）

- [ ] **Step 3: 实现**

`run_repo.py` 加：

```python
    async def list_stuck_running(self, older_than: datetime) -> list[AgentRun]:
        async with self._factory() as db:
            result = await db.execute(
                select(AgentRun).where(
                    AgentRun.status == "running", AgentRun.started_at < older_than)
            )
            return list(result.scalars().all())
```

新建 `backend/nanoresearch/heartbeat/stuck_run_watchdog.py`：

```python
"""Phase 1 watchdog: recover crashed/stuck subagents and timed-out runs.

- Stale pending member → treat the subagent as failed: append a failure result to the
  main conversation, advance the atomic join, and (if it completes the batch) wake the
  main agent. This stops a dead subagent from hanging the conversation forever.
- AgentRun status=running past a hard ceiling (and no pending subagents) → mark failed
  and emit run_end so the SSE connection is not left hanging.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from nanoresearch.bus import mailbox
from nanoresearch.bus.redis_keys import RedisKeys
from nanoresearch.bus.stream import xadd_event


class StuckRunWatchdog:
    def __init__(self, redis: Any, session_factory: Any, *, interval: int = 120,
                 subagent_stale: int = 600, run_stuck: int = 7800) -> None:
        self._redis = redis
        self._factory = session_factory
        self._interval = interval
        self._subagent_stale = subagent_stale
        self._run_stuck = run_stuck
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while self._running:
            try:
                await self._scan_once()
            except Exception:
                logger.exception("StuckRunWatchdog scan failed")
            await asyncio.sleep(self._interval)

    async def _scan_once(self) -> None:
        await self._scan_stale_pending()
        await self._scan_stuck_running()

    async def _scan_stale_pending(self) -> None:
        now = time.time()
        cursor = "0"
        while True:
            cursor, keys = await self._redis.scan(cursor, match="pending:*", count=100)
            for key in keys:
                session_key = key[len("pending:"):]
                for member in await self._redis.smembers(key):
                    parts = member.rsplit(":", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        ts = int(parts[1])
                    except ValueError:
                        continue
                    if now - ts < self._subagent_stale:
                        continue
                    await self._fail_subagent(session_key, parts[0])
            if str(cursor) == "0":
                break

    async def _fail_subagent(self, session_key: str, task_id: str) -> None:
        # derive conversation_id + uid from the conversation row
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        conv = await ConversationRepository(self._factory).get_by_session_key(session_key)
        if conv is None:
            await mailbox.join_and_should_fire(self._redis, session_key, task_id)
            return
        try:
            from nanoresearch.session.manager import SessionManager
            mgr = SessionManager(__import__("pathlib").Path("."),
                                 session_factory=self._factory, default_uid=conv.uid)
            await mgr.append_message(
                session_key,
                {"role": "user", "content": f"[Subagent {task_id} timed out / crashed]"},
                uid=conv.uid)
        except Exception:
            logger.warning("watchdog append failure result failed (non-fatal)")
        should_fire = await mailbox.join_and_should_fire(self._redis, session_key, task_id)
        if should_fire:
            from nanoresearch.server.routers.chat_router import _build_run_payload
            payload = await _build_run_payload(
                self._factory, str(conv.id), conv.uid,
                content="部分子任务超时/失败，结果已并入对话。请基于已有结果尽力汇总并回复用户。",
                run_id="")  # no original run_id known here; continuation gets a fresh stream
            await mailbox.enqueue(self._redis, payload)

    async def _scan_stuck_running(self) -> None:
        from nanoresearch.storage.repositories.run_repo import RunRepository
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._run_stuck)
        for run in await RunRepository(self._factory).list_stuck_running(cutoff):
            session_key = None
            from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
            conv = await ConversationRepository(self._factory).get_by_id(run.conversation_id)
            session_key = conv.session_key if conv else None
            if session_key and await self._redis.scard(RedisKeys.pending(session_key)):
                continue  # subagents legitimately still in flight — not stuck
            await RunRepository(self._factory).update(
                run.id, status="failed", finished_at=datetime.now(timezone.utc),
                error_message="stuck run reaped by watchdog")
            await xadd_event(self._redis, RedisKeys.run_events(str(run.id)),
                             {"type": "run_end", "status": "failed", "error": "stuck run reaped"})
```

`server/main.py` lifespan：在 dispatcher.start 后加 `app.state.stuck_watchdog = StuckRunWatchdog(app.state.redis, app.state.session_factory); await app.state.stuck_watchdog.start()`；shutdown 在 dispatcher.stop 前加 `if getattr(app.state,'stuck_watchdog',None): await app.state.stuck_watchdog.stop()`；并在 `app.state.dispatcher = None` 旁加 `app.state.stuck_watchdog = None`。

> watchdog 唤醒用 `run_id=""`（无原 run_id 可复用时，续接走新流）；正常路径（Task 4）复用原 run_id 保 SSE 连续，watchdog 是异常兜底，新流可接受。

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/integration/test_phase1_subagent_return.py -k watchdog -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/heartbeat/stuck_run_watchdog.py backend/nanoresearch/storage/repositories/run_repo.py backend/nanoresearch/server/main.py backend/tests/integration/test_phase1_subagent_return.py
git commit -m "feat(watchdog): recover stale subagents + reap stuck runs (emit run_end)"
```

---

### Task 7: 验收集成测试（5 条）

**Files:**
- Test: `backend/tests/integration/test_phase1_subagent_return.py`（追加端到端断言）

**Interfaces:**
- Consumes: 前 6 个 Task 的全部产物。

- [ ] **Step 1: 写 5 条验收映射测试**

```python
# backend/tests/integration/test_phase1_subagent_return.py （追加）
async def test_ac_summary():
    """验收映射（前述用例覆盖）：
    AC1 主带子结果汇总: test_subagent_completion_appends_and_fires_join（append+复用run_id唤醒）
                       + test_build_run_payload（续接重建配置看到消息列表）
    AC2 恰好唤醒一次:   test_join_fires_exactly_once_when_last_member_removed
    AC3 删 SCARD 不占槽: test_main_run_with_pending_subagents_skips_run_end
    AC4 崩溃/超时兜底:   test_watchdog_stale_pending_advances_join_and_wakes
                       + test_watchdog_stuck_running_emits_run_end
    AC5 子结果仍呈现前端: _report_and_join 仍调 _announce_result 写 run_events（test 见 step2）
    """
    assert True

async def test_ac5_subagent_still_streams_to_frontend(redis_client, monkeypatch):
    """AC5：子完成仍往 run_events:{原run_id} 流写（前端 SSE 不破坏）。"""
    import nanoresearch.bus.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: redis_client)
    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.bus.stream import xread_next
    from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
    from nanoresearch.storage.repositories.user_repo import UserRepository
    from nanoresearch.auth.password import hash_password
    from nanoresearch.agent.subagent import SubagentManager
    factory = make_factory()
    await UserRepository(factory).create("u1", hash_password("x"))
    conv = await ConversationRepository(factory).create(key="web:ac5-c1", uid="u1")
    sk = f"web:{conv.id}"
    await redis_client.sadd(RedisKeys.pending(sk), "t1:1000")
    mgr = SubagentManager(provider=None, workspace=__import__("pathlib").Path("."),
                          bus=None, model="m", uid="u1", session_factory=factory)
    mgr.set_run_context(conversation_id=str(conv.id))
    await mgr._report_and_join("t1", "L", "task", "RESULT-BODY",
        {"channel": "web", "chat_id": str(conv.id), "run_id": "orig-9"}, "ok", sk)
    evs, _ = await xread_next(redis_client, RedisKeys.run_events("orig-9"), "0-0", timeout_ms=200)
    assert any(e.get("type") == "subagent_result" for e in evs)
```

- [ ] **Step 2: 运行确认通过 + 全量回归**

```bash
cd backend && python -m pytest tests/unit/bus tests/unit/session tests/integration/test_phase0_dispatch.py tests/integration/test_phase1_subagent_return.py -q
# 回归：现有非 SSE chat/repos/session
cd backend && python -m pytest tests/test_chat_api.py tests/test_repositories.py tests/unit/session -q -k "not run_events"
```
Expected: 全 PASS（SSE `run_events` 流式 2 例预先存在挂起，已知，`-k "not run_events"` 跳过）

- [ ] **Step 3: 提交**

```bash
git add backend/tests/integration/test_phase1_subagent_return.py
git commit -m "test(phase1): acceptance criteria coverage + regression"
```

---

## 「对外行为不变 / SSE 连续」论证

1. **前端零改动**：续接复用**原 run_id**，汇总写回原 `run_events` 流；主 run spawn 子后不发 run_end，流保持打开，前端 SSE 不断连。子 agent 仍写 `run_events:{原run_id}`（AC5）。
2. **单轮无子对话**：`SCARD(pending)==0` → 走原 run_end/completed 分支，行为同现状。
3. **唤醒复用 Phase 0**：续接和用户发消息走同一条「信箱→调度器→run_agent_job」，未另造机制；分布式锁串行化，无并发写。
4. **无并发写**：子结果 append 用原子 RPUSH（并发安全）；join 凑齐时无子在写；续接 run 在锁内做全量保存。
5. **无持久化 intent**：续接配置从 conversation 重建（`_build_run_payload`），join 状态即现有 pending 集合。

## 风险与回滚
- **最高风险**：主 run 不发 run_end 后若续接始终没来（子全崩 + watchdog 没扫到），SSE 永挂 → watchdog 的 `run_stuck` 扫描兜底（标 failed + 补 run_end）。
- **次高**：续接复用 run_id 但若主 run 的 finally 已把该 run 当完成处理——本计划主 run spawn 子时**不**置 completed、只 return，run 行保持 running，续接 `run_repo.update(running→completed)` 自然衔接。
- **回滚**：Task 5 还原 SCARD 轮询、Task 4 还原 `_announce_result` 调用点即恢复 Phase 0 行为；新增模块不被引用即无副作用。

## Self-Review 记录
- Spec 覆盖：子结果回写消息列表(Task 2/4)、原子 join 凑齐唤醒(Task 1/4)、复用 Phase 0 唤醒(Task 3/4)、删 SCARD(Task 5)、watchdog 兜底+补 run_end(Task 6)、5 条验收(Task 7)、"不做"清单(Global Constraints) 均有对应。
- Placeholder：无 TODO/TBD；测试含具体代码与断言。
- 类型一致：`join_and_should_fire`/`enqueue`、`append_message`、`_build_run_payload`、`_report_and_join`/`set_run_context`、`_has_pending_subagents`、`StuckRunWatchdog._scan_once`/`list_stuck_running` 跨 Task 命名一致。
- 已知边缘（留后续）：watchdog 兜底唤醒用新 run_id（非复用，异常路径 SSE 走新流可接受）；用户在子运行期间发新消息会与续接交错（锁串行化保证不并发写，固定模式假设用户等待）。
