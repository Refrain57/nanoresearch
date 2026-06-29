# Phase 0：主 Agent 信箱 + 分布式锁 + 唯一入队调度器 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在平台派发主干上插入「per-(agent_id, conversation_id) 信箱 + 分布式锁 + 唯一入队调度器」三块地基，为后续异步子 agent 改造铺路，且**不改变任何用户可见行为**。

**Architecture:** HTTP 不再直接 `enqueue_job`，改为把任务 payload 投到 Redis Stream 信箱 + 往全局通知流发一条信号；一个长驻无状态调度器消费通知流，对目标信箱取 per-mailbox 分布式锁，抢到锁才 `enqueue_job` 拉起 `run_agent_job`。run 在 finally 里推进信箱游标、释放锁、若信箱还有积压则补发通知（链式排空）。出站事件流 `run_events:{run_id}` 与 SSE 端点零改动。

**Tech Stack:** Python 3.11、redis.asyncio、ARQ、FastAPI、pytest（DB 用真 Postgres，Redis 原语用真 Redis 集成测试 + 纯逻辑单测）。

## Global Constraints

- 对外行为完全不变：单轮 web 对话照常流式输出 + `run_end`；前端「拿 `run_id` 订阅 `/api/runs/{run_id}/events`」零改动、零感知。
- **不碰**：`subagent.py` 任何逻辑（含「子结果不回主」这个 bug，留 Phase 1）；SSE 前端 / `run_events` 出站流 / `chat_router` 的 `events` 端点（`chat_router.py:326-373`）；SCARD 轮询（`worker.py:398-412`）；会话存储改追加式（保持 `replace_messages` / `DEL+RPUSH` 全量覆盖）。
- 消费者组**只允许用在通知流这一层**，不得加到 agent / run_events 上。
- 分布式锁：`SET NX PX` + 唯一 token；释放用 Lua「GET 比对 then DEL」，**禁止裸 DEL**。
- 会话全量覆盖写（`manager.py:189-201`、`conversation_repo.py:128-139`）v0 不改，靠锁包住即可（单写者前提下安全）。
- **运行环境约束（实测）**：目标 Redis 为 **5.0.14**。排他区间 `(id`（XRANGE）与 `XAUTOCLAIM` 均为 6.2+，**不可用**。替代：读「严格在游标之后」用 `_next_stream_id(cursor)` + 包含式 XRANGE；PEL 重领用 `XPENDING` + `XCLAIM`（5.0 可用）。`XREADGROUP/XACK/XGROUP CREATE/XADD/XRANGE/XLEN` 在 5.0 均可用。
- 现状锚点（已复核，勿推翻）：HTTP 入口 `chat_router.py:256-313`，直接入队在 `chat_router.py:290`；`run_repo.create` 在 `:271`，返回 `{run_id}` 在 `:309-313`；`run_agent_job` 在 `worker.py:251`；`WorkerSettings`/`max_jobs=10`/`job_timeout=7200` 在 `worker.py:558-566`；`server/main.py` lifespan 里 `app.state.redis`(`:52`)、`pending_reaper.start()`(`:57-58`)、`arq_pool=create_pool`(`:68`)、shutdown(`:84-90`)。

---

## 设计要点（实现者必读）

### 数据通路
1. **信箱**（新增，入站）：`agent_inbox:{agent_id}:{conversation_id}` —— Redis Stream，每条 entry = 一次请求的完整 job payload（JSON）。
2. **信箱游标**（新增）：`agent_inbox_cursor:{agent_id}:{conversation_id}` —— String，存「最后已处理 entry id」。`XRANGE (cursor +]` 即未处理消息。
3. **通知流**（新增，全局）：`dispatch_notify` —— Redis Stream，每条 entry = `{mailbox_key, lock_key, cursor_key}`。调度器用消费者组 `dispatch_cg` 可靠消费。
4. **分布式锁**（新增）：`agent_lock:{agent_id}:{conversation_id}` —— String，`SET NX PX token`。
5. **出站流**（不动）：`run_events:{run_id}`（`redis_keys.py:22`）。

### 锁语义（关键，决定验收 2/4）
- 锁保护「同一信箱同一时刻只有一个 run 在写会话状态」。锁覆盖**整个 run 处理周期**（因为会话读-改-写贯穿整轮）。
- **PX = 30_000ms，刷新间隔 10_000ms（PX/3）**。依据：单条消息处理最长可达 ARQ `job_timeout=7200s`（深研子任务）；run 内开一个后台 refresher 每 10s 用 Lua「GET 比对 then PEXPIRE」续租；worker 一旦死亡 → 不再续租 → 锁 ≤30s 自动过期，对话不会被锁死超过 ~30s。30s 给足 3× 余量容忍事件循环卡顿/GC，又把崩溃恢复延迟压到可接受。直接满足验收 4。
- token 比对：续租/释放都先 `GET == token` 再操作，杜绝超时后误删/误续别人的锁。

### 链式排空（决定验收 2/3，且「抢不到锁的通知丢弃」）
- HTTP：`run_repo.create` → `XADD 信箱(payload)` → `XADD 通知流` → 返回 `run_id`。**1 消息 = 1 entry = 1 run_id**（保持现状 SSE 寻址）。
- 调度器：读通知 → `SET NX PX 锁`。抢到 → `XRANGE (cursor +] COUNT 1` 取下一条未处理 entry → `enqueue_job("run_agent_job", **payload, _lock 元数据 + _entry_id)`。**抢不到 → ACK 丢弃**（正在跑的 run 收尾会链式补发）。若抢到锁但无未处理 entry → 立即释放锁、ACK。
- run 收尾（finally，无论成败）：**单段 Lua 原子完成**「token 比对 → `SET cursor=_entry_id`（推进，避免毒消息重放）→ `XRANGE (cursor +] COUNT 1` 若有积压则 `XADD 通知流` → **最后** `DEL 锁`」。锁在最后删、且补发通知已在同一原子块内先于删锁完成，**不存在「锁已放但下一条未通知」的暴露态**（Must-fix 2）。token 失配 → 整段 no-op（锁已不属本 run）。
- 效果：同一信箱任意时刻只有一个 run；并发连发被锁串行化、按 entry 顺序逐条处理、零丢失零覆盖；每条消息恰好一个 run/run_id，SSE 不变。

### 调度器生命周期
- 长驻、无状态，挂在 `server/main.py` lifespan，与 `PendingReaper` 同生（`main.py:57-58` 之后 start，`:84` 之前 stop）。
- 唯一入队者：`chat_router.py:290` 的直接 `enqueue_job` 被移除，全平台入队只经调度器，杜绝双源。

---

## File Structure

- **Create** `backend/nanoresearch/bus/dist_lock.py` —— 分布式锁（acquire / refresh / release，Lua）。单一职责：锁原语。
- **Create** `backend/nanoresearch/bus/mailbox.py` —— 信箱/游标/通知流原语（post_message / post_notify / read_next_after_cursor / advance_cursor / ensure_group）。单一职责：信箱与通知流读写。
- **Create** `backend/nanoresearch/bus/dispatcher.py` —— `AgentDispatcher` 长驻调度器。单一职责：消费通知流 → 取锁 → 入队。
- **Modify** `backend/nanoresearch/bus/redis_keys.py` —— 新增 4 个 key 助手 + 常量（现有结构见 `redis_keys.py:1-75`）。
- **Modify** `backend/nanoresearch/server/routers/chat_router.py:290-307` —— 直接 enqueue 改为 投信箱 + 发通知。
- **Modify** `backend/nanoresearch/worker.py` —— `run_agent_job` 增加锁元数据可选参数 + finally 收尾（推游标/释放锁/链式补发）+ refresher；缺参时退化为现状（CLI/legacy 不受影响）。
- **Modify** `backend/nanoresearch/server/main.py` —— lifespan 起停 `AgentDispatcher`。
- **Create** `backend/tests/unit/bus/test_dist_lock.py`、`test_mailbox.py`、`test_dispatcher_logic.py`。
- **Create** `backend/tests/integration/test_phase0_dispatch.py` —— 4 条验收标准集成测试（真 Redis）。
- **Create** `backend/tests/integration/conftest.py` —— `redis_client` fixture（连真 Redis，测试前缀 flush，不可达则 skip）。

---

### Task 1: 分布式锁原语 `dist_lock.py`

**Files:**
- Create: `backend/nanoresearch/bus/dist_lock.py`
- Test: `backend/tests/unit/bus/test_dist_lock.py`

**Interfaces:**
- Produces:
  - `async def acquire(redis, key: str, *, px_ms: int = 30_000) -> str | None` —— 成功返回唯一 token，失败返回 `None`。
  - `async def refresh(redis, key: str, token: str, *, px_ms: int = 30_000) -> bool` —— token 匹配则续租并返回 `True`，否则 `False`。
  - `async def release(redis, key: str, token: str) -> bool` —— token 匹配则删除并返回 `True`，否则 `False`。
  - `RELEASE_LUA: str`、`REFRESH_LUA: str`（模块级常量）。

- [ ] **Step 1: 写失败测试**（真 Redis；不可达自动 skip）

```python
# backend/tests/unit/bus/test_dist_lock.py
import pytest
from nanoresearch.bus import dist_lock

pytestmark = pytest.mark.asyncio

async def test_acquire_then_second_acquire_fails(redis_client):
    key = "agent_lock:t:c1"
    tok1 = await dist_lock.acquire(redis_client, key, px_ms=5_000)
    assert tok1 is not None
    tok2 = await dist_lock.acquire(redis_client, key, px_ms=5_000)
    assert tok2 is None

async def test_release_with_wrong_token_is_noop(redis_client):
    key = "agent_lock:t:c2"
    tok = await dist_lock.acquire(redis_client, key, px_ms=5_000)
    assert await dist_lock.release(redis_client, key, "not-the-token") is False
    assert await dist_lock.release(redis_client, key, tok) is True
    assert await dist_lock.acquire(redis_client, key, px_ms=5_000) is not None

async def test_refresh_extends_only_with_matching_token(redis_client):
    key = "agent_lock:t:c3"
    tok = await dist_lock.acquire(redis_client, key, px_ms=2_000)
    assert await dist_lock.refresh(redis_client, key, tok, px_ms=10_000) is True
    assert await dist_lock.refresh(redis_client, key, "bad", px_ms=10_000) is False
    ttl = await redis_client.pttl(key)
    assert ttl > 2_000  # 续租生效
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/bus/test_dist_lock.py -v`
Expected: FAIL（`ModuleNotFoundError: nanoresearch.bus.dist_lock` 或 fixture 缺失）

- [ ] **Step 3: 实现 `dist_lock.py`**

```python
# backend/nanoresearch/bus/dist_lock.py
"""Per-(agent,conversation) distributed lock: SET NX PX + token, Lua release/refresh."""
from __future__ import annotations

import uuid
from typing import Any

RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""

REFRESH_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
else
    return 0
end
"""


async def acquire(redis: Any, key: str, *, px_ms: int = 30_000) -> str | None:
    token = uuid.uuid4().hex
    ok = await redis.set(key, token, nx=True, px=px_ms)
    return token if ok else None


async def refresh(redis: Any, key: str, token: str, *, px_ms: int = 30_000) -> bool:
    res = await redis.eval(REFRESH_LUA, 1, key, token, str(px_ms))
    return bool(res)


async def release(redis: Any, key: str, token: str) -> bool:
    res = await redis.eval(RELEASE_LUA, 1, key, token)
    return bool(res)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/unit/bus/test_dist_lock.py -v`
Expected: PASS（3 passed，或 Redis 不可达时 skipped）

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/bus/dist_lock.py backend/tests/unit/bus/test_dist_lock.py
git commit -m "feat(bus): per-mailbox distributed lock with token + Lua release/refresh"
```

---

### Task 2: Redis key 助手

**Files:**
- Modify: `backend/nanoresearch/bus/redis_keys.py`（在 `:60` 的 RAG 段之后、`:71` Pub/Sub 段之前插入）

**Interfaces:**
- Produces（`RedisKeys` 静态方法/常量）：
  - `agent_inbox(agent_id: str, conversation_id: str) -> str` → `f"agent_inbox:{agent_id}:{conversation_id}"`
  - `agent_inbox_cursor(agent_id: str, conversation_id: str) -> str` → `f"agent_inbox_cursor:{agent_id}:{conversation_id}"`
  - `agent_lock(agent_id: str, conversation_id: str) -> str` → `f"agent_lock:{agent_id}:{conversation_id}"`
  - `DISPATCH_NOTIFY = "dispatch_notify"`、`DISPATCH_GROUP = "dispatch_cg"`、`AGENT_INBOX_TTL = 86400`

- [ ] **Step 1: 写测试**

```python
# 追加到 backend/tests/unit/bus/test_mailbox.py 顶部（Task 3 同文件）
from nanoresearch.bus.redis_keys import RedisKeys

def test_inbox_keys_are_addressed_by_agent_and_conversation():
    assert RedisKeys.agent_inbox("a1", "c1") == "agent_inbox:a1:c1"
    assert RedisKeys.agent_inbox_cursor("a1", "c1") == "agent_inbox_cursor:a1:c1"
    assert RedisKeys.agent_lock("a1", "c1") == "agent_lock:a1:c1"
    assert RedisKeys.DISPATCH_NOTIFY == "dispatch_notify"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/bus/test_mailbox.py::test_inbox_keys_are_addressed_by_agent_and_conversation -v`
Expected: FAIL（AttributeError: agent_inbox）

- [ ] **Step 3: 实现（编辑 `redis_keys.py`，在第 60 行 `return f"chunk:..."` 方法之后插入）**

```python
    # Agent inbox / dispatch (Phase 0)
    AGENT_INBOX_TTL = 86400
    DISPATCH_NOTIFY = "dispatch_notify"
    DISPATCH_GROUP = "dispatch_cg"

    @staticmethod
    def agent_inbox(agent_id: str, conversation_id: str) -> str:
        return f"agent_inbox:{agent_id}:{conversation_id}"

    @staticmethod
    def agent_inbox_cursor(agent_id: str, conversation_id: str) -> str:
        return f"agent_inbox_cursor:{agent_id}:{conversation_id}"

    @staticmethod
    def agent_lock(agent_id: str, conversation_id: str) -> str:
        return f"agent_lock:{agent_id}:{conversation_id}"
```

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/unit/bus/test_mailbox.py::test_inbox_keys_are_addressed_by_agent_and_conversation -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/bus/redis_keys.py backend/tests/unit/bus/test_mailbox.py
git commit -m "feat(bus): redis key helpers for agent inbox / cursor / lock / notify"
```

---

### Task 3: 信箱 / 游标 / 通知流原语 `mailbox.py`

**Files:**
- Create: `backend/nanoresearch/bus/mailbox.py`
- Test: `backend/tests/unit/bus/test_mailbox.py`（与 Task 2 同文件）

**Interfaces:**
- Consumes: `RedisKeys.agent_inbox/agent_inbox_cursor/DISPATCH_NOTIFY/DISPATCH_GROUP/AGENT_INBOX_TTL`（Task 2）。
- Produces:
  - `async def post_message(redis, agent_id, conversation_id, payload: dict) -> str` —— XADD 信箱，返回 entry_id。
  - `async def post_notify(redis, *, mailbox_key, cursor_key, lock_key) -> None` —— XADD 通知流。
  - `async def read_next_after_cursor(redis, agent_id, conversation_id) -> tuple[str, dict] | None` —— `XRANGE (cursor +] COUNT 1`，返回 `(entry_id, payload)` 或 `None`。
  - `async def advance_cursor(redis, agent_id, conversation_id, entry_id: str) -> None`
  - `async def ensure_group(redis) -> None` —— 幂等创建通知流消费者组（`MKSTREAM`，已存在则忽略 BUSYGROUP）。
  - `async def finalize_and_release(redis, *, agent_id, conversation_id, lock_key, token, entry_id, ttl=AGENT_INBOX_TTL) -> bool` —— **原子收尾**（Must-fix 2）：一段 Lua 内「token 比对 → 推游标 → 若有积压则补发通知 → 释放锁」。token 失配则整段 no-op 返回 `False`（锁已不属于本 run，禁止误推游标/误放锁）。保证「锁释放的瞬间，下一条要么已通知、要么信箱确为空」，无中间暴露态。

- [ ] **Step 1: 写测试（真 Redis）**

```python
# backend/tests/unit/bus/test_mailbox.py （续）
import pytest
from nanoresearch.bus import mailbox

pytestmark = pytest.mark.asyncio

async def test_post_and_read_next_after_cursor(redis_client):
    aid, cid = "a1", "conv-roundtrip"
    e1 = await mailbox.post_message(redis_client, aid, cid, {"content": "hi-1"})
    e2 = await mailbox.post_message(redis_client, aid, cid, {"content": "hi-2"})

    got1 = await mailbox.read_next_after_cursor(redis_client, aid, cid)
    assert got1 == (e1, {"content": "hi-1"})

    await mailbox.advance_cursor(redis_client, aid, cid, e1)
    got2 = await mailbox.read_next_after_cursor(redis_client, aid, cid)
    assert got2 == (e2, {"content": "hi-2"})

    await mailbox.advance_cursor(redis_client, aid, cid, e2)
    assert await mailbox.read_next_after_cursor(redis_client, aid, cid) is None

async def test_ensure_group_is_idempotent(redis_client):
    await mailbox.ensure_group(redis_client)
    await mailbox.ensure_group(redis_client)  # 第二次不得抛

async def test_finalize_atomic_advances_renotifies_then_releases(redis_client):
    from nanoresearch.bus import dist_lock
    from nanoresearch.bus.redis_keys import RedisKeys
    aid, cid = "a1", "fin-atomic"
    e1 = await mailbox.post_message(redis_client, aid, cid, {"content": "m1"})
    await mailbox.post_message(redis_client, aid, cid, {"content": "m2"})  # 积压
    lock_key = RedisKeys.agent_lock(aid, cid)
    token = await dist_lock.acquire(redis_client, lock_key, px_ms=30_000)
    n0 = await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY)

    ok = await mailbox.finalize_and_release(
        redis_client, agent_id=aid, conversation_id=cid,
        lock_key=lock_key, token=token, entry_id=e1)
    assert ok is True
    assert (await redis_client.get(RedisKeys.agent_inbox_cursor(aid, cid))) == e1     # 推进
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) == n0 + 1               # 有积压→补发
    assert await redis_client.get(lock_key) is None                                  # 锁已放（最后一步）

async def test_finalize_is_noop_when_token_lost(redis_client):
    from nanoresearch.bus import dist_lock
    from nanoresearch.bus.redis_keys import RedisKeys
    aid, cid = "a1", "fin-lost"
    e1 = await mailbox.post_message(redis_client, aid, cid, {"content": "m1"})
    lock_key = RedisKeys.agent_lock(aid, cid)
    await dist_lock.acquire(redis_client, lock_key, px_ms=30_000)  # 别人持锁
    ok = await mailbox.finalize_and_release(
        redis_client, agent_id=aid, conversation_id=cid,
        lock_key=lock_key, token="stale-token", entry_id=e1)
    assert ok is False                                            # token 失配→整段 no-op
    assert await redis_client.get(RedisKeys.agent_inbox_cursor(aid, cid)) is None  # 未误推游标
    assert await redis_client.get(lock_key) is not None                            # 未误放别人的锁
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/bus/test_mailbox.py -v`
Expected: FAIL（`nanoresearch.bus.mailbox` 不存在）

- [ ] **Step 3: 实现 `mailbox.py`**

```python
# backend/nanoresearch/bus/mailbox.py
"""Per-(agent,conversation) inbox + cursor + global notify stream primitives."""
from __future__ import annotations

import json
from typing import Any

from nanoresearch.bus.redis_keys import RedisKeys


async def post_message(redis: Any, agent_id: str, conversation_id: str, payload: dict) -> str:
    key = RedisKeys.agent_inbox(agent_id, conversation_id)
    entry_id = await redis.xadd(key, {"data": json.dumps(payload, ensure_ascii=False)})
    await redis.expire(key, RedisKeys.AGENT_INBOX_TTL)
    return entry_id


async def post_notify(redis: Any, *, mailbox_key: str, cursor_key: str, lock_key: str) -> None:
    await redis.xadd(RedisKeys.DISPATCH_NOTIFY, {
        "mailbox_key": mailbox_key,
        "cursor_key": cursor_key,
        "lock_key": lock_key,
    })


async def read_next_after_cursor(
    redis: Any, agent_id: str, conversation_id: str
) -> tuple[str, dict] | None:
    inbox = RedisKeys.agent_inbox(agent_id, conversation_id)
    cursor_key = RedisKeys.agent_inbox_cursor(agent_id, conversation_id)
    cursor = await redis.get(cursor_key) or "0-0"
    # XRANGE start is inclusive; "(" prefix makes it exclusive of the cursor.
    res = await redis.xrange(inbox, min=f"({cursor}", max="+", count=1)
    if not res:
        return None
    entry_id, fields = res[0]
    return entry_id, json.loads(fields["data"])


async def advance_cursor(redis: Any, agent_id: str, conversation_id: str, entry_id: str) -> None:
    await redis.set(RedisKeys.agent_inbox_cursor(agent_id, conversation_id), entry_id,
                    ex=RedisKeys.AGENT_INBOX_TTL)


async def ensure_group(redis: Any) -> None:
    try:
        await redis.xgroup_create(RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP,
                                  id="0", mkstream=True)
    except Exception as e:  # redis.exceptions.ResponseError: BUSYGROUP
        if "BUSYGROUP" not in str(e):
            raise


# Must-fix 2: atomic finalize — token-check → advance cursor → (if backlog) re-notify → release,
# all in ONE Lua. The lock is DEL'd LAST and only after the next notify (if any) is already in the
# stream, so there is no "lock freed but next not notified" exposure window. token mismatch → no-op.
# KEYS[1]=lock_key KEYS[2]=inbox KEYS[3]=cursor KEYS[4]=notify_stream
# ARGV[1]=token ARGV[2]=entry_id ARGV[3]=cursor_ttl ARGV[4]=mailbox_key ARGV[5]=cursor_key ARGV[6]=lock_key
FINALIZE_LUA = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('SET', KEYS[3], ARGV[2], 'EX', ARGV[3])
local nxt = redis.call('XRANGE', KEYS[2], '(' .. ARGV[2], '+', 'COUNT', 1)
if #nxt > 0 then
    redis.call('XADD', KEYS[4], '*', 'mailbox_key', ARGV[4], 'cursor_key', ARGV[5], 'lock_key', ARGV[6])
end
redis.call('DEL', KEYS[1])
return 1
"""


async def finalize_and_release(
    redis: Any, *, agent_id: str, conversation_id: str,
    lock_key: str, token: str, entry_id: str, ttl: int = RedisKeys.AGENT_INBOX_TTL,
) -> bool:
    inbox = RedisKeys.agent_inbox(agent_id, conversation_id)
    cursor = RedisKeys.agent_inbox_cursor(agent_id, conversation_id)
    res = await redis.eval(
        FINALIZE_LUA, 4,
        lock_key, inbox, cursor, RedisKeys.DISPATCH_NOTIFY,   # KEYS
        token, entry_id, str(ttl), inbox, cursor, lock_key,   # ARGV
    )
    return bool(res)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/unit/bus/test_mailbox.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/bus/mailbox.py backend/tests/unit/bus/test_mailbox.py
git commit -m "feat(bus): inbox/cursor/notify stream primitives"
```

---

### Task 4: 调度器 `dispatcher.py`

**Files:**
- Create: `backend/nanoresearch/bus/dispatcher.py`
- Test: `backend/tests/unit/bus/test_dispatcher_logic.py`

**Interfaces:**
- Consumes: `dist_lock.acquire/release`（Task 1）、`mailbox.read_next_after_cursor/ensure_group`（Task 3）、`RedisKeys`。
- Produces:
  - `class AgentDispatcher` —— `__init__(self, redis, arq_pool, *, lock_px_ms: int = 30_000)`、`async def start()`、`async def stop()`。
  - `async def _handle_notify(self, fields: dict) -> str` —— 处理单条通知，返回决策 `"enqueued" | "dropped_locked" | "empty_released"`（便于单测，不触发真 ARQ）。`enqueue` 动作走 `self.arq_pool.enqueue_job`，测试用 fake pool 注入。

- [ ] **Step 1: 写决策单测（fake redis-lock + fake arq pool；纯逻辑）**

```python
# backend/tests/unit/bus/test_dispatcher_logic.py
import pytest
from nanoresearch.bus.dispatcher import AgentDispatcher

pytestmark = pytest.mark.asyncio

class _FakePool:
    def __init__(self): self.jobs = []
    async def enqueue_job(self, fn, **kw): self.jobs.append((fn, kw))

async def test_handle_notify_enqueues_when_lock_acquired(redis_client):
    aid, cid = "a1", "disp-c1"
    # 信箱里放一条
    from nanoresearch.bus import mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    await mailbox.post_message(redis_client, aid, cid, {"content": "x", "agent_id": aid,
                                                        "conversation_id": cid, "run_id": "r1"})
    pool = _FakePool()
    disp = AgentDispatcher(redis_client, pool)
    decision = await disp._handle_notify({
        "mailbox_key": RedisKeys.agent_inbox(aid, cid),
        "cursor_key": RedisKeys.agent_inbox_cursor(aid, cid),
        "lock_key": RedisKeys.agent_lock(aid, cid),
    })
    assert decision == "enqueued"
    assert pool.jobs and pool.jobs[0][0] == "run_agent_job"
    assert pool.jobs[0][1]["run_id"] == "r1"
    assert pool.jobs[0][1]["_lock_token"]  # 锁元数据已带上

async def test_second_notify_dropped_while_locked(redis_client):
    aid, cid = "a1", "disp-c2"
    from nanoresearch.bus import mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    await mailbox.post_message(redis_client, aid, cid, {"content": "x", "agent_id": aid,
                                                        "conversation_id": cid, "run_id": "r1"})
    pool = _FakePool()
    disp = AgentDispatcher(redis_client, pool)
    notify = {"mailbox_key": RedisKeys.agent_inbox(aid, cid),
              "cursor_key": RedisKeys.agent_inbox_cursor(aid, cid),
              "lock_key": RedisKeys.agent_lock(aid, cid)}
    assert await disp._handle_notify(notify) == "enqueued"
    # 锁仍被持有（run 还没收尾释放）→ 第二次必须丢弃，不得二次入队
    assert await disp._handle_notify(notify) == "dropped_locked"
    assert len(pool.jobs) == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/bus/test_dispatcher_logic.py -v`
Expected: FAIL（`nanoresearch.bus.dispatcher` 不存在）

- [ ] **Step 3: 实现 `dispatcher.py`**

```python
# backend/nanoresearch/bus/dispatcher.py
"""Long-lived, stateless dispatcher: the ONLY job enqueuer.

Consumes the global notify stream via a consumer group, acquires the
per-mailbox distributed lock, and enqueues run_agent_job for the next
unprocessed inbox entry. Extra notifies for a locked mailbox are dropped;
the running run re-posts a notify on finish if the inbox still has backlog.
"""
from __future__ import annotations

import asyncio
import socket
from typing import Any

from loguru import logger

from nanoresearch.bus import dist_lock, mailbox
from nanoresearch.bus.redis_keys import RedisKeys


def _parse_inbox_key(mailbox_key: str) -> tuple[str, str]:
    # "agent_inbox:{agent_id}:{conversation_id}"
    _, agent_id, conversation_id = mailbox_key.split(":", 2)
    return agent_id, conversation_id


class AgentDispatcher:
    def __init__(self, redis: Any, arq_pool: Any, *, lock_px_ms: int = 30_000) -> None:
        self._redis = redis
        self._arq = arq_pool
        self._lock_px_ms = lock_px_ms
        self._consumer = f"disp-{socket.gethostname()}-{id(self)}"
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        await mailbox.ensure_group(self._redis)
        await self._reclaim_pending()   # Adjustment 3: 重启自愈历史未 ACK 的通知
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("AgentDispatcher started (consumer={})", self._consumer)

    async def _reclaim_pending(self) -> None:
        """One-shot reclaim of notify entries left un-ACKed by a previous dispatcher
        instance (PEL), so a restart self-heals. Redis 5.0 has no XAUTOCLAIM (6.2+), so
        use XPENDING (summary) → XCLAIM. Continuous reclaim = Phase 0.1. Never raises."""
        try:
            summary = await self._redis.xpending(
                RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP)
            # redis-py xpending summary: {"pending": n, "min": .., "max": .., "consumers": [..]}
            if not summary or not summary.get("pending"):
                return
            detail = await self._redis.xpending_range(
                RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP,
                min="-", max="+", count=500)
            ids = [item["message_id"] for item in detail]
            if not ids:
                return
            claimed = await self._redis.xclaim(
                RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP, self._consumer,
                min_idle_time=0, message_ids=ids)
            for entry_id, fields in claimed:
                try:
                    await self._handle_notify(fields)
                except Exception:
                    logger.exception("reclaim _handle_notify failed")
                finally:
                    await self._redis.xack(
                        RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP, entry_id)
        except Exception:
            logger.exception("dispatcher _reclaim_pending failed (non-fatal)")

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
                resp = await self._redis.xreadgroup(
                    RedisKeys.DISPATCH_GROUP, self._consumer,
                    {RedisKeys.DISPATCH_NOTIFY: ">"}, count=20, block=5_000,
                )
                if not resp:
                    continue
                for _stream, entries in resp:
                    for entry_id, fields in entries:
                        try:
                            await self._handle_notify(fields)
                        except Exception:
                            logger.exception("dispatcher _handle_notify failed")
                        finally:
                            await self._redis.xack(
                                RedisKeys.DISPATCH_NOTIFY, RedisKeys.DISPATCH_GROUP, entry_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("dispatcher loop error; retrying")
                await asyncio.sleep(1)

    async def _handle_notify(self, fields: dict) -> str:
        mailbox_key = fields["mailbox_key"]
        lock_key = fields["lock_key"]
        agent_id, conversation_id = _parse_inbox_key(mailbox_key)

        token = await dist_lock.acquire(self._redis, lock_key, px_ms=self._lock_px_ms)
        if token is None:
            return "dropped_locked"  # a run is already processing this mailbox

        nxt = await mailbox.read_next_after_cursor(self._redis, agent_id, conversation_id)
        if nxt is None:
            await dist_lock.release(self._redis, lock_key, token)
            return "empty_released"

        entry_id, payload = nxt
        await self._arq.enqueue_job(
            "run_agent_job",
            **payload,
            _lock_key=lock_key,
            _lock_token=token,
            _entry_id=entry_id,
        )
        return "enqueued"
```

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/unit/bus/test_dispatcher_logic.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/bus/dispatcher.py backend/tests/unit/bus/test_dispatcher_logic.py
git commit -m "feat(bus): stateless AgentDispatcher — sole enqueuer via notify CG + mailbox lock"
```

---

### Task 5: worker 收尾接管（推游标 / 释放锁 / 链式补发 / 续租）

**Files:**
- Modify: `backend/nanoresearch/worker.py`（`run_agent_job` 签名 + 收尾，现状 `:251`、`:434-454` finally）
- Test: `backend/tests/integration/test_phase0_dispatch.py`（验收测试在 Task 7；本 Task 加 finally 行为的窄测）

**Interfaces:**
- Consumes: `dist_lock.refresh/release`、`mailbox.advance_cursor/read_next_after_cursor/post_notify`、`RedisKeys`。
- Produces: `run_agent_job` 新增可选 kwargs `_lock_key: str | None = None`、`_lock_token: str | None = None`、`_entry_id: str | None = None`；三者齐全才启用锁生命周期，否则行为同现状（CLI/legacy 不受影响）。

- [ ] **Step 1: 写 finally 行为窄测（真 Redis，桩掉实际 LLM 处理）**

```python
# backend/tests/integration/test_phase0_dispatch.py （第一段）
import pytest
from nanoresearch.bus import dist_lock, mailbox
from nanoresearch.bus.redis_keys import RedisKeys

pytestmark = pytest.mark.asyncio

async def test_run_finally_releases_lock_advances_cursor_and_renotifies(redis_client):
    """run_agent_job 收尾必须原子地：推进游标、（有积压则）补发通知、最后释放锁。"""
    import nanoresearch.worker as worker
    aid, cid = "a1", "fin-c1"
    # 信箱放两条；游标停在 0-0
    e1 = await mailbox.post_message(redis_client, aid, cid,
        {"content": "m1", "agent_id": aid, "conversation_id": cid, "run_id": "r1",
         "session_key": f"web:{cid}", "uid": "u1"})
    e2 = await mailbox.post_message(redis_client, aid, cid,
        {"content": "m2", "agent_id": aid, "conversation_id": cid, "run_id": "r2",
         "session_key": f"web:{cid}", "uid": "u1"})
    lock_key = RedisKeys.agent_lock(aid, cid)
    token = await dist_lock.acquire(redis_client, lock_key, px_ms=30_000)

    notify_before = await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY)
    await worker._finalize_mailbox_run(
        redis_client, agent_id=aid, conversation_id=cid,
        lock_key=lock_key, lock_token=token, entry_id=e1)

    # 游标推进到 e1
    assert (await redis_client.get(RedisKeys.agent_inbox_cursor(aid, cid))) == e1
    # 锁已释放
    assert await dist_lock.acquire(redis_client, lock_key, px_ms=1000) is not None
    # 信箱还有 e2 → 补发了通知
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) == notify_before + 1
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/integration/test_phase0_dispatch.py::test_run_finally_releases_lock_advances_cursor_and_renotifies -v`
Expected: FAIL（`worker._finalize_mailbox_run` 不存在）

- [ ] **Step 3: 实现 —— 在 `worker.py` 顶部加收尾助手 + refresher，并接到 `run_agent_job`**

新增模块级助手（放在 `run_agent_job` 之前）：

```python
async def _finalize_mailbox_run(redis, *, agent_id, conversation_id, lock_key, lock_token, entry_id):
    """Must-fix 2: atomic finalize via one Lua (cursor→re-notify→release, token-gated).
    The lock is released LAST and only after the next notify is in place — no exposure
    window. token mismatch → whole thing is a no-op. Best-effort, never raises."""
    from nanoresearch.bus import mailbox
    try:
        await mailbox.finalize_and_release(
            redis, agent_id=agent_id, conversation_id=conversation_id,
            lock_key=lock_key, token=lock_token, entry_id=entry_id)
    except Exception:
        logger.warning("finalize_mailbox_run failed (non-fatal)")


async def _lock_refresher(redis, lock_key, lock_token, px_ms, stop_evt, abort_evt, proc_task):
    """Periodically extend the lock lease while the run is alive.

    Must-fix 4: if refresh returns False (lease lost — token no longer ours), do NOT let the
    run keep writing obliviously: set abort_evt and cancel the processing task. The atomic
    finalize is token-gated, so it will correctly no-op afterwards (won't touch a lock we lost).
    """
    from nanoresearch.bus import dist_lock
    interval = px_ms / 3 / 1000.0
    try:
        while not stop_evt.is_set():
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=interval)
                return  # stop_evt set → run finished normally
            except asyncio.TimeoutError:
                pass
            if not await dist_lock.refresh(redis, lock_key, lock_token, px_ms=px_ms):
                logger.error(
                    "lock lease LOST for {} — aborting run to prevent unguarded writes", lock_key)
                abort_evt.set()
                if not proc_task.done():
                    proc_task.cancel()
                return
    except asyncio.CancelledError:
        return
```

在 `run_agent_job` 签名末尾加可选参数：

```python
    _lock_key: str | None = None,
    _lock_token: str | None = None,
    _entry_id: str | None = None,
```

紧接 `redis = get_redis()`（`worker.py:276`）之后声明锁生命周期状态（refresher 任务**稍后**起，因为它要拿 `_proc_task`）：

```python
    _mailbox_enabled = bool(_lock_key and _lock_token and _entry_id)
    _refresh_stop = asyncio.Event()
    _abort_evt = asyncio.Event()
    _refresh_task = None
```

把 `process_direct` 调用（`worker.py:376`）从 `await loop.process_direct(...)` 改为「包成 task + 启 refresher + await」：

```python
        _proc_task = asyncio.create_task(loop.process_direct(
            content,
            session_key=session_key, channel="web", chat_id=_chat_id, run_id=run_id,
            on_stream=on_stream, on_progress=on_progress, on_tool_call=on_tool_call,
            skill_names=skill_names, agent_id=agent_id, agent_override=agent_override,
            custom_persona=custom_persona, harness=harness, agents_registry=agents_registry,
            kb_bindings=kb_bindings, kb_map=kb_map, conversation_id=conversation_id,
        ))
        if _mailbox_enabled:
            _refresh_task = asyncio.create_task(_lock_refresher(
                redis, _lock_key, _lock_token, 30_000, _refresh_stop, _abort_evt, _proc_task))
        await _proc_task   # 续租失败时 refresher 会 cancel 它 → 抛 CancelledError
```

> 字段集与现状 `process_direct(...)`（`worker.py:376-394`）逐字段一致，仅多了「包 task」。

在现有 `finally`（`worker.py:442`）**最前面**追加收尾（早于现有的 `close_mcp` 等）：

```python
    finally:
        if _mailbox_enabled:
            _refresh_stop.set()
            if _refresh_task:
                try:
                    await _refresh_task
                except Exception:
                    pass
            await _finalize_mailbox_run(
                redis, agent_id=str(agent_id) if agent_id else "none",
                conversation_id=conversation_id, lock_key=_lock_key,
                lock_token=_lock_token, entry_id=_entry_id)
        # —— 以下为现有 finally 内容，保持不变（close_mcp / 删 job、cancel 键）——
        if loop is not None:
            try:
                await loop.close_mcp()
            ...
```

> 注 1：信箱寻址用 `agent_id`；`conversation.agent_id` 为空时用占位 `"none"`，与 HTTP 投信箱侧（Task 6）一致。
> 注 2（已知 Phase 0 边缘）：续租失败导致 `_proc_task` 被 cancel 时，沿用现状走 `CancelledError` 路径（`except Exception` 不接 → 不发 `run_end`），该信箱的 SSE 可能挂起直到 Phase 1 的 stuck-run watchdog 接管。续租失败在 30s/10s 的 3× 余量下属罕见路径，本期接受。

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/integration/test_phase0_dispatch.py::test_run_finally_releases_lock_advances_cursor_and_renotifies -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/worker.py backend/tests/integration/test_phase0_dispatch.py
git commit -m "feat(worker): mailbox run finalize (cursor/lock/re-notify) + lock refresher"
```

---

### Task 6: HTTP 入口改投信箱 + 发通知（移除直接入队）

**Files:**
- Modify: `backend/nanoresearch/server/routers/chat_router.py:290-307`
- Test: `backend/tests/integration/test_phase0_dispatch.py`（追加）

**Interfaces:**
- Consumes: `mailbox.post_message/post_notify`、`RedisKeys`。
- Produces: HTTP POST 不再 `arq_pool.enqueue_job`；改为 `post_message`(信箱) + `post_notify`(通知流)。`run_id` 创建（`:271`）与返回（`:309-313`）不变。

- [ ] **Step 1: 写测试 —— HTTP 后信箱有一条且通知流 +1，且未直接入队**

```python
# backend/tests/integration/test_phase0_dispatch.py （追加）
async def test_http_posts_to_inbox_and_notify_not_direct_enqueue(redis_client, monkeypatch):
    from nanoresearch.bus import mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    aid, cid = "none", "http-c1"   # agent_id 为空 → 占位 none
    payload = {"content": "hello", "agent_id": None, "conversation_id": cid,
               "run_id": "r1", "session_key": f"web:{cid}", "uid": "u1"}
    # 模拟 chat_router 的投递分支（提取为 helper enqueue_via_mailbox）
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox
    await _enqueue_via_mailbox(redis_client, payload)

    got = await mailbox.read_next_after_cursor(redis_client, aid, cid)
    assert got is not None and got[1]["content"] == "hello"
    assert await redis_client.xlen(RedisKeys.DISPATCH_NOTIFY) >= 1

async def test_idempotency_gate_blocks_duplicate_inbox_entry(redis_client):
    """Must-fix 1: 同一 job_id 第二次 SET NX 失败 → 不得再投信箱。"""
    from nanoresearch.bus import mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    aid, cid = "none", "dedup-c1"
    job_id = "dedupjob1"
    # 模拟第一次：SET NX 占位成功 → 投信箱
    won1 = await redis_client.set(RedisKeys.job(job_id), "r1", nx=True, ex=3600)
    assert won1
    await mailbox.post_message(redis_client, aid, cid, {"content": "x", "agent_id": None,
                                                        "conversation_id": cid, "run_id": "r1"})
    # 第二次（重复提交）：SET NX 必失败 → 不投信箱
    won2 = await redis_client.set(RedisKeys.job(job_id), "r2", nx=True, ex=3600)
    assert not won2
    # 信箱里只有一条
    e1 = await mailbox.read_next_after_cursor(redis_client, aid, cid)
    await mailbox.advance_cursor(redis_client, aid, cid, e1[0])
    assert await mailbox.read_next_after_cursor(redis_client, aid, cid) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/integration/test_phase0_dispatch.py::test_http_posts_to_inbox_and_notify_not_direct_enqueue -v`
Expected: FAIL（`_enqueue_via_mailbox` 不存在）

- [ ] **Step 3: 实现 —— 提取 helper 并替换 `chat_router.py:290`**

新增 helper（模块级，便于测试）：

```python
async def _enqueue_via_mailbox(redis, payload: dict) -> None:
    """Phase 0: post to per-(agent,conversation) inbox + notify the dispatcher.
    Replaces the direct arq enqueue so the dispatcher is the sole enqueuer."""
    from nanoresearch.bus import mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    agent_id = payload.get("agent_id") or "none"
    conversation_id = payload["conversation_id"]
    await mailbox.post_message(redis, agent_id, conversation_id, payload)
    await mailbox.post_notify(
        redis,
        mailbox_key=RedisKeys.agent_inbox(agent_id, conversation_id),
        cursor_key=RedisKeys.agent_inbox_cursor(agent_id, conversation_id),
        lock_key=RedisKeys.agent_lock(agent_id, conversation_id),
    )
```

保留 `chat_router.py:256-269` 现有的 GET 预检（快路径），并把 `:290-307` 的直接 `enqueue_job` 整块替换为 **「原子 `SET NX` 幂等闸门 → 命中则不投信箱直接返回 dedup → 未命中才投信箱」**（Must-fix 1）。`_redis = request.app.state.redis` 已在 `:255` 取得：

```python
    # Must-fix 1: 原子幂等闸门 —— 在投信箱前用 SET NX 占位，重复提交/前端重试
    # 不会在信箱里投出第二条 entry（否则会产生第二个 run + 二次 LLM + 二次写库，是回归）。
    _JOB_TTL = 3600
    _won = await _redis.set(RedisKeys.job(_job_id), str(run_id), nx=True, ex=_JOB_TTL)
    if not _won:
        _existing = await _redis.get(RedisKeys.job(_job_id))
        return {"run_id": _existing or str(run_id),
                "conversation_id": str(conv.id), "status": "dedup"}

    payload = {
        "run_id": str(run_id),
        "session_key": session_key,
        "conversation_id": str(conv.id),
        "content": body.content,
        "uid": uid,
        "rag_mode": body.rag_mode,
        "kb_id": body.kb_id,
        "skill_names": skill_names,
        "agent_id": str(conv.agent_id) if conv.agent_id else None,
        "agent_override": agent_override or None,
        "custom_persona": custom_persona,
        "harness": agent_harness or None,
        "agents_registry": agents_registry or None,
        "agent_kb_id": agent_kb_id,
        "job_id": _job_id,
    }
    await _enqueue_via_mailbox(_redis, payload)
```

> 字段集与原 `enqueue_job("run_agent_job", ...)`（`:290-307`）逐字段一致，worker 侧拿到的参数不变。`job_id` 仍随 payload 传入，worker 启动时 `redis.set(RedisKeys.job(job_id), run_id)`（`worker.py:283-285`）继续做二次确认。
> 已知良性副作用：并发重复请求中"输掉 SET NX"的那个，其 `run_repo.create` 产生的 run 行会成为永不处理的孤儿（status=pending），因其 run_id 从未返回给任何人、无 SSE 订阅，无害；Phase 1 的 stuck-run watchdog 可顺带回收。

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/integration/test_phase0_dispatch.py::test_http_posts_to_inbox_and_notify_not_direct_enqueue -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/server/routers/chat_router.py backend/tests/integration/test_phase0_dispatch.py
git commit -m "feat(api): route run dispatch through mailbox+notify (dispatcher is sole enqueuer)"
```

---

### Task 7: 调度器接入 lifespan + 4 条验收集成测试

**Files:**
- Modify: `backend/nanoresearch/server/main.py`（lifespan，`:57-58` 后 start，`:84` 前 stop）
- Create: `backend/tests/integration/conftest.py`（`redis_client` fixture）
- Test: `backend/tests/integration/test_phase0_dispatch.py`（追加 4 条验收）

**Interfaces:**
- Consumes: `AgentDispatcher`（Task 4）。
- Produces: `app.state.dispatcher`。

- [ ] **Step 1: 写 `redis_client` fixture + 4 条验收测试**

```python
# backend/tests/integration/conftest.py
import os, pytest, pytest_asyncio

@pytest_asyncio.fixture
async def redis_client():
    import redis.asyncio as aioredis
    url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")  # DB 15 = 测试库
    r = aioredis.from_url(url, decode_responses=True)
    try:
        await r.ping()
    except Exception:
        pytest.skip("Redis not reachable for integration tests")
    await r.flushdb()
    yield r
    await r.flushdb()
    await r.aclose()
```

```python
# backend/tests/integration/test_phase0_dispatch.py （追加，端到端用 dispatcher + fake pool）
from nanoresearch.bus.dispatcher import AgentDispatcher

class _RecordingPool:
    def __init__(self): self.jobs = []
    async def enqueue_job(self, fn, **kw): self.jobs.append((fn, kw)); return None

async def test_ac1_single_message_one_run(redis_client):
    """验收1：单条消息 → 恰好一个 run 被入队。"""
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox
    pool = _RecordingPool()
    disp = AgentDispatcher(redis_client, pool)
    await _enqueue_via_mailbox(redis_client, {
        "content": "hi", "agent_id": None, "conversation_id": "ac1", "run_id": "r1",
        "session_key": "web:ac1", "uid": "u1"})
    # 调度一轮（直接驱动 _handle_notify，避开后台 loop 计时）
    from nanoresearch.bus.redis_keys import RedisKeys
    res = await redis_client.xrange(RedisKeys.DISPATCH_NOTIFY)
    for _id, fields in res:
        await disp._handle_notify(fields)
    assert len(pool.jobs) == 1 and pool.jobs[0][1]["run_id"] == "r1"

async def test_ac3_no_double_dispatch_while_locked(redis_client):
    """验收3：同信箱第二次通知在锁持有期被丢弃，不产生第二个 run。"""
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    pool = _RecordingPool()
    disp = AgentDispatcher(redis_client, pool)
    await _enqueue_via_mailbox(redis_client, {
        "content": "m1", "agent_id": None, "conversation_id": "ac3", "run_id": "r1",
        "session_key": "web:ac3", "uid": "u1"})
    await _enqueue_via_mailbox(redis_client, {
        "content": "m2", "agent_id": None, "conversation_id": "ac3", "run_id": "r2",
        "session_key": "web:ac3", "uid": "u1"})
    for _id, fields in await redis_client.xrange(RedisKeys.DISPATCH_NOTIFY):
        await disp._handle_notify(fields)
    assert len(pool.jobs) == 1  # 锁未释放，第二条被丢弃

async def test_ac2_serialized_drain_after_finalize(redis_client):
    """验收2：第一个 run 收尾后链式拉起第二条，两条按序、无覆盖。"""
    import nanoresearch.worker as worker
    from nanoresearch.server.routers.chat_router import _enqueue_via_mailbox
    from nanoresearch.bus.redis_keys import RedisKeys
    pool = _RecordingPool()
    disp = AgentDispatcher(redis_client, pool)
    for rid, c in [("r1", "m1"), ("r2", "m2")]:
        await _enqueue_via_mailbox(redis_client, {
            "content": c, "agent_id": None, "conversation_id": "ac2", "run_id": rid,
            "session_key": "web:ac2", "uid": "u1"})
    for _id, fields in await redis_client.xrange(RedisKeys.DISPATCH_NOTIFY):
        await disp._handle_notify(fields)
    assert [j[1]["run_id"] for j in pool.jobs] == ["r1"]   # 只起了第一条
    # 模拟 run r1 收尾
    j = pool.jobs[0][1]
    await worker._finalize_mailbox_run(
        redis_client, agent_id="none", conversation_id="ac2",
        lock_key=j["_lock_key"], lock_token=j["_lock_token"], entry_id=j["_entry_id"])
    # 收尾补发了通知 → 再调度一轮，起第二条
    for _id, fields in await redis_client.xrange(RedisKeys.DISPATCH_NOTIFY, min="(" + j["_entry_id"]):
        pass
    new = await redis_client.xrange(RedisKeys.DISPATCH_NOTIFY)
    await disp._handle_notify(new[-1][1])
    assert [j[1]["run_id"] for j in pool.jobs] == ["r1", "r2"]

async def test_ac4_lock_auto_expires_on_worker_death(redis_client):
    """验收4：持锁者"死亡"（不续租）后 PX 到期，锁可被重新获取，不死锁。"""
    from nanoresearch.bus import dist_lock
    from nanoresearch.bus.redis_keys import RedisKeys
    key = RedisKeys.agent_lock("none", "ac4")
    tok = await dist_lock.acquire(redis_client, key, px_ms=300)  # 模拟短 PX
    assert tok is not None
    import asyncio; await asyncio.sleep(0.4)  # 不续租 = 模拟 worker 死亡
    assert await dist_lock.acquire(redis_client, key, px_ms=300) is not None
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/integration/test_phase0_dispatch.py -v`
Expected: FAIL（部分用例因 main.py 未接入 / 断言不满足）

- [ ] **Step 3: 接入 lifespan（`server/main.py`）**

`:57-58`（pending_reaper.start 后）追加：

```python
        from nanoresearch.bus.dispatcher import AgentDispatcher
        app.state.dispatcher = AgentDispatcher(app.state.redis, app.state.arq_pool)
        await app.state.dispatcher.start()
```

> 注意顺序：必须在 `app.state.arq_pool = await create_pool(...)`（`main.py:68`）**之后**再建 dispatcher（它依赖 arq_pool）。把上面两行移到 `:68` 之后。

`:84`（pending_reaper.stop 前）追加：

```python
        if getattr(app.state, "dispatcher", None):
            await app.state.dispatcher.stop()
```

并把启动占位 `app.state.dispatcher = None` 加到 `:99` 附近（与 `arq_pool=None` 同处）。

- [ ] **Step 4: 运行全部验收**

Run: `cd backend && python -m pytest tests/integration/test_phase0_dispatch.py tests/unit/bus -v`
Expected: PASS（Redis 不可达则相关用例 skip）

- [ ] **Step 5: 提交**

```bash
git add backend/nanoresearch/server/main.py backend/tests/integration/
git commit -m "feat(server): wire AgentDispatcher into lifespan; phase0 acceptance tests"
```

---

## 「对外行为不变」论证

1. **出站不变**：`run_agent_job` 处理逻辑、`on_stream → xadd_event(run_events:{run_id})`（`worker.py:332`）、落库、`run_end` 全未改；新增只在「入口投递」和「run 收尾」两端。
2. **SSE 不变**：`/api/runs/{run_id}/events`（`chat_router.py:326-373`）零改动；`run_id` 仍在 HTTP 同步创建并返回（`:271`、`:309-313`），前端订阅链路不变。
3. **单消息等价**：HTTP 投信箱 → 通知 → 调度器取锁 → 入队 `run_agent_job(**原 payload)`，参数集与原 `enqueue_job`（`:290-307`）逐字段一致；run 内部与现状完全相同，只是 finally 多了「推游标/释放锁/补发」三个 best-effort 步骤（失败不影响主流程）。
4. **唯一入队**：`chat_router.py:290` 的直接 enqueue 被删除，平台唯一入队点变为调度器，杜绝双源。
5. **CLI/legacy 不受影响**：`run_agent_job` 锁元数据为可选参数，缺省时不启用锁生命周期，行为同现状；CLI 路径根本不经 worker。

## 风险与回滚
- **最高风险**：调度器是唯一入队者，若它挂了则所有对话不被处理。缓解：通知流用消费者组 + **启动时一次性 `XAUTOCLAIM` 重领历史 PEL**（Adjustment 3，已在本期）；lifespan 与进程同生；**持续** autoclaim 留 Phase 0.1。
- **回滚**：单 commit 还原 `chat_router.py` 投递分支即恢复直接 enqueue；新增模块不被引用即无副作用。

## 评审反馈落实（本次修订）
- **Must-fix 1（幂等去重）**：Task 6 投信箱前加原子 `SET NX RedisKeys.job(job_id)` 闸门，命中→不投信箱、返回已有 run_id（`status:dedup`）。补 `test_idempotency_gate_blocks_duplicate_inbox_entry`。
- **Must-fix 2（finalize 原子化）**：`mailbox.finalize_and_release` 用单段 `FINALIZE_LUA`「token 比对→推游标→有积压补发通知→**最后**释放锁」，消除「锁放了但下一条未通知」暴露态。补 `test_finalize_atomic_...` / `test_finalize_is_noop_when_token_lost`。
- **Adjustment 3（一次性 XAUTOCLAIM）**：`AgentDispatcher.start()` 在 `ensure_group` 后调 `_reclaim_pending()` 重领历史 PEL，重启自愈。
- **Adjustment 4（续租失败反应）**：`_lock_refresher` 在 `refresh()` 返回 False 时 `abort_evt.set()` + `proc_task.cancel()`，不放任「锁丢仍写库」；finalize 因 token 失配自动 no-op。

## Self-Review 记录
- Spec 覆盖：信箱(Task 2/3)、分布式锁(Task 1)、唯一入队调度器(Task 4/6/7)、幂等(Task 6)、原子 finalize(Task 3/5)、续租自卫(Task 5)、一次性 reclaim(Task 4)、四条验收(Task 7)、PX 依据(设计要点)、"不做"清单(Global Constraints) 均有对应。
- Placeholder：无 TODO/TBD；测试均含具体代码与断言。
- 类型一致：`acquire/refresh/release`、`post_message/read_next_after_cursor/advance_cursor/post_notify/ensure_group/finalize_and_release`、`_handle_notify`/`_reclaim_pending`、`_finalize_mailbox_run`/`_lock_refresher(... abort_evt, proc_task)`/`_enqueue_via_mailbox` 跨 Task 命名一致。
- 已知缺口（明确留后续，非本期）：通知流**持续** `XAUTOCLAIM`（Phase 0.1，本期只做启动时一次性 reclaim）；锁丢失时的完整 fencing（Phase 1，本期做到「abort + cancel + finalize no-op」）；续租失败 cancel 路径不发 `run_end`（依赖 Phase 1 stuck-run watchdog）。
