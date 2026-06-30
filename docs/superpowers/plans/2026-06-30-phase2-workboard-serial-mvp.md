# Phase 2 看板版多主 Agent —— 串行 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在一个对话内引入"多个对等主 agent + 一块看板"的**串行接力**协作层(同一时刻全局只有一张 running 卡 / 一个主在干活),复用 Phase 0/1 的信箱/锁/join/续接机制。MVP 只做"多主 + 看板 + 串行协作"跑通;并行留作 MVP 之后的增量。

**Architecture:** 三层 = 看板层(卡片状态机 + 依赖 link)→ 多主 agent(从 Agent 角色库激活、各有真实 agent_id + 各自信箱)→ 各主私有子 agent(Phase 1 leaf,不变)。串行不变量:**全局 WIP=1**。主之间不直接通信,只通过看板交互(无环)。会话写只发生在收口(=primary 主),由 Phase 1 现有 `agent_lock` 串行,**不引入任何新会话写锁**。

**Tech Stack:** Python 3.11 / asyncio;Redis 5.0.14(SET NX PX + token + Lua;无 `(id`、无 XAUTOCLAIM);PostgreSQL + SQLAlchemy 2.0 async;ARQ worker;pytest(`asyncio_mode=auto`)。

## Global Constraints

- 不推翻 Phase 0/1 任何机制;改 Phase 1 代码处必须论证是责任拆分/扩展,不是替换,否则标 🔴。
- 子 agent 私有 leaf:不加任何寻址/message/spawn/认领能力(`agent/subagent.py:130`)。看板认领者只能是主 agent。
- 不做点对点 A2A;主之间只通过看板交互。
- 不引入中心编排 agent 做意图识别分派:主自己认领,依据 persona/description(`models.py:52`,已注入 prompt `agent/context.py:287-300`)。
- Redis 5.0.14:沿用 `_next_stream_id`(`bus/mailbox.py:20-27`)。
- 环境:venv `backend/.venv`;测试 Redis DB 15(`tests/conftest.py:90`)、PG `nanoresearch_test`(`tests/conftest.py:16`);`asyncio_mode=auto`(`pyproject.toml:144`);测试从 `backend/` 跑。
- 对外行为:看板是新增协作层,不破坏现有单主对话和 SSE 链路(`run_events:{run_id}` 不变)。
- 迁移:新表加进 `storage/models.py` → `backend/scripts/migrate_phase2_workboard.sql` → `storage/database.py:61-105` 的 `CHECKS` 加列 → `tests/conftest.py:51` 的 `truncate_all` 加表名。`Base.metadata.create_all`(`database.py:48-49`、`conftest.py:41`)自动建测试表。

---

## 前置结论(先评审这三条)

### 结论 ①:串行 MVP **不需要任何新会话写锁**;Phase 1 现有 `agent_lock`(在 agent_id 对齐后)就够。证。

**串行下会话写者的全集**(谁会对 `web:{conv}` 做 read-modify-write):
- **card-working 主 run** → Task 6 设为**会话只读**(原子 `LRANGE` 读基线,不 `save()`、不触发压缩),根本不写会话。证据:全量写在 `agent/loop.py:744/845` → `session/manager.py:190-192`,压缩副作用写在 `agent/memory.py:631/642`;只读模式两者都跳过。
- **收口 run**(= primary 主)→ 写会话(append 卡产出 + 综合 + run_end)。
- **被 defer 的用户消息 run**(= primary 主,见下)→ 写会话。
- **遗留单主交互 run** → 写会话(现状不变)。

**关键:这三类"真正写会话"的 run 全部走同一把锁 `agent_lock:{primary}:{conv}`**:
1. 用户消息进对话,路由到 primary 主信箱 `agent_inbox:{primary}:{conv}`(现状单主即路由到 `conv.agent_id`,`chat_router.py:439/480`),dispatcher 抢 `agent_lock:{primary}:{conv}` 才入队(`dispatcher.py:117-119`)。
2. 收口主 = primary(已拍板),续接抢的锁经 Task 1 对齐为 `agent_lock:{primary}:{conv}`(现状 `worker.py:328` 硬编码 `"none"`,对齐后用真实 primary id)。
3. 非 primary 主只跑 card-working(会话只读),它的 `agent_lock:{M_b}:{conv}` 只串行**它自己的信箱处理**,下面**没有任何会话写**,故不可能与 primary 的会话写竞争。

**用户点名的边角(收口 run vs 被 defer 的用户消息 run 是否两个并发会话写者)**:不是并发,二者被 Phase 1 现有机制串行——
- 协作round 进行中,用户消息到达 → dispatcher 的 `deferred_batch` 闸门(`dispatcher.py:125-129`)在"round 未交付"时 defer(Task 8 用 `board_round:{conv}` 标记接入该闸门),**用户消息 run 不启动**。
- 收口主先持 `agent_lock:{primary}:{conv}` 写会话 → run_end → finalize 释放锁 + re-notify(`worker.py:583-598` / `mailbox.py:81-93`)→ 用户消息 run 才被唤醒、再抢同一把 `agent_lock:{primary}:{conv}` → 写会话。
- 同一把锁 → 先后串行,零并发。这正是 Phase 1 今天的行为(agent_id 恒定 → agent_lock 串行会话写),串行 MVP 没有改变这一点。

**结论**:只要(a)card-working 会话只读、(b)收口 = primary、(c)用户消息路由到 primary 信箱、(d)round 进行中 defer 用户消息——则所有会话写funnel 到 `agent_lock:{primary}:{conv}` 一把锁,**与 Phase 1 完全同构,无需新锁**。并行版的 `session_write_lock`(其存在的唯一理由是"多主并行写会话 + 压缩副作用并发",见并行版结论①)在串行下前提不成立 → **整条砍掉**。

> 🔶 唯一前提是 Task 1 的 agent_id 对齐:不对齐时续接抢 `agent_lock("none")` 而用户消息走 `agent_lock:{primary}`,会是两把锁 → 才需要外部锁。对齐后同把锁,免新锁。所以 Task 1 是结论①成立的地基。

### 结论 ②:串行下收口 quiescence **没有时序竞态**。

并行版的竞态根源:多张卡并发流转,quiescence 快照读到"无 running 卡"的瞬间,另一张卡正好 running→done 或 promote 出新 ready 卡 → 快照失效 → 误判/重复 fire。

串行下 **全局 WIP=1**:任一时刻最多一张 running 卡。"是否安静"的判定**只在唯一那张 running 卡完成的那一步**做出,此刻**没有第二张卡在流转**(WIP=1)。完成卡 + 判定下一步("还有可认领的 ready 卡就接力;否则 quiesce 收口")发生在同一串行 actor 内,读到的是它自己刚改完、无并发改动的板面 → 一致读、无窗口。仍保留一个**幂等键** `collector_lock:{conv}`(continuation_lock 同构,`redis_keys.py:85`)防"完成路径重试两次重复 fire",但**不需要并行版的 card `revision` 乐观并发 + 原子 try_claim_under_concurrency**那套。

### 结论 ③:epoch 终止可大幅简化为"一个 round 标记 + 幂等键 + 成本上限";epoch 计数器整套砍掉。

并行版需要 epoch + delivered_epoch + per-card epoch 标记,是因为**多 round 可能重叠**(并行)。串行下 round 不重叠:新用户轮在当前 round 交付前被 defer(结论①(d))→ 任一时刻最多一个 round 活跃 → 用一个**布尔标记 `board_round:{conv}`**(round 开始 SET、收口交付 DEL)即可区分"round 内/外",不需要单调 epoch 计数器。

- **晚到 late-drop**:串行下收口在所有卡 done 后才 fire(quiescence = 无 running 卡),正常流程**不存在交付后才完成的卡**。"晚到"只剩 crash/watchdog 回收这一边角 → NO_REPLY 退化为"`collector_lock` 已存在/round 已 DEL 时,迟到的 fire 静默丢弃"(复用 Phase 1"续接已 run_end 不再触发"幂等,`worker.py:601-606`),无需 delivered_epoch 比较。
- **收敛保证仍需保留**:successor 卡可能无限自生 → 保留 `MAX_CARDS_PER_ROUND`(默认 50)+ `MAX_SUCCESSOR_DEPTH`(默认 8)双上限,触顶强制 quiesce。
- **砍掉**:monotonic epoch、delivered_epoch、per-card epoch 列。**保留**:`board_round:{conv}` 标记、`collector_lock:{conv}` 幂等、两个 cost cap。

---

## 相比并行版,串行 MVP 砍掉/简化了什么(证明确实更简单)

| 并行版条目 | 串行 MVP | 依据 |
|---|---|---|
| 结论① + Task 2 整个 `session_write_lock:{conv}` | **砍掉** | 结论①:串行无并发会话写 |
| Task 6 per-agent WIP + 全局 N + token 成本平衡 | **简化为全局 WIP=1** | 串行定义 |
| Task 8 并发完成 quiescence 竞态处理 + card `revision` 乐观并发 | **砍掉竞态处理,简化为 status CAS** | 结论②:WIP=1 无并发窗 |
| Task 9 epoch + delivered_epoch + per-card epoch 标记 | **砍掉,简化为 `board_round` 标记 + `collector_lock` 幂等** | 结论③:round 不重叠 |
| card-working "抑制压缩防并发撞车" | **保留会话只读,但理由改为"只读不变量",非防并发** | 见 Task 6 评估 |
| 并行 claim 的 thundering-herd / 多卡并发认领 | **砍掉,一次只 offer 一张活跃卡** | 串行定义 |

**保留(不砍)**:agent_id 对齐(Task 1)、成员表(Task 2)、卡片表+状态机(Task 3)、依赖 link+promote(Task 4)、串行认领(Task 5)、card-working 产出落卡(Task 6)、收口单写(Task 7)、终止判据简化版(Task 8)、作用域调整(Task 9)。

---

## File Structure

**新建**
- `backend/nanoresearch/bus/workboard.py` — 看板 Redis 原语(claim/heartbeat/release;claim token 复用 `dist_lock`;`board_round`/`collector_lock` 标记)。与 `bus/mailbox.py` 同层同风格。
- `backend/nanoresearch/storage/repositories/workboard_repo.py` — `WorkboardRepository`(卡片 CRUD、status CAS 转移、promote、串行 quiescence 查询、collector 认领)。参照 `run_repo.py`。
- `backend/scripts/migrate_phase2_workboard.sql` — 建 `conversation_agents` / `workboard_cards` / `workboard_card_links` 三表。
- `backend/tests/integration/test_phase2_workboard.py` — 看板集成测试(真 Redis+PG,沿用 `test_phase1_subagent_return.py` fixture)。
- `backend/tests/unit/test_phase2_agent_id_alignment.py` — agent_id 一致性单测。

**修改**
- `bus/redis_keys.py:95` 前 — 加 `workboard_claim(card_id)`、`workboard_notify`、`board_round(conv)`、`collector_lock(conv)`。
- `storage/models.py:88` 后 — 加 `ConversationAgent` / `WorkboardCard` / `WorkboardCardLink`。
- `storage/database.py:105` — `CHECKS` 加新表列。
- `agent/subagent.py:68,283` — `set_run_context` 加 agent_id;join 用真实 agent_id。
- `worker.py:328,344-345,507-518` — 续接抢锁 agent_id 对齐;收口 drain 卡产出。
- `agent/loop.py:262,718,744,784,845` — 透传 agent_id;card-working 会话只读。
- `server/routers/chat_router.py:450-487,429-447` — `_build_run_payload` 加 agent_id;多主激活读成员表。
- `bus/dispatcher.py:120-129` — `deferred_batch` 闸门接入 `board_round`。
- `heartbeat/stuck_run_watchdog.py:110` — agent_id 对齐 + 看板 stale 卡回收。
- `agent/memory.py:189-193,354-362`、`agent/context.py:69` — 摘要 conversation 作用域;L2/L3 conversation 过滤。
- `tests/conftest.py:51` — `truncate_all` 加新表。

---

## Task 1: agent_id 全程对齐(地基;结论①成立的前提;修现存 bug)

**Files:**
- Modify: `agent/subagent.py:68`(`set_run_context`)、`:283`(`continuation_lock`)、`:300-305`(续接 payload)
- Modify: `agent/loop.py:262`(`set_run_context` 调用)
- Modify: `heartbeat/stuck_run_watchdog.py:110`(`continuation_lock` agent_id)
- Modify: `worker.py:328`(续接抢 `agent_lock` 用真实 agent_id)、`:344-345`(超时 re-notify inbox/cursor agent_id)、`:595`(finalize 已 agent_id-aware,核对)
- Modify: `server/routers/chat_router.py:450`(`_build_run_payload` 加 `agent_id`)
- Test: `tests/unit/test_phase2_agent_id_alignment.py`

**Interfaces:**
- `SubagentManager.set_run_context(self, conversation_id: str | None, agent_id: str | None = None) -> None`
- `_build_run_payload(factory, conversation_id, uid, content, run_id, *, agent_id: str | None = None) -> dict`(`agent_id=None` 退回现有 `conv.agent_id` 行为,向后兼容)
- 续接路径用的 agent_id = owning-main 的真实 id(收口路径 = primary `conv.agent_id`)。

**现状不一致(已是潜在 bug)**:dispatcher 查 `continuation_lock(real_agent_id, conv)`(`dispatcher.py:115/125`),join 设 `("none", conv)`(`subagent.py:283`)→ `conv.agent_id` 非空时闸门看不到。Task 1 同时修这个。

**TDD 红→绿测试要点:**
- [ ] `test_continuation_lock_uses_real_agent_id`:`set_run_context(conv, agent_id="A")` 后 `_report_and_join`(沿用 `test_phase1_subagent_return.py:121-157` 的 `_subagent_mgr`+`_FakeArqPool`)→ 置 key == `continuation_lock("A", conv)`、payload `agent_id=="A"`。先红。
- [ ] `test_dispatcher_gate_sees_continuation_lock_for_real_agent`:置 `continuation_lock("A", conv)`,用 `agent_inbox("A", conv)` 构造 notify → `dispatcher._handle_notify` 返 `"deferred_batch"`(`dispatcher.py:129`)。先红。
- [ ] `test_build_run_payload_threads_explicit_agent_id`:`agent_id="A"` → `payload["agent_id"]=="A"`;`None` 退回 `conv.agent_id`(不回归 `test_phase1_subagent_return.py:30-40`)。
- [ ] `test_agent_id_roundtrip_invariant`:`_parse_inbox_key(agent_inbox(aid,conv))[0]==aid`,守住 inbox/lock/continuation_lock 三 key 同源。
- [ ] Commit:`fix(phase2): thread real agent_id through continuation/join path`

---

## Task 2: `conversation_agents` 成员表 + 多主激活

**Files:**
- Modify: `storage/models.py:88`(后加 `ConversationAgent`,照抄 `AgentKnowledgeBinding` `models.py:61-70`)
- Create: `backend/scripts/migrate_phase2_workboard.sql`(本表)
- Modify: `storage/database.py:105`(`CHECKS` 加 `("conversation_agents","agent_id")`)
- Modify: `storage/repositories/conversation_repo.py:176`(成员方法)
- Modify: `server/routers/chat_router.py`(激活端点 + 读成员)
- Modify: `tests/conftest.py:51`(`truncate_all` 加 `conversation_agents`)
- Test: `tests/integration/test_phase2_workboard.py`(成员部分)

**Interfaces:**
- ORM `ConversationAgent(conversation_id: UUID [PK,FK ondelete CASCADE], agent_id: UUID [PK,FK ondelete CASCADE], role: str = "main", activated_at: datetime)`
- `ConversationRepository.activate_agents(conv_id, agent_ids: list[UUID]) -> None`
- `ConversationRepository.list_member_agents(conv_id) -> list[Agent]`
- `ConversationRepository.is_member(conv_id, agent_id) -> bool`

**设计(已拍板)**:与 `Conversation.agent_id` 单 FK(`models.py:79`)共存——`agent_id` = primary 主(默认主 + 收口归属);成员表附加多主集合。默认激活 = `{conv.agent_id}`(单主不变),显式 `activate_agents` 才进多主,**不默认激活全部 Agent**。与 `session_key` 唯一约束(`models.py:75`)无冲突(一对话仍一 session_key)。

**TDD 红→绿测试要点:**
- [ ] `test_activate_and_list_member_agents`:activate 两 agent → list 返 2、`is_member` 正确。
- [ ] `test_single_main_default_membership`:未 activate 时退回 `{conv.agent_id}`,不破坏单主 `_build_run_payload`。
- [ ] `test_cascade_delete`:删 conversation → 成员行级联删。
- [ ] Commit:`feat(phase2): conversation_agents membership + activation`

---

## Task 3: `workboard_cards` 表 + 状态机 + `WorkboardRepository`

**Files:**
- Modify: `storage/models.py`(加 `WorkboardCard`)
- Modify: `backend/scripts/migrate_phase2_workboard.sql`、`storage/database.py:105`、`tests/conftest.py:51`
- Create: `storage/repositories/workboard_repo.py`
- Test: `tests/integration/test_phase2_workboard.py`(状态机部分)

**Interfaces:**
- ORM `WorkboardCard`:`id: UUID [PK]`、`conversation_id: UUID [FK ondelete CASCADE, index]`、`title: str`、`spec: Text`(指令/带产出)、`status: str default "backlog"`(`backlog|todo|ready|running|done|blocked`,`index`)、`owner_agent_id: UUID|None [FK agents.id ondelete SET NULL]`、`target_agent_id: UUID|None`(创建主建议的下一手,见 Task 5)、`claim_token: str|None`、`claimed_at/heartbeat_at: datetime|None`、`artifacts: JSONB default list`、`result: Text|None`、`created_by_agent_id: UUID|None`、`depth: int default 0`(successor 深度,Task 8 cost cap)、`created_at/updated_at`。
- `WorkboardRepository`:`create_card(...) -> WorkboardCard`、`get(card_id)`、`list_by_conversation(conv_id, statuses: set[str]|None=None) -> list[WorkboardCard]`、`transition(card_id, *, expect_status: str, to_status: str, **fields) -> bool`(**status CAS**:`WHERE id=? AND status=expect`,改成功返 True,竞争/非法返 False)。

**串行简化**:并行版的 `revision` 乐观并发列**砍掉**;`transition` 用 status CAS 足矣(WIP=1 无并发改同卡;CAS 仅防 watchdog/重试)。合法转移:`backlog→todo`、`todo→ready`、`ready→running`、`running→done|blocked`、`running→ready`(release 退回)。

**TDD 红→绿测试要点:**
- [ ] `test_create_card_defaults_backlog`。
- [ ] `test_transition_status_cas`:`transition(ready→running)` 首次 True,在已 running 上再 `expect=ready` 返 False。
- [ ] `test_illegal_transition_rejected`:`done→running` 返 False、状态不变。
- [ ] `test_list_by_conversation_filtered`:按 `{"ready","running"}` 过滤正确。
- [ ] Commit:`feat(phase2): workboard_cards + state machine + WorkboardRepository`

---

## Task 4: 依赖 link + promote(父全 done 放行子)

**Files:**
- Modify: `storage/models.py`(加 `WorkboardCardLink`)
- Modify: `migrate_phase2_workboard.sql`、`storage/database.py:105`、`tests/conftest.py:51`
- Modify: `storage/repositories/workboard_repo.py`(promote)
- Test: `tests/integration/test_phase2_workboard.py`(依赖部分)

**Interfaces:**
- ORM `WorkboardCardLink(parent_card_id: UUID [PK,FK ondelete CASCADE], child_card_id: UUID [PK,FK ondelete CASCADE])`(复合主键,照抄 `models.py:61-70`)。
- `WorkboardRepository.link(parent_card_id, child_card_id) -> None`
- `WorkboardRepository.parents_all_done(child_card_id) -> bool`(无父 → True)
- `WorkboardRepository.promote_ready_children(done_card_id) -> list[UUID]`:对 done 卡每个 `todo` 子卡,若 `parents_all_done` 则 `transition(todo→ready)`,返被 promote 子卡 id。**这是 `mailbox.py:160` `SCARD==0` 的卡片维度等价物**(串行下决定"下一张认领谁")。

**串行注**:promote 在 running 卡完成那一步串行执行,无并发 promote,无需原子保护(status CAS 已足)。

**TDD 红→绿测试要点:**
- [ ] `test_no_parents_ready_eligible`。
- [ ] `test_child_blocked_until_all_parents_done`:两父,父1 done 子仍 todo;父2 done 子被 promote 到 ready。
- [ ] `test_promote_idempotent`:对已 ready 子卡再 promote 不重复(`expect_status="todo"` CAS 保证)。
- [ ] Commit:`feat(phase2): card dependency links + parents-all-done promote`

---

## Task 5: 串行认领(全局 WIP=1)+ claim token + 看板驱动

**Files:**
- Create: `bus/workboard.py`(claim/heartbeat/release)
- Modify: `bus/redis_keys.py`(`workboard_claim(card_id)`、`workboard_notify`)
- Modify: `storage/repositories/workboard_repo.py`(全局单卡校验)
- Modify: `bus/dispatcher.py:108-144`(看板 notify → 唤醒 target 主 run 去认领)
- Test: `tests/integration/test_phase2_workboard.py`(认领部分)

**Interfaces:**
- `RedisKeys.workboard_claim(card_id: str) -> str` → `"workboard_claim:{card_id}"`
- `bus.workboard.claim_card(redis, repo, *, card_id, agent_id, conv_id, px_ms=30_000) -> str|None`:① **全局 WIP=1** 校验(本 conv 无任何 `running` 卡,`repo.list_by_conversation(conv, {"running"})` 为空)② `dist_lock.acquire(workboard_claim(card_id))` 拿 token ③ `repo.transition(card, expect_status="ready", to_status="running", owner_agent_id=agent_id, claim_token=token, claimed_at=now)`。全过返 token,任一步败则释放返 None。
- `bus.workboard.heartbeat_card(redis, *, card_id, token, px_ms=30_000) -> bool`(= `dist_lock.refresh` + 更 `heartbeat_at`)
- `bus.workboard.release_card(redis, repo, *, card_id, token, to_status="ready") -> bool`(token 校验 `dist_lock.release` + `transition(running→to_status)`)

**串行简化(对照并行 Task 6)**:**砍掉** per-agent WIP + 全局 N + token 成本平衡;WIP 校验**收敛为"本 conv 是否已有 running 卡"单条**。claim token **保留**(防其他主误改已认领卡)。

**看板驱动 + 认领归属(评审需确认)**:看板一次只 offer 一张活跃卡。当卡变 ready(创建即 ready 或 promote),`workboard_notify` → dispatcher 唤醒该卡 `target_agent_id` 主的信箱 `agent_inbox:{target}:{conv}` 投一条 "认领此卡" 消息;被唤醒的 target 主 run 执行 `claim_card`(WIP=1 保证全局只一张成功),做 card-working(Task 6)。若全局已有 running 卡,该 notify 被 dispatcher 现有 `dropped_locked`/`deferred_batch` 路径自然 defer(`dispatcher.py:117-129`),做完上一张再轮到——**串行天然由现有 dispatcher 串行性 + WIP=1 保证**。`target_agent_id` 由**创建主**按成员 persona/registry(`context.py:287-300`,已在其 prompt)写卡时填 → 分派是 peer-to-board,**非中心编排 agent**;target 主 run 可 `release` 退回(pass)以重路由。

> 评审拍板项:`target_agent_id` 由创建主指定(推荐,确定 + 复用已注入 registry)vs 广播让各主 self-claim(MVP 更重)。默认取前者。

**TDD 红→绿测试要点:**
- [ ] `test_claim_moves_ready_to_running_with_token`:ready 卡 → claim 返非空 token、status running、owner 设对、`workboard_claim:{card}` 持该 token。
- [ ] `test_claim_rejects_when_a_card_already_running`:本 conv 已有 running 卡 → 再 claim 返 None(全局 WIP=1)。
- [ ] `test_heartbeat_then_release_returns_card`:claim → heartbeat 真 → release → 卡回 ready、锁释放。
- [ ] `test_claim_token_is_dist_lock_token`:claim token 即 `dist_lock` token(非自造)。
- [ ] Commit:`feat(phase2): serial claim (global WIP=1) + claim token`

---

## Task 6: card-working 产出落卡(会话只读)

**Files:**
- Modify: `worker.py:367-392`(`run_agent_job` 加 `_card_id`/`_card_token`)、`:539-553`(working 不发 run_end、产出落卡)
- Modify: `agent/loop.py:700-846`(card-working 会话只读:基线只读 `LRANGE`,跳过 `save()` `:744/845` 与压缩调度 `:718/784/846`)
- Modify: `storage/repositories/workboard_repo.py`(写产出)
- Test: `tests/integration/test_phase2_workboard.py`(working 部分)

**Interfaces:**
- `run_agent_job(..., _card_id: str|None=None, _card_token: str|None=None)`:`_card_id` 非空 = card-working → run 末 `WorkboardRepository.attach_result(card_id, result, artifacts, token=_card_token)` 落卡、`transition(running→done)`、promote 子卡(Task 4)、**不写会话、不发 run_end**。
- `WorkboardRepository.attach_result(card_id, result: str, artifacts: list, *, token: str) -> bool`(token 不符返 False)
- `AgentLoop.process_direct(..., session_readonly: bool=False)`:`True` 时基线读用 `get_or_create`(只读),跳过 `sessions.save`(`loop.py:744/845`)与持久压缩调度(`loop.py:784/846` 的 `maybe_consolidate_by_tokens`)。
- `WorkboardRepository.create_card(...)`:`spec` 超 `WORKBOARD_MAX_SPEC_CHARS`(默认 16000)→ 拒绝/截断(防单卡指令本身爆窗)。
- card-working run 失败(provider 超窗 / `stop_reason in {"error","tool_error"}`)→ `transition(running→blocked)` + 落 error 到卡 `result`(板面不挂起,Task 8 watchdog/收口据 blocked 处理)。

**评估:card-working 抑制压缩串行下还需不需要?——抑制"持久压缩"需要(守只读不变量),但要补"基线 ephemeral 收口"避免爆窗。**
- **为什么仍抑制持久压缩**:card-working 是**会话只读不变量**——非收口 run 绝不能 mutate 共享会话(否则其内容先于收口进对话、破坏"收口单写")。持久压缩 = `LTRIM`(`memory.py:631`)+ `save()`(`memory.py:642`)会写共享 `session:msg`,故跳过。
- **爆窗边角(用户点名)实证**:`maybe_consolidate_by_tokens`(`loop.py:784/846`)只裁**先前会话历史**作基线,不裁**本轮 run 内累积的工具结果**;`_run_agent_loop`(`loop.py:409-418`)把迭代交给 `runner.run`,只传 `max_iterations`(`loop.py:413`),**无 in-loop token 预算**。故"单卡太大爆窗"**真实存在**,但它是**今天每个 run 都有的既存属性**(单主 run 同样),card-working 不在 in-run 轴上恶化。每条工具结果已 char-cap(`_TOOL_RESULT_MAX_CHARS`,`loop.py:890-891`),爆窗时 provider 报错 → `stop_reason=="error"`(`loop.py:423`)→ run 优雅失败、不污染状态。
- **MVP 兜底(三条,都不持久)**:① card-working 的**基线** ephemeral 收口——在内存里按现有 token 估算把过旧的 baseline 历史从**本 run 工作消息**裁掉(传给 `build_messages` 的 `history` 截断),**不调 `save()`/`LTRIM`**(共享会话不动,持久压缩仍归收口/primary `memory.py:553`);② `WORKBOARD_MAX_SPEC_CHARS` 卡 spec 上限;③ run 爆窗 → 卡 `blocked`(上面接口),board 不挂。

**TDD 红→绿测试要点:**
- [ ] `test_card_working_writes_card_not_session`:card-working run → 卡 result 写、status done、`session:msg` 无新增、`run_events` 无 `run_end`。
- [ ] `test_card_working_is_session_readonly`:monkeypatch `sessions.save` 抛错 → run 仍成功(证明没调 save);`get_or_create` 被调。
- [ ] `test_attach_result_token_guarded`:错 token 返 False、卡不变。
- [ ] `test_card_done_promotes_ready_child`:working run 完成 → 子卡被 promote 到 ready(接 Task 4)。
- [ ] `test_card_baseline_ephemeral_trim_does_not_persist`:给一个超长共享 baseline,card-working run 内 history 被 ephemeral 截断到预算内,但 `session:msg` 长度不变(未持久压缩)。
- [ ] `test_card_working_overflow_marks_blocked`:stub runner 返回 `stop_reason="error"`(模拟超窗)→ 卡 `transition(running→blocked)`、error 落卡 `result`、board 无 running 卡残留。
- [ ] `test_create_card_rejects_oversized_spec`:`spec` 超 `WORKBOARD_MAX_SPEC_CHARS` → 拒绝/截断。
- [ ] Commit:`feat(phase2): card-working run mode (produce-to-card, session read-only)`

---

## Task 7: 收口单写(一轮卡都 done → primary 综合 → 写对话 → run_end)

**Files:**
- Modify: `storage/repositories/workboard_repo.py`(串行 quiescence 查询 + collector 认领)
- Modify: `bus/redis_keys.py`(`collector_lock(conv)`)
- Modify: `worker.py:507-518`(续接分支扩展为收口:drain 卡产出而非 `subagent_results`)
- Modify: 卡完成路径(`worker.py` card-working 收尾或 `agent/subagent.py`)→ 判 quiescence → fire collector
- Test: `tests/integration/test_phase2_workboard.py`(收口部分)

**Interfaces:**
- `WorkboardRepository.is_board_quiesced(conv_id) -> bool`:存在 ≥1 未收口 `done` 卡 且 无 `ready|running` 卡 且 无可 promote 的 `todo` 卡。**串行下读到的是 settled 板面(结论②),无需原子快照。**
- `WorkboardRepository.try_claim_collector(conv_id) -> bool`:幂等——`collector_lock:{conv}` SET NX 成功(`dist_lock.acquire`)且 `is_board_quiesced` 真 → True;否则 False。
- `WorkboardRepository.list_done_cards_for_collection(conv_id) -> list[WorkboardCard]`
- 收口 run = `run_agent_job` 续接分支(`worker.py:510-517`)扩展:抢 `agent_lock:{primary}:{conv}`(Task 1 对齐后,**复用 Phase 1 `_continuation_acquire`,无新锁** —— 结论①)→ drain done 卡 `result/artifacts` append 进会话(类比 `_continuation_drain_and_append` `worker.py:352-364`,数据源换成卡片)→ `process_direct` 综合 → 发 `run_end` → DEL `board_round:{conv}`(Task 8)。

**收口由谁/何时/run_id(已拍板 + 结论)**:
- **何时**:`is_board_quiesced` 真(串行,无竞态)。
- **谁**:= **primary**(`conv.agent_id`,已拍板)。
- **run_id**:复用本 round 触发 run 的原 `run_id`(SSE 连续,沿用 `subagent.py:303`);watchdog 路径新建有效 run_id(沿用 `stuck_run_watchdog.py:115-123`)。
- **幂等**:`collector_lock:{conv}` 保证同 round 只一个收口。

**TDD 红→绿测试要点:**
- [ ] `test_quiesced_fires_collector_once`:3 卡全 done、无 ready/running → `try_claim_collector` 首次 True、二次 False。
- [ ] `test_not_quiesced_when_card_running`:有 running 卡 → False。
- [ ] `test_not_quiesced_when_promotable_todo_exists`:有父全 done 的 todo 卡 → False(先 promote 接力,不收口)。
- [ ] `test_collector_drains_cards_and_writes_session`:收口 run drain done 卡 append 进 `session:msg`、持 `agent_lock:{primary}:{conv}`、发 `run_end completed`、复用原 run_id。
- [ ] Commit:`feat(phase2): collector single-writer merge (primary, serial quiescence)`

---

## Task 8: 终止判据(简化版:round 标记 + 幂等 + cost cap)

**Files:**
- Modify: `storage/repositories/workboard_repo.py`(`board_round` 设/清、cost cap 校验)
- Modify: `bus/redis_keys.py`(`board_round(conv)`)
- Modify: `bus/dispatcher.py:120-129`(`deferred_batch` 闸门加 `board_round` 检查)
- Modify: `heartbeat/stuck_run_watchdog.py`(看板 stale 卡回收 + 强制 quiesce 兜底)
- Test: `tests/integration/test_phase2_workboard.py`(终止部分)

**Interfaces:**
- `RedisKeys.board_round(conversation_id) -> str` → `"board_round:{conversation_id}"`
- `bus.workboard.begin_round(redis, conv_id, px_ms) -> None`(首张卡 ready/claim 时 SET;需周期 refresh,沿用 `_lock_refresher` `worker.py:286-314` 思路,round 跨度可能 > 120s)
- `bus.workboard.end_round(redis, conv_id) -> None`(收口发完 run_end 后 DEL `board_round` + DEL `collector_lock`)
- `WorkboardRepository.can_create_successor(conv_id, parent_depth: int) -> bool`:`count(本 round 卡) < MAX_CARDS_PER_ROUND(默认50)` 且 `parent_depth+1 <= MAX_SUCCESSOR_DEPTH(默认8)`;超限返 False → 创建主不再生后继 → 板面走向 quiesce。
- dispatcher 闸门:`deferred_batch` 现有 `pending or continuation_lock`(`dispatcher.py:125-129`)→ 加 `or exists(board_round:{conv})`,使 round 进行中用户消息被 defer(结论①(d))。

**串行简化(对照并行 Task 9)**:**砍掉** monotonic epoch / delivered_epoch / per-card epoch;**保留** `board_round` 布尔标记(round 内/外)、`collector_lock` 幂等、两 cost cap。晚到 late-drop 退化为:迟到 fire 时 `collector_lock` 已存在或 `board_round` 已 DEL → 静默丢弃(NO_REPLY 等价,复用 `worker.py:601-606` 幂等)。

**TDD 红→绿测试要点:**
- [ ] `test_user_msg_deferred_during_round`:SET `board_round(conv)` → dispatcher `_handle_notify`(用户消息)返 `"deferred_batch"`;`end_round` 后再 notify → `"enqueued"`。
- [ ] `test_late_fire_after_end_round_is_dropped`:`end_round` 后(`board_round`/`collector_lock` 已清)迟到 `try_claim_collector` → False、无新 `run_end`。
- [ ] `test_max_cards_per_round_forces_quiesce`:`MAX_CARDS_PER_ROUND=3` → 第 4 张后继 `can_create_successor` False → 板面可走向 quiesce。
- [ ] `test_watchdog_reaps_stale_running_card`:running 卡心跳过期 → watchdog release/blocked → 板面不永久挂起(扩展 `stuck_run_watchdog.py:69-126` 到卡片维度)。
- [ ] Commit:`feat(phase2): serial termination (board_round flag + idempotent collector + cost caps)`

---

## Task 9: 作用域调整(共用摘要 conversation 作用域 + L2/L3 conversation 过滤)

**Files:**
- Modify: `agent/memory.py:189-193`(`MemoryStore` 落点)、`:354-362`(写 user_memory 加 conversation 维度)
- Modify: `agent/context.py:69`(history 召回加 conversation 过滤)
- Modify: `agent/conversation_knowledge_extractor.py`(抽取写入加 conversation 维度)
- Test: `tests/unit/`(作用域单测)

**Interfaces:**
- `MemoryStore(workspace, knowledge_search=None, *, conversation_id: str|None=None, agent_id: str|None=None)`:`conversation_id` 非空 → 摘要落 `workspace/conversations/{conversation_id}/memory`(共用一份),**不再按 agent_id 分目录**(消除 `memory.py:189-193` 按主分叉)。
- `KnowledgeSearch.write_user_memory_sync(items, *, uid, conversation_id=None)` 与 `search_user_memory_sync(query, *, uid, conversation_id=None)`:写带 `conversation_id` metadata,召回在现有 uid 过滤(`context.py:69`)上加 conversation 维度。

> 调作用域,非重做:摘要"按 agent_id 分叉"→"按 conversation 共用";L2/L3"uid 宽"→"conversation"。`conversation_id=None` 退回现有行为。

**TDD 红→绿测试要点:**
- [ ] `test_shared_summary_not_forked_by_agent_id`:同 conversation 两 agent_id 压缩 → 写同一 `conversations/{conv}/memory/MEMORY.md`,不再生成两个 `agents/{aid}/memory`。
- [ ] `test_l2l3_scoped_to_conversation`:同 uid 两 conversation 各写知识 → `search_user_memory_sync(conversation_id=A)` 不返 B。
- [ ] `test_uid_scope_backward_compat`:`conversation_id=None` 退回 uid 作用域,不回归单主。
- [ ] Commit:`feat(phase2): conversation-scoped shared summary + L2/L3 filter`

---

## Self-Review

**1. 串行 MVP 保留清单覆盖**:
- agent_id 对齐 → Task 1 ✓ | 成员表 → Task 2 ✓ | 卡片表+状态机 → Task 3 ✓ | 依赖 link+promote → Task 4 ✓ | 串行认领(全局单卡)→ Task 5 ✓ | card-working 产出落卡 → Task 6 ✓ | 收口单写 → Task 7 ✓ | 终止判据简化版 → Task 8 ✓ | 作用域调整 → Task 9 ✓
- 已拍板:收口=primary → Task 7 ✓ | 默认激活 `{conv.agent_id}` → Task 2 ✓
- 三结论:①无新会话写锁(证)→ 结论① + Task 7 复用 agent_lock ✓ | ②quiescence 无竞态 → 结论② + Task 7 `is_board_quiesced` ✓ | ③epoch 简化 → 结论③ + Task 8 ✓

**2. 相比并行版砍掉的(证明更简单)**:`session_write_lock`(并行结论①+Task2 整条)、per-agent WIP+全局 N+成本平衡(并行 Task6)、并发完成 quiescence 竞态处理 + card `revision` 乐观并发(并行 Task8)、epoch+delivered_epoch+per-card epoch(并行 Task9)、并行 thundering-herd 认领。MVP 任务数 9(并行版 10),且每个保留任务的并发处理面显著缩小。

**3. Placeholder/类型一致性**:`workboard_claim(card_id)`/`collector_lock(conv)`/`board_round(conv)` 命名贯穿 Task 5/7/8 一致;`transition(expect_status,to_status)` Task 3/4/5 同签名;`session_readonly`/`_card_id`/`_card_token` Task 5→6→7 串接一致;`target_agent_id` Task 3 定义、Task 5 消费。代码体按"先不写代码/测试要点"显式要求省略(用户指令优先于 skill 默认)。

**需评审拍板项(不阻塞)**:
- Task 5 认领归属:`target_agent_id` 由创建主指定(推荐)vs 广播 self-claim。
- Task 8 cost cap 默认值(`MAX_CARDS_PER_ROUND=50`、`MAX_SUCCESSOR_DEPTH=8`)。
- Task 8 `board_round` 用独立标记(推荐)vs 复用 `continuation_lock` 语义。
