# 真协作触发 + self-claim 链路 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让多主协作在**真实对话**里发生:primary 主用工具拆卡 → 卡 ready 投目标主信箱 → dispatcher 唤醒 → **目标主在自己 run 里自主 claim/pass**(路B)→ 串行接力 → collector 复用 run_id 收口。

**Architecture:** 在已建的 Phase 2 serial-MVP 看板上,把"系统替主认领"(`_drive_board` claim+enqueue)改成"投目标主信箱 + 目标主自主认领",并加一个 `decompose_to_board` 工具作触发源。最大化复用 Phase 0 信箱+dispatcher、Phase 1 defer/collector、card-working/claim/watchdog;唯一改 Phase 0 的点是 dispatcher 加 `board_offer` vs 用户消息的 kind 判别。

**Tech Stack:** Python 3.11 / asyncio;Redis 5.0.14;PostgreSQL + SQLAlchemy 2.0 async;ARQ;pytest(`asyncio_mode=auto`)。

> **格式说明(用户既定偏好,优先于 skill 默认)**:本计划给 文件:行号 + 接口签名 + TDD 红→绿测试要点,**不写完整代码体**;评审通过后再进 TDD 实现。

## Global Constraints

- 设计依据:`docs/superpowers/specs/2026-07-01-real-collaboration-trigger-design.md`。
- 不引入中心编排 agent(拆卡=primary 参与者,认领=目标主自己);不做 A2A;子保持 private leaf(`agent/subagent.py:130`)。
- 不推翻 Phase 0/1;唯一改 Phase 0 点 = dispatcher kind 判别(Task 3),是扩展不是替换。
- 会话写者 ≤1(结论①):self-claim 的 card-working 会话只读;唯一会话写者是 collector(primary)。串行 WIP=1(一次只 offer 一张)。
- Redis 5.0:沿用 `bus/mailbox.py` 现有 inbox/notify 约定。
- 环境:venv `backend/.venv`;测试 Redis DB 15、PG `nanoresearch_test`;`asyncio_mode=auto`;测试从 `backend/` 跑。
- 入站消息约定:`board_offer` 消息 payload 带 `kind:"board_offer"` + `card_id`;普通用户消息无 `kind`。
- 迁移:模型 → `backend/scripts/migrate_phase2_workboard.sql` → `storage/database.py` CHECKS。

---

## File Structure

**新建**
- `backend/nanoresearch/agent/tools/workboard_plan.py` — `DecomposeToBoardTool`(primary 拆卡工具)。仿 `agent/tools/spawn.py`。
- `backend/scripts/phase2_e2e_collab.py` — 路B 端到端冒烟(真 LLM),命门断言。

**修改**
- `backend/nanoresearch/storage/models.py` — `WorkboardCard` 加 `pass_count`。
- `backend/nanoresearch/storage/repositories/workboard_repo.py` — `record_pass`、`reset_pass`(reroute 用)。
- `backend/nanoresearch/worker.py` — `_drive_board`→`_offer_next_or_collect`(offer via inbox);`run_agent_job` self-claim 分支(`_board_offer_card_id`);`_reroute_card`;primary defer-on-board_round。
- `backend/nanoresearch/bus/dispatcher.py:108-145` — `_handle_notify` 加 board_offer kind 判别。
- `backend/nanoresearch/agent/loop.py:186-231,255-282` — 注册 `DecomposeToBoardTool` + `set_tool_context` 注入 conv/uid/primary。
- `backend/scripts/migrate_phase2_workboard.sql` + `storage/database.py` CHECKS — `pass_count`。
- `backend/tests/integration/test_phase2_workboard.py` — 改 `_drive_board` 相关测试为 offer 语义 + 新测试。

---

## Task 1: `workboard_cards.pass_count` 列 + record/reset

**Files:**
- Modify: `storage/models.py`(`WorkboardCard` 加 `pass_count`)、`scripts/migrate_phase2_workboard.sql`、`storage/database.py`(CHECKS 加 `("workboard_cards","pass_count")`)
- Modify: `storage/repositories/workboard_repo.py`(`record_pass`/`reset_pass`)
- Test: `tests/integration/test_phase2_workboard.py`

**Interfaces:**
- ORM `WorkboardCard.pass_count: Mapped[int] = mapped_column(Integer, default=0)`。
- `WorkboardRepository.record_pass(card_id, passed_agent_id: str) -> int`:`pass_count += 1`,把 `{"passed": passed_agent_id}` append 进 `artifacts`,返回新 `pass_count`。
- `WorkboardRepository.reset_pass(card_id) -> None`(可选,collector 后清理;MVP 可省)。

**TDD 红→绿测试要点:**
- [ ] `test_create_card_pass_count_zero`:新卡 `pass_count == 0`。
- [ ] `test_record_pass_increments_and_logs`:`record_pass(card, "A")` → 返 1、`pass_count==1`、`artifacts` 含 `{"passed":"A"}`;再 `record_pass(card,"B")` → 返 2。
- [ ] Commit:`feat(collab): workboard_cards.pass_count + record_pass (Task 1)`

---

## Task 2: `_offer_next_or_collect` —— 卡 ready 投目标主信箱(替代 claim+enqueue)

**Files:**
- Modify: `worker.py`(`_drive_board` 重写为 `_offer_next_or_collect`)
- Modify: `worker.py` 调用点:card-working 完成 relay(现 `_finish_card_working` 后调 `_drive_board`)、collector 不变
- Modify: `heartbeat/stuck_run_watchdog.py:_scan_stale_cards`(调用名改)
- Test: `tests/integration/test_phase2_workboard.py`(改写 `test_drive_board_*` 为 offer 语义)

**Interfaces:**
- Produces: `worker._offer_next_or_collect(redis, repo, arq, conv_id, uid) -> str`,返回 `"wip_busy" | "offered:{card_id}" | "collect:{conv_id}" | "idle"`。
  - 有 running 卡 → `"wip_busy"`。
  - 取一张 ready 卡 `c` → `mailbox.post_message(redis, str(c.target_agent_id), conv_id, {"kind":"board_offer","card_id":str(c.id),"conversation_id":conv_id,"uid":uid})` + `mailbox.post_notify(redis, mailbox_key=RedisKeys.agent_inbox(str(c.target_agent_id),conv_id), cursor_key=RedisKeys.agent_inbox_cursor(...), lock_key=RedisKeys.agent_lock(str(c.target_agent_id),conv_id))` → `"offered:{c.id}"`。**不 claim、不 enqueue card-working。**
  - 无 ready 卡 + `try_claim_collector` 成功 → enqueue collector(复用现有,`_build_run_payload(agent_id=primary)` + `_collect=True`)→ `"collect:{conv_id}"`;否则 `"idle"`。
- Consumes: `bus/mailbox.py:post_message`(`:30`)/`post_notify`(`:37`)、`bus/workboard.try_claim_collector`、`_build_run_payload`、`RedisKeys.agent_inbox/agent_lock`。

**🔶 行为变更**:`_drive_board` 不再替主认领。旧测试 `test_drive_board_claims_ready_card_and_enqueues`/`_quiesced_enqueues_collector` 改写。card-working 完成后的 relay 改调 `_offer_next_or_collect`。

**TDD 红→绿测试要点:**
- [ ] `test_offer_posts_board_offer_and_leaves_card_ready`:ready 卡(target=A)→ `_offer_next_or_collect` → `agent_inbox(A,conv)` 里有一条 `kind=="board_offer"` + `card_id`;**卡仍 `ready`、`owner_agent_id` 仍 None**(证明没替它认领);返回 `"offered:{card}"`;`_FakeArqPool` 无 job(offer 不直接 enqueue)。
- [ ] `test_offer_wip_busy_when_running`:有 running 卡 → `"wip_busy"`、无投递。
- [ ] `test_offer_quiesced_enqueues_collector`:全 done 无 ready → `"collect:{conv}"`、collector job 入队(`_collect=True`)。
- [ ] Commit:`feat(collab): _offer_next_or_collect — offer card to target inbox, not claim (Task 2)`

---

## Task 3: dispatcher `board_offer` kind 判别(🔶 唯一改 Phase 0 点)

**Files:**
- Modify: `bus/dispatcher.py:108-145`(`_handle_notify` 调整为"先读条目→按 kind 分流")
- Test: `tests/integration/test_phase2_workboard.py`

**Interfaces:**
- `_handle_notify(fields) -> str` 返回值新增 `"enqueued_self_claim"`(board_offer 路径,供测试/telemetry)。
- 逻辑(调整顺序):抢锁 → `read_next_after_cursor` → 取 `payload`:
  - `payload.get("kind") == "board_offer"` → **放行(不看 board_round/pending/continuation_lock)**:`arq.enqueue_job("run_agent_job", **payload_for_run, _board_offer_card_id=payload["card_id"], _lock_key=lock_key, _lock_token=token, _entry_id=entry_id)` → `"enqueued_self_claim"`。(`payload_for_run` = 由 dispatcher 用 `_build_run_payload(agent_id=该信箱的 agent_id)` 重建 run 配置;或直接透传 offer 的 conv/uid + agent_id=该信箱主。实现取后者:`run_id` 新建、`agent_id` = 解析出的信箱 agent_id、`session_key=web:{conv}`、`content`= 占位。)
  - 否则(普通 turn)→ 现有闸门:`scard(pending) OR exists(continuation_lock(aid,conv)) OR exists(board_round(conv))` → 释放锁 `"deferred_batch"`;否则 enqueue 普通 `run_agent_job`(现状)→ `"enqueued"`。
- Consumes: `mailbox.read_next_after_cursor`(`:45`)、`mailbox.finalize_and_release`(run 收尾,不变)。

**TDD 红→绿测试要点:**
- [ ] `test_dispatcher_board_offer_bypasses_round_gate`:`board_round(conv)` 已置 + `agent_inbox(A,conv)` 有 board_offer → `_handle_notify` 返 `"enqueued_self_claim"`,job kwargs 含 `_board_offer_card_id`、`agent_id=="A"`。
- [ ] `test_dispatcher_user_turn_still_deferred_in_round`:`board_round` 置 + 普通用户 turn → `"deferred_batch"`(回归,Task 8 of Phase 2 行为不变)。
- [ ] `test_dispatcher_user_turn_enqueues_when_idle`:无 board_round/pending → 普通 turn `"enqueued"`(回归)。
- [ ] Commit:`feat(collab): dispatcher board_offer kind distinction (Task 3)`

---

## Task 4: self-claim run 分支(目标主 run 内 claim/pass)

**Files:**
- Modify: `worker.py:run_agent_job`(新增 `_board_offer_card_id` 分支)
- Test: `tests/integration/test_phase2_workboard.py`

**Interfaces:**
- `run_agent_job(..., _board_offer_card_id: str | None = None)`:非空 → self-claim 分支:
  1. `card = WorkboardRepository(factory).get(_board_offer_card_id)`;若 `card is None` 或 `card.status != "ready"` → finalize 信箱锁、return(幂等)。
  2. 建 loop(payload 已带 target persona,经 dispatcher 的 `_build_run_payload(agent_id=target)`)。
  3. `claim = await _judge_claim(loop, card, agent_id)`(见下;返回 bool)。
  4. `claim` True → `token = await workboard.claim_card(redis, repo, card_id=card.id, agent_id=uuid(agent_id), conv_id=conversation_id)`;`token` 非空 → 复用 card-working(`process_direct(session_readonly=True)` + `_finish_card_working` + 完成调 `_offer_next_or_collect`);`token` None(WIP 忙/竞争)→ 不动卡、return。
  5. `claim` False(pass)→ `await _reroute_card(redis, repo, ctx["arq_pool"], conversation_id, uid, card, passed_agent_id=agent_id)`(Task 5)。
- Produces: `worker._judge_claim(loop, card, agent_id) -> bool`:一次轻量 LLM 调用(用 `loop.provider.chat_with_retry` + 一个 claim/pass 结构化判断 prompt:输入卡 `title/spec` + 该主 persona,输出 claim 或 pass)。**单独函数便于测试 monkeypatch。**
- Consumes: Task 2 `_offer_next_or_collect`、`workboard.claim_card`、现有 `_finish_card_working`/card-working、Task 5 `_reroute_card`。

**TDD 红→绿测试要点(monkeypatch `_judge_claim` 避开真 LLM):**
- [ ] `test_self_claim_judge_claim_runs_card_working`:monkeypatch `_judge_claim`→True,stub `loop.process_direct`/`_run_agent_loop`(避重型 loop,参照 Phase2 的 `__new__`/stub 手法或直接测 `_finish_card_working` 已覆盖)→ 卡 `running→done`、owner=target、claim 锁置过。
- [ ] `test_self_claim_judge_pass_reroutes`:monkeypatch `_judge_claim`→False → 调 `_reroute_card`(monkeypatch 验被调 + 卡未 claim)。
- [ ] `test_self_claim_idempotent_when_not_ready`:卡已 `done` → 分支幂等 return,不动卡。
- [ ] Commit:`feat(collab): self-claim run branch (judge claim/pass in target's own run) (Task 4)`

---

## Task 5: pass 重路由 + 上限 + fallback primary

**Files:**
- Modify: `worker.py`(`_reroute_card`)
- Test: `tests/integration/test_phase2_workboard.py`

**Interfaces:**
- `worker._reroute_card(redis, repo, arq, conv_id, uid, card, passed_agent_id: str) -> str`:
  1. `n = await repo.record_pass(card.id, passed_agent_id)`(Task 1)。
  2. `members = await ConversationRepository(repo._factory).list_member_agents(uuid(conv_id))`;`primary` = conv.agent_id。
  3. `tried = {passed agents from card.artifacts}`;`candidates = [m for m in members if str(m.id)!=primary and str(m.id) not in tried]`。
  4. 有候选 → 设 `card.target_agent_id = candidates[0].id`(`transition` 不改 status,用一个 `repo.set_target(card_id, agent_id)` 或直接 update)→ `_offer_next_or_collect`(重新 offer)→ `"rerouted:{cand}"`。
  5. 无候选(全试过)或 `n >= len(members)` → **fallback**:`card.target_agent_id = primary` → `_offer_next_or_collect` → `"fallback_primary"`。(primary 对自拆卡的 `_judge_claim` 默认 claim。)
- Produces: `WorkboardRepository.set_target(card_id, agent_id) -> None`(update `target_agent_id`)。

**TDD 红→绿测试要点:**
- [ ] `test_reroute_offers_next_member`:2 主 A/B,卡 target=A,A pass → `_reroute_card(...,passed=A)` → target 变 B、重新 offer 到 B 信箱、`pass_count==1`。
- [ ] `test_reroute_fallback_primary_when_all_passed`:所有非 primary 主都 pass → target 变 primary、offer 到 primary 信箱、返回 `"fallback_primary"`。
- [ ] Commit:`feat(collab): pass reroute + cap + fallback primary (Task 5)`

---

## Task 6: `decompose_to_board` 工具(primary 拆卡触发)

**Files:**
- Create: `agent/tools/workboard_plan.py`(`DecomposeToBoardTool`)
- Modify: `agent/loop.py:186-231`(注册)、`:255-282`(`_set_tool_context` 注入 conv_id/uid/primary_agent_id)
- Test: `tests/integration/test_phase2_workboard.py`

**Interfaces:**
- `DecomposeToBoardTool(repo_factory, redis, arq_getter)`,`set_context(conversation_id, uid, primary_agent_id, agents_registry)`,`name="decompose_to_board"`,参数 `cards:[{title,spec,target_agent,depends_on:[int]}]`。
- `execute(cards) -> str`:
  1. 解析每个 `target_agent`(按名字/id 在 `agents_registry` 匹配)→ `target_agent_id`;匹配不到 → 返回错误串让 LLM 改。
  2. `can_create_successor` 数量校验。
  3. 建卡:无 `depends_on`→`status="ready"`,有→`status="todo"` + `link`;`target_agent_id`/`depth` 写入(复用 `create_card`/`link`)。
  4. `activate_agents(conv, 去重 target ids ∪ {primary})`。
  5. `begin_round(redis, conv)`。
  6. `_offer_next_or_collect(redis, repo, arq, conv, uid)`(offer 第一张 ready)。
  7. 返回回执串(如"已拆成 N 张,分派给 …")。
- Consumes: `WorkboardRepository`、`ConversationRepository.activate_agents`、`bus/workboard.begin_round`、Task 2 `_offer_next_or_collect`、`agents_registry`(`agent/context.py:287-300` 已在 prompt)。
- 注册条件:web 平台 + 有 arq(后台)能力时暴露(参照 spawn 仅后台场景)。

**TDD 红→绿测试要点(直接构造工具,不走完整 loop):**
- [ ] `test_decompose_creates_cards_and_links`:`execute(cards=[{研究,target:研究主,deps:[]},{写作,target:写作主,deps:[0]}])` → 研究卡 ready、写作卡 todo + link(研究→写作);两主被 `activate_agents`。
- [ ] `test_decompose_activates_and_offers_first`:执行后 `board_round` 置、第一张 ready 卡 offer 到 研究主信箱(board_offer)。
- [ ] `test_decompose_unknown_target_returns_error`:`target_agent` 匹配不到 registry → 返回错误串、不建卡。
- [ ] Commit:`feat(collab): decompose_to_board tool (primary decomposes task into cards) (Task 6)`

---

## Task 7: primary defer-on-board_round

**Files:**
- Modify: `worker.py:run_agent_job`(defer 判定,现 `_has_pending_subagents` `:551`)
- Test: `tests/integration/test_phase2_workboard.py`

**Interfaces:**
- 扩 defer 判定:`if await _has_pending_subagents(redis, session_key) or await redis.exists(RedisKeys.board_round(conversation_id)): 留 running、不发 run_end、return`(Phase 0 `finally` 仍释放信箱锁)。
- Produces: `worker._should_defer_run_end(redis, session_key, conversation_id) -> bool`(抽出,便于测试)。

**TDD 红→绿测试要点:**
- [ ] `test_should_defer_when_board_round_set`:`board_round(conv)` 置 → `_should_defer_run_end` True;清掉 → False(仅当 pending 也空)。
- [ ] `test_primary_run_defers_no_run_end_on_board_round`(集成,stub loop):primary run 跑完、`board_round` 在 → `run_events` 无 `run_end`、run 状态留 running。
- [ ] Commit:`feat(collab): primary defers run_end while a board round is in flight (Task 7)`

---

## Task 8: 路B 端到端冒烟(真 LLM,命门)

**Files:**
- Create: `backend/scripts/phase2_e2e_collab.py`(扩自 `phase2_e2e_smoke.py`)
- 手动可跑;非 pytest(真 LLM)。

**做什么:**
- 激活 研究主 + 写作主;给 primary 一条真任务;**不手动塞卡**——让 primary **真调** `decompose_to_board` 拆卡。
- 用 `_CapturingPool` 捕获 dispatcher/relay 的 enqueue,inline 驱动 self-claim run(真 loop、真 LLM、真 `_judge_claim`)。
- **路B 命门断言(关键)**:
  - 拆卡 + offer 之后、目标主 run 之前:第一张卡 `status=="ready"` 且 `owner_agent_id is None`(证明驱动器**没替它认领**)。
  - 目标主 self-claim run 之后:卡 `running/done` 且 `owner_agent_id == 目标主`(证明认领发生在**目标主自己的 run 里**)。
- 仍带结论①命门:card-working 期间 `session:msg` 长度不变。
- 末:用户收到 collector 综合答复(复用 run_id),`RESULT: ALL PASS`。自清理 `p2collab_*`。

**Steps:**
- [ ] 写脚本(参照 `phase2_e2e_smoke.py` 结构 + 上面命门断言)。
- [ ] 在 dev DB 跑(已建 workboard 表)→ 确认 `RESULT: ALL PASS` + 路B 命门 PASS。
- [ ] Commit:`test(collab): real-LLM e2e smoke proving 路B self-claim (Task 8)`

---

## Self-Review

**1. Spec coverage**(对 spec §4 组件):
- 4.1 decompose 工具 → Task 6 ✓ | 4.2 primary defer → Task 7 ✓ | 4.3 _offer_next_or_collect → Task 2 ✓ | 4.4 dispatcher kind 判别 → Task 3 ✓ | 4.5 self-claim run → Task 4 ✓ | 4.6 pass reroute → Task 5 ✓ | 4.7 collector → 复用(无新 Task)✓ | §10 pass_count → Task 1 ✓ | §8 测试(单元+e2e 命门)→ 各 Task test points + Task 8 ✓

**2. Placeholder scan**:无 TBD/TODO;每 Task 有具体文件:行号 + 签名 + 具名测试。代码体按用户既定偏好省略(优先于 skill full-code 默认)。

**3. Type consistency**:`_offer_next_or_collect(redis,repo,arq,conv_id,uid)` Task 2/5/6 一致;`_board_offer_card_id` Task 3 enqueue→Task 4 消费一致;`record_pass(card_id,passed_agent_id)->int` Task 1 定义→Task 5 用一致;`board_offer` payload `{kind,card_id,conversation_id,uid}` Task 2 产→Task 3 消费一致;`_judge_claim(loop,card,agent_id)->bool` Task 4 定义+消费一致。

**建议执行顺序**:1(数据)→ 2(offer)→ 3(dispatcher)→ 4(self-claim)→ 5(reroute)→ 6(拆卡工具)→ 7(primary defer)→ 8(e2e 命门)。1-7 每个独立可测;8 是真 LLM 总验收。
