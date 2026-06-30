# 真协作触发 + self-claim 链路 — Design Spec

**Status:** Approved design (brainstorm) — 2026-07-01. Next: implementation plan (writing-plans).
**Branch:** `feature/consolidation-compaction`
**Builds on:** Phase 2 serial-MVP workboard (commits `d1394a39`→`e331eaac`). 见 `docs/superpowers/plans/2026-07-01-phase2-workboard-serial-mvp.md`。

---

## 1. 目标与现状

**目标**:让"多个主 agent 在一个对话里真协作"在**真实对话**里发生——而不仅仅是机制跑通。

**现状(诚实陈述)**:
- Phase 2 serial-MVP 把"多主 + 看板 + 串行接力 + 收口"的**机制**全部建好并端到端验证(冒烟脚本 `backend/scripts/phase2_e2e_smoke.py`,真 LLM 真跑过)。
- **但真实对话里不会产生任何协作**:没有任何代码把"用户任务"变成"看板任务"。用户发一条消息走的还是 Phase 0/1 单主老路——一个主、一个 run、零卡片、零第二个 agent。看板只在冒烟脚本手动塞卡时才有内容。
- 而且 serial-MVP 取了一个捷径:`_drive_board`(`backend/nanoresearch/worker.py`)**替**目标主认领卡片(claim+enqueue)。这是"自动化编排",不是"协作"——目标主没有自主认领的动作。

**本设计补两样东西,使协作真发生**:
1. **拆卡触发**:primary 主把用户任务拆成看板卡片(它自己用一个工具拆,不引入中心编排 agent)。
2. **self-claim 链路(路B)**:卡片 ready 时不再由系统替主认领,而是**唤醒目标主、由它在自己的 run 里自主认领**(轻量 LLM 判断 claim/pass)。这是 Phase 2 真正的临门一脚。

---

## 2. 已锁定的决策(brainstorm 结论,勿再 litigate)

1. **谁拆卡** = **primary 主自己拆**,经一个 `decompose_to_board` 工具。拆卡者是参与的主,不是专职调度 → 不破"不引入中心编排 agent"约束,与 Task 5 拍板的"target_agent_id 由创建主指定"一脉相承。
2. **primary 收尾** = **复用 Phase 1 spawn 模式**:拆完卡 defer run_end、留 running;collector(=primary)**复用原 run_id** 发最终综合答复。用户看到一条连续 run。
3. **认领侧** = **路B(self-claim)**:卡 ready → 投目标主信箱 → dispatcher 唤醒目标主 → **目标主 run 内自主认领**。**不走路A**(系统替认领)。理由:路A 是编排不是协作;本轮出发点就是让协作真发生。
4. **认领决策** = **轻量 LLM 判断 claim/pass**(真 agency:能 pass / 重路由),否则塌回路A。
5. **串行形状** = **一次只 offer 一张卡**(完成再 offer 下一张),保证 WIP=1、不在忙时唤醒别人。**并行不在本期范围**——但本设计是并行的前置(并行 = 把"offer 一张给 target"扩成"offer 多张给多候选竞争",是扩展不是重做)。
6. **pass 出口** = 退回 ready + 重路由给下一个已激活主;带 pass 上限;超限 fallback 强制由 primary 执行。保证终止。

---

## 3. 架构总览

```
用户任务
  │  (走现有 _enqueue_via_mailbox → dispatcher → run_agent_job 单主 run,primary)
  ▼
primary 主 run (planning turn)
  │  LLM 判断:简单 → 正常答复结束;需多专长协作 → 调 decompose_to_board 工具
  ▼  decompose_to_board:
  │    建 cards(无依赖→ready, 有依赖→todo+link) + activate_agents + begin_round
  │    + offer 第一张 ready 卡(投 target 信箱)
  │  primary run: board_round 在 → defer run_end、留 running(复用 Phase 1 defer)
  ▼
[self-claim 循环 — 串行,一次一张]
  卡 ready → post board_offer 到 agent_inbox:{target}:{conv} + notify
    → dispatcher 唤醒 target 主 run(Phase 0,board_offer 不被 round 闸门 defer)
      → target run: 轻量 LLM 判断 claim / pass
          claim → claim_card(抢锁,WIP=1 兜底)→ card-working(会话只读,产出落卡)
                  → 完成 _finish_card_working(done + promote 子卡)→ offer 下一张 ready
          pass  → 退回 ready + pass_count++ → 重路由给下一个已激活主
                  (pass 超限 → fallback primary 执行)
  无 ready 卡 + quiesced → try_claim_collector(fire-once)
  ▼
collector run (primary,复用 planning run 的 run_id)
  抢 agent_lock:{primary}:{conv}(唯一会话写者)→ _collect_cards_into_session(合并各卡产出)
  → process_direct 综合 → run_end → end_round
  ▼
用户收到一条连续 run 的最终综合答复
```

---

## 4. 组件设计

每个组件标注 **复用** / **扩展** / **新建**,并给接口与文件落点。

### 4.1 `decompose_to_board` 工具(新建)
- **职责**:primary 主把任务拆成卡片放上看板,激活目标主,开 round,offer 第一张卡。是触发协作的唯一入口。
- **落点**:`backend/nanoresearch/agent/tools/workboard_plan.py`(新),仿 `agent/tools/spawn.py:SpawnTool` 的形态(工具持 conversation/run 上下文,经 `set_context` 注入)。在 `agent/loop.py:_register_default_tools`(`:186-231`)注册;仅对 web 平台 + 有激活成员能力时暴露。
- **接口(LLM 可见)**:
  ```
  name: decompose_to_board
  description: 仅当任务需要多个不同专长的主 agent 协作时使用;把任务拆成看板卡片分派给同伴主。
               简单问答不要用,直接回答。
  parameters:
    cards: [ {title: str, spec: str, target_agent: str(同伴主的名字/id,取自 Agent Registry),
             depends_on: [int]  # 本批内的卡片下标,父全 done 才放行} ]
  ```
- **执行逻辑**:
  1. 解析 `target_agent`(按名字/id 在 `agents_registry` 里匹配;已在 prompt,见 `agent/context.py:287-300`)。无法匹配 → 工具返回错误让 LLM 改。
  2. 受 cost cap:`WorkboardRepository.can_create_successor`(`workboard_repo.py`,MAX_CARDS_PER_ROUND=50)校验数量。
  3. 建卡:无 `depends_on` → `status="ready"`,有 → `status="todo"` + `link(parent, child)`(复用 `create_card`/`link`)。`depth` 按依赖层级。`target_agent_id` 写入卡。
  4. `activate_agents(conv_id, 去重的 target ids ∪ {primary})`(复用 `ConversationRepository.activate_agents`)。
  5. `begin_round(redis, conv_id)`(复用 `bus/workboard.py:begin_round`)。
  6. `_offer_next_or_collect`(见 4.3)offer 第一张 ready 卡。
  7. 工具返回给 primary LLM 一句回执(如"已拆成 N 张,分派给 研究主/写作主")。
- **依赖**:`WorkboardRepository`、`ConversationRepository`、`bus/workboard`、`agents_registry`。

### 4.2 primary run 的 defer(扩展)
- **职责**:primary 调完 `decompose_to_board` 后,这次 run 不收尾(把最终答复让给 collector,复用原 run_id)。
- **落点**:`backend/nanoresearch/worker.py:run_agent_job`,现有 Phase 1 defer 判定 `_has_pending_subagents`(`worker.py:551`)。**扩展**该判定:`if _has_pending_subagents(...) OR EXISTS(board_round:{conv}): 留 running、不发 run_end、return`(Phase 0 `finally` 仍释放信箱锁)。
- **复用**:Phase 1 的"defer run_end + collector 复用 run_id"整套(`subagent.py` 续接 enqueue 同构;collector 分支已在 `worker.py`,`_collect=True`)。
- **新建**:仅"board_round 也触发 defer"这一个条件。

### 4.3 `_offer_next_or_collect`:卡 ready → 投目标主信箱(扩展自 `_drive_board`)
- **职责**:串行驱动。取下一张 ready 卡,**不再 claim+enqueue**,改为**投 board_offer 到目标主信箱 + notify**;无 ready 卡且 quiesced → fire collector。
- **落点**:`backend/nanoresearch/worker.py:_drive_board` **重写**为 `_offer_next_or_collect(redis, repo, conv_id, uid)`:
  1. 有 `running` 卡 → return `"wip_busy"`(串行兜底)。
  2. 取一张 `ready` 卡 `c`:
     - `post_message(redis, c.target_agent_id, conv_id, {kind:"board_offer", card_id:str(c.id)})`(复用 `bus/mailbox.py:post_message` `:30`)。
     - `post_notify(redis, mailbox_key=agent_inbox(target,conv), cursor_key=…, lock_key=agent_lock(target,conv))`(复用 `:37`)。
     - return `"offered:{card_id}"`。
  3. 无 ready 卡:`try_claim_collector` 成功 → enqueue collector run(`_collect=True`,复用 `_build_run_payload(agent_id=primary)`)→ `"collect:{conv}"`;否则 `"idle"`。
- **复用**:Phase 0 inbox 原语、`try_claim_collector`、collector enqueue。
- **新建**:把"claim+enqueue"换成"post board_offer + notify"(不再由驱动器替主认领)。`board_offer` 是新的入站消息 `kind`。
- **谁调它**:① `decompose_to_board`(首张)② self-claim 的 card-working 完成后(relay,替代旧 `_finish_card_working` 末尾的 `_drive_board`)③ pass 重路由后 ④ watchdog 回收卡后。

### 4.4 dispatcher kind 判别(扩展,🔶 唯一改 Phase 0 的点)
- **背景**:现 dispatcher 闸门(`dispatcher.py:120-131`)在 `board_round` 在时 defer 一切。但 `board_offer`(走 target 信箱)是 round 的引擎,**不能 defer**;只有**用户消息**(走 primary 信箱)在 round 内才 defer。
- **落点**:`backend/nanoresearch/bus/dispatcher.py:_handle_notify`(`:108-145`)。**调整顺序**:抢锁 → **先 `read_next_after_cursor` 读出下一条入站条目** → 看 `payload.kind`:
  - `kind == "board_offer"` → **放行**(即使 board_round 在):enqueue 一个 **self-claim run**(见 4.5),带 `_board_offer_card_id` + 信箱锁三件套(`_lock_key/_lock_token/_entry_id`)。
  - 否则(普通用户 turn)→ 沿用现有闸门:`scard(pending) OR exists(continuation_lock(agent_id,conv)) OR exists(board_round(conv))` → defer;否则 enqueue 普通 `run_agent_job`。
- **不推翻**:闸门逻辑、信箱锁生命周期、PEL reclaim 全不变;只加一个"先读条目、按 kind 分流"的判别。
- **不变量保持**:board_offer 走 `agent_lock:{target}:{conv}`(target 自己的信箱锁),与 primary 的 `agent_lock:{primary}:{conv}` 不同 key;但 card-working 会话只读 → 不写会话 → 不与 primary/collector 的会话写竞争(结论①仍成立)。串行靠"一次只 offer 一张"保证不会有两个 self-claim run 并发。

### 4.5 self-claim run 模式(新建 — 路B 核心)
- **职责**:被唤醒的目标主在**自己的 run 里**判断 claim/pass,claim 则干活、pass 则重路由。
- **落点**:`backend/nanoresearch/worker.py:run_agent_job` 新增分支 `_board_offer_card_id` 非空:
  1. 取卡 `c = repo.get(card_id)`;若已非 ready(被别的路径动过)→ 直接 finalize 信箱锁 return(幂等)。
  2. 建 loop(payload 经 `_build_run_payload(agent_id=target)` → 带 **target 主的 persona**,见已修的 per-target persona)。
  3. **轻量 LLM 判断 claim/pass**:一次 LLM 调用(结构化输出或 claim/pass 工具),输入 = 卡 `title/spec` + 该主 persona,问"这张卡是否该由你做"。
     - **claim** → `claim_card(redis, repo, card_id, target, conv)`(复用 `bus/workboard.py:claim_card`,WIP=1 + claim 锁兜底)。成功 → 跑 card-working(`process_direct(session_readonly=True)` + `_finish_card_working`,复用现有 `_card_id` 路径)→ 完成调 `_offer_next_or_collect`(relay)。claim 失败(WIP 忙/竞争)→ 不动卡,finalize return(下次 relay 会再 offer)。
     - **pass** → `_reroute_card`(见 4.6)。
  4. 信箱锁 finalize(Phase 0,`finally`)。
- **复用**:card-working 全套(只读、落卡、heartbeat、finish、promote)、`claim_card`。
- **新建**:run 开头的 claim/pass LLM 判断 + claim/pass 分流。

### 4.6 pass 重路由(新建)
- **职责**:目标主 pass 后,把卡重路由,带上限保证终止。
- **落点**:`backend/nanoresearch/worker.py:_reroute_card(redis, repo, conv_id, uid, card)`:
  - `pass_count++`(`workboard_cards.pass_count` 列,见 §10)。已 pass 过的主记在卡的 `artifacts`(append `{"passed": agent_id}`),供"减去已试过的主"。
  - 候选 = 已激活主(`list_member_agents`)减去 primary 减去已 pass 过的;取下一个 → 设 `target_agent_id` = 它 → `_offer_next_or_collect` 重新 offer。
  - **上限**:`pass_count >= 已激活主数`(每主一次机会)→ **fallback**:`target_agent_id = primary`,offer 给 primary(primary 强制执行这张卡);primary 的 self-claim 判断对"自己拆的卡"默认 claim。这保证有限步内必有人做或 blocked。
- **数据**:`workboard_cards` 加 `pass_count INTEGER DEFAULT 0`(迁移 + CHECKS + 模型)。

### 4.7 collector(复用,不改)
- 已建:`is_board_quiesced` / `try_claim_collector` / `_collect_cards_into_session` / collector 分支(`_collect=True`,抢 `agent_lock:{primary}`,合并卡产出 → `process_direct` 综合 → run_end → `end_round`)。复用原 run_id(planning run 的)。

---

## 5. 串行不变量(WIP=1)如何维持
- **一次只 offer 一张**:`_offer_next_or_collect` 每次只取一张 ready 卡 offer。下一张的 offer 只在当前卡 **完成**(`_finish_card_working` → relay)或 **pass 重路由** 时发生。
- 故任一时刻最多一个 self-claim run 在跑、最多一张 running 卡。`claim_card` 的 WIP=1 校验(`list_by_conversation(running)` 非空则拒)是二次兜底。
- 用户消息在 round 内被 dispatcher 闸门 defer(`board_round`),不插队;round 由 collector `end_round` 结束后放行。

---

## 6. 边界与容错
| 情况 | 处理 |
|---|---|
| primary 拆卡但任务其实简单 | LLM 自己不调 `decompose_to_board`,正常单主答复(工具 description 明确"简单别用")。 |
| 目标主 pass | `_reroute_card` 重路由,pass 上限 → fallback primary 执行。 |
| 所有主都 pass | 上限触发 → primary 强制执行(primary 对自拆卡默认 claim)。 |
| 被唤醒时 WIP 忙(claim 失败) | self-claim run 不动卡、finalize return;当前卡完成时的 relay 会再 offer。串行下"一次一张"基本不触发,作兜底。 |
| board_offer 唤醒后卡已非 ready | 幂等 return(被 watchdog/其它路径动过)。 |
| self-claim run 崩溃 / card 卡死 | watchdog `_scan_stale_cards`(已建)回收 running 卡 → blocked + 重 drive(改为 `_offer_next_or_collect`)。 |
| 用户在 round 中途发消息 | dispatcher `board_round` 闸门 defer;collector `end_round` 后放行(已建)。 |
| 收口后晚到的卡完成 | `collected` 标记 → `is_board_quiesced` False → `try_claim_collector` False,静默丢弃(NO_REPLY,已建)。 |
| 后继卡无限自生 | `can_create_successor` 双 cap(50/8,已建)→ 必 quiesce。 |

---

## 7. 硬约束保持(自检)
- **不引入中心编排 agent**:拆卡由 primary(参与者)经工具完成;认领由目标主自己做。无专职调度 agent。✅
- **不做 A2A**:主之间不直接通信,全程经看板(卡片 + 信箱 board_offer 是看板的投递,不是主对主喊话)。✅
- **子 agent 私有 leaf**:不动(`subagent.py:130` 无 message/spawn)。看板认领者只能是主。✅
- **不推翻 Phase 0/1**:dispatcher 仅加 kind 判别;defer 仅加 board_round 条件;其余(信箱锁、claim、card-working、collector、watchdog、cost cap)全复用。🔶 唯一改 Phase 0 的点是 4.4 的 kind 判别,是扩展不是替换。✅
- **Redis 5.0**:沿用现有 inbox/notify 约定(`mailbox._next_stream_id`),无新 Redis 特性。✅
- **会话写者 ≤1(结论①)**:self-claim 的 card-working 会话只读;唯一会话写者仍是 collector(primary)持 `agent_lock:{primary}`。串行保证无并发 self-claim run。✅
- **串行 WIP=1**:一次只 offer 一张 + claim 兜底。✅

---

## 8. 测试策略

**单元/集成(真 Redis+PG,无 LLM)** — `tests/integration/test_phase2_workboard.py` 续写:
- `decompose_to_board` 工具:建卡(ready/todo+link)、activate_agents、begin_round、offer 第一张(断言 board_offer 进 target 信箱 + notify,卡**仍 ready 未被 claim**)。
- dispatcher kind 判别:`board_offer` 在 board_round 在时**放行**(enqueue self-claim run);用户 turn 在 board_round 在时 `deferred_batch`。
- self-claim 分支(stub LLM 判断):claim → 卡 running+owner=target;pass → `_reroute_card` 重路由 + pass_count++;pass 超限 → fallback primary。
- `_offer_next_or_collect`:有 ready→offer(notify target);无 ready+quiesced→collect。
- primary defer:board_round 在 → primary run 留 running、不发 run_end(扩 `_has_pending_subagents` 等价测)。

**端到端冒烟(真 LLM)** — 扩 `backend/scripts/phase2_e2e_smoke.py`(或新 `phase2_e2e_collab.py`):
- 不再手动塞卡 + 手动 `_drive_board`;改为:给 primary 一条真任务 → primary **真的调** `decompose_to_board` 拆卡 → 目标主**真的被唤醒、在自己 run 里真的自主 claim** → 接力 → 收口 → 用户收到综合答复。
- **路B 真假命门(关键断言)**:offer 之后、目标主 run 之前,卡是 **ready 且无 owner**(证明驱动器**没替它认领**);目标主 run 之后卡 running/done 且 owner=target(证明认领发生在**目标主自己的 run 里**)。这是路A/路B 的判别点。
- 仍带结论①命门:card-working 期间会话 `session:msg` 不变。

---

## 9. 不在本期范围(留作增量)
- **并行**:多个候选主同时被唤醒竞争认领、多卡同跑、动态响应。本设计是其前置(把"offer 一张给 target"扩成"offer 多张给多候选"),并行是扩展不是重做。
- **看板状态前端视图**:本轮先做触发让协作真发生;视图(只读端点 + 组件)随后单独做(届时真实对话已有卡片可展示)。
- **拆卡质量优化**:primary 拆卡的 prompt 调优、何时该多主的判据细化,先用工具 description + persona 引导,后续按真实 badcase 调。

---

## 10. 数据/迁移增量
- `workboard_cards` 加 `pass_count INTEGER NOT NULL DEFAULT 0`(模型 + `migrate_phase2_workboard.sql` + `database.py` CHECKS + conftest 已含表)。
- 入站消息新增 `kind` 字段约定:`"board_offer"`(payload 带 `card_id`);现有用户消息无 `kind`(视为普通 turn)。无 schema 变更(信箱是 Redis Stream,payload 是 JSON)。
