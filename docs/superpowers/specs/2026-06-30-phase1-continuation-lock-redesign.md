# Phase 1 锁模型结构性重设计:continuation_lock 闸门 + subagent_results 暂存

> 状态:**已实现并合入分支 feature/phase1-subagent-async-return**(T1 join_and_fire / T4 闸门 / T6 子暂存 / T5 worker 续接 / T7 watchdog + 清理)。47 redesign+Phase0 测试 + 37 非 SSE 回归全绿。下方为设计原文(评审已通过)。
>
> 原状态:设计待评审(不写代码)。本文档**取代** Phase 1 计划里的锁/数据流模型(原 `join_and_acquire` 占 agent_lock + 子直接 append 会话),修复两个已实证的洞:
> - **洞 1(已实证)**:`join_and_acquire` 占的是 agent_lock,R1 持锁期间最后一个子 join 的 `SET NX` 失败 → `fired=False` → pending 已空却无续接 → 挂死等 watchdog。
> - **洞 2(分析确认)**:R1 的会话保存是全量 `DEL+RPUSH`,子的 `append_message` 是裸 `RPUSH` 不拿锁 → 快子/秒错子在 R1 保存前 append 会被 `DEL` 冲掉。

---

## 0. 核心原则(一句话)

**join 绝不触碰任何「正在执行的父 run 持有的锁」**;改用一个父 run 从不持有的闸门标记 `continuation_lock`(join 原子置位);子结果改为**暂存**(`subagent_results`),由续接 run 作为**唯一写者**落库,躲开 R1-保存覆盖。

---

## 1. Redis 键总览(新增 2 个)

| 键 | 类型 | 作用 | 谁置/谁清 | TTL |
|---|---|---|---|---|
| `agent_lock:none:{conv}` | String(token) | **会话写互斥**(用户消息 run + 续接 run 写会话前都要持有) | dispatcher(用户 run)/ 续接 run 自己抢;持有者 finalize 释放 | PX 30s + 续租 |
| `continuation_lock:none:{conv}` | String(token) | **闸门标记**:有 batch 待续接 / 续接进行中。父 run 从不持有它 | join 原子置位;续接 run 续租;续接 finalize DEL | PX 120s + 续租 |
| `pending:{session_key}` | Set | batch 在飞(成员 `{task_id}:{ts}`) | 子 spawn 时 SADD;子完成 join 时 SREM | 无(reaper/watchdog 清) |
| `subagent_results:{session_key}` | List | **子结果暂存**(append-only),给续接读 | 子完成 RPUSH;续接读后 DEL | `SESSION_TTL`(2h) |
| `run_events:{run_id}` | Stream | 出站给前端 SSE(不变) | run/子写;SSE 读 | 24h |

> `session_key` 平台恒为 `web:{conv}`,即 `f"web:{conversation_id}"`,与 `agent_lock`/`continuation_lock` 的 `{conv}` 同一 conversation_id。

---

## 2. 闸门(dispatcher `_handle_notify`)

抢到 `agent_lock` 后:

```
if SCARD(pending) > 0  OR  EXISTS(continuation_lock):
    release(agent_lock); return "deferred_batch"
else:
    读下一条 inbox entry → enqueue 用户 run(带 agent_lock token + entry_id)
```

**用户消息只在「无 batch 在飞 且 无续接待处理/进行中」时才跑。** 二者由 join 原子同步切换(见 §3),无空档。

---

## 3. continuation_lock 完整生命周期(你要看的第①样)

### 3.1 置位 —— join 原子步(唯一置位者)
新 `join_and_fire(redis, session_key, task_id, cont_lock_key, cont_token, px) -> bool`,一段 Lua:

```
-- 按 task_id 前缀移除本成员
SREM pending <member>
if SCARD(pending) == 0:
    SET continuation_lock = cont_token PX <px>     -- 覆盖式 SET(只有清空者到这,无并发置位)
    return 1                                         -- fired
return 0
```

- **关键**:置位的是 `continuation_lock`,**不是** `agent_lock`。父 run(R1/续接)持有的是 `agent_lock`,从不持有 `continuation_lock` → 这个 `SET` **永不失败** → 洞 1 消除。
- 「恰好一次」仍由原子 SREM-to-zero 保证(只有清空 pending 的那个子到达 `SET`)。
- 覆盖式 `SET`(非 NX):极端时序下若上一层 `continuation_lock` 尚未被上一个续接 DEL,新 batch 的 join 覆盖之;旧续接 finalize 的 DEL 是 **token 门控**(见 3.3),token 不匹配则不 DEL,新值得以存活。

### 3.2 续租 —— 续接 run
续接 run 启动后开一个 refresher,每 `px/3` 用 token 门控 `PEXPIRE continuation_lock`(同 Phase 0 锁续租)。崩溃即停续租。

### 3.3 DEL —— 续接 finalize(token 门控)
续接 run 结束时:`GET continuation_lock == my_token then DEL`(token 门控,防误删后继续接的锁)+ re-notify(让被 defer 的用户消息重试)。

### 3.4 崩溃兜底(新模型最关键的新失效点)
| 崩溃点 | 现象 | 兜底 |
|---|---|---|
| batch 期(continuation_lock 未置位)子崩/卡死 | pending 永不空 | watchdog 扫超期 pending → 触发 join_and_fire(置 continuation_lock + 拉续接) |
| join 已置 continuation_lock,但续接 run 没起来(worker 挤压/崩) | 无人续租 | continuation_lock **PX 120s 到期**自动清(留足 enqueue→pickup 余量)→ 闸门放开;原 run_id 仍 running → watchdog `_scan_stuck_running` 标 failed + 补 run_end |
| 续接 run 跑一半崩 | continuation_lock 续租停 → PX 到期清;原 run_id running | 同上:watchdog 补 run_end;暂存结果 TTL 自然过期 |
| **续接抢 agent_lock 超时**(R1 卡住没释放) | 续接拿不到锁 | **续接自清**(它持 continuation_lock):token 门控 DEL continuation_lock + 标 run failed + 补 run_end + re-notify。**不留给 watchdog、不依赖 PX**(见下) |

### 3.5 时间参数自洽(确认 3)
| 参数 | 值 | 关系 |
|---|---|---|
| `agent_lock` PX / 续租 | 30s / 10s | Phase 0 不变 |
| **续接抢 agent_lock 重试上限** | **30s** | 抢锁期间续接**活着、持续续租 continuation_lock**(每 `PX/3`≈40s),故 continuation_lock 在抢锁期不会过期 |
| `continuation_lock` PX / 续租 | **120s / 40s** | PX(120) > 抢锁上限(30) + 余量;抢锁期靠续租维持,不靠 PX |
| watchdog 扫描周期 | 120s | —— |

- **抢锁超时(30s)路径**:续接**自己清** continuation_lock + 发 run_end(上表新增行)→ **无 PX-vs-watchdog 时间窗**,闸门不会出现「锁已过期 + 续接仍被处理」的漏窗。残留只剩「R1 自身卡住占着 agent_lock」,那是 Phase 0 既有行为(job_timeout 7200s 兜底),Phase 1 不放大。
- **续接进程崩溃路径**(无法自清):continuation_lock 靠 PX(120s)自愈放开闸门。此窗内**没有活跃续接在写**(进程已死)→ 用户消息进来只是读到「R1 已存 + 部分暂存」的降级基线,**无并发写、无覆盖**;孤儿 run_id 由 watchdog `_scan_stuck_running` 在 run_stuck 后补 run_end。这是崩溃恢复的可接受降级,不是正确性洞。
- 因此「抢锁超时」走自清(无窗),「进程崩溃」走 PX 自愈(窗内无活跃写者,良性)。两条路径都自洽。

---

## 4. subagent_results 暂存完整生命周期(你要看的第②样)

### 4.1 谁写 —— 子(成功 / 失败都写),join 前
`_report_and_join` 顺序:
1. `_announce_result`(写 `run_events` 给前端,**不变**,AC5)。
2. **RPUSH `subagent_results:{session_key}`** 一条结果 payload(成功=结果体,失败=错误 marker)。**有限重试 2 次**;仍失败 → RPUSH 一条极小 `[result unavailable]` marker(payload 小、更易成功)。
3. **只要 RPUSH(结果体或 marker)成功 → 该子自己调 `join_and_fire` 推进它的 pending 成员**(原子 SREM + 判空 + 置 continuation_lock)。**marker 即该子的降级结果,有真实推进作用 —— 不是 log、不是等 watchdog。** batch 不会被这个子卡住,续接读暂存时会看到 `[result unavailable]`。
4. **唯一不推进的情况**:连极小 marker 都 RPUSH 失败(= Redis 全挂,灾难级)→ 不 SREM/不 join → 成员留 pending → 交 watchdog 长 stale(此时整个系统已不可用,长挂可接受)。

> 确认 2:marker 的推进由「写 marker 的那个子自己 `join_and_fire`」完成,**不依赖 watchdog**。watchdog 只兜「连 marker 都写不进」的 Redis 全挂场景。

> 子**不再直接写会话列表** → 躲开 R1-全量保存覆盖(洞 2)。子只写独立 append-only 列表。

### 4.2 谁读 / 谁清 —— 续接 run,落库前
**硬要求(确认 1,最关键):续接的一切会话读写都在「抢到 agent_lock 之后」。** 单写者只保证"只有续接写",还要"续接写时基线最新",两者都满足才无覆盖。续接的执行序:

1. 抢到 `agent_lock`(见 §5 有界重试)。**在此之前不读消息列表基线**——`_build_run_payload`(由 join 触发者在锁外调)只读 `conversation` 行取 agent 配置,**绝不读 messages**。
2. 锁内:原子 `LRANGE subagent_results 0 -1` + `DEL subagent_results`(一段 Lua,读清同一步)。
3. 锁内:把读出的每条 **append 进会话列表**(RPUSH 追加式,不读基线)。
4. 锁内:`process_direct(content=汇总指令)` → 其 `get_or_create` 此刻才**读基线**(= R1/上层已 save 后的最新状态,因为续接持锁意味着上层已释放锁=已 save)→ 推理 → 全量 `DEL+RPUSH` 保存。
5. 反例(禁止):若锁前(如 payload 阶段)就读了 messages 基线,而那时 R1 还没 save 完,续接用过期基线全量保存 → **续接覆盖 R1 的写**(与"子覆盖"同构)。所以基线读(`get_or_create`)**必须在锁内**(即 `process_direct` 在 acquire 之后调用)。

### 4.3 清 vs 写的并发(你点名的)
- **本层不冲突**:join 触发的前提是 pending 空 = 本 batch 所有子都已完成(每个子是「先 RPUSH 后 SREM/join」,故最后一个 SREM 清空时所有 RPUSH 已落)→ 续接 drain 时**无子在写**。
- **跨层不冲突**:续接 drain+DEL 在 `process_direct` **之前**完成;续接若再 spawn 二层 batch,是在 `process_direct` **之中/之后**,二层子写的是 DEL 之后的**新**列表 → 与本次 drain 不重叠。
- 因此「续接清的时候新 batch 在写」**不会发生**:新 batch 在 drain+DEL 之后才产生。

---

## 5. 多层递归不变量表(你要看的第③样 —— 逐层论证)

设主回合 R1 → spawn 一层 batch → 续接 C1 → C1 再 spawn 二层 batch → 续接 C2(不再 spawn)。

| 时刻 | 谁在写会话(≤1) | agent_lock | continuation_lock | pending | 用户消息为何被 defer |
|---|---|---|---|---|---|
| R1 处理用户消息 | R1 | R1 持(dispatcher 给) | 无 | 空 | agent_lock 被 R1 持(SET NX 失败) |
| R1 spawn 一层、save、return、finalize | 无(R1 已存盘) | R1 释放 | 无 | >0(一层) | 闸门 `pending>0` |
| 一层 batch 跑(子写 staging,不写会话) | 无 | 空 | 无 | >0 | 闸门 `pending>0` |
| 一层最后子 join(原子) | 无 | 空 | **置位**(join SET) | 空 | 闸门 `EXISTS(continuation_lock)` |
| C1 起:抢 agent_lock(罩早 join 残留的 R1 写)→ drain+落库+汇总 | C1(抢到 agent_lock,唯一写者) | C1 持 | C1 持(续租) | 空 | 闸门 `EXISTS(continuation_lock)` |
| C1 spawn 二层、save、return、finalize | 无(C1 已存盘) | C1 释放 | C1 DEL(token 门控) | >0(二层) | 闸门 `pending>0`(二层) |
| 二层 batch 跑 | 无 | 空 | 无 | >0 | 闸门 `pending>0` |
| 二层最后子 join | 无 | 空 | 置位 | 空 | 闸门 `EXISTS(continuation_lock)` |
| C2 起:抢 agent_lock → drain+落库+汇总(无再 spawn) | C2 | C2 持 | C2 持 | 空 | 闸门 `EXISTS(continuation_lock)` |
| C2 完成、`SCARD(pending)==0` → 发 run_end、finalize | 无 | C2 释放 | C2 DEL + re-notify | 空 | **闸门全清 → 用户消息可跑** |

**两条不变量逐层成立:**
1. **互斥写**:任一时刻「谁在写会话」≤1。父 run 都在 `process_direct` 末尾 save 后才 return;续接抢 `agent_lock` 后才写;早 join(秒错子在父 save 前触发)由「续接抢 agent_lock(有界重试,等父释放)」罩住 —— 续接拿不到 agent_lock 就等,绝不与还在 save 的父并发写。
2. **闸门无空档**:从 R1-spawn 到 C2-finalize,每一刻 `pending>0` 或 `continuation_lock 在`(二者由 join 原子切换:同一 Lua 内 SREM 清空 pending 与 SET continuation_lock 同时发生)→ 用户消息全程被 defer。
3. **join 永不撞父锁**:join 只 SET `continuation_lock`,父 run 持的是 `agent_lock`,跨层皆然 → 递归安全。

> 续接为什么仍要抢 `agent_lock`(而非只靠 continuation_lock 闸门)?闸门只挡**新用户消息**;但「早 join」下**父 run 可能还在 save**,续接必须与之互斥 → 抢 agent_lock。正常时序父早已释放,续接瞬间抢到。

---

## 6. 必改 2/4 在新暂存模型下的重新定位(你要看的第④样)

- **必改 2(保护重心迁移)**:原「append 进会话成功才推进 join」→ 现「**RPUSH 进 `subagent_results` 成功才推进 join**」。durability 点从「子写会话」移到「子写暂存列表」。RPUSH 失败 → 不 SREM/不 join → 成员留 pending → watchdog 兜。**仍不允许「结果没落地却判齐」。**
- **必改 4(append 失败 2h 挂起,采纳「重试 + 极小 marker」)**:
  - 暂存 RPUSH **有限重试 2 次**;
  - 仍失败 → RPUSH 极小 `[result unavailable]` marker(payload 小、更易成功)→ marker 成功即可正常 join(对话照常推进,只是该子结果缺失);
  - 连 marker 都失败 = Redis 全挂(灾难级)→ 才留长 stale。**常见 DB/瞬时抖动不再挂 2h**(已被重试 + marker 化解)。
  - 备选(若你要更强保障):给「已完成但暂存失败」的成员单独 `staging_failed:{conv}` 短 stale 集合(120s),watchdog 快速兜。默认走前者(更简单),需要再上后者。

---

## 7. 对计划任务的影响(改哪些)

| 任务 | 改动 |
|---|---|
| T1 `mailbox` | `join_and_acquire` → `join_and_fire`(SET `continuation_lock` 非 agent_lock);新增 `continuation_lock` key(`redis_keys.py`);暂存读清 Lua `drain_subagent_results`(LRANGE+DEL 原子) |
| T4 dispatcher | 闸门 `SCARD(pending)>0 OR EXISTS(continuation_lock)` |
| T5 worker | 续接 path:持 `continuation_lock`(token) + **抢 agent_lock(有界重试 30s)→ 锁内 drain 暂存→落库→`process_direct`(基线读在锁内,§4.2)**;双锁续租;finalize = 释放 agent_lock + token 门控 DEL continuation_lock + re-notify;**抢锁超时 → 自清(DEL continuation_lock + 补 run_end,§3.5)** |
| T6 subagent | `_report_and_join`:不再写会话;改 **RPUSH 暂存(重试+marker)** → `join_and_fire`(置 continuation_lock + 带 token 直接 enqueue 续接) |
| T7 watchdog | 兜底 `join_and_fire`(置 continuation_lock 后 enqueue 续接,真 run_id);continuation_lock 僵死由 PX 自愈;`_scan_stuck_running` 不变 |
| 必改 2/4 | 重心迁到暂存 RPUSH;重试 + 极小 marker |

---

## 8. 红测试新预期(TDD red→green 锚点)

`test_join_fires_even_when_parent_run_holds_agent_lock`(R1 持 `agent_lock` 期间最后一个子 join):
- **现状(已实证)**:`fired=False`、无续接 → 挂死。
- **新模型预期**:join 置 `continuation_lock`(与 agent_lock 不同 key)→ **`fired=True`**、续接被 enqueue(复用原 run_id)、`continuation_lock` 已置位。

补充必测:
- `test_gate_defers_when_continuation_lock_present`:pending 空但 continuation_lock 在 → dispatcher `deferred_batch`。
- `test_continuation_drains_staging_and_appends_to_session`:暂存两条 → 续接 drain+落库 → 会话列表含两条 + 暂存被 DEL。
- `test_continuation_acquires_agent_lock_before_write`:父持 agent_lock 时续接等待(不并发写)。
- `test_recursive_second_batch_keeps_invariant`:C1 再 spawn → 二层 batch 期 `pending>0`、用户消息仍 defer。
- `test_staging_rpush_retry_then_marker`:暂存 RPUSH 头两次失败、第三次(marker)成功 → 仍 join。

---

## 9. 确认 1/2/3 落定 + 仍需你拍板一点

- **确认 1(基线读在锁内)**:已 codify 进 §4.2 —— 续接 `get_or_create` 在 acquire 之后,锁前不读 messages。
- **确认 2(marker 有推进作用)**:已 codify 进 §4.1 —— 写 marker 的子自己 `join_and_fire` 推进,不靠 watchdog。
- **确认 3(时间窗自洽)**:已 codify 进 §3.5 —— 抢锁超时走「续接自清」(无窗),进程崩溃走「PX 自愈」(窗内无活跃写者,良性);PX(120) > 抢锁上限(30) + 余量。

**仍需你拍板一点**:必改 4 默认走「重试 2 次 + 极小 marker」(§4.1,简单);要不要再叠加「`staging_failed` 短 stale 集合(120s)」做更强兜底(多一套,只在「连 marker 都失败」的 Redis 全挂场景才有额外价值)?**默认不叠加**,你要更强保障就说。

定了这一点 + §1–§8 自洽你认了,即进 TDD(先固化 §8 红测试)。
