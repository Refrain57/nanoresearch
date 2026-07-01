# Consolidation 压缩修订 — 触发/留尾/时点 + confidence gate

**Date:** 2026-06-28
**Status:** Design — pending user review
**范围（用户已拍板）:** 一份 spec 装两件最高严重度、互不撞车的修复：
1. **T1 压缩三件事** —— idle gate + 轮次计数 + 留尾 + end_idx 时点（改 `loop.py:530-560`）。
2. **confidence gate（形态三）** —— 压缩产物被 0.7 阈值静默丢弃、从未入库（改 `memory.py:355`）。
外加一个 **前置阻塞修复**：idle gate 依赖的 `session.updated_at` 时区往返不一致（本轮一并修掉，不留到 plan）。

**Supersedes:** `2026-06-28-consolidation-anchor-retention-design.md` 的修法 1。该 spec 的修法 2A/2B 本轮不做（见 §9）。

---

## 0. 证据等级约定

- **【代码确认】** 已读源码到行。
- **【日志确认】** 用户真实日志佐证。
- **【需进一步查】** 推断，落地前须验证。

---

## 1. 调查结论：压缩触发器全景

### 1.1 三问直答

**有几个压缩触发器？** 自动 2 个 + 手动 1 个。【代码确认：grep `consolidat|archive_messages` 全仓只此三处推进 `last_consolidated` / 调 `MemoryStore.consolidate`】

| # | 名称 | 入口 | 性质 |
|---|---|---|---|
| T1 | startup 计数触发 | `loop.py:530` `_check_pending_consolidation` | 自动，每 job 进来先跑 |
| T2 | token 压力触发 | `memory.py:548` `maybe_consolidate_by_tokens` | 自动，按 prompt token 估算 |
| T3 | `/new` 手动归档 | `builtin.py:69` `cmd_new` | 用户主动重置，整段归档后 `clear()` |

**`pending_count >= 5` 属于谁、数什么？** 属于 **T1**，是其唯一阈值，非独立触发器。`loop.py:541` `pending_count = len(session.messages) - session.last_consolidated` 数的是 **OpenAI message 条数**，不是对话轮次。`_save_turn`（`loop.py:875-899`）每条 message 都 append，"调 3 次工具的一轮 = 1+3×2+1 = 8 条" → 单轮即过 5 线。**确认：把"轮次"实现成了"消息条数"。**【代码确认】

**三者关系？** T1、T2 **并行独立**，但在同一 `_process_message` 内顺序跑、共享同一 `last_consolidated` 偏移：
```
_process_message(loop.py:667)
  line 752  _check_pending_consolidation   ← T1（前台，本turn消息尚未append）
  line 760  maybe_consolidate_by_tokens     ← T2（前台，build prompt 前）
  line 800  _run_agent_loop
  line 820  _save_turn                        ← 本turn消息此刻才进 messages
  line 822  maybe_consolidate_by_tokens(bg)   ← T2（后台）
（system-message 路径 loop.py:688-723 同构：694 T2前台 / 719 _save_turn / 721 T2后台，无 T1）
```
不是层级，是两机制共享一个指针。`pending_count>=5` 只在 T1。【代码确认】

### 1.2 现状/目标对照表

| 维度 | **T1** 现状 | **T1** 目标 | **T2** 现状 | **T2** 目标 |
|---|---|---|---|---|
| 入口 | `loop.py:530` | 同 | `memory.py:548` | 同 |
| 触发条件 | `pending_count(条数)<5` 跳过(`:541-542`)，**无 idle** | **idle gate** + **轮次计数 ≥ MIN_TURNS** | `estimated≥budget`(`:561,566`) | 不变 |
| 计数对象 | **消息条数**【bug】 | **对话轮次**（数 user 条） | token 估算 | 同 |
| 压缩范围 | **全部 pending**(`:551`)，**不留尾**【bug】 | `pick_consolidation_boundary(tail_protect=N)` 留尾 | 已 `tail_protect=5`(`:590`)【代码确认】 | N 可配、与 T1 对齐 |
| `last_consolidated` 推进 | 推到 **`len(messages)`**(`:555`)，发生在本turn append 之前【bug】 | 推到 **`end_idx`** | 已 `end_idx`(`:636`)【代码确认】 | 不变 |
| 留尾 | **否** | **是（N 条）** | **是（5）** | **是（N，可配）** |
| 去重 gate | 实例 set(`:152,538`)，每 job 清零【bug】 | **idle gate（读持久 updated_at）去重**；实例 set 降为同-job 短路 | 每 session 一把 Lock(`:559`) | 不变 |

**关键纠正：T2 的留尾 + 正确推进时点在现有代码里已对**【代码确认 `memory.py:590,636`】。问题 1/3/4 的 bug 面集中在 T1；T2 仅需把写死参数化。

---

## 2. 根因定性（systematic 调查结论）

> 先定根因再开方。

- **问题 1 频繁压缩**【日志：33273 token=55% budget 被压】：根因 = T1 数条数 + 无 idle + gate 不持久。55% 时 T2 不可能触发（`estimated<budget` 直接 return，`memory.py:566`），点火的是 T1 条数阈值；单工具轮 ≈8 条即过线；gate 每 job 清零 → 背靠背 turn 反复点火。**修对不删**：数轮次 + idle gate。
- **问题 2 startup 重复触发**【代码 `loop.py:152`】：把"该跨 job 持久的去重状态"放进用完即弃的实例 set，违反"无状态执行机"。**用持久 `updated_at` 做 idle gate** 替代，既外置又不锁死兜底（压缩后 updated_at 刷新 → 30min 内再来被跳过=去重；超 30min 再来 → 重新触发=不锁死）。**不整把持久化 set。**
- **问题 3 推进时点错**【代码 `loop.py:555` vs `:820`】：T1 在 `:752` 推到"此刻 len"，但本 turn 消息要到 `:820 _save_turn` 才 append → 这批消息下个 job 又成 pending。**修法**：走 boundary、推到 `end_idx`，boundary 恒 ≤ `len−N`，不会推过头。
- **问题 4 不留尾**【日志：agent 回"没有前面的上下文"；代码 `loop.py:551`+`manager.py:68`】：T1 取全部 pending 压扁 + `get_history` 起点就是 `last_consolidated` → 下一轮 `history=[]`，"这篇/第1个"锚点没了。**修法**：T1 走 `pick_consolidation_boundary(tail_protect=N)`，最近 N 条原文永留 prompt。

---

## 3. 修复设计

### 3.1 修法 A — T1 改写（`loop.py:530-560`）

idle gate + 轮次计数 + tail protect + end_idx 推进，一个函数闭环：

```python
async def _check_pending_consolidation(self, session, agent_id=None):
    if session.key in self._startup_consolidated:   # 同-job 廉价短路，不再承担正确性
        return

    # (a-1) idle gate —— 背靠背 turn 直接跳过，灭频繁压缩
    if _utcnow_aware() - _as_aware_utc(session.updated_at) < IDLE_THRESHOLD:
        self._startup_consolidated.add(session.key)
        return

    # (a-2) 轮次计数 —— 数 pending 内 user 消息，不数条数
    pending = session.messages[session.last_consolidated:]
    pending_turns = sum(1 for m in pending if m.get("role") == "user")
    if pending_turns < MIN_PENDING_TURNS:
        self._startup_consolidated.add(session.key)
        return

    # (b) 留尾 —— 复用成熟 boundary 逻辑
    boundary = self.memory_consolidator.pick_consolidation_boundary(
        session, tokens_to_remove=1, tail_protect=TAIL_PROTECT)
    if boundary is None:
        self._startup_consolidated.add(session.key)
        return
    end_idx, _ = boundary
    chunk = session.messages[session.last_consolidated:end_idx]
    if not chunk:
        self._startup_consolidated.add(session.key)
        return

    success = await self.memory_consolidator.consolidate_messages(
        chunk, agent_id=agent_id, uid=self._uid)
    if success:
        session.last_consolidated = end_idx       # (c) 只推进到 boundary
        await self.sessions.save(session)
        self._startup_consolidated.add(session.key)
```
`_utcnow_aware` / `_as_aware_utc` 见 §3.3。T2、T3 不动（T2 已对，T3 是主动重置）。

### 3.2 修法 B — confidence gate（形态三）

**故障**【代码确认】：consolidation 成功后写会话摘要（唯一含论文名/方法名的产物）到 user_memory，`confidence: 0.6`（`memory.py:355`）；但 `write_user_memory_sync` 第一行 `[m for m in memories if m.get("confidence",0) >= 0.7]`（`knowledge_search.py:153`）→ 0.6<0.7 → **整条丢弃，返回 (0,0)，永不入库**。`_raw_archive` 0.5（`memory.py:393`）同样被丢。后果：摘要从未进向量库，下一轮 query 再准也召不回。这是"压了找不回"的根因总开关。

**落地前两项查证（用户要求）：**

**(a) 0.7 阈值还有谁依赖？** —— **有。不能降全局阈值。**【代码确认】
全部 `write_user_memory[_sync]` 调用面：
- `memory.py:351` consolidation_summary，conf **0.6** → 被丢
- `memory.py:389` raw_archive，conf **0.5** → 被丢
- `conversation_knowledge_extractor.py:140` → `write_user_memory`（async）→ 委托 `write_user_memory_sync`（`knowledge_search.py:202`，**同一道 0.7 过滤**），conf = LLM 给的 `ui.confidence`（`:133`，动态 0~1）
- `scripts/*`（migrate / test，非运行时）
extractor 路径**依赖** 0.7 当质量闸：LLM 标的低置信用户信息靠它挡掉。降全局 0.7 会让 extractor 的 0.6~0.69 低质记忆涌进库。**故修在 consolidation 写入点，不动 gate。**

**(b) 0.6 是 bug 还是有意？** —— **判定为 bug（数值与意图矛盾）。**【代码确认】
`consolidate()`（`memory.py:348-357`）显式写 user_memory 的目的就是让摘要可被 `build_history_context` 召回。一个保证 no-op 的写入是死代码——若作者真认为"会话摘要本就中等可信、不该持久"，就不会去写它。0.6 只是低于闸门的笔误，不是"摘要不该入库"的设计决策。

**推荐改法**：把 `memory.py:355` 的 `consolidation_summary` confidence **0.6 → 0.7**（落在闸门上沿，让写入真正落地，同时全局质量闸对 extractor 仍生效）。`_raw_archive`（`memory.py:393`）0.5 同属"archive 意图却被静默丢"的同类——降级兜底的本意就是保证持久，**推荐一并提到 0.7**（决策点，见 §6）。

> 不改 `knowledge_search.py:153`。改写入点而非闸门，blast radius 最小且不污染 extractor 路径。

### 3.3 前置阻塞修复 — `updated_at` 时区往返（idle gate 依赖）

**故障**【代码确认】：idle gate 算 `now - session.updated_at`，但 `updated_at` 的写入/读取时区不一致，两处 skew：
1. Redis 正常 save 写 **local-naive** `datetime.now().isoformat()`（`manager.py:188`）；而 Lua LTRIM 写 **UTC-naive** `datetime.utcnow().isoformat()`（`memory.py:631`）。一次 token 压缩后，Redis meta 的 updated_at 变 UTC。
2. DB 列是 **tz-aware** `DateTime(timezone=True)` 且 `onupdate=_utcnow`（`models.py:86-87`，存 UTC），load 时 `.replace(tzinfo=None)` 砍成 **UTC-naive**（`manager.py:267`）。
两者都与 idle gate 的 local `datetime.now()` 比较 → 在 UTC+8 部署里，session 看起来"多 idle ~8 小时" → idle gate 几乎必触发，**部分抵消本 spec 要修的频繁压缩**。故为 C1 硬前置。

**推荐改法（统一到 aware-UTC，bounded 到持久边界 + gate）：**
| 点位 | 现状 | 改为 |
|---|---|---|
| `manager.py:188` Redis save | `datetime.now().isoformat()` | `datetime.now(timezone.utc).isoformat()` |
| `memory.py:631` Lua meta | `datetime.utcnow().isoformat()` | `datetime.now(timezone.utc).isoformat()`（aware） |
| `manager.py:159` Redis load | `fromisoformat(...)`（naive） | 同上；输出 aware（带 offset 时 fromisoformat 自然 aware） |
| `manager.py:267` DB load | `.replace(tzinfo=None)`（砍成 naive UTC） | 保留 tzinfo（已是 aware UTC），不砍 |
| idle gate 计算 | — | `datetime.now(timezone.utc) - updated_at`，两边 aware；负 delta → 视为活跃跳过（防残余 skew） |

封装两个 helper：`_utcnow_aware()` 与 `_as_aware_utc(dt)`（dt 已 aware 直接返回；naive 按 UTC 兜底 + 告警），idle gate 用后者吃任何来源的 `updated_at`，杜绝再现 skew。`created_at` 仅用于展示/age，不在本次范围。

> **【需进一步查→本轮验证掉】** PG 把 `update_meta`（`conversation_repo.py:102` 显式赋 `session.updated_at`）的 naive 值写进 timestamptz 列的实际时区语义。单测覆盖：local-naive / UTC-naive / aware-UTC 三种 `updated_at` 输入，idle gate 结果一致。

---

## 4. 三件事 × 逐触发器（哪个改动加在哪）

| | (a) 触发条件 | (b) 留尾 | (c) 推进时点 |
|---|---|---|---|
| **T1** | idle gate + 轮次计数（修法 A） | `pick_consolidation_boundary(tail_protect=N)`（新增，修法 A） | `= end_idx`（修法 A） |
| **T2** | 不改（token 爆该压就压）；仅 target 比例参数化 | 已有 `tail_protect=5`，提成可配 N | 已 `= end_idx`，不动 |
| **T3** | 不改（主动重置） | 不需要（整段归档） | `clear()` 归零 |

---

## 5. 数据流（Turn 起步示意）

```
背靠背 Turn（间隔秒级）:
  session = get_or_create  → updated_at = 上一轮(几秒前, aware-UTC)
  _check_pending_consolidation:
    now_utc - updated_at = 几秒 < IDLE(30min) → skip  ← 不再频繁压缩
  history = messages[last_consolidated:] = 完整原文 → "这篇" 锚点在 → 指代成功

idle > 30min 再来:
  now_utc - updated_at = 45min > IDLE
  pending_turns = 3 ≥ MIN_TURNS(2) → 触发
  boundary = pick(tail_protect=8) → end_idx
  consolidate messages[last_consolidated:end_idx]
    → 会话摘要 conf 0.7 写 user_memory（修法 B 后真正落地）
  last_consolidated = end_idx   ← 留尾 8 条原文仍在 history
```

---

## 6. 关键决策推荐值（用户拍板）

| 决策 | 现状 | 推荐 | 理由 |
|---|---|---|---|
| T1 计数对象 | 消息条数(`loop.py:541`) | **对话轮次** | 直接消除"工具消息爆炸→误触发"（问题 1） |
| `MIN_PENDING_TURNS` | 5（条） | **2（轮）** | 配 idle gate，阈值只为滤掉"idle 后仅 1 轮"的琐碎会话 |
| `TAIL_PROTECT` (N) | T1 无 / T2=5 | **8** | 含 2~3 工具调用的轮≈6-8 条；留 8≈最近 1~2 整轮，够指代消解；boundary 只在 user 边界切，tail 恒为整轮 |
| `IDLE_THRESHOLD` | 无 | **1800s** | 会话边界常识值；29min 回环风险由 T2 token 兜底 |
| 新增 `last_consolidation_at` 字段 | 无 | **不加** | idle gate 读 `updated_at` 已同时给"去重"+"不锁死"；`updated_at` 本就持久，无需新字段 |
| T2 token 阈值 | `estimated≥budget`，target=budget//2 | **沿用**（仅把 0.5 参数化） | budget//2 给足回落空间，无证据需调 |
| consolidation_summary confidence | 0.6(`memory.py:355`) | **0.7** | 落在 0.7 闸上沿让写入落地，不动全局闸、不污染 extractor 路径 |
| raw_archive confidence | 0.5(`memory.py:393`) | **0.7**（待拍板） | archive 本意是保证持久；同被静默丢。若想保留"兜底=低质不入库"语义则维持 0.5 |

---

## 7. 本轮做 / 不做

### 做
- 修法 A：T1 idle gate + 轮次计数 + 留尾 + end_idx（`loop.py:530-560`）。
- 修法 B：confidence gate（`memory.py:355`，可选含 `:393`）。
- 前置：`updated_at` 时区统一（`manager.py:188,159,267` + `memory.py:631` + gate helper）。
- 参数化：`TAIL_PROTECT`/`IDLE_THRESHOLD`/`MIN_PENDING_TURNS`/`TOKEN_TARGET_RATIO`。

### 不做（明确写入，留到以后单独一轮）
- **形态一**：consolidation prompt 加实体保留（`_CONSOLIDATION_SYSTEM_PROMPT` `memory.py:39-121`）。
- **形态二**：召回 query 扩展（`context.py:346` topic / `loop.py:779`）。
- **RECENT_TOPICS / 上一版修法 2A·2B**。
- per-uid Chroma collection、MCP 子进程 uid 透传、`default_uid="admin"` fallback。
- **反抖死代码**【代码确认，仅标注】`memory.py:567-576`：`savings_ratio<0.1` 只 log 无 return，外层 `:584` 无条件 return，反抖当前空操作——不在本轮范围，提醒勿误以为它在保护。

**为何能拆**：形态一/二改的是 `_CONSOLIDATION_SYSTEM_PROMPT` 和召回链路，**与本轮三处改动（loop.py T1 / memory.py:355 / 时区点位）零文件重叠**；它们彼此共享 prompt 与召回链路，必须打包一起做。本轮的留尾已修复"背靠背指代失败"最痛场景；形态一/二只对"idle>30min 真归档后"的远期召回有用，是不同场景增强。

---

## 8. Commit 切分（独立可 revert）

| Commit | 内容 | 文件:行 | revert 影响 |
|---|---|---|---|
| **C1** | `updated_at` 时区统一 + gate helper（**A 的前置**） | `manager.py:188,159,267`、`memory.py:631`、新 helper | 退回旧时区行为；不影响压缩逻辑 |
| **C2** | T1 改写（idle gate+轮次+留尾+end_idx） | `loop.py:530-560` | 退回旧 T1；T2 不受影响 |
| **C3** | confidence gate 修复 | `memory.py:355`（+`:393` 待拍板） | 退回 → 摘要重新被丢 |
| **C4** | 阈值参数化、T1/T2 共用 tail_protect | `loop.py`、`memory.py:561-562,590` | 回落写死默认值，行为不变 |

顺序：C1 → C2（C1 是 C2 前置）；C3、C4 独立，任意序。每个均可单独 revert。

### 配置点

| 名称 | 默认 | 暴露 | 触发器 |
|---|---|---|---|
| `STARTUP_CONSOLIDATION_IDLE_SECONDS` | 1800 | env/config | T1 |
| `STARTUP_MIN_PENDING_TURNS` | 2 | env/常量 | T1 |
| `CONSOLIDATION_TAIL_PROTECT` | 8 | env/常量 | T1+T2 |
| `TOKEN_CONSOLIDATION_TARGET_RATIO` | 0.5 | 常量 | T2 |
| `CONSOLIDATION_SUMMARY_CONFIDENCE` | 0.7 | 常量 | 写入 |

---

## 9. 风险

- **R1 时区残余 skew**：§3.3 未覆盖到的 datetime 来源。Mitigation：`_as_aware_utc` 兜底 naive→UTC + 告警；单测覆盖三种输入；负 delta 视为活跃。
- **R2 29min 回环**：idle 永不达成 → 旧会话不归档。Mitigation：T2 token 兜底。可接受。
- **R3 tail_protect=8 致 token 路径压不动**：一轮 >8 条且全在保护区 → boundary=None 该轮不压。Mitigation：T2 多轮循环（`_MAX_CONSOLIDATION_ROUNDS=5` `memory.py:405`）下轮可切；8 远小于触发窗口。
- **R4 轮次阈值=2 仍偏频**：调高 `STARTUP_MIN_PENDING_TURNS`（已配置化）。
- **R5 confidence 提到 0.7 让边缘摘要也入库**：本就是期望（让摘要可召回）；去重 `similarity_threshold=0.85`（`knowledge_search.py:146`）挡重复。

---

## 10. 验证

1. 单测：
   - idle gate：`now-updated_at < IDLE` 跳过；`> IDLE` 触发；**三种时区输入结果一致**（C1 关键）。
   - 轮次计数：pending 8 条但仅 1 user → `pending_turns=1<2` 不触发（**复现问题 1**）。
   - 留尾：触发后 `last_consolidated==end_idx`（非 len），`messages[last_consolidated:]` 仍含最近 N 条原文（**复现问题 3/4**）。
   - confidence：consolidation 写入 conf 0.7 → `write_user_memory_sync` 返回 `(≥1, *)`，向量库可 query 回（**复现形态三**）；extractor 的 0.6 项仍被挡。
   - boundary=None 不抛、mark seen。
2. e2e：Turn1 谈 X → 背靠背 Turn2 含"这篇" → 必须解出（留尾生效）；Turn1 谈 X → 倒拨 31min → Turn2 触发归档 → 摘要入库 → 后续可召回（confidence 生效）。
3. 监控：worker 日志 `Found N unconsolidated messages from previous session` 频率应大幅下降（仅 idle>30min 出现）；`KnowledgeSearch: wrote N user memories` 在 consolidation 后 N≥1（不再恒 0）。

---

**Spec self-review:**
- ✅ Placeholder scan：无 TBD/TODO。
- ✅ 证据分级：每条标 代码/日志/需查；T2 已正确处明确标注，未把"有接口"当"已实现"（confidence gate 即反例：有写入调用但被闸丢弃）。
- ✅ Scope：单 plan，4 commit 独立可 revert，参数全配置化；"不做"清单含文件:行，与本轮零文件重叠论证在案。
- ✅ 覆盖：a/b/c × T1/T2/T3 逐格；问题 1-4 + 形态三各自定根因落到具体行；confidence (a)(b) 两项查证 + updated_at 时区前置均查清给推荐。
- ⚠️ 单一【需进一步查】（PG naive→timestamptz 语义）已转为本轮单测验证项，非遗留到 plan。
