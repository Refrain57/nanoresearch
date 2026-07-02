# Cron 生产级重做 — 设计规格 (Spec, for review)

> **状态**: 待评审。这一轮只出设计 + Task 拆分 + 接口 + misfire 策略 + TDD 测试点,**不写实现代码体**。
> **评审顺序(用户指定)**: 先看下面「两个先决结论」→ 逐 Task 评审 → 确认后才 TDD。
> **目标**: 把现有 `nanoresearch.cron`(JSON + 单进程 asyncio 定时器)重做成在 **web/serve 生产部署**下正常工作的定时任务子系统,复用现有 P0/P1 基础设施,不再发明锁/持久化/异步执行。

---

## 0. 两个先决结论(请先评审这两条)

### 结论 ① — misfire 默认策略:`fire_once` + 宽限窗(coalescing within grace),任务级可配

**默认**:每个到点任务(不论错过多少次)在**恢复后最多补触发一次**,然后把 `next_run_at` 前滚到「下一个未来时刻」。是否补触发取决于**宽限窗**:

- 定义「错过」:`next_run_at < now - misfire_grace_s`(默认 `misfire_grace_s = 3600` 秒 = 1h)。
- **窗内**(`next_run_at >= now - grace`,进程只是短暂重启):**补触发一次**。
- **窗外**(`next_run_at < now - grace`,进程长期宕机):**不触发**,只记一条 `missed` 运行记录,然后前滚 / 删除。

分任务类型:

| 类型 | 窗内(recent) | 窗外(stale) |
|---|---|---|
| `at` 一次性 | 补跑一次 → 然后删除/禁用 | **不跑,标 `missed`,删除/禁用** ← 直接修掉现状问题 2(僵尸任务) |
| `every` / `cron` 重复 | 补跑一次(合并 N 次错过为 1 次)→ next 前滚到未来 | 不跑,标 `missed` → next 前滚到未来(不追 N 次) |

**依据**:
1. **不 spam**。重复任务停机 8 小时错过 96 次,补跑 96 个完整 agent turn = 成本爆炸 + 用户被 96 条通知轰炸 + LLM 限流。业界调度器(APScheduler `coalesce=True`、Quartz `MISFIRE_INSTRUCTION_FIRE_ONCE_NOW`)默认也是合并成一次。
2. **一条迟到的提醒有价值,一条过期太久的提醒是噪音**。1h 内迟到的「开会提醒」仍有用;两天后的就是垃圾 → 宽限窗把「有用的迟到」和「噪音」切开。
3. **一次性任务必须有终态**。现状 `at` 错过后 `next_run=None` 但 `enabled=True`,永远卡住(`cron/service.py:22-23` + `:296-301`)。新策略强制窗外 `at` 走「标 missed + 删除/禁用」,消灭僵尸。

**任务级可配** `misfire_policy` 字段(枚举):
- `fire_once`(默认)— 上表行为。
- `skip` — 恢复后**从不**补触发,直接前滚 / 禁用(适合低价值周期任务,例如「每小时刷新缓存」错过就错过)。
- `fire_all` — 补跑每一个错过的时刻,**硬上限 `CRON_FIRE_ALL_CAP = 10`**(适合极少数「每个 tick 都不能丢」的任务;`at` 天然单次,等价于 `fire_once`)。

> **需要你判断的点**:默认宽限窗 `3600s` 是否合理?`fire_all` 上限 `10` 是否合理?`misfire_policy` 三档够不够?

### 结论 ② — 执行身份接入:每个 cron 任务一条**专属 conversation**,`agent_lock` 天然与用户对话隔离,**不冲突**

**现状**:cron 用 `session_key=f"cron:{job.id}"` 的孤立会话,在 gateway 进程内直接 `agent.process_direct`(`cli/commands.py:678-683`),不进 mailbox。

**新设计**:cron 任务接入 mailbox → dispatcher → worker,身份如下:

- **conversation**:每个 cron 任务在**创建时**建一条**专属 `Conversation` 行**(`storage/models.py:73`),`uid` = 任务所有者,`agent_id` = 任务选定的 agent(可为 NULL),`metadata` 标记 `{"cron_job_id": <id>}`。其 `conversation_id` 存回 `cron_jobs.conversation_id`。
- **session_key** = `web:{cron_conversation_id}`(走 `_build_run_payload` 默认格式,`chat_router.py:478`)—— 与 dispatcher 的批处理门(`dispatcher.py:126` 查 `pending("web:"+conv_id)`)、P1 continuation 完全兼容,**零特判**。
- **agent_id** = 任务的 `agent_id`(或 `"none"`,与 Phase 0 默认一致,`chat_router.py:439`)。
- **锁** = `agent_lock:{agent_id}:{cron_conversation_id}`(`redis_keys.py:81`)。

**为什么不冲突**:cron 任务的 conversation 与任何交互式用户对话是**不同的 conversation_id** → `agent_lock` 键不同 → cron run 与用户聊天**永远不争同一把锁**,cron 也不会把一个意外 turn 插进用户正在进行的对话。而**同一个 cron 任务的两次触发**会 serialize 在它自己那把 conversation 锁上(这是**期望行为**:同一任务不该自我重叠执行)。若 cron run 内部 spawn 子 agent,P1 continuation 机制原样适用(它就是一个普通 run)。

> **需要你判断的点**:cron 专属 conversation 是否要在前端对话列表**隐藏**(靠 `metadata.cron_job_id` 过滤)?还是作为「定时任务记录」正常展示?

---

## 1. 现状与问题(带文件:行号)

现有 `nanoresearch.cron` = 「skill 文档 + agent tool + 单进程 asyncio 定时器 + JSON 文件」:

- Skill 文档:`backend/nanoresearch/skills/cron/SKILL.md`(纯提醒 / 动态任务 / 一次性 三语义)。
- Agent 工具:`backend/nanoresearch/agent/tools/cron.py:12`(`CronTool`,add/list/remove),注册于 `agent/loop.py:205-208`(仅当 `cron_service` 注入)。
- 调度服务:`backend/nanoresearch/cron/service.py:63`(`CronService`)。单进程自重排定时器 `_arm_timer:228` / `_on_timer:247`;`croniter` 仅用于算下次时间 `_compute_next_run:20-46`。
- 数据模型:`backend/nanoresearch/cron/types.py`(`CronJob/CronSchedule/CronPayload/CronJobState/CronStore`)。
- 持久化:`<workspace>/cron/jobs.json`,`_load_store:80` / `_save_store:141`。
- 执行:`on_cron_job`(`cli/commands.py:661-718`)→ `agent.process_direct(session_key="cron:{id}")`,gateway 进程内直接跑;deliver 门控 `evaluate_response`(`:707`)→ `bus.publish_outbound`(`:712`)。

### 致命问题(新设计必须解决)

1. 🔴 **web 下永不触发**。定时器只在 `gateway` 命令 `cron.start()`(`commands.py:807`)。`serve`(FastAPI,`commands.py:1392`)、ARQ worker(`build_loop_config`,`commands.py:474`→`worker.py:128`)、交互式 `agent`(`commands.py:866`)都只挂了 `cron` 工具、**从不 `start()`、从不设 `on_job`**。FastAPI lifespan(`server/main.py:33-80`)启动 dispatcher/reaper/watchdog 但**没有 cron**。→ web 用户设的定时任务静默落盘、永不执行。
2. **一次性 `at` 错过 → 僵尸**。重启 `_recompute_next_runs:211` 对过期 `at` 算出 `None`(`:22-23`),`enabled=True` 但 `next_run=None`,永不触发也永不删除。
3. **无 misfire 补偿**。`every` 重启重置为 `now+interval`(`:29`);`cron` 从当前往后取下一个 → 停机期间错过的 tick 全部丢弃。
4. 🔴 **无跨进程锁**。单进程一个 timer task。两个进程指向同一 `jobs.json` 会各自把每个任务触发一遍 → 重复执行。对比 web 路径有 `agent_lock`/`dist_lock`,cron 一把锁没有。
5. **长任务阻塞定时器**。`_on_timer:259-260` 顺序 `await` 每个到点任务的完整 agent turn,慢任务拖住其它 + 延后重排。
6. **JSON 非原子写**。`_save_store:192` 原地覆写;解析失败 fallback 空 store(`:133-135`)→ 静默丢全部任务。

---

## 2. 新架构总览:cron = 一个「定时触发的 dispatcher」

**核心洞察**:cron 调度器和现有 `AgentDispatcher` 是同一种东西,只是触发源不同。

| | AgentDispatcher(现有) | CronScheduler(新增) |
|---|---|---|
| 触发源 | **事件**:消费 notify 流(`dispatcher.py:88`) | **时间**:定期扫 DB 找到点任务 |
| 抢锁 | `dist_lock.acquire(agent_lock)`(`dispatcher.py:117`) | `dist_lock.acquire(cron_lock:{job_id})` |
| 派发 | `post` 已在 inbox → `arq.enqueue_job`(`dispatcher.py:137`) | 认领后 `post_message`+`post_notify` → 交给现有 dispatcher |
| 执行 | worker `run_agent_job`(`worker.py:368`) | **同一个 worker,同一条路径** |

新形态四点:
1. 任务存 **DB(`cron_jobs` 表)**,不再 JSON。
2. 一个轻量常驻 **`CronScheduler` 哨兵**(形如 `StuckRunWatchdog`,`heartbeat/stuck_run_watchdog.py:30-63`),在 `serve` 的 FastAPI lifespan 启动,定期扫 DB 找到点任务,认领后**只负责投进 mailbox**(不自己执行)。
3. 执行**完全复用现有 worker**——cron 任务和用户消息走同一套 mailbox→dispatcher→worker,同样的并发保护 / 崩溃恢复 / 即弃协程。顺便消灭「cron 是独立第二套 runtime」。
4. `cron` 工具(add/list/remove)与 `SKILL.md` 语义保留,只把底层从 JSON 换成 DB。

### 复用清单(全部已勘察确认签名)

| 生产级需要 | 复用 | 位置 |
|---|---|---|
| 分布式锁 | `dist_lock.acquire/refresh/release`(SET NX PX + token + Lua) | `bus/dist_lock.py:32/39/45` |
| 幂等去重 | `SET job:{id} NX EX` + `RedisKeys.job` | `chat_router.py:294` / `redis_keys.py:16` |
| 投递入信箱 | `mailbox.post_message` / `post_notify` | `bus/mailbox.py:30/37` |
| 派发执行 | 现有 `AgentDispatcher` + `run_agent_job` | `bus/dispatcher.py:108` / `worker.py:368` |
| run payload 构造 | `_build_run_payload` | `chat_router.py:450` |
| 常驻哨兵范式 | `StuckRunWatchdog`(bg asyncio loop + interval scan) | `heartbeat/stuck_run_watchdog.py:42-63` |
| lifespan 启动范式 | dispatcher/reaper/watchdog 启动块 | `server/main.py:57-80` |
| 表 + repo + 迁移范式 | model(`mapped_column`/JSONB/Index)+ 迁移脚本 + `check_schema_migrations` CHECKS + conftest truncate | `storage/models.py:103` / `scripts/migrate_agent_harness.py` / `database.py:61` / `tests/conftest.py:45` |
| deliver 门控 | `evaluate_response` | `utils/evaluator.py:53` |
| run 持久化 | `RunRepository.create/update` | `storage/repositories/run_repo.py:18/42` |

---

## 3. 关键设计点逐个回答

### 3.1 misfire 策略 → 见结论 ①。落地位置:纯函数 `cron/schedule.py::resolve_misfire`(Task 3),被哨兵调用。

### 3.2 哨兵扫描机制

- **扫描间隔**:`CRON_SCAN_INTERVAL_S` 默认 **30s**(与秒级 `every` 任务的最坏延迟一致;比现状 asyncio 精确到毫秒差,但对提醒类任务 30s 抖动可接受;可配)。
- **高效查到点任务**:`cron_jobs` 上建**复合索引 `(enabled, next_run_at)`**,扫描 SQL `WHERE enabled AND next_run_at <= now() ORDER BY next_run_at LIMIT N`(`CronJobRepository.list_due`,Task 2)。
- **扫描→认领→投递→前滚 的原子性**(避免「扫到但投递前崩 → 漏触发」/「投递后前滚前崩 → 重复」):见 3.3 的锁契约。核心 = **post-first + 确定性幂等键 + DB CAS 前滚**,把重复收敛为幂等去重,把漏触发窗口压到与现有 `create_run` 相同的亚秒级。

### 3.3 防重复触发的锁契约(哨兵 per-job 流程)

哨兵对每个到点 job 执行(**次序即契约**):

```
1. token = dist_lock.acquire(cron_lock:{job.id}, px=60_000)      # 同一 tick 内多哨兵去重
   若 token is None → 跳过(别的哨兵在处理这个 job)
2. 重读该 job(拿最新 next_run_at);observed = job.next_run_at
   若 observed 已 > now(别人已前滚) → release → 跳过
3. decision = resolve_misfire(policy, kind, observed, now, grace, ...)   # 结论①
4. 若 decision 要 fire:
   a. det_job_id = "cron-" + sha256(f"{job.id}:{observed_ms}")[:20]   # 确定性:同一 occurrence 唯一
   b. won = redis.set(job:{det_job_id}, run_id_placeholder, nx=True, ex=3600)   # 复用 chat_router:294 幂等
      若 not won → 该 occurrence 已被投递(跨锁过期的重试)→ 跳到 5(只前滚)
   c. run = RunRepository.create(conversation_id=job.conversation_id, uid=job.uid, agent_id=job.agent_id)
   d. redis.set(job:{det_job_id}, str(run.id), ex=3600)
   e. payload = build_cron_run_payload(factory, job, str(run.id))        # Task 5
   f. mailbox.post_message + post_notify                                 # 交给现有 dispatcher
5. cron_repo.advance(job.id, observed, decision.next_run, last_status=..., run_at=now)   # DB CAS:WHERE next_run_at = observed
   一次性且窗外/已跑完 → cron_repo.finish_one_shot(delete 或 disable)
6. dist_lock.release(cron_lock:{job.id}, token)
```

**三层保证**:
- `cron_lock:{job_id}`(Redis 单持有,`dist_lock`):同一 tick 内、多哨兵间,单个 job 只被一个哨兵处理。
- **确定性 occurrence 幂等键** `job:{sha(job_id, occurrence_ms)}`:即使 cron_lock 过期后另一个哨兵重试,同一 occurrence 也只入队一个 worker job(复用现有 `RedisKeys.job` + SET NX 机制)。**注意**:键含 `occurrence_ms`,所以「今天 9am」与「明天 9am」是不同键 —— 不会把下一次误判成重复。
- **DB CAS 前滚**(`UPDATE ... WHERE id=:id AND next_run_at=:observed`):`next_run_at` 只前滚一次,并定义下一 occurrence 何时可被认领;并发的第二个 UPDATE 命中 0 行 → no-op。

**崩溃语义**:
- 崩在 post 之后、advance 之前 → 下个 tick:cron_lock 已过期、`next_run_at` 未变 → 重进 → 同一 det 键 SET NX 失败 → **不重投** → 只补做 advance。✅ 无重复。
- 崩在 SET NX(4b)之后、post(4f)之前 → 亚秒级窗口,占位键已设但 run 未投 → 该 occurrence 丢失。**这与现有 `create_run` 的同一窗口同级**(`chat_router.py:294`SET NX → `:318`post 之间崩溃同样会孤儿化一个 run)。列为已知有界限制;后续可加「pending cron run reaper」补偿(本 spec 不含)。
- 崩在 advance 之后 → occurrence 已完成,`next_run` 在未来,不再触发。✅

对照现有 dispatcher 锁用法:`dispatcher.py:117` 抢锁 → `:126-129` 门控 → `:137` 入队 —— 本设计的哨兵是「时间版」的同构体。

### 3.4 执行路径接入 → 见结论 ②(专属 conversation + `web:{conv}` session_key + `agent_lock` 天然隔离)。

### 3.5 多进程 / 多 worker 下只触发一次

- **有几个哨兵**:每个 `serve`(FastAPI)副本在 lifespan 各起**一个** `CronScheduler`(与 dispatcher/watchdog 每副本一个一致,`server/main.py:73/78`)。N 副本 = N 个哨兵。
- **只触发一次的论证**:N 个哨兵同一 tick 都扫到同一到点 job,`dist_lock.acquire(cron_lock:{job.id})` 只有一个成功(SET NX 原子,`dist_lock.py:35`),其余拿 None 跳过;再叠加 occurrence 幂等键 + DB CAS,三层收敛为「恰好一次入队」。worker 侧不需改动(和处理用户消息一模一样)。

### 3.6 web 部署真正生效(直接修问题 1)

在 `server/main.py` lifespan 里,紧随 `StuckRunWatchdog`(`:77-80`)启动 `CronScheduler`:
```
app.state.cron_scheduler = CronScheduler(app.state.redis, app.state.session_factory)
await app.state.cron_scheduler.start()
```
并在 lifespan 收尾(`:96-101` 区)`await app.state.cron_scheduler.stop()`。→ 只要 `serve` + worker 在跑,web 用户设的定时任务就会触发。**这是本次重做的第一目标。**

### 3.7 deliver 门控保留(⚠️ 含一个现状暴露的跨进程缺口)

现状 deliver 门控在 gateway 进程内(`commands.py:706-716`):`evaluate_response` → `bus.publish_outbound`。**但 `MessageBus` 是进程内 `asyncio.Queue`(`bus/queue.py:8-18/28-30`)**——新路径下 run 在 **worker 进程**执行,worker 的 `bus` 是另一个进程的独立实例,`publish_outbound` 发出去**到不了** `serve` 进程的 channel manager。

**分两层落地**:
- **核心(默认,无跨进程依赖)**:cron run 的结果天然落在**专属 cron conversation**,web 用户在 UI 里就能看到该会话的历史。无需任何投递即「生效」。
- **外部渠道投递(deliver=True,如 Telegram/飞书)**:在 `run_agent_job` 成功路径(`worker.py:566` 附近)加一个 `cron_delivery` 处理:若 payload 带 `cron_delivery` 且 `deliver`,调用 `evaluate_response(response, task_context, provider, model)`(`evaluator.py:53`);为 True 时,**通过 Redis 跨进程 outbound 桥**投递(而非进程内 bus)。此桥是**现状架构缺口**(worker→channel 无跨进程 outbound 通道),作为 **Task 6** 单列并标风险。

---

## 4. Task 列表

> 每个 Task:文件(创建/修改 + 行号)+ 接口签名 + TDD 红→绿测试点。**本轮不写实现代码体**,确认后再 TDD。
> 全局约束:Redis 5.0.14(无 `XAUTOCLAIM`/无排他 `(` 区间,沿用 `mailbox.py:20-27` 的 `_next_stream_id` 手法);`backend/.venv`;测试 Redis DB 15、PG `nanoresearch_test`;`asyncio_mode=auto`;`decode_responses=True`。

### Task 1 — `cron_jobs` 表 + ORM 模型 + 迁移脚本 + schema 校验 + test truncate

**Files**:
- Modify: `backend/nanoresearch/storage/models.py`(追加 `CronJob` 模型,末尾)
- Create: `backend/scripts/migrate_cron_jobs.py`(幂等 CREATE TABLE + 索引)
- Modify: `backend/nanoresearch/storage/database.py:61`(`CHECKS` 加 `("cron_jobs", "id")`、`("cron_jobs", "next_run_at")`)
- Modify: `backend/tests/conftest.py:51`(`TRUNCATE` 列表加 `cron_jobs`)
- Test: `backend/tests/cron/test_cron_model.py`

**模型字段**(SQLAlchemy `mapped_column`,风格照 `models.py:103` AgentRun):
```
CronJob(__tablename__="cron_jobs"):
  id: UUID pk
  uid: str FK users.uid, index
  agent_id: UUID | None FK agents.id ondelete SET NULL
  conversation_id: UUID | None FK conversations.id ondelete SET NULL   # 专属会话(结论②)
  name: str
  enabled: bool default True
  schedule_kind: str            # 'at' | 'every' | 'cron'
  schedule_at: datetime(tz) | None       # 一次性绝对时刻
  schedule_every_s: int | None
  schedule_expr: str | None
  schedule_tz: str | None
  message: str (Text)
  misfire_policy: str default 'fire_once'   # 'fire_once' | 'skip' | 'fire_all'
  misfire_grace_s: int default 3600
  deliver: bool default False
  deliver_channel: str | None
  deliver_to: str | None
  next_run_at: datetime(tz) | None, index          # 扫描键
  last_run_at: datetime(tz) | None
  last_status: str | None        # 'ok' | 'error' | 'missed' | 'skipped'
  last_error: str | None (Text)
  run_history: list JSONB default list             # 截断 20 条
  delete_after_run: bool default False
  created_at / updated_at: datetime(tz)
  __table_args__ = (Index("ix_cron_jobs_enabled_next", "enabled", "next_run_at"),)
```

**TDD**:
- `test_cron_jobs_table_created`:`create_tables()` 后 `information_schema` 有 `cron_jobs` 且含 `next_run_at`。红:无表/无列;绿:模型 + 迁移。
- `test_cron_job_roundtrip`:插入一行(`schedule_kind='cron'`, expr, tz)→ 读回字段一致,`run_history` 默认 `[]`。
- `test_migrate_cron_jobs_idempotent`:对已存在表跑迁移脚本 → 打印 skip、不抛。

### Task 2 — `CronJobRepository`(含原子认领 CAS)

**Files**:
- Create: `backend/nanoresearch/storage/repositories/cron_repo.py`
- Test: `backend/tests/cron/test_cron_repo.py`

**接口**(照 `run_repo.py:14` 范式,注入 `async_sessionmaker`):
```
class CronJobRepository:
    async def create(self, *, uid, name, agent_id, conversation_id,
                     schedule_kind, schedule_at=None, schedule_every_s=None,
                     schedule_expr=None, schedule_tz=None, message,
                     misfire_policy="fire_once", misfire_grace_s=3600,
                     deliver=False, deliver_channel=None, deliver_to=None,
                     next_run_at, delete_after_run=False) -> CronJob
    async def get(self, job_id: uuid.UUID) -> CronJob | None
    async def list_by_uid(self, uid: str, *, include_disabled=False) -> list[CronJob]
    async def list_due(self, now: datetime, *, limit: int = 100) -> list[CronJob]   # WHERE enabled AND next_run_at<=now ORDER BY next_run_at
    async def advance(self, job_id, observed_next_run_at, new_next_run_at, *,
                      last_status, last_error=None, run_at, run_record: dict | None) -> bool   # CAS: WHERE next_run_at=observed → rowcount>0
    async def finish_one_shot(self, job_id, observed_next_run_at, *, delete: bool,
                              last_status, run_at, run_record: dict | None) -> bool             # CAS delete 或 disable
    async def remove(self, job_id: uuid.UUID) -> bool
    async def set_enabled(self, job_id: uuid.UUID, enabled: bool) -> CronJob | None
```

**TDD**(用 `make_factory()` + `truncate_all()`):
- `test_list_due_returns_only_past_enabled`:插 3 行(过去/未来/disabled)→ `list_due(now)` 只返回过去且 enabled 的。
- `test_advance_cas_wins_once`:两次以**同一** `observed` 调 `advance` → 第一次 True、第二次 False(CAS 只成一次)。← 3.3 核心保证。
- `test_advance_appends_run_history_capped`:连续 advance 25 次 → `run_history` 长度 == 20(保留最近)。
- `test_finish_one_shot_delete_removes_row` / `test_finish_one_shot_disable_keeps_row_next_null`。

### Task 3 — 调度数学纯函数 `cron/schedule.py`(替代 `service.py:_compute_next_run`)

**Files**:
- Create: `backend/nanoresearch/cron/schedule.py`
- Test: `backend/tests/cron/test_cron_schedule.py`

**接口**(纯函数,无 I/O,含时区):
```
def compute_next_run(kind, *, at=None, every_s=None, expr=None, tz=None,
                     after: datetime) -> datetime | None
@dataclass
class MisfireDecision:
    action: Literal["fire", "skip"]
    fire_count: int          # fire_all 时 >1,上限 CRON_FIRE_ALL_CAP=10;否则 0/1
    next_run: datetime | None
    missed: bool
def resolve_misfire(policy, kind, scheduled: datetime, now: datetime, grace_s, *,
                    every_s=None, expr=None, tz=None) -> MisfireDecision
```

**TDD**:
- `test_compute_next_every` / `test_compute_next_cron_with_tz`(`croniter` + `ZoneInfo`,对齐 `service.py:31-42` 逻辑)/ `test_compute_next_at_past_returns_none`。
- `test_misfire_within_grace_fires_once`:`scheduled = now-10min`, grace 3600 → `action='fire'`, `fire_count==1`, `missed==False`, `next_run>now`。
- `test_misfire_outside_grace_skips_and_rolls_forward`:`scheduled=now-3h`, grace 3600 → `action='skip'`, `missed==True`, `next_run>now`。
- `test_misfire_policy_skip_never_fires`。
- `test_misfire_policy_fire_all_capped_at_10`:错过 50 次 → `fire_count==10`。
- `test_misfire_at_outside_grace_marks_missed_next_none`(一次性窗外:`next_run is None`,交由 finish_one_shot 删除/禁用)。
- `test_cron_dst_boundary`(America/Vancouver 春季跳变附近下一个时刻正确)。

### Task 4 — `CronScheduler` 哨兵(定时 dispatcher)

**Files**:
- Create: `backend/nanoresearch/cron/scheduler.py`
- Test: `backend/tests/cron/test_cron_scheduler.py`

**接口**(形如 `StuckRunWatchdog`,`stuck_run_watchdog.py:30-63`;`dispatch_fn` 可注入以便测试):
```
class CronScheduler:
    def __init__(self, redis, session_factory, *, interval_s: int = 30,
                 lock_px_ms: int = 60_000,
                 dispatch_fn: Callable[[Any, dict], Awaitable[None]] | None = None)
        # dispatch_fn 默认 = mailbox 投递(post_message + post_notify);测试传桩
    async def start(self) -> None          # create_task(self._run())
    async def stop(self) -> None
    async def _run(self) -> None           # while running: _scan_once(); sleep(interval_s)
    async def _scan_once(self) -> None     # list_due → 逐个 _dispatch_due_job
    async def _dispatch_due_job(self, job: CronJob) -> str   # 返回 "fired"|"skipped"|"locked"|"deduped"(测试/telemetry)
```
`_dispatch_due_job` 内即 3.3 契约:`dist_lock.acquire(cron_lock:{id})` → 重读 → `resolve_misfire` → (fire 分支)确定性 `job:{det}` SET NX → `RunRepository.create` → `build_cron_run_payload` → `mailbox.post_message`+`post_notify` → `cron_repo.advance`/`finish_one_shot` → `release`。

**TDD**(`redis_client` DB15 + `make_factory`;`dispatch_fn` 用记录型桩计数):
- `test_due_job_dispatched_once`:1 个到点 job → `_scan_once` 后 dispatch 桩被调用 1 次、`advance` 生效(`next_run` 前滚)。
- `test_two_schedulers_fire_once`:并发跑两个 `CronScheduler._scan_once`(同一 redis+DB)→ dispatch 桩总计 1 次(cron_lock + 幂等键)。← 3.5。
- `test_crash_between_post_and_advance_no_double`:第一次调用后**跳过 advance**(模拟崩)→ 再 `_scan_once` → 同 occurrence det 键使 dispatch 不重复(桩仍 1 次),仅补 advance。← 3.3 崩溃语义。
- `test_stale_job_applies_misfire_skip`:`next_run` 远早于 now(窗外)→ 不 fire(桩 0 次)、`last_status='missed'`、`next_run` 前滚。
- `test_one_shot_fired_then_deleted`:`kind='at'` 窗内 → fire 1 次 → 行被删(`delete_after_run=True`)。
- `test_disabled_job_not_dispatched`。

### Task 5 — cron run payload 构造 + 专属 conversation

**Files**:
- Create: `backend/nanoresearch/cron/payload.py`(`build_cron_run_payload`)
- Modify: `backend/nanoresearch/server/routers/chat_router.py`(复用 `_build_run_payload:450`;如需在其它模块复用,提取为可 import 的 helper——**不改其签名**)
- Test: `backend/tests/cron/test_cron_payload.py`

**接口**:
```
async def build_cron_run_payload(factory, job: CronJob, run_id: str) -> dict
    # 复用 chat_router._build_run_payload(factory, str(job.conversation_id), job.uid,
    #     content=_wrap_cron_task(job), run_id=run_id, agent_id=str(job.agent_id) or None)
    # 再注入 payload["cron_delivery"] = {"deliver": job.deliver, "channel": job.deliver_channel,
    #     "to": job.deliver_to, "task_context": job.message}
def _wrap_cron_task(job: CronJob) -> str    # 复刻 commands.py:667-671 的 "[Scheduled Task] ..." 包装
```

**TDD**:
- `test_payload_targets_cron_conversation`:`session_key == f"web:{job.conversation_id}"`、`agent_id == job.agent_id`。
- `test_payload_wraps_task_message`:`content` 含 "[Scheduled Task]" 与 `job.message`。
- `test_payload_carries_cron_delivery`:`cron_delivery.deliver/channel/to/task_context` 正确。

### Task 6 — deliver 门控(worker 侧)+ 跨进程 outbound 桥 ⚠️

**Files**:
- Create: `backend/nanoresearch/bus/outbound_bridge.py`(Redis 跨进程 outbound:`publish_outbound(redis, OutboundMessage)` → Redis Stream `outbound_notify`;serve 进程 channel 侧 `consume` 并 `bus.publish_outbound` 落回本进程 channel manager)
- Modify: `backend/nanoresearch/worker.py:566`(成功路径:若 `cron_delivery` 存在且 `deliver` → `evaluate_response` → 桥投递)
- Modify: `backend/nanoresearch/server/main.py`(lifespan 起一个 outbound 桥消费协程,把跨进程 outbound 交给 `channel_loop` 的 bus)
- Test: `backend/tests/cron/test_cron_delivery.py`、`backend/tests/bus/test_outbound_bridge.py`

**接口**:
```
# outbound_bridge.py
async def publish_outbound(redis, msg: OutboundMessage) -> None          # XADD outbound_notify
async def consume_outbound(redis, *, block_ms=5000) -> list[OutboundMessage]
# worker.py 内新增 helper
async def _maybe_deliver_cron_result(redis, cron_delivery: dict, response: str,
                                     provider, model) -> None            # gate → 桥投递
```

**TDD**:
- `test_deliver_gate_true_publishes`(桩 `evaluate_response`→True)→ `outbound_notify` 有一条。
- `test_deliver_gate_false_suppresses`→ 无。
- `test_no_cron_delivery_unchanged`:普通用户 run(无 `cron_delivery`)→ worker 行为不变(回归保护)。
- `test_outbound_bridge_roundtrip`:publish → consume 得回等价 `OutboundMessage`。

> ⚠️ **风险**:此 Task 触及 worker→channel 跨进程投递这一**现状缺口**(`MessageBus` 仅进程内,`queue.py:8-18`)。需先确认当前 channel 出站是否已有任何跨进程通道;若无,`outbound_bridge` 是新增基础设施,体量与风险高于其它 Task。**可作为二期**:一期先让 cron 结果落 cron conversation(核心已生效),deliver-to-channel 后置。

### Task 7 — `CronTool` 改写为写 DB(add/list/remove)+ 建专属 conversation

**Files**:
- Modify: `backend/nanoresearch/agent/tools/cron.py`(`CronTool.__init__` 改注入 `CronJobRepository`+`ConversationRepository`+`uid`,替代 `CronService`;`_add_job:126`/`_list_jobs:215`/`_remove_job:227` 改走 repo;保留 `_in_cron_context` 防自调度守卫 `:117-118`、tz 校验 `:36-43`、at/every/cron 解析 `:144-168`)
- Modify: `backend/nanoresearch/agent/loop.py:205-208`(工具注册:从 `cron_service` 改为注入 repo/uid;保留 `set_context` 传 channel/chat_id,`loop.py:268-276`)
- Modify: `backend/nanoresearch/skills/cron/SKILL.md`(用户语义不变;补一句「一次性任务错过超过宽限窗将不再触发」)
- Test: `backend/tests/cron/test_cron_tool_db.py`(迁移并改写 `tests/cron/test_cron_tool_list.py` 现有断言)

**接口**:
```
class CronTool(Tool):
    def __init__(self, cron_repo: CronJobRepository, conv_repo: ConversationRepository,
                 uid: str, default_timezone: str = "UTC")
    # name/description/parameters 不变(add|list|remove,同 cron.py:56-103)
    async def execute(self, action, message="", every_seconds=None, cron_expr=None,
                      tz=None, at=None, job_id=None, **kw) -> str
    # add: 校验 → compute_next_run → conv = conv_repo.create(uid, agent_id, metadata={"cron_job_id":...})
    #      → cron_repo.create(..., conversation_id=conv.id, next_run_at=...)
```

**TDD**:
- `test_add_creates_job_and_conversation`:`add(every_seconds=1200,...)` → `cron_jobs` 一行 + 一条带 `metadata.cron_job_id` 的 conversation。
- `test_add_computes_next_run`:`next_run_at` = now + 1200s(±容差)。
- `test_list_reads_from_db` / `test_remove_deletes_row`。
- `test_self_schedule_blocked`:`_in_cron_context=True` 时 `add` 返回错误(复刻 `cron.py:117-118`)。
- `test_tz_only_with_cron_expr` / `test_unknown_tz_rejected`(保留 `cron.py:138-142`)。

### Task 8 — 接入 serve lifespan + 退役 gateway 内嵌 cron

**Files**:
- Modify: `backend/nanoresearch/server/main.py`(lifespan `:77-80` 后起 `CronScheduler`;`:96-101` 区停)
- Modify: `backend/nanoresearch/cli/commands.py`:`serve`(`:1392` 区)构造 `CronJobRepository`,注入给 loop 用的 `CronTool`;移除对 `CronService` 的依赖;`gateway`(`:661-718` `on_cron_job`、`:807` `cron.start()`、`:824` `cron.stop()`)移除旧内嵌 cron 路径。
- Test: `backend/tests/cron/test_cron_scheduler_lifecycle.py`

**TDD**:
- `test_scheduler_start_stop`:`start()` 建 task、`stop()` 取消且不抛(镜像 `stuck_run_watchdog` 生命周期)。
- `test_serve_lifespan_boots_cron_scheduler`(轻量:mock lifespan 依赖,断言 `app.state.cron_scheduler` 被 start)。
- 回归:`gateway` 移除旧路径后启动不报错(冒烟)。

> **需要你判断的点(gateway 去向)**:新执行路径依赖 dispatcher+worker(Redis+ARQ),而 `gateway` 命令(`commands.py:575`)不跑这套(它 `agent.run()` 进程内执行)。选项:(A)`gateway` 一并起 dispatcher+worker(改动大);(B)`gateway` 的 cron 直接退役,生产 cron 只在 `serve`+worker 下工作(推荐,契合你「要能 web 部署」的目标)。本 spec 默认 (B),在 §6 标为风险。

### Task 9 — 数据迁移 jobs.json → cron_jobs

**Files**:
- Create: `backend/scripts/migrate_cron_jobs_from_json.py`
- Test: `backend/tests/cron/test_cron_json_migration.py`

**接口**:
```
async def migrate_jobs_json(json_path: Path, factory, *, default_uid: str) -> int
    # 读 <workspace>/cron/jobs.json(格式见 service.py:92-131)→ 逐 job:
    #   建 conversation(uid=default_uid, metadata cron_job_id)→ cron_repo.create
    #   映射:atMs→schedule_at, everyMs→schedule_every_s, expr/tz→schedule_*,
    #        payload.message→message, deliver/channel/to→deliver_*,
    #        misfire_policy 缺省 'fire_once', grace 3600;重算 next_run_at
    # 幂等:按 (uid, name, schedule 指纹) 去重
```

**TDD**:
- `test_migrate_sample_jobs`:构造含 3 类 schedule 的 jobs.json → 迁出 3 行 + 3 conversation,字段映射正确。
- `test_migrate_idempotent`:重复跑不产生重复行。
- `test_migrate_missing_file_noop`:文件不存在 → 返回 0、不抛。

### Task 10 — 退役旧 CronService / JSON 路径(清理)

**Files**:
- Delete: `backend/nanoresearch/cron/service.py`、`backend/nanoresearch/cron/types.py`(字段已并入 ORM 模型)
- Modify: `backend/nanoresearch/cron/__init__.py`(导出改为 `CronScheduler` / `CronJobRepository`)
- Modify: `backend/nanoresearch/dashboard/server.py:166`(`_read_cron_jobs` 从读 JSON 改为读 DB;或标记 dashboard 该视图待迁移)
- Modify: `backend/nanoresearch/cli/commands.py:558`(移除 `_migrate_cron_store`,由 Task 9 脚本替代)
- Delete/迁移: `backend/tests/cron/test_cron_service.py`、`tests/agent/test_loop_cron_timezone.py`(逻辑迁往 Task 3/4 的新测试)

**TDD**:
- 全量 `pytest backend/tests/cron -q` 绿。
- `grep -rn "CronService\|cron/jobs.json\|_migrate_cron_store"` 反向确认无残留引用(除迁移脚本)。

---

## 5. 迁移方案

1. **schema**:上线前跑 `backend/scripts/migrate_cron_jobs.py`(幂等 CREATE TABLE + 索引);`check_schema_migrations`(`database.py:52`)加 `cron_jobs` CHECKS 后,缺表会 fail-fast。
2. **数据**:跑 `migrate_cron_jobs_from_json.py`(Task 9),把 `<workspace>/cron/jobs.json` 里的存量任务导入 `cron_jobs` + 建专属 conversation。`default_uid` 取单机安装的 admin uid。
3. **切换**:部署新 `serve` + worker → lifespan 自动起 `CronScheduler` → 存量与新任务开始正常触发。
4. **回滚**:新旧持久化不同介质(JSON vs DB),`serve` 不动 JSON;回滚到旧 `gateway` 仍可读原 `jobs.json`(未删),无数据破坏。确认稳定后再执行 Task 10 删除旧文件与 JSON。

---

## 6. 风险与体量结论

**风险**:
- 🟠 **Task 6 跨进程 outbound 桥**是最大不确定项——`MessageBus` 仅进程内(`queue.py:8-18`),worker→channel 投递是现状缺口。**建议分期**:一期不含外部渠道投递(cron 结果落 cron conversation 即生效);二期补 outbound 桥。这样一期就直接修掉「web 永不触发」这个致命问题。
- 🟠 **gateway 去向**(Task 8 决策点):默认退役 gateway 内嵌 cron,生产 cron 只在 `serve`+worker 下工作。若你仍要单进程 `gateway` 跑 cron,需让它也起 dispatcher+worker(体量翻倍)。
- 🟡 **亚秒级漏触发窗口**(3.3):与现有 `create_run` 同级、有界;需要更强保证时后加「pending cron run reaper」(不在本 spec)。
- 🟡 **扫描延迟**:30s 间隔下秒级 `every` 任务最坏迟 30s;对提醒/任务类可接受,若需更准可下调 `CRON_SCAN_INTERVAL_S`。
- 🟢 **锁/幂等/执行/持久化全复用**现成基础设施,新增面主要是 `cron_jobs` 表 + 一个哨兵 + 纯函数调度数学,与现有 dispatcher/watchdog 同构,风险可控。

**体量**(粗估):
- 一期(Task 1–5, 7–9,不含 Task 6 外部投递 + Task 10 清理):~9 个新文件/改文件,核心逻辑集中在 `cron/scheduler.py` + `cron/schedule.py` + `cron_repo.py`,其余是薄接线。**中等体量**,单人 TDD 数日可完成。
- Task 6(跨进程 outbound)+ Task 10(清理)另计,建议二期。

**一句话**:把 cron 从「gateway 专属的进程内 JSON 定时器」改成「serve lifespan 里常驻的时间触发 dispatcher」——扫 DB 找到点任务、`cron_lock` + occurrence 幂等键 + DB CAS 保证跨进程恰好一次、投进现有 mailbox→dispatcher→worker 执行,一次性解决 web 不触发 / 僵尸 / 无 misfire / 重复触发 / 阻塞 / JSON 损坏 六个问题。

---

## 附:Spec 自检(对照 §3 关键点)

- 结论① misfire → §3.1 + Task 3 ✅ 结论② 身份 → §3.4 + Task 5 ✅
- 扫描机制 §3.2 → Task 2/4 ✅ 锁契约 §3.3 → Task 2(CAS)+ Task 4(哨兵)✅
- 多进程一次 §3.5 → Task 4 `test_two_schedulers_fire_once` ✅
- web 生效 §3.6 → Task 8 ✅ deliver §3.7 → Task 6 ✅
- 迁移 §5 → Task 9 ✅ 清理 → Task 10 ✅
- 六个致命问题:①web不触发→Task8;②僵尸→Task3;③无misfire→Task3;④重复→Task2/4;⑤阻塞→Task2/4(哨兵只认领不执行,执行在worker);⑥JSON损坏→Task1(DB)✅
