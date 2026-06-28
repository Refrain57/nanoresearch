# Consolidation Anchor Retention — Design

**Date:** 2026-06-28
**Status:** Design — pending user review
**Origin:** Real e2e test 2026-06-28 Turn 3 failed coreference resolution. Bug B (`SessionManager._redis_save` 双切片) 已修 (commit `3d126a51`)，但发现独立的 consolidation 设计问题：刚说完的话被 startup consolidation 一锅端，外层 Agent 看不到任何近期对话锚点。

## 1. Problem

外层 Agent 在 Turn 3 起步时看不到 Turn 1/2 的对话锚点，导致代词消解失败（用户问"这篇有和与nerf的对比吗？"，Agent 反问"您提到的'这篇'是指哪篇论文呢？"）。

### Root causes (两条独立链路)

**链路 1 — `history` 被 consolidation 切空**：
- `loop.py:771 history = session.get_history(max_messages=0)` 返回 `messages[last_consolidated:]`
- `_check_pending_consolidation` (`loop.py:530-560`) 在 `pending_count >= 5` 时整把吃掉所有未消化消息，把 `last_consolidated` 推到 `len(messages)`
- 触发条件不带 idle 判断：用户刚说完话就触发；token 才 8535/65536，远未到压力线也照压
- 结果：consolidation 之后 `history = []`，外层 Agent 只看到 system prompt + 当前 user message

**链路 2 — prompt 的 dynamic 段没有近期对话锚点**：
- `MEMORY.md` 只装"6 个月不变"的稳定事实——临时论文名定义上不进 (`memory.py:39-121` 的 `_CONSOLIDATION_SYSTEM_PROMPT`)
- `user_memory` (Chroma) 是 RAG 召回，query 是当前 user message（`context.py:114-126`），"这篇有和与nerf的对比吗？"不含 CityGaussianV2，召回不到对应 history_entry

## 2. Goals & Non-goals

### Goals
1. consolidation 之后的下一轮，外层 Agent 仍能从原始 history tail 看到最近 1-2 轮对话
2. 真正归档的内容（idle 后才 consolidate 的）仍能通过 prompt dynamic 段被下一次召回
3. 每个修法独立可 revert，可单独配置阈值

### Non-goals
- 不动 `maybe_consolidate_by_tokens` 主路径（token 真爆时该消化的还是要消化）
- 不动 MCP 子进程 uid 透传（已识别但属独立 spec）
- 不动 `user_memory` Chroma 切 per-uid collection（产品演进，非本次 bug 修复范畴）
- 不动 SessionManager 的 `default_uid="admin"` fallback

## 3. Design

### 3.1 修法 1: startup consolidation idle gating + tail protect

**改动位置**：`backend/nanoresearch/agent/loop.py:530-560` 的 `_check_pending_consolidation`

**新行为**：
```python
async def _check_pending_consolidation(self, session, agent_id=None):
    if session.key in self._startup_consolidated:
        return

    # NEW: idle gate
    IDLE_THRESHOLD = timedelta(minutes=30)
    if datetime.now() - session.updated_at < IDLE_THRESHOLD:
        self._startup_consolidated.add(session.key)
        return

    pending_count = len(session.messages) - session.last_consolidated
    if pending_count < 5:
        self._startup_consolidated.add(session.key)
        return

    # NEW: 用 pick_consolidation_boundary 留 tail
    boundary = self.memory_consolidator.pick_consolidation_boundary(
        session,
        tokens_to_remove=1,
        tail_protect=5,
    )
    if boundary is None:
        self._startup_consolidated.add(session.key)
        return

    end_idx, _ = boundary
    chunk = session.messages[session.last_consolidated:end_idx]
    if not chunk:
        self._startup_consolidated.add(session.key)
        return

    success = await self.memory_consolidator.consolidate_messages(
        chunk, agent_id=agent_id, uid=self._uid
    )
    if success:
        session.last_consolidated = end_idx  # ← 只推进到 boundary，留 tail
        await self.sessions.save(session)
        self._startup_consolidated.add(session.key)
```

**关键决策**：
- `IDLE_THRESHOLD = 30 分钟`：用户走开半小时以上才算"该归档了"
- `tail_protect = 5`：跟现有 `pick_consolidation_boundary` 默认对齐
- 复用 `pick_consolidation_boundary`（已存在，逻辑成熟），不写新的 boundary 选择代码
- 触发但找不到合法 boundary（如全部都在 tail 保护区内）时直接 mark seen 退出

**配置点**：`IDLE_THRESHOLD` 通过 env var 或 loop_config 暴露，默认 1800 秒。

### 3.2 修法 2 Part A: consolidation prompt 加 RECENT_TOPICS section

**改动位置**：`backend/nanoresearch/agent/memory.py:39-121` 的 `_CONSOLIDATION_SYSTEM_PROMPT`

**新增内容**（追加到现有模板）：

```markdown
### RECENT_TOPICS Section
- 列出本次对话涉及的具体实体（论文名、方法名、KB 标题、人物、专有名词）
- 每条 1 行，格式：`- {实体名} — {一句话上下文}`
- 滚动覆盖：保留最近 5 条，每次 consolidation 由 LLM 整段 rewrite
- 旧实体可保留（如仍相关），新对话提到的实体优先入

示例：
## RECENT_TOPICS
- CityGaussianV2 — 用户问的大规模场景重建方法，对比了 GauU-Scene/MatrixCity
- PGSR — 早期对比的 GS 变体，用于结构精度比较
```

**MEMORY.md 模板更新**：`RECENT_TOPICS` 段位于 `FOCUS_AREAS` 之后，跟 FACTS/USER_PROFILE/FOCUS_AREAS 同级。

**写入路径**：仍走现有 `MemoryStore.write_long_term(update)`，整文件 rewrite。

**注入路径**：自动通过 `<memory>` 段进入 prompt（`context.py:336-342` 已有 `<memory>` wrapper，整个 `MEMORY.md` 内容会被注入）。无需改 context.py。

### 3.3 修法 2 Part B: 召回 query 扩展

**改动位置**：`backend/nanoresearch/agent/loop.py` `_process_message` 里调用 `build_messages` 之前对 `topic` 参数的构造。

**当前**：
```python
# loop.py 大概 770-780
history = session.get_history(max_messages=0)
messages = self.context_builder.build_messages(
    history=history,
    current_message=msg.content,
    topic=msg.content,  # ← 召回 query 就是当前用户输入
    ...
)
```

**新行为**：
```python
def _build_recall_topic(history, current_msg, n=3):
    """Concat last N user messages + current message as recall query."""
    user_msgs = [m["content"] for m in history if m.get("role") == "user"][-n:]
    user_msgs.append(current_msg)
    return "\n".join(user_msgs)

topic = _build_recall_topic(history, msg.content, n=3)
```

**关键决策**：
- N = 3：拼最近 3 轮 user message
- 只拼 user role（跳过 assistant/tool）：避免噪音
- 当前 user message 总是最后一行
- 用 `\n` 拼接：vector embed 综合捕获
- 函数放在 `loop.py` 内作为私有方法 `self._build_recall_topic`

**配置点**：N 通过常量定义，可改但不暴露 env（小调优）。

## 4. Data Flow (Turn 3 起步示意)

```
Turn 3 起步:
  session = sessions.get_or_create(key)
    └─ Redis hit 返回 Session(messages=[m0..m9], last_consolidated=0, updated_at=<5分钟前>)

  _check_pending_consolidation(session):
    └─ now - updated_at = 5min < 30min IDLE_THRESHOLD
    └─ skip; mark seen

  history = session.get_history(0) = messages[0:] = [m0..m9]
    └─ Turn 1/2 全部原文都在

  topic = _build_recall_topic(history, current="这篇有和与nerf的对比吗？", n=3)
       = "查查看3dgs\n看看你说的，那个大规模...\n这篇有和与nerf的对比吗？"

  build_messages(history=history, topic=topic):
    system: [workspace + agent + dynamic]
      dynamic:
        <memory>
          FACTS: ...
          USER_PROFILE: ...
          FOCUS_AREAS: ...
          RECENT_TOPICS:        ← 即使现在还没攒进去（idle 未到），下次 Turn 4 起步 idle 触发后会有
        </memory>
        <history>
          (RAG 召回，topic 扩展后命中 user_memory)  ← consolidation 后才有
        </history>
    history: [m0..m9]            ← 完整 Turn 1/2 原文
    user: 这篇有和与nerf的对比吗？

  外层 Agent → 看到 m0..m9 原文里的 CityGaussianV2 → 指代消解成功
```

idle > 30min 之后再来 Turn：
```
  _check_pending_consolidation:
    └─ now - updated_at = 45min > 30min
    └─ pending_count = 10, 触发
    └─ pick_consolidation_boundary(tail=5) 返回 boundary=5
    └─ consolidate messages[0:5] → MEMORY.md 新增 RECENT_TOPICS, user_memory 写 history_entry
    └─ session.last_consolidated = 5
    └─ messages[5:] 仍保留 (Turn 2 末尾 + Turn 3 user)

  history = messages[5:]  ← 还有 5 条 tail
  topic = ...
```

## 5. Components & Files

### Modified
- `backend/nanoresearch/agent/loop.py`
  - `_check_pending_consolidation` 改逻辑（idle gate + tail protect）
  - `_process_message` 调用前组装新 topic（用 `_build_recall_topic` 私有方法）
- `backend/nanoresearch/agent/memory.py`
  - `_CONSOLIDATION_SYSTEM_PROMPT` 追加 RECENT_TOPICS section 说明

### New tests
- `backend/tests/unit/agent/test_check_pending_consolidation.py`
  - idle gate: now-updated_at < threshold 不触发
  - idle gate: now-updated_at > threshold 触发
  - tail protect: 触发后 last_consolidated == boundary（非 len(messages)）
  - boundary 找不到时不抛错、mark seen
- `backend/tests/unit/agent/test_consolidation_prompt.py`
  - prompt template 包含 RECENT_TOPICS 段说明
  - mock LLM 返回带 RECENT_TOPICS 的 memory_update → MemoryStore.write 后 MEMORY.md 含该 section
- `backend/tests/unit/agent/test_topic_expansion.py`
  - `_build_recall_topic(history, current, n=3)`：返回最近 3 条 user + current 的 \n 拼接
  - history 不足 3 条 user 时只拼现有的
  - history 全是 assistant/tool 时只返回 current

### Integration test
- `backend/tests/integration/test_consolidation_anchor_e2e.py`
  - Turn 1 谈 X → 模拟 session.updated_at 倒拨 31min → Turn 2 触发 startup consolidation → 验证 last_consolidated 推进但 tail 保留 → Turn 3 user 含代词 → 验证 build_messages 的 history 仍含 X 的原文，topic 包含最近 3 轮 user
  - 不调真 LLM，全程 mock provider

## 6. Error handling

- `pick_consolidation_boundary` 返回 None：mark seen 退出（已有逻辑），不抛
- consolidation LLM 失败：现有 `_fail_or_raw_archive` 机制接管，不影响 idle gate 路径
- `session.updated_at` 缺失/异常：fallback 当 0（即视为 idle 很久，仍按现有逻辑触发），不应阻塞 main path
- `_build_recall_topic` history 为空：返回单条 current_msg

## 7. Configuration

| 名称 | 默认 | 暴露方式 |
|---|---|---|
| `STARTUP_CONSOLIDATION_IDLE_SECONDS` | 1800 (30 min) | env var or loop_config |
| `STARTUP_CONSOLIDATION_TAIL_PROTECT` | 5 | 模块常量 |
| `RECALL_TOPIC_USER_TURNS` | 3 | 模块常量 |

## 8. Risks

- **R1 — RECENT_TOPICS 滚动覆盖丢历史实体**：LLM 决定保留哪些旧 entity 可能不稳定，旧重要实体被新对话挤掉。
  - Mitigation: 在 prompt 里加 "如果旧实体仍可能被后续提及，保留之"。长期还有 user_memory 兜底。
- **R2 — topic 拼接过长 → embedding 失真**：3 条 user message 合起来如果都很长会让 embedding 偏向某一条。
  - Mitigation: 每条截到 200 字符。
- **R3 — IDLE_THRESHOLD 误差导致 boundary 漏触发**：用户 29 分钟回来 + 29 分钟回来 ... session 永远不归档。
  - Mitigation: `maybe_consolidate_by_tokens` 主路径在 token 真正接近爆时仍会触发，这是兜底。
- **R4 — RECENT_TOPICS 内容跨 agent 串扰**：同一 uid 多 agent 用同一 MEMORY.md（agent_id=None 时）。
  - Mitigation: 这是现有设计问题，不在本次范畴；MEMORY.md 路径仍走 agent_id 切分。

## 9. Out of scope

- consolidation summary 写到 user_memory 后的召回稀释（Chroma 共享 collection）
- MCP 子进程 uid 透传
- worker pool 化 / `default_uid="admin"` fallback 撤除
- token budget 主路径 `maybe_consolidate_by_tokens` 调优
- `_CONSOLIDATION_SYSTEM_PROMPT` 的 FACTS/USER_PROFILE/FOCUS_AREAS 段不动

## 10. Migration / Rollback

- 修法 1/2A/2B 各自独立 commit，可单独 revert
- 现有 session 数据兼容：`updated_at` 已存在，无 schema 变更
- MEMORY.md 增加 RECENT_TOPICS 段对老的 reader 无害（按 markdown 解析）
- 紧急回滚：env var `STARTUP_CONSOLIDATION_IDLE_SECONDS=0` 退化为旧行为

## 11. Verification

落地后通过以下方式验证：
1. 单测：6 个新 unit test + 1 个 integration test 全过
2. 真实 e2e：复现 2026-06-28 故障场景
   - Turn 1 谈 CityGaussianV2 → 立刻 Turn 2 含"这篇" → 必须解出
   - Turn 1 谈 X → 等 30+ 分钟 → Turn 2 含"刚才那个" → 仍必须解出（验证 long-term 召回链路）
3. 监控：worker 日志里 `Found N unconsolidated messages from previous session` 频率应大幅下降

---

**Spec self-review (filled in by author):**

- ✅ **Placeholder scan**: 无 TBD / TODO / vague
- ✅ **Internal consistency**: 修法 1 的 boundary 推进与 `pick_consolidation_boundary` 现有逻辑一致；修法 2A 的 RECENT_TOPICS 注入路径复用现有 `<memory>` wrapper，不引入新注入点
- ✅ **Scope check**: 单一 plan 范围，3 个独立 commit 可分阶段实施
- ✅ **Ambiguity check**:
  - "RECENT_TOPICS 滚动 5 条" — 明确为 LLM 整段 rewrite，client 不截断
  - "topic N=3" — 明确只数 user role 消息
  - "IDLE_THRESHOLD" — 明确为 `datetime.now() - session.updated_at`，单位秒
