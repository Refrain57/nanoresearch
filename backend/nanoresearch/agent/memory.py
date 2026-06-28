"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

# Atomically trim the session:msg list to start at the new compaction boundary,
# then update updated_at in session:meta.
# KEYS[1]=msg_key  KEYS[2]=meta_key
# ARGV[1]=keep_from_idx (number of entries to drop from front)
# ARGV[2]=ISO timestamp
_LUA_LTRIM = """
local keep = tonumber(ARGV[1])
local len  = redis.call('LLEN', KEYS[1])
if len > keep then
    redis.call('LTRIM', KEYS[1], keep, -1)
else
    redis.call('DEL', KEYS[1])
end
redis.call('HSET', KEYS[2], 'updated_at', ARGV[2])
return 1
"""

from loguru import logger

CONSOLIDATION_TAIL_PROTECT = int(os.environ.get("CONSOLIDATION_TAIL_PROTECT", "8"))
TOKEN_CONSOLIDATION_TARGET_RATIO = float(os.environ.get("TOKEN_CONSOLIDATION_TARGET_RATIO", "0.5"))
CONSOLIDATION_SUMMARY_CONFIDENCE = float(os.environ.get("CONSOLIDATION_SUMMARY_CONFIDENCE", "0.7"))

from nanoresearch.utils.helpers import as_aware_utc, ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanoresearch.providers.base import LLMProvider
    from nanoresearch.session.manager import Session, SessionManager


_CONSOLIDATION_SYSTEM_PROMPT = r"""You are a memory consolidation agent. Analyze the conversation and update the memory following the exact format below.

## 内容分类规则（关键）

### 写入 MEMORY.md（稳定事实，6个月后仍成立）
- 用户偏好：语言偏好、工具偏好、工作习惯
- 环境约定：工作目录、API 配置、模型选择
- 长期决策：架构决策、技术选型
- 用户画像：角色、背景、专业领域

### 不写入 MEMORY.md（临时内容）
- 任务进度：当前任务状态、待办事项
- 讨论结论：本次讨论的结论、发现
- 临时焦点：当前调试目标、短期关注点
- 工具调用细节：具体的搜索结果、代码片段

判断标准：这条信息 6 个月后还成立吗？
→ 成立 → 写入 MEMORY.md
→ 不成立/不确定 → 只写入 history_entry，不进 MEMORY.md

## Output Format for save_memory

### memory_update（MEMORY.md）
只包含稳定事实，格式：

```markdown
# User Memory

## FACTS
- 用户偏好 Python
- 工作目录: D:\Code\nanoresearch
- 使用 Claude 模型

## USER_PROFILE
资深工程师，专注 AI Agent 开发。

## FOCUS_AREAS
- AI Agent 架构设计
- RAG 系统优化
```

### history_entry（HISTORY.md）
结构化摘要，格式：

```markdown
## Session Summary [YYYY-MM-DD HH:MM]
- Active Task: 当前正在进行的任务（如有）
- Completed Actions: 已完成的操作（简要）
- Key Decisions: 做出的关键决策
- Tools Used: 使用的工具列表
- Blocked/Issues: 遇到的阻碍或问题
- Stable Facts: 可进入 MEMORY.md 的事实（如有新发现）
```

注意：
- 临时任务结论不要写入 memory_update 的 FACTS 或 FOCUS_AREAS
- history_entry 使用结构化字段，便于后续解析
- 如果某字段无内容，写 "无" 或跳过该字段

## Memory Update Rules

### FACTS Section
- 只添加稳定事实（用户偏好、环境约定、长期决策）
- 移除被否定/过时的事实
- 每条一行，grep 可搜索
- 不重复已有事实

### USER_PROFILE Section
- 用户透露的新信息时更新
- 移除过时信息
- 最多 3 句

### FOCUS_AREAS Section
- 只保留长期关注点（非临时任务）
- 最多 5 个
- 如果本次对话没有新的长期焦点，保持原有不变

### history_entry
- 以 ## Session Summary [YYYY-MM-DD HH:MM] 开头
- 使用固定字段格式
- 关键词 grep 可搜索

Call the save_memory tool with your consolidation."""


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown with fixed sections: "
                        "FACTS (bullet list), USER_PROFILE (max 3 sentences), FOCUS_AREAS (bullet list, max 5). "
                        "Include all existing content plus new information. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Long-term memory stored in MEMORY.md."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path, knowledge_search: Any = None, agent_id: str | None = None):
        if agent_id:
            self.memory_dir = ensure_dir(workspace / "agents" / agent_id / "memory")
        else:
            self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self._consecutive_failures = 0
        self._cached_hash: str | None = None
        self._knowledge_search = knowledge_search

    def get_content_hash(self) -> str:
        """Calculate stable SHA-256 hash of memory content.

        Used to detect whether memory has changed between requests,
        enabling efficient cache invalidation.
        """
        content = self.read_long_term()
        if not content:
            return "empty"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def has_changed(self, last_hash: str | None) -> bool:
        """Check if memory has changed since last known hash.

        Args:
            last_hash: Previously recorded hash, or None if no prior record.

        Returns:
            True if memory content differs from last_hash.
        """
        if last_hash is None:
            return True
        return self.get_content_hash() != last_hash

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def get_memory_context(self) -> str:
        """Return raw MEMORY.md content for XML wrapping in context builder."""
        return self.read_long_term()

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """Format messages for consolidation with full tool context."""
        lines = []
        for message in messages:
            timestamp = message.get('timestamp', '?')[:16]
            role = message.get('role', 'unknown')
            content = message.get('content', '')

            if role == "tool":
                # 工具返回：显示工具名和调用ID
                tool_name = message.get("name", "unknown_tool")
                tool_call_id = message.get("tool_call_id", "")[:8]
                # 裁剪长输出
                if len(content) > 500:
                    content = content[:200] + "\n...[truncated]...\n" + content[-200:]
                lines.append(f"[{timestamp}] TOOL({tool_name})[{tool_call_id}]: {content}")

            elif role == "assistant":
                # 助手消息：显示 tool_calls 详情
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        tc_func = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                        tc_name = tc_func.get("name", "unknown")
                        tc_args = tc_func.get("arguments", "{}")
                        if isinstance(tc_args, dict):
                            tc_args = json.dumps(tc_args, ensure_ascii=False)
                        # 截断参数避免过长
                        args_preview = tc_args[:100] + "..." if len(tc_args) > 100 else tc_args
                        lines.append(f"[{timestamp}] CALL {tc_name}({args_preview})")
                if content:
                    lines.append(f"[{timestamp}] ASSISTANT: {content}")

            elif role == "user":
                lines.append(f"[{timestamp}] USER: {content}")

            else:
                lines.append(f"[{timestamp}] {role.upper()}: {content}")

        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        uid: str | None = None,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""## Current Memory
{current_memory or "(empty)"}

## New Conversation to Process
{self._format_messages(messages)}

Call save_memory with your updated memory following the exact format specified."""

        chat_messages = [
            {"role": "system", "content": _CONSOLIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages, uid=uid)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages, uid=uid)

            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages, uid=uid)

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning("Memory consolidation: save_memory payload contains null required fields")
                return self._fail_or_raw_archive(messages, uid=uid)

            entry = _ensure_text(entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages, uid=uid)

            # Write to user_memory (only if knowledge_search is available)
            if self._knowledge_search:
                from datetime import datetime
                self._knowledge_search.write_user_memory_sync([{
                    "text": entry,
                    "type": "consolidation_summary",
                    "confidence": CONSOLIDATION_SUMMARY_CONFIDENCE,
                    "is_evergreen": False,
                    "created_at": datetime.now().isoformat(),
                }], uid=uid)

            update = _ensure_text(update)
            if update != current_memory:
                self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages, uid=uid)

    def _fail_or_raw_archive(self, messages: list[dict], uid: str | None = None) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages, uid=uid)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict], uid: str | None = None) -> None:
        """Fallback: dump raw messages to user_memory without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Format messages as searchable text
        formatted = self._format_messages(messages)
        text = f"[{ts}] [RAW] {len(messages)} messages\n{formatted}"

        # Write to user_memory (sync version)
        if self._knowledge_search:
            self._knowledge_search.write_user_memory_sync([{
                "text": text,
                "type": "raw_archive",
                "confidence": CONSOLIDATION_SUMMARY_CONFIDENCE,
                "is_evergreen": False,
                "created_at": datetime.now().isoformat(),
            }], uid=uid)

        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        knowledge_search: Any = None,
    ):
        self._workspace = workspace
        self._knowledge_search = knowledge_search
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

        # Conversation knowledge extractor (lazy init)
        self._knowledge_extractor = None

        # Anti-shake: track last token count to avoid repeated small consolidations
        self._last_session_tokens: dict[str, int] = {}

    def _get_store(self, agent_id: str | None = None) -> MemoryStore:
        return MemoryStore(self._workspace, knowledge_search=self._knowledge_search, agent_id=agent_id)

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]], agent_id: str | None = None, uid: str | None = None) -> bool:
        """Archive a selected message chunk into persistent memory."""
        success = await self._get_store(agent_id).consolidate(messages, self.provider, self.model, uid=uid)

        # After successful consolidation, extract knowledge from conversation
        if success and self._knowledge_search:
            await self._extract_conversation_knowledge(messages, uid=uid)

            # Run structural lint (report only, no auto-fix)
            try:
                from nanoresearch.research.knowledge_lint import KnowledgeLint
                lint = KnowledgeLint(self._knowledge_search, provider=None, model=None)
                structural_issues = await lint.lint_structural(fix=False)
                if structural_issues:
                    logger.info(
                        f"MemoryConsolidator: lint found {len(structural_issues)} structural issues (report only)"
                    )
            except Exception as e:
                logger.warning(f"Structural lint failed: {e}")

        return success

    async def _extract_conversation_knowledge(self, messages: list[dict[str, object]], uid: str | None = None) -> None:
        """Extract knowledge claims from conversation messages.

        This is called after consolidation to extract claims from agent statements.
        """
        try:
            # Lazy init the extractor
            if self._knowledge_extractor is None:
                from nanoresearch.agent.conversation_knowledge_extractor import ConversationKnowledgeExtractor
                self._knowledge_extractor = ConversationKnowledgeExtractor(
                    provider=self.provider,
                    model=self.model,
                    knowledge_search=self._knowledge_search,
                )

            await self._knowledge_extractor.extract_from_messages(messages, uid=uid)
        except Exception as e:
            logger.warning(f"Conversation knowledge extraction failed: {e}")

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
        tail_protect: int = CONSOLIDATION_TAIL_PROTECT,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens.

        Head/Tail Protection:
        - Head: 系统提示 + 首轮交互（通过 last_consolidated 起点保护）
        - Tail: 最近 N 条消息（tail_protect）不会被压缩

        Args:
            session: The session to analyze.
            tokens_to_remove: Minimum tokens to remove.
            tail_protect: Number of recent messages to protect (default CONSOLIDATION_TAIL_PROTECT).
        """
        start = session.last_consolidated
        # Tail protection: don't consolidate beyond this index
        max_end = len(session.messages) - tail_protect
        if start >= max_end or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, max_end):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        # Return last valid boundary within protected range
        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]], agent_id: str | None = None, uid: str | None = None) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(MemoryStore._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(messages, agent_id=agent_id, uid=uid):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session, agent_id: str | None = None, uid: str | None = None) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.

        Anti-shake: Skip consolidation if savings < 10% of last check.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = int(budget * TOKEN_CONSOLIDATION_TARGET_RATIO)
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < budget:
                # Anti-shake: check if savings too small
                last_tokens = self._last_session_tokens.get(session.key, estimated)
                self._last_session_tokens[session.key] = estimated
                if last_tokens > 0:
                    savings_ratio = (last_tokens - estimated) / last_tokens
                    if savings_ratio < 0.1:  # Savings < 10%
                        logger.debug(
                            "Token consolidation skipped (anti-shake): savings={:.1%}",
                            savings_ratio,
                        )
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target), tail_protect=CONSOLIDATION_TAIL_PROTECT)
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                old_last_consolidated = session.last_consolidated
                end_idx = boundary[0]
                chunk = session.messages[old_last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk, agent_id=agent_id, uid=uid):
                    return

                # Lua LTRIM: atomically advance Redis list start to new boundary (fast-path).
                # Failure here is non-fatal — sessions.save() will rewrite the correct window.
                if uid is not None and ":" in session.key:
                    try:
                        from nanoresearch.bus.redis_client import get_redis
                        from nanoresearch.bus.redis_keys import RedisKeys
                        ch, chat_id = session.key.split(":", 1)
                        _redis = get_redis()
                        keep_from_idx = end_idx - old_last_consolidated
                        await _redis.eval(
                            _LUA_LTRIM, 2,
                            RedisKeys.session_msg(uid, ch, chat_id),
                            RedisKeys.session_meta(uid, ch, chat_id),
                            str(keep_from_idx),
                            datetime.now(timezone.utc).isoformat(),
                        )
                    except Exception as _lua_err:
                        logger.warning("Lua LTRIM failed (non-fatal): {}", _lua_err)

                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return


def plan_startup_consolidation(
    session,
    *,
    now_utc,
    idle_threshold,
    min_turns,
    tail_protect,
    pick_boundary,
):
    """Decide the startup-consolidation chunk for a session, or None.

    Returns (start, end_idx) where start == session.last_consolidated, or None
    when the session is too active (idle gate), has too few turns, or no safe
    tail-protected boundary exists.
    """
    if now_utc - as_aware_utc(session.updated_at) < idle_threshold:
        return None

    start = session.last_consolidated
    pending = session.messages[start:]
    pending_turns = sum(1 for m in pending if m.get("role") == "user")
    if pending_turns < min_turns:
        return None

    boundary = pick_boundary(session, tokens_to_remove=1, tail_protect=tail_protect)
    if boundary is None:
        return None
    end_idx, _ = boundary
    if end_idx <= start:
        return None
    return (start, end_idx)
