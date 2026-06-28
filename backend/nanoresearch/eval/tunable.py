"""Text-class TunableObject interface and Phase 1 implementations.

Design constraints (from SDD §4.1):
- Interface covers ONLY text-class objects.  Numeric/rule-class objects (top-k,
  budget, thresholds) must NOT be added here — that is interface dilution.
- Interface is frozen until Phase 6 completes.  New requirements go in callers
  as patches; no new methods are added here until the next interface review.
- Two instances in Phase 1: PersonaObject and ToolDescriptionObject.
  Three data points required before a new abstraction is warranted.

PersonaObject note (SDD §6 Phase 1 #4 rationale):
  System prompt is assembled dynamically by ContextBuilder from multiple
  segments (SOUL.md structure, skills summary, KB bindings, dynamic suffix).
  The ONLY tunable segment stored in the database is agents.persona.
  PersonaObject.read/apply therefore operate exclusively on that field.
  This is an intentional constraint, NOT an incomplete implementation.
  Optimizing the other segments requires code changes — not this object.

ToolDescriptionObject note:
  generate_candidates produces candidate description text (LLM call only).
  Scoring candidates by running the agent requires Phase 4 sandbox layering.
  Until Phase 4, OptimizationAgent._score_candidate raises NotImplementedError
  for tool_description targets.  The interface is complete; the scoring path is not.
  PHASE_STATUS.md documents this explicitly.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanoresearch.providers.base import LLMProvider
    from nanoresearch.storage.models import AgentRunSnapshot
    from nanoresearch.storage.repositories.agent_eval_repo import AgentEvalRepository
    from nanoresearch.storage.repositories.agent_repo import AgentRepository


# ---------------------------------------------------------------------------
# Shared value type (imported by optimizer.py to avoid circular deps)
# ---------------------------------------------------------------------------

@dataclass
class OptimizationCandidate:
    prompt: str      # candidate text for this tunable object
    rationale: str   # why this candidate is expected to improve things
    # Phase 2: dual-set scores.  Structure:
    #   {"fix_set": {"keyword_coverage": 0.8, ...}, "health_set": {"keyword_coverage": 0.9, ...}}
    # Populated by OptimizationAgent after scoring on both sets.
    # Empty dict means scoring was skipped (e.g. tool_description before Phase 4).
    scores: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = {}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TunableTextObject(ABC):
    """Uniform interface for text-class tunable objects.

    Implementations must be two-point-connected (2 concrete classes exist before
    any abstraction change is considered valid).  Do not add methods here without
    a Phase 6 interface review.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Registry kind key: 'system_prompt' | 'tool_description'."""

    @property
    @abstractmethod
    def target_id(self) -> str:
        """Identifies the specific object within its kind (agent_id | tool_name)."""

    @abstractmethod
    async def read(self) -> str:
        """Return the current content of this object."""

    @abstractmethod
    async def generate_candidates(
        self, badcases: list["AgentRunSnapshot"]
    ) -> list[OptimizationCandidate]:
        """Generate 3-5 candidate improvements given representative badcases."""

    @abstractmethod
    async def apply(self, content: str) -> str:
        """Write content to version registry (active=True) and persist to storage.

        Returns the new version_id (UUID string).
        Previous active version for this kind+target_id is deactivated.
        """

    @abstractmethod
    async def get_current_version(self) -> str | None:
        """Return the currently active version_id, or None if no version exists."""

    @abstractmethod
    async def rollback(self, version_id: str) -> None:
        """Re-apply a historical version's content via apply().

        Historical rows are immutable — rollback writes a NEW row, not a mutation.
        Raises ValueError if version_id not found.
        """


# ---------------------------------------------------------------------------
# Shared LLM generation helper
# ---------------------------------------------------------------------------

_MAX_EXAMPLE_CHAIN_CHARS = 500
_MAX_EXAMPLE_RESP_CHARS = 300
_MAX_EXAMPLES = 5


def _build_badcase_examples(snapshots: list["AgentRunSnapshot"]) -> list[str]:
    examples: list[str] = []
    for snap in snapshots[:_MAX_EXAMPLES]:
        chain = snap.tool_call_chain or []
        recent = chain[-2:] if len(chain) > 2 else chain
        chain_str = json.dumps(recent, ensure_ascii=False, default=str)[:_MAX_EXAMPLE_CHAIN_CHARS]
        resp = (snap.final_response or "(无回复)")[:_MAX_EXAMPLE_RESP_CHARS]
        examples.append(
            f"- 用户输入: {(snap.user_input or '')[:200]}\n"
            f"  工具调用(最近): {chain_str}\n"
            f"  最终回复: {resp}"
        )
    return examples


async def _llm_generate_candidates(
    provider: "LLMProvider",
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> list[OptimizationCandidate]:
    try:
        response = await provider.chat_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            max_tokens=2048,
            temperature=0.3,
        )
        raw = (response.content or "").strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        items: list[dict] = json.loads(match.group())
        return [
            OptimizationCandidate(
                prompt=str(item.get("prompt", "")),
                rationale=str(item.get("rationale", "")),
            )
            for item in items
            if item.get("prompt")
        ]
    except Exception as exc:
        logger.warning("TunableTextObject._llm_generate_candidates failed: {}", exc)
        return []


# ---------------------------------------------------------------------------
# PersonaObject
# ---------------------------------------------------------------------------

_PERSONA_GENERATE_SYSTEM = """\
你是一位 AI Agent persona（人格/行为设定）优化专家。
给定当前的 persona 文本和代表性 badcase，生成 3-5 条候选 persona 改进方案。

重要约束：persona 是 system prompt 中通过数据库可调整的部分。
其余部分（技能摘要、工具描述、知识库绑定、动态后缀）由系统代码固定拼接，
不在优化范围内——不要在候选中引用或修改这些部分的内容。

以 JSON 数组输出，格式：[{"prompt": "改进后的完整 persona 文本", "rationale": "一句话理由"}]
不要输出任何其他内容。
"""


class PersonaObject(TunableTextObject):
    """Tunable object for agents.persona — the only DB-stored tunable text segment.

    SCOPE BOUNDARY: read/apply operate ONLY on agents.persona.
    The rest of the system prompt (structure, skills, KB bindings, dynamic suffix)
    is assembled by ContextBuilder from code and is NOT tunable via this object.
    """

    kind = "system_prompt"

    def __init__(
        self,
        agent_id: str,
        agent_repo: "AgentRepository",
        eval_repo: "AgentEvalRepository",
        provider: "LLMProvider",
        model: str | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._agent_repo = agent_repo
        self._eval_repo = eval_repo
        self._provider = provider
        self._model = model or os.environ.get("EVAL_JUDGE_MODEL") or provider.get_default_model()

    @property
    def target_id(self) -> str:
        return self._agent_id

    async def read(self) -> str:
        agent = await self._agent_repo.get_by_id(uuid.UUID(self._agent_id))
        if agent is None:
            return ""
        return agent.persona or ""

    async def generate_candidates(
        self, badcases: list["AgentRunSnapshot"]
    ) -> list[OptimizationCandidate]:
        current = await self.read()
        examples = _build_badcase_examples(badcases)
        user_prompt = (
            f"当前 persona 文本：\n{current or '（未设置）'}\n\n"
            f"代表性 badcase 样本（共 {len(examples)} 条）：\n"
            + "\n\n".join(examples)
            + "\n\n请生成 3-5 条候选 persona 改进方案（JSON 数组）。"
        )
        return await _llm_generate_candidates(
            self._provider, self._model, _PERSONA_GENERATE_SYSTEM, user_prompt
        )

    async def apply(self, content: str) -> str:
        version_id = await self._eval_repo.create_tunable_version(
            kind=self.kind,
            target_id=self.target_id,
            content=content,
            created_by="system",
        )
        await self._agent_repo.update(uuid.UUID(self._agent_id), persona=content)
        return str(version_id)

    async def get_current_version(self) -> str | None:
        v = await self._eval_repo.get_current_tunable_version(self.kind, self.target_id)
        return str(v.id) if v is not None else None

    async def rollback(self, version_id: str) -> None:
        v = await self._eval_repo.get_tunable_version_by_id(uuid.UUID(version_id))
        if v is None:
            raise ValueError(f"tunable version {version_id} not found")
        await self.apply(v.content)


# ---------------------------------------------------------------------------
# ToolDescriptionObject
# ---------------------------------------------------------------------------

_TOOL_DESC_GENERATE_SYSTEM = """\
你是一位 AI 工具描述（tool description）优化专家。
给定某工具的当前描述和代表性 badcase（该工具被错误调用、未被调用、或调用参数有误），
生成 3-5 条候选工具描述改进方案。

工具描述的优化目标：让模型更准确地理解何时调用该工具、传入什么参数。

以 JSON 数组输出，格式：[{"prompt": "改进后的完整工具描述文本", "rationale": "一句话理由"}]
不要输出任何其他内容。
"""


class ToolDescriptionObject(TunableTextObject):
    """Tunable object for a tool's description field in agents.tools_config.

    SCORING NOTE (Phase 1 constraint):
    generate_candidates produces candidate text (LLM only, no agent run).
    Evaluating candidates by running the agent through a sandboxed replay requires
    Phase 4 sandbox layering.  Until Phase 4, OptimizationAgent._score_candidate
    raises NotImplementedError for tool_description targets — this is explicit
    and intentional, not a silent failure.
    See PHASE_STATUS.md: "ToolDescriptionObject scoring blocked until Phase 4".
    """

    kind = "tool_description"

    def __init__(
        self,
        tool_name: str,
        agent_id: str,
        agent_repo: "AgentRepository",
        eval_repo: "AgentEvalRepository",
        provider: "LLMProvider",
        model: str | None = None,
    ) -> None:
        self._tool_name = tool_name
        self._agent_id = agent_id
        self._agent_repo = agent_repo
        self._eval_repo = eval_repo
        self._provider = provider
        self._model = model or os.environ.get("EVAL_JUDGE_MODEL") or provider.get_default_model()

    @property
    def target_id(self) -> str:
        return self._tool_name

    async def read(self) -> str:
        agent = await self._agent_repo.get_by_id(uuid.UUID(self._agent_id))
        if agent is None:
            return ""
        for tool in (agent.tools_config or []):
            if tool.get("name") == self._tool_name:
                return tool.get("description") or ""
        return ""

    async def generate_candidates(
        self, badcases: list["AgentRunSnapshot"]
    ) -> list[OptimizationCandidate]:
        current = await self.read()
        examples = _build_badcase_examples(badcases)
        user_prompt = (
            f"工具名称：{self._tool_name}\n"
            f"当前描述：\n{current or '（未设置）'}\n\n"
            f"代表性 badcase 样本（共 {len(examples)} 条）：\n"
            + "\n\n".join(examples)
            + "\n\n请生成 3-5 条候选工具描述改进方案（JSON 数组）。"
        )
        return await _llm_generate_candidates(
            self._provider, self._model, _TOOL_DESC_GENERATE_SYSTEM, user_prompt
        )

    async def apply(self, content: str) -> str:
        version_id = await self._eval_repo.create_tunable_version(
            kind=self.kind,
            target_id=self.target_id,
            content=content,
            created_by="system",
        )
        agent = await self._agent_repo.get_by_id(uuid.UUID(self._agent_id))
        if agent is None:
            raise ValueError(f"agent {self._agent_id} not found")
        tools = list(agent.tools_config or [])
        updated = False
        for tool in tools:
            if tool.get("name") == self._tool_name:
                tool["description"] = content
                updated = True
                break
        if not updated:
            logger.warning(
                "ToolDescriptionObject.apply: tool '{}' not found in agent {} tools_config",
                self._tool_name, self._agent_id,
            )
        await self._agent_repo.update(uuid.UUID(self._agent_id), tools_config=tools)
        return str(version_id)

    async def get_current_version(self) -> str | None:
        v = await self._eval_repo.get_current_tunable_version(self.kind, self.target_id)
        return str(v.id) if v is not None else None

    async def rollback(self, version_id: str) -> None:
        v = await self._eval_repo.get_tunable_version_by_id(uuid.UUID(version_id))
        if v is None:
            raise ValueError(f"tunable version {version_id} not found")
        await self.apply(v.content)
