"""DecomposeToBoardTool — primary main decomposes a multi-domain task into workboard cards.

Phase 2 Task 6: the tool that kicks off real multi-main collaboration.  The LLM calls this
tool; everything else (offer → self-claim → card-working → collector) is already built.
"""
from __future__ import annotations

import uuid
from typing import Any

from nanoresearch.agent.tools.base import Tool


class DecomposeToBoardTool(Tool):
    """Decompose a complex multi-domain task into workboard cards for specialist mains."""

    def __init__(self, session_factory: Any, arq_pool: Any) -> None:
        self._session_factory = session_factory
        self._arq_pool = arq_pool
        # Injected per-run by set_context
        self._conversation_id: str | None = None
        self._uid: str | None = None
        self._primary_agent_id: str | None = None
        self._agents_registry: list[dict] = []

    def set_context(
        self,
        conversation_id: str | None,
        uid: str | None,
        primary_agent_id: str | None,
        agents_registry: list[dict] | None,
    ) -> None:
        """Called by the loop before each run to inject runtime routing info."""
        self._conversation_id = conversation_id
        self._uid = uid
        self._primary_agent_id = primary_agent_id
        self._agents_registry = agents_registry or []

    @property
    def name(self) -> str:
        return "decompose_to_board"

    @property
    def description(self) -> str:
        return (
            "将一个需要多个不同专长主 Agent 协作的复杂任务拆分成工作板卡片，分派给各专长主，"
            "启动多主协作流程。"
            "【重要限制】仅当任务必须由多个不同专长的主 Agent 共同完成时才调用此工具。"
            "单领域问题、简单问答、或可由当前 Agent 独立回答的请求，请直接回答，不要拆卡。"
            "工具调用后协作立即启动；所有卡片完成后主 Agent 汇总结果回复用户。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "description": (
                        "要创建的卡片列表。每张卡片分派给一个专长主 Agent，"
                        "且 spec 必须包含足以让目标 Agent 独立执行的完整指令。"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "卡片标题（简短概括，30 字以内）",
                            },
                            "spec": {
                                "type": "string",
                                "description": (
                                    "给目标 Agent 的完整执行指令。"
                                    "应足够独立，使 Agent 无需其他上下文即可完成任务。"
                                ),
                            },
                            "target_agent": {
                                "type": "string",
                                "description": (
                                    "目标 Agent 的名称或 id，必须来自 Agent Registry 中列出的条目。"
                                ),
                            },
                            "depends_on": {
                                "type": "array",
                                "description": (
                                    "此卡片所依赖的其他卡片在本 cards 数组中的索引（0 起）。"
                                    "列出的父卡片全部完成后，此卡片才会进入 ready 状态。"
                                    "无依赖时传空数组 []。"
                                ),
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["title", "spec", "target_agent", "depends_on"],
                    },
                }
            },
            "required": ["cards"],
        }

    def _resolve_agent(self, target: str) -> str | None:
        """Resolve target_agent string (name or id) → agent id string. None if not found."""
        target_stripped = (target or "").strip()
        target_lower = target_stripped.lower()
        for entry in self._agents_registry:
            if entry.get("id") == target_stripped:
                return entry["id"]
            if entry.get("name", "").lower() == target_lower:
                return entry["id"]
        return None

    async def execute(self, cards: list[dict], **kwargs: Any) -> str:  # type: ignore[override]
        """Create workboard cards and start the collaboration round."""
        # Guard: requires a web conversation context
        if not self._conversation_id:
            return (
                "Error: decompose_to_board 只能在 Web 对话中使用。"
                "CLI / 非对话场景不支持多主协作工作板。"
            )

        if not cards:
            return "Error: cards 列表不能为空，至少需要一张卡片。"

        # Resolve all target_agent values upfront — fail before any DB writes
        resolved_agent_ids: list[str] = []
        for i, card in enumerate(cards):
            agent_id_str = self._resolve_agent(card.get("target_agent", ""))
            if agent_id_str is None:
                available = ", ".join(
                    a.get("name", a.get("id", "?")) for a in self._agents_registry
                )
                return (
                    f"Error: 卡片 [{i}]（{card.get('title', '')!r}）的"
                    f" target_agent={card.get('target_agent')!r}"
                    f" 在 Agent Registry 中找不到。"
                    f" 可用 Agent：{available or '（无）'}。"
                    f" 请使用上面列出的名称或 id 重试，不要创建任何卡片。"
                )
            resolved_agent_ids.append(agent_id_str)

        # Lazy imports to avoid import cycles
        from nanoresearch.bus import workboard
        from nanoresearch.bus.redis_client import get_redis
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        from nanoresearch.storage.repositories.workboard_repo import WorkboardRepository

        conv_uuid = uuid.UUID(self._conversation_id)
        repo = WorkboardRepository(self._session_factory)

        # Cap check: can we still add cards to this round?
        if not await repo.can_create_successor(conv_uuid, parent_depth=0):
            return (
                "Error: 工作板已达到本轮卡片上限，无法继续拆分。"
                "请缩减任务范围或等待当前轮次完成。"
            )

        primary_uuid = (
            uuid.UUID(self._primary_agent_id) if self._primary_agent_id else None
        )

        # Create cards in declaration order; depth = 0 for roots, 1 for dependants
        created_ids: list[uuid.UUID] = []
        for i, card in enumerate(cards):
            depends = card.get("depends_on") or []
            status = "ready" if not depends else "todo"
            depth = 0 if not depends else 1
            target_uuid = uuid.UUID(resolved_agent_ids[i])
            created = await repo.create_card(
                conversation_id=conv_uuid,
                title=card["title"],
                spec=card.get("spec", ""),
                status=status,
                target_agent_id=target_uuid,
                created_by_agent_id=primary_uuid,
                depth=depth,
            )
            created_ids.append(created.id)

        # Wire dependency links
        for i, card in enumerate(cards):
            for parent_idx in (card.get("depends_on") or []):
                await repo.link(created_ids[parent_idx], created_ids[i])

        # Activate agents: unique set of target ids ∪ {primary}
        all_target_uuids = {uuid.UUID(r) for r in resolved_agent_ids}
        if primary_uuid is not None:
            all_target_uuids.add(primary_uuid)
        await ConversationRepository(self._session_factory).activate_agents(
            conv_uuid, list(all_target_uuids)
        )

        # Begin collaboration round (gate defers user messages while round is in flight)
        redis = get_redis()
        await workboard.begin_round(redis, self._conversation_id)

        # Offer the first ready card to its target's inbox
        from nanoresearch.worker import _offer_next_or_collect  # lazy — avoid import cycles

        await _offer_next_or_collect(
            redis, repo, self._arq_pool, self._conversation_id, self._uid
        )

        # Build a human-readable receipt
        name_by_id: dict[str, str] = {
            e["id"]: e.get("name", e["id"]) for e in self._agents_registry
        }
        unique_names = list(dict.fromkeys(name_by_id.get(r, r) for r in resolved_agent_ids))
        names_str = "、".join(unique_names)
        return (
            f"已把任务拆成 {len(cards)} 张卡片，分派给 {names_str}；"
            f"协作已开始，完成后我会综合答复。"
        )
