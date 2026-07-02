"""Rebuild derived memory layers (events / conv-summaries / 画像) from the PG conversation log.

P4 of the memory-layering redesign. Per-uid **serial** over conversations in **chronological
order**, sharing ONE fact store so contradictory profiles **converge** (plan C1) — later
conversations' diffs land on the earlier-built 画像. The bulk run invokes the real consolidation
LLM and is a documented MANUAL step (P4.3); the chunking + orchestration below are unit-tested.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from loguru import logger

# consolidate_fn(messages, uid, conversation_id, turn_start, turn_end) -> awaitable
ConsolidateFn = Callable[[list[dict], str, str, int, int], Awaitable[Any]]


def plan_rebuild_chunks(messages: list[dict]) -> list[tuple[int, int]]:
    """Split a conversation into consolidation chunks at user-turn boundaries, covering ALL
    messages (contiguous, non-overlapping, union == whole conversation)."""
    if not messages:
        return []
    starts = sorted({0, *[i for i, m in enumerate(messages) if m.get("role") == "user"]})
    chunks: list[tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(messages)
        if e > s:
            chunks.append((s, e))
    return chunks


async def rebuild_uid(uid: str, conversations: list[tuple[str, Any, list[dict]]],
                      consolidate_fn: ConsolidateFn, limit: int | None = None,
                      dry_run: bool = False) -> int:
    """Rebuild one user's derived memory.

    `conversations` = [(conversation_id, created_at, messages)]. Processed in chronological order
    (by created_at) against the SHARED fact store so later conversations override earlier
    contradictory facts (convergence, C1). Returns the number of chunks (would-be) consolidated.

    `limit`: process at most this many conversations (试水: limit=1 → 仅最早一个对话).
    `dry_run`: don't call consolidate_fn — just log the plan (conversations + chunk counts).
    """
    ordered = sorted(conversations, key=lambda c: c[1])
    if limit is not None:
        ordered = ordered[:limit]
    n = 0
    for conv_id, _created, messages in ordered:
        chunks = plan_rebuild_chunks(messages)
        if dry_run:
            logger.info("[dry-run] uid={} conv={} chunks={} msgs={}",
                        uid, conv_id, len(chunks), len(messages))
            n += len(chunks)
            continue
        for start, end in chunks:
            await consolidate_fn(messages[start:end], uid, conv_id, start, end)
            n += 1
    return n


async def rebuild_from_pg(uid: str, repo: Any, consolidate_fn: ConsolidateFn,
                          limit: int | None = None, dry_run: bool = False,
                          conversation_id: str | None = None) -> int:
    """Load a uid's conversations from PG (via ConversationRepository) and rebuild them serially.

    `conversation_id`: if given, only that one conversation (试水: 指定某个对话).
    `limit` / `dry_run`: forwarded to rebuild_uid.
    """
    convs = await repo.list_conversations(uid, limit=100_000)
    conversations: list[tuple[str, Any, list[dict]]] = []
    for c in convs:
        if conversation_id and str(c.id) != conversation_id:
            continue
        msgs = await repo.get_messages(c.id)
        conversations.append((str(c.id), c.created_at, [m.content for m in msgs]))
    logger.info("rebuild_from_pg: uid={} conversations={} (limit={}, dry_run={})",
                uid, len(conversations), limit, dry_run)
    return await rebuild_uid(uid, conversations, consolidate_fn, limit=limit, dry_run=dry_run)


def main() -> None:  # pragma: no cover — manual bulk run (needs live LLM + DB + config)
    """Manual bulk run. Wire a real consolidation fn from your runtime config, then rebuild.

    The per-chunk consolidation uses MemoryStore.consolidate directly (no token-budgeting needed
    for a rebuild). Because this needs a live LLM + populated PG, it is NOT run in CI; see the
    P4.3 note in docs/superpowers/plans/2026-07-01-memory-events-summary-migration-P2P3P4.md.
    """
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Rebuild memory layers from the PG log.")
    parser.add_argument("--uid", action="append", dest="uids", help="uid(s) to rebuild")
    parser.add_argument("--limit", type=int, default=None,
                        help="max conversations per uid (试水: --limit 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划,不写入")
    parser.add_argument("--conversation-id", dest="conversation_id", default=None,
                        help="只重建指定 conversation")
    parser.add_argument("--config", default=None, help="config path")
    args = parser.parse_args()

    async def _run() -> None:
        from pathlib import Path

        from nanoresearch.agent.memory import MemoryStore
        from nanoresearch.cli.commands import build_loop_config
        from nanoresearch.storage.repositories.conversation_repo import ConversationRepository
        from nanoresearch.storage.repositories.user_repo import UserRepository

        cfg = await build_loop_config(config=args.config)
        factory = cfg["session_factory"]
        provider = cfg["provider"]
        model = cfg.get("model")
        knowledge_search = cfg.get("knowledge_search")
        base = Path(cfg["base_workspace"])
        repo = ConversationRepository(factory)

        uids = args.uids or [u.uid for u in await UserRepository(factory).list_all()]
        for uid in uids:
            store = MemoryStore(base / "users" / uid, knowledge_search=knowledge_search,
                                session_factory=factory)

            async def consolidate_fn(msgs, _uid, cid, s, e):
                return await store.consolidate(msgs, provider, model, uid=_uid,
                                               conversation_id=cid, turn_start=s, turn_end=e)

            count = await rebuild_from_pg(uid, repo, consolidate_fn, limit=args.limit,
                                          dry_run=args.dry_run, conversation_id=args.conversation_id)
            logger.info("rebuilt uid={}: {} chunks (dry_run={})", uid, count, args.dry_run)

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
