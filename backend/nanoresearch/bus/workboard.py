"""Workboard claim primitives (Phase 2 serial-MVP).

Serial invariant: global WIP=1 — at most one running card per conversation. `claim_card`
enforces it before acquiring the per-card lock (the claim token IS the dist_lock token) and
flipping the card ready→running under that token. heartbeat/release reuse the same dist_lock
primitives, so a stray non-owner can never extend or release someone else's claim.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from nanoresearch.bus import dist_lock
from nanoresearch.bus.redis_keys import RedisKeys


async def claim_card(
    redis: Any, repo: Any, *, card_id: Any, agent_id: Any, conv_id: Any, px_ms: int = 30_000
) -> str | None:
    """Claim a ready card for *agent_id*. Returns the claim token on success, else None.

    Order: ① global WIP=1 (no card already running in this conversation) ② acquire the per-card
    lock (claim token) ③ status CAS ready→running under that token. Any failure releases and
    returns None."""
    if await repo.list_by_conversation(conv_id, statuses={"running"}):
        return None  # serial: a card is already running

    lock_key = RedisKeys.workboard_claim(str(card_id))
    token = await dist_lock.acquire(redis, lock_key, px_ms=px_ms)
    if token is None:
        return None

    now = datetime.now(timezone.utc)
    moved = await repo.transition(
        card_id, expect_status="ready", to_status="running",
        owner_agent_id=agent_id, claim_token=token, claimed_at=now, heartbeat_at=now,
    )
    if not moved:
        await dist_lock.release(redis, lock_key, token)
        return None
    return token


async def heartbeat_card(redis: Any, repo: Any, *, card_id: Any, token: str, px_ms: int = 30_000) -> bool:
    """Extend the claim lease iff *token* still owns it; refresh heartbeat_at on success."""
    ok = await dist_lock.refresh(redis, RedisKeys.workboard_claim(str(card_id)), token, px_ms=px_ms)
    if ok:
        await repo.touch_heartbeat(card_id)
    return ok


async def release_card(redis: Any, repo: Any, *, card_id: Any, token: str, to_status: str = "ready") -> bool:
    """Token-gated release: drop the lock (token-checked) then move the card running→to_status,
    clearing owner/claim_token. Returns False if *token* no longer owns the lock."""
    released = await dist_lock.release(redis, RedisKeys.workboard_claim(str(card_id)), token)
    if not released:
        return False
    await repo.transition(
        card_id, expect_status="running", to_status=to_status,
        owner_agent_id=None, claim_token=None,
    )
    return True
