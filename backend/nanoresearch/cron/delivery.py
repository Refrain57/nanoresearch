"""Deliver a cron run's result back to the origin web conversation.

Cron runs execute in a dedicated `[cron]` conversation (lock isolation), so their output would
otherwise be invisible to the user — who set the task up in a *different* chat. For the `web`
channel no cross-process outbound bridge is needed (the spec's Task 6 bridge is only for external
IM channels): the result is persisted straight into the origin conversation (`deliver_to`) AND
pushed onto that conversation's live SSE stream, so an already-open frontend renders it
immediately, with no polling.
"""
from __future__ import annotations

from typing import Any

from loguru import logger


async def deliver_cron_result_web(
    redis: Any,
    sessions: Any,
    *,
    uid: str,
    cron: dict | None,
    response_text: str | None,
) -> bool:
    """Persist + live-push a cron result into its origin web conversation.

    Returns True iff a delivery was made. No-op (False) unless the job opted into delivery on the
    web channel and there is a non-empty response. Both writes are best-effort: the persist makes
    the result show on reload; the live push makes it appear immediately in an open frontend.
    """
    if not cron or not cron.get("deliver"):
        return False
    if cron.get("channel") != "web":
        # External channels (feishu/whatsapp/…) need the cross-process outbound bridge (deferred
        # Task 6), not this web-only path.
        return False
    to = cron.get("to")
    text = (response_text or "").strip()
    if not to or not text:
        return False

    from nanoresearch.bus.redis_keys import RedisKeys
    from nanoresearch.bus.stream import xadd_event

    session_key = f"web:{to}"
    # Persist into the origin conversation (Redis session + DB) so it also shows on reload.
    # append_message is the same cross-run append used for subagent results; a concurrent
    # full-overwrite save in this exact conversation is a narrow, benign race (§ manager.py).
    try:
        await sessions.append_message(
            session_key,
            {"role": "assistant", "content": {"text": text}, "cron": True},
            uid=uid,
        )
    except Exception as e:
        logger.warning("cron delivery: append to {} failed (non-fatal): {}", session_key, e)

    # Live push so an already-open frontend renders it now (no polling).
    try:
        await xadd_event(redis, RedisKeys.conv_live(to), {
            "type": "cron_message",
            "role": "assistant",
            "content": {"text": text},
            "task": cron.get("task_context"),
        })
    except Exception as e:
        logger.warning("cron delivery: live push to conv_live:{} failed (non-fatal): {}", to, e)

    return True
