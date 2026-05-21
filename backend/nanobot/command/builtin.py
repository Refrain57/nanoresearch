"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.utils.helpers import build_status_content


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    # Prefer actual usage from provider over tiktoken estimate (which is inaccurate for non-OpenAI models)
    ctx_est = loop._last_usage.get("prompt_tokens", 0)
    if ctx_est <= 0:
        try:
            ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
        except Exception:
            pass
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
        ),
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    lines = [
        "🐈 nanobot commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/research — Start a research task",
        "/help — Show available commands",
    ]
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata={"render_as": "text"},
    )


async def cmd_research(ctx: CommandContext) -> OutboundMessage:
    """Start or manage research tasks."""
    loop = ctx.loop
    msg = ctx.msg
    args = ctx.args.strip()

    if not args:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=(
                "用法: /research <研究方向>\n"
                "示例: /research LLM在医疗领域的应用\n"
                "\n"
                "支持深度参数: /research --depth=quick|normal|deep <研究方向>"
            ),
        )

    # Parse depth parameter
    depth = "normal"
    topic = args
    if args.startswith("--depth="):
        parts = args.split(None, 1)
        if len(parts) == 2:
            depth = parts[0].split("=")[1]
            topic = parts[1]
        else:
            topic = args

    # Get research_tool
    research_tool = loop.tools.get("research")
    if not research_tool:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="研究模块未启用。请在配置中启用 tools.research.enabled = true",
        )

    # Execute research
    try:
        result = await research_tool.execute(action="start", topic=topic, depth=depth)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=result,
        )
    except Exception as e:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"研究失败: {e}",
        )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/help", cmd_help)
    router.exact("/research", cmd_research)  # exact match for usage display
    router.prefix("/research ", cmd_research)  # prefix match to accept arguments

