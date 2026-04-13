"""Spawn tool for creating background subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "在后台启动一个子 Agent 来执行耗时任务，前台可以继续对话，任务完成后会自动通知用户。"
            "使用场景："
            "1. 深度研究（research 工具）：当用户需要全面研究某个主题时"
            "2. 复杂文件分析：当需要处理大量数据或代码时"
            "3. 耗时任务：明确告诉用户'在后台运行'、'研究完通知我'、'后台执行'时"
            "重要提示：这是一个用于**后台执行**的工具，当你被要求后台运行时必须使用。"
            "不要在前台执行耗时任务，而应该使用此工具将任务放到后台。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "后台任务的具体指令。例如：'使用 research 工具深度研究车载雷达'",
                },
                "label": {
                    "type": "string",
                    "description": "任务标签，用于显示（可选）",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )
