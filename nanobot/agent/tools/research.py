"""Research tool — provides autonomous research capability to the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.research.runner import ResearchRunner
from nanobot.research.types import ResearchConfig, ResearchResult

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class ResearchTool(Tool):
    """Tool for launching and managing autonomous research tasks."""

    name = "research"
    description = (
        "启动自主网络研究任务。当用户需要深入了解某个话题、对比多个观点、生成研究报告时使用。"
        "此工具会主动搜索网络、抓取网页、分析信息、生成完整报告。"
        "注意：这是网络研究工具，不需要先查询本地知识库。"
        "支持 action: start/status/list。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "status", "list"],
                "description": "start=启动研究, status=查看进度, list=列出历史",
            },
            "topic": {
                "type": "string",
                "description": "研究方向（仅 start 时需要）",
            },
            "depth": {
                "type": "string",
                "enum": ["quick", "normal", "deep"],
                "description": "研究深度（默认 normal）",
            },
            "research_id": {
                "type": "string",
                "description": "研究ID（仅 status 时需要）",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        web_search_tool: Any,
        web_fetch_tool: Any,
        config: ResearchConfig | None = None,
    ) -> None:
        self._runner = ResearchRunner(
            provider=provider,
            model=model,
            web_search_tool=web_search_tool,
            web_fetch_tool=web_fetch_tool,
            config=config,
        )
        self._results: dict[str, ResearchResult] = {}

    async def execute(
        self,
        action: str,
        topic: str | None = None,
        depth: str = "normal",
        research_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if action == "start":
            if not topic:
                return "Error: topic is required for start action"
            logger.info("ResearchTool: starting research on '{}'", topic)
            result = await self._runner.run(topic, depth=depth)
            self._results[result.id] = result

            # Build summary
            quality_str = f"{result.quality_score:.1f}/10" if result.quality_score else "N/A"
            summary = [
                f"研究完成！",
                f"",
                f"**主题**: {result.topic}",
                f"**研究ID**: {result.id}",
                f"**迭代轮次**: {result.iterations}",
                f"**信息来源**: {result.total_sources} 篇",
                f"**质量自评**: {quality_str}",
                f"",
                f"---",
                f"",
            ]
            if result.report:
                summary.append(result.report)
            else:
                summary.append("报告生成失败，请检查日志。")
            return "\n".join(summary)

        elif action == "status":
            if research_id:
                result = self._runner.get_result(research_id) or self._results.get(research_id)
            else:
                active = [rid for rid, r in self._runner._results.items()]
                return f"状态: 活跃研究中 {len(active)} 个\nID列表: {active}" if active else "暂无活跃研究"

            if not result:
                return f"Error: research '{research_id}' not found"

            quality_str = f"{result.quality_score:.1f}/10" if result.quality_score else "N/A"
            return (
                f"**研究ID**: {result.id}\n"
                f"**主题**: {result.topic}\n"
                f"**状态**: {result.status.value}\n"
                f"**迭代**: {result.iterations}\n"
                f"**来源**: {result.total_sources} 篇\n"
                f"**质量**: {quality_str}\n"
                f"**创建时间**: {result.created_at.strftime('%Y-%m-%d %H:%M')}"
            )

        elif action == "list":
            all_results = list(self._runner._results.values()) + list(self._results.values())
            if not all_results:
                return "暂无研究记录"
            lines = []
            for r in sorted(all_results, key=lambda x: x.created_at, reverse=True):
                quality_str = f"{r.quality_score:.1f}/10" if r.quality_score else "N/A"
                lines.append(
                    f"- **{r.id}** [{r.status.value}] {r.topic} (质量: {quality_str})"
                )
            return "研究历史:\n" + "\n".join(lines)

        else:
            return f"Unknown action: {action}"
