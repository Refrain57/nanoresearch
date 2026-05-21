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
        "启动自主网络研究任务（耗时 10-30min）。"
        "适用于需要大量新信息支撑的深度研究场景。"
        "调用前请先判断：如果知识库（research_claims/research_insights）已有相关结论，"
        "直接使用 RAG 检索即可，无需启动完整研究流程。"
        "重要：这是耗时任务，必须通过 spawn 工具后台执行。"
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
                "default": "normal",
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
        knowledge_search: Any = None,
        rag_store: Any = None,
    ) -> None:
        # Create knowledge processor if knowledge_search is available
        knowledge_processor = None
        if knowledge_search:
            from nanobot.research.knowledge_processor import KnowledgeProcessor
            from nanobot.research.insight_tracker import InsightTracker
            tracker = InsightTracker()
            knowledge_processor = KnowledgeProcessor(
                provider=provider,
                model=model,
                knowledge_search=knowledge_search,
                tracker=tracker,
                rag_store=rag_store,
            )

        self._runner = ResearchRunner(
            provider=provider,
            model=model,
            web_search_tool=web_search_tool,
            web_fetch_tool=web_fetch_tool,
            config=config,
            knowledge_search=knowledge_search,
            knowledge_processor=knowledge_processor,
            rag_store=rag_store,
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
            ]

            # Include knowledge write result if available
            if result.knowledge_result:
                kr = result.knowledge_result
                summary.append(f"**知识写入**: {kr.claims_written} claims, {kr.insights_written} insights")
                if kr.duplicates_skipped > 0:
                    summary.append(f"**去重跳过**: {kr.duplicates_skipped} 条")
                if kr.conflicts_detected > 0:
                    summary.append(f"**冲突检测**: {kr.conflicts_detected} 条")
            else:
                summary.append(f"**知识写入**: 未配置知识处理器")

            summary.append("")
            summary.append("---")
            summary.append("")

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
