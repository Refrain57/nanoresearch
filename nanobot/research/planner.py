"""Research planner — decomposes a topic into sub-questions via LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.research.types import ResearchPlan, SubQuestion

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_RESEARCH_PLAN_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "research_plan",
            "description": "输出研究规划，将研究方向拆解为具体的子问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "子问题的具体描述",
                                },
                                "keywords_zh": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "中文搜索关键词组合",
                                },
                                "keywords_en": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "英文搜索关键词组合",
                                },
                                "priority": {
                                    "type": "integer",
                                    "description": "优先级 1-5，1 最高",
                                },
                            },
                            "required": ["question", "priority"],
                        },
                    },
                },
                "required": ["sub_questions"],
            },
        },
    }
]

_SYSTEM_PROMPT = """你是一个专业的研究规划专家。用户的任务是深入研究某个主题。

请将研究方向拆解为 3-6 个具体的子问题，每个子问题：
1. 有独立的研究价值
2. 可以通过搜索引擎找到答案
3. 避免重复或重叠

为每个子问题提供：
- question: 具体的问题描述（中文）
- keywords_zh: 2-4 个中文关键词
- keywords_en: 2-4 个英文关键词
- priority: 优先级 1-5（1 = 最重要）

研究深度说明：
- quick: 拆解 2-3 个核心问题，快速概览
- normal: 拆解 4-5 个问题，平衡深度和广度
- deep: 拆解 5-6 个问题，追求全面覆盖"""

_USER_TEMPLATE = """## 研究方向
{topic}

## 研究深度
{depth}

请调用 research_plan 工具返回规划结果。"""


class ResearchPlanner:
    """Decomposes a research topic into sub-questions using LLM."""

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def plan(self, topic: str, depth: str = "normal") -> ResearchPlan:
        """Generate a research plan from a topic.

        Args:
            topic: The research topic or question.
            depth: One of "quick", "normal", "deep".

        Returns:
            ResearchPlan with sub-questions and keywords.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(topic=topic, depth=depth)},
        ]

        logger.info("ResearchPlanner: planning topic={}", topic)
        llm_response = await self.provider.chat_with_retry(
            messages=messages,
            tools=_RESEARCH_PLAN_TOOL,
            model=self.model,
            max_tokens=2048,
            temperature=0.3,
        )

        if not llm_response.has_tool_calls:
            # Try to parse from content if LLM didn't use tool_call but returned structured text
            if llm_response.content:
                parsed = self._parse_plan_from_content(topic, llm_response.content)
                if parsed.sub_questions:
                    logger.info("ResearchPlanner: parsed {} sub-questions from content", len(parsed.sub_questions))
                    return parsed
            logger.warning("ResearchPlanner: no tool call returned, using fallback")
            return self._fallback_plan(topic)

        args = llm_response.tool_calls[0].arguments
        sub_questions_raw = args.get("sub_questions", [])

        sub_questions: list[SubQuestion] = []
        for idx, sq_data in enumerate(sub_questions_raw):
            keywords = []
            keywords.extend(sq_data.get("keywords_zh") or [])
            keywords.extend(sq_data.get("keywords_en") or [])
            sub_questions.append(
                SubQuestion(
                    id=idx + 1,
                    question=sq_data.get("question", ""),
                    keywords=keywords,
                    priority=sq_data.get("priority", 3),
                )
            )

        if not sub_questions:
            logger.warning("ResearchPlanner: empty sub_questions, using fallback")
            return self._fallback_plan(topic)

        logger.info("ResearchPlanner: generated {} sub-questions", len(sub_questions))
        return ResearchPlan(topic=topic, sub_questions=sub_questions)

    def _fallback_plan(self, topic: str) -> ResearchPlan:
        """Fallback plan when LLM fails to return structured output."""
        t = topic.strip()
        return ResearchPlan(
            topic=t,
            sub_questions=[
                SubQuestion(id=1, question=f"{t} 概述与核心技术原理", keywords=[t, "3DGS", "Gaussian Splatting", "原理"], priority=1),
                SubQuestion(id=2, question=f"{t} 的主要应用场景与案例", keywords=[t, "应用", "application", "SLAM", "重建"], priority=2),
                SubQuestion(id=3, question=f"{t} 的发展趋势与挑战", keywords=[t, "发展", "挑战", "challenge", "未来趋势"], priority=3),
            ],
        )

    def _parse_plan_from_content(self, topic: str, content: str) -> ResearchPlan:
        """Parse a research plan from plain text content (fallback when tool_call unavailable)."""
        import json
        import re

        # Try JSON block first
        json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                sq_raw = data.get("sub_questions") or data.get("questions") or []
                if sq_raw:
                    sub_questions = []
                    for idx, sq_data in enumerate(sq_raw):
                        keywords = []
                        keywords.extend(sq_data.get("keywords_zh") or [])
                        keywords.extend(sq_data.get("keywords_en") or [])
                        keywords.extend(sq_data.get("keywords") or [])
                        if not keywords:
                            keywords = [sq_data.get("question", "")]
                        sub_questions.append(
                            SubQuestion(
                                id=idx + 1,
                                question=sq_data.get("question", ""),
                                keywords=keywords,
                                priority=sq_data.get("priority", idx + 1),
                            )
                        )
                    if sub_questions:
                        return ResearchPlan(topic=topic, sub_questions=sub_questions)
            except (json.JSONDecodeError, ValueError):
                pass

        # Parse numbered list: "1. question" or "- question"
        lines = content.split("\n")
        sub_questions = []
        for line in lines:
            line = line.strip()
            m = re.match(r"^[\d\-\*]+\.?\s*(.+)", line)
            if m and len(m.group(1)) > 5:
                q = m.group(1).strip().rstrip(".,;:")
                sub_questions.append(
                    SubQuestion(id=len(sub_questions) + 1, question=q, keywords=[q], priority=len(sub_questions) + 1)
                )
        if len(sub_questions) >= 2:
            return ResearchPlan(topic=topic, sub_questions=sub_questions[:6])
        return ResearchPlan(topic=topic, sub_questions=[])
