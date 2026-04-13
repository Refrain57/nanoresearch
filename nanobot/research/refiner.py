"""Research refiner — decides whether to continue iterating and identifies supplementary searches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.research.types import (
    ResearchConfig,
    ResearchPlan,
    SubQuestion,
    SynthesisResult,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_REFINE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "refine_research",
            "description": "决定是否需要补充搜索，以及补充哪些搜索方向",
            "parameters": {
                "type": "object",
                "properties": {
                    "should_continue": {
                        "type": "boolean",
                        "description": "是否需要补充搜索",
                    },
                    "new_sub_questions": {
                        "type": "array",
                        "description": "需要新增的子问题",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "搜索关键词（中英文混合）",
                                },
                                "priority": {"type": "integer"},
                                "reason": {
                                    "type": "string",
                                    "description": "为什么要新增这个子问题",
                                },
                            },
                            "required": ["question", "keywords", "priority"],
                        },
                    },
                    "additional_keywords": {
                        "type": "array",
                        "description": "在现有子问题上补充的关键词",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sub_question_id": {"type": "integer"},
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "reason": {
                        "type": "string",
                        "description": "继续或停止的理由",
                    },
                },
                "required": ["should_continue"],
            },
        },
    }
]

_SYSTEM_PROMPT = """你是一个研究质量评估专家。给定当前的研究进展，请判断：

1. **是否需要继续搜索？** 在以下情况下继续：
   - 某个子问题信息不够充分（coverage = insufficient/partial）
   - 存在重要的知识空白
   - 关键发现存在矛盾需要更多来源验证

2. **如何补充？** 如果需要继续，提供：
   - 新增的子问题（如果当前子问题集有重大遗漏）
   - 现有子问题的补充关键词（更精确/更广泛的搜索词）
   - 补充的理由

3. **如果不需要继续**：说明现有信息已经足够支撑报告的原因。

请调用 refine_research 工具返回决策结果。"""

_USER_TEMPLATE = """## 研究主题
{topic}

## 当前子问题
{sub_questions}

## 覆盖度评估
{coverage_assessment}

## 矛盾点
{contradictions}

## 知识空白
{knowledge_gaps}

## 当前迭代轮次
{iteration} / {max_iterations}

请决定是否需要补充搜索。"""


class ResearchRefiner:
    """Decides whether to iterate and identifies supplementary search directions."""

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    def should_continue(
        self,
        synthesis: SynthesisResult,
        iteration: int,
        config: ResearchConfig,
    ) -> bool:
        """Quick check — can we stop iterating?"""
        if iteration >= config.max_iterations:
            return False
        if synthesis.coverage_score >= config.min_coverage_threshold:
            return False
        return True

    async def refine(
        self,
        plan: ResearchPlan,
        synthesis: SynthesisResult,
        config: ResearchConfig,
    ) -> ResearchPlan | None:
        """Analyze synthesis and decide whether to refine the plan.

        Args:
            plan: Current research plan.
            synthesis: Current synthesis result.
            config: Research configuration.

        Returns:
            Updated ResearchPlan with new sub-questions/keywords, or None if done.
        """
        logger.info(
            "ResearchRefiner: coverage={:.2f}, iteration={}, max_iter={}",
            synthesis.coverage_score, plan.iteration, config.max_iterations,
        )

        # Quick exit conditions
        if plan.iteration >= config.max_iterations:
            logger.info("ResearchRefiner: max iterations reached, stopping")
            return None
        if synthesis.coverage_score >= config.min_coverage_threshold:
            logger.info("ResearchRefiner: coverage threshold met, stopping")
            return None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    topic=plan.topic,
                    sub_questions=self._format_sub_questions(plan.sub_questions),
                    coverage_assessment=self._format_coverage(synthesis),
                    contradictions=self._format_contradictions(synthesis),
                    knowledge_gaps=self._format_gaps(synthesis),
                    iteration=plan.iteration,
                    max_iterations=config.max_iterations,
                ),
            },
        ]

        logger.info("ResearchRefiner: LLM deciding refinement strategy")
        llm_response = await self.provider.chat_with_retry(
            messages=messages,
            tools=_REFINE_TOOL,
            model=self.model,
            max_tokens=1536,
            temperature=0.2,
        )

        if not llm_response.has_tool_calls:
            logger.warning("ResearchRefiner: no tool call, auto-advising continuation")
            return self._auto_refine(plan, synthesis)

        args = llm_response.tool_calls[0].arguments
        should_continue_flag = args.get("should_continue", False)
        reason = args.get("reason", "")
        logger.info("ResearchRefiner: should_continue={}, reason={}", should_continue_flag, reason)

        if not should_continue_flag:
            return None

        # Apply refinements
        new_plan = self._apply_refinements(plan, args)
        new_plan.iteration = plan.iteration + 1
        return new_plan

    def _apply_refinements(self, plan: ResearchPlan, args: dict[str, Any]) -> ResearchPlan:
        """Apply LLM-suggested refinements to the plan."""
        # Add new sub-questions
        next_id = max((sq.id for sq in plan.sub_questions), default=0) + 1
        for sq_data in args.get("new_sub_questions") or []:
            plan.sub_questions.append(
                SubQuestion(
                    id=next_id,
                    question=sq_data.get("question", ""),
                    keywords=sq_data.get("keywords") or [],
                    priority=sq_data.get("priority", 4),
                    status="pending",
                )
            )
            next_id += 1

        # Add additional keywords to existing sub-questions
        for kw_data in args.get("additional_keywords") or []:
            sq_id = kw_data.get("sub_question_id")
            keywords = kw_data.get("keywords") or []
            for sq in plan.sub_questions:
                if sq.id == sq_id:
                    for kw in keywords:
                        if kw not in sq.keywords:
                            sq.keywords.append(kw)
                    break

        return plan

    def _auto_refine(self, plan: ResearchPlan, synthesis: SynthesisResult) -> ResearchPlan | None:
        """Fallback: auto-generate supplementary searches for low-coverage areas."""
        if synthesis.coverage_score < 0.3:
            # Very low coverage — add a catch-all sub-question
            next_id = max((sq.id for sq in plan.sub_questions), default=0) + 1
            plan.sub_questions.append(
                SubQuestion(
                    id=next_id,
                    question=f"{plan.topic} 的全面综述",
                    keywords=[plan.topic, "survey", "overview", "综述"],
                    priority=5,
                    status="pending",
                )
            )
            plan.iteration = plan.iteration + 1
            return plan
        return None

    def _format_sub_questions(self, sub_questions: list[SubQuestion]) -> str:
        return "\n".join(
            f"- [{sq.id}] {sq.question} (优先级 {sq.priority}, 状态 {sq.status}, 来源 {len(sq.results)} 个)"
            for sq in sub_questions
        )

    def _format_coverage(self, synthesis: SynthesisResult) -> str:
        if synthesis.coverage_score < 0.4:
            return f"低覆盖度 ({synthesis.coverage_score:.0%})"
        elif synthesis.coverage_score < 0.7:
            return f"中覆盖度 ({synthesis.coverage_score:.0%})"
        return f"高覆盖度 ({synthesis.coverage_score:.0%})"

    def _format_contradictions(self, synthesis: SynthesisResult) -> str:
        if not synthesis.contradictions:
            return "无明显矛盾"
        return "\n".join(
            f"- {c.topic}: {c.viewpoint_a[:50]} vs {c.viewpoint_b[:50]}"
            for c in synthesis.contradictions[:3]
        )

    def _format_gaps(self, synthesis: SynthesisResult) -> str:
        if not synthesis.knowledge_gaps:
            return "无明显知识空白"
        return "\n".join(f"- {g.description}" for g in synthesis.knowledge_gaps[:3])
