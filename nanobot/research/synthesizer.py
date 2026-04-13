"""Information synthesizer — multi-source analysis, contradiction detection, source mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.research.types import (
    Contradiction,
    Finding,
    KnowledgeGap,
    ResearchPlan,
    SearchResult,
    SourceAssignment,
    SynthesisResult,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_SYNTHESIZE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "synthesize",
            "description": "分析搜索结果，提取高层发现、来源映射、矛盾点和知识空白",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "description": "核心发现列表（高层陈述，不包含详细证据）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement": {
                                    "type": "string",
                                    "description": "发现的核心结论（一句话）",
                                },
                                "source_urls": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "来源 URL 列表",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "置信度 0-1",
                                },
                            },
                            "required": ["statement", "source_urls"],
                        },
                    },
                    "source_assignments": {
                        "type": "array",
                        "description": "来源到子问题的映射（source → sub_question 索引）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_url": {"type": "string"},
                                "sub_question_id": {"type": "integer"},
                                "relevance": {
                                    "type": "number",
                                    "description": "该来源对此子问题的相关度 0-1",
                                },
                            },
                            "required": ["source_url", "sub_question_id", "relevance"],
                        },
                    },
                    "contradictions": {
                        "type": "array",
                        "description": "矛盾观点列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "viewpoint_a": {"type": "string"},
                                "viewpoint_b": {"type": "string"},
                                "source_a": {"type": "string"},
                                "source_b": {"type": "string"},
                            },
                            "required": ["topic", "viewpoint_a", "viewpoint_b"],
                        },
                    },
                    "knowledge_gaps": {
                        "type": "array",
                        "description": "知识空白列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "related_sub_question_id": {"type": "integer"},
                                "suggested_searches": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["description"],
                        },
                    },
                    "coverage_assessment": {
                        "type": "array",
                        "description": "各子问题的覆盖度评估",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sub_question_id": {"type": "integer"},
                                "coverage": {
                                    "type": "string",
                                    "enum": ["sufficient", "partial", "insufficient"],
                                },
                            },
                        },
                    },
                },
            },
        },
    }
]

_SYSTEM_PROMPT = """你是一个专业的信息分析专家。给你一批搜索结果，请完成以下任务（不要压缩内容）：

1. **高层发现**：提取 6-12 个核心结论，每个发现只需一句话核心陈述 + 来源 URL。不要提取详细证据——Reporter 会直接从原文中引用。

2. **来源映射**：为每个来源分配它最相关的子问题。
   - 每个来源可分配给 1-2 个子问题
   - 给出 relevance 评分（0-1）
   - 用途：帮助 Reporter 按章节分批读取相关原文

3. **矛盾观点**：哪些问题不同来源说法不一致？列出观点 A 和观点 B。

4. **知识空白**：哪些方面信息不足？关联到子问题，提供补充搜索建议。

5. **覆盖度评估**：评估每个子问题的信息覆盖程度。

请调用 synthesize 工具返回分析结果。"""

_USER_TEMPLATE = """## 研究主题
{topic}

## 子问题列表
{sub_questions}

## 搜索结果（共 {n} 个来源）
{formatted_results}

请调用 synthesize 工具进行分析。"""


class InformationSynthesizer:
    """Synthesizes multi-source information into structured findings and source mapping."""

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def synthesize(
        self,
        results: list[SearchResult],
        plan: ResearchPlan,
    ) -> SynthesisResult:
        """Synthesize search results into structured output.

        Args:
            results: Scored search results (passed through unchanged).
            plan: The research plan with sub-questions.

        Returns:
            SynthesisResult with findings, source_assignments, contradictions, and gaps.
            sources field contains original SearchResult objects (not truncated).
        """
        if not results:
            logger.warning("InformationSynthesizer: no results to synthesize")
            return SynthesisResult()

        formatted_results = self._format_results_for_prompt(results)
        sub_q_str = "\n".join(
            f"- [{sq.id}] {sq.question} (优先级 {sq.priority})"
            for sq in plan.sub_questions
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    topic=plan.topic,
                    sub_questions=sub_q_str,
                    n=len(results),
                    formatted_results=formatted_results,
                ),
            },
        ]

        logger.info("InformationSynthesizer: synthesizing {} results", len(results))
        llm_response = await self.provider.chat_with_retry(
            messages=messages,
            tools=_SYNTHESIZE_TOOL,
            model=self.model,
            max_tokens=8192,
            temperature=0.2,
        )

        if not llm_response.has_tool_calls:
            if llm_response.content:
                parsed = self._parse_synthesis_from_content(llm_response.content, results, plan)
                if parsed.findings:
                    logger.info("InformationSynthesizer: parsed {} findings from content", len(parsed.findings))
                    return parsed
            logger.warning("InformationSynthesizer: no tool call, using fallback")
            return self._fallback_synthesis(results, plan)

        args = llm_response.tool_calls[0].arguments

        # Parse findings (statement + source_urls only, no evidence)
        findings: list[Finding] = []
        for f_data in args.get("findings") or []:
            findings.append(
                Finding(
                    statement=f_data.get("statement", ""),
                    source_urls=f_data.get("source_urls") or [],
                    confidence=f_data.get("confidence", 0.5),
                )
            )

        # Parse source assignments
        source_assignments: list[SourceAssignment] = []
        url_to_title: dict[str, str] = {r.url: r.title for r in results}
        for sa_data in args.get("source_assignments") or []:
            source_assignments.append(
                SourceAssignment(
                    source_url=sa_data.get("source_url", ""),
                    source_title=url_to_title.get(sa_data.get("source_url", ""), ""),
                    sub_question_id=sa_data.get("sub_question_id", 0),
                    relevance_to_sq=sa_data.get("relevance", 0.5),
                )
            )

        # Parse contradictions
        contradictions: list[Contradiction] = []
        for c_data in args.get("contradictions") or []:
            contradictions.append(
                Contradiction(
                    topic=c_data.get("topic", ""),
                    viewpoint_a=c_data.get("viewpoint_a", ""),
                    viewpoint_b=c_data.get("viewpoint_b", ""),
                    source_a=c_data.get("source_a", ""),
                    source_b=c_data.get("source_b", ""),
                )
            )

        # Parse knowledge gaps
        knowledge_gaps: list[KnowledgeGap] = []
        for g_data in args.get("knowledge_gaps") or []:
            knowledge_gaps.append(
                KnowledgeGap(
                    description=g_data.get("description", ""),
                    related_sub_question_id=g_data.get("related_sub_question_id"),
                    suggested_searches=g_data.get("suggested_searches") or [],
                )
            )

        # Coverage assessment
        coverage_score = self._calc_coverage_score(args.get("coverage_assessment"), plan)

        logger.info(
            "InformationSynthesizer: {} findings, {} source_assignments, {} contradictions, {} gaps, coverage={:.2f}",
            len(findings), len(source_assignments), len(contradictions), len(knowledge_gaps), coverage_score,
        )

        # Pass through original sources unchanged
        return SynthesisResult(
            findings=findings,
            contradictions=contradictions,
            knowledge_gaps=knowledge_gaps,
            coverage_score=coverage_score,
            source_assignments=source_assignments,
            sources=results,  # pass-through: no truncation
        )

    def _format_results_for_prompt(self, results: list[SearchResult]) -> str:
        """Format search results for the LLM prompt."""
        blocks: list[str] = []
        for i, r in enumerate(results, 1):
            block = f"""--- 来源 {i} ---
ID: {i}
标题: {r.title}
URL: {r.url}
类型: {r.source_type}
评分: 可靠性={r.credibility_score:.1f}, 相关度={r.relevance_score:.1f}, 时效性={r.recency_score:.1f}
内容: {r.content[:10000]}
"""
            blocks.append(block)
        return "\n".join(blocks)

    def _calc_coverage_score(
        self,
        coverage_data: Any,
        plan: ResearchPlan,
    ) -> float:
        """Calculate overall coverage score from per-sub-question assessments."""
        if not coverage_data:
            if not plan.sub_questions:
                return 0.5
            total_results = sum(len(sq.results) for sq in plan.sub_questions)
            expected = len(plan.sub_questions) * 3
            return min(1.0, total_results / expected) if expected > 0 else 0.5

        score_map = {"sufficient": 1.0, "partial": 0.6, "insufficient": 0.2}
        items: list[tuple[str, str]] = []
        if isinstance(coverage_data, dict):
            for key, val in coverage_data.items():
                if isinstance(val, str):
                    items.append((key, val))
                elif isinstance(val, dict):
                    cov = val.get("coverage", "") if isinstance(val, dict) else str(val)
                    items.append((key, cov))
        elif isinstance(coverage_data, list):
            for item in coverage_data:
                if isinstance(item, dict):
                    cov = item.get("coverage", "")
                    items.append((str(item.get("sub_question_id", "")), cov))
                elif isinstance(item, str):
                    items.append(("", item))

        if not items:
            if not plan.sub_questions:
                return 0.5
            total_results = sum(len(sq.results) for sq in plan.sub_questions)
            expected = len(plan.sub_questions) * 3
            return min(1.0, total_results / expected) if expected > 0 else 0.5

        total = sum(score_map.get(cov, 0.5) for _, cov in items)
        return total / len(items)

    def _fallback_synthesis(
        self,
        results: list[SearchResult],
        plan: ResearchPlan,
    ) -> SynthesisResult:
        """Fallback when LLM returns no structured output."""
        findings: list[Finding] = []
        for r in results[:10]:
            findings.append(
                Finding(
                    statement=r.title,
                    source_urls=[r.url],
                    confidence=r.final_score,
                )
            )
        coverage_score = min(1.0, len(results) / (len(plan.sub_questions) * 3 + 1))
        return SynthesisResult(
            findings=findings,
            contradictions=[],
            knowledge_gaps=[],
            coverage_score=coverage_score,
            source_assignments=[],
            sources=results,
        )

    def _parse_synthesis_from_content(
        self,
        content: str,
        results: list[SearchResult],
        plan: ResearchPlan,
    ) -> SynthesisResult:
        """Parse synthesis results from plain text (fallback when tool_call unavailable)."""
        import json
        import re

        json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                findings = []
                for f_data in data.get("findings") or []:
                    findings.append(Finding(
                        statement=f_data.get("statement", ""),
                        source_urls=f_data.get("source_urls") or [],
                        confidence=f_data.get("confidence", 0.5),
                    ))
                contradictions = []
                for c_data in data.get("contradictions") or []:
                    contradictions.append(Contradiction(
                        topic=c_data.get("topic", ""),
                        viewpoint_a=c_data.get("viewpoint_a", ""),
                        viewpoint_b=c_data.get("viewpoint_b", ""),
                    ))
                gaps = []
                for g_data in data.get("knowledge_gaps") or []:
                    gaps.append(KnowledgeGap(
                        description=g_data.get("description", ""),
                        suggested_searches=g_data.get("suggested_searches") or [],
                    ))
                coverage_score = data.get("coverage_score", 0.5)
                if findings:
                    return SynthesisResult(
                        findings=findings,
                        contradictions=contradictions,
                        knowledge_gaps=gaps,
                        coverage_score=coverage_score,
                        source_assignments=[],
                        sources=results,
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        return self._fallback_synthesis(results, plan)
