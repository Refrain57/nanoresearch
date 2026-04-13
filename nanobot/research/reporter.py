"""Report generator — produces structured Markdown reports with section-by-section writing."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.research.types import (
    Finding,
    ReportMetrics,
    ResearchPlan,
    SearchResult,
    SynthesisResult,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_REPORT_SECTION_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_section",
            "description": "撰写报告的一节",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
        },
    }
]

_EVALUATE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_report",
            "description": "评估研究报告的质量",
            "parameters": {
                "type": "object",
                "properties": {
                    "completeness": {
                        "type": "number",
                        "description": "完整性评分 0-10",
                    },
                    "accuracy": {"type": "number", "description": "准确性评分 0-10"},
                    "readability": {"type": "number", "description": "可读性评分 0-10"},
                    "overall": {"type": "number", "description": "综合评分 0-10"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "weaknesses": {"type": "array", "items": {"type": "string"}},
                    "suggestions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["overall", "completeness", "accuracy", "readability"],
            },
        },
    }
]

_SECTION_SYSTEM_PROMPT = """你是一个专业的研究报告撰写专家。

任务：根据提供的原始来源，撰写报告的一节。

要求：
1. 直接引用原文中的具体数据、案例、技术细节
2. 不要泛泛而谈，要有事实支撑
3. 字数：600-800字（充实但不要过度展开）
4. 使用 Markdown 格式
5. 引用来源时标注 [来源X]

请调用 write_section 工具返回结果。"""

_SECTION_USER_TEMPLATE = """## 报告主题
{topic}

## 本节任务
标题：{section_title}
说明：{section_desc}

## 本节相关发现摘要（作为全局参考）
{findings_summary}

## 本节相关来源（原文全文）
{sources_formatted}

---
请撰写本节内容（600-800字），直接引用原文细节。"""

_INTEGRATE_SYSTEM_PROMPT = """你是一个专业的报告整合专家。

任务：整合各章节，润色为完整报告。

要求：
1. 润色每节内容，消除重复
2. 撰写执行摘要（一句话核心结论）
3. 插入"矛盾与争议"章节
4. 插入"知识空白"章节
5. 调整章节顺序使其流畅
6. 最终格式：Markdown

请调用 generate_report 工具返回结果。"""

_INTEGRATE_USER_TEMPLATE = """## 研究主题
{topic}

## 各章节草稿
{sections}

## 矛盾与争议
{contradictions}

## 知识空白
{knowledge_gaps}

---
请整合为完整报告。"""

_INTEGRATE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "生成整合后的完整报告",
            "parameters": {
                "type": "object",
                "properties": {
                    "executive_summary": {
                        "type": "string",
                        "description": "一句话执行摘要",
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }
]

# Token budget per source in section writing
MAX_CHARS_PER_SOURCE = 8000


class ReportGenerator:
    """Generates structured reports with section-by-section writing."""

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def generate(
        self,
        topic: str,
        synthesis: SynthesisResult,
        plan: ResearchPlan,
    ) -> str:
        """Generate a structured report by writing sections individually.

        Args:
            topic: Original research topic.
            synthesis: Synthesis result with findings, source_assignments, sources.
            plan: Research plan with sub-questions.

        Returns:
            Markdown-formatted report string.
        """
        sections: list[dict[str, str]] = []
        findings_summary = self._format_findings_map(synthesis.findings)

        # Write one section per sub-question
        for sq in plan.sub_questions:
            relevant_sources = self._select_sources_for_sq(
                sq.id, synthesis.source_assignments, synthesis.sources, top_k=3
            )

            if not relevant_sources:
                logger.warning("ReportGenerator: no sources for sub-question {}", sq.id)
                continue

            section_text = await self._write_section(
                topic=topic,
                section_title=sq.question,
                section_desc=f"回答子问题：{sq.question}",
                findings_summary=findings_summary,
                sources=relevant_sources,
            )
            sections.append({"title": sq.question, "content": section_text})
            logger.info("ReportGenerator: wrote section '{}' ({} chars)", sq.question, len(section_text))

        if not sections:
            logger.warning("ReportGenerator: no sections generated, using fallback")
            return self._fallback_report(topic, synthesis, plan)

        # Integrate all sections
        final_report = await self._integrate(
            topic=topic,
            sections=sections,
            contradictions=synthesis.contradictions,
            knowledge_gaps=synthesis.knowledge_gaps,
        )

        logger.info("ReportGenerator: final report generated ({} chars)", len(final_report))
        return final_report

    def _select_sources_for_sq(
        self,
        sq_id: int,
        assignments: list[Any],
        all_sources: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Select top sources most relevant to a sub-question."""
        relevant = sorted(
            [a for a in assignments if a.sub_question_id == sq_id],
            key=lambda x: x.relevance_to_sq,
            reverse=True,
        )[:top_k]

        if not relevant:
            # Fallback: take first few sources
            return all_sources[:top_k]

        url_set = {a.source_url for a in relevant}
        return [s for s in all_sources if s.url in url_set]

    async def _write_section(
        self,
        topic: str,
        section_title: str,
        section_desc: str,
        findings_summary: str,
        sources: list[SearchResult],
    ) -> str:
        """Write a single section by reading relevant sources directly."""
        sources_formatted = self._format_sources_for_section(sources)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _SECTION_USER_TEMPLATE.format(
                    topic=topic,
                    section_title=section_title,
                    section_desc=section_desc,
                    findings_summary=findings_summary,
                    sources_formatted=sources_formatted,
                ),
            },
        ]

        llm_response = await self.provider.chat_with_retry(
            messages=messages,
            tools=_REPORT_SECTION_TOOL,
            model=self.model,
            max_tokens=2048,
            temperature=0.3,
        )

        if llm_response.has_tool_calls:
            args = llm_response.tool_calls[0].arguments
            return args.get("content", "")

        # Fallback: use content directly if structured output failed
        if llm_response.content and len(llm_response.content) > 100:
            return llm_response.content

        # Last resort: summarize sources
        return f"## {section_title}\n\n" + "\n\n".join(
            f"[来源: {s.title}]({s.url})\n{s.content[:500]}" for s in sources[:2]
        )

    def _format_sources_for_section(self, sources: list[SearchResult]) -> str:
        """Format sources for section writing prompt with truncation control."""
        blocks = []
        for i, s in enumerate(sources, 1):
            content = s.content[:MAX_CHARS_PER_SOURCE]
            truncated_note = ""
            if len(s.content) > MAX_CHARS_PER_SOURCE:
                truncated_note = "\n[内容较长，已截取前段用于本节写作]"
            blocks.append(
                f"--- 来源 {i} ---\n"
                f"标题: {s.title}\n"
                f"URL: {s.url}\n"
                f"内容:\n{content}{truncated_note}\n"
            )
        return "\n".join(blocks)

    async def _integrate(
        self,
        topic: str,
        sections: list[dict[str, str]],
        contradictions: list[Any],
        knowledge_gaps: list[Any],
    ) -> str:
        """Integrate sections into final report."""
        sections_text = "\n\n".join(
            f"## {sec['title']}\n\n{sec['content']}" for sec in sections
        )
        contradictions_text = self._format_contradictions(contradictions)
        gaps_text = self._format_gaps(knowledge_gaps)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _INTEGRATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _INTEGRATE_USER_TEMPLATE.format(
                    topic=topic,
                    sections=sections_text,
                    contradictions=contradictions_text,
                    knowledge_gaps=gaps_text,
                ),
            },
        ]

        llm_response = await self.provider.chat_with_retry(
            messages=messages,
            tools=_INTEGRATE_TOOL,
            model=self.model,
            max_tokens=16384,
            temperature=0.2,
        )

        if llm_response.has_tool_calls:
            args = llm_response.tool_calls[0].arguments
            exec_summary = args.get("executive_summary", "")
            final_sections = args.get("sections", [])
            return self._build_markdown(topic, exec_summary, final_sections, contradictions, knowledge_gaps)

        # Fallback: use content directly or assemble manually
        if llm_response.content and len(llm_response.content) > 500:
            return llm_response.content

        # Manual assembly fallback
        lines = [f"# {topic} 研究报告", "", "---", ""]
        for sec in sections:
            lines.append(f"## {sec['title']}")
            lines.append("")
            lines.append(sec["content"])
            lines.append("")
        if contradictions:
            lines.append("## 矛盾与争议")
            for c in contradictions:
                lines.append(f"- {c.topic}: {c.viewpoint_a[:50]} vs {c.viewpoint_b[:50]}")
        if knowledge_gaps:
            lines.append("## 知识空白")
            for g in knowledge_gaps:
                lines.append(f"- {g.description}")
        return "\n".join(lines)

    def _build_markdown(
        self,
        topic: str,
        exec_summary: str,
        sections: list[dict[str, Any]],
        contradictions: list[Any],
        knowledge_gaps: list[Any],
    ) -> str:
        """Build the final Markdown report."""
        lines = [
            f"# {topic} 研究报告",
            "",
            f"**执行摘要**: {exec_summary}",
            "",
            "---",
            "",
        ]

        for sec in sections:
            lines.append(f"## {sec.get('title', '')}")
            lines.append("")
            lines.append(sec.get("content", ""))
            lines.append("")

        if contradictions:
            lines.append("---")
            lines.append("")
            lines.append("## 矛盾与争议")
            for c in contradictions:
                lines.append(f"### {c.topic}")
                lines.append(f"- **观点 A**: {c.viewpoint_a}")
                lines.append(f"- **观点 B**: {c.viewpoint_b}")
                lines.append("")

        if knowledge_gaps:
            lines.append("---")
            lines.append("")
            lines.append("## 知识空白")
            for g in knowledge_gaps:
                lines.append(f"- **{g.description}**")
                if g.suggested_searches:
                    lines.append(f"  建议搜索: {', '.join(g.suggested_searches)}")

        return "\n".join(lines)

    def _format_findings_map(self, findings: list[Finding]) -> str:
        """Format findings as a reference map for section writing."""
        if not findings:
            return "无"
        blocks = []
        for i, f in enumerate(findings[:12], 1):
            sources = ", ".join(f.source_urls[:2]) if f.source_urls else "无来源"
            blocks.append(f"{i}. {f.statement} (置信度: {f.confidence:.0%}, 来源: {sources})")
        return "\n".join(blocks)

    def _format_contradictions(self, contradictions: list[Any]) -> str:
        if not contradictions:
            return "无明显矛盾"
        return "\n".join(
            f"- **{c.topic}**: {c.viewpoint_a[:100]} vs {c.viewpoint_b[:100]}"
            for c in contradictions[:5]
        )

    def _format_gaps(self, gaps: list[Any]) -> str:
        if not gaps:
            return "无明显知识空白"
        return "\n".join(f"- {g.description}" for g in gaps[:5])

    def _fallback_report(
        self,
        topic: str,
        synthesis: SynthesisResult,
        plan: ResearchPlan,
    ) -> str:
        """Fallback report when section writing fails."""
        lines = [
            f"# {topic} 研究报告",
            "",
            f"**执行摘要**: 关于 {topic} 的研究发现见下文。",
            "",
            "---",
            "",
        ]
        for i, sq in enumerate(plan.sub_questions, 1):
            lines.append(f"## {i}. {sq.question}")
            lines.append("")
            lines.append("（详细内容待补充）")
            lines.append("")
        return "\n".join(lines)

    async def self_evaluate(
        self,
        report: str,
        synthesis: SynthesisResult,
        plan: ResearchPlan,
    ) -> ReportMetrics:
        """Evaluate the generated report."""
        sub_q_text = "\n".join(f"- [{sq.id}] {sq.question}" for sq in plan.sub_questions)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "评估研究报告质量。"},
            {
                "role": "user",
                "content": f"""## 研究主题
{plan.topic}

## 原始子问题
{sub_q_text}

## 待评估报告
---
{report[:4000]}
---
""",
            },
        ]

        llm_response = await self.provider.chat_with_retry(
            messages=messages,
            tools=_EVALUATE_TOOL,
            model=self.model,
            max_tokens=1024,
            temperature=0.1,
        )

        if llm_response.has_tool_calls:
            args = llm_response.tool_calls[0].arguments
            return ReportMetrics(
                completeness=float(args.get("completeness", 5.0)),
                accuracy=float(args.get("accuracy", 5.0)),
                readability=float(args.get("readability", 5.0)),
                overall=float(args.get("overall", 5.0)),
                strengths=args.get("strengths") or [],
                weaknesses=args.get("weaknesses") or [],
                suggestions=args.get("suggestions") or [],
            )

        return self._default_metrics()

    def _default_metrics(self) -> ReportMetrics:
        return ReportMetrics(
            completeness=5.0,
            accuracy=5.0,
            readability=5.0,
            overall=5.0,
            strengths=["报告已生成"],
            weaknesses=["评估过程异常"],
            suggestions=["请人工复核"],
        )
