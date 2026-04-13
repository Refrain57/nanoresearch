"""Core evaluation framework for Agent components.

This module provides:
- LLM-as-Judge evaluation for Research Agent and Memory Consolidation
- Ground-truth based evaluation for RAG retrieval
- Tool call correctness evaluation
- Standardized metrics and reporting
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import tiktoken


# ============================================================================
# Metrics Definitions
# ============================================================================

class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"  # Binary: correct or not
    QUALITY = "quality"  # Score 0-10
    COVERAGE = "coverage"  # Percentage covered
    RELEVANCE = "relevance"  # 0-1 score
    FIDELITY = "fidelity"  # How well it matches ground truth


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    name: str
    metric_type: MetricType
    value: float
    max_value: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        """Normalize value to 0-1 range."""
        return self.value / self.max_value if self.max_value > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "max_value": self.max_value,
            "normalized": self.normalized,
            "details": self.details,
        }


@dataclass
class EvaluationResult:
    """Result of a complete evaluation run."""
    test_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: list[MetricResult] = field(default_factory=list)
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, metric: MetricResult) -> None:
        self.metrics.append(metric)

    def overall_score(self) -> float:
        """Calculate overall score as weighted average."""
        if not self.metrics:
            return 0.0
        return sum(m.normalized for m in self.metrics) / len(self.metrics)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Evaluation: {self.test_name}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Overall Score: {self.overall_score():.2%}",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            "",
            "Metrics:",
        ]
        for m in self.metrics:
            lines.append(f"  - {m.name}: {m.value:.2f}/{m.max_value} ({m.normalized:.1%})")
        if self.details:
            lines.append("")
            lines.append("Details:")
            for k, v in self.details.items():
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score(),
            "passed": self.passed,
            "metrics": [m.to_dict() for m in self.metrics],
            "details": self.details,
        }


# ============================================================================
# LLM-as-Judge Evaluator
# ============================================================================

class LLMJudgeProtocol(Protocol):
    """Protocol for LLM provider used as judge."""

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        """Send chat request and return response."""
        ...


_JUDGE_SYSTEM_PROMPT = """你是一个严格的研究报告评估专家。你的职责是评估AI生成的研究报告质量。

评分标准（每项 0-10 分）：
1. 完整性 (completeness): 报告是否回答了所有子问题？
2. 准确性 (accuracy): 事实是否正确？引用来源是否可靠？
3. 可读性 (readability): 结构是否清晰？逻辑是否通顺？
4. 深度 (depth): 是否有足够的细节和分析？

评分说明：
- 8-10: 优秀，满足标准且有独到见解
- 5-7: 合格，基本满足要求但有改进空间
- 3-4: 较差，部分满足要求
- 0-2: 不合格，严重缺失或错误

请用JSON格式返回评估结果：
{
    "completeness": 0-10,
    "accuracy": 0-10,
    "readability": 0-10,
    "depth": 0-10,
    "overall": 0-10,
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["缺点1", "缺点2"],
    "suggestions": ["改进建议1", "改进建议2"]
}
"""

_JUDGE_CONFIDENCE_PROMPT = """你是一个事实核查专家。请评估以下陈述是否准确。

陈述：{statement}
相关来源：{sources}

请判断：
1. 准确度 (accuracy): 陈述与来源的吻合程度 (0-1)
2. 置信度 (confidence): 你对判断的确信程度 (0-1)

JSON格式返回：
{
    "accurate": true/false,
    "accuracy": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reason": "判断理由"
}
"""


_MEMORY_QUALITY_PROMPT = """你是一个记忆质量评估专家。请评估以下记忆内容的质量。

长时记忆内容：
{memory_content}

对话历史：
{conversation_history}

评估标准：
1. 准确性：记忆内容是否与对话历史一致？
2. 完整性：是否捕捉了关键信息？
3. 实用性：这些记忆对未来的对话是否有帮助？
4. 简洁性：是否避免了冗余信息？

JSON格式返回：
{
    "accuracy": 0-10,
    "completeness": 0-10,
    "utility": 0-10,
    "conciseness": 0-10,
    "overall": 0-10,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1"]
}
"""


class LLMasJudge:
    """LLM-as-Judge evaluator for agent outputs."""

    def __init__(self, judge_provider: LLMJudgeProtocol, model: str = "gpt-4o"):
        self.provider = judge_provider
        self.model = model
        self._encoding = tiktoken.get_encoding("cl100k_base")

    async def evaluate_research_report(
        self,
        report: str,
        research_topic: str,
        sub_questions: list[str],
        sources: list[str],
    ) -> EvaluationResult:
        """Evaluate a research report using LLM judge."""
        prompt = f"""## 研究主题
{research_topic}

## 子问题
{chr(10).join(f"- {q}" for q in sub_questions)}

## 检索到的来源
{chr(10).join(f"- {s}" for s in sources[:10])}

## 待评估的报告
{report}

{_JUDGE_SYSTEM_PROMPT}"""

        messages = [
            {"role": "system", "content": "你是一个研究报告评估专家。"},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat(messages=messages, model=self.model, max_tokens=2048)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            import json
            data = json.loads(content)

            result = EvaluationResult(
                test_name=f"research_report_{research_topic[:20]}",
                passed=data.get("overall", 0) >= 6.0,
                details={
                    "topic": research_topic,
                    "sub_questions_count": len(sub_questions),
                    "sources_count": len(sources),
                    "report_length": len(report),
                    "report_tokens": len(self._encoding.encode(report)),
                }
            )

            result.add_metric(MetricResult(
                name="completeness",
                metric_type=MetricType.QUALITY,
                value=data.get("completeness", 0),
                max_value=10,
                details={"reason": "LLM judge evaluation"}
            ))
            result.add_metric(MetricResult(
                name="accuracy",
                metric_type=MetricType.QUALITY,
                value=data.get("accuracy", 0),
                max_value=10,
            ))
            result.add_metric(MetricResult(
                name="readability",
                metric_type=MetricType.QUALITY,
                value=data.get("readability", 0),
                max_value=10,
            ))
            result.add_metric(MetricResult(
                name="depth",
                metric_type=MetricType.QUALITY,
                value=data.get("depth", 0),
                max_value=10,
            ))
            result.add_metric(MetricResult(
                name="overall",
                metric_type=MetricType.QUALITY,
                value=data.get("overall", 0),
                max_value=10,
                details={
                    "strengths": data.get("strengths", []),
                    "weaknesses": data.get("weaknesses", []),
                    "suggestions": data.get("suggestions", []),
                }
            ))

            return result

        except Exception as e:
            return EvaluationResult(
                test_name=f"research_report_{research_topic[:20]}",
                passed=False,
                details={"error": str(e)}
            )

    async def evaluate_memory_quality(
        self,
        memory_content: str,
        conversation_history: str,
    ) -> EvaluationResult:
        """Evaluate memory consolidation quality."""
        prompt = f"""## 长时记忆内容
{memory_content or "(空)"}"""

        messages = [
            {"role": "system", "content": "你是一个记忆质量评估专家。"},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat(messages=messages, model=self.model, max_tokens=2048)
            content = response.content if hasattr(response, 'content') else str(response)
            data = json.loads(content)

            result = EvaluationResult(
                test_name="memory_consolidation",
                passed=data.get("overall", 0) >= 6.0,
                details={
                    "memory_length": len(memory_content),
                    "conversation_length": len(conversation_history),
                }
            )

            result.add_metric(MetricResult(
                name="accuracy",
                metric_type=MetricType.QUALITY,
                value=data.get("accuracy", 0),
                max_value=10,
            ))
            result.add_metric(MetricResult(
                name="completeness",
                metric_type=MetricType.QUALITY,
                value=data.get("completeness", 0),
                max_value=10,
            ))
            result.add_metric(MetricResult(
                name="utility",
                metric_type=MetricType.QUALITY,
                value=data.get("utility", 0),
                max_value=10,
            ))
            result.add_metric(MetricResult(
                name="conciseness",
                metric_type=MetricType.QUALITY,
                value=data.get("conciseness", 0),
                max_value=10,
            ))

            return result

        except Exception as e:
            return EvaluationResult(
                test_name="memory_consolidation",
                passed=False,
                details={"error": str(e)}
            )

    async def evaluate_statement_accuracy(
        self,
        statement: str,
        sources: list[str],
    ) -> tuple[bool, float, float]:
        """Evaluate if a statement is accurate based on sources.

        Returns:
            (is_accurate, accuracy_score, confidence)
        """
        prompt = _JUDGE_CONFIDENCE_PROMPT.format(
            statement=statement,
            sources="\n".join(f"- {s}" for s in sources[:5])
        )

        messages = [
            {"role": "system", "content": "你是一个事实核查专家。"},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat(messages=messages, model=self.model, max_tokens=512)
            content = response.content if hasattr(response, 'content') else str(response)
            data = json.loads(content)

            return (
                data.get("accurate", False),
                data.get("accuracy", 0.0),
                data.get("confidence", 0.0),
            )
        except Exception:
            return False, 0.0, 0.0


# ============================================================================
# Ground-Truth Based Evaluator (for RAG)
# ============================================================================

@dataclass
class RetrievalTestCase:
    """A single retrieval test case with ground truth."""
    query: str
    relevant_chunks: list[str]  # Chunk IDs that should be retrieved
    relevant_sources: list[str]  # Source URLs that should appear
    category: str = ""  # e.g., "technical", "factual", "comparison"


@dataclass
class RetrievalEvaluationResult:
    """Result of retrieval evaluation."""
    query: str
    retrieved_chunks: list[str]
    relevant_chunks: list[str]
    recall: float  # Percentage of relevant chunks retrieved
    precision: float  # Percentage of retrieved chunks that are relevant
    mrr: float  # Mean Reciprocal Rank of first relevant result
    ndcg: float  # Normalized Discounted Cumulative Gain
    category: str = ""


class RAGEvaluator:
    """Ground-truth based evaluator for RAG retrieval."""

    def __init__(self, test_cases: list[RetrievalTestCase]):
        self.test_cases = {tc.query: tc for tc in test_cases}

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: list[str],
    ) -> RetrievalEvaluationResult:
        """Evaluate a single retrieval result against ground truth."""
        if query not in self.test_cases:
            raise ValueError(f"No ground truth for query: {query}")

        test_case = self.test_cases[query]
        relevant = set(test_case.relevant_chunks)
        retrieved = set(retrieved_chunks)

        # Calculate metrics
        true_positives = len(relevant & retrieved)
        false_positives = len(retrieved - relevant)
        false_negatives = len(relevant - retrieved)

        # Precision & Recall
        precision = true_positives / (true_positives + false_positives) if retrieved else 0.0
        recall = true_positives / (true_positives + false_negatives) if relevant else 0.0

        # MRR
        mrr = 0.0
        for i, chunk in enumerate(retrieved_chunks, 1):
            if chunk in relevant:
                mrr = 1.0 / i
                break

        # NDCG
        ndcg = self._calculate_ndcg(retrieved_chunks, relevant)

        return RetrievalEvaluationResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            relevant_chunks=test_case.relevant_chunks,
            recall=recall,
            precision=precision,
            mrr=mrr,
            ndcg=ndcg,
            category=test_case.category,
        )

    def evaluate_batch(
        self,
        results: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Evaluate multiple retrieval results."""
        evals = []
        for query, retrieved in results.items():
            if query in self.test_cases:
                evals.append(self.evaluate_retrieval(query, retrieved))

        if not evals:
            return {"error": "No matching test cases"}

        # Aggregate metrics
        avg_precision = sum(e.precision for e in evals) / len(evals)
        avg_recall = sum(e.recall for e in evals) / len(evals)
        avg_mrr = sum(e.mrr for e in evals) / len(evals)
        avg_ndcg = sum(e.ndcg for e in evals) / len(evals)

        # F1 score
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

        # Per-category breakdown
        by_category: dict[str, list[RetrievalEvaluationResult]] = {}
        for e in evals:
            if e.category:
                by_category.setdefault(e.category, []).append(e)

        category_stats = {}
        for cat, cat_evals in by_category.items():
            category_stats[cat] = {
                "count": len(cat_evals),
                "avg_precision": sum(x.precision for x in cat_evals) / len(cat_evals),
                "avg_recall": sum(x.recall for x in cat_evals) / len(cat_evals),
                "avg_ndcg": sum(x.ndcg for x in cat_evals) / len(cat_evals),
            }

        return {
            "overall": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": f1,
                "mrr": avg_mrr,
                "ndcg": avg_ndcg,
                "total_queries": len(evals),
            },
            "by_category": category_stats,
            "passed": avg_ndcg >= 0.6,  # Threshold
        }

    @staticmethod
    def _calculate_ndcg(retrieved: list[str], relevant: set[str]) -> float:
        """Calculate NDCG@K."""
        k = len(retrieved)
        if k == 0 or not relevant:
            return 0.0

        # DCG
        dcg = 0.0
        for i, chunk in enumerate(retrieved, 1):
            if chunk in relevant:
                dcg += 1.0 / (i ** 0.5)  # Position weight

        # IDCG (ideal DCG)
        ideal_retrieved = list(relevant)[:k]
        idcg = sum(1.0 / ((i + 1) ** 0.5) for i in range(len(ideal_retrieved)))

        return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# Tool Call Evaluator
# ============================================================================

@dataclass
class ToolCallTestCase:
    """A test case for tool call evaluation."""
    user_intent: str
    conversation_history: list[dict[str, str]]
    expected_tools: list[str]  # Tools that should be called
    expected_params: dict[str, Any] | None = None  # Expected parameters


@dataclass
class ToolCallResult:
    """Result of tool call evaluation."""
    expected_tool: str
    actual_tool: str | None
    matched: bool
    parameter_match: float  # 0-1, how well params match
    details: dict[str, Any] = field(default_factory=dict)


class ToolCallEvaluator:
    """Evaluates tool selection and parameter correctness."""

    def __init__(self, test_cases: list[ToolCallTestCase]):
        self.test_cases = test_cases

    def evaluate_single(
        self,
        test_case: ToolCallTestCase,
        actual_tool_calls: list[dict[str, Any]],
    ) -> dict[str, ToolCallResult]:
        """Evaluate a single tool call against expected."""
        results = {}

        # Check if expected tools were called
        actual_tools = {tc["name"] for tc in actual_tool_calls}

        for expected in test_case.expected_tools:
            matched = expected in actual_tools
            actual = None
            for tc in actual_tool_calls:
                if tc["name"] == expected:
                    actual = expected
                    break

            # Parameter match
            param_match = 1.0
            if test_case.expected_params and actual:
                for tc in actual_tool_calls:
                    if tc["name"] == expected:
                        params = tc.get("arguments", {})
                        expected_params = test_case.expected_params
                        matches = sum(
                            1 for k, v in expected_params.items()
                            if k in params and params[k] == v
                        )
                        param_match = matches / len(expected_params) if expected_params else 1.0
                        break

            results[expected] = ToolCallResult(
                expected_tool=expected,
                actual_tool=actual,
                matched=matched,
                parameter_match=param_match,
            )

        return results

    def evaluate_batch(
        self,
        results: list[dict[str, ToolCallResult]],
    ) -> dict[str, Any]:
        """Aggregate evaluation results."""
        if not results:
            return {"error": "No results to evaluate"}

        total_tools = sum(len(r) for r in results)
        matched_tools = sum(1 for r in results for v in r.values() if v.matched)
        avg_param_match = sum(
            v.parameter_match for r in results for v in r.values() if v.matched
        ) / max(1, matched_tools)

        tool_match_rate = matched_tools / total_tools if total_tools > 0 else 0.0

        # Per-tool breakdown
        tool_stats: dict[str, dict[str, Any]] = {}
        for batch_result in results:
            for tool_name, result in batch_result.items():
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {"matched": 0, "total": 0, "param_scores": []}
                tool_stats[tool_name]["total"] += 1
                if result.matched:
                    tool_stats[tool_name]["matched"] += 1
                tool_stats[tool_name]["param_scores"].append(result.parameter_match)

        for tool_name in tool_stats:
            s = tool_stats[tool_name]
            s["match_rate"] = s["matched"] / s["total"] if s["total"] > 0 else 0.0
            s["avg_param_score"] = sum(s["param_scores"]) / len(s["param_scores"]) if s["param_scores"] else 0.0

        return {
            "overall": {
                "tool_match_rate": tool_match_rate,
                "avg_param_score": avg_param_match,
                "total_tests": len(results),
            },
            "by_tool": tool_stats,
            "passed": tool_match_rate >= 0.7,  # 70% threshold
        }


# ============================================================================
# Report Generator
# ============================================================================

class EvaluationReporter:
    """Generate evaluation reports in various formats."""

    @staticmethod
    def generate_report(
        results: list[EvaluationResult],
        output_path: Path | None = None,
        format: str = "text",
    ) -> str:
        """Generate evaluation report."""
        if format == "json":
            report = json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False)
        elif format == "markdown":
            report = EvaluationReporter._generate_markdown(results)
        else:
            report = "\n\n".join(r.summary() for r in results)

        if output_path:
            output_path.write_text(report, encoding="utf-8")

        return report

    @staticmethod
    def _generate_markdown(results: list[EvaluationResult]) -> str:
        """Generate markdown report."""
        lines = [
            "# Agent Evaluation Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary",
        ]

        overall_scores = [r.overall_score() for r in results]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        passed_count = sum(1 for r in results if r.passed)

        lines.extend([
            f"- Total Tests: {len(results)}",
            f"- Passed: {passed_count}",
            f"- Failed: {len(results) - passed_count}",
            f"- Average Score: {avg_score:.1%}",
            "\n## Detailed Results",
        ])

        for i, result in enumerate(results, 1):
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            lines.extend([
                f"\n### {i}. {result.test_name} [{status}]",
                f"\n**Overall Score:** {result.overall_score():.1%}",
                "",
                "| Metric | Score |",
                "|--------|-------|",
            ])
            for m in result.metrics:
                lines.append(f"| {m.name} | {m.value:.1f}/{m.max_value} ({m.normalized:.0%}) |")

        return "\n".join(lines)
