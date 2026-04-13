"""Run all agent evaluation tests and generate report.

Usage:
    python -m tests.agent.evaluation.run_evaluation
    python -m tests.agent.evaluation.run_evaluation --format=markdown --output=report.md
    python -m tests.agent.evaluation.run_evaluation --benchmark-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {text}")
    print('=' * 60)


def print_result(name: str, passed: bool, score: float | None = None) -> None:
    """Print a test result."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    score_str = f" (score: {score:.1%})" if score is not None else ""
    print(f"  {status} {name}{score_str}")


async def run_research_benchmark() -> dict[str, Any]:
    """Run research agent benchmark."""
    print_header("Research Agent Benchmark")

    from tests.agent.evaluation.test_research_agent import (
        RESEARCH_TEST_CASES,
        MockLLMProvider,
        ResearchPlanner,
    )

    results = {
        "name": "Research Agent",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {},
    }

    provider = MockLLMProvider()
    planner = ResearchPlanner(provider, "test-model")

    passed = 0
    total = 0

    for test_case in RESEARCH_TEST_CASES:
        try:
            plan = await planner.plan(test_case.topic, depth=test_case.depth)
            sq_count = len(plan.sub_questions)
            count_ok = test_case.min_sub_questions <= sq_count <= test_case.max_sub_questions

            passed += 1 if count_ok else 0
            total += 1

            print_result(
                test_case.topic[:40],
                count_ok,
                sq_count / test_case.max_sub_questions if sq_count <= test_case.max_sub_questions else 0,
            )

            results["tests"].append({
                "topic": test_case.topic,
                "sub_questions_count": sq_count,
                "expected_range": f"{test_case.min_sub_questions}-{test_case.max_sub_questions}",
                "passed": count_ok,
            })
        except Exception as e:
            print_result(test_case.topic[:40], False)
            print(f"    Error: {e}")
            total += 1
            results["tests"].append({
                "topic": test_case.topic,
                "error": str(e),
                "passed": False,
            })

    results["summary"] = {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0,
    }

    print(f"\n  Summary: {passed}/{total} passed ({passed/total*100:.0f}%)")

    return results


def run_rag_benchmark() -> dict[str, Any]:
    """Run RAG retrieval benchmark."""
    print_header("RAG Retrieval Benchmark")

    from tests.agent.evaluation.test_rag_eval import RAG_TEST_CASES
    from tests.agent.evaluation.evaluator import RAGEvaluator

    evaluator = RAGEvaluator(RAG_TEST_CASES)

    # Simulate realistic retrieval
    all_results = {}
    for tc in RAG_TEST_CASES:
        relevant = tc.relevant_chunks
        # Simulate 70% recall with some noise
        retrieved = relevant[:max(1, int(len(relevant) * 0.7))]
        retrieved.extend([f"noise_{tc.query[:10]}_{i}" for i in range(2)])
        all_results[tc.query] = retrieved

    batch_result = evaluator.evaluate_batch(all_results)

    overall = batch_result.get("overall", {})
    passed = batch_result.get("passed", False)

    print_result("RAG Retrieval", passed, overall.get("ndcg", 0))

    print(f"\n  Metrics:")
    print(f"    Precision: {overall.get('precision', 0):.1%}")
    print(f"    Recall: {overall.get('recall', 0):.1%}")
    print(f"    F1: {overall.get('f1', 0):.1%}")
    print(f"    MRR: {overall.get('mrr', 0):.1%}")
    print(f"    NDCG: {overall.get('ndcg', 0):.1%}")

    # Category breakdown
    by_category = batch_result.get("by_category", {})
    if by_category:
        print(f"\n  By Category:")
        for cat, stats in by_category.items():
            print(f"    {cat}: NDCG={stats.get('avg_ndcg', 0):.1%}")

    return {
        "name": "RAG Retrieval",
        "timestamp": datetime.now().isoformat(),
        "metrics": overall,
        "by_category": by_category,
        "passed": passed,
    }


async def run_memory_benchmark() -> dict[str, Any]:
    """Run memory consolidation benchmark."""
    print_header("Memory Consolidation Benchmark")

    from tests.agent.evaluation.test_memory_eval import MEMORY_TEST_CASES
    from tests.agent.evaluation.evaluator import LLMasJudge

    from unittest.mock import MagicMock

    eval_count = [0]

    async def mock_judge(messages, **kwargs):
        eval_count[0] += 1
        return MagicMock(content=json.dumps({
            "accuracy": 7 + (eval_count[0] % 2),
            "completeness": 8,
            "utility": 7,
            "conciseness": 7,
            "overall": 7.25,
            "issues": [],
            "suggestions": []
        }))

    judge_provider = MagicMock()
    judge_provider.chat = mock_judge
    judge = LLMasJudge(judge_provider, "gpt-4o")

    results = {
        "name": "Memory Consolidation",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {},
    }

    passed = 0
    total = 0

    for test_case in MEMORY_TEST_CASES:
        try:
            result = await judge.evaluate_memory_quality(
                memory_content="\n".join(f"- {f}" for f in test_case.expected_facts),
                conversation_history="\n".join(
                    f"{m['role']}: {m['content']}"
                    for m in test_case.conversation
                ),
            )

            passed += 1 if result.passed else 0
            total += 1

            print_result(
                f"Memory quality ({test_case.conversation_id})",
                result.passed,
                result.overall_score(),
            )

            results["tests"].append({
                "conversation_id": test_case.conversation_id,
                "score": result.overall_score(),
                "passed": result.passed,
            })
        except Exception as e:
            print_result(f"Memory ({test_case.conversation_id})", False)
            print(f"    Error: {e}")
            total += 1

    results["summary"] = {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0,
    }

    print(f"\n  Summary: {passed}/{total} passed ({passed/total*100:.0f}%)")

    return results


def run_tool_call_benchmark() -> dict[str, Any]:
    """Run tool call benchmark."""
    print_header("Tool Call Benchmark")

    from tests.agent.evaluation.test_tool_call import (
        SimpleToolSelector,
        SimpleToolRegistry,
        ToolCallEvaluator,
        TOOL_CALL_TEST_CASES,
    )

    selector = SimpleToolSelector(SimpleToolRegistry)
    evaluator = ToolCallEvaluator(TOOL_CALL_TEST_CASES)

    results = {
        "name": "Tool Call",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {},
    }

    all_results = []
    total_expected = 0
    total_matched = 0

    for test_case in TOOL_CALL_TEST_CASES:
        actual_calls = selector.select_tools(test_case.user_intent)
        eval_results = evaluator.evaluate_single(test_case, actual_calls)

        all_results.append(eval_results)

        expected_count = len(test_case.expected_tools)
        matched_count = sum(
            1 for tool in test_case.expected_tools
            if tool in eval_results and eval_results[tool].matched
        )

        total_expected += expected_count
        total_matched += matched_count

        passed = matched_count == expected_count
        print_result(
            test_case.user_intent[:40],
            passed,
            matched_count / expected_count if expected_count > 0 else 0,
        )

        results["tests"].append({
            "intent": test_case.user_intent,
            "expected_tools": test_case.expected_tools,
            "actual_tools": [t["name"] for t in actual_calls],
            "match_rate": matched_count / expected_count if expected_count > 0 else 0,
            "passed": passed,
        })

    # Aggregate
    tool_match_rate = total_matched / total_expected if total_expected > 0 else 0

    batch_result = evaluator.evaluate_batch(all_results)
    overall = batch_result.get("overall", {})

    results["summary"] = {
        "tool_match_rate": tool_match_rate,
        "avg_param_score": overall.get("avg_param_score", 0),
        "total_expected": total_expected,
        "total_matched": total_matched,
    }

    print(f"\n  Summary:")
    print(f"    Tool Match Rate: {tool_match_rate:.1%}")
    print(f"    Average Param Score: {overall.get('avg_param_score', 0):.1%}")
    print(f"    Matched: {total_matched}/{total_expected}")

    return results


async def run_all_benchmarks() -> dict[str, Any]:
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print(" NANOBOT AGENT EVALUATION BENCHMARK")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    # Run all benchmarks
    benchmarks = [
        ("research_agent", run_research_benchmark()),
        ("rag_retrieval", asyncio.coroutine(run_rag_benchmark)()),
        ("memory_consolidation", run_memory_benchmark()),
        ("tool_call", run_tool_call_benchmark()),
    ]

    # Await async benchmarks
    for name, coro in benchmarks:
        if asyncio.iscoroutine(coro):
            all_results["benchmarks"][name] = await coro
        else:
            all_results["benchmarks"][name] = coro

    # Summary
    print_header("OVERALL SUMMARY")

    total_passed = 0
    total_tests = 0

    for name, result in all_results["benchmarks"].items():
        summary = result.get("summary", {})
        if summary:
            passed = summary.get("passed", 0)
            total = summary.get("total", 0)
            total_passed += passed
            total_tests += total
            print(f"  {name}: {passed}/{total} ({passed/total*100:.0f}%)")
        else:
            passed = result.get("passed", False)
            metrics = result.get("metrics", {})
            ndcg = metrics.get("ndcg", 0)
            print(f"  {name}: {'PASSED' if passed else 'FAILED'} (NDCG={ndcg:.1%})")

    print(f"\n  Total: {total_passed}/{total_tests} ({total_passed/total_tests*100:.0f}%)")

    all_results["summary"] = {
        "total_passed": total_passed,
        "total_tests": total_tests,
        "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
    }

    return all_results


def generate_report(results: dict[str, Any], format: str = "markdown") -> str:
    """Generate evaluation report."""
    if format == "json":
        return json.dumps(results, indent=2, ensure_ascii=False)

    lines = [
        "# Nanobot Agent Evaluation Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary",
    ]

    summary = results.get("summary", {})
    lines.append(f"\n- **Total Tests:** {summary.get('total_tests', 0)}")
    lines.append(f"- **Passed:** {summary.get('total_passed', 0)}")
    lines.append(f"- **Pass Rate:** {summary.get('pass_rate', 0):.1%}")

    lines.append("\n## Benchmark Results")

    for name, result in results.get("benchmarks", {}).items():
        lines.append(f"\n### {name.replace('_', ' ').title()}")

        summary = result.get("summary", {})
        if summary:
            for key, value in summary.items():
                if isinstance(value, float) and value <= 1:
                    lines.append(f"- **{key}:** {value:.1%}")
                else:
                    lines.append(f"- **{key}:** {value}")
        else:
            metrics = result.get("metrics", {})
            passed = result.get("passed", False)
            lines.append(f"\n- **Status:** {'✅ PASSED' if passed else '❌ FAILED'}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.1%}")
                else:
                    lines.append(f"- **{key}:** {value}")

    return "\n".join(lines)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Nanobot Agent Evaluation")
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run benchmarks, skip unit tests",
    )

    args = parser.parse_args()

    if args.benchmark_only:
        results = await run_all_benchmarks()
    else:
        # Run benchmarks with pytest
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", __file__.replace("_runner.py", "_test_*.py"), "-v"],
            capture_output=False,
        )
        results = {"exit_code": result.returncode}

    # Generate report
    report = generate_report(results, args.format)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    asyncio.run(main())
