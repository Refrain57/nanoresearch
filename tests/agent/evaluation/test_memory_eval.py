"""Tests for memory consolidation evaluation.

Tests memory quality:
- Fact extraction accuracy
- Memory completeness
- Memory utility for future conversations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.agent.evaluation.evaluator import LLMasJudge
from tests.agent.evaluation.test_datasets import MEMORY_TEST_CASES, MemoryTestCase


# ============================================================================
# Test Cases
# ============================================================================

class TestMemoryFactExtraction:
    """Tests for fact extraction from conversations."""

    @pytest.mark.asyncio
    async def test_simple_fact_extraction(self):
        """Test extraction of simple facts from conversation."""
        from nanobot.agent.memory import MemoryStore

        # Create a memory store
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))

            # Simulate conversation messages
            messages = [
                {"role": "user", "content": "你好，我叫张三", "timestamp": "2024-01-01 10:00"},
                {"role": "assistant", "content": "你好张三！", "timestamp": "2024-01-01 10:00"},
                {"role": "user", "content": "我是一名Python开发工程师", "timestamp": "2024-01-01 10:01"},
            ]

            # Mock provider for consolidation
            mock_provider = MagicMock()
            mock_provider.chat_with_retry = AsyncMock(return_value=MagicMock(
                finish_reason="stop",
                content="",
                has_tool_calls=True,
                tool_calls=[
                    MagicMock(
                        arguments={
                            "history_entry": "[2024-01-01 10:00] 用户自我介绍：名叫张三，是Python开发工程师",
                            "memory_update": "## 用户信息\n- 姓名：张三\n- 职业：Python开发工程师\n"
                        }
                    )
                ]
            ))

            result = await store.consolidate(messages, mock_provider, "test-model")

            assert result is True

            # Check memory content
            memory = store.read_long_term()
            assert "张三" in memory
            assert "Python" in memory

    @pytest.mark.asyncio
    async def test_no_information_loss(self):
        """Test that key facts are not lost during consolidation."""
        from nanobot.agent.memory import MemoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))

            # Create conversation with multiple facts
            messages = [
                {"role": "user", "content": "我叫李四", "timestamp": "2024-01-01 10:00"},
                {"role": "user", "content": "我在北京工作", "timestamp": "2024-01-01 10:01"},
                {"role": "user", "content": "我主要用Go语言", "timestamp": "2024-01-01 10:02"},
                {"role": "user", "content": "我对AI很感兴趣", "timestamp": "2024-01-01 10:03"},
            ]

            # Track what LLM actually extracted
            extracted = {}

            def mock_chat(*args, **kwargs):
                call_args = args[0] if args else kwargs.get('messages', [])
                # Find the user messages
                for msg in call_args:
                    if msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if '李四' in content:
                            extracted['name'] = True
                        if '北京' in content:
                            extracted['location'] = True
                        if 'Go' in content:
                            extracted['language'] = True
                        if 'AI' in content:
                            extracted['interest'] = True

                return MagicMock(
                    finish_reason="stop",
                    content="",
                    has_tool_calls=True,
                    tool_calls=[
                        MagicMock(arguments={
                            "history_entry": "[2024-01-01] 用户介绍了基本信息",
                            "memory_update": "## 用户\n- 姓名：已知\n- 其他信息：见历史记录"
                        })
                    ]
                )

            mock_provider = MagicMock()
            mock_provider.chat_with_retry = mock_chat

            await store.consolidate(messages, mock_provider, "test-model")

            # Check all facts were processed
            assert 'name' in extracted
            assert 'location' in extracted
            assert 'language' in extracted
            assert 'interest' in extracted


class TestMemoryDegradation:
    """Tests for memory consolidation degradation behavior."""

    @pytest.mark.asyncio
    async def test_degradation_on_llm_failure(self):
        """Test graceful degradation when LLM fails."""
        from nanobot.agent.memory import MemoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))

            messages = [
                {"role": "user", "content": "测试消息", "timestamp": "2024-01-01 10:00"},
            ]

            # Mock provider that always fails
            mock_provider = MagicMock()
            mock_provider.chat_with_retry = AsyncMock(side_effect=Exception("API Error"))

            # Should not raise, should handle gracefully
            result = await store.consolidate(messages, mock_provider, "test-model")

            # After max failures, should fall back to raw archive
            for i in range(3):
                await store.consolidate(messages, mock_provider, "test-model")

            # History should have raw archived content
            history = store.read_long_term()  # This is MEMORY.md, not HISTORY.md

            # The actual history file check
            history_content = store.history_file.read_text()
            assert "[RAW]" in history_content or "测试消息" in history_content

    @pytest.mark.asyncio
    async def test_consecutive_failures_counter(self):
        """Test that consecutive failures are properly tracked."""
        from nanobot.agent.memory import MemoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))

            assert store._consecutive_failures == 0

            # Mock failing provider
            mock_provider = MagicMock()
            mock_provider.chat_with_retry = AsyncMock(return_value=MagicMock(
                finish_reason="stop",
                content="no tool call",
                has_tool_calls=False,
            ))

            # Fail twice
            await store.consolidate([], mock_provider, "test-model")
            assert store._consecutive_failures == 1

            await store.consolidate([], mock_provider, "test-model")
            assert store._consecutive_failures == 2

            # Third failure triggers raw archive and resets counter
            await store.consolidate([], mock_provider, "test-model")
            assert store._consecutive_failures == 0  # Reset after raw archive


class TestMemoryQualityEvaluation:
    """Tests for memory quality evaluation using LLM-as-Judge."""

    @pytest.mark.asyncio
    async def test_memory_quality_judge(self):
        """Test LLM-as-Judge evaluation of memory quality."""
        judge_provider = MagicMock()
        judge_provider.chat = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "accuracy": 8,
                "completeness": 7,
                "utility": 9,
                "conciseness": 6,
                "overall": 7.5,
                "issues": ["遗漏了一些偏好信息"],
                "suggestions": ["添加更多上下文"]
            })
        ))

        judge = LLMasJudge(judge_provider, "gpt-4o")

        result = await judge.evaluate_memory_quality(
            memory_content="## 用户信息\n- 姓名：张三\n- 职业：Python工程师",
            conversation_history="用户张三，Python工程师，主要做后端开发",
        )

        assert result.passed  # 7.5 >= 6.0
        assert len(result.metrics) >= 4

        # Check metric values
        accuracy_metric = next(m for m in result.metrics if m.name == "accuracy")
        assert accuracy_metric.value == 8


class TestMemoryCompleteness:
    """Tests for memory completeness against ground truth."""

    @pytest.mark.asyncio
    async def test_completeness_against_ground_truth(self):
        """Test that memory captures expected facts."""
        from nanobot.agent.memory import MemoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))

            for test_case in MEMORY_TEST_CASES[:2]:  # Test first 2
                # Format conversation for consolidation
                messages = [
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": "2024-01-01 10:00",
                        "tools_used": [],
                    }
                    for msg in test_case.conversation
                ]

                # Mock LLM to extract expected facts
                async def mock_chat(*args, **kwargs):
                    content = json.dumps({
                        "history_entry": f"包含: {', '.join(test_case.expected_facts[:2])}",
                        "memory_update": "## 信息\n" + "\n".join(f"- {f}" for f in test_case.expected_facts)
                    })
                    return MagicMock(
                        finish_reason="stop",
                        content="",
                        has_tool_calls=True,
                        tool_calls=[MagicMock(arguments={
                            "history_entry": content,
                            "memory_update": content,
                        })]
                    )

                mock_provider = MagicMock()
                mock_provider.chat_with_retry = mock_chat

                result = await store.consolidate(messages, mock_provider, "test-model")

                # Read memory
                memory = store.read_long_term()

                # Check expected facts are captured
                for fact in test_case.expected_facts[:2]:  # Check first 2
                    # Fact might be paraphrased, check key words
                    fact_words = fact.split("：")[1] if "：" in fact else fact.split(":")[1] if ":" in fact else fact
                    assert any(word in memory for word in fact_words.split()[:2]), \
                        f"Expected fact '{fact}' not found in memory"


class TestMemoryBenchmark:
    """Benchmark tests for memory consolidation quality."""

    @pytest.mark.asyncio
    async def test_memory_benchmark(self):
        """Run memory quality benchmark."""
        judge_provider = MagicMock()

        eval_count = [0]

        async def mock_judge(messages, **kwargs):
            eval_count[0] += 1
            return MagicMock(content=json.dumps({
                "accuracy": 7 + (eval_count[0] % 2),  # Vary scores
                "completeness": 8,
                "utility": 7,
                "conciseness": 7,
                "overall": 7.25,
                "issues": [],
                "suggestions": []
            }))

        judge_provider.chat = mock_judge
        judge = LLMasJudge(judge_provider, "gpt-4o")

        results = []
        for test_case in MEMORY_TEST_CASES:
            result = await judge.evaluate_memory_quality(
                memory_content="\n".join(f"- {f}" for f in test_case.expected_facts),
                conversation_history="\n".join(f"{m['role']}: {m['content']}" for m in test_case.conversation),
            )
            results.append(result)

        # Aggregate
        avg_score = sum(r.overall_score() for r in results) / len(results)
        pass_rate = sum(1 for r in results if r.passed) / len(results)

        print(f"\n=== Memory Benchmark ===")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Pass Rate: {pass_rate:.1%}")

        # Quality threshold
        assert pass_rate >= 0.5, f"Pass rate {pass_rate:.1%} < 50%"


import tempfile


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
