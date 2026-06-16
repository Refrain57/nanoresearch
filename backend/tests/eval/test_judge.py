"""Tests for LLMJudge — score parsing, calibration, prompt building."""

from __future__ import annotations

import pytest

from nanobot.eval.judge import LLMJudge, CalibrationResult
from nanobot.eval.snapshot import RunSnapshotData
from nanobot.eval.judge import _parse_scores, _build_prompt


class _DummyProvider:
    def __init__(self, responses=None):
        self._responses = list(responses) if responses else []

    async def chat_with_retry(self, messages, model=None, **_):
        if self._responses:
            return self._responses.pop(0)
        return type("R", (), {"content": '{"scores": {"task_completion": 5}, "reasoning": "ok"}'})()

    def get_default_model(self):
        return "test-model"


def _make_snapshot(**overrides) -> RunSnapshotData:
    defaults = dict(
        run_id="r1",
        user_input="test input",
        tool_call_chain=[{"name": "search", "params": {"q": "test"}, "result": "found x", "order": 1, "duration_ms": 100, "error": False}],
        llm_calls=[{"input_tokens": 50, "output_tokens": 100, "model": "m"}],
        final_response="here is the result",
        run_status="success",
        total_input_tokens=50,
        total_output_tokens=100,
        ttft_ms=None,
        total_duration_ms=500,
        tool_call_count=1,
        llm_call_count=1,
        retry_count=0,
    )
    defaults.update(overrides)
    return RunSnapshotData(**defaults)


# ---------------------------------------------------------------------------
# _parse_scores unit tests
# ---------------------------------------------------------------------------


class TestParseScores:
    def test_parse_valid_json(self):
        raw = '{"scores": {"tool_rationality": 4, "task_completion": 5}, "reasoning": "good"}'
        result = _parse_scores(raw)
        assert result == {"tool_rationality": 0.75, "task_completion": 1.0}

    def test_parse_normalizes_to_zero_one(self):
        raw = '{"scores": {"d1": 1, "d2": 3, "d3": 5}}'
        result = _parse_scores(raw)
        assert result == {"d1": 0.0, "d2": 0.5, "d3": 1.0}

    def test_parse_handles_markdown_fence(self):
        raw = '```json\n{"scores": {"task_completion": 4}}\n```'
        result = _parse_scores(raw)
        assert result == {"task_completion": 0.75}

    def test_parse_returns_empty_on_invalid_json(self):
        assert _parse_scores("not json") == {}

    def test_parse_returns_empty_on_empty(self):
        assert _parse_scores("") == {}

    def test_parse_ignores_out_of_range(self):
        raw = '{"scores": {"d1": 0, "d2": 6, "d3": "bad"}}'
        result = _parse_scores(raw)
        assert result == {}

    def test_parse_includes_hallucination(self):
        raw = '{"scores": {"task_completion": 4, "hallucination": 5}}'
        result = _parse_scores(raw)
        assert "hallucination" in result
        assert result["hallucination"] == 1.0


# ---------------------------------------------------------------------------
# _build_prompt unit tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_basic_prompt_has_sections(self,):
        data = _make_snapshot()
        prompt = _build_prompt(data, None, None)
        assert "## 用户输入" in prompt
        assert "## 工具调用链" in prompt
        assert "## Agent 最终回复" in prompt
        assert "## 评分要求" in prompt
        assert "test input" in prompt
        assert "here is the result" in prompt

    def test_prompt_with_session_history(self,):
        data = _make_snapshot()
        history = [{"role": "user", "content": "previous question"}]
        prompt = _build_prompt(data, None, history)
        assert "## 历史对话" in prompt
        assert "previous question" in prompt
        assert "multi_turn" in prompt.lower() or "连贯" in prompt

    def test_prompt_with_expected_keywords(self,):
        data = _make_snapshot()
        tc = type("TC", (), {"expected_keywords": ["result", "data"]})()
        prompt = _build_prompt(data, tc, None)
        assert "期望关键词" in prompt
        assert "result" in prompt

    def test_prompt_truncates_long_chains(self,):
        long_chain = [{"name": "t", "params": {}, "result": "x" * 1000, "order": i, "duration_ms": 1, "error": False} for i in range(10)]
        data = _make_snapshot(tool_call_chain=long_chain)
        prompt = _build_prompt(data, None, None)
        assert len(prompt) < 10000

    def test_prompt_empty_tool_call_chain(self,):
        data = _make_snapshot(tool_call_chain=[])
        prompt = _build_prompt(data, None, None)
        assert "（无工具调用）" in prompt


# ---------------------------------------------------------------------------
# LLMJudge integration tests
# ---------------------------------------------------------------------------


class TestLLMJudge:
    def test_model_fallback(self):
        class _NoDefaultProvider:
            def get_default_model(self):
                return "fallback-model"

        judge = LLMJudge(_NoDefaultProvider())
        assert judge._model == "fallback-model"

    def test_model_explicit(self):
        class _P:
            def get_default_model(self): return "default"
        judge = LLMJudge(_P(), model="explicit")
        assert judge._model == "explicit"

    @pytest.mark.asyncio
    async def test_score_returns_empty_on_provider_error(self):
        class _FailingProvider:
            def get_default_model(self): return "m"
            async def chat_with_retry(self, **kw): raise RuntimeError("fail")

        data = _make_snapshot()
        judge = LLMJudge(_FailingProvider())
        scores, raw = await judge.score(data)
        assert scores == {}
        assert raw == ""

    @pytest.mark.asyncio
    async def test_calibrate_returns_passed_when_no_human_scores(self):
        class _P:
            def get_default_model(self): return "m"
            async def chat_with_retry(self, **kw):
                return type("R", (), {"content": '{"scores": {"task_completion": 4}}'})()

        data = _make_snapshot()
        tc = type("TC", (), {"human_score": None})()
        judge = LLMJudge(_P())
        result = await judge.calibrate([(data, tc)])
        assert result.passed is True
        assert result.mad == 0.0
        assert result.sample_count == 0
