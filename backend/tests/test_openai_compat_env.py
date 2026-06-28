"""Verify OpenAICompatProvider construction does not pollute os.environ."""

import os

from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider
from nanoresearch.providers.registry import find_by_name


def _snapshot_env(keys: list[str]) -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in keys}


def test_construct_gateway_provider_no_env_write(monkeypatch):
    """Gateway provider (openrouter) used to do os.environ[env_key] = api_key — must not."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    spec = find_by_name("openrouter")
    assert spec is not None

    OpenAICompatProvider(
        api_key="sk-or-test-1",
        api_base="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-3.5-sonnet",
        spec=spec,
    )
    assert os.environ.get("OPENROUTER_API_KEY") is None


def test_construct_standard_provider_no_env_write(monkeypatch):
    """Non-gateway (deepseek) used to setdefault — must not."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    spec = find_by_name("deepseek")
    assert spec is not None

    OpenAICompatProvider(
        api_key="sk-deepseek-test",
        api_base=None,
        default_model="deepseek-chat",
        spec=spec,
    )
    assert os.environ.get("DEEPSEEK_API_KEY") is None


def test_zhipu_env_extras_no_write(monkeypatch):
    """zhipu spec has env_extras (ZHIPUAI_API_KEY=...) — must not be written either."""
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    spec = find_by_name("zhipu")
    assert spec is not None

    OpenAICompatProvider(
        api_key="zai-test",
        api_base=None,
        default_model="glm-4",
        spec=spec,
    )
    assert os.environ.get("ZAI_API_KEY") is None
    assert os.environ.get("ZHIPUAI_API_KEY") is None
