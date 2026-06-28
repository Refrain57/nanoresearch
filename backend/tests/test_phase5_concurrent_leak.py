"""Verify ModelFactory + provider construction doesn't leak across uids."""

import asyncio
import os
import pytest

from nanoresearch.providers.model_factory import ModelFactory, ModelRole
from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider
from nanoresearch.providers.registry import find_by_name


def test_two_uids_get_distinct_api_keys(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    user_a_providers = [{
        "id": "a", "name": "deepseek", "api_key": "sk-USER-A",
        "api_base": None, "models": ["deepseek-chat"],
    }]
    user_b_providers = [{
        "id": "b", "name": "deepseek", "api_key": "sk-USER-B",
        "api_base": None, "models": ["deepseek-chat"],
    }]

    spec_a = ModelFactory.resolve(
        ModelRole.CHAT, user_providers=user_a_providers,
        user_model="deepseek-chat", mode="server",
    )
    spec_b = ModelFactory.resolve(
        ModelRole.CHAT, user_providers=user_b_providers,
        user_model="deepseek-chat", mode="server",
    )
    assert spec_a.api_key == "sk-USER-A"
    assert spec_b.api_key == "sk-USER-B"

    # 构造两个 provider 实例，确认 os.environ 不被任何一方污染
    ps = find_by_name("deepseek")
    OpenAICompatProvider(api_key=spec_a.api_key, spec=ps, default_model="deepseek-chat")
    OpenAICompatProvider(api_key=spec_b.api_key, spec=ps, default_model="deepseek-chat")
    assert os.environ.get("DEEPSEEK_API_KEY") is None
