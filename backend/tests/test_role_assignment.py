"""ModelFactory consults user_roles map when present."""

from __future__ import annotations

import pytest

from nanoresearch.providers.model_factory import (
    ModelFactory,
    ModelResolutionError,
    ModelRole,
)


PROVIDERS = [
    {"id": "u1", "name": "我的 DeepSeek", "provider": "deepseek",
     "api_key": "sk-1", "api_base": None, "models": ["deepseek-chat"]},
    {"id": "u2", "name": "通义", "provider": "dashscope",
     "api_key": "sk-2", "api_base": "https://x", "models": ["qwen-plus", "text-embedding-v3"]},
]


def test_chat_role_by_id_resolves_to_explicit_provider(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    roles = {"chat": {"provider_id": "u2", "model": "qwen-plus"}}
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=PROVIDERS,
        user_roles=roles,
    )
    assert spec.api_key == "sk-2"
    assert spec.model == "qwen-plus"
    assert spec.base_url == "https://x"


def test_role_provider_id_unknown_falls_back_to_existing_logic(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    roles = {"chat": {"provider_id": "ghost", "model": "deepseek-chat"}}
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=PROVIDERS,
        user_roles=roles,
    )
    # Falls back to _match_user_provider_by_model with model="deepseek-chat" → u1
    assert spec.api_key == "sk-1"


def test_role_none_falls_through_to_existing_resolver(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    roles = {"chat": None}
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=PROVIDERS,
        user_roles=roles,
        user_model="qwen-plus",  # Drives model-match fallback to u2
    )
    assert spec.api_key == "sk-2"


def test_embedding_role_by_id(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    roles = {"embedding": {"provider_id": "u2", "model": "text-embedding-v3"}}

    class _Emb:
        provider = "dashscope"
        model = "fallback-model"
        api_key = None
        base_url = None

    class _Settings:
        embedding = _Emb()
        llm = None
        vision_llm = None

    spec = ModelFactory.resolve(
        ModelRole.EMBEDDING,
        user_providers=PROVIDERS,
        user_roles=roles,
        rag_settings=_Settings(),
    )
    assert spec.api_key == "sk-2"
    assert spec.model == "text-embedding-v3"


def test_server_mode_with_roles_uses_assigned_provider(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    roles = {"chat": {"provider_id": "u1", "model": "deepseek-chat"}}
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=PROVIDERS,
        user_roles=roles,
    )
    assert spec.api_key == "sk-1"


def test_server_mode_no_roles_no_match_raises_missing_provider(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    with pytest.raises(ModelResolutionError) as exc:
        ModelFactory.resolve(
            ModelRole.CHAT,
            user_providers=[],
            user_roles={"chat": None},
        )
    assert exc.value.missing_role == "chat"


def test_match_user_provider_by_id_helper():
    assert ModelFactory._match_user_provider_by_id("u1", PROVIDERS)["api_key"] == "sk-1"
    assert ModelFactory._match_user_provider_by_id("ghost", PROVIDERS) is None
    assert ModelFactory._match_user_provider_by_id("", PROVIDERS) is None
