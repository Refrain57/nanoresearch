"""Tests for ModelFactory mode-aware resolution."""

import pytest

from nanoresearch.providers.model_factory import (
    ModelFactory,
    ModelResolutionError,
    ModelRole,
)


def _user_providers(api_key: str = "sk-user", model: str | None = None):
    return [{
        "id": "p1",
        "name": "deepseek",
        "api_key": api_key,
        "api_base": "https://api.deepseek.com",
        "models": [model] if model else [],
    }]


def test_server_mode_uses_user_provider_when_present(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=_user_providers("sk-user", "deepseek-chat"),
        user_model="deepseek-chat",
    )
    assert spec.api_key == "sk-user"
    assert spec.model == "deepseek-chat"


def test_server_mode_raises_when_user_provider_missing(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    with pytest.raises(ModelResolutionError) as exc:
        ModelFactory.resolve(
            ModelRole.CHAT,
            user_providers=[],
            user_model="gpt-4o",
            mode="server",
        )
    assert exc.value.missing_role == "chat"


def test_local_mode_unchanged_behavior(monkeypatch):
    """local mode 维持现状：无 user_providers + 无 config + 无 settings → 返回空 spec（旧行为）。"""
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    spec = ModelFactory.resolve(
        ModelRole.CHAT,
        user_providers=[],
        user_model="gpt-4o",
        mode="local",
    )
    # 旧路径返回 ModelSpec(model="gpt-4o") 但无 api_key
    assert spec.model == "gpt-4o"
    assert spec.api_key is None


def test_mode_param_overrides_env(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    with pytest.raises(ModelResolutionError):
        ModelFactory.resolve(
            ModelRole.CHAT,
            user_providers=[],
            user_model="gpt-4o",
            mode="server",  # 显式 server 覆盖 env
        )


def test_missing_role_field_on_error():
    err = ModelResolutionError("missing key", missing_role="embedding")
    assert err.missing_role == "embedding"
