"""Pydantic schema + merge helpers for the new roles map and provider preset field."""

from __future__ import annotations

import pytest

from nanoresearch.server.routers.settings_router import (
    ProviderIn,
    UserSettingsUpdate,
    _merge_providers,
    _to_dict,
)


def test_provider_in_accepts_provider_preset_field():
    p = ProviderIn(name="我的 DeepSeek", provider="deepseek", api_key="sk-x")
    assert p.provider == "deepseek"


def test_provider_in_provider_field_defaults_to_none():
    p = ProviderIn(name="legacy", api_key="sk-x")
    assert p.provider is None


def test_user_settings_update_accepts_roles_field():
    u = UserSettingsUpdate(roles={
        "chat": {"provider_id": "uuid-1", "model": "deepseek-chat"},
        "embedding": None,
    })
    assert u.roles["chat"]["provider_id"] == "uuid-1"
    assert u.roles["embedding"] is None


def test_merge_providers_preserves_provider_preset_field():
    existing = [{"id": "u1", "name": "X", "provider": "deepseek", "api_key": "sk-old", "api_base": None, "models": []}]
    incoming = [ProviderIn(id="u1", name="X", provider="deepseek", api_key=None, models=[])]
    merged = _merge_providers(existing, incoming)
    assert merged[0]["provider"] == "deepseek"
    assert merged[0]["api_key"] == "sk-old"  # None means keep


def test_merge_providers_accepts_new_provider_field_on_new_row():
    incoming = [ProviderIn(name="New", provider="openai", api_key="sk-new", models=[])]
    merged = _merge_providers([], incoming)
    assert merged[0]["provider"] == "openai"
    assert merged[0]["api_key"] == "sk-new"


def test_to_dict_surfaces_roles_and_provider_field():
    class Row:
        model = None
        max_iterations = None
        extra = {
            "providers": [
                {"id": "u1", "name": "X", "provider": "deepseek", "api_key": "sk", "api_base": None, "models": []}
            ],
            "roles": {
                "chat": {"provider_id": "u1", "model": "deepseek-chat"},
                "embedding": None,
            },
        }

    out = _to_dict(Row(), defaults={})
    assert out["roles"]["chat"]["provider_id"] == "u1"
    assert out["roles"]["embedding"] is None
    assert out["providers"][0]["provider"] == "deepseek"
    assert out["providers"][0]["api_key_set"] is True
    assert "api_key" not in out["providers"][0]  # masked
