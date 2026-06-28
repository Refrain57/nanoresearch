"""Idempotent legacy-extra migration for the new provider preset + roles fields."""

from __future__ import annotations

from nanoresearch.storage.repositories.user_settings_repo import _migrate_legacy_extra


def test_legacy_extra_no_roles_gets_roles_inferred():
    extra = {
        "providers": [
            {"id": "u1", "name": "我的 DeepSeek", "api_key": "sk-1", "api_base": None, "models": ["deepseek-chat"]},
            {"id": "u2", "name": "通义 dashscope", "api_key": "sk-2", "api_base": "https://x", "models": ["qwen-plus", "text-embedding-v3"]},
        ]
    }
    out, changed = _migrate_legacy_extra(extra)
    assert changed is True
    assert out["providers"][0]["provider"] == "deepseek"
    assert out["providers"][1]["provider"] == "dashscope"
    assert out["roles"]["chat"]["provider_id"] == "u1"
    assert out["roles"]["ingestion_llm"]["provider_id"] == "u1"
    assert out["roles"]["embedding"]["provider_id"] == "u2"
    assert out["roles"]["vision"] is None
    assert out["roles"]["eval_generator"] is None
    assert out["roles"]["eval_evaluator"] is None


def test_legacy_extra_unknown_name_falls_back_to_openai_compatible():
    extra = {"providers": [{"id": "u1", "name": "MyCustom", "api_key": "sk", "api_base": "https://x", "models": []}]}
    out, changed = _migrate_legacy_extra(extra)
    assert changed is True
    assert out["providers"][0]["provider"] == "openai_compatible"


def test_migration_is_idempotent_when_roles_present():
    extra = {
        "providers": [{"id": "u1", "name": "X", "provider": "deepseek", "api_key": "sk", "api_base": None, "models": []}],
        "roles": {"chat": {"provider_id": "u1", "model": "deepseek-chat"}, "embedding": None},
    }
    out, changed = _migrate_legacy_extra(extra)
    assert changed is False
    assert out is extra  # no copy when no work


def test_migration_skips_when_no_providers():
    extra = {"fast_model": "x"}
    out, changed = _migrate_legacy_extra(extra)
    assert changed is False


def test_embedding_role_left_null_when_no_embedding_capable_provider():
    extra = {
        "providers": [
            {"id": "u1", "name": "DeepSeek", "api_key": "sk", "api_base": None, "models": []},
        ]
    }
    out, changed = _migrate_legacy_extra(extra)
    assert changed is True
    assert out["roles"]["chat"]["provider_id"] == "u1"
    assert out["roles"]["embedding"] is None


def test_chat_role_left_null_when_no_provider_has_api_key():
    extra = {
        "providers": [
            {"id": "u1", "name": "DeepSeek", "api_key": "", "api_base": None, "models": []},
        ]
    }
    out, changed = _migrate_legacy_extra(extra)
    assert changed is True
    assert out["roles"]["chat"] is None
    assert out["providers"][0]["provider"] == "deepseek"  # preset field still inferred
