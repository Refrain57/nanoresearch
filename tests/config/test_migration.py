"""Tests for nanoresearch.config.migration — migrate_llm_keys()."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_yaml():
    """Return a temporary settings.yaml path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.close()
    yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def config_json():
    """Return a temporary config.json path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    f.close()
    yield Path(f.name)
    os.unlink(f.name)


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMigrateLlmKeys:
    """migrate_llm_keys() copies settings.yaml api_keys into config.json providers."""

    def test_migrates_llm_key(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "dashscope", "api_key": "sk-ds-123", "model": "qwen-plus"},
        })
        _write_json(config_json, {
            "providers": {"dashscope": {"apiKey": ""}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 1
        assert report["migrated"][0]["section"] == "llm"
        assert report["migrated"][0]["provider"] == "dashscope"

        cfg = _read_json(config_json)
        assert cfg["providers"]["dashscope"]["apiKey"] == "sk-ds-123"

    def test_migrates_embedding_key(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "embedding": {"provider": "openai", "api_key": "sk-emb-456", "model": "text-embedding-3-small"},
        })
        _write_json(config_json, {
            "providers": {"openai": {"apiKey": ""}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 1
        cfg = _read_json(config_json)
        assert cfg["providers"]["openai"]["apiKey"] == "sk-emb-456"

    def test_migrates_vision_key(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "vision_llm": {"enabled": True, "provider": "dashscope", "model": "qwen-vl-max", "api_key": "sk-vl-789"},
        })
        _write_json(config_json, {
            "providers": {"dashscope": {"apiKey": ""}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 1
        cfg = _read_json(config_json)
        assert cfg["providers"]["dashscope"]["apiKey"] == "sk-vl-789"

    def test_skips_when_config_already_has_key(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "dashscope", "api_key": "sk-ds-old"},
        })
        _write_json(config_json, {
            "providers": {"dashscope": {"apiKey": "sk-ds-new"}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 0
        assert len(report["skipped_already_exists"]) == 1
        # config.json value unchanged
        cfg = _read_json(config_json)
        assert cfg["providers"]["dashscope"]["apiKey"] == "sk-ds-new"

    def test_skips_when_no_matching_provider_in_config(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "unknown_provider", "api_key": "sk-xxx"},
        })
        _write_json(config_json, {
            "providers": {"dashscope": {"apiKey": ""}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 0
        assert len(report["skipped_no_provider"]) == 1
        assert report["skipped_no_provider"][0]["provider"] == "unknown_provider"

    def test_skips_when_section_has_no_key(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "dashscope", "model": "qwen-plus"},
        })
        _write_json(config_json, {
            "providers": {"dashscope": {"apiKey": ""}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 0

    def test_dry_run_does_not_write(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "dashscope", "api_key": "sk-ds-dry"},
        })
        _write_json(config_json, {
            "providers": {"dashscope": {"apiKey": ""}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=True,
        )

        assert len(report["migrated"]) == 1
        # config.json should NOT be updated
        cfg = _read_json(config_json)
        assert cfg["providers"]["dashscope"]["apiKey"] == ""

    def test_missing_settings_yaml_returns_empty_report(self, config_json):
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path="/nonexistent/settings.yaml",
            config_path=config_json,
        )

        assert len(report["migrated"]) == 0

    def test_missing_config_json_returns_empty_report(self, settings_yaml):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "dashscope", "api_key": "sk-ds"},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path="/nonexistent/config.json",
        )

        assert len(report["migrated"]) == 0

    def test_empty_key_not_copied(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "dashscope", "api_key": ""},
        })
        _write_json(config_json, {
            "providers": {"dashscope": {"apiKey": ""}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 0

    def test_normalises_openai_compat_to_custom(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "openai-compat", "api_key": "sk-custom"},
        })
        _write_json(config_json, {
            "providers": {"custom": {"apiKey": ""}},
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 1
        assert report["migrated"][0]["provider"] == "custom"
        cfg = _read_json(config_json)
        assert cfg["providers"]["custom"]["apiKey"] == "sk-custom"

    def test_migrates_all_three_sections(self, settings_yaml, config_json):
        _write_yaml(settings_yaml, {
            "llm": {"provider": "dashscope", "api_key": "sk-llm"},
            "embedding": {"provider": "openai", "api_key": "sk-emb"},
            "vision_llm": {"enabled": True, "provider": "siliconflow", "model": "deepseek-vl", "api_key": "sk-vl"},
        })
        _write_json(config_json, {
            "providers": {
                "dashscope": {"apiKey": ""},
                "openai": {"apiKey": ""},
                "siliconflow": {"apiKey": ""},
            },
        })
        from nanoresearch.config.migration import migrate_llm_keys

        report = migrate_llm_keys(
            settings_path=settings_yaml,
            config_path=config_json,
            dry_run=False,
        )

        assert len(report["migrated"]) == 3
        cfg = _read_json(config_json)
        assert cfg["providers"]["dashscope"]["apiKey"] == "sk-llm"
        assert cfg["providers"]["openai"]["apiKey"] == "sk-emb"
        assert cfg["providers"]["siliconflow"]["apiKey"] == "sk-vl"