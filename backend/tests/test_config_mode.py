"""Tests for NANORESEARCH_MODE switch and helpers."""

import pytest


def test_get_mode_defaults_to_local(monkeypatch):
    monkeypatch.delenv("NANORESEARCH_MODE", raising=False)
    from nanoresearch.config.loader import get_mode
    assert get_mode() == "local"


def test_get_mode_reads_server(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    from nanoresearch.config.loader import get_mode
    assert get_mode() == "server"


def test_get_mode_invalid_raises(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "weird")
    from nanoresearch.config.loader import get_mode
    with pytest.raises(ValueError, match="NANORESEARCH_MODE"):
        get_mode()


def test_env_key_fallback_allowed_local(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    from nanoresearch.config.loader import env_key_fallback_allowed
    assert env_key_fallback_allowed() is True


def test_env_key_fallback_allowed_server(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    from nanoresearch.config.loader import env_key_fallback_allowed
    assert env_key_fallback_allowed() is False
