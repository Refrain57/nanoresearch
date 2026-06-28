"""Tests for nanoresearch.utils.env_compat."""

import os
import warnings

import pytest

from nanoresearch.utils.env_compat import apply_legacy_env_compat, _reset_for_tests


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset compat state and clear test-relevant env vars between tests."""
    _reset_for_tests()
    for key in list(os.environ):
        if key.startswith("NANOBOT_") or key.startswith("NANORESEARCH_"):
            monkeypatch.delenv(key, raising=False)
    yield
    _reset_for_tests()


def test_copies_legacy_to_new_with_warning(monkeypatch):
    monkeypatch.setenv("NANOBOT_FOO", "value-foo")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        copied = apply_legacy_env_compat()
    assert ("NANOBOT_FOO", "NANORESEARCH_FOO") in copied
    assert os.environ["NANORESEARCH_FOO"] == "value-foo"
    assert any(
        issubclass(item.category, DeprecationWarning)
        and "NANOBOT_FOO" in str(item.message)
        and "NANORESEARCH_FOO" in str(item.message)
        for item in w
    )


def test_respects_explicit_new_name(monkeypatch):
    """If user sets BOTH, new name wins and old name still warns."""
    monkeypatch.setenv("NANOBOT_FOO", "old-value")
    monkeypatch.setenv("NANORESEARCH_FOO", "new-value")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        copied = apply_legacy_env_compat()
    assert copied == []  # nothing copied — new name already set
    assert os.environ["NANORESEARCH_FOO"] == "new-value"  # untouched
    assert any(
        issubclass(item.category, DeprecationWarning)
        and "NANOBOT_FOO is set alongside" in str(item.message)
        for item in w
    )


def test_idempotent(monkeypatch):
    monkeypatch.setenv("NANOBOT_FOO", "v")
    first = apply_legacy_env_compat()
    second = apply_legacy_env_compat()
    assert first == [("NANOBOT_FOO", "NANORESEARCH_FOO")]
    assert second == []  # second call is no-op


def test_ignores_unrelated_env_vars(monkeypatch):
    monkeypatch.setenv("PATH", "/some/path")
    monkeypatch.setenv("HOME", "/home/user")
    copied = apply_legacy_env_compat()
    assert copied == []


def test_handles_all_five_legacy_vars(monkeypatch):
    """The 5 documented NANOBOT_ vars all roundtrip correctly."""
    legacy_vars = {
        "NANOBOT_MAX_CONCURRENT_REQUESTS": "5",
        "NANOBOT_PYTHON": "/usr/bin/python3",
        "NANOBOT_WORKSPACE": "/tmp/ws",
        "NANOBOT_TMUX_SOCKET_DIR": "/tmp/sockets",
        "NANOBOT_FOO_PYDANTIC_FIELD": "bar",  # represents any Pydantic env_prefix var
    }
    for k, v in legacy_vars.items():
        monkeypatch.setenv(k, v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        apply_legacy_env_compat()
    for old, val in legacy_vars.items():
        new = "NANORESEARCH_" + old[len("NANOBOT_"):]
        assert os.environ[new] == val, f"{old} → {new} failed"
