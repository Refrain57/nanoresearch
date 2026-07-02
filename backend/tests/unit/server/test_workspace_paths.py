"""Unit tests for shared workspace path helpers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from nanoresearch.server.routers.workspace_paths import safe_resolve, user_workspace


def _fake_request(base: str | None):
    state = SimpleNamespace(loop_config={"base_workspace": base} if base else {})
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_user_workspace_builds_per_user_path(tmp_path):
    req = _fake_request(str(tmp_path))
    assert user_workspace(req, "alice") == tmp_path / "users" / "alice"


def test_safe_resolve_allows_inside(tmp_path):
    assert safe_resolve(tmp_path, "skills/foo") == (tmp_path / "skills" / "foo").resolve()


def test_safe_resolve_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException) as exc:
        safe_resolve(tmp_path, "../../etc/passwd")
    assert exc.value.status_code == 403
