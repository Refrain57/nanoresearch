"""Tests for workspace file DELETE endpoint + traversal guard."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.server.routers.workspace_router import router, _safe_resolve


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.loop_config = {"base_workspace": str(tmp_path)}
    app.dependency_overrides[get_current_user] = lambda: "u1"
    ws = tmp_path / "users" / "u1"
    ws.mkdir(parents=True)
    return TestClient(app), ws


def test_delete_file(client):
    c, ws = client
    (ws / "note.txt").write_text("hi", encoding="utf-8")
    resp = c.delete("/api/workspace/files/note.txt")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "note.txt"}
    assert not (ws / "note.txt").exists()


def test_delete_directory_recursive(client):
    c, ws = client
    d = ws / "sub"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")
    resp = c.delete("/api/workspace/files/sub")
    assert resp.status_code == 200
    assert not d.exists()


def test_delete_system_file_forbidden(client):
    c, ws = client
    (ws / "SOUL.md").write_text("soul", encoding="utf-8")
    resp = c.delete("/api/workspace/files/SOUL.md")
    assert resp.status_code == 403
    assert (ws / "SOUL.md").exists()


def test_delete_missing_file(client):
    c, ws = client
    resp = c.delete("/api/workspace/files/nope.txt")
    assert resp.status_code == 404


def test_safe_resolve_blocks_traversal(tmp_path):
    # 直接测越界守卫，避开 HTTP 客户端对 URL 中 ".." 的规范化
    with pytest.raises(HTTPException) as ei:
        _safe_resolve(tmp_path, "../../secret.txt")
    assert ei.value.status_code == 403
