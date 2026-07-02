# backend/tests/test_workspace_paths.py
from pathlib import Path
from nanoresearch.server.routers.workspace_paths import build_attachment_descriptors


def test_in_workspace_file_maps_to_descriptor(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    f = ws / "report.md"
    f.write_text("hello", encoding="utf-8")
    out = build_attachment_descriptors([str(f)], ws)
    assert out == [{"path": "report.md", "name": "report.md", "size": 5}]


def test_nested_subdir_path_preserved(tmp_path):
    ws = tmp_path / "users" / "alice"
    sub = ws / "sub" / "deep"
    sub.mkdir(parents=True)
    f = sub / "a.pdf"
    f.write_bytes(b"1234")
    out = build_attachment_descriptors([str(f)], ws)
    assert out == [{"path": "sub/deep/a.pdf", "name": "a.pdf", "size": 4}]


def test_outside_workspace_dropped(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    other = tmp_path / "users" / "bob" / "secret.txt"
    other.parent.mkdir(parents=True)
    other.write_text("x", encoding="utf-8")
    assert build_attachment_descriptors([str(other)], ws) == []


def test_missing_file_dropped(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    assert build_attachment_descriptors([str(ws / "nope.md")], ws) == []


def test_none_and_empty(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    assert build_attachment_descriptors(None, ws) == []
    assert build_attachment_descriptors([], ws) == []
