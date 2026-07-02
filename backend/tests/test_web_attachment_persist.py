"""Web-bridge persistence attaches workspace-relative media descriptors.

build_sent_attachment_messages(contents, medias, workspace_root) builds the
assistant message dicts folded into the saved turn; media descriptors are
attached only when present and within the workspace.
"""
from pathlib import Path

from nanoresearch.agent.loop import build_sent_attachment_messages


def test_build_messages_adds_media_when_present(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    (ws / "r.md").write_text("hi", encoding="utf-8")
    out = build_sent_attachment_messages(
        ["see file", "plain"],
        [[str(ws / "r.md")], []],
        ws,
    )
    assert out == [
        {"role": "assistant", "content": "see file",
         "media": [{"path": "r.md", "name": "r.md", "size": 2}]},
        {"role": "assistant", "content": "plain"},
    ]


def test_out_of_workspace_media_omitted(tmp_path):
    ws = tmp_path / "users" / "alice"
    ws.mkdir(parents=True)
    outside = tmp_path / "x.txt"
    outside.write_text("x", encoding="utf-8")
    out = build_sent_attachment_messages(["hi"], [[str(outside)]], ws)
    assert out == [{"role": "assistant", "content": "hi"}]  # media dropped -> key omitted
