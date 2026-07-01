"""Structured 画像 (profile) facts + incremental-diff projection (P1 of memory-layering).

Profile = provenance-tagged facts (source: extracted|manual). Consolidation applies an
incremental diff (never wholesale overwrite); manual facts are never auto-removed.
MEMORY.md is a one-way rendered projection of the active facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SECTIONS = ("facts", "user_profile", "focus_areas")
_SECTION_TITLES = {"facts": "FACTS", "user_profile": "USER_PROFILE", "focus_areas": "FOCUS_AREAS"}
_TITLE_TO_SECTION = {
    "FACTS": "facts",
    "USER_PROFILE": "user_profile", "USER PROFILE": "user_profile",
    "FOCUS_AREAS": "focus_areas", "FOCUS AREAS": "focus_areas",
}


@dataclass
class Fact:
    text: str
    section: str
    source: str = "extracted"          # extracted | manual
    id: str | None = None
    uid: str | None = None
    derived_from: list[str] = field(default_factory=list)
    confidence: float | None = None
    edited_by: str | None = None
    edited_at: str | None = None
    active: bool = True


@dataclass
class ProfileDiff:
    add: list[tuple[str, str]] = field(default_factory=list)   # (section, text)
    remove_ids: list[str] = field(default_factory=list)        # Fact.id to deactivate


def normalize(text: str) -> str:
    t = text.strip().lstrip("-*").strip()
    t = re.sub(r"\s+", " ", t)
    return t.casefold()


def parse_memory_md(md_text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    section: str | None = None
    for raw in (md_text or "").splitlines():
        line = raw.rstrip()
        h = re.match(r"^#{1,6}\s*(.+?)\s*$", line)
        if h:
            section = _TITLE_TO_SECTION.get(h.group(1).strip().upper())
            continue
        if section is None:
            continue
        b = re.match(r"^\s*[-*]\s+(.+?)\s*$", line)
        text = b.group(1).strip() if b else line.strip()
        if text:
            out.append((section, text))
    return out


def render_memory_md(facts: list[Fact]) -> str:
    active = [f for f in facts if f.active]
    lines = ["# User Memory", ""]
    for sec in SECTIONS:
        sec_facts = [f for f in active if f.section == sec]
        if not sec_facts:
            continue
        lines.append(f"## {_SECTION_TITLES[sec]}")
        lines.extend(f"- {f.text}" for f in sec_facts)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compute_profile_diff(current: list[Fact], new_lines: list[tuple[str, str]]) -> ProfileDiff:
    active = [f for f in current if f.active]
    cur_keys = {(f.section, normalize(f.text)): f for f in active}
    new_keys = {(sec, normalize(txt)) for sec, txt in new_lines}
    diff = ProfileDiff()
    seen: set[tuple[str, str]] = set()
    for sec, txt in new_lines:
        key = (sec, normalize(txt))
        if key in seen:
            continue
        seen.add(key)
        if key not in cur_keys:
            diff.add.append((sec, txt))
    for key, fact in cur_keys.items():
        if key not in new_keys and fact.source == "extracted" and fact.id:
            diff.remove_ids.append(fact.id)
    return diff
