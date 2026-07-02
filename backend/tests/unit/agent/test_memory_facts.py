from nanoresearch.agent.memory_facts import (
    Fact, ProfileDiff, normalize, parse_memory_md, render_memory_md, compute_profile_diff,
)


def test_normalize_strips_bullet_and_casefolds():
    assert normalize("-  Prefers  Python ") == normalize("prefers python")


def test_parse_memory_md_sections_and_bullets():
    md = "# User Memory\n\n## FACTS\n- 偏好 Python\n- 用 Git\n\n## USER_PROFILE\n资深工程师。\n\n## FOCUS_AREAS\n- RAG\n"
    got = parse_memory_md(md)
    assert ("facts", "偏好 Python") in got
    assert ("facts", "用 Git") in got
    assert ("user_profile", "资深工程师。") in got
    assert ("focus_areas", "RAG") in got


def test_render_roundtrips_active_only():
    facts = [
        Fact(text="偏好 Python", section="facts"),
        Fact(text="过时", section="facts", active=False),
        Fact(text="RAG", section="focus_areas"),
    ]
    out = render_memory_md(facts)
    assert "偏好 Python" in out and "RAG" in out
    assert "过时" not in out
    assert "## FACTS" in out and "## FOCUS_AREAS" in out


def test_compute_diff_adds_new_lines():
    cur = [Fact(id="1", text="偏好 Python", section="facts", source="extracted")]
    new = [("facts", "偏好 Python"), ("facts", "喜欢 TDD")]
    diff = compute_profile_diff(cur, new)
    assert ("facts", "喜欢 TDD") in diff.add
    assert diff.remove_ids == []


def test_compute_diff_removes_absent_extracted():
    cur = [Fact(id="1", text="旧偏好", section="facts", source="extracted")]
    diff = compute_profile_diff(cur, [("facts", "新偏好")])
    assert diff.remove_ids == ["1"]
    assert ("facts", "新偏好") in diff.add


def test_compute_diff_never_removes_manual():
    cur = [Fact(id="m1", text="人工写的", section="facts", source="manual")]
    diff = compute_profile_diff(cur, [("facts", "别的")])  # manual absent from new
    assert "m1" not in diff.remove_ids


def test_compute_diff_dedups_new_and_existing():
    cur = [Fact(id="1", text="偏好 Python", section="facts", source="extracted")]
    new = [("facts", "偏好 Python"), ("facts", "偏好  python")]  # dup of existing + self-dup
    diff = compute_profile_diff(cur, new)
    assert diff.add == []
