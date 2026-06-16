"""Tests for YAML frontmatter parsing in MarkdownLoader."""

import tempfile
from pathlib import Path

from nanobot.rag.libs.loader.markdown_loader import MarkdownLoader, _parse_frontmatter


def test_parse_frontmatter_normal():
    """Normal frontmatter is parsed correctly."""
    text = "---\nsource: research\ntitle: Test Report\nquality_score: 8.5\n---\n\n# Hello\nWorld"
    fm, remaining = _parse_frontmatter(text)
    assert fm == {"source": "research", "title": "Test Report", "quality_score": 8.5}
    assert remaining == "# Hello\nWorld"


def test_parse_frontmatter_no_frontmatter():
    """File without frontmatter returns empty dict."""
    text = "# Just a heading\n\nContent here"
    fm, remaining = _parse_frontmatter(text)
    assert fm == {}
    assert remaining == text


def test_parse_frontmatter_no_closing_marker():
    """Missing closing --- returns empty dict and original text."""
    text = "---\nkey: val\n\n# Content"
    fm, remaining = _parse_frontmatter(text)
    assert fm == {}
    assert remaining == text


def test_parse_frontmatter_empty():
    """Empty frontmatter (---\\n---) returns empty dict, strips markers."""
    text = "---\n---\n\n# Content"
    fm, remaining = _parse_frontmatter(text)
    assert fm == {}
    assert remaining == "# Content"


def test_parse_frontmatter_malformed_yaml():
    """Malformed YAML falls back silently."""
    text = "---\n  : invalid yaml :\n---\n# Content"
    fm, remaining = _parse_frontmatter(text)
    assert fm == {}
    assert remaining == text


def test_frontmatter_in_metadata():
    """MarkdownLoader merges frontmatter into document metadata."""
    content = "---\nsource: research\nresearch_id: abc123\n---\n# Test Doc\n\nBody text."
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write(content)
        tmp_path = f.name

    try:
        loader = MarkdownLoader()
        doc = loader.load(tmp_path)
        assert doc.metadata["source"] == "research"
        assert doc.metadata["research_id"] == "abc123"
        assert doc.metadata["source_path"] == tmp_path
        assert doc.metadata["doc_type"] == "markdown"
        # Frontmatter strippped from text
        assert doc.text.startswith("# Test Doc")
        assert "source: research" not in doc.text
    finally:
        Path(tmp_path).unlink()


def test_system_fields_override_frontmatter():
    """System fields (source_path, doc_hash, file_name) take priority."""
    content = "---\nsource_path: /fake/override\nfile_name: fake.md\n---\n# Real Doc"
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write(content)
        tmp_path = f.name

    try:
        loader = MarkdownLoader()
        doc = loader.load(tmp_path)
        assert doc.metadata["source_path"] == tmp_path  # system value, not /fake/override
        assert doc.metadata["file_name"] == Path(tmp_path).name  # system value, not fake.md
    finally:
        Path(tmp_path).unlink()
