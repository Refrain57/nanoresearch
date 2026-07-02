"""P4.2 guard: no source references the retired user_memory API or the deleted extractor."""
from pathlib import Path

BANNED = [
    "ConversationKnowledgeExtractor",
    "search_user_memory",
    "write_user_memory",
    "cleanup_old_user_memory",
    "user_memory_store",
]


def test_no_legacy_memory_symbols_in_source():
    root = Path(__file__).resolve().parents[2] / "nanoresearch"
    offenders = []
    for p in root.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for sym in BANNED:
            if sym in text:
                offenders.append(f"{p.relative_to(root)}: {sym}")
    assert not offenders, f"legacy memory symbols still referenced: {offenders}"
