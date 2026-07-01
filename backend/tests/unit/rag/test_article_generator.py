from nanoresearch.rag.wiki.article_generator import (
    build_article_prompt, evidence_signature, build_citations,
)


def test_evidence_signature_stable_and_order_independent():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "x"}]
    assert evidence_signature(a) == evidence_signature(b)
    assert evidence_signature(a) != evidence_signature([{"chunk_id": "z"}])


def test_build_citations_numbers_from_one_and_truncates():
    ev = [{"chunk_id": "x", "content": "c" * 400, "source": "p.pdf", "page": 2}]
    cites = build_citations(ev)
    assert cites[0]["index"] == 1 and cites[0]["source"] == "p.pdf" and cites[0]["page"] == 2
    assert len(cites[0]["snippet"]) <= 300


def test_build_article_prompt_includes_numbered_evidence_and_facts():
    system, user = build_article_prompt(
        "3dgs",
        [{"source": "3dgs", "label": "faster_than", "target": "nerf", "doc_count": 2}],
        [{"chunk_id": "x", "content": "explicit points", "source": "p.pdf"}],
    )
    assert "3dgs" in user
    assert "[1]" in user and "explicit points" in user   # numbered evidence
    assert "faster_than" in user                          # facts included
    assert "[^" in user                                   # instructs [^n] citation
