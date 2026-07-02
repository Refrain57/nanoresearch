from nanoresearch.rag.wiki.article_generator import (
    build_article_prompt, evidence_signature, build_citations,
    build_concept_prompt, build_overview_prompt, overview_signature,
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


def test_build_concept_prompt_has_topic_numbered_evidence_and_citation_instr():
    system, user = build_concept_prompt(
        "体渲染",
        [{"chunk_id": "x", "content": "volume rendering integrates radiance", "source": "p.pdf"}],
    )
    assert "体渲染" in user
    assert "[1]" in user and "volume rendering" in user
    assert "[^" in user               # instructs [^n]
    assert "不" in system              # grounding guard (只依据/不编造)


def test_build_overview_prompt_lists_top_entities_and_relations():
    system, user = build_overview_prompt(
        [{"name": "3dgs", "mentions": 12}, {"name": "nerf", "mentions": 9}],
        [{"source": "3dgs", "label": "faster_than", "target": "nerf"}],
    )
    assert "3dgs" in user and "nerf" in user
    assert "faster_than" in user
    assert "导览" in user or "总览" in user


def test_overview_signature_stable_and_sensitive():
    a = overview_signature([{"name": "3dgs"}], [{"source": "3dgs", "label": "x", "target": "nerf"}])
    b = overview_signature([{"name": "3dgs"}], [{"source": "3dgs", "label": "x", "target": "nerf"}])
    c = overview_signature([{"name": "nerf"}], [])
    assert a == b and a != c
