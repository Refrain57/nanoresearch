# backend/tests/rag/test_build_citations_from_chunks.py
from nanoresearch.rag.mcp_server.tools.agentic.shared import build_citations_from_chunks


def test_build_basic_and_dedup():
    chunks = [
        {"chunk_id": "c1", "score": 0.9, "text": "T" * 300,
         "metadata": {"source_path": "a.pdf", "page": 3, "doc_id": "d1"}},
        {"chunk_id": "c1", "score": 0.8, "text": "dup", "metadata": {"source_path": "a.pdf"}},
        {"chunk_id": "c2", "score": 0.7, "text": "hello",
         "metadata": {"source": "b.md"}},
    ]
    out = build_citations_from_chunks(chunks)
    assert [c["index"] for c in out] == [1, 2]          # 去重后两条，序号连续
    assert out[0]["chunk_id"] == "c1"
    assert out[0]["source"] == "a.pdf"
    assert out[0]["page"] == 3
    assert out[0]["doc_id"] == "d1"
    assert len(out[0]["snippet"]) <= 203                 # 200 + "..."
    assert out[1]["source"] == "b.md"                    # 回退到 metadata.source
    assert out[1]["page"] is None
