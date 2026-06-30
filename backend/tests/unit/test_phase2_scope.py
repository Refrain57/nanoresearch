"""Phase 2 Task 9: conversation-scoped shared summary + L2/L3 conversation filter."""


def test_memory_store_conversation_scoped_not_forked_by_agent(tmp_path):
    """Two mains (different agent_id) on the same conversation share one MEMORY.md — the summary
    is no longer forked per agent (serial-MVP scope adjustment)."""
    from nanoresearch.agent.memory import MemoryStore
    s_a = MemoryStore(tmp_path, agent_id="A", conversation_id="conv-1")
    s_b = MemoryStore(tmp_path, agent_id="B", conversation_id="conv-1")
    s_a.write_long_term("shared summary")
    assert s_a.memory_dir == s_b.memory_dir
    assert s_b.read_long_term() == "shared summary"


def test_memory_store_backward_compat_agent_dir(tmp_path):
    """No conversation_id → legacy per-agent dir (single-main unchanged)."""
    from nanoresearch.agent.memory import MemoryStore
    s = MemoryStore(tmp_path, agent_id="A")
    assert s.memory_dir == tmp_path / "agents" / "A" / "memory"


def test_search_user_memory_filters_by_conversation():
    """search_user_memory_sync filters candidates to the given conversation_id (L2/L3 scoping)."""
    from nanoresearch.research.knowledge_search import KnowledgeSearch
    ks = KnowledgeSearch.__new__(KnowledgeSearch)

    class _Store:
        def query(self, vector, top_k):
            return [
                {"id": "1", "score": 1.0, "metadata": {"uid": "u1", "conversation_id": "A", "text": "a"}},
                {"id": "2", "score": 1.0, "metadata": {"uid": "u1", "conversation_id": "B", "text": "b"}},
            ]

    ks.user_memory_store = _Store()
    ks._embed = lambda q: [0.0]
    ks._bm25_rank = lambda q, c: c
    ks._rrf_fuse = lambda c, b, top_k: c
    ks._get_reranker = lambda: None
    ks._apply_decay = lambda f: f

    res = ks.search_user_memory_sync("q", uid="u1", conversation_id="A")
    assert [r["id"] for r in res] == ["1"]

    # no conversation_id → no conversation filter (backward compatible)
    res_all = ks.search_user_memory_sync("q", uid="u1")
    assert {r["id"] for r in res_all} == {"1", "2"}
