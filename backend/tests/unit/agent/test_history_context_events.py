"""P2.3: <history> recall reads mem_events (not the flat user_memory)."""
from nanoresearch.agent.context import ContextBuilder


class _FakeKS:
    def __init__(self, rows):
        self.rows = rows
        self.events_called = 0
        self.user_memory_called = 0

    def search_events_sync(self, query, top_k=5, apply_decay=True, uid=None):
        self.events_called += 1
        return self.rows

    def search_user_memory_sync(self, *a, **k):
        self.user_memory_called += 1
        return []


def test_history_context_reads_events(tmp_path):
    rows = [{
        "id": "ev_1",
        "text": "3DGS compared VA-GS vs HoGS",
        "metadata": {"text": "3DGS compared VA-GS vs HoGS", "confidence": 0.8,
                     "created_at": "2026-07-01T10:00:00"},
    }]
    ks = _FakeKS(rows)
    cb = ContextBuilder(workspace=tmp_path, knowledge_search=ks, uid="u1")
    out = cb.build_history_context("3DGS", token_budget=500, uid="u1")
    assert "相关历史记忆" in out
    assert "3DGS compared" in out
    assert ks.events_called == 1 and ks.user_memory_called == 0


def test_history_context_empty_without_ks(tmp_path):
    cb = ContextBuilder(workspace=tmp_path, knowledge_search=None, uid="u1")
    assert cb.build_history_context("x", token_budget=500, uid="u1") == ""
