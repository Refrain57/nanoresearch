"""Contract test for mem_events write/search — fake store + fake encoder (no live Chroma)."""
from nanoresearch.research.knowledge_search import KnowledgeSearch


class _FakeEncoder:
    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


class _FakeStore:
    def __init__(self, canned=None):
        self.inserted = []
        self._canned = canned or []

    def insert_batch(self, items):
        self.inserted.extend(items)

    def query(self, vector, top_k):
        return list(self._canned)


def _ks(events_store):
    return KnowledgeSearch(dense_encoder=_FakeEncoder(), settings=None, mem_events_store=events_store)


def test_write_events_composes_text_and_metadata_and_returns_ids():
    store = _FakeStore()
    ks = _ks(store)
    ids = ks.write_events_sync(
        [{"topic": "3DGS", "action": "compared", "result": "VA-GS wins",
          "time": "2026-07-01T10:00:00", "conversation_id": "c1"}],
        uid="u1",
    )
    assert len(ids) == 1 and ids[0].startswith("ev_")
    md = store.inserted[0]["metadata"]
    assert md["type"] == "event" and md["uid"] == "u1" and md["conversation_id"] == "c1"
    assert md["text"] == "3DGS | compared | VA-GS wins"
    assert md["topic"] == "3DGS" and md["result"] == "VA-GS wins"


def test_search_events_filters_by_uid():
    canned = [
        {"id": "ev_a", "metadata": {"uid": "u1", "text": "3DGS compared", "created_at": "2026-07-01T10:00:00"}},
        {"id": "ev_b", "metadata": {"uid": "u2", "text": "other user", "created_at": "2026-07-01T10:00:00"}},
    ]
    ks = _ks(_FakeStore(canned=canned))
    out = ks.search_events_sync("3DGS", uid="u1")
    ids = {r["id"] for r in out}
    assert "ev_a" in ids and "ev_b" not in ids


def test_search_events_empty_when_no_store():
    ks = KnowledgeSearch(dense_encoder=_FakeEncoder(), settings=None, mem_events_store=None)
    assert ks.search_events_sync("x", uid="u1") == []


def test_write_events_noop_without_store():
    ks = KnowledgeSearch(dense_encoder=_FakeEncoder(), settings=None, mem_events_store=None)
    assert ks.write_events_sync([{"topic": "t", "action": "a", "result": "r"}], uid="u1") == []
