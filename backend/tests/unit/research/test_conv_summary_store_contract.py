"""Contract test for mem_conv_summaries write/search/list — fake store + encoder."""
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


def _ks(store):
    return KnowledgeSearch(dense_encoder=_FakeEncoder(), settings=None,
                           mem_conv_summaries_store=store)


def test_write_conv_summary_metadata_and_id():
    store = _FakeStore()
    rid = _ks(store).write_conv_summary_sync(
        "discussed 3DGS", uid="u1", conversation_id="c1", turn_start=0, turn_end=4, topic="3DGS")
    assert rid.startswith("cs_")
    md = store.inserted[0]["metadata"]
    assert md["type"] == "conv_summary" and md["uid"] == "u1"
    assert md["conversation_id"] == "c1" and md["turn_start"] == 0 and md["turn_end"] == 4
    assert md["topic"] == "3DGS" and md["text"] == "discussed 3DGS"


def test_search_filters_by_conversation_and_uid_and_excludes():
    canned = [
        {"id": "cs_a", "metadata": {"uid": "u1", "conversation_id": "c1", "text": "3DGS a", "created_at": "2026-07-01T10:00:00"}},
        {"id": "cs_b", "metadata": {"uid": "u1", "conversation_id": "c2", "text": "3DGS b", "created_at": "2026-07-01T10:00:00"}},
        {"id": "cs_c", "metadata": {"uid": "u2", "conversation_id": "c1", "text": "3DGS c", "created_at": "2026-07-01T10:00:00"}},
    ]
    out = _ks(_FakeStore(canned)).search_conv_summaries_sync(
        "3DGS", uid="u1", conversation_id="c1", exclude_ids=["cs_a"])
    ids = {r["id"] for r in out}
    assert ids == set()  # cs_a excluded; cs_b wrong conv; cs_c wrong uid


def test_list_returns_all_for_conversation():
    canned = [
        {"id": "cs_a", "metadata": {"uid": "u1", "conversation_id": "c1", "turn_end": 4, "text": "a"}},
        {"id": "cs_b", "metadata": {"uid": "u1", "conversation_id": "c1", "turn_end": 9, "text": "b"}},
        {"id": "cs_x", "metadata": {"uid": "u1", "conversation_id": "c2", "turn_end": 4, "text": "x"}},
    ]
    out = _ks(_FakeStore(canned)).list_conv_summaries_sync(uid="u1", conversation_id="c1")
    assert {r["id"] for r in out} == {"cs_a", "cs_b"}
