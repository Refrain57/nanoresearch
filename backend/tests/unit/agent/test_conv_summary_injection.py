"""P3.3: <conversation_summary> = near (deterministic) + far (conv-scoped semantic recall)."""
from nanoresearch.agent.context import ContextBuilder


class _FakeKS:
    def __init__(self, all_sums, far):
        self._all = all_sums
        self._far = far
        self.list_called = 0
        self.search_kwargs = None

    def list_conv_summaries_sync(self, uid, conversation_id):
        self.list_called += 1
        return self._all

    def search_conv_summaries_sync(self, query, uid=None, conversation_id=None,
                                   exclude_ids=None, top_k=5, apply_decay=True):
        self.search_kwargs = {"conversation_id": conversation_id, "exclude_ids": exclude_ids}
        return self._far

    def search_events_sync(self, *a, **k):
        return []


def test_conversation_summary_block_near_and_far(tmp_path):
    all_sums = [
        {"id": "cs_old", "metadata": {"text": "早期讨论 A", "turn_end": 2}},
        {"id": "cs_new", "metadata": {"text": "最近讨论 B", "turn_end": 9}},
    ]
    far = [{"id": "cs_far", "metadata": {"text": "语义召回的早期段"}}]
    ks = _FakeKS(all_sums, far)
    cb = ContextBuilder(workspace=tmp_path, knowledge_search=ks, uid="u1")
    prompt = cb.build_system_prompt(topic="A vs B", conversation_id="c1")
    assert "<conversation_summary>" in prompt
    assert "最近讨论 B" in prompt          # near — newest, deterministic
    assert "语义召回的早期段" in prompt      # far — semantic recall
    # far search is conv-scoped and excludes the near id
    assert ks.search_kwargs["conversation_id"] == "c1"
    assert "cs_new" in ks.search_kwargs["exclude_ids"]


def test_no_conversation_id_no_block(tmp_path):
    ks = _FakeKS([], [])
    cb = ContextBuilder(workspace=tmp_path, knowledge_search=ks, uid="u1")
    prompt = cb.build_system_prompt(topic="x")  # no conversation_id
    assert "<conversation_summary>" not in prompt
    assert ks.list_called == 0
