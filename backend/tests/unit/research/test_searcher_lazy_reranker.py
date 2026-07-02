"""Regression tests: the research reranker must NOT load its (heavy) model eagerly.

A SearchOrchestrator is constructed per agent run, but almost all runs (plain chat) never
invoke research/rerank. Eagerly loading the Cross-Encoder model in __init__ made every message
pay a multi-second model load. These tests pin the lazy-load contract and the process-level
model cache that makes repeated research runs reuse a single loaded model.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from nanoresearch.research.searcher import SearchOrchestrator
from nanoresearch.research.types import ResearchConfig


def _make_orchestrator() -> SearchOrchestrator:
    cfg = ResearchConfig()  # defaults: rerank_enabled=True, provider="cross_encoder"
    assert cfg.rerank_enabled and cfg.rerank_provider != "none"
    return SearchOrchestrator(MagicMock(), MagicMock(), cfg)


def test_reranker_not_loaded_on_construction():
    """Constructing the orchestrator (even with rerank enabled) must not build the reranker."""
    with patch.object(SearchOrchestrator, "_init_reranker", autospec=True) as spy:
        orch = _make_orchestrator()
        spy.assert_not_called()
    assert orch._reranker is None
    assert orch._reranker_ready is False
    assert orch._rerank_wanted is True


def test_reranker_built_once_on_first_use():
    """_ensure_reranker builds the reranker exactly once, only on first call."""
    def _fake_init(self):
        self._reranker = object()

    with patch.object(SearchOrchestrator, "_init_reranker", autospec=True,
                      side_effect=_fake_init) as spy:
        orch = _make_orchestrator()
        spy.assert_not_called()

        orch._ensure_reranker()
        assert spy.call_count == 1
        assert orch._reranker is not None

        orch._ensure_reranker()  # idempotent
        assert spy.call_count == 1


def test_disabled_reranker_never_builds():
    """provider='none' → no reranker wanted, _ensure_reranker is a no-op."""
    cfg = ResearchConfig(rerank_provider="none")
    with patch.object(SearchOrchestrator, "_init_reranker", autospec=True) as spy:
        orch = SearchOrchestrator(MagicMock(), MagicMock(), cfg)
        orch._ensure_reranker()
        spy.assert_not_called()
    assert orch._rerank_wanted is False


def test_cross_encoder_model_cached_across_instances():
    """The Cross-Encoder model loads once per process even across reranker instances."""
    from nanoresearch.rag.libs.reranker import cross_encoder_reranker as cer

    cer._MODEL_CACHE.clear()
    fake_ctor = MagicMock(side_effect=lambda name: MagicMock(name=f"model:{name}"))
    settings = SimpleNamespace(rerank=SimpleNamespace(model="dummy/model"))

    # sentence_transformers may not be installed in the dev env; inject a stub so the
    # `from sentence_transformers import CrossEncoder` inside the loader resolves to our fake.
    fake_st = ModuleType("sentence_transformers")
    fake_st.CrossEncoder = fake_ctor
    with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
        r1 = cer.CrossEncoderReranker(settings)
        r2 = cer.CrossEncoderReranker(settings)

    assert fake_ctor.call_count == 1          # only one real load
    assert r1.model is r2.model               # shared instance
    cer._MODEL_CACHE.clear()
