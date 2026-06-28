"""C4: token trigger T2 uses the shared tail_protect / target-ratio config."""
from __future__ import annotations

import nanoresearch.agent.memory as memory_mod
from nanoresearch.agent.memory import MemoryConsolidator, CONSOLIDATION_TAIL_PROTECT
from nanoresearch.session.manager import Session


def test_pick_boundary_default_tail_protect_is_shared_constant():
    consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
    rows = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(40)]
    session = Session(key="web:t2", messages=rows, last_consolidated=0)

    # Tail of CONSOLIDATION_TAIL_PROTECT messages must never be selected.
    boundary = consolidator.pick_consolidation_boundary(session, tokens_to_remove=1)
    assert boundary is not None
    end_idx, _ = boundary
    assert end_idx <= len(rows) - CONSOLIDATION_TAIL_PROTECT


def test_target_ratio_constant_is_used(monkeypatch):
    """maybe_consolidate_by_tokens must derive `target` from the ratio constant."""
    import inspect
    src = inspect.getsource(MemoryConsolidator.maybe_consolidate_by_tokens)
    assert "TOKEN_CONSOLIDATION_TARGET_RATIO" in src
    assert "// 2" not in src
