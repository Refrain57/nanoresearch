from nanoresearch.eval.compare import compare_runs


def _c(name, **params):
    return {"name": name, "params": params}


def _run(chain, final="ans"):
    return {"tool_call_chain": chain, "final_response": final}


def test_identical_runs_all_verdicts_true():
    chain = [_c("search", q="a"), _c("read", url="x")]
    out = compare_runs(_run(chain), _run(list(chain)))
    assert out["first_divergence"] is None
    assert out["verdicts"] == {"strict": True, "unordered": True, "subset": True, "superset": True}
    assert out["degraded_tool_set_match"] is True
    assert out["final_response_equal"] is True


def test_reordered_is_unordered_not_strict():
    base = [_c("a"), _c("b")]
    cand = [_c("b"), _c("a")]
    out = compare_runs(_run(base), _run(cand))
    assert out["verdicts"]["strict"] is False
    assert out["verdicts"]["unordered"] is True


def test_candidate_extra_tool_is_superset_not_subset():
    base = [_c("search", q="a")]
    cand = [_c("search", q="a"), _c("read", url="x")]
    out = compare_runs(_run(base), _run(cand))
    # candidate ⊇ baseline
    assert out["verdicts"]["superset"] is True
    assert out["verdicts"]["subset"] is False


def test_degraded_set_match_after_divergence():
    # diverge at step 0 (param), but the remaining tool SETS match
    base = [_c("search", q="a"), _c("read", url="x")]
    cand = [_c("search", q="b"), _c("read", url="x")]
    out = compare_runs(_run(base), _run(cand))
    assert out["first_divergence"] == 0
    assert out["degraded_tool_set_match"] is True


def test_final_response_inequality():
    out = compare_runs(_run([], "yes"), _run([], "no"))
    assert out["final_response_equal"] is False
