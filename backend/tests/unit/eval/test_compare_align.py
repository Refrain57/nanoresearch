from nanoresearch.eval.compare import align_steps, get_args_matcher

M = get_args_matcher("exact")


def _c(name, **params):
    return {"name": name, "params": params}


def test_identical_all_match():
    chain = [_c("search", q="a"), _c("read", url="x")]
    steps, first = align_steps(chain, list(chain), M)
    assert [s["status"] for s in steps] == ["match", "match"]
    assert first is None


def test_param_diff_localized():
    steps, first = align_steps([_c("search", q="a")], [_c("search", q="b")], M)
    assert [s["status"] for s in steps] == ["param_diff"]
    assert first == 0


def test_inserted_call_does_not_cascade():
    # candidate inserts an extra 'think' between the two shared calls
    base = [_c("search", q="a"), _c("read", url="x")]
    cand = [_c("search", q="a"), _c("think"), _c("read", url="x")]
    steps, first = align_steps(base, cand, M)
    assert [s["status"] for s in steps] == ["match", "added", "match"]
    assert first == 1  # only the insertion diverges; 'read' still matches


def test_removed_call():
    base = [_c("search", q="a"), _c("read", url="x")]
    cand = [_c("search", q="a")]
    steps, first = align_steps(base, cand, M)
    assert [s["status"] for s in steps] == ["match", "removed"]
    assert first == 1
