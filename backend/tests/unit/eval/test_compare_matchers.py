from nanoresearch.eval.compare import get_args_matcher


def test_exact():
    m = get_args_matcher("exact")
    assert m({"q": "a"}, {"q": "a"})
    assert not m({"q": "a"}, {"q": "b"})


def test_ignore():
    m = get_args_matcher("ignore")
    assert m({"q": "a"}, {"q": "totally different"})


def test_subset_superset():
    # subset: baseline args are a subset of candidate args
    assert get_args_matcher("subset")({"q": "a"}, {"q": "a", "k": 1})
    assert not get_args_matcher("subset")({"q": "a", "k": 1}, {"q": "a"})
    # superset: baseline args are a superset of candidate args
    assert get_args_matcher("superset")({"q": "a", "k": 1}, {"q": "a"})


def test_ignore_fields():
    m = get_args_matcher("exact", ignore_fields=["ts"])
    assert m({"q": "a", "ts": 1}, {"q": "a", "ts": 999})
    assert not m({"q": "a", "ts": 1}, {"q": "b", "ts": 1})
