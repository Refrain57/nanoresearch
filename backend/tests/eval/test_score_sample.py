"""Tests for the ScoreSample dataclass used in σ-weighted gate."""
import math
import pytest

from nanoresearch.eval.score_sample import ScoreSample


def test_from_observations_computes_mean_and_std():
    s = ScoreSample.from_observations([0.5, 0.6, 0.7])
    assert math.isclose(s.mean, 0.6, abs_tol=1e-9)
    assert math.isclose(s.std, math.sqrt(((0.5 - 0.6) ** 2 + 0 + (0.7 - 0.6) ** 2) / 2), abs_tol=1e-9)
    assert s.n == 3


def test_from_observations_single_sample_has_zero_std():
    s = ScoreSample.from_observations([0.5])
    assert s.mean == 0.5
    assert s.std == 0.0
    assert s.n == 1


def test_from_observations_empty_raises():
    with pytest.raises(ValueError, match="at least one observation"):
        ScoreSample.from_observations([])


def test_serializable_to_dict():
    s = ScoreSample(mean=0.6, std=0.1, n=3)
    assert s.to_dict() == {"mean": 0.6, "std": 0.1, "n": 3}


def test_roundtrip_dict():
    s = ScoreSample(mean=0.6, std=0.1, n=3)
    assert ScoreSample.from_dict(s.to_dict()) == s
