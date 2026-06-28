"""ScoreSample: scalar score with empirical standard deviation and sample count.

Used by the σ-weighted optimization gate (B2 of A1 Phase 1) to distinguish
real improvements from variance within the baseline noise envelope.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ScoreSample:
    mean: float
    std: float
    n: int

    @classmethod
    def from_observations(cls, observations: Iterable[float]) -> "ScoreSample":
        obs = list(observations)
        if not obs:
            raise ValueError("ScoreSample requires at least one observation")
        n = len(obs)
        mean = sum(obs) / n
        if n == 1:
            return cls(mean=mean, std=0.0, n=1)
        variance = sum((x - mean) ** 2 for x in obs) / (n - 1)  # sample variance (Bessel)
        return cls(mean=mean, std=math.sqrt(variance), n=n)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreSample":
        return cls(mean=float(d["mean"]), std=float(d["std"]), n=int(d["n"]))
