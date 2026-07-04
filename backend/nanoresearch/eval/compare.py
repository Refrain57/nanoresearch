"""Pure, symmetric compare of two agent runs' tool_call_chain.

One LCS alignment produces a canonical diff; the four verdicts
(strict/unordered/subset/superset) are functions over that same data.

Tool-args matchers are adapted from agentevals (MIT, commit 4b68015)
trajectory/utils.py:137-171.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable


# --- tool-args matchers -------------------------------------------------------
# Adapted from agentevals (MIT, commit 4b68015) trajectory/utils.py:137-171.

def _exact(a: dict, b: dict) -> bool:
    return a == b


def _subset(a: dict, b: dict) -> bool:
    # every key/value in a exists in b
    return all(k in b and b[k] == v for k, v in a.items())


def _superset(a: dict, b: dict) -> bool:
    # every key/value in b exists in a
    return all(k in a and a[k] == v for k, v in b.items())


def _ignore(a: dict, b: dict) -> bool:
    return True


def get_args_matcher(
    mode: str = "exact",
    ignore_fields: list[str] | None = None,
) -> Callable[[dict, dict], bool]:
    """Return a two-arg predicate comparing tool-call param dicts.

    mode: exact | ignore | subset | superset.
    ignore_fields: when given, compare all keys EXCEPT these (top-level), under
    the chosen mode's equality — implemented by stripping the fields first.
    """
    base = {"exact": _exact, "subset": _subset, "superset": _superset, "ignore": _ignore}.get(mode)
    if base is None:
        raise ValueError(f"Invalid args match mode: {mode!r}")
    if not ignore_fields:
        return base

    def _strip(d: dict) -> dict:
        out = dict(d)
        for path in ignore_fields:
            top = path.split(".")[0]
            out.pop(top, None)
        return out

    def matcher(a: dict, b: dict) -> bool:
        return base(_strip(a), _strip(b))

    return matcher


# --- LCS alignment ---------------------------------------------------------------


def _extract_names(chain: list[dict]) -> list[str]:
    return [str(c.get("name", "")) for c in chain]


def _lcs_ops(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    """LCS backbone over tool NAMES → ordered list of alignment ops.

    Each op is (i, j): (i,j)=aligned pair, (i,None)=baseline-only, (None,j)=candidate-only.
    LCS aligns on name so an inserted/removed call does not cascade-misalign the tail.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    ops: list[tuple[int | None, int | None]] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            ops.append((i, j)); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            ops.append((i, None)); i += 1
        else:
            ops.append((None, j)); j += 1
    while i < n:
        ops.append((i, None)); i += 1
    while j < m:
        ops.append((None, j)); j += 1
    return ops


def align_steps(
    baseline_chain: list[dict],
    candidate_chain: list[dict],
    matcher: Callable[[dict, dict], bool],
) -> tuple[list[dict], int | None]:
    """One LCS pass → per-step status list + index of first non-match (or None)."""
    a_names, b_names = _extract_names(baseline_chain), _extract_names(candidate_chain)
    steps: list[dict] = []
    first_divergence: int | None = None
    for i, j in _lcs_ops(a_names, b_names):
        if i is not None and j is not None:
            b_entry, c_entry = baseline_chain[i], candidate_chain[j]
            same = matcher(b_entry.get("params") or {}, c_entry.get("params") or {})
            status = "match" if same else "param_diff"
            step = {"status": status, "name": a_names[i], "baseline": b_entry, "candidate": c_entry}
        elif i is not None:
            status = "removed"
            step = {"status": status, "name": a_names[i], "baseline": baseline_chain[i], "candidate": None}
        else:
            status = "added"
            step = {"status": status, "name": b_names[j], "baseline": None, "candidate": candidate_chain[j]}
        if status != "match" and first_divergence is None:
            first_divergence = len(steps)
        steps.append(step)
    return steps, first_divergence


# --- verdicts ---------------------------------------------------------------


def _multiset_contains(a: list[dict], b: list[dict], matcher: Callable[[dict, dict], bool]) -> bool:
    """True if every call in b has an unused name+args match in a (greedy).

    Adapted from agentevals (MIT, commit 4b68015) _is_trajectory_superset.
    """
    used: set[int] = set()
    for bc in b:
        found = False
        for idx, ac in enumerate(a):
            if idx in used or ac.get("name") != bc.get("name"):
                continue
            if matcher(ac.get("params") or {}, bc.get("params") or {}):
                used.add(idx); found = True; break
        if not found:
            return False
    return True


def compare_runs(
    baseline: dict,
    candidate: dict,
    *,
    args_match_mode: str = "exact",
    ignore_fields: list[str] | None = None,
) -> dict:
    """Symmetric compare of two runs. One alignment; four verdicts read the same data."""
    matcher = get_args_matcher(args_match_mode, ignore_fields)
    b_chain = list(baseline.get("tool_call_chain") or [])
    c_chain = list(candidate.get("tool_call_chain") or [])

    steps, first_divergence = align_steps(b_chain, c_chain, matcher)

    # verdicts — computed together off the same two chains / the same steps
    strict = all(s["status"] == "match" for s in steps)
    superset = _multiset_contains(c_chain, b_chain, matcher)   # candidate ⊇ baseline
    subset = _multiset_contains(b_chain, c_chain, matcher)     # candidate ⊆ baseline
    unordered = superset and subset

    # post-divergence degrade: ignore order, compare remaining tool-name multisets
    if first_divergence is None:
        degraded = True
    else:
        tail = steps[first_divergence:]
        b_names = Counter(s["name"] for s in tail if s["baseline"] is not None)
        c_names = Counter(s["name"] for s in tail if s["candidate"] is not None)
        degraded = b_names == c_names

    return {
        "steps": steps,
        "first_divergence": first_divergence,
        "verdicts": {"strict": strict, "unordered": unordered, "subset": subset, "superset": superset},
        "degraded_tool_set_match": degraded,
        "final_response_equal": (baseline.get("final_response") == candidate.get("final_response")),
    }
