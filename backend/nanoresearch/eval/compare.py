"""Pure, symmetric compare of two agent runs' tool_call_chain.

One LCS alignment produces a canonical diff; the four verdicts
(strict/unordered/subset/superset) are functions over that same data.

Tool-args matchers are adapted from agentevals (MIT, commit 4b68015)
trajectory/utils.py:137-212.
"""

from __future__ import annotations

from typing import Any, Callable


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


def _get_nested(d: dict, path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def get_args_matcher(
    mode: str = "exact",
    ignore_fields: list[str] | None = None,
) -> Callable[[dict, dict], bool]:
    """Return a two-arg predicate comparing tool-call param dicts.

    mode: exact | ignore | subset | superset.
    ignore_fields: when given, compare all keys EXCEPT these (dot-path), under
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
