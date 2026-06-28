"""Legacy env var compatibility shim for the nanobot → nanoresearch rename.

Reads any `NANOBOT_*` environment variable that does NOT have a corresponding
`NANORESEARCH_*` already set, copies the value to the new name, and emits a
deprecation warning. Must be called once at process startup (CLI entry,
server entry, worker entry) BEFORE any code reads env vars or Pydantic
Settings loads.

The compat layer will be removed in v0.3.0.
"""

from __future__ import annotations

import os
import warnings

_LEGACY_PREFIX = "NANOBOT_"
_NEW_PREFIX = "NANORESEARCH_"
_REMOVED_IN = "0.3.0"

_applied = False


def apply_legacy_env_compat() -> list[tuple[str, str]]:
    """Copy NANOBOT_* env vars to NANORESEARCH_* with deprecation warnings.

    Idempotent: subsequent calls are no-ops. Returns the list of
    (old_name, new_name) pairs that were copied this call (empty after the
    first call).
    """
    global _applied
    if _applied:
        return []
    _applied = True

    copied: list[tuple[str, str]] = []
    for old_name, value in list(os.environ.items()):
        if not old_name.startswith(_LEGACY_PREFIX):
            continue
        new_name = _NEW_PREFIX + old_name[len(_LEGACY_PREFIX):]
        if new_name in os.environ:
            warnings.warn(
                f"{old_name} is set alongside {new_name}; "
                f"{old_name} is deprecated and will be removed in v{_REMOVED_IN}. "
                f"Using {new_name}.",
                DeprecationWarning,
                stacklevel=2,
            )
            continue
        os.environ[new_name] = value
        copied.append((old_name, new_name))
        warnings.warn(
            f"{old_name} is deprecated; use {new_name}. "
            f"{old_name} will be removed in v{_REMOVED_IN}.",
            DeprecationWarning,
            stacklevel=2,
        )
    return copied


def _reset_for_tests() -> None:
    """Test-only helper to reset the idempotency guard between tests."""
    global _applied
    _applied = False
