# Repo Cleanup Phase 4: Configurable Base Path via `NANORESEARCH_HOME`

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every hard-coded `Path.home() / ".nanoresearch"` in production code with a single helper `get_nanoresearch_home()` that reads the `NANORESEARCH_HOME` environment variable, defaulting to `~/.nanoresearch`, to unblock multi-tenant and containerized deployments where the base directory must be redirected (e.g. `/data/tenant_x`).

**Architecture:** One source-of-truth helper in `backend/nanoresearch/config/loader.py`. All current `~/.nanoresearch` constructions — in `paths.py`, `loader.py`, RAG settings, server entries, MCP collections tools, and CLI scripts — funnel through this helper. Legacy `NANOBOT_HOME` automatically translates to `NANORESEARCH_HOME` via the existing `apply_legacy_env_compat()` shim; helper calls it defensively so script entries that bypass `nanoresearch.__main__` still pick it up.

**Tech Stack:** Python 3.11+, `pathlib`, `pytest>=9.0`, existing `nanoresearch.utils.env_compat` shim. No new dependencies.

## Global Constraints

- Default unchanged when `NANORESEARCH_HOME` unset: `Path.home() / ".nanoresearch"`.
- Accept `~/x`, absolute paths, and relative paths via `Path(value).expanduser()` (no `.resolve()` — keep symlinks intact and avoid touching FS on read).
- `NANOBOT_HOME` compatibility handled by existing `apply_legacy_env_compat()` shim. Do **not** write new compat code.
- mkdir / permission failures propagate; never silently swallowed.
- Python invocation: `./backend/.venv/Scripts/python` from repo root (`D:\Code\nanobot`). System python is a Windows Store stub returning exit 49 — never use bare `python`.
- Repo-root `tests/` and `backend/tests/` both exist. Targeted pytest invocations only (full `pytest tests/` collection is broken by `tests/research/` module-level imports — known, out of scope).
- Git discipline: no `git add .` / `-A`; list files explicitly. No `--no-verify`, no `--amend`, no `--force` on push. Push failure → stop and report verbatim, do not retry-with-different-strategy.
- Each task ends with a commit and an explicit halt for human approval before the next task begins.

## Scope Decisions (defaults — user can override before Task 1)

1. **Task 2 + Task 3 (bypass sites outside `paths.py`) are included in Phase 4.** Reason: if `NANORESEARCH_HOME=/data/tenant_x` only redirects `paths.py` but `loader.py:25`, `rag/core/settings.py:18,413`, `server/main.py:125`, `server/routers/eval_router.py:1011`, `collections.py:×4`, and the two scripts still hit `~/.nanoresearch`, the feature is half-broken and multi-tenancy doesn't work. Spec literal wording is "改 paths.py 内部" but the spec goal (multi-tenant/serverside base) cannot be met without these sites.
2. **Helper lives in `loader.py`, not a new `config/base.py` module.** Reason: `paths.py` already imports from `loader.py` (`get_config_path`); reusing this dependency direction avoids inventing a third config-layer module for one function.
3. **Docs commit (Task 4) included in Phase 4.** Reason: README.md, SECURITY.md, docker-compose.yml currently document `~/.nanoresearch` as the only path. Without docs, users won't discover the env var.

## Hit Map (production code, 12 sites)

| File | Lines | Site count |
|---|---|---|
| `backend/nanoresearch/config/paths.py` | 39, 45, 46, 52, 57, 62 | 6 |
| `backend/nanoresearch/config/loader.py` | 25 | 1 |
| `backend/nanoresearch/rag/core/settings.py` | 18, 413 | 2 |
| `backend/nanoresearch/server/main.py` | 125 | 1 |
| `backend/nanoresearch/server/routers/eval_router.py` | 1011 | 1 |
| `backend/nanoresearch/rag/mcp_server/tools/agentic/collections.py` | 595, 666, 793, 854 | 4 |
| `backend/scripts/audit_duplicate_chunks.py` | 12 | 1 |
| `backend/scripts/migrate_sessions.py` | 109 | 1 |

Test sites (updated, not removed):
- `tests/config/test_config_paths.py` — 5 hits
- `backend/tests/eval/test_ragas_e2e.py:26` — 1 hit (script, not pytest target)

Out-of-scope dev scripts retaining literal `C:/Users/Augix/.nanoresearch/config.json`:
- `backend/test_themes_diagnostic.py`, `backend/test_ragas_transforms.py` — flagged but not modified; they are local dev-only scratch files, not under CI.

---

### Task 1: Helper + `loader.py` + `paths.py` + test suite

**Files:**
- Modify: `backend/nanoresearch/config/loader.py` (add helper, change `get_config_path` fallback)
- Modify: `backend/nanoresearch/config/paths.py` (6 sites)
- Modify: `tests/config/test_config_paths.py` (replace 5 assertions, add 3 new tests)
- Modify: `backend/tests/eval/test_ragas_e2e.py:26` (1 site)

**Interfaces:**
- Produces: `nanoresearch.config.loader.get_nanoresearch_home() -> pathlib.Path`
  - Reads `os.environ["NANORESEARCH_HOME"]`. Returns `Path(value).expanduser()` if set and non-empty, otherwise `Path.home() / ".nanoresearch"`.
  - Calls `apply_legacy_env_compat()` defensively (idempotent) at entry so `NANOBOT_HOME` is translated to `NANORESEARCH_HOME` before the env read, even when invoked from a script that bypasses `nanoresearch.__main__`.
  - Does **not** create the directory. Callers that need ensured directories use `ensure_dir()` downstream (existing pattern).
- Consumes: existing `nanoresearch.utils.env_compat.apply_legacy_env_compat`.

**Steps:**

- [ ] **Step 1.1: Capture baseline grep — file count and line list**

Run from repo root:
```bash
./backend/.venv/Scripts/python -c "import nanoresearch; print(nanoresearch.__file__)"
```
Expected: prints path under `D:\Code\nanobot\backend\nanoresearch\__init__.py` (source, not site-packages). If it resolves elsewhere, stop and report — venv is wrong.

Then capture the baseline:
```bash
grep -rn 'Path\.home()' backend/nanoresearch/ backend/scripts/ tests/ backend/tests/ > /tmp/phase4-baseline-pathhome.txt
wc -l /tmp/phase4-baseline-pathhome.txt
```
Expected: ≥12 matches. Record the exact count for Task 1 end-of-task verification (should drop by 6 after this task).

- [ ] **Step 1.2: Write failing test for default behavior**

Append to `tests/config/test_config_paths.py` (after the existing tests):

```python
def test_get_nanoresearch_home_defaults_to_user_home(monkeypatch):
    from nanoresearch.config.loader import get_nanoresearch_home

    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_HOME", raising=False)

    assert get_nanoresearch_home() == Path.home() / ".nanoresearch"
```

- [ ] **Step 1.3: Run test to verify it fails**

Run from repo root:
```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py::test_get_nanoresearch_home_defaults_to_user_home -v
```
Expected: FAIL with `ImportError: cannot import name 'get_nanoresearch_home' from 'nanoresearch.config.loader'`.

- [ ] **Step 1.4: Implement `get_nanoresearch_home()` in `loader.py`**

Edit `backend/nanoresearch/config/loader.py`. Replace lines 1–25 with:

```python
"""Configuration loading utilities."""

import json
import os
from pathlib import Path

import pydantic
from loguru import logger

from nanoresearch.config.schema import Config
from nanoresearch.utils.env_compat import apply_legacy_env_compat

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None


def get_nanoresearch_home() -> Path:
    """Return the base directory for all NanoResearch runtime state.

    Reads ``NANORESEARCH_HOME``; falls back to ``~/.nanoresearch``. Calls the
    legacy env-compat shim first so ``NANOBOT_HOME`` is honoured for scripts
    that bypass the main entrypoints. Does not create the directory.
    """
    apply_legacy_env_compat()
    raw = os.environ.get("NANORESEARCH_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".nanoresearch"


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return get_nanoresearch_home() / "config.json"
```

- [ ] **Step 1.5: Verify the default-behavior test now passes**

Run:
```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py::test_get_nanoresearch_home_defaults_to_user_home -v
```
Expected: PASS.

- [ ] **Step 1.6: Write failing test for `NANORESEARCH_HOME` env override**

Append to `tests/config/test_config_paths.py`:

```python
def test_get_nanoresearch_home_respects_env_override(monkeypatch, tmp_path):
    from nanoresearch.config.loader import get_nanoresearch_home

    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / "tenant_x"))

    assert get_nanoresearch_home() == tmp_path / "tenant_x"


def test_get_nanoresearch_home_expands_tilde(monkeypatch):
    from nanoresearch.config.loader import get_nanoresearch_home

    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", "~/custom-nr-root")

    assert get_nanoresearch_home() == Path.home() / "custom-nr-root"
```

- [ ] **Step 1.7: Run both new tests; verify both pass**

Run:
```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py::test_get_nanoresearch_home_respects_env_override tests/config/test_config_paths.py::test_get_nanoresearch_home_expands_tilde -v
```
Expected: 2 PASS.

- [ ] **Step 1.8: Write failing test for `NANOBOT_HOME` legacy translation**

Append to `tests/config/test_config_paths.py`:

```python
def test_nanobot_home_legacy_env_is_translated(monkeypatch, tmp_path, recwarn):
    from nanoresearch.config.loader import get_nanoresearch_home
    from nanoresearch.utils.env_compat import _reset_for_tests

    _reset_for_tests()
    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.setenv("NANOBOT_HOME", str(tmp_path / "legacy_tenant"))

    result = get_nanoresearch_home()

    assert result == tmp_path / "legacy_tenant"
    deprecation_msgs = [
        str(w.message) for w in recwarn.list
        if issubclass(w.category, DeprecationWarning) and "NANOBOT_HOME" in str(w.message)
    ]
    assert any("NANORESEARCH_HOME" in m for m in deprecation_msgs), (
        f"expected a DeprecationWarning naming both NANOBOT_HOME and NANORESEARCH_HOME, got {deprecation_msgs}"
    )
```

- [ ] **Step 1.9: Run; verify PASS**

Run:
```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py::test_nanobot_home_legacy_env_is_translated -v
```
Expected: PASS. The existing `apply_legacy_env_compat()` already handles `NANOBOT_*` → `NANORESEARCH_*` translation and emits a DeprecationWarning; no new compat code needed.

If FAIL with "expected DeprecationWarning ... got []": this likely means `apply_legacy_env_compat()` was already called earlier in the test session and the `_applied` guard short-circuits. The `_reset_for_tests()` call at the top is intended to defeat this; if it does not, debug `env_compat.py` rather than mutating the test expectation.

- [ ] **Step 1.10: Refactor `paths.py` — replace 6 `~/.nanoresearch` constructions**

Edit `backend/nanoresearch/config/paths.py`. Replace lines 1–8 with:

```python
"""Runtime path helpers derived from the active config context."""

from __future__ import annotations

from pathlib import Path

from nanoresearch.config.loader import get_config_path, get_nanoresearch_home
from nanoresearch.utils.helpers import ensure_dir
```

Replace lines 37–62 (the 6 hard-coded sites) with:

```python
def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure the agent workspace path."""
    path = Path(workspace).expanduser() if workspace else get_nanoresearch_home() / "workspace"
    return ensure_dir(path)


def is_default_workspace(workspace: str | Path | None) -> bool:
    """Return whether a workspace resolves to NanoResearch's default workspace path."""
    current = Path(workspace).expanduser() if workspace is not None else get_nanoresearch_home() / "workspace"
    default = get_nanoresearch_home() / "workspace"
    return current.resolve(strict=False) == default.resolve(strict=False)


def get_cli_history_path() -> Path:
    """Return the shared CLI history file path."""
    return get_nanoresearch_home() / "history" / "cli_history"


def get_bridge_install_dir() -> Path:
    """Return the shared WhatsApp bridge installation directory."""
    return get_nanoresearch_home() / "bridge"


def get_legacy_sessions_dir() -> Path:
    """Return the legacy global session directory used for migration fallback."""
    return get_nanoresearch_home() / "sessions"
```

- [ ] **Step 1.11: Update existing assertions in `tests/config/test_config_paths.py`**

In `test_shared_and_legacy_paths_remain_global`, `test_workspace_path_is_explicitly_resolved`, and `test_is_default_workspace_distinguishes_default_and_custom_paths`, the 5 existing assertions of the form `Path.home() / ".nanoresearch" / ...` must be updated to also exercise the env-override path. Replace those three tests with:

```python
def test_shared_and_legacy_paths_remain_global(monkeypatch):
    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    assert get_cli_history_path() == Path.home() / ".nanoresearch" / "history" / "cli_history"
    assert get_bridge_install_dir() == Path.home() / ".nanoresearch" / "bridge"
    assert get_legacy_sessions_dir() == Path.home() / ".nanoresearch" / "sessions"


def test_shared_and_legacy_paths_follow_nanoresearch_home(monkeypatch, tmp_path):
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / "tenant_y"))
    assert get_cli_history_path() == tmp_path / "tenant_y" / "history" / "cli_history"
    assert get_bridge_install_dir() == tmp_path / "tenant_y" / "bridge"
    assert get_legacy_sessions_dir() == tmp_path / "tenant_y" / "sessions"


def test_workspace_path_is_explicitly_resolved(monkeypatch):
    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    assert get_workspace_path() == Path.home() / ".nanoresearch" / "workspace"
    assert get_workspace_path("~/custom-workspace") == Path.home() / "custom-workspace"


def test_workspace_path_follows_nanoresearch_home(monkeypatch, tmp_path):
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / "tenant_z"))
    assert get_workspace_path() == tmp_path / "tenant_z" / "workspace"


def test_is_default_workspace_distinguishes_default_and_custom_paths(monkeypatch):
    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    assert is_default_workspace(None) is True
    assert is_default_workspace(Path.home() / ".nanoresearch" / "workspace") is True
    assert is_default_workspace("~/custom-workspace") is False
```

(`is_default_workspace` behaviour with env-override is implicitly covered by the workspace tests.)

- [ ] **Step 1.12: Update `backend/tests/eval/test_ragas_e2e.py:26`**

This file is a standalone script, not a pytest target. Update line 26 only:

Replace:
```python
    cfg_path = pathlib.Path.home() / ".nanoresearch" / "config.json"
```
with:
```python
    from nanoresearch.config.loader import get_nanoresearch_home
    cfg_path = get_nanoresearch_home() / "config.json"
```

- [ ] **Step 1.13: Run the full `test_config_paths.py` suite**

```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py -v
```
Expected: all tests PASS. The suite now contains the original 4 tests (+ rewritten variants) plus 6 new env-override tests.

- [ ] **Step 1.14: Sanity-check `paths.py` hit count dropped from 6 to 0**

```bash
grep -nE 'Path\.home\(\)' backend/nanoresearch/config/paths.py
```
Expected: no output (zero matches).

```bash
grep -nE 'Path\.home\(\)' backend/nanoresearch/config/loader.py
```
Expected: **exactly 1 hit**, on the line inside `get_nanoresearch_home()` reading `return Path.home() / ".nanoresearch"`. This is the intentional default-fallback when `NANORESEARCH_HOME` is unset. No other `Path.home()` should remain in `loader.py`.

- [ ] **Step 1.15: Smoke-test multi-tenant env override end-to-end**

```bash
NANORESEARCH_HOME=/tmp/nr_phase4_smoke ./backend/.venv/Scripts/python -c "
from nanoresearch.config.paths import get_workspace_path, get_cli_history_path, get_bridge_install_dir
print(get_workspace_path())
print(get_cli_history_path())
print(get_bridge_install_dir())
"
```
Expected output (Windows path separators OK):
```
/tmp/nr_phase4_smoke/workspace
/tmp/nr_phase4_smoke/history/cli_history
/tmp/nr_phase4_smoke/bridge
```
(`get_workspace_path` ensures the directory, so `/tmp/nr_phase4_smoke/workspace/` will be created. Other two are read-only.)

Then verify the legacy env shim:
```bash
NANOBOT_HOME=/tmp/nr_phase4_legacy ./backend/.venv/Scripts/python -W default -c "
import warnings
warnings.simplefilter('always', DeprecationWarning)
from nanoresearch.config.loader import get_nanoresearch_home
print(get_nanoresearch_home())
" 2>&1
```
Expected: prints `/tmp/nr_phase4_legacy` and a DeprecationWarning naming both `NANOBOT_HOME` and `NANORESEARCH_HOME`.

- [ ] **Step 1.16: Commit Task 1**

```bash
git add backend/nanoresearch/config/loader.py backend/nanoresearch/config/paths.py tests/config/test_config_paths.py backend/tests/eval/test_ragas_e2e.py
git status
git commit -m "$(cat <<'EOF'
feat(config): NANORESEARCH_HOME env var for configurable base path

Introduce get_nanoresearch_home() in config/loader.py reading
NANORESEARCH_HOME (default ~/.nanoresearch). Funnel paths.py 6 sites and
loader.py config-fallback through it. Test coverage: default, env
override, tilde expansion, NANOBOT_HOME legacy shim translation.

Refs: docs/superpowers/specs/2026-06-26-repo-cleanup-design.md L197-199
EOF
)"
git log -1 --stat
```

**Halt for review.** Do not start Task 2 without human approval of Commit A.

---

### Task 2: Bypass sites — server/scripts/settings layer (6 sites in 5 files)

**Files:**
- Modify: `backend/nanoresearch/rag/core/settings.py:18,413` (constant → lazy)
- Modify: `backend/nanoresearch/config/migration.py:82,85` (use lazy accessor)
- Modify: `backend/nanoresearch/server/main.py:125`
- Modify: `backend/nanoresearch/server/routers/eval_router.py:1011`
- Modify: `backend/scripts/audit_duplicate_chunks.py:12`
- Modify: `backend/scripts/migrate_sessions.py:109`

**Interfaces:**
- Consumes (from Task 1): `nanoresearch.config.loader.get_nanoresearch_home() -> Path`.
- Produces: `nanoresearch.rag.core.settings.default_settings_path() -> Path`
  - Replaces the module-level constant `DEFAULT_SETTINGS_PATH`. Lazy — resolved on each call so env changes take effect after-import.

**Why settings.py needs the constant-to-function conversion:** `DEFAULT_SETTINGS_PATH: Path = Path.home() / ".nanoresearch" / "settings.yaml"` evaluates at import time. If we replace `Path.home() / ".nanoresearch"` with `get_nanoresearch_home()` while keeping the module-level binding, the value freezes at the first import — before the test or runtime override has any chance to take effect. Lazy resolution via a function is the only correct fix.

**Steps:**

- [ ] **Step 2.1: Write a failing integration test that asserts settings.yaml default path follows the env**

Append to `tests/config/test_config_paths.py`:

```python
def test_default_settings_path_follows_nanoresearch_home(monkeypatch, tmp_path):
    from nanoresearch.rag.core.settings import default_settings_path

    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / "rag_tenant"))

    assert default_settings_path() == tmp_path / "rag_tenant" / "settings.yaml"
```

Run:
```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py::test_default_settings_path_follows_nanoresearch_home -v
```
Expected: FAIL with `ImportError: cannot import name 'default_settings_path'`.

- [ ] **Step 2.2: Implement `default_settings_path()` in `settings.py` and remove the constant**

Edit `backend/nanoresearch/rag/core/settings.py` line 18. Replace:
```python
# Default settings path - user-level config
DEFAULT_SETTINGS_PATH: Path = Path.home() / ".nanoresearch" / "settings.yaml"
```
with:
```python
# Default settings path - user-level config (lazy so NANORESEARCH_HOME overrides take effect)
def default_settings_path() -> Path:
    """Return the active default ``settings.yaml`` path under the current NANORESEARCH_HOME."""
    from nanoresearch.config.loader import get_nanoresearch_home
    return get_nanoresearch_home() / "settings.yaml"
```

(Local import in the function avoids any circular-import risk between `rag/core/settings.py` and `config/loader.py`. They are in different package subtrees and currently independent; keeping the import local makes that explicit and cheap.)

Then update the two readers inside `settings.py`. Around line 365:
```python
            path = DEFAULT_SETTINGS_PATH
```
becomes:
```python
            path = default_settings_path()
```

Around line 367:
```python
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
```
becomes:
```python
    settings_path = Path(path) if path is not None else default_settings_path()
```

And the legacy-warning function at line 411-422 — replace line 413:
```python
    rag_settings = Path.home() / ".nanoresearch" / "rag" / "settings.yaml"
```
with:
```python
    from nanoresearch.config.loader import get_nanoresearch_home
    rag_settings = get_nanoresearch_home() / "rag" / "settings.yaml"
```

- [ ] **Step 2.3: Update the one external reader in `config/migration.py`**

Edit `backend/nanoresearch/config/migration.py` lines 82, 85. Replace:
```python
    from nanoresearch.rag.core.settings import DEFAULT_SETTINGS_PATH
    ...
    settings_path = settings_path or DEFAULT_SETTINGS_PATH
```
with:
```python
    from nanoresearch.rag.core.settings import default_settings_path
    ...
    settings_path = settings_path or default_settings_path()
```

- [ ] **Step 2.4: Run the new test; verify it passes**

```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py::test_default_settings_path_follows_nanoresearch_home -v
```
Expected: PASS.

- [ ] **Step 2.5: Fix `server/main.py:125`**

Edit `backend/nanoresearch/server/main.py`. Replace lines 123–126:
```python
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    rag_images_dir = Path.home() / ".nanoresearch" / "rag" / "images"
    rag_images_dir.mkdir(parents=True, exist_ok=True)
```
with:
```python
    from fastapi.staticfiles import StaticFiles
    from nanoresearch.config.loader import get_nanoresearch_home
    rag_images_dir = get_nanoresearch_home() / "rag" / "images"
    rag_images_dir.mkdir(parents=True, exist_ok=True)
```
(The now-redundant `from pathlib import Path` at line 123 can be dropped because the symbol isn't used elsewhere in this block. If `Path` is referenced later in `main.py`, leave the import; check `grep -n 'Path' backend/nanoresearch/server/main.py` before deciding.)

- [ ] **Step 2.6: Fix `server/routers/eval_router.py:1011`**

Edit `backend/nanoresearch/server/routers/eval_router.py`. Replace line 1011:
```python
    base: Path = cfg.get("base_workspace") or Path.home() / ".nanoresearch" / "workspace"
```
with:
```python
    from nanoresearch.config.loader import get_nanoresearch_home
    base: Path = cfg.get("base_workspace") or get_nanoresearch_home() / "workspace"
```
(Local import inside the function — this is a request-scoped path and matches the existing pattern of inline imports already used in this function block at lines 1001-1005.)

- [ ] **Step 2.7: Fix `backend/scripts/audit_duplicate_chunks.py:12`**

Edit `backend/scripts/audit_duplicate_chunks.py`. Replace lines 5–14:
```python
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import chromadb

CHROMA_PATH = Path.home() / ".nanoresearch" / "rag" / "chroma"
if not CHROMA_PATH.exists():
    sys.exit(f"ChromaDB not found at {CHROMA_PATH}")
```
with:
```python
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import chromadb

from nanoresearch.config.loader import get_nanoresearch_home

CHROMA_PATH = get_nanoresearch_home() / "rag" / "chroma"
if not CHROMA_PATH.exists():
    sys.exit(f"ChromaDB not found at {CHROMA_PATH}")
```
(`get_nanoresearch_home()` calls `apply_legacy_env_compat()` defensively, so this standalone script picks up `NANOBOT_HOME` correctly even without an explicit shim invocation.)

- [ ] **Step 2.8: Fix `backend/scripts/migrate_sessions.py:109`**

Edit `backend/scripts/migrate_sessions.py`. Replace line 109:
```python
        default=str(Path.home() / ".nanoresearch" / "workspace"),
```
with:
```python
        default=str(get_nanoresearch_home() / "workspace"),
```

Then add the import. At the top of the file, alongside `from pathlib import Path`, add:
```python
from nanoresearch.config.loader import get_nanoresearch_home
```

(If the existing imports are alphabetical or grouped stdlib-then-firstparty, place it with other `nanoresearch.*` imports. Check the existing top-of-file structure.)

- [ ] **Step 2.9: Verify `tests/config/test_config_paths.py` still all green**

```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py -v
```
Expected: all PASS (original suite + 7 new env-override tests cumulatively across Tasks 1 and 2).

- [ ] **Step 2.10: Smoke-test server import path under env override**

```bash
NANORESEARCH_HOME=/tmp/nr_phase4_task2 ./backend/.venv/Scripts/python -c "
from nanoresearch.server.main import create_app
app = create_app()
print('OK')
" 2>&1 | tail -5
```
Expected: prints `OK` (or a known-unrelated runtime warning followed by `OK`). If it crashes with `Path` undefined or import error, fix before committing.

- [ ] **Step 2.11: Sanity-check hit-count reduction**

```bash
grep -rn 'Path\.home\(\)\s*/\s*"\.nanoresearch"' backend/nanoresearch/ backend/scripts/
```
Expected: the only remaining hits are in `backend/nanoresearch/rag/mcp_server/tools/agentic/collections.py` (4 hits — handled in Task 3). All others must be gone.

- [ ] **Step 2.12: Commit Task 2**

```bash
git add backend/nanoresearch/rag/core/settings.py backend/nanoresearch/config/migration.py backend/nanoresearch/server/main.py backend/nanoresearch/server/routers/eval_router.py backend/scripts/audit_duplicate_chunks.py backend/scripts/migrate_sessions.py tests/config/test_config_paths.py
git status
git commit -m "$(cat <<'EOF'
feat(config): plumb NANORESEARCH_HOME through server, scripts, RAG settings

DEFAULT_SETTINGS_PATH (module-level constant frozen at import) replaced
with lazy default_settings_path() so env overrides take effect.
server/main.py rag-images dir, eval_router.py base workspace fallback,
two scripts/, and settings.yaml legacy-warning path all funnel through
get_nanoresearch_home(). config/migration.py reader updated for the
constant→function rename.

Out of scope: collections.py 4 sites (Task 3).
EOF
)"
git log -1 --stat
```

**Halt for review.** Do not start Task 3 without human approval of Commit B.

---

### Task 3: Bypass sites — `collections.py` (4 sites)

**Files:**
- Modify: `backend/nanoresearch/rag/mcp_server/tools/agentic/collections.py:595,666,793,854`

**Interfaces:**
- Consumes (from Task 1): `nanoresearch.config.loader.get_nanoresearch_home() -> Path`.

**Why a separate task:** This file is in the MCP server tool surface — agentic RAG ingest and retrieval flows. The 4 sites are mechanically identical to Task 2 but worth a separate review gate so a reviewer can independently confirm the MCP-tool layer is touched as expected. The four sites use slightly different naming conventions (`Path` at 595, 793, 854; `_Path` at 666), so each must be verified individually rather than search-replaced.

**Steps:**

- [ ] **Step 3.1: Fix line 595 (`config.json` lookup, ingest flow)**

Edit `backend/nanoresearch/rag/mcp_server/tools/agentic/collections.py`. In the function containing line 595 (around `_get_ks` → provider resolution), replace:
```python
            import json
            from pathlib import Path
            from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider

            config_path = Path.home() / ".nanoresearch" / "config.json"
```
with:
```python
            import json
            from pathlib import Path
            from nanoresearch.config.loader import get_nanoresearch_home
            from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider

            config_path = get_nanoresearch_home() / "config.json"
```

- [ ] **Step 3.2: Fix line 666 (`config.json` lookup, second flow with aliased imports)**

The site at line 666 aliases `Path as _Path` and `json as _json`. The aliasing is preserved; only the path construction changes. Replace:
```python
            import json as _json
            from pathlib import Path as _Path
            from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider

            config_path = _Path.home() / ".nanoresearch" / "config.json"
```
with:
```python
            import json as _json
            from pathlib import Path as _Path
            from nanoresearch.config.loader import get_nanoresearch_home as _get_nanoresearch_home
            from nanoresearch.providers.openai_compat_provider import OpenAICompatProvider

            config_path = _get_nanoresearch_home() / "config.json"
```

- [ ] **Step 3.3: Fix line 793 (relative file_path → workspace anchoring)**

Around lines 791–793:
```python
            path = Path(file_path)
            if not path.is_absolute():
                path = Path.home() / ".nanoresearch" / "workspace" / file_path
```
becomes:
```python
            path = Path(file_path)
            if not path.is_absolute():
                from nanoresearch.config.loader import get_nanoresearch_home
                path = get_nanoresearch_home() / "workspace" / file_path
```

- [ ] **Step 3.4: Fix line 854 (permanent document storage directory)**

Around line 854:
```python
            doc_dir = Path.home() / ".nanoresearch" / "rag" / "documents" / kb_id
            doc_dir.mkdir(parents=True, exist_ok=True)
```
becomes:
```python
            from nanoresearch.config.loader import get_nanoresearch_home
            doc_dir = get_nanoresearch_home() / "rag" / "documents" / kb_id
            doc_dir.mkdir(parents=True, exist_ok=True)
```

(All four sites use local imports of `get_nanoresearch_home` rather than a top-of-file import. Rationale: `collections.py` is large and the existing pattern in this file is to inline imports inside `try:` blocks for provider/tool integrations. Adding a top-level import would be cosmetically nicer but would diverge from the file's existing style. Reviewer can flag if they prefer a single top-level import — fold that into a follow-up clean-up if so.)

- [ ] **Step 3.5: Verify the file has zero `Path.home()` hits**

```bash
grep -nE 'Path\.home\(\)' backend/nanoresearch/rag/mcp_server/tools/agentic/collections.py
```
Expected: no output.

- [ ] **Step 3.6: Smoke-test the file imports cleanly**

```bash
./backend/.venv/Scripts/python -c "
from nanoresearch.rag.mcp_server.tools.agentic import collections
print('OK')
"
```
Expected: prints `OK`. If `ImportError` or other failure → fix before commit.

- [ ] **Step 3.7: Global hit-count verification**

```bash
grep -rn 'Path\.home\(\)\s*/\s*"\.nanoresearch"' backend/nanoresearch/ backend/scripts/
```
Expected: **exactly 1 hit**, on the line inside `backend/nanoresearch/config/loader.py`'s `get_nanoresearch_home()` reading `return Path.home() / ".nanoresearch"`. This is the intentional default-fallback. All 11 other production hits are now gone.

```bash
grep -rn 'Path\.home\(\)' backend/nanoresearch/
```
Expected: **exactly 1 hit**, the same fallback line in `loader.py:get_nanoresearch_home()`. The whole `nanoresearch/` tree is otherwise fully env-driven.

- [ ] **Step 3.8: Final test pass**

```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py -v
```
Expected: all PASS (no regression from Task 3 since collections.py has no dedicated unit test in this suite).

- [ ] **Step 3.9: Commit Task 3**

```bash
git add backend/nanoresearch/rag/mcp_server/tools/agentic/collections.py
git status
git commit -m "$(cat <<'EOF'
feat(config): plumb NANORESEARCH_HOME through MCP collections tools

4 sites in rag/mcp_server/tools/agentic/collections.py — config.json
lookups (×2), workspace-relative file resolution, and permanent doc
storage dir — all switched to get_nanoresearch_home(). Local imports
preserved to match the existing inline-import style in this file.

Phase 4 production code now has zero hard-coded ~/.nanoresearch.
EOF
)"
git log -1 --stat
```

**Halt for review.** Do not start Task 4 without human approval of Commit C.

---

### Task 4: Documentation

**Files:**
- Modify: `README.md` (add NANORESEARCH_HOME paragraph in Configuration section)
- Modify: `SECURITY.md` (note env var in file-permission guidance)
- Modify: `docker-compose.yml` (add a commented-out example of the env override + volume swap)

**Interfaces:** None — docs only.

**Why included in Phase 4:** Without docs, multi-tenant operators won't discover the env var. The docker-compose mention is especially important — the current `~/.nanoresearch:/root/.nanoresearch` volume binding will not work if NANORESEARCH_HOME points elsewhere inside the container.

**Steps:**

- [ ] **Step 4.1: Add an env-var section to `README.md`**

Locate the Configuration section (around line 189 onward, where it documents `~/.nanoresearch/config.json`). Append a new subsection after that paragraph:

```markdown
### Custom base directory (multi-tenant / containerized deployments)

By default, NanoResearch stores all runtime state under `~/.nanoresearch`.
Set the `NANORESEARCH_HOME` environment variable to relocate this base
directory — for example to support multiple tenants on a single host or
to mount a non-home volume inside a container:

```bash
export NANORESEARCH_HOME=/data/tenant_alice
nr serve
```

Tilde expansion (`~/custom-root`) and absolute paths are both supported.
The directory is created automatically on first write. The legacy
`NANOBOT_HOME` environment variable is also accepted for backward
compatibility and will be removed in v0.3.0.
```

- [ ] **Step 4.2: Update `SECURITY.md` to mention env-var paths**

Locate the file-permission guidance section (around line 145–149). Replace the block:
```
   chmod 700 ~/.nanoresearch
   chmod 600 ~/.nanoresearch/config.json
   chmod 700 ~/.nanoresearch/whatsapp-auth
```
with:
```
   # If you have set NANORESEARCH_HOME, substitute its value for ~/.nanoresearch below.
   chmod 700 ~/.nanoresearch
   chmod 600 ~/.nanoresearch/config.json
   chmod 700 ~/.nanoresearch/whatsapp-auth
```

- [ ] **Step 4.3: Add a commented multi-tenant example to `docker-compose.yml`**

After the existing `~/.nanoresearch:/root/.nanoresearch` volume binding (around line 30 and line 58), append a commented block:

```yaml
    # Multi-tenant / custom-base example:
    #   environment:
    #     NANORESEARCH_HOME: /data/tenant_x
    #   volumes:
    #     - ./tenant_x:/data/tenant_x
```

(Place once near line 30 in the first service block. The second binding at line 58 is the same comment — DRY: cross-reference rather than duplicating the block.)

- [ ] **Step 4.4: Verify Markdown renders and YAML still parses**

```bash
./backend/.venv/Scripts/python -c "
import yaml
with open('docker-compose.yml') as f:
    yaml.safe_load(f)
print('docker-compose.yml: valid YAML')
"
```
Expected: prints `docker-compose.yml: valid YAML`.

For README/SECURITY, open the files in an editor or rely on the PR-renderer; no automated check required.

- [ ] **Step 4.5: Commit Task 4**

```bash
git add README.md SECURITY.md docker-compose.yml
git status
git commit -m "$(cat <<'EOF'
docs(config): document NANORESEARCH_HOME env var

README: new subsection covering tilde/absolute paths, NANOBOT_HOME
deprecation. SECURITY: note that chmod paths follow the env var.
docker-compose: commented multi-tenant example next to the default
volume binding.
EOF
)"
git log -1 --stat
```

**Halt for review.** Phase 4 complete. Do not push without human approval.

---

## End-of-Phase Verification (run after Task 4 commit, before push)

- [ ] **EV.1: Production `Path.home()` references reduced to the single intentional fallback**

```bash
grep -rn 'Path\.home\(\)' backend/nanoresearch/ backend/scripts/
```
Expected: **exactly 1 hit**, on the line inside `backend/nanoresearch/config/loader.py`'s `get_nanoresearch_home()` reading `return Path.home() / ".nanoresearch"` — the intentional default when `NANORESEARCH_HOME` is unset. All other 11 production hits gone. (Tests, dev-scratch files at `backend/test_themes_diagnostic.py` and `backend/test_ragas_transforms.py`, and docs/specs may still contain literal references — those are intentionally out of scope per Scope Decisions section.)

- [ ] **EV.2: Targeted pytest suite green**

```bash
./backend/.venv/Scripts/python -m pytest tests/config/test_config_paths.py -v
```
Expected: 100% PASS.

- [ ] **EV.3: Multi-tenant smoke**

```bash
NANORESEARCH_HOME=/tmp/nr_phase4_final ./backend/.venv/Scripts/python -c "
from nanoresearch.config.paths import get_workspace_path, get_cli_history_path
from nanoresearch.config.loader import get_config_path, get_nanoresearch_home
from nanoresearch.rag.core.settings import default_settings_path
print('home          ', get_nanoresearch_home())
print('config        ', get_config_path())
print('workspace     ', get_workspace_path())
print('cli_history   ', get_cli_history_path())
print('rag_settings  ', default_settings_path())
"
```
Expected: every printed path is rooted at `/tmp/nr_phase4_final`.

- [ ] **EV.4: Legacy `NANOBOT_HOME` smoke**

```bash
NANOBOT_HOME=/tmp/nr_phase4_legacy ./backend/.venv/Scripts/python -W default -c "
import warnings
warnings.simplefilter('always', DeprecationWarning)
from nanoresearch.config.loader import get_nanoresearch_home
print(get_nanoresearch_home())
" 2>&1
```
Expected: prints `/tmp/nr_phase4_legacy`. At least one DeprecationWarning naming both env vars is emitted to stderr.

- [ ] **EV.5: Server entry import smoke**

```bash
./backend/.venv/Scripts/python -c "
from nanoresearch.server.main import create_app
from nanoresearch.cli import commands  # noqa: F401
print('OK')
"
```
Expected: prints `OK`.

- [ ] **EV.6: Push (only after human approval)**

```bash
git log --oneline origin/main..HEAD
git push origin main
```
Expected: 4 commits pushed (A: helper+paths, B: server/scripts/settings, C: collections.py, D: docs). On any push failure, **stop and report verbatim**. Do not retry with `--force`, do not amend, do not bypass hooks.

## Reverse-Grep Allowlist

If `EV.1` shows hits, only the following are acceptable and may be ignored:
- `backend/test_themes_diagnostic.py`, `backend/test_ragas_transforms.py` — local dev scratch with hardcoded `C:/Users/Augix/...`, intentionally out of scope.
- `docs/superpowers/specs/2026-06-26-repo-cleanup-design.md`, `docs/superpowers/plans/2026-06-27-repo-cleanup-phase3.md` — historical documentation, not code.
- `tests/config/test_config_paths.py` — intentional comparisons against `Path.home() / ".nanoresearch"` to verify default behaviour.
- `backend/tests/eval/test_ragas_e2e.py` — already updated in Task 1.12 to use `get_nanoresearch_home()`, but the file is a standalone script not collected by pytest.

Any other hit means a site was missed — return to the appropriate task and add it.
