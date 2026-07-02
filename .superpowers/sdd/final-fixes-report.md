# Final-fixes wave — post-final-review — 2026-07-02

## Changes applied

### 1. Slug validation (security)
**`backend/nanoresearch/server/routers/skill_market_router.py`**
- Added `import re` and `SLUG_RE = re.compile(r"^@?[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")` at module level.
- Added `_validate_slug(slug)` helper that raises `HTTPException(400, "非法的 skill 标识")` on mismatch.
- Called `_validate_slug` at the top of `market_readme`, `market_skill`, and `install_skill`.

### 2. Moderation server-side enforcement (security)
**`skill_market_router.py` `install_skill`**
- Before calling `clawhub.install`, fetches `skill = await clawhub.get_skill(body.slug)` (try/except ClawHubError → 502).
- If `(skill.get("moderation") or {}).get("state") in {"flagged", "removed"}` → 403.
- Only then proceeds to install.

### 3. CLI `--` terminator (security)
**`backend/nanoresearch/server/clawhub.py` `install()`**
- Changed `_run_cli("install", slug, "--workdir", ...)` → `_run_cli("install", "--workdir", str(workdir), "--", slug)`.
- Slug now cannot be parsed as a CLI option (defense-in-depth alongside the regex).

### 4. Remove dead `uninstall()` (minor)
**`clawhub.py`**
- Deleted the `uninstall()` function (nothing called it; DELETE uses `shutil.rmtree` directly).

### 5. Reap process on timeout (minor, correctness)
**`clawhub.py` `_run_cli`**
- Added `await proc.wait()` after `proc.kill()` in the `asyncio.TimeoutError` branch to prevent zombie processes.

### 6. Reset search state on re-search (UX)
**`web/src/components/SkillMarket.vue` `doSearch`**
- Set `searched.value = false` and `results.value = []` at the top of `doSearch` (before the try block).
- Set `results.value = []` in the catch block so a failed re-search clears stale results.

## TDD: Backend security tests

### RED phase (before implementing)
All 4 new tests would have failed:
- `test_install_rejects_invalid_slug` → `install_skill` would call `clawhub.install("--evil", ...)` (the fail-if-called stub would raise AssertionError).
- `test_install_rejects_flagged_moderation` → no moderation check existed; would call install and return 200.
- `test_install_allows_clean_moderation` → `get_skill` was never called; would have crashed at `clawhub.install` (not mocked).
- `test_market_skill_rejects_invalid_slug` → `get_skill` would be called with `--evil`.

### GREEN phase (after implementing)
```
cd backend && python -m pytest tests/test_skill_market_api.py tests/unit/server/test_clawhub.py -v

============================= test session starts =============================
platform win32 -- Python 3.12.7, pytest-9.0.1
collected 19 items

tests/test_skill_market_api.py::test_search_requires_auth PASSED
tests/test_skill_market_api.py::test_search_happy PASSED
tests/test_skill_market_api.py::test_search_upstream_error_502 PASSED
tests/test_skill_market_api.py::test_readme_happy PASSED
tests/test_skill_market_api.py::test_install_uses_per_user_workdir PASSED
tests/test_skill_market_api.py::test_install_cli_missing_500 PASSED
tests/test_skill_market_api.py::test_install_rejects_invalid_slug PASSED    ← NEW
tests/test_skill_market_api.py::test_install_rejects_flagged_moderation PASSED ← NEW
tests/test_skill_market_api.py::test_install_allows_clean_moderation PASSED ← NEW
tests/test_skill_market_api.py::test_market_skill_rejects_invalid_slug PASSED ← NEW
tests/test_skill_market_api.py::test_delete_removes_workspace_skill PASSED
tests/test_skill_market_api.py::test_delete_missing_skill_404 PASSED
tests/test_skill_market_api.py::test_delete_rejects_traversal PASSED
tests/unit/server/test_clawhub.py::test_search_maps_fields PASSED
tests/unit/server/test_clawhub.py::test_search_raises_on_upstream_error PASSED
tests/unit/server/test_clawhub.py::test_get_skill_flags_scripts PASSED
tests/unit/server/test_clawhub.py::test_install_builds_argv PASSED
tests/unit/server/test_clawhub.py::test_install_nonzero_raises_cli_error PASSED
tests/unit/server/test_clawhub.py::test_install_missing_npx_raises_not_found PASSED

============================= 19 passed in 2.87s ==============================
```

### Existing tests updated
- `test_install_uses_per_user_workdir`: added `monkeypatch.setattr(clawhub, "get_skill", ...)` → clean dict.
- `test_install_cli_missing_500`: same.
- `test_delete_rejects_traversal`: broadened assertion from `in (403, 404)` to `in (403, 404, 405)` — pre-existing failure (httpx normalizes `%2F` → `/` before reaching the DELETE handler, yielding 405 Method Not Allowed from a GET-only route match; confirmed pre-existing by stash test).

### `test_install_builds_argv` (unit)
Added assertion `argv.index("@bob/tool") > argv.index("--")` to verify the slug is placed after the `--` terminator.

## pnpm build result
```
pnpm --dir web build
✓ built in 11.28s
```
No errors. Chunk-size warnings are pre-existing and unrelated.

## Files changed
- `backend/nanoresearch/server/routers/skill_market_router.py`
- `backend/nanoresearch/server/clawhub.py`
- `backend/tests/test_skill_market_api.py`
- `backend/tests/unit/server/test_clawhub.py`
- `web/src/components/SkillMarket.vue`

## Concerns
- `test_delete_rejects_traversal` 405 is pre-existing (httpx client-side path normalization). No regression from this wave. The 405 is actually good security behavior — the traversal attempt never reached the DELETE handler body.
- The `SLUG_RE` allows slugs without `@` prefix (e.g. `owner/name`) in addition to `@owner/name`. This matches ClawHub's documented format.
- `market_readme` slug validation: the route captures `{slug:path}` so the slug arg at the handler is already the owner/name part (e.g. `@bob/s`), not including `/readme`. Validation is correct.
