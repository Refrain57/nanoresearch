# F1 — 前端 Skill 市场 / ClawHub 安装 UI

**Date:** 2026-07-02
**Status:** Design approved, pending spec review
**Branch:** `worktree-feat+skill-marketplace-ui`

## Problem

The backend already integrates ClawHub (clawhub.ai — a public skill registry, "npm for AI agents"), but only through the built-in `clawhub` *agent* skill: the agent shells out to `npx clawhub@latest search/install/...`. There is **no web UI**. In the frontend, users can only attach skills that are *already in the pool* to an agent (`AgentDetailView.vue` → 「添加 Skill」modal, fed by `GET /api/skills`); they cannot discover or install new skills.

F1 adds a marketplace + install flow to the frontend so users can search ClawHub, preview a skill, install it into their own workspace (where it enters the available pool), and remove workspace-installed skills.

## Scope

**In scope**
- Search ClawHub and browse results.
- Preview a skill (rendered SKILL.md + trust signals) before installing.
- Install a skill into the *caller's own* workspace (`users/{uid}/skills/`).
- List workspace-installed skills and remove them.
- Fix `GET /api/skills` so it scans the per-user workspace (today it scans the built-in dir, so installed skills never appear in the pool).

**Out of scope (explicitly deferred)**
- Updating installed skills (`clawhub update`).
- Authoring a new skill from scratch (skill-creator UI).
- Cleaning up dangling `skills_config` entries after an uninstall (graceful-skip is enough for MVP; cleanup is later polish).

## Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| First-version scope | Search + install **and** list + remove installed |
| Trust model | **Self-serve + mandatory preview/confirm.** Any logged-in user installs into their own workspace only. Install requires viewing the SKILL.md + trust signals first. |
| UI placement | **Inside the agent「添加 Skill」modal** (no new nav route). Remove lives on workspace-source items inside the 可用池 tab. |
| Read ops (search/detail/preview) | Backend **proxies** the ClawHub HTTP API. |
| Install / remove | Backend shells out to the `clawhub` **CLI** (`npx clawhub@latest`). |
| Installed/pool listing | **Filesystem scan** of the user's `workspace/skills` via `SkillsLoader`. |
| Server prereq | Node/`npx` reachable from the server process (already assumed by the existing `clawhub` agent skill). Pure-Python install is the documented fallback if this is not acceptable. |

## ClawHub interface (verified 2026-07-02)

HTTP API — base `https://clawhub.ai`, overridable via `CLAWHUB_REGISTRY`; search/detail/file are **public, no auth**:
- `GET /api/v1/search?q=<query>&limit=<n>` → results: `slug, displayName, summary, version, updatedAt, ownerHandle, score, owner{handle, displayName, image}`.
- `GET /api/v1/skills/{slug}` → `slug, displayName, summary, topics, tags, stats, createdAt, updatedAt, latestVersion{version, changelog}, metadata{OS restrictions, Nix systems}, owner, moderation`.
- `GET /api/v1/skills/{slug}/file?path=SKILL.md` → raw file text (200 KB limit; defaults to latest version).
- `GET /api/v1/skills/{slug}/versions/{version}` → version metadata + files list.
- `GET /api/v1/download?slug=<slug>&tag=latest` → hosted-ZIP bytes **or** a "GitHub source-handoff" (metadata pointing at GitHub) for GitHub-backed skills without a hosted version.

CLI (`npx clawhub@latest`):
- `search <query> [--limit n]` — **no `--json`** (human output only) → we do not use the CLI for search.
- `install @owner/slug --workdir <dir>` — downloads ZIP via the API and extracts to `<workdir>/skills/<slug>`, writes a lockfile. Handles both hosted-ZIP and GitHub source-handoff. **No documented post-install scripts** (install is file extraction, not `npm postinstall`).
- `uninstall <skill> --yes --workdir <dir>` — removes the skill directory + lockfile entry.

**Why CLI for install (not pure-Python):** install must handle hosted-ZIP *and* GitHub source-handoff *and* the lockfile. The CLI already does all three; reimplementing them in Python is fragile. Read ops use the HTTP API directly because it returns clean JSON and the CLI `search` has no machine-readable output.

## Existing code touchpoints

- Pool endpoint (to fix): `backend/nanoresearch/server/routers/agent_router.py:61` `list_skills` — currently `SkillsLoader(workspace=BUILTIN_SKILLS_DIR.parent, ...)`; must use the authenticated user's workspace.
- Per-user workspace helper (to reuse): `backend/nanoresearch/server/routers/workspace_router.py:23` `_user_workspace(request, uid)` and `:32` `_safe_resolve` (path-traversal guard).
- Skills loader: `backend/nanoresearch/agent/skills.py` `SkillsLoader.list_skills / get_skill_metadata`.
- Auth dependency: `backend/nanoresearch/server/middleware/auth.py` `get_current_user`.
- Router registration: `backend/nanoresearch/server/main.py`.
- Front-end API layer: `web/src/apis/base.js` (`apiGet/apiPost/apiDelete`), `web/src/apis/agents.js` (`listSkills`).
- Modal + skill logic: `web/src/views/AgentDetailView.vue` (「添加 Skill」modal `:205`, `skillsToAdd` `:286`, `addSkill` `:352`, `toggleSkill` `:343`).
- Agent skill state: `web/src/stores/agent.js` (`skills`).

## Architecture

```
Browser (AgentDetailView 「添加 Skill」modal)
  ├─ Tab 可用池        ── GET  /api/skills                 (per-user workspace scan)
  │    └─ workspace item ── DELETE /api/skills/{name}       (uninstall via CLI)
  └─ Tab 从市场安装
       ├─ search       ── GET  /api/skills/market/search    ─┐
       ├─ preview      ── GET  /api/skills/market/{slug}      ├─ backend proxies clawhub.ai/api/v1
       │                  GET  /api/skills/market/{slug}/readme ┘
       └─ install      ── POST /api/skills/install            (install via CLI into users/{uid})
```

- `uid` is always taken from the JWT (`get_current_user`) — never a request parameter. A user can only read/modify their own workspace.
- The installed pool is the filesystem (`users/{uid}/skills/`), so it reflects reality regardless of how a skill got there.

## Backend API (new `skill_market_router.py`; register in `main.py`)

All endpoints require `get_current_user`.

| Method + path | Body / query | Returns | Notes |
|---|---|---|---|
| `GET /api/skills/market/search` | `q`, `limit` | `[{slug, name, summary, version, owner, stars, score, moderation}]` | Proxy `/api/v1/search`. `502` if ClawHub unreachable. |
| `GET /api/skills/market/{slug}` | — | metadata + trust signals (`moderation, stats, latestVersion{version,changelog}, metadata{os/nix}, files`) | Proxy `/api/v1/skills/{slug}` (+ versions for the file list). |
| `GET /api/skills/market/{slug}/readme` | — | `{content}` (raw SKILL.md text) | Proxy `/api/v1/skills/{slug}/file?path=SKILL.md`. |
| `POST /api/skills/install` | `{slug}` | installed skill info `{name, description, source}` | `npx clawhub@latest install <slug> --workdir <users/{uid}>`. Idempotent for an already-installed slug. |
| `DELETE /api/skills/{name}` | — | `204` | Uninstall from caller's workspace. Reject built-in skills (`404`/`403`). Path-guarded to `users/{uid}/skills/`. |
| `GET /api/skills` *(fix existing)* | — | pool `[{name, description, source}]` | Scan `users/{uid}/skills` + built-in; move from `agent_router` behavior to a per-user workspace scan. |

Backend uses `httpx` (already a project dependency, `>=0.28.0`, widely used across `nanoresearch`) for the proxy calls, and `asyncio.create_subprocess_exec` for the CLI calls with a timeout.

## Frontend

- **`web/src/apis/skills.js`** (new): `searchMarket(q, limit)`, `getMarketSkill(slug)`, `getMarketReadme(slug)`, `installSkill(slug)`, `uninstallSkill(name)`, `listSkills()` (move/alias from `agents.js`).
- **`AgentDetailView.vue`「添加 Skill」modal → `a-tabs`:**
  - **Tab 可用池** — existing pick-to-attach behavior (`addSkill`). Each `source === 'workspace'` item also shows a small 「卸载」 action (`a-popconfirm` → `uninstallSkill` → refresh pool). Built-in items have no uninstall. This tab is the installed-skills manager for workspace skills.
  - **Tab 从市场安装** — search input → result cards (name, owner handle, ★ stars, moderation badge, version). Each card: 「预览」opens a preview surface; 「安装」is only reachable after preview.
- **Preview surface** (`a-drawer` or nested `a-modal`): rendered SKILL.md + trust signals (owner, moderation state, stars/downloads, version + changelog, OS/Nix restrictions, **file list with a warning badge if the skill bundles files beyond SKILL.md, i.e. scripts**). Contains the 「安装」button.
- **Post-install:** success toast, refresh the pool so the skill appears under 可用池, and remind the user that workspace skills load on a **new session** (per the `clawhub` SKILL.md note).
- Components use Ant Design Vue + the existing Anthropic warm-clay theme (`web/src/styles/theme.css`, `App.vue` tokens).

## Security / trust UX

- **Mandatory preview before install** (per the self-serve + preview/confirm decision): the 「安装」button lives inside the preview surface, so the user always sees SKILL.md + trust signals first.
- **Moderation gating:** if ClawHub `moderation` is flagged/removed, disable install and show a warning.
- **Script warning:** if the version's file list contains files other than `SKILL.md` (e.g. `scripts/`), show a clear "this skill bundles executable files" warning in the preview.
- **Own-workspace only:** install/uninstall always target `users/{uid}/skills/` derived from the JWT; blast radius is the installing user's own workspace (per-user isolation already enforced elsewhere).
- **Path safety:** `DELETE /api/skills/{name}` resolves within `users/{uid}/skills/` using the same guard pattern as `workspace_router._safe_resolve`; the built-in skills directory is never writable via the API.

## Edge cases & error handling

- ClawHub unreachable / non-200 from proxy → `502` with a friendly message (「技能市场暂时不可用」).
- `npx`/Node missing on the server → clear error (「服务器未安装 Node/clawhub CLI」); surfaced as a toast.
- Install of an already-installed slug → idempotent; report「已安装」.
- Uninstall of a skill currently bound to one or more agents → popconfirm warns it is in use, then allows. Dangling `skills_config` entries are skipped gracefully (`SkillsLoader.load_skill` returns `None` for a missing skill; agent context already tolerates this — verify during implementation).
- GitHub source-handoff skills that need `git`/network the server lacks → surfaced as an install error from the CLI's stderr.
- `npx` cold start can add seconds to the first install; install endpoint is synchronous with a spinner in the UI and a generous subprocess timeout. (Optional later optimization: pre-install `clawhub` globally.)

## Testing

**Backend (pytest)**
- Market proxy endpoints with the ClawHub HTTP client mocked (search / detail / readme), including the `502`-on-unreachable path.
- Install / uninstall with the subprocess mocked: correct `--workdir` (per-user), idempotent install, built-in-skill uninstall rejected.
- Path-traversal guard on `DELETE /api/skills/{name}`.
- `GET /api/skills` scans the per-user workspace (a skill placed in `users/{uid}/skills/` appears; another user's does not).

**Frontend**
- Modal-tabs render; install button is disabled until the preview is opened and moderation is acceptable.
- Manual E2E: search → preview → install → skill appears in 可用池 → attach to agent → remove.

## Implementation order (for the plan)

1. Backend: fix `GET /api/skills` to per-user scan (+ test).
2. Backend: `skill_market_router` proxy endpoints (+ tests, HTTP mocked).
3. Backend: install/uninstall endpoints (+ tests, subprocess mocked).
4. Frontend: `apis/skills.js`.
5. Frontend: 从市场安装 tab + preview surface.
6. Frontend: 可用池 tab uninstall affordance.
7. Wire-up, manual E2E, polish (post-install session reminder, error toasts).
