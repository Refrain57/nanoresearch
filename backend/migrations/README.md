# Migrations

Raw PostgreSQL SQL migrations. Not Alembic. Applied manually via `psql` against the target database.

## Convention (in force from A1 Phase 1, 2026-06-26)

Every NEW migration ships as a pair:

- `<descriptive_name>.sql` — the up migration (idempotent: use `IF NOT EXISTS` / `IF EXISTS`)
- `<descriptive_name>_down.sql` — the down migration (idempotent: use `IF EXISTS`)

Pre-2026-06-26 migrations do NOT have down companions and are NOT retrofitted.

## Application order

Migrations are applied in alphabetical order of filename. Use a numeric prefix only if ordering matters: `001_<name>.sql`, `002_<name>.sql`. Otherwise descriptive names are fine.

## Two-phase column additions

When adding a NOT NULL column to an existing populated table, use TWO migrations:

1. `add_<name>.sql` — adds the column as nullable
2. After running the backfill script, `add_<name>_enforce.sql` — applies the NOT NULL constraint

The backfill script lives in `backend/scripts/backfill_<name>.py`.
