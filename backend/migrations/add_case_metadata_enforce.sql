-- Migration: enforce NOT NULL on B4 metadata fields after backfill.
-- Run AFTER scripts/backfill_case_metadata.py --apply has populated all rows.
-- Verifies no NULLs remain before applying constraint (will error otherwise — safe).

ALTER TABLE agent_test_cases
    ALTER COLUMN target_dimension SET NOT NULL,
    ALTER COLUMN added_at SET NOT NULL,
    ALTER COLUMN added_by SET NOT NULL,
    ALTER COLUMN coverage_tags SET NOT NULL;
