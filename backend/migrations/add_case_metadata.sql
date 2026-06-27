-- Migration: add B4 metadata fields to agent_test_cases
-- Phase 1 of A1 tool-layer harness refactor.
-- Adds origin_badcase_id, target_dimension, added_at, added_by, coverage_tags.
-- Backfill required before NOT NULL constraint is added — run scripts/backfill_case_metadata.py --apply.

ALTER TABLE agent_test_cases
    ADD COLUMN IF NOT EXISTS origin_badcase_id UUID REFERENCES agent_run_snapshots(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS target_dimension TEXT,
    ADD COLUMN IF NOT EXISTS added_at TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS added_by TEXT,
    ADD COLUMN IF NOT EXISTS coverage_tags TEXT[];

CREATE INDEX IF NOT EXISTS idx_agent_test_cases_target_dimension ON agent_test_cases(target_dimension);
CREATE INDEX IF NOT EXISTS idx_agent_test_cases_origin_badcase_id ON agent_test_cases(origin_badcase_id);

-- NOT NULL constraints applied in a second migration AFTER backfill — see add_case_metadata_enforce.sql.
