-- Down-migration: revert add_case_metadata.sql
-- Drops the five B4 metadata columns and their indexes.
-- IRREVERSIBLE FOR DATA: any populated metadata is lost.

DROP INDEX IF EXISTS idx_agent_test_cases_origin_badcase_id;
DROP INDEX IF EXISTS idx_agent_test_cases_target_dimension;

ALTER TABLE agent_test_cases
    DROP COLUMN IF EXISTS coverage_tags,
    DROP COLUMN IF EXISTS added_by,
    DROP COLUMN IF EXISTS added_at,
    DROP COLUMN IF EXISTS target_dimension,
    DROP COLUMN IF EXISTS origin_badcase_id;
