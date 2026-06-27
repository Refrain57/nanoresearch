-- Down-migration: drop NOT NULL constraints from B4 metadata fields.
ALTER TABLE agent_test_cases
    ALTER COLUMN coverage_tags DROP NOT NULL,
    ALTER COLUMN added_by DROP NOT NULL,
    ALTER COLUMN added_at DROP NOT NULL,
    ALTER COLUMN target_dimension DROP NOT NULL;
