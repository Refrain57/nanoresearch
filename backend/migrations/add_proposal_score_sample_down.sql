-- Down-migration: drop score_sample column.
ALTER TABLE optimization_proposals
    DROP COLUMN IF EXISTS score_sample;
