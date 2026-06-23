-- Migration: Phase 0 — context trace column
-- Adds structured context assembly decisions to agent_run_snapshots.
-- Run once against the target database before deploying the matching code.
--
-- UP
ALTER TABLE agent_run_snapshots
    ADD COLUMN IF NOT EXISTS context_trace JSONB DEFAULT NULL;

-- DOWN (to undo this migration)
-- ALTER TABLE agent_run_snapshots DROP COLUMN IF EXISTS context_trace;
