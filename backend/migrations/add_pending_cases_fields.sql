-- Migration: add flywheel fields to agent_test_cases
-- Adds status, generated_from_snapshot_id, and generation_reason to support
-- the data flywheel pending-review queue.

ALTER TABLE agent_test_cases
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS generated_from_snapshot_id UUID REFERENCES agent_run_snapshots(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS generation_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_agent_test_cases_status ON agent_test_cases(status);
