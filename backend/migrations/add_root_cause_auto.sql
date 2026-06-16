-- Add auto root-cause classification columns to agent_run_snapshots
ALTER TABLE agent_run_snapshots
  ADD COLUMN IF NOT EXISTS root_cause_auto VARCHAR(32),
  ADD COLUMN IF NOT EXISTS root_cause_auto_confidence VARCHAR(16),
  ADD COLUMN IF NOT EXISTS root_cause_auto_reason VARCHAR(500);
