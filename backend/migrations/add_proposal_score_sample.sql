-- Migration: add score_sample column to optimization_proposals for B2 σ-weighted gate.
-- Stores per-case repeat statistics: {case_id: {"mean": float, "std": float, "n": int}}.

ALTER TABLE optimization_proposals
    ADD COLUMN IF NOT EXISTS score_sample JSONB DEFAULT NULL;
