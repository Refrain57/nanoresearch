-- Migration: document new proposal status value 'signal_unreliable' (B2/B6 — Phase 1 of A1).
-- The status column is a plain VARCHAR with no enum constraint, so no DDL change is required.
-- This file exists as a placeholder / changelog marker — the value is enforced in application code.

-- (Intentionally empty — see backend/nanoresearch/storage/models.py OptimizationProposal.status column comment.)
