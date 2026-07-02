-- add_memory_facts.sql — 画像 provenance store (memory layering P1)
-- Source of truth for the user profile; MEMORY.md is a one-way rendered projection.
-- Idempotent.

CREATE TABLE IF NOT EXISTS memory_facts (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uid          VARCHAR NOT NULL,
    section      VARCHAR NOT NULL,                 -- facts | user_profile | focus_areas
    text         TEXT NOT NULL,
    source       VARCHAR DEFAULT 'extracted',      -- extracted | manual
    derived_from JSONB DEFAULT '[]'::jsonb,         -- event ids (provenance, P2/C4)
    confidence   DOUBLE PRECISION,
    edited_by    VARCHAR,                          -- set for source=manual
    edited_at    TIMESTAMPTZ,                      -- set for source=manual
    active       BOOLEAN DEFAULT TRUE,             -- soft-delete (diff removal deactivates)
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_memory_facts_uid    ON memory_facts (uid);
CREATE INDEX IF NOT EXISTS ix_memory_facts_active ON memory_facts (active);
