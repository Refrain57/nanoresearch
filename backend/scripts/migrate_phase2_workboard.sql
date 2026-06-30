-- Phase 2 serial-MVP: multi-main workboard tables.
-- Idempotent; run against the production DB before deploying Phase 2.
-- Tables: conversation_agents (Task 2), workboard_cards (Task 3), workboard_card_links (Task 4).

-- Task 2: Conversation ↔ Agent membership (additive to conversations.agent_id).
CREATE TABLE IF NOT EXISTS conversation_agents (
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    agent_id        UUID        NOT NULL REFERENCES agents(id)        ON DELETE CASCADE,
    role            VARCHAR     NOT NULL DEFAULT 'main',
    activated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, agent_id)
);
