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

-- Task 3: workboard cards (state machine backlog→todo→ready→running→done|blocked).
CREATE TABLE IF NOT EXISTS workboard_cards (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    title               VARCHAR     NOT NULL,
    spec                TEXT,
    status              VARCHAR     NOT NULL DEFAULT 'backlog',
    owner_agent_id      UUID        REFERENCES agents(id) ON DELETE SET NULL,
    target_agent_id     UUID        REFERENCES agents(id) ON DELETE SET NULL,
    created_by_agent_id UUID        REFERENCES agents(id) ON DELETE SET NULL,
    claim_token         VARCHAR,
    claimed_at          TIMESTAMPTZ,
    heartbeat_at        TIMESTAMPTZ,
    artifacts           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    result              TEXT,
    depth               INTEGER     NOT NULL DEFAULT 0,
    collected           BOOLEAN     NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_workboard_cards_conversation ON workboard_cards (conversation_id);
CREATE INDEX IF NOT EXISTS idx_workboard_cards_status ON workboard_cards (status);

-- Task 1 (collab layer): pass_count tracks how many agents have "passed" on this card.
ALTER TABLE workboard_cards ADD COLUMN IF NOT EXISTS pass_count INTEGER NOT NULL DEFAULT 0;

-- Task 4: parent → child dependency (child waits in todo until all parents are done).
CREATE TABLE IF NOT EXISTS workboard_card_links (
    parent_card_id UUID NOT NULL REFERENCES workboard_cards(id) ON DELETE CASCADE,
    child_card_id  UUID NOT NULL REFERENCES workboard_cards(id) ON DELETE CASCADE,
    PRIMARY KEY (parent_card_id, child_card_id)
);
