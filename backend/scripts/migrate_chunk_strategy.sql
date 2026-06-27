ALTER TABLE knowledge_bases
    ADD COLUMN IF NOT EXISTS chunk_strategy VARCHAR DEFAULT 'auto';
