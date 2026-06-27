ALTER TABLE kb_documents
    ADD COLUMN IF NOT EXISTS pdf_parser VARCHAR DEFAULT 'marker';
