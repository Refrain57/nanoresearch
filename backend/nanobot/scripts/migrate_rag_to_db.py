"""Migrate existing ChromaDB knowledge base into PostgreSQL tables.

Usage:
    python -m nanobot.scripts.migrate_rag_to_db [--collection default] [--kb-name "默认知识库"] [--uid admin] [--dry-run]

What it does:
  1. Reads ingestion_history.db for all successfully processed files
  2. Reads all chunks from ChromaDB (grouped by source_path)
  3. Creates one KnowledgeBase row per collection
  4. Creates one KbDocument row per source file
  5. Creates KbChunk rows for all chunks

Idempotent: skips if KnowledgeBase with same name already exists (use --force to overwrite).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path


def _get_doc_hash(source_path: str) -> str:
    """Replicate VectorUpserter's source_hash: sha256(source_path)[:8]."""
    return hashlib.sha256(source_path.encode()).hexdigest()[:8]


async def migrate(
    collection: str = "default",
    kb_name: str = "默认知识库",
    uid: str = "admin",
    dry_run: bool = False,
    force: bool = False,
) -> None:
    # --- init DB ---
    from nanobot.storage.database import init_engine, get_session_factory, init_db
    from nanobot.storage.repositories.knowledge_repo import KnowledgeRepository
    from nanobot.storage.models import KbChunk

    init_engine()
    await init_db()
    factory = get_session_factory()
    repo = KnowledgeRepository(factory)

    # --- check existing KB ---
    existing_kbs = await repo.list_by_uid(uid)
    existing = next((k for k in existing_kbs if k.name == kb_name), None)
    if existing and not force:
        print(f"[skip] KnowledgeBase '{kb_name}' already exists (id={existing.id}). Use --force to re-migrate.")
        return
    if existing and force:
        print(f"[force] Deleting existing KB '{kb_name}' and re-migrating...")
        if not dry_run:
            await repo.delete(existing.id)

    # --- load RAG settings + open stores ---
    from nanobot.rag.core.settings import load_settings, resolve_path
    from nanobot.rag.libs.vector_store.vector_store_factory import VectorStoreFactory
    from nanobot.rag.libs.loader.file_integrity import SQLiteIntegrityChecker

    settings = load_settings()
    integrity_db = str(resolve_path("~/.nanoresearch/rag/ingestion_history.db"))

    if not Path(integrity_db).exists():
        print(f"[warn] Integrity DB not found at {integrity_db}. Nothing to migrate.")
        return

    integrity = SQLiteIntegrityChecker(db_path=integrity_db)
    processed = integrity.list_processed(collection=collection)

    if not processed:
        print(f"[info] No successfully processed files found in collection '{collection}'.")
        return

    print(f"[info] Found {len(processed)} processed files in collection '{collection}'.")

    # --- read all chunks from ChromaDB ---
    print(f"[info] Loading chunks from ChromaDB collection '{collection}'...")
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    all_chunks = vector_store.get_all_documents()
    print(f"[info] Total chunks in ChromaDB: {len(all_chunks)}")

    # Group chunks by source_path (from chunk metadata)
    chunks_by_source: dict[str, list[dict]] = defaultdict(list)
    for chunk in all_chunks:
        meta = chunk.get("metadata", {})
        src = meta.get("source_path", "")
        if src:
            chunks_by_source[src].append(chunk)

    # Also index processed files by file_path for metadata lookup
    file_info_by_path: dict[str, dict] = {r["file_path"]: r for r in processed}

    # Build set of source paths to process: intersection of integrity DB + ChromaDB
    all_source_paths = set(file_info_by_path.keys()) | set(chunks_by_source.keys())
    print(f"[info] Unique source paths to migrate: {len(all_source_paths)}")

    if dry_run:
        print("[dry-run] Would create:")
        print(f"  1 KnowledgeBase: '{kb_name}' (collection='{collection}')")
        for src in sorted(all_source_paths):
            n_chunks = len(chunks_by_source.get(src, []))
            print(f"  KbDocument: {Path(src).name}  ({n_chunks} chunks)")
        return

    # --- create KnowledgeBase ---
    kb = await repo.create(
        uid=uid,
        name=kb_name,
        description=f"从现有 ChromaDB collection '{collection}' 迁移",
        embedding_model=getattr(settings.embedding, "model", None) if hasattr(settings, "embedding") else None,
        chroma_collection=collection,
    )
    print(f"[ok] Created KnowledgeBase id={kb.id}")

    total_docs = 0
    total_chunks = 0

    for src_path in sorted(all_source_paths):
        filename = Path(src_path).name
        file_info = file_info_by_path.get(src_path, {})
        doc_chunks = chunks_by_source.get(src_path, [])

        # Determine file size from disk if available
        file_size: int | None = None
        if Path(src_path).exists():
            file_size = Path(src_path).stat().st_size

        # Infer mime type from extension
        ext = Path(src_path).suffix.lower()
        mime_map = {".pdf": "application/pdf", ".md": "text/markdown", ".txt": "text/plain", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
        mime_type = mime_map.get(ext)

        doc = await repo.create_document(
            kb_id=kb.id,
            filename=filename,
            file_path=src_path,
            file_size=file_size,
            mime_type=mime_type,
        )

        # Sort chunks by chunk_index from metadata
        doc_chunks_sorted = sorted(doc_chunks, key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

        chunk_rows: list[KbChunk] = []
        for idx, chunk in enumerate(doc_chunks_sorted):
            meta = chunk.get("metadata", {})
            chunk_rows.append(KbChunk(
                kb_id=kb.id,
                document_id=doc.id,
                chroma_id=chunk.get("id"),
                chunk_index=meta.get("chunk_index", idx),
                content=chunk.get("text", ""),
                token_count=meta.get("token_count"),
                char_start=meta.get("char_start"),
                char_end=meta.get("char_end"),
                chunk_metadata={k: v for k, v in meta.items() if k not in ("source_path",)},
            ))

        if chunk_rows:
            await repo.create_chunks(chunk_rows)

        # Mark document as indexed
        await repo.update_document_status(
            doc.id,
            status="indexed",
            chunk_count=len(chunk_rows),
        )

        total_docs += 1
        total_chunks += len(chunk_rows)
        print(f"  [{total_docs:3d}] {filename}  ({len(chunk_rows)} chunks)")

    # Update KB counts
    await repo.increment_counts(kb.id, doc_delta=total_docs, chunk_delta=total_chunks)

    print(f"\n[done] Migrated {total_docs} documents, {total_chunks} chunks into KB '{kb_name}' (id={kb.id})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate existing ChromaDB KB to PostgreSQL")
    parser.add_argument("--collection", default="default", help="ChromaDB collection name (default: 'default')")
    parser.add_argument("--kb-name", default="默认知识库", help="Name for the new KnowledgeBase record")
    parser.add_argument("--uid", default="admin", help="Owner user UID (default: 'admin')")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")
    parser.add_argument("--force", action="store_true", help="Delete existing KB with same name and re-migrate")
    args = parser.parse_args()

    asyncio.run(migrate(
        collection=args.collection,
        kb_name=args.kb_name,
        uid=args.uid,
        dry_run=args.dry_run,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
