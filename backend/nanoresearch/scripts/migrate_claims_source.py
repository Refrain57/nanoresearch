"""Migrate existing claims to add source="research" field.

Usage:
    python -m nanoresearch.scripts.migrate_claims_source
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger


async def migrate():
    logger.info("Starting claims migration: adding source='research' to all existing claims")

    try:
        from nanoresearch.rag.core.settings import load_settings
        from nanoresearch.rag.libs.vector_store.chroma_store import ChromaStore

        settings = load_settings()

        # Open the research_claims collection
        store = ChromaStore(settings, collection_name="research_claims")
        count = store.count()

        if count == 0:
            logger.info("No claims found in research_claims collection")
            return

        logger.info(f"Found {count} claims, fetching all...")

        # Get all claims
        all_claims = store.get_all_documents()

        # Find claims that need migration (no source field or wrong source)
        to_update = []
        for claim in all_claims:
            meta = claim.get("metadata", {})
            current_source = meta.get("source")

            if current_source is None or current_source == "":
                # Missing source field - needs migration
                to_update.append((claim["id"], {"source": "research", "verified": True}))
            elif current_source not in ("research", "conversation"):
                # Unknown source - treat as research
                to_update.append((claim["id"], {"source": "research"}))
            # If source is already "research" or "conversation", skip

        if not to_update:
            logger.info(f"All {count} claims already have source field set")
            return

        logger.info(f"Migrating {len(to_update)} claims...")

        # Batch update in chunks of 100
        chunk_size = 100
        migrated = 0
        for i in range(0, len(to_update), chunk_size):
            chunk = to_update[i:i + chunk_size]
            store.update_batch(chunk)
            migrated += len(chunk)
            logger.info(f"Migrated {migrated}/{len(to_update)} claims...")

        logger.info(f"Migration complete: {len(to_update)} claims updated")

        # Print summary
        logger.info("Summary:")
        logger.info(f"  - Total claims: {count}")
        logger.info(f"  - Claims updated: {len(to_update)}")
        logger.info(f"  - Claims already migrated: {count - len(to_update)}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(migrate())
