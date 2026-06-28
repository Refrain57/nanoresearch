"""Migrate conversation-derived claims from research_claims to user_memory.

This script migrates existing data where source="conversation" from the
research_claims collection to the new user_memory collection.

Usage:
    python -m nanoresearch.scripts.migrate_to_user_memory

After running this script, you may optionally delete the migrated claims
from research_claims using:
    python -m nanoresearch.scripts.migrate_to_user_memory --delete
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


async def migrate(delete_after: bool = False):
    logger.info("Starting migration: conversation claims -> user_memory")

    try:
        from nanoresearch.rag.core.settings import load_settings
        from nanoresearch.research.knowledge_search import KnowledgeSearch

        settings = load_settings()
        ks = KnowledgeSearch.from_settings(settings)

        # Query all source=conversation claims
        conversation_claims = ks.claim_store.query_by_metadata(
            filters={"source": "conversation"}
        )

        if not conversation_claims:
            logger.info("No conversation claims found in research_claims collection")
            return

        logger.info(f"Found {len(conversation_claims)} conversation claims to migrate")

        # Transform to user_memory format
        memories = []
        for claim in conversation_claims:
            metadata = claim.get("metadata", {})
            memories.append({
                "text": metadata.get("text", ""),
                "type": metadata.get("claim_type", "factual"),
                "confidence": metadata.get("confidence", 0.7),
                "is_evergreen": metadata.get("is_evergreen", False),
                "created_at": metadata.get("created_at", ""),
            })

        # Write to user_memory collection
        written, skipped = ks.write_user_memory_sync(memories)
        logger.info(f"Migration complete: wrote {written} memories, skipped {skipped} duplicates")

        # Optionally delete old data
        if delete_after and written > 0:
            ids_to_delete = [c["id"] for c in conversation_claims]
            ks.claim_store.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} migrated claims from research_claims")

        # Print summary
        logger.info("Summary:")
        logger.info(f"  - Claims to migrate: {len(conversation_claims)}")
        logger.info(f"  - Memories written: {written}")
        logger.info(f"  - Duplicates skipped: {skipped}")
        if delete_after:
            logger.info(f"  - Claims deleted from research_claims: {len(ids_to_delete)}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Migrate conversation claims to user_memory collection"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete migrated claims from research_claims after successful migration",
    )
    args = parser.parse_args()

    asyncio.run(migrate(delete_after=args.delete))


if __name__ == "__main__":
    main()