"""Insight Tracker — tracks unprocessed candidate insights."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from loguru import logger


class InsightTracker:
    """Track unprocessed candidate insights.

    Uses a simple JSON file as external index to solve the "list all candidates" problem.
    This is separate from the vector store (ChromaDB) which doesn't support efficient
    "get all" operations with custom filters.
    """

    def __init__(self, storage_path: str = "~/.nanoresearch/insight_tracker.json"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def add_candidate(self, insight_id: str, insight: dict) -> None:
        """Add a new candidate insight.

        Args:
            insight_id: Unique identifier for the insight.
            insight: Dictionary containing insight data (must have 'insight' key).
        """
        data = self._load()
        data[insight_id] = {
            **insight,
            "added_at": datetime.now().isoformat(),
        }
        self._save(data)
        logger.debug(f"InsightTracker: added candidate {insight_id}")

    def list_candidates(self) -> list[dict]:
        """List all unprocessed candidate insights.

        Returns:
            List of candidate insight dictionaries, each including 'id' key.
        """
        data = self._load()
        return [v | {"id": k} for k, v in data.items()]

    def mark_processed(self, insight_id: str) -> None:
        """Mark a candidate as processed (remove from tracker).

        Args:
            insight_id: The ID of the insight to mark as processed.
        """
        data = self._load()
        if insight_id in data:
            del data[insight_id]
            self._save(data)
            logger.debug(f"InsightTracker: marked {insight_id} as processed")

    def get_candidate(self, insight_id: str) -> dict | None:
        """Get a specific candidate by ID.

        Args:
            insight_id: The ID of the insight to retrieve.

        Returns:
            The candidate dict with 'id' key, or None if not found.
        """
        data = self._load()
        if insight_id in data:
            return data[insight_id] | {"id": insight_id}
        return None

    def count(self) -> int:
        """Count of unprocessed candidates."""
        return len(self._load())

    def _load(self) -> dict[str, dict]:
        """Load tracker data from disk."""
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"InsightTracker: failed to load, starting fresh: {e}")
        return {}

    def _save(self, data: dict[str, dict]) -> None:
        """Save tracker data to disk."""
        try:
            self.storage_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except IOError as e:
            logger.error(f"InsightTracker: failed to save: {e}")
            raise