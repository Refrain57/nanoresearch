"""Correction Tracker — tracks corrections to research knowledge.

This module provides a simple JSON-based storage for tracking when
existing claims or insights are contradicted by new research findings.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class CorrectionTracker:
    """Track corrections to research knowledge.

    Uses a simple JSON file to store correction records.
    Each correction records:
    - old_value: The existing knowledge that was contradicted
    - new_value: The new finding that contradicts it
    - source_url: Source of the correction
    - reason: Why this is a correction
    """

    def __init__(self, storage_path: str = "~/.nanoresearch/corrections.json"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def add_correction(self, correction: dict[str, Any]) -> None:
        """Add a new correction record.

        Args:
            correction: Dictionary containing correction details.
                Must have: old_value, new_value, source_url, reason
        """
        data = self._load()
        correction_id = f"corr_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(data):04d}"
        correction["id"] = correction_id
        correction["created_at"] = datetime.now().isoformat()
        data[correction_id] = correction
        self._save(data)
        logger.info(
            f"CorrectionTracker: recorded correction {correction_id}: "
            f"{correction.get('old_value', '')[:50]}... → {correction.get('new_value', '')[:50]}..."
        )

    def list_corrections(
        self,
        limit: int = 100,
        correction_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List correction records.

        Args:
            limit: Maximum number of corrections to return.
            correction_type: Optional filter by type ("claim" or "insight").

        Returns:
            List of correction records, most recent first.
        """
        data = self._load()
        corrections = [v | {"id": k} for k, v in data.items()]

        if correction_type:
            corrections = [c for c in corrections if c.get("correction_type") == correction_type]

        # Sort by created_at descending
        corrections.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return corrections[:limit]

    def get_correction(self, correction_id: str) -> dict[str, Any] | None:
        """Get a specific correction by ID.

        Args:
            correction_id: The ID of the correction to retrieve.

        Returns:
            The correction dict with 'id' key, or None if not found.
        """
        data = self._load()
        if correction_id in data:
            return data[correction_id] | {"id": correction_id}
        return None

    def count(self) -> int:
        """Count of total corrections."""
        return len(self._load())

    def clear_old_corrections(self, days: int = 30) -> int:
        """Clear corrections older than specified days.

        Args:
            days: Number of days to keep corrections.

        Returns:
            Number of corrections removed.
        """
        data = self._load()
        now = datetime.now()
        to_remove = []

        for corr_id, corr in data.items():
            created_at_str = corr.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if (now - created_at).days > days:
                        to_remove.append(corr_id)
                except (ValueError, TypeError):
                    pass

        for corr_id in to_remove:
            del data[corr_id]

        if to_remove:
            self._save(data)
            logger.info(f"CorrectionTracker: cleared {len(to_remove)} old corrections")

        return len(to_remove)

    def _load(self) -> dict[str, dict]:
        """Load tracker data from disk."""
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"CorrectionTracker: failed to load, starting fresh: {e}")
        return {}

    def _save(self, data: dict[str, dict]) -> None:
        """Save tracker data to disk."""
        try:
            self.storage_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except IOError as e:
            logger.error(f"CorrectionTracker: failed to save: {e}")
            raise
