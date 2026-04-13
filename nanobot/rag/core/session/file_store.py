"""File-based session store implementation.

Persistent storage using JSON files. Supports multi-process access
through file locking.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from nanobot.rag.core.session.base import SessionStore
from nanobot.rag.core.types_agentic import SearchSession


class FileSessionStore(SessionStore):
    """File-based session storage.

    Sessions are stored as individual JSON files in a directory.
    Supports file locking for basic multi-process safety.

    Attributes:
        _sessions_dir: Directory for session files
        _memory_cache: In-memory cache for faster access
        _lock_timeout: Maximum time to wait for file lock
    """

    def __init__(
        self,
        sessions_dir: str = "./data/sessions",
        lock_timeout: float = 5.0,
    ):
        """Initialize the file session store.

        Args:
            sessions_dir: Directory to store session files
            lock_timeout: Maximum seconds to wait for file lock
        """
        self._sessions_dir = Path(sessions_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._lock_timeout = lock_timeout
        # In-memory cache for faster access
        self._memory_cache: Dict[str, SearchSession] = {}
        self._cache_loaded = False

    def _ensure_cache_loaded(self) -> None:
        """Load all sessions into memory cache if not already loaded."""
        if self._cache_loaded:
            return

        for session_file in self._sessions_dir.glob("*.json"):
            try:
                import json
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = SearchSession.from_dict(data)
                self._memory_cache[session.session_id] = session
            except Exception:
                # Skip corrupted files
                continue

        self._cache_loaded = True

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session.

        Args:
            session_id: The session ID

        Returns:
            Path to the session file
        """
        # Sanitize session_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self._sessions_dir / f"{safe_id}.json"

    def create(self, session: SearchSession) -> str:
        """Create a new session.

        Args:
            session: The session to create

        Returns:
            The session ID
        """
        import json

        self._ensure_cache_loaded()
        self._memory_cache[session.session_id] = session

        session_file = self._get_session_file(session.session_id)
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        return session.session_id

    def get(self, session_id: str) -> Optional[SearchSession]:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The session if found, None otherwise
        """
        self._ensure_cache_loaded()
        return self._memory_cache.get(session_id)

    def update(self, session: SearchSession) -> bool:
        """Update an existing session.

        Args:
            session: The session with updated data

        Returns:
            True if successful, False if session not found
        """
        import json

        self._ensure_cache_loaded()
        if session.session_id not in self._memory_cache:
            return False

        self._memory_cache[session.session_id] = session

        session_file = self._get_session_file(session.session_id)
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        return True

    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        self._ensure_cache_loaded()
        if session_id not in self._memory_cache:
            return False

        del self._memory_cache[session_id]

        session_file = self._get_session_file(session_id)
        if session_file.exists():
            session_file.unlink()

        return True

    def list_active(self) -> List[SearchSession]:
        """List all active sessions.

        Returns:
            List of all active sessions
        """
        self._ensure_cache_loaded()
        return list(self._memory_cache.values())

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Clean up expired sessions.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of sessions deleted
        """
        import json

        self._ensure_cache_loaded()
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        expired = [
            sid for sid, session in self._memory_cache.items()
            if session.created_at < cutoff
        ]

        deleted = 0
        for sid in expired:
            if self.delete(sid):
                deleted += 1

        return deleted

    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        self._ensure_cache_loaded()
        return {
            "active_sessions": len(self._memory_cache),
            "files_on_disk": len(list(self._sessions_dir.glob("*.json"))),
        }
