"""In-memory session store implementation.

Simple in-memory storage for testing and single-process scenarios.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from nanobot.rag.core.session.base import SessionStore
from nanobot.rag.core.types_agentic import SearchSession


class MemorySessionStore(SessionStore):
    """In-memory session storage.

    Sessions are stored in a dictionary and are lost when the process exits.
    Suitable for testing and single-process deployments.

    Attributes:
        _sessions: Internal session storage dictionary
    """

    def __init__(self):
        """Initialize the memory store."""
        self._sessions: Dict[str, SearchSession] = {}

    def create(self, session: SearchSession) -> str:
        """Create a new session.

        Args:
            session: The session to create

        Returns:
            The session ID
        """
        self._sessions[session.session_id] = session
        return session.session_id

    def get(self, session_id: str) -> Optional[SearchSession]:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The session if found, None otherwise
        """
        return self._sessions.get(session_id)

    def update(self, session: SearchSession) -> bool:
        """Update an existing session.

        Args:
            session: The session with updated data

        Returns:
            True if successful, False if session not found
        """
        if session.session_id not in self._sessions:
            return False
        self._sessions[session.session_id] = session
        return True

    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_active(self) -> List[SearchSession]:
        """List all active sessions.

        Returns:
            List of all active sessions
        """
        return list(self._sessions.values())

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Clean up expired sessions.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        expired = [
            sid for sid, session in self._sessions.items()
            if session.created_at < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)
