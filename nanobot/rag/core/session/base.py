"""Base session store interface.

Defines the abstract interface for session storage implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from nanobot.rag.core.types_agentic import SearchSession


class SessionStore(ABC):
    """Abstract base class for session storage.

    Implementations can use in-memory storage, file-based storage,
    or external systems like Redis.
    """

    @abstractmethod
    def create(self, session: SearchSession) -> str:
        """Create a new session.

        Args:
            session: The session to create

        Returns:
            The session ID
        """
        pass

    @abstractmethod
    def get(self, session_id: str) -> Optional[SearchSession]:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The session if found, None otherwise
        """
        pass

    @abstractmethod
    def update(self, session: SearchSession) -> bool:
        """Update an existing session.

        Args:
            session: The session with updated data

        Returns:
            True if successful, False if session not found
        """
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def list_active(self) -> List[SearchSession]:
        """List all active sessions.

        Returns:
            List of all active sessions
        """
        pass

    @abstractmethod
    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Clean up expired sessions.

        Args:
            max_age_seconds: Maximum age in seconds before a session is expired

        Returns:
            Number of sessions deleted
        """
        pass

    def exists(self, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            session_id: The session ID to check

        Returns:
            True if session exists
        """
        return self.get(session_id) is not None
