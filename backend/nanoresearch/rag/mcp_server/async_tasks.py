"""Async task manager for long-running operations.

Provides background task execution with status tracking,
allowing MCP tools to return immediately while processing continues.

Usage:
    # Submit a task
    task_id = await task_manager.submit(
        "ingest_document",
        lambda: run_pipeline("paper.pdf"),
        metadata={"file_path": "paper.pdf", "collection": "default"}
    )

    # Query status
    status = task_manager.get_status(task_id)
    # {"status": "running", "progress": 0.5, "message": "Processing..."}

    # Get result (blocks until complete)
    result = await task_manager.get_result(task_id)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Information about a background task."""
    id: str
    task_type: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": (self.completed_at or time.time()) - self.created_at,
            "progress": self.progress,
            "message": self.message,
            "metadata": self.metadata,
            "error": self.error,
        }


class AsyncTaskManager:
    """Manager for background task execution.

    Features:
    - Submit tasks for background execution
    - Track task status and progress
    - Retrieve results when complete
    - Auto-cleanup old completed tasks
    """

    def __init__(self, max_workers: int = 4, cleanup_interval: int = 300):
        """Initialize task manager.

        Args:
            max_workers: Maximum number of concurrent worker threads
            cleanup_interval: Seconds between cleanup of old completed tasks
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, TaskInfo] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def submit(
        self,
        task_type: str,
        func: Callable[[], Any],
        metadata: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """Submit a task for background execution.

        Args:
            task_type: Type of task (e.g., "ingest_document")
            func: Function to execute (should be blocking)
            metadata: Optional metadata about the task
            on_progress: Optional callback for progress updates

        Returns:
            Task ID for status queries
        """
        task_id = str(uuid.uuid4())[:8]
        now = time.time()

        task_info = TaskInfo(
            id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
            metadata=metadata or {},
        )

        with self._lock:
            self._tasks[task_id] = task_info

        def _run():
            # Update status to running
            with self._lock:
                self._tasks[task_id].status = TaskStatus.RUNNING
                self._tasks[task_id].started_at = time.time()

            try:
                # Execute the function
                result = func()

                # Update status to completed
                with self._lock:
                    self._tasks[task_id].status = TaskStatus.COMPLETED
                    self._tasks[task_id].completed_at = time.time()
                    self._tasks[task_id].result = result
                    self._tasks[task_id].progress = 1.0
                    self._tasks[task_id].message = "Completed"

                logger.info(f"Task {task_id} completed successfully")
                return result

            except Exception as e:
                # Update status to failed
                with self._lock:
                    self._tasks[task_id].status = TaskStatus.FAILED
                    self._tasks[task_id].completed_at = time.time()
                    self._tasks[task_id].error = str(e)
                    self._tasks[task_id].message = f"Failed: {e}"

                logger.error(f"Task {task_id} failed: {e}")
                raise

        # Submit to executor
        future = self._executor.submit(_run)
        with self._lock:
            self._futures[task_id] = future

        logger.info(f"Submitted task {task_id} ({task_type})")
        return task_id

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status.

        Args:
            task_id: Task ID

        Returns:
            Task status dict, or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return task.to_dict()
        return None

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result, blocking until complete.

        Args:
            task_id: Task ID
            timeout: Optional timeout in seconds

        Returns:
            Task result

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If task failed
        """
        with self._lock:
            future = self._futures.get(task_id)
            task = self._tasks.get(task_id)

        if not future or not task:
            raise ValueError(f"Task {task_id} not found")

        # Wait for completion
        result = future.result(timeout=timeout)

        # Check if task failed
        with self._lock:
            task = self._tasks[task_id]
            if task.status == TaskStatus.FAILED:
                raise RuntimeError(task.error)

        return result

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled, False if not running
        """
        with self._lock:
            future = self._futures.get(task_id)
            task = self._tasks.get(task_id)

        if not future or not task:
            return False

        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            cancelled = future.cancel()
            if cancelled:
                with self._lock:
                    self._tasks[task_id].status = TaskStatus.CANCELLED
                    self._tasks[task_id].completed_at = time.time()
                    self._tasks[task_id].message = "Cancelled"
                logger.info(f"Task {task_id} cancelled")
            return cancelled

        return False

    def list_tasks(
        self,
        task_type: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 20,
    ) -> list[Dict[str, Any]]:
        """List tasks, optionally filtered.

        Args:
            task_type: Filter by task type
            status: Filter by status
            limit: Maximum number of tasks to return

        Returns:
            List of task status dicts
        """
        with self._lock:
            tasks = list(self._tasks.values())

        # Filter
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        # Limit
        tasks = tasks[:limit]

        return [t.to_dict() for t in tasks]

    def cleanup(self, max_age: int = 3600) -> int:
        """Clean up old completed tasks.

        Args:
            max_age: Maximum age in seconds for completed tasks

        Returns:
            Number of tasks removed
        """
        now = time.time()
        to_remove = []

        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    if task.completed_at and (now - task.completed_at) > max_age:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]
                self._futures.pop(task_id, None)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old tasks")

        return len(to_remove)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: Wait for pending tasks to complete
        """
        self._executor.shutdown(wait=wait)
        logger.info("Task manager shutdown")


# Global singleton
_task_manager: Optional[AsyncTaskManager] = None


def get_task_manager() -> AsyncTaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = AsyncTaskManager()
    return _task_manager
