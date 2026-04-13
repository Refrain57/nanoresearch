"""
Splitter Module.

This package contains text splitter abstractions and implementations:
- Base splitter class
- Splitter factory
- Implementations (Recursive, Semantic, FixedLength)
"""

from nanobot.rag.libs.splitter.base_splitter import BaseSplitter
from nanobot.rag.libs.splitter.splitter_factory import SplitterFactory

# Import concrete implementations (they auto-register with factory)
try:
    from nanobot.rag.libs.splitter.recursive_splitter import RecursiveSplitter
except ImportError:
    RecursiveSplitter = None  # type: ignore[assignment, misc]

__all__ = [
    "BaseSplitter",
    "SplitterFactory",
    "RecursiveSplitter",
]
