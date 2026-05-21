"""
Loader Module.

This package contains document loader components:
- Base loader class
- PDF loader (MarkItDown)
- Marker loader (GPU-accelerated, high-quality)
- File integrity checker

Note: MarkerLoader is lazily imported to avoid slow initialization
(~9s) of the marker package with its heavy dependencies (google.genai).
"""

from nanobot.rag.libs.loader.base_loader import BaseLoader
from nanobot.rag.libs.loader.pdf_loader import PdfLoader
from nanobot.rag.libs.loader.markdown_loader import MarkdownLoader
from nanobot.rag.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker

# Lazy import for MarkerLoader to avoid slow initialization
def get_marker_loader():
    """Get MarkerLoader class (lazy import)."""
    from nanobot.rag.libs.loader.marker_loader import MarkerLoader
    return MarkerLoader

# For backwards compatibility, provide MarkerLoader as a property-like accessor
class _MarkerLoaderAccessor:
    """Accessor that lazily imports MarkerLoader on first use."""
    def __call__(self, *args, **kwargs):
        return get_marker_loader()(*args, **kwargs)

MarkerLoader = _MarkerLoaderAccessor()

__all__ = [
    "BaseLoader",
    "PdfLoader",
    "MarkerLoader",
    "MarkdownLoader",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
    "get_marker_loader",
]
