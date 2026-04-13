"""
Loader Module.

This package contains document loader components:
- Base loader class
- PDF loader
- File integrity checker
"""

from nanobot.rag.libs.loader.base_loader import BaseLoader
from nanobot.rag.libs.loader.pdf_loader import PdfLoader
from nanobot.rag.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker

__all__ = [
    "BaseLoader",
    "PdfLoader",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
]
