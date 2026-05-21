"""Marker PDF Loader implementation with GPU acceleration.

This module implements PDF parsing using Marker (marker-pdf) for high-quality
extraction of academic papers with formulas and tables.

Features:
- GPU-accelerated PDF parsing via Marker
- Lazy model loading (avoid loading at initialization)
- Image extraction and storage
- Image placeholder insertion with metadata tracking
- Graceful fallback to MarkItDown if Marker fails

Usage:
    loader = MarkerLoader(device="cuda")
    doc = loader.load("paper.pdf")

Or via pipeline:
    result = run_pipeline("paper.pdf", pdf_parser="marker")
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from nanobot.rag.core.types import Document
from nanobot.rag.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Marker availability check
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False


class MarkerLoader(BaseLoader):
    """PDF Loader using Marker with GPU acceleration.

    This loader:
    1. Uses Marker for high-quality text extraction (especially formulas/tables)
    2. Extracts images and saves to storage directory
    3. Inserts image placeholders in the format [IMAGE: {image_id}]
    4. Records image metadata in Document.metadata.images

    Configuration:
        device: Device for GPU acceleration ("cuda" or "cpu", default: "cuda")
        extract_images: Enable/disable image extraction (default: True)
        image_storage_dir: Base directory for image storage

    Note:
        - Models are loaded lazily on first use
        - Falls back to MarkItDown if Marker fails
    """

    def __init__(
        self,
        device: str = "cuda",
        extract_images: bool = True,
        image_storage_dir: str | Path = "~/.nanoresearch/rag/images"
    ):
        """Initialize Marker Loader.

        Args:
            device: Device for model inference ("cuda" or "cpu")
            extract_images: Whether to extract images from PDFs
            image_storage_dir: Base directory for storing extracted images
        """
        if not MARKER_AVAILABLE:
            raise ImportError(
                "Marker is not installed. Install with: uv pip install marker-pdf"
            )

        self.device = device
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._models = None
        self._converter = None

    @property
    def models(self):
        """Lazy load Marker models."""
        if self._models is None:
            logger.info(f"Loading Marker models on {self.device}...")
            self._models = create_model_dict(device=self.device)
            logger.info("Marker models loaded")
        return self._models

    @property
    def converter(self) -> PdfConverter:
        """Get or create PDF converter."""
        if self._converter is None:
            self._converter = PdfConverter(artifact_dict=self.models)
        return self._converter

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a PDF file using Marker.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Document with Markdown text and metadata.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If parsing fails critically.
        """
        # Validate file
        path = self._validate_file(file_path)
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {path}")

        # Compute document hash for unique ID and image directory
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        try:
            # Parse PDF with Marker
            rendered = self.converter(str(path))
            text_content, _, images = text_from_rendered(rendered)

            # If text is empty, fallback to MarkItDown
            if not text_content or not text_content.strip():
                logger.warning("Marker returned empty content, falling back to MarkItDown")
                return self._fallback_to_markitdown(path, doc_id, doc_hash)

        except Exception as e:
            logger.warning(f"Marker parsing failed: {e}, falling back to MarkItDown")
            return self._fallback_to_markitdown(path, doc_id, doc_hash)

        # Initialize metadata
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "parser": "marker",  # Track which parser was used
        }

        # Extract title from first heading if available
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title

        # Handle image extraction
        if self.extract_images and images:
            try:
                text_content, images_metadata = self._process_images(
                    path, text_content, doc_hash, images
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(f"Image processing failed: {e}, continuing with text-only")

        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )

    def _fallback_to_markitdown(
        self,
        path: Path,
        doc_id: str,
        doc_hash: str
    ) -> Document:
        """Fallback to MarkItDown if Marker fails."""
        try:
            from markitdown import MarkItDown

            logger.info(f"Using MarkItDown fallback for {path.name}")
            md = MarkItDown()
            result = md.convert(str(path))
            text_content = result.text_content if hasattr(result, 'text_content') else str(result)

            metadata: Dict[str, Any] = {
                "source_path": str(path),
                "doc_type": "pdf",
                "doc_hash": doc_hash,
                "parser": "markitdown-fallback",
            }

            title = self._extract_title(text_content)
            if title:
                metadata["title"] = title

            return Document(
                id=doc_id,
                text=text_content,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Both Marker and MarkItDown failed: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown heading or first non-empty line."""
        lines = text.split('\n')

        for line in lines[:20]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 0:
                return line

        return None

    def _process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str,
        images: Dict[str, Any]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Process and save images extracted by Marker.

        Args:
            pdf_path: Path to PDF file
            text_content: Current text content
            doc_hash: Document hash for image IDs
            images: Dict of image_id -> PIL.Image from Marker

        Returns:
            Tuple of (modified_text, images_metadata_list)
        """
        from PIL import Image

        images_metadata = []

        # Create image storage directory
        image_dir = self.image_storage_dir / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)

        # Marker images dict: {filename: PIL.Image}
        for filename, img in images.items():
            if not isinstance(img, Image.Image):
                continue

            try:
                # Generate image ID from filename
                # Filename format: _page_N_Figure_M.jpeg or similar
                image_id = self._generate_image_id_from_filename(filename, doc_hash)

                # Save as PNG
                image_filename = f"{image_id}.png"
                image_path = image_dir / image_filename
                img.save(image_path, "PNG")

                # Get image dimensions
                width, height = img.size

                # Create placeholder
                placeholder = f"[IMAGE: {image_id}]"
                text_content += f"\n{placeholder}\n"

                # Record metadata
                # Try to extract page number from filename
                page_num = self._extract_page_from_filename(filename)

                image_metadata = {
                    "id": image_id,
                    "path": str(image_path.absolute()),
                    "page": page_num,
                    "text_offset": len(text_content) - len(placeholder),
                    "text_length": len(placeholder),
                    "position": {
                        "width": width,
                        "height": height,
                        "index": len(images_metadata)
                    }
                }
                images_metadata.append(image_metadata)

                logger.debug(f"Saved image {image_id} to {image_path}")

            except Exception as e:
                logger.warning(f"Failed to process image {filename}: {e}")
                continue

        if images_metadata:
            logger.info(f"Processed {len(images_metadata)} images from {pdf_path}")

        return text_content, images_metadata

    def _generate_image_id_from_filename(self, filename: str, doc_hash: str) -> str:
        """Generate unique image ID from Marker filename.

        Args:
            filename: Original filename from Marker (e.g., "_page_0_Figure_3.jpeg")
            doc_hash: Document hash prefix

        Returns:
            Unique image ID (e.g., "abc123_0_figure_3")
        """
        # Remove leading underscore and extension
        name = filename.lstrip('_').rsplit('.', 1)[0]

        # Convert to consistent format: page_N_type_M -> page_N_type_M
        # Replace spaces and special chars with underscores
        name = name.lower().replace(' ', '_').replace('-', '_')

        return f"{doc_hash[:8]}_{name}"

    def _extract_page_from_filename(self, filename: str) -> int:
        """Extract page number from Marker filename."""
        import re
        match = re.search(r'_page_(\d+)', filename)
        if match:
            return int(match.group(1)) + 1  # Convert to 1-based
        return 0