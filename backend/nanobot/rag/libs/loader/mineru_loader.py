"""MinerU PDF Loader — local library mode and HTTP API mode.

Local mode: requires `magic-pdf[full]` (or `mineru`) package.
HTTP mode:  set `mineru_url` to a running MinerU API server endpoint.

Usage:
    # local
    loader = MinerULoader(mode="local")
    doc = loader.load("paper.pdf")

    # HTTP API
    loader = MinerULoader(mode="http", mineru_url="http://localhost:8080")
    doc = loader.load("paper.pdf")
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from nanobot.rag.core.types import Document
from nanobot.rag.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MinerULoader(BaseLoader):
    """PDF Loader using MinerU (magic-pdf) for high-quality OCR extraction.

    Supports two modes:
    - "local": uses installed magic-pdf package (GPU recommended)
    - "http":  sends file to a MinerU API server via multipart POST

    No fallback is performed on failure — the caller must handle exceptions
    if they want to retry with a different parser.
    """

    def __init__(
        self,
        mode: str = "local",
        mineru_url: str = "http://localhost:8080",
        timeout: int = 300,
        image_storage_dir: str | Path = "~/.nanoresearch/rag/images",
    ):
        """Initialize MinerU Loader.

        Args:
            mode: "local" or "http"
            mineru_url: Base URL of MinerU API server (only used in http mode)
            timeout: HTTP request timeout in seconds
            image_storage_dir: Base dir for saving extracted images (local mode)
        """
        self.mode = mode
        self.mineru_url = mineru_url.rstrip("/")
        self.timeout = timeout
        self.image_storage_dir = Path(image_storage_dir).expanduser()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        if self.mode == "http":
            return self._load_http(path, doc_id, doc_hash)
        return self._load_local(path, doc_id, doc_hash)

    # ------------------------------------------------------------------
    # Local mode
    # ------------------------------------------------------------------

    def _load_local(self, path: Path, doc_id: str, doc_hash: str) -> Document:
        markdown_text, image_types = self._run_magic_pdf(path, doc_hash)

        if not markdown_text or not markdown_text.strip():
            raise RuntimeError(f"MinerU returned empty content for {path.name}")

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "parser": "mineru-local",
        }
        title = self._extract_title(markdown_text)
        if title:
            metadata["title"] = title
        if image_types:
            metadata["image_types"] = image_types

        return Document(id=doc_id, text=markdown_text, metadata=metadata)

    def _run_magic_pdf(self, path: Path, doc_hash: str) -> tuple[str, dict]:
        """Run magic-pdf pipeline and return (markdown, image_types).

        image_types maps absolute image path → "table" | "figure" for every
        image extracted from the document.  Used by the ingestion pipeline to
        propagate per-chunk type hints into ImageCaptioner.

        Supports magic-pdf 0.6.x (UNIPipe) and 1.x (PymuDocDataset) APIs.
        """
        img_dir = self.image_storage_dir / doc_hash
        img_dir.mkdir(parents=True, exist_ok=True)

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        # Pre-import torch before paddle/paddleocr to avoid Windows DLL conflict
        # where paddle's MKL DLLs shadow torch's shm.dll dependencies.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass

        # Try v1.x API (PymuDocDataset — magic-pdf >= 1.0)
        v1_available = False
        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            v1_available = True
        except ImportError:
            pass

        if v1_available:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

            ds = PymuDocDataset(pdf_bytes)
            image_writer = FileBasedDataWriter(str(img_dir))

            try:
                from magic_pdf.config.enums import SupportedPdfParseMethod
                use_ocr = ds.classify() == SupportedPdfParseMethod.OCR
            except Exception:
                use_ocr = False

            infer_result = ds.apply(doc_analyze, ocr=use_ocr)
            # 1.x: pipe_txt_mode / pipe_ocr_mode return PipeResult directly (no separate classify/analyze/parse)
            pipe_result = infer_result.pipe_ocr_mode(image_writer) if use_ocr else infer_result.pipe_txt_mode(image_writer)
            markdown = pipe_result.get_markdown(str(img_dir))
            image_types = self._extract_image_types(pipe_result, img_dir)
            return markdown, image_types

        # Fallback: v0.6.x API (UNIPipe)
        import magic_pdf.model as model_config
        model_config.__use_inside_model__ = True

        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

        image_writer = DiskReaderWriter(str(img_dir))
        jso_useful_key = {"_pdf_type": "", "model_list": []}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        return pipe.get_markdown(str(img_dir)), {}

    def _extract_image_types(self, pipe_result: Any, img_dir: Path) -> dict:
        """Try to extract image type info (table vs figure) from MinerU content_list.

        Returns a dict mapping absolute image path (as it appears in the markdown)
        to "table" or "figure".  Returns empty dict on any error so callers
        always get a valid (possibly empty) mapping.
        """
        try:
            content_list = pipe_result.get_content_list(str(img_dir))
        except Exception:
            return {}

        image_types: dict = {}
        for item in content_list:
            item_type = item.get("type", "")
            label = "table" if item_type == "table" else "figure"
            # MinerU may use different field names for the image path
            img_path_str = (
                item.get("img_path")
                or item.get("img_name")
                or item.get("image_path")
                or ""
            )
            if not img_path_str:
                continue
            img_path = Path(img_path_str)
            # Resolve to absolute path within img_dir when relative
            abs_path = img_path if img_path.is_absolute() else img_dir / img_path
            image_types[str(abs_path)] = label
            # Also index by basename so pipeline can match by filename alone
            image_types[img_path.name] = label
        return image_types

    # ------------------------------------------------------------------
    # HTTP mode
    # ------------------------------------------------------------------

    def _load_http(self, path: Path, doc_id: str, doc_hash: str) -> Document:
        import base64
        import httpx

        url = f"{self.mineru_url}/api/v1/extract"
        logger.info(f"Sending {path.name} to MinerU API at {url}")

        with open(path, "rb") as f:
            try:
                resp = httpx.post(
                    url,
                    files={"file": (path.name, f, "application/pdf")},
                    timeout=self.timeout,
                )
            except httpx.TransportError as e:
                raise RuntimeError(f"Cannot reach MinerU server at {self.mineru_url}: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"MinerU API returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        markdown_text: str = (
            data.get("markdown")
            or (data.get("result") or {}).get("markdown")
            or ""
        )

        if not markdown_text.strip():
            raise RuntimeError(f"MinerU API returned empty markdown for {path.name}")

        # Save images returned from server to local image_storage_dir
        image_types: Dict[str, str] = {}
        raw_images: dict = data.get("images") or {}
        if raw_images:
            img_dir = self.image_storage_dir / doc_hash
            img_dir.mkdir(parents=True, exist_ok=True)
            for img_name, img_info in raw_images.items():
                img_bytes = base64.b64decode(img_info["data"])
                img_path = img_dir / img_name
                img_path.write_bytes(img_bytes)
                img_type = img_info.get("type", "figure")
                image_types[str(img_path)] = img_type
                image_types[img_name] = img_type

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "parser": "mineru-http",
        }
        title = self._extract_title(markdown_text)
        if title:
            metadata["title"] = title
        if image_types:
            metadata["image_types"] = image_types

        return Document(id=doc_id, text=markdown_text, metadata=metadata)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_file_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_title(self, text: str) -> Optional[str]:
        for line in text.split("\n")[:20]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        for line in text.split("\n")[:10]:
            line = line.strip()
            if line:
                return line
        return None
