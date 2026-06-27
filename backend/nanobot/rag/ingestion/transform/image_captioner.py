"""Image Captioner transform for enriching chunks with image descriptions.

Performance Optimizations:
1. Only processes images that are actually referenced in chunk text (via [IMAGE: id] placeholder)
2. Uses caption cache to avoid redundant Vision API calls for the same image
3. Skips chunks without image references entirely
4. Parallel processing of unique images with thread-safe caching

Output format: ![caption](url) — native markdown image with stable URL (Layer 2).
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

from nanobot.rag.core.settings import Settings
from nanobot.rag.core.types import Chunk
from nanobot.rag.core.trace.trace_context import TraceContext
from nanobot.rag.ingestion.transform.base_transform import BaseTransform
from nanobot.rag.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from nanobot.rag.libs.llm.llm_factory import LLMFactory
from nanobot.rag.observability.logger import get_logger

logger = get_logger(__name__)

# Regex to find image placeholders: [IMAGE: some_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

# Default max parallel workers for Vision API calls
DEFAULT_MAX_WORKERS = 3  # Lower than text LLM due to higher cost/latency

_TABLE_PROMPT = (
    "这是一个表格图片。请提取表格中所有数据，以 Markdown 表格格式（使用 | 分隔符）原样输出，"
    "不要遗漏任何行或列，不要添加额外解释。"
)
_DEFAULT_PROMPT = (
    "请详细描述这张图片的内容，包括所有可见的文字、数据、图形和结构信息，以便用于文档检索。"
)


class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks using Vision LLM.
    
    This transform identifies chunks containing image references, uses a Vision LLM
    to generate descriptive captions, and enriches the chunk text/metadata with
    these captions to improve retrieval for visual content.
    
    Key Features:
    - Only processes images actually referenced in chunk text (not all images in metadata)
    - Caches captions to avoid redundant Vision API calls
    - Thread-safe caption cache for potential future parallelization
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseVisionLLM] = None
    ):
        self.settings = settings
        self.llm = None
        # Caption cache: image_id -> caption string (thread-safe with lock)
        self._caption_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()

        # Image URL configuration (Layer 2)
        ingestion = settings.ingestion
        self._image_storage_root = Path(
            ingestion.image_storage_root if ingestion else "~/.nanoresearch/rag/images"
        ).expanduser()
        self._image_base_url = (
            ingestion.image_base_url if ingestion else "http://localhost:18790/rag-images"
        )
        
        # Check if vision LLM is enabled in settings
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
                 # We don't raise here to allow pipeline to continue without captioning
                 # effectively falling back to no-op for this transform
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")
        
        self.prompt = _DEFAULT_PROMPT

    def _build_image_url(self, img_path: str, img_id: str = "") -> str:
        """Convert an absolute image path to a stable HTTP URL."""
        if not img_path:
            return ""
        p = Path(img_path)
        try:
            rel = p.relative_to(self._image_storage_root)
            return f"{self._image_base_url}/{rel.as_posix()}"
        except ValueError:
            return f"{self._image_base_url}/{p.name}"

    # Matches existing ![alt](local_path) where path looks like an absolute local path
    _LOCAL_IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    def _normalize_local_image_urls(self, text: str) -> str:
        """Replace ![alt](local_path) with ![caption](http_url) for local file paths.

        If a caption was generated for this path (via _caption_local_images), it
        replaces the alt text; otherwise the original alt is preserved.
        """
        def _replace(m: re.Match) -> str:
            alt, path = m.group(1), m.group(2)
            if path.startswith("http://") or path.startswith("https://"):
                return m.group(0)
            url = self._build_image_url(path)
            if not url:
                return m.group(0)
            with self._cache_lock:
                caption = self._caption_cache.get(path)
            if caption:
                return f"![{caption.replace(chr(10), ' ').strip()}]({url})"
            return f"![{alt}]({url})"

        return self._LOCAL_IMG_RE.sub(_replace, text)

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """Extract image IDs actually referenced in the chunk text.
        
        Args:
            text: Chunk text content
            
        Returns:
            List of image IDs found in [IMAGE: id] placeholders
        """
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _get_caption(
        self,
        img_id: str,
        img_path: str,
        trace: Optional[TraceContext] = None,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Get caption for an image, using cache if available. Thread-safe.
        
        Args:
            img_id: Image identifier
            img_path: Path to image file
            trace: Optional trace context
            
        Returns:
            Caption string or None if failed
        """
        # Check cache first (thread-safe read)
        with self._cache_lock:
            if img_id in self._caption_cache:
                logger.debug(f"Caption cache hit for image {img_id}")
                return self._caption_cache[img_id]
        
        # Validate path
        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None
        
        try:
            image_input = ImageInput(path=img_path)
            response = self.llm.chat_with_image(
                text=prompt or self.prompt,
                image=image_input,
                trace=trace
            )
            caption = response.content
            
            # Cache the result (thread-safe write)
            with self._cache_lock:
                self._caption_cache[img_id] = caption
            logger.debug(f"Generated and cached caption for image {img_id}")
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    def _caption_local_images(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> None:
        """Caption MinerU-style ![](local_path) images using per-chunk image_types.

        Populates self._caption_cache keyed by the local path string so that
        _normalize_local_image_urls can apply the caption when rewriting to HTTP URLs.
        """
        # Collect unique (path, prompt) pairs across all chunks
        path_to_prompt: Dict[str, str] = {}
        for chunk in chunks:
            image_types = chunk.metadata.get("image_types", {})
            for m in self._LOCAL_IMG_RE.finditer(chunk.text):
                ref = m.group(2)
                if ref.startswith(("http://", "https://")):
                    continue
                if ref in self._caption_cache:
                    continue
                img_type = image_types.get(ref, "")
                used_prompt = _TABLE_PROMPT if img_type == "table" else _DEFAULT_PROMPT
                path_to_prompt[ref] = used_prompt

        if not path_to_prompt:
            return

        logger.info(f"ImageCaptioner: captioning {len(path_to_prompt)} MinerU local images")
        for img_path, used_prompt in path_to_prompt.items():
            self._get_caption(img_path, img_path, trace, prompt=used_prompt)

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks and add captions for referenced images.
        
        Only processes images that are actually referenced in chunk text
        via [IMAGE: id] placeholders. Uses caching to avoid redundant API calls.
        Parallel processing for unique images.
        """
        # Always normalize local image URLs (applies even without VLM)
        # Caption cache populated below is used by _normalize_local_image_urls.
        if not self.llm:
            for chunk in chunks:
                chunk.text = self._normalize_local_image_urls(chunk.text)
            return chunks

        # Build image lookup from all chunks' metadata
        image_lookup: Dict[str, dict] = {}
        for chunk in chunks:
            if chunk.metadata and "images" in chunk.metadata:
                for img_meta in chunk.metadata.get("images", []):
                    img_id = img_meta.get("id")
                    if img_id and img_id not in image_lookup:
                        image_lookup[img_id] = img_meta
        
        logger.info(f"Found {len(image_lookup)} unique images in document")
        
        # Clear cache for new document processing
        with self._cache_lock:
            self._caption_cache.clear()
        
        # First pass: collect all unique image IDs that need captioning
        images_to_caption: Dict[str, str] = {}  # img_id -> img_path
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            for img_id in referenced_ids:
                if img_id not in images_to_caption:
                    img_meta = image_lookup.get(img_id)
                    if img_meta and img_meta.get("path"):
                        images_to_caption[img_id] = img_meta.get("path")
        
        # Parallel caption generation for all unique images
        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)

        # Caption MinerU direct images (![](local_path) format, keyed by image_types)
        self._caption_local_images(chunks, trace)

        # Second pass: apply captions + normalize local image URLs in all chunks
        processed_chunks = []
        total_captions_added = 0

        for chunk in chunks:
            # Always normalize any ![](local_path) → ![](http_url), regardless of LLM
            chunk.text = self._normalize_local_image_urls(chunk.text)

            referenced_ids = self._find_referenced_image_ids(chunk.text)

            if not referenced_ids:
                processed_chunks.append(chunk)
                continue
            
            new_text = chunk.text
            captions = []

            for img_id in referenced_ids:
                img_id_stripped = img_id.strip()

                # Get caption from cache (already populated by parallel processing)
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_stripped)

                img_meta = image_lookup.get(img_id_stripped, {})
                img_path = img_meta.get("path", "")
                url = self._build_image_url(img_path, img_id_stripped)

                if caption:
                    captions.append({"id": img_id_stripped, "caption": caption, "url": url})
                    alt = caption.replace("\n", " ").strip()
                    replacement = f"![{alt}]({url})"
                    new_text = new_text.replace(f"[IMAGE: {img_id}]", replacement)
                    total_captions_added += 1
                elif url:
                    # No caption but URL available — emit plain image link
                    new_text = new_text.replace(f"[IMAGE: {img_id}]", f"![]({url})")
                    
            chunk.text = new_text
            
            if captions:
                if "image_captions" not in chunk.metadata:
                    chunk.metadata["image_captions"] = []
                chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
        
        with self._cache_lock:
            api_calls = len(self._caption_cache)
        logger.info(f"Added {total_captions_added} captions, API calls: {api_calls}")
            
        return processed_chunks
    
    def _generate_captions_parallel(
        self, 
        images_to_caption: Dict[str, str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """Generate captions for multiple images in parallel.
        
        Args:
            images_to_caption: Dict of img_id -> img_path
            trace: Optional trace context
        """
        if not images_to_caption:
            return
        
        max_workers = min(DEFAULT_MAX_WORKERS, len(images_to_caption))
        logger.debug(f"Generating captions for {len(images_to_caption)} images (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_caption, img_id, img_path, trace): img_id
                for img_id, img_path in images_to_caption.items()
            }
            
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    caption = future.result()
                    if caption:
                        logger.debug(f"Caption generated for {img_id}")
                except Exception as e:
                    logger.error(f"Failed to generate caption for {img_id}: {e}")
