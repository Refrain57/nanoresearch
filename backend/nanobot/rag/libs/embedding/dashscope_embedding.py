"""DashScope (通义千问) Embedding implementation.

Uses the OpenAI-compatible API that DashScope provides, targeting
the base_url configured in settings.embedding.base_url.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from nanobot.rag.libs.embedding.base_embedding import BaseEmbedding


class DashScopeEmbeddingError(RuntimeError):
    """Raised when DashScope Embeddings API call fails."""


class DashScopeEmbedding(BaseEmbedding):
    """DashScope Embedding provider (OpenAI-compatible API).

    DashScope exposes an OpenAI-compatible embeddings endpoint at:
      https://dashscope.aliyuncs.com/compatible-mode/v1

    Configure in settings.yaml:
      embedding:
        provider: dashscope
        model: text-embedding-v3
        api_key: sk-...
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model = settings.embedding.model
        self.dimensions = getattr(settings.embedding, "dimensions", None)

        self.api_key = (
            api_key
            or getattr(settings.embedding, "api_key", None)
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "DashScope API key not provided. Set in settings.yaml "
                "(embedding.api_key) or DASHSCOPE_API_KEY / OPENAI_API_KEY env var."
            )

        settings_base_url = getattr(settings.embedding, "base_url", None)
        self.base_url = settings_base_url or self.DEFAULT_BASE_URL
        self._extra_config = kwargs

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        self.validate_texts(texts)

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package not installed. Install with: pip install openai"
            ) from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        api_params: dict[str, Any] = {
            "input": texts,
            "model": self.model,
        }
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None:
            api_params["dimensions"] = dimensions

        try:
            response = client.embeddings.create(**api_params)
        except Exception as e:
            raise DashScopeEmbeddingError(
                f"DashScope Embeddings API call failed: {e}"
            ) from e

        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise DashScopeEmbeddingError(
                f"Failed to parse DashScope Embeddings API response: {e}"
            ) from e

        if len(embeddings) != len(texts):
            raise DashScopeEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        return embeddings

    def get_dimension(self) -> Optional[int]:
        if self.dimensions is not None:
            return self.dimensions
        model_dimensions = {
            "text-embedding-v1": 1536,
            "text-embedding-v2": 1536,
            "text-embedding-v3": 1024,
        }
        return model_dimensions.get(self.model)
