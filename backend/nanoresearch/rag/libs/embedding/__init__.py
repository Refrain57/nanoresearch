"""
Embedding Module.

This package contains embedding service abstractions and implementations:
- Base embedding class
- Embedding factory
- Provider implementations (OpenAI, Azure, Ollama)
"""

from nanoresearch.rag.libs.embedding.azure_embedding import AzureEmbedding
from nanoresearch.rag.libs.embedding.base_embedding import BaseEmbedding
from nanoresearch.rag.libs.embedding.embedding_factory import EmbeddingFactory
from nanoresearch.rag.libs.embedding.ollama_embedding import OllamaEmbedding
from nanoresearch.rag.libs.embedding.openai_embedding import OpenAIEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "OpenAIEmbedding",
    "AzureEmbedding",
    "OllamaEmbedding",
]
