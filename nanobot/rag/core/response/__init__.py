"""
Response Module.

This package contains response building components:
- Response builder
- Citation generator
- Multimodal assembler
"""

from nanobot.rag.core.response.citation_generator import Citation, CitationGenerator
from nanobot.rag.core.response.multimodal_assembler import (
    ImageContent,
    ImageReference,
    MultimodalAssembler,
)
from nanobot.rag.core.response.response_builder import MCPToolResponse, ResponseBuilder

__all__ = [
    "Citation",
    "CitationGenerator",
    "ImageContent",
    "ImageReference",
    "MCPToolResponse",
    "MultimodalAssembler",
    "ResponseBuilder",
]
