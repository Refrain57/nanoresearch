"""
LLM Module.

This package contains LLM client abstractions and implementations:
- Base LLM class (text-only)
- Base Vision LLM class (multimodal: text + image)
- LLM factory
- Provider implementations (OpenAI, Azure, Ollama, DeepSeek)
"""

from nanobot.rag.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from nanobot.rag.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from nanobot.rag.libs.llm.llm_factory import LLMFactory
from nanobot.rag.libs.llm.openai_llm import OpenAILLM, OpenAILLMError
from nanobot.rag.libs.llm.openai_vision_llm import OpenAIVisionLLM, OpenAIVisionLLMError
from nanobot.rag.libs.llm.azure_llm import AzureLLM, AzureLLMError
from nanobot.rag.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from nanobot.rag.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError

# Register text-only LLM providers with factory
LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("azure", AzureLLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("ollama", OllamaLLM)

# Note: Vision LLM providers will be registered in task B9+

__all__ = [
    # Base classes
    "BaseLLM",
    "BaseVisionLLM",
    # Data types
    "ChatResponse",
    "Message",
    "ImageInput",
    # Factory
    "LLMFactory",
    # Text-only LLM implementations
    "OpenAILLM",
    "OpenAILLMError",
    "AzureLLM",
    "AzureLLMError",
    "DeepSeekLLM",
    "DeepSeekLLMError",
    "OllamaLLM",
    "OllamaLLMError",
    # Vision LLM implementations
    "OpenAIVisionLLM",
    "OpenAIVisionLLMError",
]
