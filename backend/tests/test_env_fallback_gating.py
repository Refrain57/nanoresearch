"""Server mode must not let API keys leak via env var fallback."""

import os
import pytest


def test_env_key_or_raise_local_returns_env(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    from nanoresearch.config.loader import env_key_or_raise
    assert env_key_or_raise("OPENAI_API_KEY", role="chat") == "sk-from-env"


def test_env_key_or_raise_local_missing_raises(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from nanoresearch.config.loader import env_key_or_raise
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        env_key_or_raise("OPENAI_API_KEY", role="chat")


def test_env_key_or_raise_server_raises(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leaked")  # 即使 env 有也不读
    from nanoresearch.config.loader import env_key_or_raise
    with pytest.raises(RuntimeError, match="server mode"):
        env_key_or_raise("OPENAI_API_KEY", role="chat")


def test_openai_embedding_raises_in_server_mode(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_MODE", "server")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
    from types import SimpleNamespace
    from nanoresearch.rag.libs.embedding.openai_embedding import OpenAIEmbedding
    # Build a minimal settings object matching what OpenAIEmbedding.__init__ reads
    mock_embedding = SimpleNamespace(model="text-embedding-3-small", dimensions=None, api_key=None,
                                     azure_endpoint=None, api_version=None, base_url=None)
    mock_settings = SimpleNamespace(embedding=mock_embedding)
    with pytest.raises(RuntimeError, match="server mode"):
        OpenAIEmbedding(settings=mock_settings, api_key=None)
