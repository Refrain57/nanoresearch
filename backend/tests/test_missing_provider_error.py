"""Server-mode missing-provider API contract test."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nanoresearch.providers.model_factory import ModelResolutionError


def _make_test_app() -> FastAPI:
    """Tiny app with only the exception handler under test installed.

    Bypasses create_app() because of a pre-existing TunableObjectVersion
    ImportError that's unrelated to Phase 5.
    """
    from nanoresearch.server.main import _missing_provider_handler  # added by this task
    app = FastAPI()
    app.add_exception_handler(ModelResolutionError, _missing_provider_handler)

    @app.get("/_raise_chat")
    def _raise_chat():
        raise ModelResolutionError(
            "no api key", sources_checked=["user_providers"], missing_role="chat"
        )

    @app.get("/_raise_embedding")
    def _raise_embedding():
        raise ModelResolutionError(
            "no api key", sources_checked=["user_providers"], missing_role="embedding"
        )

    @app.get("/_raise_no_role")
    def _raise_no_role():
        raise ModelResolutionError("no api key")

    return app


def test_missing_provider_returns_422_with_role():
    client = TestClient(_make_test_app())
    resp = client.get("/_raise_chat")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "missing_provider"
    assert body["role"] == "chat"
    assert "no api key" in body["message"]


def test_missing_provider_role_embedding():
    client = TestClient(_make_test_app())
    resp = client.get("/_raise_embedding")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "missing_provider"
    assert body["role"] == "embedding"
    assert "no api key" in body["message"]


def test_missing_provider_role_empty_when_unset():
    client = TestClient(_make_test_app())
    resp = client.get("/_raise_no_role")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "missing_provider"
    assert body["role"] == ""
    assert "no api key" in body["message"]
