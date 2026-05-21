"""Unit tests for JWT and password — no database required."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from nanobot.auth.password import hash_password, verify_password
from nanobot.auth.jwt import create_token, verify_token

SECRET = "a" * 64


# ── Password ──────────────────────────────────────────────────────────────────

def test_password_hash_and_verify():
    hashed = hash_password("mysecret")
    assert verify_password("mysecret", hashed)
    assert not verify_password("wrong", hashed)


def test_different_plaintexts_produce_different_hashes():
    h1 = hash_password("password1")
    h2 = hash_password("password2")
    assert h1 != h2


def test_same_plaintext_produces_different_hashes():
    # bcrypt uses a random salt each time
    h1 = hash_password("same")
    h2 = hash_password("same")
    assert h1 != h2
    assert verify_password("same", h1)
    assert verify_password("same", h2)


# ── JWT ───────────────────────────────────────────────────────────────────────

def test_jwt_round_trip(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", SECRET)
    token = create_token("user123")
    assert isinstance(token, str)
    uid = verify_token(token)
    assert uid == "user123"


def test_jwt_missing_secret_raises(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        create_token("user123")


def test_jwt_invalid_token_raises_401(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", SECRET)
    with pytest.raises(HTTPException) as exc:
        verify_token("this.is.garbage")
    assert exc.value.status_code == 401


def test_jwt_wrong_secret_raises_401(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", SECRET)
    token = create_token("user123")
    monkeypatch.setenv("JWT_SECRET_KEY", "b" * 64)
    with pytest.raises(HTTPException) as exc:
        verify_token(token)
    assert exc.value.status_code == 401


def test_jwt_empty_token_raises_401(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", SECRET)
    with pytest.raises(HTTPException) as exc:
        verify_token("")
    assert exc.value.status_code == 401
