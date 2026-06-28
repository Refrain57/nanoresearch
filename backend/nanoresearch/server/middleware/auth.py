"""JWT auth dependency for FastAPI endpoints."""

from __future__ import annotations

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    from nanoresearch.auth.jwt import verify_token
    return verify_token(token)
