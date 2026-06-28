"""JWT token creation and verification."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from jose import JWTError, jwt

ALGORITHM = "HS256"
EXPIRE_DAYS = 7


def _get_secret() -> str:
    key = os.environ.get("JWT_SECRET_KEY")
    if not key:
        raise RuntimeError(
            "JWT_SECRET_KEY 环境变量未设置。"
            "请运行：python -c \"import secrets; print(secrets.token_hex(32))\""
            " 并将结果写入 .env 文件：JWT_SECRET_KEY=<生成的值>"
        )
    return key


def create_token(uid: str) -> str:
    payload = {
        "sub": uid,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=EXPIRE_DAYS),
    }
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def verify_token(token: str) -> str:
    """Verify token and return uid. Raises HTTP 401 on failure."""
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
        uid: str | None = payload.get("sub")
        if not uid:
            raise JWTError("missing sub")
        return uid
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或已过期的 token",
            headers={"WWW-Authenticate": "Bearer"},
        )
