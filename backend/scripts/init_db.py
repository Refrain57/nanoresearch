#!/usr/bin/env python
"""
Initialize the database: create tables and seed default admin user + agent.

Usage:
    cd backend
    uv run scripts/init_db.py --username admin --password <your-password>
"""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
from pathlib import Path

# Ensure the backend package is importable when run from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(username: str, password: str) -> None:
    # Hint if JWT_SECRET_KEY is missing
    if not os.environ.get("JWT_SECRET_KEY"):
        suggested = secrets.token_hex(32)
        print("\n⚠️  未检测到 JWT_SECRET_KEY，请将以下内容写入 .env 文件：")
        print(f"   JWT_SECRET_KEY={suggested}\n")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL 环境变量未设置", file=sys.stderr)
        sys.exit(1)

    from nanobot.storage.database import init_engine, init_db
    from nanobot.storage.repositories.user_repo import UserRepository
    from nanobot.storage.repositories.agent_repo import AgentRepository
    from nanobot.auth.password import hash_password

    init_engine(database_url)
    await init_db()
    print("✓ 表结构创建完成")

    from nanobot.storage.database import get_session_factory
    factory = get_session_factory()

    # Seed admin user
    user_repo = UserRepository(factory)
    if not await user_repo.exists(username):
        await user_repo.create(
            uid=username,
            password_hash=hash_password(password),
            role="admin",
        )
        print(f"✓ 创建用户: {username}")
    else:
        print(f"  用户已存在，跳过: {username}")

    # Seed default agent
    agent_repo = AgentRepository(factory)
    if not await agent_repo.default_exists():
        # Try to read model from config
        model = os.environ.get("DEFAULT_MODEL", "anthropic/claude-opus-4-5")
        provider = os.environ.get("DEFAULT_PROVIDER", "anthropic")
        await agent_repo.create({
            "name": "Nano Research",
            "description": "深度研究智能体，支持联网搜索、代码执行、知识库检索",
            "version": "1.0.0",
            "capabilities": {
                "streaming": True,
                "web_search": True,
                "code_execution": True,
                "knowledge_base": True,
                "deep_research": True,
            },
            "default_model": model,
            "provider": provider,
            "is_default": True,
            "created_by": username,
        })
        print(f"✓ 创建默认 agent (model={model})")
    else:
        print("  默认 agent 已存在，跳过")

    print("\n✅ 初始化完成")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize NanoResearch database")
    parser.add_argument("--username", default="admin", help="Admin username (default: admin)")
    parser.add_argument("--password", required=True, help="Admin password")
    args = parser.parse_args()

    asyncio.run(main(args.username, args.password))
