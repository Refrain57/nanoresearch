#!/usr/bin/env python
"""
One-time migration: import JSONL session files into PostgreSQL.

Usage:
    cd backend
    uv run scripts/migrate_sessions.py --workspace ~/.nanoresearch/workspace --uid admin

Idempotent: sessions already in DB (by session_key) are skipped.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(workspace: Path, uid: str) -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL 环境变量未设置", file=sys.stderr)
        sys.exit(1)

    from nanobot.storage.database import init_engine, get_session_factory
    from nanobot.storage.repositories.conversation_repo import ConversationRepository

    init_engine(database_url)
    factory = get_session_factory()
    repo = ConversationRepository(factory)

    sessions_dir = workspace / "sessions"
    if not sessions_dir.exists():
        print(f"sessions 目录不存在: {sessions_dir}")
        return

    files = sorted(sessions_dir.glob("*.jsonl"))
    if not files:
        print("未找到 JSONL 文件")
        return

    success = skipped = failed = 0

    for path in files:
        try:
            messages = []
            metadata: dict = {}
            created_at = None
            last_consolidated = 0
            key = None

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        key = data.get("key") or path.stem.replace("_", ":", 1)
                        metadata = data.get("metadata", {})
                        raw_ts = data.get("created_at")
                        if raw_ts:
                            from datetime import datetime
                            created_at = datetime.fromisoformat(raw_ts)
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            if not key:
                key = path.stem.replace("_", ":", 1)

            # Idempotency check
            existing = await repo.get_by_session_key(key)
            if existing is not None:
                skipped += 1
                continue

            conv = await repo.create(
                key=key,
                uid=uid,
                metadata=metadata,
                created_at=created_at,
            )
            await repo.replace_messages(conv.id, messages)
            if last_consolidated:
                from datetime import datetime, timezone
                await repo.update_meta(conv.id, last_consolidated, metadata, datetime.now(timezone.utc))

            success += 1
            print(f"  ✓ {key}  ({len(messages)} messages)")

        except Exception as e:
            failed += 1
            print(f"  ✗ {path.name}: {e}")

    print(f"\n迁移完成: 成功 {success} / 跳过 {skipped} (已存在) / 失败 {failed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate JSONL sessions to PostgreSQL")
    parser.add_argument(
        "--workspace",
        default=str(Path.home() / ".nanoresearch" / "workspace"),
        help="Workspace path (default: ~/.nanoresearch/workspace)",
    )
    parser.add_argument("--uid", default="admin", help="User UID to assign sessions to (default: admin)")
    args = parser.parse_args()

    asyncio.run(main(Path(args.workspace).expanduser(), args.uid))
