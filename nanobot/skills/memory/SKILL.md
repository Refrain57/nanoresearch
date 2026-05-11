---
name: memory
description: Two-layer memory system with RAG-based recall.
always: true
---

# Memory

## Structure

- `memory/MEMORY.md` — Long-term facts (preferences, project context, relationships). Always loaded into your context.
- `user_memory` collection (ChromaDB) — Conversation history. Retrieved via RAG based on relevance.

## How Memory Works

### Long-term Facts (MEMORY.md)
Write important facts immediately using `edit_file` or `write_file`:
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")

### Conversation Memory (RAG)
User memory is automatically extracted from conversations and stored in ChromaDB.
Relevant memories are retrieved based on the conversation topic — no manual search needed.

## Auto-consolidation

Old conversations are automatically processed:
1. Important facts → extracted to MEMORY.md
2. Conversation knowledge → stored in user_memory collection (RAG-retrievable)

You don't need to manage this process.
