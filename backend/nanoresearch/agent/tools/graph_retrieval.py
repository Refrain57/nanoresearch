"""Graph-based entity retrieval tool — cross-document concept tracking via KG."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanoresearch.agent.tools.base import Tool

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker


class RetrieveByEntityTool(Tool):
    """Retrieve chunks that mention a specific entity across the knowledge base."""

    name = "retrieve_by_entity"
    side_effect = False  # read-only KG query
    description = (
        "当你在已检索内容中发现一个关键概念或实体，需要了解该概念在"
        "其他文档中如何被论述时，使用此工具。"
        "它通过知识图谱精确追踪实体出现过的所有 chunk，不依赖向量相似度，"
        "适合跨文档概念追踪和多跳推理。"
        "需要先用 retrieve_hybrid 找到初步相关内容，识别关键实体后再用此工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": "要追踪的实体名称（自然语言，会做归一化处理）",
            },
            "collection": {
                "type": "string",
                "description": "知识库集合名称，格式为 {uid}_{kb_uuid}",
            },
            "top_k": {
                "type": "integer",
                "description": "返回的最大 chunk 数量（默认 10）",
                "default": 10,
            },
        },
        "required": ["entity_name", "collection"],
    }

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def execute(
        self,
        entity_name: str,
        collection: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> str:
        from nanoresearch.storage.repositories.graph_repo import GraphRepository
        from nanoresearch.storage.repositories.knowledge_repo import KnowledgeRepository

        # Extract kb_id UUID from collection name: format is {uid}_{kb_uuid}
        import re as _re
        m = _re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            collection,
            _re.IGNORECASE,
        )
        if not m:
            return f"错误：无法从集合名 '{collection}' 中解析知识库 ID"
        kb_id = uuid.UUID(m.group())

        graph_repo = GraphRepository(self._session_factory)
        knowledge_repo = KnowledgeRepository(self._session_factory)

        try:
            chunk_ids = await graph_repo.get_chunks_by_entity_name(kb_id, entity_name)
        except Exception as e:
            logger.error("retrieve_by_entity: graph query failed: {}", e)
            return f"错误：图谱查询失败 — {e}"

        if not chunk_ids:
            return f"知识图谱中未找到实体「{entity_name}」的相关内容（可能尚未建图，请先执行图构建）"

        limited_ids = chunk_ids[:top_k]
        try:
            chunks = await knowledge_repo.get_chunks_by_ids(limited_ids)
        except Exception as e:
            logger.error("retrieve_by_entity: chunk fetch failed: {}", e)
            return f"错误：获取 chunk 内容失败 — {e}"

        if not chunks:
            return f"找到实体提及记录，但无法加载对应 chunk 内容"

        lines = [f"实体「{entity_name}」在知识库中出现于 {len(chunk_ids)} 个片段（返回前 {len(chunks)} 个）：\n"]
        for i, chunk in enumerate(chunks, 1):
            source = (chunk.chunk_metadata or {}).get("source_path", str(chunk.document_id or chunk.id))
            lines.append(f"--- 片段 {i} (来源: {source}) ---")
            lines.append(chunk.content)
            lines.append("")

        return "\n".join(lines)
