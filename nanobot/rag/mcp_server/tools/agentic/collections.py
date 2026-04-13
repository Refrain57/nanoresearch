"""Collection and document management tools for Agentic RAG.

Allows Agent to:
- List available collections
- List documents in a collection
- Ingest new documents into the knowledge base
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from nanobot.rag.mcp_server.tools.agentic.shared import build_json_response

logger = logging.getLogger(__name__)


class ListCollectionsTool:
    """MCP tool for listing available collections."""

    def __init__(self):
        self._initialized = False
        self._collections = []

    def _ensure_initialized(self) -> None:
        """Initialize by querying ChromaDB for collections."""
        if self._initialized:
            return

        try:
            from nanobot.rag.core.settings import load_settings, resolve_path
            from nanobot.rag.libs.vector_store.chroma_store import ChromaStore

            settings = load_settings()

            # Reuse ChromaStore's client cache to avoid conflicts
            # ChromaDB allows only one PersistentClient per path
            # Creating a ChromaStore instance populates _client_cache
            temp_store = ChromaStore(settings)
            client_key = str(temp_store.persist_directory)
            client = ChromaStore._client_cache.get(client_key)

            if client:
                for coll in client.list_collections():
                    self._collections.append({
                        "name": coll.name,
                        "type": "chroma",
                        "metadata": coll.metadata or {},
                        "count": coll.count() if hasattr(coll, "count") else None,
                    })

            # Check BM25 indices
            bm25_dir = resolve_path("~/.nanobot/rag/bm25")
            if bm25_dir.exists():
                for item in bm25_dir.iterdir():
                    if item.is_dir():
                        existing = [c for c in self._collections if c["name"] == item.name]
                        if not existing:
                            self._collections.append({
                                "name": item.name,
                                "type": "bm25_only",
                                "path": str(item),
                            })

            self._initialized = True
            logger.info(f"Found {len(self._collections)} collections via ChromaDB API")

        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
            self._initialized = True

    @property
    def name(self) -> str:
        return "list_collections"

    @property
    def description(self) -> str:
        return """List all available knowledge base collections.

When to use:
- Before retrieval, to see what collections exist
- When user asks 'what do you know about' or 'what's in the knowledge base'
- To discover available data sources

Returns:
    JSON with list of collections, each containing:
    - name: Collection identifier
    - type: Storage type (chroma, bm25_only)
    - path: Storage location"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self) -> "MCPToolResponse":
        """Execute the list collections tool."""
        import asyncio

        await asyncio.to_thread(self._ensure_initialized)

        return build_json_response({
            "collections": self._collections,
            "total": len(self._collections),
        })


class ListDocumentsTool:
    """MCP tool for listing documents in a collection."""

    def __init__(self):
        self._initialized = False

    @property
    def name(self) -> str:
        return "list_documents"

    @property
    def description(self) -> str:
        return """List documents in a specific collection.

When to use:
- After list_collections, to see documents in a collection
- When user asks about specific collection contents
- To verify if a document was ingested

Args:
    collection: Collection name (default: 'default')

Returns:
    JSON with list of documents, each containing:
    - doc_id: Document identifier
    - source_path: Original file path
    - chunk_count: Number of chunks
    - metadata: Document metadata"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "default": "default",
                    "description": "Collection name to list documents from",
                },
            },
        }

    async def execute(self, collection: str = "default") -> "MCPToolResponse":
        """Execute the list documents tool."""
        import asyncio

        try:
            from nanobot.rag.core.settings import load_settings, resolve_path
            from nanobot.rag.ingestion.document_manager import DocumentManager
            from nanobot.rag.ingestion.storage.bm25_indexer import BM25Indexer
            from nanobot.rag.ingestion.storage.image_storage import ImageStorage
            from nanobot.rag.libs.vector_store.chroma_store import ChromaStore
            from nanobot.rag.libs.loader.file_integrity import SQLiteIntegrityChecker

            settings = load_settings()

            def _list():
                # Initialize all required components
                chroma_store = ChromaStore(settings, collection_name=collection)
                bm25_indexer = BM25Indexer(
                    index_dir=str(resolve_path(f"~/.nanobot/rag/bm25/{collection}"))
                )
                image_storage = ImageStorage(
                    db_path=str(resolve_path("~/.nanobot/rag/image_index.db")),
                    images_root=str(resolve_path("~/.nanobot/rag/images"))
                )
                integrity_checker = SQLiteIntegrityChecker(
                    db_path=str(resolve_path("~/.nanobot/rag/ingestion_history.db"))
                )

                manager = DocumentManager(
                    chroma_store=chroma_store,
                    bm25_indexer=bm25_indexer,
                    image_storage=image_storage,
                    file_integrity=integrity_checker
                )
                return manager.list_documents(collection)

            documents = await asyncio.to_thread(_list)

            # Format response
            doc_list = []
            for doc in documents:
                doc_list.append({
                    "source_path": doc.source_path,
                    "source_hash": doc.source_hash,
                    "collection": doc.collection,
                    "chunk_count": doc.chunk_count,
                    "image_count": doc.image_count,
                    "processed_at": doc.processed_at,
                })

            return build_json_response({
                "collection": collection,
                "documents": doc_list,
                "total": len(doc_list),
            })

        except Exception as e:
            logger.exception(f"list_documents failed: {e}")
            return build_json_response({
                "collection": collection,
                "documents": [],
                "total": 0,
                "error": str(e),
            })


class IngestDocumentTool:
    """MCP tool for ingesting a document into the knowledge base."""

    def __init__(self):
        self._initialized = False

    @property
    def name(self) -> str:
        return "ingest_document"

    @property
    def description(self) -> str:
        return """Ingest a document (PDF or Markdown) into the knowledge base.

When to use:
- User wants to add a document to the knowledge base
- User says 'learn this document', 'add this file', 'import this PDF/Markdown'
- After ingesting, you can retrieve from it with retrieve_hybrid

IMPORTANT:
- File path must be absolute or relative to workspace
- Supported formats: PDF, Markdown (.md, .markdown)
- Markdown files are ideal for structure-aware chunking (preserves heading hierarchy)
- Large files may take time to process (chunking + embedding)

Args:
    file_path: Path to the file to ingest (PDF or Markdown)
    collection: Collection name (default: 'default')

Returns:
    JSON with:
    - success: true/false
    - doc_id: Document identifier (SHA256 hash)
    - chunk_count: Number of chunks created
    - error: Error message if failed"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file to ingest",
                },
                "collection": {
                    "type": "string",
                    "default": "default",
                    "description": "Collection name to store the document",
                },
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        file_path: str,
        collection: str = "default",
    ) -> "MCPToolResponse":
        """Execute the ingest document tool."""
        import asyncio
        from pathlib import Path

        try:
            from nanobot.rag.core.settings import load_settings
            from nanobot.rag.ingestion.pipeline import IngestionPipeline

            # Validate file exists
            path = Path(file_path)
            if not path.is_absolute():
                # Try relative to workspace
                settings = load_settings()
                workspace = Path(settings.agents.defaults.workspace).expanduser()
                path = workspace / file_path

            if not path.exists():
                return build_json_response({
                    "success": False,
                    "file_path": file_path,
                    "error": f"File not found: {path}",
                })

            # Check supported file types
            supported_extensions = {".pdf", ".md", ".markdown", ".mdown"}
            if path.suffix.lower() not in supported_extensions:
                return build_json_response({
                    "success": False,
                    "file_path": file_path,
                    "error": f"Unsupported file type: {path.suffix}. Supported: PDF, Markdown (.md)",
                })

            # Run ingestion pipeline
            settings = load_settings()

            def _ingest():
                pipeline = IngestionPipeline(settings, collection=collection)
                return pipeline.run(str(path))

            result = await asyncio.to_thread(_ingest)

            return build_json_response({
                "success": result.success,
                "file_path": file_path,
                "doc_id": result.doc_id,
                "chunk_count": result.chunk_count,
                "collection": collection,
                "error": result.error,
            })

        except Exception as e:
            logger.exception(f"ingest_document failed: {e}")
            return build_json_response({
                "success": False,
                "file_path": file_path,
                "error": str(e),
            })


# MCP Tool Handlers
async def list_collections_handler() -> "MCPToolResponse":
    """Handler for list_collections MCP tool."""
    tool = ListCollectionsTool()
    return await tool.execute()


async def list_documents_handler(collection: str = "default") -> "MCPToolResponse":
    """Handler for list_documents MCP tool."""
    tool = ListDocumentsTool()
    return await tool.execute(collection=collection)


async def ingest_document_handler(
    file_path: str,
    collection: str = "default",
) -> "MCPToolResponse":
    """Handler for ingest_document MCP tool."""
    tool = IngestDocumentTool()
    return await tool.execute(file_path=file_path, collection=collection)


def register_tools(protocol_handler: Any) -> None:
    """Register collection management tools."""
    protocol_handler.register_tool(
        name="list_collections",
        description=ListCollectionsTool().description,
        input_schema=ListCollectionsTool().input_schema,
        handler=list_collections_handler,
    )
    logger.info("Registered agentic tool: list_collections")

    protocol_handler.register_tool(
        name="list_documents",
        description=ListDocumentsTool().description,
        input_schema=ListDocumentsTool().input_schema,
        handler=list_documents_handler,
    )
    logger.info("Registered agentic tool: list_documents")

    protocol_handler.register_tool(
        name="ingest_document",
        description=IngestDocumentTool().description,
        input_schema=IngestDocumentTool().input_schema,
        handler=ingest_document_handler,
    )
    logger.info("Registered agentic tool: ingest_document")
