"""Collection and document management tools for Agentic RAG.

Allows Agent to:
- List available collections
- List documents in a collection
- Ingest new documents into the knowledge base
"""

from __future__ import annotations

import asyncio

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
            bm25_dir = resolve_path("~/.nanoresearch/rag/bm25")
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
                    index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{collection}"))
                )
                image_storage = ImageStorage(
                    db_path=str(resolve_path("~/.nanoresearch/rag/image_index.db")),
                    images_root=str(resolve_path("~/.nanoresearch/rag/images"))
                )
                integrity_checker = SQLiteIntegrityChecker(
                    db_path=str(resolve_path("~/.nanoresearch/rag/ingestion_history.db"))
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


class ListResearchKnowledgeTool:
    """MCP tool for research knowledge overview and browsing."""

    @property
    def name(self) -> str:
        return "list_research_knowledge"

    @property
    def description(self) -> str:
        return """Get overview or browse research knowledge (claims and insights).

## Actions

### stats (default)
Returns statistics:
- Count of claims, insights, and chunks
- Domains/topics covered (aggregated from metadata)
- Recent timestamps

### list_claims
List claims with optional filters:
- domain: Filter by applicable domain
- source: Filter by source type ("research", "conversation", "all")
- limit: Max results (default 10)

### list_insights
List insights with optional filters:
- maturity: Filter by maturity ("candidate", "confirmed")
- limit: Max results (default 10)

### browse_domains
List all unique domains from insights' applicable_domains.

### lint_structural
Run structural lint checks (no LLM required):
- orphan_claims: Claims not used in any insight
- broken_refs: Insights referencing non-existent claims
- missing_evidence: Research claims without evidence_ids
- empty_domains: Insights without applicable_domains
- invalid_confidence: Confidence values outside [0, 1]
Returns report only (fix=False).

### lint_semantic
Run semantic lint with LLM (checks if claims are self-contained):
- Requires LLM provider configuration
- Processes conversation-source claims
- Three-tier verdict: keep, demote, delete
- Optional fix: apply demote/delete changes

### lint_all
Run both structural and semantic checks combined.

For actual content retrieval, use retrieve_hybrid with collections=["research_claims", "research_insights"]."""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["stats", "list_claims", "list_insights", "browse_domains", "lint_structural", "lint_semantic", "lint_all"],
                    "default": "stats",
                    "description": "What to do: stats, list_claims, list_insights, browse_domains, lint_structural, lint_semantic, or lint_all",
                },
                "domain": {
                    "type": "string",
                    "description": "Filter by domain (for list_claims/list_insights)",
                },
                "source": {
                    "type": "string",
                    "enum": ["research", "conversation", "all"],
                    "default": "research",
                    "description": "Filter by source type (for list_claims)",
                },
                "maturity": {
                    "type": "string",
                    "enum": ["candidate", "confirmed"],
                    "description": "Filter by maturity (for list_insights)",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max results to return",
                },
                "fix": {
                    "type": "boolean",
                    "default": False,
                    "description": "Apply fixes for semantic lint (demote/delete). Structural lint always runs in report-only mode.",
                },
            },
        }

    async def execute(
        self,
        action: str = "stats",
        domain: Optional[str] = None,
        source: str = "research",
        maturity: Optional[str] = None,
        limit: int = 10,
        fix: bool = False,
    ) -> "MCPToolResponse":
        """Execute the list research knowledge tool."""
        import asyncio

        try:
            from nanobot.rag.core.settings import load_settings
            from nanobot.rag.libs.vector_store.chroma_store import ChromaStore

            settings = load_settings()

            if action == "stats":
                return await self._get_stats(settings)
            elif action == "list_claims":
                return await self._list_claims(settings, domain, source, limit)
            elif action == "list_insights":
                return await self._list_insights(settings, domain, maturity, limit)
            elif action == "browse_domains":
                return await self._browse_domains(settings)
            elif action == "lint_structural":
                return await self._lint_structural(settings, fix=fix)
            elif action == "lint_semantic":
                return await self._lint_semantic(settings, fix=fix)
            elif action == "lint_all":
                return await self._lint_all(settings, fix=fix)
            else:
                return build_json_response({"error": f"Unknown action: {action}"})

        except Exception as e:
            logger.exception(f"list_research_knowledge failed: {e}")
            return build_json_response({"error": str(e)})

    async def _get_stats(self, settings) -> "MCPToolResponse":
        """Get statistics for all research collections."""
        import asyncio

        def get_collection_stats(collection_name: str) -> dict:
            from nanobot.rag.libs.vector_store.chroma_store import ChromaStore
            store = ChromaStore(settings, collection_name=collection_name)
            count = store.count()

            if count == 0:
                return {"count": 0, "domains": [], "recent_timestamps": []}

            # Sample records to extract metadata
            results = store.query(vector=[0] * 1024, top_k=min(count, 100))

            domains = set()
            timestamps = []
            for r in results:
                meta = r.get("metadata", {})
                # Aggregate domains (stored as comma-separated string)
                raw_domains = meta.get("applicable_domains", "")
                if raw_domains:
                    for d in raw_domains.split(","):
                        d = d.strip()
                        if d:
                            domains.add(d)
                # Collect timestamps
                if "created_at" in meta:
                    timestamps.append(meta["created_at"])

            # Sort timestamps, keep recent 5
            timestamps.sort(reverse=True)

            return {
                "count": count,
                "domains": list(domains)[:10],
                "recent_timestamps": timestamps[:5],
            }

        def _get_all_stats():
            return {
                "claims": get_collection_stats("research_claims"),
                "insights": get_collection_stats("research_insights"),
                "chunks": get_collection_stats("research_chunks"),
            }

        stats = await asyncio.to_thread(_get_all_stats)

        return build_json_response({
            "claims": stats["claims"],
            "insights": stats["insights"],
            "chunks": stats["chunks"],
            "usage_hint": "Use retrieve_hybrid with collections=['research_claims', 'research_insights'] to search content",
        })

    async def _list_claims(
        self, settings, domain: Optional[str], source: str, limit: int
    ) -> "MCPToolResponse":
        """List claims with optional filters."""
        import asyncio
        from nanobot.rag.libs.vector_store.chroma_store import ChromaStore

        def _list():
            store = ChromaStore(settings, collection_name="research_claims")
            count = store.count()
            if count == 0:
                return []

            # Build filters
            filters = {}
            if source != "all":
                filters["source"] = source

            # Query with dummy vector to get all records
            results = store.query_by_metadata(filters=filters, limit=limit * 2)

            # Filter by domain if specified
            if domain:
                results = [
                    r for r in results
                    if domain.lower() in str(r.get("metadata", {}).get("applicable_domains", "")).lower()
                ]

            return results[:limit]

        claims = await asyncio.to_thread(_list)

        return build_json_response({
            "action": "list_claims",
            "source": source,
            "domain": domain,
            "claims": [
                {
                    "id": c.get("id"),
                    "text": c.get("text", ""),
                    "source": c.get("metadata", {}).get("source", "research"),
                    "confidence": c.get("metadata", {}).get("confidence", 0),
                    "created_at": c.get("metadata", {}).get("created_at", ""),
                }
                for c in claims
            ],
            "total": len(claims),
        })

    async def _list_insights(
        self, settings, domain: Optional[str], maturity: Optional[str], limit: int
    ) -> "MCPToolResponse":
        """List insights with optional filters."""
        import asyncio
        from nanobot.rag.libs.vector_store.chroma_store import ChromaStore

        def _list():
            store = ChromaStore(settings, collection_name="research_insights")
            count = store.count()
            if count == 0:
                return []

            # Build filters
            filters = {}
            if maturity:
                filters["maturity"] = maturity

            results = store.query_by_metadata(filters=filters, limit=limit * 2)

            # Filter by domain if specified
            if domain:
                results = [
                    r for r in results
                    if domain.lower() in str(r.get("metadata", {}).get("applicable_domains", "")).lower()
                ]

            return results[:limit]

        insights = await asyncio.to_thread(_list)

        def _parse_domains(raw: str) -> list[str]:
            """Parse comma-separated domains string into list."""
            if not raw:
                return []
            return [d.strip() for d in raw.split(",") if d.strip()]

        return build_json_response({
            "action": "list_insights",
            "maturity": maturity,
            "domain": domain,
            "insights": [
                {
                    "id": i.get("id"),
                    "text": i.get("text", ""),
                    "maturity": i.get("metadata", {}).get("maturity", "candidate"),
                    "confidence": i.get("metadata", {}).get("confidence", 0),
                    "domains": _parse_domains(i.get("metadata", {}).get("applicable_domains", "")),
                    "created_at": i.get("metadata", {}).get("created_at", ""),
                }
                for i in insights
            ],
            "total": len(insights),
        })

    async def _browse_domains(self, settings) -> "MCPToolResponse":
        """Browse all unique domains from insights."""
        import asyncio
        from nanobot.rag.libs.vector_store.chroma_store import ChromaStore

        def _browse():
            store = ChromaStore(settings, collection_name="research_insights")
            count = store.count()
            if count == 0:
                return []

            # Get all records
            results = store.get_all_documents()

            # Extract unique domains
            domains = set()
            for r in results:
                meta = r.get("metadata", {})
                applicable_domains = meta.get("applicable_domains", "")
                if applicable_domains:
                    # Split comma-separated domains
                    for d in applicable_domains.split(","):
                        d = d.strip()
                        if d:
                            domains.add(d)

            return sorted(list(domains))

        domains = await asyncio.to_thread(_browse)

        return build_json_response({
            "action": "browse_domains",
            "domains": domains,
            "total": len(domains),
        })

    async def _lint_structural(self, settings, fix: bool = False) -> "MCPToolResponse":
        """Run structural lint checks (no LLM required)."""
        from nanobot.research.knowledge_lint import KnowledgeLint
        from nanobot.research.knowledge_search import KnowledgeSearch

        def _get_ks():
            return KnowledgeSearch.from_settings(settings)

        ks = await asyncio.to_thread(_get_ks)
        lint = KnowledgeLint(ks, provider=None, model=None)
        issues = await lint.lint_structural(fix=False)  # Always report-only

        # Group issues by check type
        by_check = {}
        for issue in issues:
            check = issue.check
            if check not in by_check:
                by_check[check] = []
            by_check[check].append({
                "id": issue.id,
                "verdict": issue.verdict.value,
                "reason": issue.reason,
                "original_confidence": issue.original_confidence,
            })

        return build_json_response({
            "action": "lint_structural",
            "total_issues": len(issues),
            "issues_by_check": by_check,
            "fix_applied": False,
            "note": "Structural lint runs in report-only mode. Use lint_semantic with fix=true to apply changes.",
        })

    async def _lint_semantic(self, settings, fix: bool = False) -> "MCPToolResponse":
        """Run semantic lint with LLM."""
        from nanobot.research.knowledge_lint import KnowledgeLint
        from nanobot.research.knowledge_search import KnowledgeSearch

        def _get_ks():
            return KnowledgeSearch.from_settings(settings)

        ks = await asyncio.to_thread(_get_ks)

        # Get LLM provider from config
        provider = None
        model = "gpt-4o"
        try:
            import json
            from pathlib import Path
            from nanobot.providers.openai_compat_provider import OpenAICompatProvider

            config_path = Path.home() / ".nanoresearch" / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text())
                agents_config = config.get("agents", {}).get("defaults", {})
                providers_config = config.get("providers", {})

                provider_name = agents_config.get("provider", "")
                model = agents_config.get("model", "gpt-4o")

                if provider_name and provider_name in providers_config:
                    pconf = providers_config[provider_name]
                    provider = OpenAICompatProvider(
                        api_key=pconf.get("apiKey", ""),
                        api_base=pconf.get("apiBase", ""),
                    )
        except Exception as e:
            logger.warning(f"Failed to create provider from config: {e}")

        if provider is None:
            return build_json_response({
                "action": "lint_semantic",
                "error": "No LLM provider available. Semantic lint requires a configured LLM provider.",
                "hint": "Ensure the agent has a provider configured in config.json",
            })

        lint = KnowledgeLint(ks, provider=provider, model=model)
        result = await lint.lint_semantic(limit=100)

        # Apply fixes if requested
        demoted = 0
        deleted = 0
        if fix and result.issues:
            demoted, deleted = await lint._apply_fixes(result.issues)

        return build_json_response({
            "action": "lint_semantic",
            "total_checked": result.total_checked,
            "llm_calls": result.total_checked,
            "issues_found": len(result.issues),
            "demoted": demoted if fix else len([i for i in result.issues if i.verdict.value == "demote"]),
            "deleted": deleted if fix else len([i for i in result.issues if i.verdict.value == "delete"]),
            "fix_applied": fix,
            "issues": [
                {
                    "id": i.id,
                    "verdict": i.verdict.value,
                    "reason": i.reason,
                    "original_confidence": i.original_confidence,
                }
                for i in result.issues[:20]  # Limit output
            ],
        })

    async def _lint_all(self, settings, fix: bool = False) -> "MCPToolResponse":
        """Run both structural and semantic lint."""
        from nanobot.research.knowledge_lint import KnowledgeLint
        from nanobot.research.knowledge_search import KnowledgeSearch

        def _get_ks():
            return KnowledgeSearch.from_settings(settings)

        ks = await asyncio.to_thread(_get_ks)

        # Get LLM provider from config
        provider = None
        model = "gpt-4o"
        try:
            import json as _json
            from pathlib import Path as _Path
            from nanobot.providers.openai_compat_provider import OpenAICompatProvider

            config_path = _Path.home() / ".nanoresearch" / "config.json"
            if config_path.exists():
                config = _json.loads(config_path.read_text())
                agents_config = config.get("agents", {}).get("defaults", {})
                providers_config = config.get("providers", {})

                provider_name = agents_config.get("provider", "")
                model = agents_config.get("model", "gpt-4o")

                if provider_name and provider_name in providers_config:
                    pconf = providers_config[provider_name]
                    provider = OpenAICompatProvider(
                        api_key=pconf.get("apiKey", ""),
                        api_base=pconf.get("apiBase", ""),
                    )
        except Exception as e:
            logger.warning(f"Failed to create provider from config: {e}")

        lint = KnowledgeLint(ks, provider=provider, model=model)
        report = await lint.lint_all(fix=fix)

        return build_json_response({
            "action": "lint_all",
            "structural_issues": len(report.structural_issues),
            "semantic_issues": len(report.semantic_issues),
            "total_issues": report.total_issues,
            "kept": report.kept,
            "demoted": report.demoted,
            "deleted": report.deleted,
            "llm_calls": report.llm_calls,
            "fix_applied": fix,
            "structural_by_check": {
                check: len([i for i in report.structural_issues if i.check == check])
                for check in set(i.check for i in report.structural_issues)
            },
        })


class IngestDocumentTool:
    """MCP tool for ingesting a document into the knowledge base.

    This tool runs ingestion in the background and returns immediately
    with a task_id. Use get_task_status to check progress and results.
    """

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
- File path must include the correct extension (.pdf or .md)
- Supported formats: PDF (.pdf), Markdown (.md, .markdown)
- Path can be absolute or relative to ~/.nanoresearch/workspace/
- This tool runs ASYNCHRONOUSLY - it returns a task_id immediately
- Use get_task_status with the task_id to check progress and get results

Args:
    file_path: Path to the file to ingest (must include .pdf or .md extension)
    collection: Collection name (default: 'default')
    pdf_parser: PDF parser to use - "markitdown" (fast, default) or "marker" (high-quality for academic papers, requires GPU)

Returns:
    JSON with:
    - accepted: true (always, if file exists and is valid)
    - task_id: Task ID to query status with get_task_status
    - file_path: The file being processed
    - message: Status message"""

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
                "pdf_parser": {
                    "type": "string",
                    "enum": ["markitdown", "marker"],
                    "default": "marker",
                    "description": "PDF parser: marker (high-quality for academic papers with formulas/tables, requires GPU) or markitdown (fast, general-purpose)",
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force re-processing even if file was already processed",
                },
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        file_path: str,
        collection: str = "default",
        pdf_parser: str = "marker",
        force: bool = False,
    ) -> "MCPToolResponse":
        """Execute the ingest document tool asynchronously."""
        from pathlib import Path
        import time

        logger.info(f"[ingest] Starting with file={file_path}, pdf_parser={pdf_parser}, force={force}")
        t_start = time.time()

        try:
            logger.info("[ingest] Step 1: Import modules...")
            from nanobot.rag.core.settings import load_settings
            from nanobot.rag.ingestion.pipeline import IngestionPipeline
            from nanobot.rag.mcp_server.async_tasks import get_task_manager
            logger.info(f"[ingest] Step 1 done in {time.time()-t_start:.2f}s")

            # Validate file exists FIRST (synchronous, fast)
            logger.info("[ingest] Step 2: Validate file...")
            path = Path(file_path)
            if not path.is_absolute():
                workspace = Path.home() / ".nanoresearch" / "workspace"
                path = workspace / file_path

            if not path.exists():
                return build_json_response({
                    "accepted": False,
                    "file_path": file_path,
                    "error": f"File not found: {path}",
                })

            supported_extensions = {".pdf", ".md", ".markdown", ".mdown"}
            if path.suffix.lower() not in supported_extensions:
                return build_json_response({
                    "accepted": False,
                    "file_path": file_path,
                    "error": f"Unsupported file type: {path.suffix}. Supported: PDF, Markdown (.md)",
                })
            logger.info(f"[ingest] Step 2 done in {time.time()-t_start:.2f}s")

            # Load settings (should be fast, cached)
            logger.info("[ingest] Step 3: Load settings...")
            settings = load_settings()
            logger.info(f"[ingest] Step 3 done in {time.time()-t_start:.2f}s")

            # Get task manager
            logger.info("[ingest] Step 4: Get task manager...")
            task_manager = get_task_manager()
            logger.info(f"[ingest] Step 4 done in {time.time()-t_start:.2f}s")

            # Create a minimal task that just creates Pipeline and runs it in background
            # This ensures all heavy initialization happens in the background thread
            def _ingest():
                logger.info(f"[ingest] Background task started for {path}")
                pipeline = IngestionPipeline(settings, collection=collection, pdf_parser=pdf_parser, force=force)
                return pipeline.run(str(path))

            # Submit to background - this should return immediately
            logger.info("[ingest] Step 5: Submit task...")
            task_id = task_manager.submit(
                task_type="ingest_document",
                func=_ingest,
                metadata={
                    "file_path": str(path),
                    "collection": collection,
                    "pdf_parser": pdf_parser,
                    "force": force,
                }
            )
            logger.info(f"[ingest] Step 5 done in {time.time()-t_start:.2f}s, task_id={task_id}")

            # Return immediately with task_id
            logger.info(f"[ingest] Returning immediately with task_id={task_id}")
            return build_json_response({
                "accepted": True,
                "task_id": task_id,
                "file_path": str(path),
                "collection": collection,
                "pdf_parser": pdf_parser,
                "message": f"Ingestion started. Use get_task_status(task_id='{task_id}') to check progress.",
            })

        except Exception as e:
            import traceback
            logger.exception(f"[ingest] Failed: {e}")
            return build_json_response({
                "accepted": False,
                "file_path": file_path,
                "error": str(e),
            })


class DeleteDocumentTool:
    """MCP tool for deleting a document from the knowledge base."""

    @property
    def name(self) -> str:
        return "delete_document"

    @property
    def description(self) -> str:
        return """删除知识库中的文档。

## 何时使用
- 用户想要从知识库中删除某个文档
- 用户说"删除这个文档"、"从知识库移除这个文件"
- 需要清理过期或错误的文档

## 参数说明
- source_path: 文档的原始文件路径（必填）
- collection: 所属集合名称（默认 "default"）
- source_hash: 可选的文档 SHA-256 哈希值（如已知可提供）

## 返回格式
返回 JSON 格式结果，包含：
- success: 是否成功
- chunks_deleted: 从 ChromaDB 删除的 chunk 数量
- bm25_removed: BM25 索引是否已清理
- images_deleted: 删除的图片数量
- integrity_removed: 摄取记录是否已删除
- errors: 错误信息列表

## 示例
```python
# 删除文档
delete_document(source_path="/path/to/document.pdf")

# 指定集合
delete_document(source_path="/path/to/document.pdf", collection="papers")
```"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "原始文档的文件路径",
                },
                "collection": {
                    "type": "string",
                    "default": "default",
                    "description": "文档所属的集合名称",
                },
                "source_hash": {
                    "type": "string",
                    "description": "可选的文档 SHA-256 哈希值",
                },
            },
            "required": ["source_path"],
        }

    async def execute(
        self,
        source_path: str,
        collection: str = "default",
        source_hash: Optional[str] = None,
    ) -> "MCPToolResponse":
        """Execute the delete document tool."""
        import asyncio

        try:
            from nanobot.rag.core.settings import load_settings, resolve_path
            from nanobot.rag.ingestion.document_manager import DocumentManager
            from nanobot.rag.ingestion.storage.bm25_indexer import BM25Indexer
            from nanobot.rag.ingestion.storage.image_storage import ImageStorage
            from nanobot.rag.libs.vector_store.chroma_store import ChromaStore
            from nanobot.rag.libs.loader.file_integrity import SQLiteIntegrityChecker

            settings = load_settings()

            def _delete():
                # Initialize all required components
                chroma_store = ChromaStore(settings, collection_name=collection)
                bm25_indexer = BM25Indexer(
                    index_dir=str(resolve_path(f"~/.nanoresearch/rag/bm25/{collection}"))
                )
                image_storage = ImageStorage(
                    db_path=str(resolve_path("~/.nanoresearch/rag/image_index.db")),
                    images_root=str(resolve_path("~/.nanoresearch/rag/images"))
                )
                integrity_checker = SQLiteIntegrityChecker(
                    db_path=str(resolve_path("~/.nanoresearch/rag/ingestion_history.db"))
                )

                manager = DocumentManager(
                    chroma_store=chroma_store,
                    bm25_indexer=bm25_indexer,
                    image_storage=image_storage,
                    file_integrity=integrity_checker
                )
                return manager.delete_document(
                    source_path=source_path,
                    collection=collection,
                    source_hash=source_hash,
                )

            result = await asyncio.to_thread(_delete)

            return build_json_response({
                "success": result.success,
                "source_path": source_path,
                "collection": collection,
                "chunks_deleted": result.chunks_deleted,
                "bm25_removed": result.bm25_removed,
                "images_deleted": result.images_deleted,
                "integrity_removed": result.integrity_removed,
                "errors": result.errors,
            })

        except Exception as e:
            logger.exception(f"delete_document failed: {e}")
            return build_json_response({
                "success": False,
                "source_path": source_path,
                "error": str(e),
            })


class DeleteClaimTool:
    """MCP tool for deleting a single claim from knowledge base."""

    @property
    def name(self) -> str:
        return "delete_claim"

    @property
    def description(self) -> str:
        return """删除知识库中的单个 claim。

## 何时使用
- 用户想要删除特定的 claim
- 清理测试数据或错误的 claim
- lint 检查后删除无用的 claim

## 参数说明
- claim_id: 要删除的 claim ID（必填）

## 返回格式
返回 JSON 格式结果，包含：
- success: 是否成功
- claim_id: 被删除的 claim ID
- error: 错误信息（如果失败）

## 示例
```python
delete_claim(claim_id="claim_20260422193246_c70b8b73")
```"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "claim_id": {
                    "type": "string",
                    "description": "要删除的 claim ID",
                },
            },
            "required": ["claim_id"],
        }

    async def execute(self, claim_id: str) -> "MCPToolResponse":
        """Execute the delete claim tool."""
        import asyncio

        try:
            from nanobot.rag.core.settings import load_settings
            from nanobot.rag.libs.vector_store.chroma_store import ChromaStore

            settings = load_settings()

            def _delete():
                store = ChromaStore(settings, collection_name="research_claims")
                store.delete(ids=[claim_id])
                return True

            await asyncio.to_thread(_delete)

            return build_json_response({
                "success": True,
                "claim_id": claim_id,
                "message": f"Claim {claim_id} deleted successfully",
            })

        except Exception as e:
            logger.exception(f"delete_claim failed: {e}")
            return build_json_response({
                "success": False,
                "claim_id": claim_id,
                "error": str(e),
            })


class GetTaskStatusTool:
    """MCP tool for checking the status of background tasks."""

    @property
    def name(self) -> str:
        return "get_task_status"

    @property
    def description(self) -> str:
        return """查询后台任务的状态和结果。

When to use:
- After calling ingest_document, use this to check progress
- Poll this tool until status is "completed" or "failed"
- Get the final result when status becomes "completed"

Args:
    task_id: The task ID returned by ingest_document
    wait: If true and task is still running, wait up to timeout seconds for completion

Returns:
    JSON with:
    - task_id: Task identifier
    - status: "pending", "running", "completed", or "failed"
    - progress: Progress percentage (0.0 to 1.0)
    - message: Status message
    - result: Final result if completed (doc_id, chunk_count, etc.)
    - error: Error message if failed
    - duration: Time taken in seconds"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID returned by ingest_document",
                },
                "wait": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, wait up to 60 seconds for task completion",
                },
            },
            "required": ["task_id"],
        }

    async def execute(
        self,
        task_id: str,
        wait: bool = False,
    ) -> "MCPToolResponse":
        """Execute the get task status tool."""
        try:
            from nanobot.rag.mcp_server.async_tasks import get_task_manager

            task_manager = get_task_manager()
            status = task_manager.get_status(task_id)

            if status is None:
                return build_json_response({
                    "found": False,
                    "task_id": task_id,
                    "error": f"Task {task_id} not found",
                })

            # If wait=True and task is running, wait for completion
            if wait and status["status"] == "running":
                try:
                    result = task_manager.get_result(task_id, timeout=60)
                    status = task_manager.get_status(task_id)
                    # Get the result from the PipelineResult
                    if hasattr(result, 'to_dict'):
                        status["result"] = result.to_dict()
                    else:
                        status["result"] = {
                            "success": getattr(result, 'success', None),
                            "doc_id": getattr(result, 'doc_id', None),
                            "chunk_count": getattr(result, 'chunk_count', None),
                        }
                except TimeoutError:
                    status["message"] = "Task still running after 60s wait"
                except Exception as e:
                    status["error"] = str(e)

            return build_json_response(status)

        except Exception as e:
            logger.exception(f"get_task_status failed: {e}")
            return build_json_response({
                "found": False,
                "task_id": task_id,
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


async def list_research_knowledge_handler(
    action: str = "stats",
    domain: Optional[str] = None,
    source: str = "research",
    maturity: Optional[str] = None,
    limit: int = 10,
    fix: bool = False,
) -> "MCPToolResponse":
    """Handler for list_research_knowledge MCP tool."""
    tool = ListResearchKnowledgeTool()
    return await tool.execute(
        action=action,
        domain=domain,
        source=source,
        maturity=maturity,
        limit=limit,
        fix=fix,
    )


async def ingest_document_handler(
    file_path: str,
    collection: str = "default",
    pdf_parser: str = "markitdown",
    force: bool = False,
) -> "MCPToolResponse":
    """Handler for ingest_document MCP tool."""
    tool = IngestDocumentTool()
    return await tool.execute(file_path=file_path, collection=collection, pdf_parser=pdf_parser, force=force)


async def delete_document_handler(
    source_path: str,
    collection: str = "default",
    source_hash: Optional[str] = None,
) -> "MCPToolResponse":
    """Handler for delete_document MCP tool."""
    tool = DeleteDocumentTool()
    return await tool.execute(
        source_path=source_path,
        collection=collection,
        source_hash=source_hash,
    )


async def delete_claim_handler(claim_id: str) -> "MCPToolResponse":
    """Handler for delete_claim MCP tool."""
    tool = DeleteClaimTool()
    return await tool.execute(claim_id=claim_id)


async def get_task_status_handler(
    task_id: str,
    wait: bool = False,
) -> "MCPToolResponse":
    """Handler for get_task_status MCP tool."""
    tool = GetTaskStatusTool()
    return await tool.execute(task_id=task_id, wait=wait)


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
        name="list_research_knowledge",
        description=ListResearchKnowledgeTool().description,
        input_schema=ListResearchKnowledgeTool().input_schema,
        handler=list_research_knowledge_handler,
    )
    logger.info("Registered agentic tool: list_research_knowledge")

    protocol_handler.register_tool(
        name="ingest_document",
        description=IngestDocumentTool().description,
        input_schema=IngestDocumentTool().input_schema,
        handler=ingest_document_handler,
    )
    logger.info("Registered agentic tool: ingest_document")

    protocol_handler.register_tool(
        name="delete_document",
        description=DeleteDocumentTool().description,
        input_schema=DeleteDocumentTool().input_schema,
        handler=delete_document_handler,
    )
    logger.info("Registered agentic tool: delete_document")

    protocol_handler.register_tool(
        name="delete_claim",
        description=DeleteClaimTool().description,
        input_schema=DeleteClaimTool().input_schema,
        handler=delete_claim_handler,
    )
    logger.info("Registered agentic tool: delete_claim")

    protocol_handler.register_tool(
        name="get_task_status",
        description=GetTaskStatusTool().description,
        input_schema=GetTaskStatusTool().input_schema,
        handler=get_task_status_handler,
    )
    logger.info("Registered agentic tool: get_task_status")
