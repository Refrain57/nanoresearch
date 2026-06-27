"""Search orchestrator — parallel web search, deduplication, and quality scoring."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from nanobot.research.types import (
    ResearchConfig,
    ResearchPlan,
    SearchResult,
    SubQuestion,
)

# Lazy import reranker to avoid dependency issues
_reranker_factory = None


def _get_reranker_factory():
    """Lazy import reranker factory."""
    global _reranker_factory
    if _reranker_factory is None:
        from nanobot.rag.libs.reranker.reranker_factory import RerankerFactory
        _reranker_factory = RerankerFactory
    return _reranker_factory

# High-credibility TLDs and domains (open access / no paywall)
_HIGH_CREDIBILITY = {
    ".gov", ".edu", ".org",
    "arxiv.org", "arxiv.org/abs", "arxiv.org/pdf",
    "nature.com", "pubmed.gov", "scholar.google.com",
    "github.com", "github.com/",  # open source repos
    "stackoverflow.com", "stackexchange.com",
    "dev.to", "medium.com", "blog",
}
# Medium credibility (mostly accessible)
_MEDIUM_CREDIBILITY = {
    "medium.com", "substack.com", "dev.to",
    "stackoverflow.com", "stackexchange.com",
    "npmjs.com", "pypi.org", "crates.io",
    "huggingface.co", "paperswithcode.com",
}
# Domains known to have persistent SSL/connection issues — skip immediately
_KNOWN_DEAD_DOMAINS = {
    "ngrok.cn", "joypage.cn", "vue-js.com",
}

# Blacklisted domains that are never relevant for research
_BLACKLIST_DOMAINS = {
    "google.com", "google.co", "bing.com", "baidu.com", "yahoo.com",
    "apple.com/app", "apps.apple.com", "play.google.com",
    "chrome.google.com", "addons.mozilla.org",
    "amazon.com", "aliexpress.com", "taobao.com", "jd.com",
    "youtube.com/watch", "youtu.be",
    "twitter.com", "x.com/status",
    "facebook.com", "instagram.com", "reddit.com",
    "github.com/issues", "github.com/discussions",
    "grokipedia.com", "junglescout.cn", "appstore.com",
    "openi.cn", "ofweek.com", "adquan.com",
}


def _calculate_credibility(url: str) -> float:
    """Score credibility 0-1 based on domain."""
    url_lower = url.lower()
    for domain in _HIGH_CREDIBILITY:
        if domain in url_lower:
            return 0.9
    for domain in _MEDIUM_CREDIBILITY:
        if domain in url_lower:
            return 0.6
    return 0.4


def _calculate_recency(content: str, publish_date: str | None) -> float:
    """Score recency 0-1 based on publish date or content freshness indicators."""
    if publish_date:
        try:
            # Try common date formats
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y"):
                try:
                    pub = datetime.strptime(publish_date[:10], "%Y-%m-%d")
                    age_days = (datetime.now() - pub).days
                    if age_days <= 90:
                        return 1.0
                    elif age_days <= 365:
                        return 0.8
                    elif age_days <= 730:
                        return 0.6
                    else:
                        return 0.4
                except ValueError:
                    continue
        except Exception:
            pass

    # Heuristic: look for "2024" or "2025" in content
    content_lower = content.lower()
    if re.search(r"202[5-9]", content_lower):
        return 0.9
    if re.search(r"202[3-4]", content_lower):
        return 0.7
    if re.search(r"202[0-2]", content_lower):
        return 0.5
    return 0.4


def _detect_source_type(url: str, content: str) -> str:
    """Classify the source type."""
    url_lower = url.lower()
    if any(kw in url_lower for kw in ["arxiv", "nature", "science", "pubmed", "ieee", "acm"]):
        return "paper"
    if any(kw in url_lower for kw in ["news", "reuters", "bloomberg", "wsj", "nytimes"]):
        return "news"
    if any(kw in url_lower for kw in ["github", "stackoverflow", "dev.to", "npm", "pypi"]):
        return "social"
    if any(kw in url_lower for kw in [".gov", ".edu"]):
        return "official"
    if "blog" in url_lower or len(content) < 500:
        return "blog"
    return "unknown"


def _parse_search_output(output: str) -> list[dict[str, Any]]:
    """Parse the raw text output from WebSearchTool into structured items."""
    items = []
    # Format: "1. Title\n   url\n   snippet"
    blocks = re.split(r"\n(?=\d+\.)", output.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue
        # First line: "N. Title"
        title_match = re.match(r"\d+\.\s+(.+)", lines[0])
        if not title_match:
            continue
        title = title_match.group(1).strip()
        url = ""
        snippet = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("http"):
                url = stripped.split()[0]
            elif stripped and not url:
                # Part of title
                pass
            elif stripped:
                snippet = stripped
        if url:
            items.append({"title": title, "url": url, "content": snippet})
    return items


class SearchOrchestrator:
    """Parallel search across sub-questions with deduplication and scoring."""

    # 并发控制：限制同时进行的子问题搜索和URL抓取数量
    _MAX_CONCURRENT_SUBQ = 2  # 最多同时搜索2个子问题
    _MAX_CONCURRENT_FETCH = 3  # 最多同时抓取3个URL

    def __init__(
        self,
        web_search_tool: Any,
        web_fetch_tool: Any,
        config: ResearchConfig,
        *,
        search_count: int = 10,
    ) -> None:
        self.web_search = web_search_tool
        self.web_fetch = web_fetch_tool
        self.config = config
        self.search_count = search_count  # results per keyword
        self._subq_sem = asyncio.Semaphore(self._MAX_CONCURRENT_SUBQ)
        self._fetch_sem = asyncio.Semaphore(self._MAX_CONCURRENT_FETCH)

        # Cross-iteration deduplication (Bug 2C)
        self._global_seen_urls: set[str] = set()
        # Domain failure counter: SSL/connection errors increment by 3, HTTP errors by 1.
        # Domains with count >= 3 are skipped on subsequent fetches.
        self._failed_domains: dict[str, int] = {}

        # Initialize reranker if enabled
        self._reranker = None
        if config.rerank_enabled and config.rerank_provider != "none":
            self._init_reranker()

    def _init_reranker(self) -> None:
        """Initialize reranker based on config."""
        try:
            RerankerFactory = _get_reranker_factory()

            # Create a minimal settings object for the reranker
            class MinimalSettings:
                def __init__(self, config):
                    self.rerank = type('RerankSettings', (), {
                        'enabled': config.rerank_enabled,
                        'provider': config.rerank_provider,
                        'model': config.rerank_model,
                        'top_k': config.rerank_top_k,
                    })()

            settings = MinimalSettings(self.config)
            self._reranker = RerankerFactory.create(settings)
            logger.info("SearchOrchestrator: initialized reranker with provider={}, model={}",
                       self.config.rerank_provider, self.config.rerank_model)
        except Exception as e:
            logger.warning("SearchOrchestrator: failed to initialize reranker: {}, falling back to scoring only", e)
            self._reranker = None

    async def search(self, plan: ResearchPlan) -> tuple[list[SearchResult], list[dict]]:
        """Run parallel searches for all sub-questions.

        Args:
            plan: ResearchPlan with sub-questions.

        Returns:
            Tuple of (deduplicated results, rerank details).
            rerank_details is a list of {"url": ..., "original_score": ..., "rerank_score": ...}
        """
        # Only search sub-questions that haven't been searched yet (Bug 2B)
        pending = [sq for sq in plan.sub_questions if sq.status == "pending"]
        logger.info(
            "SearchOrchestrator: searching {} pending sub-questions ({} total)",
            len(pending), len(plan.sub_questions),
        )

        if not pending:
            logger.info("SearchOrchestrator: no pending sub-questions, skipping search")
            return [], []

        tasks = [self._search_sub_question(sq) for sq in pending]
        results_per_sq = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[SearchResult] = []
        for idx, result in enumerate(results_per_sq):
            if isinstance(result, Exception):
                logger.warning("SearchOrchestrator: sub-question {} failed: {}", idx, result)
                continue
            all_results.extend(result)

        # Deduplicate: local (within-iteration) + global (cross-iteration, Bug 2C)
        deduped = self._dedupe(all_results)
        logger.info("SearchOrchestrator: {} raw results, {} after dedup", len(all_results), len(deduped))

        rerank_details: list[dict] = []

        # Apply reranking if enabled
        if self._reranker and deduped:
            deduped, rerank_details = self._rerank_results(plan.topic, deduped)

        return deduped, rerank_details

    def _rerank_results(self, query: str, results: list[SearchResult]) -> tuple[list[SearchResult], list[dict]]:
        """Apply reranker to search results.

        Args:
            query: The original research topic (used as rerank query).
            results: List of SearchResult to rerank.

        Returns:
            Tuple of (reranked results, rerank details).
            rerank_details is a list of {"url": ..., "original_score": ..., "rerank_score": ...}
        """
        rerank_details: list[dict] = []

        try:
            # Convert SearchResult to dict format expected by reranker
            candidates = [
                {"id": r.url, "text": r.content, "title": r.title, "url": r.url, "original_score": r.final_score}
                for r in results
            ]

            # Run reranker (synchronous call)
            reranked = self._reranker.rerank(query, candidates, top_k=self.config.rerank_top_k)

            # Map back to SearchResult with updated scores
            url_to_result = {r.url: r for r in results}
            final_results: list[SearchResult] = []

            for item in reranked:
                url = item.get("url") or item.get("id")
                if url in url_to_result:
                    sr = url_to_result[url]
                    original_score = sr.final_score
                    rerank_score = item.get("rerank_score", 0.0)
                    # Update final_score with rerank_score
                    sr.final_score = rerank_score
                    final_results.append(sr)

                    # Record rerank detail
                    rerank_details.append({
                        "url": url,
                        "title": sr.title[:50] if sr.title else "",
                        "original_score": round(original_score, 3),
                        "rerank_score": round(rerank_score, 3),
                    })

            logger.info("SearchOrchestrator: reranked {} results", len(final_results))
            return final_results, rerank_details

        except Exception as e:
            logger.warning("SearchOrchestrator: reranking failed: {}, using original order", e)
            return results, rerank_details

    async def _search_sub_question(self, sq: SubQuestion) -> list[SearchResult]:
        """Search a single sub-question across all keywords."""
        async with self._subq_sem:
            sq.status = "searching"
            results: list[SearchResult] = []

            # Build search queries from keywords
            queries: list[str] = []
            for kw in sq.keywords:
                if kw.strip():
                    queries.append(kw.strip())

            if not queries:
                queries = [sq.question]

            async def _search_one(query: str) -> list[SearchResult]:
                try:
                    output = await self.web_search.execute(query=query, count=self.search_count)
                    items = _parse_search_output(output)
                    sq_results: list[SearchResult] = []
                    for item in items:
                        url = item.get("url", "")
                        if not url:
                            continue
                        # Fetch content with concurrency control
                        async with self._fetch_sem:
                            content = await self._fetch_content(url)
                        sr = self._score_result(item, sq, content)
                        if sr is None:
                            continue
                        sq_results.append(sr)
                    return sq_results
                except Exception as e:
                    logger.warning("SearchOrchestrator: query '{}' failed: {}", query, e)
                    return []

            # Run all keyword searches in parallel
            keyword_results = await asyncio.gather(*[_search_one(q) for q in queries], return_exceptions=True)
            for kr in keyword_results:
                if isinstance(kr, Exception):
                    continue
                results.extend(kr)

            sq.status = "completed"
            sq.results = results
            return results

    def _get_domain(self, url: str) -> str:
        """Extract the netloc/domain from a URL for failure tracking."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except Exception:
            return url

    async def _fetch_content(self, url: str) -> str:
        """Fetch content from a URL using web_fetch tool.

        Skips domains in _KNOWN_DEAD_DOMAINS or whose failure count has reached
        the threshold (>= 3), avoiding repeated 10-30s timeout chains.
        SSL/connection errors count as 3 (permanent); HTTP errors count as 1 (session).
        """
        domain = self._get_domain(url)

        # Skip statically known dead domains
        for dead in _KNOWN_DEAD_DOMAINS:
            if dead in domain:
                logger.debug("SearchOrchestrator: skipping known dead domain {}", domain)
                return ""

        # Skip dynamically detected failing domains
        if self._failed_domains.get(domain, 0) >= 3:
            logger.debug("SearchOrchestrator: skipping failing domain {} (count={})", domain, self._failed_domains[domain])
            return ""

        try:
            raw = await self.web_fetch.execute(url=url, extractMode="text", maxChars=30000)
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                    return data.get("text", "")
                except json.JSONDecodeError:
                    return raw
            elif isinstance(raw, dict):
                return raw.get("text", "")
            return ""
        except Exception as e:
            err_str = str(e).lower()
            # SSL and connection errors are permanent — count as 3 to hit threshold immediately
            if any(kw in err_str for kw in ("ssl", "certificate", "connection refused", "name or service not known")):
                self._failed_domains[domain] = self._failed_domains.get(domain, 0) + 3
                logger.debug("SearchOrchestrator: SSL/connection error for {} (domain count now {})", url, self._failed_domains[domain])
            else:
                self._failed_domains[domain] = self._failed_domains.get(domain, 0) + 1
                logger.debug("SearchOrchestrator: fetch failed for {}: {}", url, e)
            return ""

    def _score_result(self, item: dict[str, Any], sq: SubQuestion, content: str) -> SearchResult | None:
        """Score a raw search result. Returns None if result is blacklisted or content too short."""
        url = item.get("url", "")
        title = item.get("title", "")
        snippet = item.get("content", "")

        # Skip blacklisted domains
        url_lower = url.lower()
        for blocked in _BLACKLIST_DOMAINS:
            if blocked in url_lower:
                return None

        # Skip results with too little content (both snippet and fetch must be useful)
        # Exception: high-credibility sources like arxiv/nature get lower threshold
        total_content = len(snippet) + len(content)
        MIN_CONTENT_LEN = 50  # minimum characters
        if total_content < MIN_CONTENT_LEN:
            # High credibility sources (papers) get a pass with just snippet
            credibility = _calculate_credibility(url)
            if credibility < 0.8 and len(snippet) < 30:
                return None

        # Relevance: how well title/snippet matches keywords
        text_to_score = f"{title} {snippet}".lower()
        keyword_matches = sum(1 for kw in sq.keywords if kw.lower() in text_to_score)
        relevance = min(1.0, keyword_matches / max(1, len(sq.keywords) * 0.5))

        # Boost if topic keywords appear in title
        topic_keywords = sq.question.lower().split()
        for kw in topic_keywords:
            if len(kw) > 3 and kw in title.lower():
                relevance = min(1.0, relevance + 0.2)

        # Credibility from domain
        credibility = _calculate_credibility(url)

        # Recency from content/date hints
        recency = _calculate_recency(f"{snippet} {content}", None)

        # Source type
        source_type = _detect_source_type(url, snippet)

        # Use fetched content if available, otherwise use snippet
        # Combine for richer context
        final_content = content if len(content) > len(snippet) else snippet
        if content and snippet and len(content) < len(snippet):
            # Fetched content was shorter (rate limited/partial), use snippet + whatever we got
            final_content = f"{snippet}\n\n[补充信息]: {content[:500]}"

        sr = SearchResult(
            url=url,
            title=title,
            content=final_content,
            source_type=source_type,
            credibility_score=credibility,
            relevance_score=relevance,
            recency_score=recency,
        )
        # final_score computed in __post_init__
        return sr

    def _dedupe(self, results: list[SearchResult]) -> list[SearchResult]:
        """Deduplicate by URL: local (within-iteration) + global (cross-iteration).

        URLs already seen in previous iterations are dropped immediately.
        New URLs are registered into the global set so future iterations skip them.
        """
        seen_local: dict[str, SearchResult] = {}
        for r in results:
            url = r.url
            # Skip URLs seen in any previous iteration (Bug 2C)
            if url in self._global_seen_urls:
                continue
            if url not in seen_local or r.final_score > seen_local[url].final_score:
                seen_local[url] = r

        # Register all new URLs so future iterations won't re-fetch them
        self._global_seen_urls.update(seen_local.keys())

        # Sort by final_score descending
        sorted_results = sorted(seen_local.values(), key=lambda x: x.final_score, reverse=True)

        # Filter: keep only results with relevance >= 0.15 or high credibility
        MIN_RELEVANCE = 0.15
        filtered = [r for r in sorted_results if r.relevance_score >= MIN_RELEVANCE or r.credibility_score >= 0.8]

        # Additional filter: skip results with very short content (likely rate-limited/empty)
        MIN_CONTENT_CHARS = 80
        quality_filtered = []
        for r in filtered:
            if len(r.content) >= MIN_CONTENT_CHARS:
                quality_filtered.append(r)
            elif r.credibility_score >= 0.8:  # High-credibility sources (papers) get a pass
                quality_filtered.append(r)
        filtered = quality_filtered

        # Limit total
        max_total = self.config.max_sources_per_question * 3
        return filtered[:max_total]
