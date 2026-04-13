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

# High-credibility TLDs and domains
_HIGH_CREDIBILITY = {".gov", ".edu", ".org", "arxiv.org", "nature.com", "sciencedirect.com", "pubmed.gov", "scholar.google.com"}
_MEDIUM_CREDIBILITY = {"medium.com", "substack.com", "dev.to", "stackoverflow.com", "github.com", "zhihu.com"}

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

    async def search(self, plan: ResearchPlan) -> list[SearchResult]:
        """Run parallel searches for all sub-questions.

        Args:
            plan: ResearchPlan with sub-questions.

        Returns:
            Deduplicated, scored list of SearchResults.
        """
        logger.info("SearchOrchestrator: searching {} sub-questions", len(plan.sub_questions))

        tasks = [self._search_sub_question(sq) for sq in plan.sub_questions]
        results_per_sq = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[SearchResult] = []
        for idx, result in enumerate(results_per_sq):
            if isinstance(result, Exception):
                logger.warning("SearchOrchestrator: sub-question {} failed: {}", idx, result)
                continue
            all_results.extend(result)

        # Deduplicate by URL
        deduped = self._dedupe(all_results)
        logger.info("SearchOrchestrator: {} raw results, {} after dedup", len(all_results), len(deduped))
        return deduped

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

    async def _fetch_content(self, url: str) -> str:
        """Fetch content from a URL using web_fetch tool."""
        try:
            raw = await self.web_fetch.execute(url=url, extractMode="text", maxChars=30000)
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                    return data.get("text", "")
                except json.JSONDecodeError:
                    # Return full raw content, not truncated
                    return raw
            elif isinstance(raw, dict):
                return raw.get("text", "")
            return ""
        except Exception as e:
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
        """Deduplicate by URL, keeping highest-scoring entry per URL. Filters low-relevance."""
        seen: dict[str, SearchResult] = {}
        for r in results:
            url = r.url
            if url not in seen or r.final_score > seen[url].final_score:
                seen[url] = r

        # Sort by final_score descending
        sorted_results = sorted(seen.values(), key=lambda x: x.final_score, reverse=True)

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
