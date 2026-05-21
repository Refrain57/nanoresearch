"""Cache metrics tracking for prompt caching optimization."""

from dataclasses import dataclass, field
from typing import Any, Optional
from loguru import logger
import time


@dataclass
class CacheMetrics:
    """Cache performance metrics container."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_creation_tokens: int = 0
    total_read_tokens: int = 0
    _last_log_time: float = field(default_factory=time.time)
    _log_interval: int = 10  # Log every N requests

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def avg_creation_tokens(self) -> float:
        """Average tokens per cache creation (cache miss)."""
        if self.cache_misses == 0:
            return 0.0
        return self.total_creation_tokens / self.cache_misses

    @property
    def avg_read_tokens(self) -> float:
        """Average tokens per cache read (cache hit)."""
        if self.cache_hits == 0:
            return 0.0
        return self.total_read_tokens / self.cache_hits

    @property
    def savings_ratio(self) -> float:
        """Token savings ratio from caching."""
        total = self.total_creation_tokens + self.total_read_tokens
        if total == 0:
            return 0.0
        return self.total_read_tokens / total

    def record(self, cache_read_tokens: int, cache_creation_tokens: int) -> None:
        """Record cache stats from a single request.

        Args:
            cache_read_tokens: Tokens read from cache (cache hit)
            cache_creation_tokens: Tokens used to create cache (cache miss)
        """
        self.total_requests += 1

        if cache_read_tokens > 0:
            self.cache_hits += 1
            self.total_read_tokens += cache_read_tokens
        else:
            self.cache_misses += 1
            self.total_creation_tokens += cache_creation_tokens

        # Periodic logging
        if self.total_requests % self._log_interval == 0:
            self.log_summary()

    def log_summary(self) -> None:
        """Log current cache metrics summary."""
        logger.info(
            "Cache Metrics [requests={}]: hit_rate={:.1%}, hits={}, misses={}, "
            "avg_creation={:.0f}tokens, avg_read={:.0f}tokens, savings={:.1%}",
            self.total_requests,
            self.hit_rate,
            self.cache_hits,
            self.cache_misses,
            self.avg_creation_tokens,
            self.avg_read_tokens,
            self.savings_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "total_creation_tokens": self.total_creation_tokens,
            "total_read_tokens": self.total_read_tokens,
            "avg_creation_tokens": round(self.avg_creation_tokens, 2),
            "avg_read_tokens": round(self.avg_read_tokens, 2),
            "savings_ratio": round(self.savings_ratio, 4),
        }


# Global instance for easy import
_global_metrics: Optional[CacheMetrics] = None


def get_cache_metrics() -> CacheMetrics:
    """Get or create the global cache metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = CacheMetrics()
    return _global_metrics


def record_cache_stats(usage: dict[str, Any]) -> None:
    """Record cache stats from LLM response usage.

    Args:
        usage: Usage dict from LLM response, should contain:
            - cache_read_input_tokens: tokens read from cache
            - cache_creation_input_tokens: tokens used to create cache
    """
    if not usage:
        return

    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_creation = usage.get("cache_creation_input_tokens", 0)

    if cache_read > 0 or cache_creation > 0:
        get_cache_metrics().record(cache_read, cache_creation)