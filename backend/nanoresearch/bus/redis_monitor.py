"""Background monitor for Redis health and per-prefix memory usage.

3-A: Poll INFO stats every stats_interval seconds; fire WARNING if evicted_keys delta > 0.
3-B: Every memory_interval seconds, SCAN a sample of keys per prefix via pipeline
     MEMORY USAGE, then append a JSON Lines record to logs/redis_metrics.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from nanoresearch.bus.redis_client import get_redis

logger = logging.getLogger(__name__)

_PREFIXES = [
    "session:msg:",
    "session:meta:",
    "agent:",
    "user_settings:",
    "kb:meta:",
    "embedding:",
    "chunk:",
    "pending:",
    "cancel:",
    "job:",
    "run_events:",
    "chat_events:",
]

_SAMPLE_SIZE = 50

# parents[0]=bus/, parents[1]=nanoresearch/, parents[2]=backend/
_DEFAULT_METRICS_PATH = Path(__file__).resolve().parents[2] / "logs" / "redis_metrics.jsonl"


class RedisMonitor:
    """Periodically poll Redis eviction stats and sample per-prefix memory usage."""

    def __init__(
        self,
        stats_interval: int = 60,
        memory_interval: int = 300,
        sample_size: int = _SAMPLE_SIZE,
        metrics_path: Path | None = None,
    ) -> None:
        self._stats_interval = stats_interval
        self._memory_interval = memory_interval
        self._sample_size = sample_size
        env_path = os.environ.get("REDIS_METRICS_PATH", "")
        self._metrics_path = Path(env_path) if env_path else (metrics_path or _DEFAULT_METRICS_PATH)
        self._task: asyncio.Task[Any] | None = None
        self._last_evicted: int | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        last_memory_check = 0.0
        while True:
            try:
                await self._check_stats()
            except Exception:
                logger.exception("RedisMonitor: stats poll failed")

            now = time.monotonic()
            if now - last_memory_check >= self._memory_interval:
                try:
                    await self._scan_memory()
                    last_memory_check = now
                except Exception:
                    logger.exception("RedisMonitor: memory scan failed")

            await asyncio.sleep(self._stats_interval)

    async def _check_stats(self) -> None:
        """Poll INFO stats; emit WARNING if evicted_keys delta > 0 (3-A)."""
        redis = get_redis()
        info: dict[str, Any] = await redis.info("stats")
        current_evicted = int(info.get("evicted_keys", 0))
        if self._last_evicted is not None:
            delta = current_evicted - self._last_evicted
            if delta > 0:
                logger.warning(
                    "RedisMonitor: eviction alert — %d key(s) evicted in last %ds "
                    "(total evicted_keys=%d)",
                    delta,
                    self._stats_interval,
                    current_evicted,
                )
        self._last_evicted = current_evicted

    async def _scan_memory(self) -> None:
        """SCAN + pipeline MEMORY USAGE per prefix; write JSON Lines record (3-B)."""
        redis = get_redis()
        ts = time.time()
        results: list[dict[str, Any]] = []

        for prefix in _PREFIXES:
            sample_keys: list[str] = []
            cursor: Any = "0"
            while len(sample_keys) < self._sample_size:
                cursor, keys = await redis.scan(
                    cursor,
                    match=f"{prefix}*",
                    count=self._sample_size,
                )
                sample_keys.extend(keys)
                if cursor == 0:
                    break
            sample_keys = sample_keys[: self._sample_size]

            total_bytes = 0
            if sample_keys:
                try:
                    async with redis.pipeline(transaction=False) as pipe:
                        for key in sample_keys:
                            pipe.memory_usage(key)
                        mem_results = await pipe.execute()
                    for mem in mem_results:
                        if mem is not None:
                            total_bytes += mem
                except Exception:
                    pass

            avg_bytes = total_bytes // len(sample_keys) if sample_keys else 0
            results.append({
                "prefix": prefix,
                "sampled_keys": len(sample_keys),
                "total_sample_bytes": total_bytes,
                "avg_bytes_per_key": avg_bytes,
            })

        await asyncio.to_thread(self._write_metrics, ts, results)

    def _write_metrics(self, ts: float, results: list[dict[str, Any]]) -> None:
        """Append a single JSON Lines record to the metrics file (called via to_thread)."""
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": ts,
            "type": "redis_memory_scan",
            "prefixes": results,
        }
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            logger.info(
                "RedisMonitor: memory scan complete — %d prefixes written to %s",
                len(results),
                self._metrics_path,
            )
        except OSError:
            logger.exception("RedisMonitor: failed to write metrics file")
