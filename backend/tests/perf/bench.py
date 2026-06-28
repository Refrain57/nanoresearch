"""
Nanoresearch infrastructure benchmark — LLM calls mocked out.

Three layers:
  A  Redis Stream roundtrip       (xadd → xread, pure Redis, no HTTP)
  B  Agent config cache hit P99   (hgetall under concurrent load, no HTTP)
  C  Full pipeline E2E            (HTTP POST /api/runs → ARQ → mock worker → SSE)

Layer C requires:
  1. server running  (uvicorn nanoresearch.server.main:app)
  2. worker running  (arq nanoresearch.worker.WorkerSettings)
  3. worker has the __PERF_TEST__ bypass (already patched in worker.py)
  4. AUTH_TOKEN and BASE_URL set below (or via env)

Run:
  cd backend
  python -m tests.perf.bench            # A + B only (no server needed)
  python -m tests.perf.bench --e2e      # A + B + C
  python -m tests.perf.bench --layer A  # single layer
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid

# ── config ───────────────────────────────────────────────────────────────────
REDIS_URL  = os.getenv("REDIS_URL",  "redis://localhost:6379")
BASE_URL   = os.getenv("BASE_URL",   "http://localhost:8000")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")   # Bearer token for the HTTP API


# ── stats ─────────────────────────────────────────────────────────────────────
def _pct(data: list[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def print_stats(label: str, ms: list[float]) -> None:
    if not ms:
        print(f"\n{label}: no data")
        return
    print(f"\n{'─'*55}")
    print(f"  {label}  (n={len(ms)})")
    print(f"  P50  = {_pct(ms, 50):.2f} ms")
    print(f"  P95  = {_pct(ms, 95):.2f} ms")
    print(f"  P99  = {_pct(ms, 99):.2f} ms")
    if len(ms) >= 1000:
        print(f"  P999 = {_pct(ms, 99.9):.2f} ms")
    print(f"  max  = {max(ms):.2f} ms")


# ── Layer A: Redis Stream roundtrip ───────────────────────────────────────────
async def bench_stream(n: int = 1000) -> None:
    import redis.asyncio as aioredis
    from nanoresearch.bus.stream import xadd_event, xread_next

    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    latencies: list[float] = []

    for _ in range(n):
        key = f"run_events:bench-{uuid.uuid4()}"
        t0 = time.perf_counter()
        await xadd_event(r, key, {"type": "message_delta", "chunk": "x"})
        await xread_next(r, key, "0-0", timeout_ms=2000)
        latencies.append((time.perf_counter() - t0) * 1000)
        await r.delete(key)

    await r.aclose()
    print_stats("Layer A — Redis Stream xadd→xread roundtrip", latencies)


# ── Layer B: Agent config cache hit under concurrency ────────────────────────
async def bench_cache(concurrency: int = 50, per_worker: int = 200) -> None:
    """
    Pre-populate 10 fake agents in Redis, then hammer hgetall with
    `concurrency` coroutines each doing `per_worker` reads.
    Tests whether P99 degrades as connection pool saturates.
    """
    import redis.asyncio as aioredis
    from nanoresearch.bus.redis_keys import RedisKeys

    # 共享连接池：max_connections 设为并发数的一半足够（asyncio 不会真的同时占用所有连接）
    pool = aioredis.ConnectionPool.from_url(
        REDIS_URL, decode_responses=True, max_connections=max(64, concurrency // 2)
    )
    r = aioredis.Redis(connection_pool=pool)

    agent_ids = [str(uuid.uuid4()) for _ in range(10)]
    fake_hash = {
        "id": "placeholder", "name": "perf-agent", "description": "",
        "persona": "", "default_model": "gpt-4o", "is_default": "false",
        "skills_config": "[]", "tools_config": "[]", "harness": "{}",
        "capabilities": "{}", "max_iterations": "40", "version": "1.0.0",
        "provider": "", "created_by": "",
    }
    for aid in agent_ids:
        await r.hset(RedisKeys.agent(aid), mapping={**fake_hash, "id": aid})
        await r.expire(RedisKeys.agent(aid), 600)

    all_latencies: list[float] = []

    async def worker(idx: int) -> None:
        lats: list[float] = []
        for i in range(per_worker):
            aid = agent_ids[(idx * per_worker + i) % len(agent_ids)]
            t0 = time.perf_counter()
            await r.hgetall(RedisKeys.agent(aid))
            lats.append((time.perf_counter() - t0) * 1000)
        all_latencies.extend(lats)

    await asyncio.gather(*[worker(i) for i in range(concurrency)])

    for aid in agent_ids:
        await r.delete(RedisKeys.agent(aid))
    await r.aclose()

    total = concurrency * per_worker
    print_stats(
        f"Layer B — agent cache hgetall  ({concurrency} concurrent, {total} total)",
        all_latencies,
    )


# ── Layer C: Full pipeline E2E ────────────────────────────────────────────────
async def bench_e2e(
    concurrency: int = 10,
    iterations: int = 5,
    mock_chunks: int = 5,
) -> None:
    """
    Each iteration fires `concurrency` parallel runs through the real HTTP
    stack: POST /api/runs → ARQ → mock worker → SSE run_end.

    Measures wall-clock from POST to receiving run_end.
    A single shared AsyncClient is used so connection count stays bounded
    under high concurrency (e.g. 1000 parallel coroutines).
    """
    try:
        import httpx
    except ImportError:
        print("\nLayer C skipped — install httpx:  pip install httpx")
        return

    if not AUTH_TOKEN:
        print("\nLayer C skipped — set AUTH_TOKEN env var")
        return

    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    # Cap TCP connections; asyncio multiplexes many coroutines over fewer sockets.
    limits = httpx.Limits(max_connections=min(concurrency, 500), max_keepalive_connections=100)

    async with httpx.AsyncClient(
        base_url=BASE_URL, headers=headers,
        timeout=httpx.Timeout(connect=10, read=120, write=30, pool=30),
        limits=limits,
    ) as client:
        r = await client.post("/api/conversations", json={"title": "perf-bench"})
        r.raise_for_status()
        conv_id = r.json()["id"]
        print(f"\nLayer C — using conversation {conv_id}  (concurrency={concurrency}, iters={iterations})")

        latencies: list[float] = []
        errors = 0

        async def one_run() -> None:
            nonlocal errors
            content = f"__PERF_TEST__:{mock_chunks}:{uuid.uuid4()}"
            t0 = time.perf_counter()
            try:
                r = await client.post("/api/runs", json={
                    "conversation_id": conv_id,
                    "content": content,
                })
                r.raise_for_status()
                run_id = r.json()["run_id"]

                async with client.stream("GET", f"/api/runs/{run_id}/events") as s:
                    async for line in s.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        ev = json.loads(line[5:].strip())
                        if ev.get("type") == "run_end":
                            break

                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception as exc:
                errors += 1
                print(f"  run error: {exc}")

        # Ramp: test at each concurrency level to find the degradation point
        levels = [1, concurrency // 2, concurrency] if concurrency > 2 else [1, concurrency]
        levels = sorted(set(levels))

        for level in levels:
            level_lats: list[float] = []
            for _ in range(iterations):
                tasks = [one_run() for _ in range(level)]
                before = len(latencies)
                await asyncio.gather(*tasks)
                level_lats.extend(latencies[before:])
            print_stats(
                f"Layer C — E2E  concurrency={level}  ({len(level_lats)} samples)",
                level_lats,
            )

        if errors:
            print(f"\n  ⚠  {errors} errors during Layer C")


# ── entrypoint ────────────────────────────────────────────────────────────────
async def main(layers: set[str], e2e: bool, concurrency: int = 10, iterations: int = 5) -> None:
    if "A" in layers:
        print("\nRunning Layer A — Redis Stream roundtrip (n=1000) …")
        await bench_stream(n=1000)

    if "B" in layers:
        print("\nRunning Layer B — Cache hit concurrency (1000×100 = 100 000 reads) …")
        await bench_cache(concurrency=1000, per_worker=100)

    if e2e or "C" in layers:
        print(f"\nRunning Layer C — Full E2E pipeline ({concurrency} concurrent, {iterations} iterations) …")
        await bench_e2e(concurrency=concurrency, iterations=iterations, mock_chunks=5)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=["A", "B", "C"], action="append",
                        dest="layers", help="run specific layer(s)")
    parser.add_argument("--e2e", action="store_true",
                        help="include Layer C (requires running server + worker)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Layer C: number of parallel runs per iteration (default: 10)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Layer C: iterations per concurrency level (default: 5)")
    args = parser.parse_args()

    chosen = set(args.layers) if args.layers else {"A", "B"}
    asyncio.run(main(chosen, args.e2e, args.concurrency, args.iterations))
