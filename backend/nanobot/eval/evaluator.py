"""Rule-based evaluator for agent run snapshots."""

from __future__ import annotations

import asyncio
import functools
import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from nanobot.eval.snapshot import RunSnapshotData
    from nanobot.storage.models import AgentTestCase

# English → Chinese equivalent terms for bilingual keyword matching.
# When a keyword misses in the response, each alias is also tried.
# NOTE: kept for now; will be removed once semantic matching is proven stable.
BILINGUAL_MAP: dict[str, list[str]] = {
    "multiresolution hash encoding": ["多分辨率哈希编码", "哈希编码"],
    "hash table": ["哈希表"],
    "feature vector": ["特征向量"],
    "trilinear interpolation": ["三线性插值"],
    "grid": ["网格"],
    "cone tracing": ["锥体追踪"],
    "integrated positional encoding": ["IPE", "积分位置编码"],
    "anti-aliasing": ["抗锯齿"],
    "frustum": ["截锥体"],
    "hash encoding": ["哈希编码"],
    "contraction": ["空间收缩"],
    "point cloud": ["点云"],
    "radiance field": ["辐射场"],
    "volume rendering": ["体渲染"],
    "rasterization": ["光栅化"],
    "spherical harmonics": ["球谐函数", "SH"],
    "unbounded": ["无界场景", "无界"],
    "surface reconstruction": ["表面重建"],
}

_SENTENCE_SPLIT_RE = re.compile(r'[。！？!?\n]+')
_SEMANTIC_KW_THRESHOLD = 0.65  # cross-lingual embeddings (e.g. EN keyword vs ZH response) produce lower similarity than monolingual
_SEMANTIC_TOOL_QUERY_THRESHOLD = 0.6


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out very short fragments."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if len(s.strip()) >= 5]


_EMBED_BATCH_SIZE = 10  # conservative limit for DashScope / most APIs


def _batch_embed(
    embedding_fn: Callable[[list[str]], list[list[float]]],
    texts: list[str],
) -> list[list[float]]:
    """Call embedding_fn in chunks to respect API batch size limits."""
    results: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        chunk = texts[i:i + _EMBED_BATCH_SIZE]
        results.extend(embedding_fn(chunk))
    return results


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _kw_hit_str(kw: str, text_lower: str) -> bool:
    """String-only keyword hit check with bilingual fallback."""
    if kw.lower() in text_lower:
        return True
    for alias in BILINGUAL_MAP.get(kw.lower(), []):
        if alias.lower() in text_lower:
            return True
    return False


class RuleEvaluator:
    """Scores a run against a test case using deterministic rules.

    Only dimensions with expected values set are scored — missing expectations
    are skipped entirely, never recorded as 0.

    If embedding_fn is provided, keyword_coverage uses a two-stage approach:
    string match first, then semantic similarity for unmatched keywords.
    web_search tool calls also get parameter-level validation via embedding.
    """

    def __init__(
        self,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn

    async def evaluate(
        self,
        snapshot: "RunSnapshotData",
        test_case: "AgentTestCase",
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # Token budget — informational only, not a pass/fail gate.
        # ------------------------------------------------------------------ #
        if test_case.token_budget is not None:
            total = snapshot.total_input_tokens + snapshot.total_output_tokens
            scores["token_budget"] = 1.0 if total <= test_case.token_budget else 0.0

        # ------------------------------------------------------------------ #
        # Tool call evaluation (presence + optional param validation).
        # ------------------------------------------------------------------ #
        web_search_queries: list[str] = []   # accumulate for embedding batch
        _param_check_needed = False

        if test_case.expected_tools:
            actual_calls = snapshot.tool_call_chain or []
            actual_names = {c["name"] for c in actual_calls}
            expected = set(test_case.expected_tools)
            hits = actual_names & expected

            if not hits:
                scores["tool_skip"] = 0.0
            elif hits == expected:
                scores["tool_skip"] = 1.0
            else:
                scores["tool_skip"] = 0.7

            # Collect web_search queries for param validation (Direction 6).
            if hits and self._embedding_fn:
                for call in actual_calls:
                    if call["name"] == "web_search" and "web_search" in expected:
                        q = (call.get("params") or {}).get("query", "")
                        if q:
                            web_search_queries.append(q)
                            _param_check_needed = True

        # ------------------------------------------------------------------ #
        # retrieve_by_entity param validation — pure string, no embedding.
        # ------------------------------------------------------------------ #
        _retrieve_param_ok: bool | None = None
        if (
            test_case.expected_tools
            and "retrieve_by_entity" in set(test_case.expected_tools)
            and test_case.expected_keywords
        ):
            kw_lower_set = {kw.lower() for kw in test_case.expected_keywords}
            for call in (snapshot.tool_call_chain or []):
                if call["name"] == "retrieve_by_entity":
                    params = call.get("params") or {}
                    entity = (params.get("entity") or params.get("query") or "").lower()
                    if any(kw in entity or entity in kw for kw in kw_lower_set):
                        _retrieve_param_ok = True
                        break
            if _retrieve_param_ok is None and any(
                c["name"] == "retrieve_by_entity" for c in (snapshot.tool_call_chain or [])
            ):
                _retrieve_param_ok = False  # called but entity didn't match

        # ------------------------------------------------------------------ #
        # Keyword coverage — string match first, semantic fallback later.
        # ------------------------------------------------------------------ #
        str_hits: dict[str, bool] = {}
        unmatched_keywords: list[str] = []

        if test_case.expected_keywords:
            resp_lower = (snapshot.final_response or "").lower()
            for kw in test_case.expected_keywords:
                hit = _kw_hit_str(kw, resp_lower)
                str_hits[kw] = hit
                if not hit:
                    unmatched_keywords.append(kw)

            # Contextual recall via string match.
            if snapshot.tool_call_chain:
                retrieved_text = " ".join(
                    str(entry.get("result") or "") for entry in snapshot.tool_call_chain
                ).lower()
                covered_in_chunks = sum(
                    1 for kw in test_case.expected_keywords
                    if _kw_hit_str(kw, retrieved_text)
                )
                scores["contextual_recall"] = round(
                    covered_in_chunks / max(len(test_case.expected_keywords), 1), 4
                )

        # ------------------------------------------------------------------ #
        # Single batch embedding call for semantic fallback + tool param check.
        # ------------------------------------------------------------------ #
        semantic_kw_hits: dict[str, bool] = {}
        web_search_param_ok: bool | None = None

        needs_embedding = (
            self._embedding_fn is not None
            and (unmatched_keywords or _param_check_needed)
        )

        if needs_embedding:
            sentences = _split_sentences(snapshot.final_response or "")

            # Build the combined text list for one batch call:
            #   [unmatched_kw_0, ..., sent_0, ..., ws_query_0, ..., user_input]
            all_texts: list[str] = list(unmatched_keywords) + sentences
            ws_offset = len(all_texts)
            if web_search_queries:
                all_texts += web_search_queries
                all_texts.append(snapshot.user_input)

            try:
                vectors: list[list[float]] = await asyncio.get_event_loop().run_in_executor(
                    None, functools.partial(_batch_embed, self._embedding_fn, all_texts)
                )

                n_kw = len(unmatched_keywords)
                n_sent = len(sentences)
                kw_vecs = vectors[:n_kw]
                sent_vecs = vectors[n_kw:n_kw + n_sent]

                # Semantic keyword hits.
                for i, kw in enumerate(unmatched_keywords):
                    if sent_vecs:
                        best = max(_cosine_sim(kw_vecs[i], sv) for sv in sent_vecs)
                        semantic_kw_hits[kw] = best >= _SEMANTIC_KW_THRESHOLD
                    else:
                        semantic_kw_hits[kw] = False

                # web_search param validation.
                if web_search_queries:
                    ws_query_vecs = vectors[ws_offset:ws_offset + len(web_search_queries)]
                    ui_vec = vectors[-1]
                    sims = [_cosine_sim(qv, ui_vec) for qv in ws_query_vecs]
                    web_search_param_ok = all(s >= _SEMANTIC_TOOL_QUERY_THRESHOLD for s in sims)

            except Exception:
                # Embedding failures degrade gracefully to string-only mode.
                pass

        # ------------------------------------------------------------------ #
        # Finalize keyword_coverage.
        # ------------------------------------------------------------------ #
        if test_case.expected_keywords:
            covered = sum(
                1 for kw in test_case.expected_keywords
                if str_hits.get(kw) or semantic_kw_hits.get(kw, False)
            )
            scores["keyword_coverage"] = round(
                covered / max(len(test_case.expected_keywords), 1), 4
            )

        # ------------------------------------------------------------------ #
        # Finalize tool_skip with param degradation.
        # ------------------------------------------------------------------ #
        if "tool_skip" in scores and scores["tool_skip"] > 0.0:
            # Determine if any param check failed.
            param_failed = (
                (web_search_param_ok is not None and not web_search_param_ok)
                or (_retrieve_param_ok is not None and not _retrieve_param_ok)
            )
            if param_failed:
                scores["tool_skip"] = 0.5

        return scores

    def is_passed(self, scores: dict[str, float], test_case: "AgentTestCase") -> tuple[bool, list[str]]:
        """Pass/fail gate: keyword_coverage >= threshold only.

        Keyword threshold is dynamic based on keyword count:
        - ≤4 keywords: threshold 0.8
        - ≤7 keywords: threshold 0.6
        - >7 keywords: threshold 0.5

        tool_skip is a behaviour signal recorded in failed_dims but does NOT
        affect the pass/fail decision.
        tool_param_wrong (tool_skip == 0.5) is similarly recorded but doesn't gate.
        token_budget and contextual_recall are excluded entirely (informational only).
        """
        kw_score = scores.get("keyword_coverage", 1.0)
        tool_score = scores.get("tool_skip", 1.0)

        n_kw = len(test_case.expected_keywords or [])
        if n_kw <= 4:
            kw_threshold = 0.8
        elif n_kw <= 7:
            kw_threshold = 0.6
        else:
            kw_threshold = 0.5

        passed = kw_score >= kw_threshold

        failed: list[str] = []
        if kw_score < kw_threshold:
            failed.append("keyword_coverage")
        if tool_score < 0.5 and test_case.expected_tools:
            failed.append("tool_skip")
        elif tool_score == 0.5 and test_case.expected_tools:
            failed.append("tool_param_wrong")

        return passed, failed
