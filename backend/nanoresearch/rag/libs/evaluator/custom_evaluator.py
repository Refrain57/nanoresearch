"""Custom evaluator: Recall@K, Precision@K, MRR, F1@K for retrieval quality checks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from nanoresearch.rag.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Deterministic retrieval evaluator — no LLM needed."""

    SUPPORTED_METRICS = {"recall@k", "f1@k", "precision@k", "mrr"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(m).strip().lower() for m in (metrics or [])]
        if not normalized:
            normalized = ["recall@k", "precision@k", "mrr"]

        unsupported = [m for m in normalized if m not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        retrieved_ids = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
        ground_truth_ids = self._extract_ground_truth_ids(ground_truth)

        results: Dict[str, float] = {}
        if "recall@k" in self.metrics:
            results["recall@k"] = self._compute_recall(retrieved_ids, ground_truth_ids)
        if "precision@k" in self.metrics:
            results["precision@k"] = self._compute_precision(retrieved_ids, ground_truth_ids)
        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr(retrieved_ids, ground_truth_ids)
        if "f1@k" in self.metrics:
            results["f1@k"] = self._compute_f1(retrieved_ids, ground_truth_ids)
        return results

    # ------------------------------------------------------------------

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        if settings is None:
            return []
        metrics = getattr(getattr(settings, "evaluation", None), "metrics", None)
        return [str(m) for m in metrics] if metrics else []

    def _extract_ground_truth_ids(self, ground_truth: Optional[Any]) -> List[str]:
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, dict):
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                return self._extract_ids(ground_truth["ids"], label="ground_truth.ids")
            return self._extract_ids([ground_truth], label="ground_truth")
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth, label="ground_truth")
        raise ValueError(f"Unsupported ground_truth type: {type(ground_truth).__name__}")

    def _extract_ids(self, items: Iterable[Any], label: str) -> List[str]:
        ids: List[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue
            raise ValueError(
                f"Unable to extract id from {label}[{index}] of type {type(item).__name__}"
            )
        return ids

    def _compute_recall(self, retrieved: Sequence[str], gold: Sequence[str]) -> float:
        if not gold:
            return 0.0
        gold_set = set(gold)
        hits = sum(1 for r in retrieved if r in gold_set)
        return hits / len(gold_set)

    def _compute_precision(self, retrieved: Sequence[str], gold: Sequence[str]) -> float:
        if not retrieved:
            return 0.0
        gold_set = set(gold)
        hits = sum(1 for r in retrieved if r in gold_set)
        return hits / len(retrieved)

    def _compute_mrr(self, retrieved: Sequence[str], gold: Sequence[str]) -> float:
        if not gold:
            return 0.0
        gold_set = set(gold)
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in gold_set:
                return 1.0 / rank
        return 0.0

    def _compute_f1(self, retrieved: Sequence[str], gold: Sequence[str]) -> float:
        if not gold or not retrieved:
            return 0.0
        gold_set = set(gold)
        ret_set = set(retrieved)
        tp = len(ret_set & gold_set)
        if tp == 0:
            return 0.0
        precision = tp / len(ret_set)
        recall = tp / len(gold_set)
        return 2 * precision * recall / (precision + recall)
