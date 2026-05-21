"""Knowledge Lint — quality checks for claims and insights.

This module implements a lint mechanism for knowledge quality assurance.
The core question: "Does this claim hold up without its source context?"

Three-tier verdict system (avoiding accidental deletion):
- KEEP: Self-contained, no action needed
- DEMOTE: May depend on context, confidence halved, marked needs_review=True
- DELETE: Clearly nonsensical or contradictory (strict criteria)

Two categories of checks:
- Structural (fast, no LLM): metadata-based checks like orphan_claims, broken_refs
- Semantic (LLM-based): self_contained check, contradiction detection (future)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.research.knowledge_search import KnowledgeSearch


class LintVerdict(Enum):
    """Three-tier verdict for lint results."""
    KEEP = "keep"      # Self-contained, no action needed
    DEMOTE = "demote"  # May depend on context, reduce confidence
    DELETE = "delete"  # Clearly nonsensical or contradictory


@dataclass
class LintIssue:
    """A single issue found by lint."""
    id: str                         # Record ID
    check: str                      # Which check found it
    verdict: LintVerdict            # keep / demote / delete
    reason: str                     # Why this verdict
    original_confidence: float      # Original confidence score
    suggested_fix: str | None = None


@dataclass
class SemanticLintResult:
    """Result of semantic lint with actual LLM call count."""
    issues: list[LintIssue]
    total_checked: int  # Actual claims processed = LLM calls made


@dataclass
class LintReport:
    """Full lint report combining structural and semantic results."""
    structural_issues: list[LintIssue] = field(default_factory=list)
    semantic_issues: list[LintIssue] = field(default_factory=list)
    kept: int = 0
    demoted: int = 0
    deleted: int = 0
    llm_calls: int = 0

    @property
    def total_issues(self) -> int:
        return len(self.structural_issues) + len(self.semantic_issues)


# Prompt for self-contained check
_SELF_CONTAINED_PROMPT = """判断这条 claim 是否可以独立成立，不需要来源上下文。

Claim: "{claim}"

判断标准：
1. 是否包含模糊引用（如"该方法"、"上述实验"、"它"）
2. 是否依赖未说明的前提条件
3. 是否需要特定领域背景才能理解

返回 JSON（只返回 JSON，不要其他内容）：
{{
  "verdict": "keep" | "demote" | "delete",
  "reason": "判断理由",
  "missing_context": "如果 demote/delete，缺少什么上下文"
}}

判断标准：
- keep: 完全独立，任何人都能理解和验证
- demote: 有模糊引用或隐含前提，但核心内容有价值
- delete: 严重依赖上下文，脱离后无意义或无法理解
"""


class KnowledgeLint:
    """Lint mechanism for knowledge quality assurance.

    Two categories of checks:
    - Structural (fast, no LLM): orphan_claims, broken_refs, missing_evidence, etc.
    - Semantic (LLM-based): self_contained check

    Usage:
        lint = KnowledgeLint(knowledge_search, provider=llm_provider, model="gpt-4o")
        report = await lint.lint_all(fix=True)
    """

    CONCURRENCY_LIMIT = 10  # Max concurrent LLM calls

    def __init__(
        self,
        knowledge_search: "KnowledgeSearch",
        provider: "LLMProvider | None" = None,
        model: str = "gpt-4o",
    ):
        self.ks = knowledge_search
        self.provider = provider
        self.model = model

    def _require_provider(self) -> None:
        """Raise clear error if provider is not set for semantic lint."""
        if self.provider is None:
            raise RuntimeError(
                "KnowledgeLint: provider is None. "
                "Semantic lint requires a LLM provider. "
                "Pass provider=your_provider in constructor."
            )

    # =========================================================================
    # Structural Checks (No LLM)
    # =========================================================================

    async def lint_structural(self, fix: bool = False) -> list[LintIssue]:
        """Run all structural checks (no LLM required).

        Args:
            fix: If True, apply automatic fixes where possible.

        Returns:
            List of issues found.
        """
        issues: list[LintIssue] = []

        # Check 1: orphan_claims — claims not used in any insight
        orphan_issues = await self._check_orphan_claims(fix=fix)
        issues.extend(orphan_issues)

        # Check 2: broken_refs — insights referencing non-existent claims
        broken_issues = await self._check_broken_refs(fix=fix)
        issues.extend(broken_issues)

        # Check 3: missing_evidence — research claims without evidence_ids
        evidence_issues = await self._check_missing_evidence()
        issues.extend(evidence_issues)

        # Check 4: empty_domains — insights without applicable_domains
        domain_issues = await self._check_empty_domains()
        issues.extend(domain_issues)

        # Check 5: invalid_confidence — confidence outside [0, 1]
        confidence_issues = await self._check_invalid_confidence(fix=fix)
        issues.extend(confidence_issues)

        return issues

    async def _check_orphan_claims(self, fix: bool = False) -> list[LintIssue]:
        """Check for claims not used in any insight.

        Note: Newly created claims may legitimately be orphans until
        they're referenced by an insight. This check reports only.
        """
        # ChromaDB doesn't match empty strings well, use get_all_documents
        claims = self.ks.claim_store.get_all_documents()
        if not claims:
            return []

        # Filter for orphan claims (empty used_in_insight_ids)
        orphan_claims = [
            c for c in claims
            if not c.get("metadata", {}).get("used_in_insight_ids", "")
        ]

        # Report only — orphans may be valid newly created claims
        return [
            LintIssue(
                id=c.get("id", "unknown"),
                check="orphan_claims",
                verdict=LintVerdict.KEEP,  # Don't auto-delete
                reason="Claim not used in any insight (may be newly created)",
                original_confidence=c.get("metadata", {}).get("confidence", 0.5),
            )
            for c in orphan_claims
        ]

    async def _check_broken_refs(self, fix: bool = False) -> list[LintIssue]:
        """Check for insights referencing non-existent claims."""
        insights = self.ks.insight_store.get_all_documents()
        if not insights:
            return []

        # Get all claim IDs
        all_claims = self.ks.claim_store.get_all_documents()
        valid_claim_ids = {c.get("id") for c in all_claims}

        issues = []
        to_update = []

        for insight in insights:
            meta = insight.get("metadata", {})
            raw_claim_ids = meta.get("supporting_claim_ids", "")
            if not raw_claim_ids:
                continue

            claim_ids = [cid.strip() for cid in raw_claim_ids.split(",") if cid.strip()]
            invalid_ids = [cid for cid in claim_ids if cid not in valid_claim_ids]

            if invalid_ids:
                # Fix: remove invalid refs
                valid_ids = [cid for cid in claim_ids if cid in valid_claim_ids]
                if fix and valid_ids:
                    to_update.append((
                        insight.get("id"),
                        {"supporting_claim_ids": ",".join(valid_ids)},
                    ))

                issues.append(LintIssue(
                    id=insight.get("id", "unknown"),
                    check="broken_refs",
                    verdict=LintVerdict.KEEP,  # Don't delete insight
                    reason=f"Insight references non-existent claims: {invalid_ids}",
                    original_confidence=meta.get("confidence", 0.5),
                ))

        if fix and to_update:
            self.ks.insight_store.update_batch(to_update)
            logger.info(f"Lint: fixed {len(to_update)} insights with broken refs")

        return issues

    async def _check_missing_evidence(self) -> list[LintIssue]:
        """Check for research claims without evidence_ids."""
        # ChromaDB doesn't support multi-condition AND in simple query
        # First get research claims, then filter in memory
        claims = self.ks.claim_store.query_by_metadata(
            filters={"source": "research"},
            limit=500,
        )
        if not claims:
            return []

        # Filter for missing evidence_ids
        issues = []
        for c in claims:
            meta = c.get("metadata", {})
            if not meta.get("evidence_ids", ""):
                issues.append(LintIssue(
                    id=c.get("id", "unknown"),
                    check="missing_evidence",
                    verdict=LintVerdict.KEEP,  # Report only
                    reason="Research claim has no evidence_ids",
                    original_confidence=meta.get("confidence", 0.5),
                ))

        return issues

    async def _check_empty_domains(self) -> list[LintIssue]:
        """Check for insights without applicable_domains."""
        # ChromaDB doesn't match empty strings well, use get_all_documents
        insights = self.ks.insight_store.get_all_documents()
        if not insights:
            return []

        # Filter for empty applicable_domains
        issues = []
        for i in insights:
            meta = i.get("metadata", {})
            if not meta.get("applicable_domains", ""):
                issues.append(LintIssue(
                    id=i.get("id", "unknown"),
                    check="empty_domains",
                    verdict=LintVerdict.KEEP,  # Report only
                    reason="Insight has no applicable_domains",
                    original_confidence=meta.get("confidence", 0.5),
                ))

        return issues

    async def _check_invalid_confidence(self, fix: bool = False) -> list[LintIssue]:
        """Check for confidence values outside [0, 1] range."""
        claims = self.ks.claim_store.get_all_documents()
        issues = []
        to_update = []

        for claim in claims:
            meta = claim.get("metadata", {})
            conf = meta.get("confidence", 0.5)
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                issues.append(LintIssue(
                    id=claim.get("id", "unknown"),
                    check="invalid_confidence",
                    verdict=LintVerdict.KEEP,
                    reason=f"Invalid confidence: {conf}",
                    original_confidence=0.5,
                ))
                if fix:
                    # Clamp to valid range
                    new_conf = max(0.0, min(1.0, float(conf) if isinstance(conf, (int, float)) else 0.5))
                    to_update.append((claim.get("id"), {"confidence": new_conf}))

        if fix and to_update:
            self.ks.claim_store.update_batch(to_update)
            logger.info(f"Lint: fixed {len(to_update)} claims with invalid confidence")

        return issues

    # =========================================================================
    # Semantic Checks (LLM-based)
    # =========================================================================

    async def lint_semantic(self, limit: int = 100) -> SemanticLintResult:
        """Run semantic checks with LLM.

        Args:
            limit: Maximum claims to process.

        Returns:
            SemanticLintResult with issues and actual LLM call count.
        """
        self._require_provider()

        # Get conversation claims (not research-verified)
        claims = self.ks.claim_store.query_by_metadata(
            filters={"source": "conversation"},
            limit=limit,
        )

        if not claims:
            return SemanticLintResult(issues=[], total_checked=0)

        issues = await self._check_self_contained_concurrent(claims)
        return SemanticLintResult(issues=issues, total_checked=len(claims))

    async def _check_self_contained_concurrent(self, claims: list[dict]) -> list[LintIssue]:
        """Process claims concurrently with semaphore to avoid batch bias."""
        semaphore = asyncio.Semaphore(self.CONCURRENCY_LIMIT)

        async def check_one(claim: dict) -> LintIssue | None:
            async with semaphore:
                # Get claim text from either top-level or metadata
                claim_text = claim.get("text", "") or claim.get("metadata", {}).get("text", "")
                if not claim_text:
                    logger.warning(f"Lint: claim {claim.get('id', 'unknown')} has no text, skipping")
                    return None

                try:
                    response = await self.provider.chat_with_retry(
                        messages=[
                            {"role": "system", "content": "You are a knowledge quality evaluator. Return only JSON."},
                            {"role": "user", "content": _SELF_CONTAINED_PROMPT.format(claim=claim_text)},
                        ],
                        model=self.model,
                        max_tokens=200,
                        temperature=0.0,
                    )

                    # Handle different response formats
                    content = response.content
                    if isinstance(content, list):
                        content = "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )

                    result = json.loads(content)
                    verdict = LintVerdict(result.get("verdict", "keep"))

                    if verdict != LintVerdict.KEEP:
                        return LintIssue(
                            id=claim.get("id", "unknown"),
                            check="self_contained",
                            verdict=verdict,
                            reason=result.get("reason", ""),
                            original_confidence=claim.get("metadata", {}).get("confidence", 0.5),
                        )
                    return None

                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning(f"Lint: failed to parse LLM response for claim {claim.get('id', 'unknown')}: {e}")
                    return None

        # Concurrent processing
        results = await asyncio.gather(*[check_one(c) for c in claims])
        return [r for r in results if r is not None]

    # =========================================================================
    # Combined Lint
    # =========================================================================

    async def lint_all(self, fix: bool = False) -> LintReport:
        """Run both structural and semantic checks.

        Args:
            fix: If True, apply fixes for semantic lint (demote/delete).
                 Structural lint always runs in report-only mode.

        Returns:
            LintReport with all findings.
        """
        # Structural checks (report only, don't auto-fix)
        structural = await self.lint_structural(fix=False)

        # Semantic checks (optional, requires provider)
        semantic_issues: list[LintIssue] = []
        llm_calls = 0
        demoted = 0
        deleted = 0
        kept = 0

        if self.provider:
            result = await self.lint_semantic(limit=100)
            semantic_issues = result.issues
            llm_calls = result.total_checked

            # Calculate verdict counts
            demoted = len([i for i in semantic_issues if i.verdict == LintVerdict.DEMOTE])
            deleted = len([i for i in semantic_issues if i.verdict == LintVerdict.DELETE])
            kept = llm_calls - demoted - deleted

            # Apply fixes if requested
            if fix and semantic_issues:
                demoted, deleted = await self._apply_fixes(semantic_issues)
                kept = llm_calls - demoted - deleted

        return LintReport(
            structural_issues=structural,
            semantic_issues=semantic_issues,
            kept=kept,
            demoted=demoted,
            deleted=deleted,
            llm_calls=llm_calls,
        )

    async def _apply_fixes(self, issues: list[LintIssue]) -> tuple[int, int]:
        """Apply demote/delete fixes.

        Args:
            issues: List of issues with verdicts.

        Returns:
            Tuple of (demoted_count, deleted_count).
        """
        to_demote = [i for i in issues if i.verdict == LintVerdict.DEMOTE]
        to_delete = [i for i in issues if i.verdict == LintVerdict.DELETE]

        # Demote: confidence halved, mark needs_review
        if to_demote:
            updates = [
                (
                    i.id,
                    {
                        "confidence": i.original_confidence * 0.5,
                        "needs_review": True,
                        "lint_reason": i.reason,
                    },
                )
                for i in to_demote
            ]
            await self.ks.claim_store.update_batch(updates)
            logger.info(f"Lint: demoted {len(to_demote)} claims")

        # Delete: remove from store
        if to_delete:
            await self.ks.claim_store.delete(ids=[i.id for i in to_delete])
            logger.info(f"Lint: deleted {len(to_delete)} claims")

        return len(to_demote), len(to_delete)
