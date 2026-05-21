"""Knowledge Processor — processes research results into knowledge base.

This module handles the knowledge loop's two phases:
1. Immediate: Extract claims and candidate insights after each research
2. Batch: Refine candidate insights into confirmed insights

Three-tier architecture:
- RAG Chunks: Original evidence (source_type: "research" | "document")
- Claims: Atomic facts with evidence_ids pointing to RAG chunks
- Insights: Cross-domain patterns with supporting_claim_ids pointing to Claims
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.research.insight_tracker import InsightTracker
from nanobot.research.knowledge_search import KnowledgeSearch
from nanobot.research.types import (
    BatchRefineResult,
    Claim,
    Insight,
    KnowledgeProcessResult,
    ResearchResult,
    SynthesisResult,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.research.correction_tracker import CorrectionTracker
    from nanobot.rag.libs.vector_store.chroma_store import ChromaStore


# ============== Prompt Templates ==============

_EXTRACT_CLAIMS_PROMPT = """从以下研究结论中提取原子级的事实陈述（Claims）。

## 研究主题
{topic}

## 研究报告摘要
{report_summary}

## 研究发现
{findings}

## 任务
请提取 5-15 条原子级的事实陈述，每条应满足：
1. 独立的事实陈述，不可再拆分
2. 有明确的来源 URL（如果提供了）
3. 可以是以下类型之一：
   - factual: 客观事实（如"3DGS 的训练时间比 NeRF 短"）
   - interpretation: 解释性陈述（如"3DGS 适合实时渲染是因为其高斯 Splatting 方式"）
   - method: 方法论陈述（如"3DGS 使用 3D 高斯函数表示场景"）
   - insight: 洞察性陈述（如"基于显式表示的方法在效率上优于隐式表示"）

## 输出格式
请输出 JSON 格式：
```json
{{
  "claims": [
    {{
      "claim": "事实陈述内容",
      "type": "factual|interpretation|method|insight",
      "confidence": 0.8,
      "is_evergreen": true,
      "source_urls": ["url1", "url2"]
    }}
  ]
}}
```

confidence: 0.0-1.0，表示这个事实的可信度
is_evergreen: true 表示这个事实在很长时间内不会改变（如基本原理），false 表示可能随时间变化（如性能数据）"""


_EXTRACT_INSIGHTS_PROMPT = """从以下 Claims 中提炼可迁移的洞察（Insights）。

## Claims（来自研究 "{topic}"）
{claims_text}

## 任务
请从 Claims 中提炼 2-5 条可迁移的洞察：
1. 洞察应该是跨 topic 可复用的深层规律
2. 需要从多个 Claims 综合得出
3. 标注适用领域（applicable_domains）
4. 判断是否 "常青"（is_evergreen）

## 示例
Claims:
- [0] "3DGS 在薄表面渲染上存在 artifact"
- [1] "NeRF 对透明物体有伪影"
- [2] "Mip-NeRF 对边缘处理较差"

→ Insight: "基于点/高斯表示的渲染方法对薄结构和透明物体普遍处理较差"
   supporting_claim_ids: ["0", "1", "2"]
   适用领域: ["3D重建", "渲染", "视图合成"]

## 输出格式
```json
{{
  "insights": [
    {{
      "insight": "洞察内容",
      "supporting_claim_ids": ["0", "1", "2"],  // Claims 的索引号
      "applicable_domains": ["领域1", "领域2"],
      "confidence": 0.75,
      "is_evergreen": true
    }}
  ]
}}
```"""


_BATCH_REFINE_PROMPT = """从以下候选 Insights 中提炼跨域规律。

## 候选 Insights (来自多次研究)
{candidates}

## 任务
1. 识别重复或相似的候选，合并为一条
2. 识别跨域模式，提炼为更通用的规律
3. 标注适用领域和边界条件

## 示例
候选1: "3DGS 对薄结构处理较差" (来自 3DGS 研究)
候选2: "NeRF 对透明物体有伪影" (来自 NeRF 研究)

→ 提炼: "基于点/高斯表示的渲染方法对薄结构和透明物体普遍处理较差，原因是缺乏体积约束"
   适用领域: ["3D重建", "渲染"]

## 输出格式
```json
{{
  "insights": [
    {{
      "insight": "提炼后的规律",
      "supporting_candidate_ids": ["id1", "id2"],
      "applicable_domains": ["domain1", "domain2"],
      "confidence": 0.9,
      "is_evergreen": true
    }}
  ]
}}
```
"""


class KnowledgeProcessor:
    """Process research results into knowledge base.

    This class orchestrates the knowledge loop's processing pipeline:
    1. Extract claims from research results
    2. Extract candidate insights from claims
    3. Deduplicate and assess confidence
    4. Write to knowledge base
    5. Track candidate insights for batch refinement
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        knowledge_search: KnowledgeSearch,
        tracker: InsightTracker,
        correction_tracker: "CorrectionTracker | None" = None,
        similarity_threshold: float = 0.85,
        rag_store: "ChromaStore | None" = None,
    ):
        self.provider = provider
        self.model = model
        self.knowledge_search = knowledge_search
        self.tracker = tracker
        self.correction_tracker = correction_tracker
        self.similarity_threshold = similarity_threshold
        self.rag_store = rag_store

    async def process(self, result: ResearchResult) -> KnowledgeProcessResult:
        """Process a research result into claims and insights.

        This is the immediate phase: runs after each research completion.

        Flow:
        1. Write report to RAG → get chunk_ids
        2. Extract Claims → match with chunk_ids (evidence_ids)
        3. Deduplicate Claims → write to claim store → get claim_ids
        4. Extract Insights → using claim_ids
        5. Deduplicate Insights → write to insight store

        Args:
            result: The research result to process.

        Returns:
            KnowledgeProcessResult with counts of written/duplicated items.
        """
        logger.info(f"KnowledgeProcessor: processing result for topic '{result.topic}'")

        # Step 1: Write report to RAG → get chunk_ids
        chunk_ids = await self._write_report_to_rag(result.report, result.topic)
        logger.info(f"KnowledgeProcessor: wrote {len(chunk_ids)} chunks to RAG")

        # Step 2: Extract Claims
        claims = await self._extract_claims(result)
        if not claims:
            logger.info("KnowledgeProcessor: no claims extracted")
            return KnowledgeProcessResult(
                claims_written=0,
                insights_written=0,
                duplicates_skipped=0,
                conflicts_detected=0,
            )

        # Step 3: Match Claims to chunks using vector similarity
        if chunk_ids:
            claim_chunk_map = await self._match_chunks_to_claims(claims, chunk_ids)
            for claim in claims:
                claim.evidence_ids = claim_chunk_map.get(claim.claim, [])
            logger.info(f"KnowledgeProcessor: matched claims to chunks")

        # Step 4: Confidence assessment
        claims = self._assess_confidence(claims, result.synthesis)

        # Step 5: Write Claims with automatic deduplication (now handled inside write_claims)
        # source="research", verified=True by default for research-sourced claims
        write_result = await self.knowledge_search.write_claims(claims, source="research", verified=True)
        claim_ids = write_result.claim_ids

        logger.info(
            f"KnowledgeProcessor: {write_result.claims_written} new claims, "
            f"{write_result.claims_updated} updated, {write_result.duplicates_skipped} skipped"
        )

        # Step 6: Extract Insights using claim_ids
        insights = await self._extract_insights(claims, claim_ids, result.topic)

        # Step 7: Assess insight confidence
        insights = self._assess_insight_confidence(insights, claims, claim_ids)

        # Step 8: Deduplicate insights
        new_insights = await self._check_insight_duplicates(insights)
        logger.info(
            f"KnowledgeProcessor: {len(new_insights)} new insights, "
            f"{len(insights) - len(new_insights)} duplicates"
        )

        # Step 9: Write insights to store
        if new_insights:
            await self.knowledge_search.write_insights(new_insights, maturity="candidate")

            # Record to tracker
            for insight in new_insights:
                insight_id = f"cand_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(insight.insight) % 100000:05d}"
                self.tracker.add_candidate(insight_id, {
                    "insight": insight.insight,
                    "supporting_claim_ids": insight.supporting_claim_ids,
                    "applicable_domains": insight.applicable_domains,
                    "confidence": insight.confidence,
                    "is_evergreen": insight.is_evergreen,
                    "maturity": "candidate",
                })

        logger.info(
            f"KnowledgeProcessor: wrote {len(claim_ids)} claims, "
            f"{len(new_insights)} candidate insights"
        )

        return KnowledgeProcessResult(
            claims_written=write_result.claims_written,
            insights_written=len(new_insights),
            duplicates_skipped=write_result.duplicates_skipped,
            conflicts_detected=0,  # Conflicts now handled by confidence comparison in write_claims
        )

    async def batch_refine_insights(self) -> BatchRefineResult:
        """Refine candidate insights into confirmed insights.

        This is the batch phase: runs periodically or on-demand.
        Combines multiple candidate insights into higher-level confirmed insights.
        """
        logger.info("KnowledgeProcessor: starting batch refinement")

        # Step 1: Get all candidates
        candidates = self.tracker.list_candidates()
        if not candidates:
            logger.info("KnowledgeProcessor: no candidates to refine")
            return BatchRefineResult(processed=0, confirmed=0)

        logger.info(f"KnowledgeProcessor: refining {len(candidates)} candidates")

        # Step 2: LLM refine
        candidates_text = "\n".join(
            f"[{c['id']}] {c['insight']}" for c in candidates
        )

        messages = [
            {"role": "system", "content": "You are an expert research analyst."},
            {"role": "user", "content": _BATCH_REFINE_PROMPT.format(candidates=candidates_text)},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                model=self.model,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"KnowledgeProcessor: LLM refine failed: {e}")
            return BatchRefineResult(processed=0, confirmed=0)

        # Step 3: Parse response
        confirmed_insights = self._parse_refine_response(response.content if response else "{}")

        if not confirmed_insights:
            logger.warning("KnowledgeProcessor: no confirmed insights from refine")
            return BatchRefineResult(processed=len(candidates), confirmed=0)

        # Step 4: Write confirmed insights (write first, ensure success)
        confirmed_insight_objects = [
            Insight(
                insight=i["insight"],
                supporting_claim_ids=i.get("supporting_candidate_ids", []),
                applicable_domains=i.get("applicable_domains", []),
                confidence=i.get("confidence", 0.7),
                is_evergreen=i.get("is_evergreen", True),
                maturity="confirmed",
            )
            for i in confirmed_insights
        ]

        confirmed_ids = await self.knowledge_search.write_insights(
            confirmed_insight_objects, maturity="confirmed"
        )

        # Step 5: Delete old candidates (from both store and tracker)
        old_ids = [c["id"] for c in candidates]
        await self.knowledge_search.delete_insights(old_ids)
        for old_id in old_ids:
            self.tracker.mark_processed(old_id)

        logger.info(
            f"KnowledgeProcessor: batch refine complete: "
            f"{len(candidates)} candidates → {len(confirmed_ids)} confirmed"
        )

        return BatchRefineResult(processed=len(candidates), confirmed=len(confirmed_ids))

    # ============== Private Methods ==============

    async def _extract_claims(self, result: ResearchResult) -> list[Claim]:
        """Extract claims from research result using LLM."""
        # Prepare context
        findings_text = ""
        if result.synthesis and result.synthesis.findings:
            findings_text = "\n".join(
                f"- {f.statement}" for f in result.synthesis.findings
            )

        report_summary = ""
        if result.report:
            # Take first 2000 chars as summary
            report_summary = result.report[:2000]

        # Collect all source URLs
        all_urls = set()
        if result.synthesis and result.synthesis.sources:
            for source in result.synthesis.sources:
                if source.url:
                    all_urls.add(source.url)
        if result.synthesis and result.synthesis.findings:
            for f in result.synthesis.findings:
                all_urls.update(f.source_urls)

        messages = [
            {"role": "system", "content": "You are an expert research analyst."},
            {
                "role": "user",
                "content": _EXTRACT_CLAIMS_PROMPT.format(
                    topic=result.topic,
                    report_summary=report_summary,
                    findings=findings_text or "无",
                ),
            },
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                model=self.model,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"KnowledgeProcessor: claim extraction failed: {e}")
            return []

        return self._parse_claims(response.content if response else "{}", list(all_urls))

    async def _extract_insights(
        self, claims: list[Claim], claim_ids: list[str], topic: str
    ) -> list[Insight]:
        """Extract candidate insights from claims using LLM.

        Args:
            claims: List of claims to derive insights from.
            claim_ids: List of claim IDs (same order as claims).
            topic: Research topic.

        Returns:
            List of extracted insights with supporting_claim_ids.
        """
        if len(claims) < 2:
            return []

        # Format claims with indices for LLM reference
        claims_text = "\n".join(
            f"[{i}] [{c.type}] {c.claim} (confidence={c.confidence:.2f})"
            for i, c in enumerate(claims)
        )

        messages = [
            {"role": "system", "content": "You are an expert research analyst."},
            {
                "role": "user",
                "content": _EXTRACT_INSIGHTS_PROMPT.format(
                    topic=topic,
                    claims_text=claims_text,
                ),
            },
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                model=self.model,
                max_tokens=2048,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"KnowledgeProcessor: insight extraction failed: {e}")
            return []

        return self._parse_insights(response.content if response else "{}", claim_ids)

    async def _check_claim_duplicates(
        self, claims: list[Claim]
    ) -> tuple[list[Claim], int, int]:
        """Check for duplicate claims using vector similarity and bigram.

        When conflicts are detected (high similarity but different content):
        - If new claim has higher confidence, record as correction and add it
        - If old claim has higher confidence, skip the new one

        Returns:
            Tuple of (new_claims, duplicate_count, conflict_count).
        """
        new_claims = []
        duplicate_count = 0
        conflict_count = 0

        for claim in claims:
            # Search for similar claims
            results = await self.knowledge_search.search_claims(
                claim.claim, top_k=3, apply_decay=False
            )

            if not results:
                new_claims.append(claim)
                continue

            top_match = results[0]
            similarity = top_match.get("score", 0)

            if similarity >= self.similarity_threshold:
                # Similar claim found - check if actually same
                if self._is_same_claim(claim.claim, top_match.get("text", "")):
                    duplicate_count += 1
                else:
                    # High similarity but different content = potential conflict
                    conflict_count += 1

                    # Get old claim's confidence
                    old_confidence = top_match.get("metadata", {}).get("confidence", 0.5)

                    # If new claim has higher confidence, it's a correction
                    if claim.confidence > old_confidence:
                        # Record the correction
                        await self._record_correction(
                            old_value=top_match.get("text", ""),
                            new_value=claim.claim,
                            old_confidence=old_confidence,
                            new_confidence=claim.confidence,
                            source_url=claim.source_urls[0] if claim.source_urls else "",
                            reason="New research finding with higher confidence",
                            correction_type="claim",
                            old_id=top_match.get("id", ""),
                        )
                        # Add the new claim (it replaces the old one)
                        new_claims.append(claim)
                    # else: skip the new claim, old one is more reliable
            else:
                new_claims.append(claim)

        return new_claims, duplicate_count, conflict_count

    async def _record_correction(
        self,
        old_value: str,
        new_value: str,
        old_confidence: float,
        new_confidence: float,
        source_url: str,
        reason: str,
        correction_type: str,
        old_id: str = "",
    ) -> None:
        """Record a correction when new knowledge contradicts existing.

        Args:
            old_value: The existing claim/insight that was contradicted.
            new_value: The new claim/insight that contradicts it.
            old_confidence: Confidence of the old value.
            new_confidence: Confidence of the new value.
            source_url: Source URL for the correction.
            reason: Why this is a correction.
            correction_type: "claim" or "insight".
            old_id: ID of the old record in storage.
        """
        if not self.correction_tracker:
            return

        from nanobot.research.types import Correction

        correction = Correction(
            old_value=old_value,
            new_value=new_value,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            source_url=source_url,
            reason=reason,
            correction_type=correction_type,
            related_claim_id=old_id,
        )

        self.correction_tracker.add_correction(correction.to_dict())

        logger.info(
            f"KnowledgeProcessor: recorded {correction_type} correction: "
            f"'{old_value[:50]}...' → '{new_value[:50]}...'"
        )

    def _is_same_claim(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are semantically the same using character bigrams.

        This method works for both English and Chinese text.
        """
        def bigrams(s: str) -> set[tuple[str, str]]:
            s = s.strip()
            if len(s) <= 1:
                return set()
            return set(zip(s[:-1], s[1:]))

        b1, b2 = bigrams(claim1), bigrams(claim2)
        if not b1 or not b2:
            return False

        jaccard = len(b1 & b2) / len(b1 | b2)
        return jaccard >= 0.6

    def _assess_confidence(
        self, claims: list[Claim], synthesis: "SynthesisResult | None"
    ) -> list[Claim]:
        """Adjust claim confidence based on synthesis quality.

        Rules:
        - If synthesis.coverage_score is high (>0.8), boost confidence
        - If synthesis has many contradictions, reduce confidence
        - Claims with source_urls get a boost

        Args:
            claims: List of claims to assess.
            synthesis: The synthesis result from research.

        Returns:
            List of claims with adjusted confidence.
        """
        if not synthesis:
            return claims

        coverage_boost = 0.1 if synthesis.coverage_score > 0.8 else 0
        contradiction_penalty = 0.1 * len(synthesis.contradictions)

        adjusted_claims = []
        for claim in claims:
            # Base adjustment from synthesis quality
            adjusted_confidence = claim.confidence + coverage_boost - contradiction_penalty

            # Boost for well-sourced claims
            if claim.source_urls and len(claim.source_urls) >= 2:
                adjusted_confidence += 0.05

            # Clamp to [0, 1]
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

            adjusted_claims.append(Claim(
                claim=claim.claim,
                type=claim.type,
                is_evergreen=claim.is_evergreen,
                source_urls=claim.source_urls,
                confidence=adjusted_confidence,
                evidence_ids=claim.evidence_ids,  # Preserve evidence_ids
                created_at=claim.created_at,
            ))

        return adjusted_claims

    def _assess_insight_confidence(
        self, insights: list[Insight], claims: list[Claim], claim_ids: list[str]
    ) -> list[Insight]:
        """Adjust insight confidence based on supporting claims quality.

        Rules:
        - Insight confidence = weighted average of supporting claims
        - More supporting claims → higher confidence
        - Higher claim confidence → higher insight confidence

        Args:
            insights: List of insights to assess.
            claims: List of claims that support the insights.
            claim_ids: List of claim IDs (same order as claims).

        Returns:
            List of insights with adjusted confidence.
        """
        if not insights or not claims:
            return insights

        # Build claim_id to confidence mapping
        claim_id_to_conf = {str(i): c.confidence for i, c in enumerate(claims)}

        adjusted_insights = []
        for insight in insights:
            if not insight.supporting_claim_ids:
                # No supporting claims, keep original confidence
                adjusted_insights.append(insight)
                continue

            # Calculate weighted confidence from supporting claims
            supporting_scores = []
            for claim_id_str in insight.supporting_claim_ids:
                if claim_id_str in claim_id_to_conf:
                    supporting_scores.append(claim_id_to_conf[claim_id_str])

            if supporting_scores:
                # Weighted average: more support = higher confidence
                avg_support = sum(supporting_scores) / len(supporting_scores)
                support_boost = min(0.1, len(supporting_scores) * 0.02)  # Max 0.1 boost
                adjusted_confidence = max(0.0, min(1.0, avg_support + support_boost))
            else:
                adjusted_confidence = insight.confidence

            adjusted_insights.append(Insight(
                insight=insight.insight,
                supporting_claim_ids=insight.supporting_claim_ids,
                applicable_domains=insight.applicable_domains,
                confidence=adjusted_confidence,
                is_evergreen=insight.is_evergreen,
                maturity=insight.maturity,
                created_at=insight.created_at,
            ))

        return adjusted_insights

    async def _check_insight_duplicates(
        self, insights: list[Insight]
    ) -> list[Insight]:
        """Check for duplicate insights using vector similarity.

        Similar to _check_claim_duplicates but for insights.

        Args:
            insights: List of insights to check for duplicates.

        Returns:
            List of new insights (duplicates removed).
        """
        new_insights = []

        for insight in insights:
            # Search for similar insights
            results = await self.knowledge_search.search_insights(
                insight.insight, top_k=3, maturity=None
            )

            if not results:
                new_insights.append(insight)
                continue

            top_match = results[0]
            similarity = top_match.get("score", 0)

            if similarity >= self.similarity_threshold:
                # Check if semantically same
                if self._is_same_claim(insight.insight, top_match.get("text", "")):
                    logger.debug(
                        f"KnowledgeProcessor: skipping duplicate insight: {insight.insight[:50]}..."
                    )
                    continue

            new_insights.append(insight)

        return new_insights

    def _parse_claims(self, content: str, fallback_urls: list[str]) -> list[Claim]:
        """Parse claims from LLM response."""
        claims = []

        try:
            # Try to extract JSON from content
            json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(content)

            for item in data.get("claims", []):
                claims.append(Claim(
                    claim=item.get("claim", ""),
                    type=item.get("type", "factual"),
                    confidence=item.get("confidence", 0.5),
                    is_evergreen=item.get("is_evergreen", False),
                    source_urls=item.get("source_urls") or fallback_urls[:3],
                ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"KnowledgeProcessor: failed to parse claims: {e}")

        return claims

    def _parse_insights(self, content: str, claim_ids: list[str]) -> list[Insight]:
        """Parse insights from LLM response.

        Args:
            content: LLM response content.
            claim_ids: List of claim IDs to map indices to actual IDs.

        Returns:
            List of parsed insights with supporting_claim_ids.
        """
        insights = []

        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(content)

            for item in data.get("insights", []):
                # Convert index strings to actual claim IDs
                supporting_indices = item.get("supporting_claim_ids", [])
                actual_claim_ids = []
                for idx_str in supporting_indices:
                    try:
                        idx = int(idx_str)
                        if 0 <= idx < len(claim_ids):
                            actual_claim_ids.append(claim_ids[idx])
                    except (ValueError, TypeError):
                        pass

                insights.append(Insight(
                    insight=item.get("insight", ""),
                    supporting_claim_ids=actual_claim_ids,
                    applicable_domains=item.get("applicable_domains", []),
                    confidence=item.get("confidence", 0.5),
                    is_evergreen=item.get("is_evergreen", True),
                    maturity="candidate",
                ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"KnowledgeProcessor: failed to parse insights: {e}")

        return insights

    def _parse_refine_response(self, content: str) -> list[dict[str, Any]]:
        """Parse refined insights from LLM response."""
        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(content)

            return data.get("insights", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"KnowledgeProcessor: failed to parse refine response: {e}")
            return []

    # ============== RAG Methods (Three-tier Architecture) ==============

    async def _write_report_to_rag(
        self, report: str | None, topic: str
    ) -> list[str]:
        """Write report chunks to RAG store.

        This is the first step in the three-tier architecture:
        1. Split report into chunks
        2. Write chunks to RAG with metadata
        3. Return chunk IDs for later claim matching

        Args:
            report: The research report text.
            topic: Research topic for metadata.

        Returns:
            List of chunk IDs.
        """
        if not report or not self.rag_store:
            return []

        # Split report into chunks
        chunks = self._split_into_chunks(report)

        if not chunks:
            return []

        records = []
        for chunk in chunks:
            chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
            records.append({
                "id": chunk_id,
                "vector": self._embed(chunk),
                "metadata": {
                    "source_type": "research",
                    "topic": topic,
                    "text": chunk,
                    "created_at": datetime.now().isoformat(),
                },
            })

        if records:
            self.rag_store.upsert(records)
            logger.info(f"KnowledgeProcessor: wrote {len(records)} chunks to RAG")

        return [r["id"] for r in records]

    def _split_into_chunks(
        self, text: str, max_chars: int = 1000
    ) -> list[str]:
        """Split text into chunks by paragraphs.

        Args:
            text: Text to split.
            max_chars: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If single paragraph exceeds max_chars, split by sentences
            if len(para) > max_chars:
                if current.strip():
                    chunks.append(current.strip())
                chunks.extend(self._split_long_paragraph(para, max_chars))
                current = ""
            elif len(current) + len(para) <= max_chars:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = para + "\n\n"

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _split_long_paragraph(self, text: str, max_chars: int) -> list[str]:
        """Split a long paragraph by sentences or fixed length.

        Args:
            text: Long text to split.
            max_chars: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        chunks = []
        current = ""

        # Try to split by sentences (Chinese: 。？！, English: . ! ?)
        import re
        sentences = re.split(r'(?<=[。？！.!?])\s*', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) <= max_chars:
                current += sentence + " "
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If single sentence is too long, truncate it
                if len(sentence) > max_chars:
                    chunks.extend(self._split_by_length(sentence, max_chars))
                    current = ""
                else:
                    current = sentence + " "

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _split_by_length(self, text: str, max_chars: int) -> list[str]:
        """Split text by fixed length.

        Args:
            text: Text to split.
            max_chars: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        chunks = []
        for i in range(0, len(text), max_chars):
            chunk = text[i:i + max_chars].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    async def _match_chunks_to_claims(
        self, claims: list[Claim], chunk_ids: list[str]
    ) -> dict[str, list[str]]:
        """Match claims to RAG chunks using vector similarity.

        Args:
            claims: List of claims to match.
            chunk_ids: List of chunk IDs from RAG.

        Returns:
            Dict mapping claim text to list of matching chunk IDs.
        """
        if not claims or not chunk_ids or not self.rag_store:
            return {c.claim: [] for c in claims}

        # Get chunk text and metadata using get_by_ids
        try:
            chunk_results = self.rag_store.get_by_ids(chunk_ids)
            chunk_data = {}
            for result in chunk_results:
                if result:
                    # Use text from metadata or document
                    text = result.get("metadata", {}).get("text", "") or result.get("text", "")
                    if text:
                        chunk_data[result["id"]] = text
        except Exception as e:
            logger.warning(f"KnowledgeProcessor: failed to get chunk data: {e}")
            chunk_data = {}

        if not chunk_data:
            return {c.claim: [] for c in claims}

        # Re-embed chunk texts and match
        claim_chunk_map = {}
        for claim in claims:
            claim_vec = self._embed(claim.claim)
            best_chunk_id = None
            best_score = -1

            for cid, chunk_text in chunk_data.items():
                chunk_vec = self._embed(chunk_text)
                score = self._cosine_sim(claim_vec, chunk_vec)
                if score > best_score:
                    best_score = score
                    best_chunk_id = cid

            # Only include if similarity is above threshold
            if best_chunk_id and best_score > 0.5:
                claim_chunk_map[claim.claim] = [best_chunk_id]
            else:
                claim_chunk_map[claim.claim] = []

        return claim_chunk_map

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text using knowledge search's encoder."""
        return self.knowledge_search._embed(text)

    def _cosine_sim(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score.
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

