"""Data structures for the Auto Research Agent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ResearchStatus(Enum):
    """Research pipeline status."""

    PLANNING = "planning"
    SEARCHING = "searching"
    SYNTHESIZING = "synthesizing"
    ITERATING = "iterating"
    COMPLETED = "completed"
    FAILED = "failed"


class DepthLevel(Enum):
    """Research depth level."""

    QUICK = "quick"
    NORMAL = "normal"
    DEEP = "deep"


@dataclass
class SubQuestion:
    """A sub-question extracted from the research topic."""

    id: int
    question: str
    keywords: list[str] = field(default_factory=list)
    priority: int = 1  # 1 = highest
    status: str = "pending"  # pending / searching / completed
    results: list["SearchResult"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "keywords": self.keywords,
            "priority": self.priority,
            "status": self.status,
        }


@dataclass
class ResearchPlan:
    """Research plan containing the topic and sub-questions."""

    topic: str
    sub_questions: list[SubQuestion] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "created_at": self.created_at.isoformat(),
            "iteration": self.iteration,
        }


@dataclass
class SearchResult:
    """A single search result with quality scoring."""

    url: str
    title: str
    content: str = ""
    source_type: str = "unknown"  # paper / news / blog / official / social
    credibility_score: float = 0.5  # 0-1: based on domain (.gov/.edu = high)
    relevance_score: float = 0.5  # 0-1: keyword match to sub-question
    recency_score: float = 0.5  # 0-1: based on publish date
    final_score: float = 0.0
    publish_date: str | None = None
    fetched_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.final_score == 0.0:
            self.final_score = self.credibility_score * 0.4 + self.relevance_score * 0.4 + self.recency_score * 0.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content[:200] if self.content else "",
            "source_type": self.source_type,
            "credibility_score": self.credibility_score,
            "relevance_score": self.relevance_score,
            "recency_score": self.recency_score,
            "final_score": self.final_score,
            "publish_date": self.publish_date,
        }


@dataclass
class KnowledgeGap:
    """An identified knowledge gap."""

    description: str
    related_sub_question_id: int | None = None
    suggested_searches: list[str] = field(default_factory=list)


@dataclass
class Contradiction:
    """A contradiction detected across sources."""

    topic: str
    viewpoint_a: str
    viewpoint_b: str
    source_a: str = ""
    source_b: str = ""


@dataclass
class Finding:
    """A core finding extracted from sources."""

    statement: str
    source_urls: list[str] = field(default_factory=list)
    confidence: float = 0.5  # 0-1
    evidence: str = ""  # optional, reporter will extract from sources directly


@dataclass
class SourceAssignment:
    """Maps a source to its most relevant sub-questions."""

    source_url: str
    source_title: str
    sub_question_id: int
    relevance_to_sq: float  # 0-1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "source_title": self.source_title,
            "sub_question_id": self.sub_question_id,
            "relevance_to_sq": self.relevance_to_sq,
        }


@dataclass
class SynthesisResult:
    """Information synthesis result."""

    findings: list[Finding] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    knowledge_gaps: list[KnowledgeGap] = field(default_factory=list)
    coverage_score: float = 0.0  # 0-1: how well sources cover all sub-questions
    source_assignments: list[SourceAssignment] = field(default_factory=list)  # source → sub_question mapping
    sources: list[SearchResult] = field(default_factory=list)  # pass-through original sources
    synthesized_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [
                {"statement": f.statement, "evidence": getattr(f, "evidence", ""), "source_urls": f.source_urls}
                for f in self.findings
            ],
            "contradictions": [
                {
                    "topic": c.topic,
                    "viewpoint_a": c.viewpoint_a,
                    "viewpoint_b": c.viewpoint_b,
                }
                for c in self.contradictions
            ],
            "knowledge_gaps": [g.description for g in self.knowledge_gaps],
            "coverage_score": self.coverage_score,
        }


@dataclass
class ReportMetrics:
    """Self-evaluation metrics for a research report."""

    completeness: float = 0.0  # 0-10: are all sub-questions answered?
    accuracy: float = 0.0  # 0-10: are findings well-supported?
    readability: float = 0.0  # 0-10: is the report well-structured?
    overall: float = 0.0  # weighted average
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "readability": self.readability,
            "overall": self.overall,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
        }


@dataclass
class ResearchResult:
    """Final research result with report and metrics."""

    topic: str
    status: ResearchStatus
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    report: str | None = None
    plan: ResearchPlan | None = None
    synthesis: SynthesisResult | None = None
    metrics: ReportMetrics | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    iterations: int = 0
    total_sources: int = 0
    quality_score: float = 0.0  # derived from metrics.overall

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "status": self.status.value,
            "report": self.report,
            "iterations": self.iterations,
            "total_sources": self.total_sources,
            "quality_score": self.quality_score,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ResearchConfig:
    """Configuration for the research pipeline."""

    max_iterations: int = 3
    max_sources_per_question: int = 10
    min_coverage_threshold: float = 0.7
    search_timeout: int = 30
    default_depth: str = "normal"  # quick / normal / deep
    enable_self_evaluation: bool = True
    evaluation_threshold: float = 6.0  # trigger retry if overall score < threshold
