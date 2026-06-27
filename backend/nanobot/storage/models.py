"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import ARRAY, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from nanobot.storage.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    uid: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class UserSettings(Base):
    __tablename__ = "user_settings"

    uid: Mapped[str] = mapped_column(String, primary_key=True)
    model: Mapped[str | None] = mapped_column(String)
    max_iterations: Mapped[int | None] = mapped_column(Integer)
    extra: Mapped[dict] = mapped_column(JSONB, default=dict)  # fast_model, future fields
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String, default="1.0.0")
    capabilities: Mapped[dict] = mapped_column(JSONB, default=dict)
    skills_config: Mapped[list] = mapped_column(JSONB, default=list)
    tools_config: Mapped[list] = mapped_column(JSONB, default=list)
    default_model: Mapped[str | None] = mapped_column(String)
    provider: Mapped[str | None] = mapped_column(String)
    persona: Mapped[str | None] = mapped_column(Text)
    harness: Mapped[dict] = mapped_column(JSONB, default=dict)
    max_iterations: Mapped[int] = mapped_column(Integer, default=40)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str | None] = mapped_column(String, ForeignKey("users.uid"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class AgentKnowledgeBinding(Base):
    """Many-to-many: Agent ↔ KnowledgeBase."""
    __tablename__ = "agent_knowledge_bindings"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), primary_key=True
    )


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (UniqueConstraint("session_key", name="uq_conversations_session_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uid: Mapped[str] = mapped_column(String, ForeignKey("users.uid"), nullable=False, index=True)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(String)
    channel: Mapped[str] = mapped_column(String, default="web")
    channel_chat_id: Mapped[str | None] = mapped_column(String)
    session_key: Mapped[str | None] = mapped_column(String)  # "channel:chat_id"
    conv_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    last_consolidated: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)  # full message dict
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    uid: Mapped[str] = mapped_column(String, ForeignKey("users.uid"), nullable=False, index=True)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    input_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=True)
    output_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=True)
    model_used: Mapped[str | None] = mapped_column(String)
    tool_calls: Mapped[list] = mapped_column(JSONB, default=list)
    thinking: Mapped[str | None] = mapped_column(Text)
    tokens_used: Mapped[dict] = mapped_column(JSONB, default=dict)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    sandbox_id: Mapped[str | None] = mapped_column(String)
    artifacts: Mapped[list] = mapped_column(JSONB, default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Phase 4: Knowledge Base & Evaluation
# ---------------------------------------------------------------------------

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uid: Mapped[str] = mapped_column(String, ForeignKey("users.uid"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str | None] = mapped_column(String)
    chroma_collection: Mapped[str | None] = mapped_column(String)
    chunk_strategy: Mapped[str] = mapped_column(String, default="auto")
    chunk_size: Mapped[int] = mapped_column(Integer, default=512)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=50)
    enable_graph_expansion: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String, default="active")
    doc_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class KbDocument(Base):
    __tablename__ = "kb_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str | None] = mapped_column(String)
    file_size: Mapped[int | None] = mapped_column(Integer)
    mime_type: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="uploaded")  # uploaded|parsing|indexing|indexed|error
    pdf_parser: Mapped[str] = mapped_column(String, default="marker")  # markitdown|marker|mineru
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_msg: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KbChunk(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("kb_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chroma_id: Mapped[str | None] = mapped_column(String, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Knowledge Graph Tables
# ---------------------------------------------------------------------------

class KgEntity(Base):
    __tablename__ = "kg_entities"
    __table_args__ = (UniqueConstraint("kb_id", "name", "label", name="uq_kg_entity_kb_name_label"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False, default="Entity")
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KgEntityMention(Base):
    __tablename__ = "kg_entity_mentions"
    __table_args__ = (UniqueConstraint("entity_id", "chunk_id", name="uq_kg_entity_mention"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("kb_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KgTriple(Base):
    __tablename__ = "kg_triples"
    __table_args__ = (UniqueConstraint("kb_id", "source_id", "target_id", "label", name="uq_kg_triple"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String, nullable=False, default="RELATED_TO")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KgTripleMention(Base):
    __tablename__ = "kg_triple_mentions"
    __table_args__ = (UniqueConstraint("triple_id", "chunk_id", name="uq_kg_triple_mention"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    triple_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("kg_triples.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("kb_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class EvalDataset(Base):
    __tablename__ = "eval_datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class EvalDatasetItem(Base):
    __tablename__ = "eval_dataset_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("eval_datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    item_index: Mapped[int | None] = mapped_column(Integer)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    gold_chunk_ids: Mapped[list | None] = mapped_column(JSONB)  # list of chunk id strings
    gold_answer: Mapped[str | None] = mapped_column(Text)
    question_type: Mapped[str | None] = mapped_column(Text)  # single_hop | multi_context


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("eval_datasets.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str | None] = mapped_column(String)
    eval_type: Mapped[str] = mapped_column(String, default="quick")  # quick|ragas
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|running|completed|failed
    retrieval_config: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    overall_score: Mapped[float | None] = mapped_column(Float)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class EvalRunItem(Base):
    __tablename__ = "eval_run_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    query: Mapped[str | None] = mapped_column(Text)
    gold_answer: Mapped[str | None] = mapped_column(Text)
    generated_answer: Mapped[str | None] = mapped_column(Text)
    retrieved_contexts: Mapped[list | None] = mapped_column(JSONB)  # chunk ids (quick) or texts (ragas)
    item_metrics: Mapped[dict] = mapped_column("metrics", JSONB, default=dict)
    question_type: Mapped[str | None] = mapped_column(Text)


# ---------------------------------------------------------------------------
# Agent Evaluation — eval batch runs, run snapshots, test cases, badcases
# ---------------------------------------------------------------------------

class AgentEvalRun(Base):
    __tablename__ = "agent_eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    dataset_type: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)  # pending/running/completed/failed
    created_by: Mapped[str] = mapped_column(String, ForeignKey("users.uid"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    passed_cases: Mapped[int] = mapped_column(Integer, default=0)
    failed_cases: Mapped[int] = mapped_column(Integer, default=0)
    summary_scores: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    # v2 regression fields
    baseline_eval_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_eval_runs.id", ondelete="SET NULL"), nullable=True)
    regression_diffs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    has_regression: Mapped[bool] = mapped_column(Boolean, default=False)


class AgentRunSnapshot(Base):
    __tablename__ = "agent_run_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    uid: Mapped[str] = mapped_column(String, ForeignKey("users.uid"), nullable=False, index=True)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt_version: Mapped[str | None] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    # Process metrics
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    ttft_ms: Mapped[float | None] = mapped_column(Float)
    total_duration_ms: Mapped[float | None] = mapped_column(Float)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    llm_call_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    run_status: Mapped[str] = mapped_column(String, default="success")
    # Structured run data
    tool_call_chain: Mapped[list] = mapped_column(JSONB, default=list)
    llm_calls: Mapped[list] = mapped_column(JSONB, default=list)
    final_response: Mapped[str | None] = mapped_column(Text)
    # Evaluation scores (written back after scoring)
    scores: Mapped[dict | None] = mapped_column(JSONB)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    failed_dimensions: Mapped[list | None] = mapped_column(JSONB)
    # Badcase fields (inline — avoids a separate join table)
    is_badcase: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    badcase_trigger: Mapped[str | None] = mapped_column(String)
    badcase_category: Mapped[str | None] = mapped_column(String)
    badcase_status: Mapped[str] = mapped_column(String, default="pending")
    root_cause: Mapped[str | None] = mapped_column(Text)
    fix_notes: Mapped[str | None] = mapped_column(Text)
    annotated_by: Mapped[str | None] = mapped_column(String)
    annotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Batch eval run link (nullable — live snapshots have no batch run)
    eval_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_eval_runs.id", ondelete="SET NULL"), nullable=True, index=True)
    # v2 eval fields
    tool_recordings: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    semantic_category: Mapped[str | None] = mapped_column(String, nullable=True)
    judge_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Auto root-cause classification (root_cause stores human annotation)
    root_cause_auto: Mapped[str | None] = mapped_column(String(32), nullable=True)
    root_cause_auto_confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)
    root_cause_auto_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Phase 0: structured context assembly decisions (ids + numbers + query text, no injected full text)
    context_trace: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class AgentTestCase(Base):
    __tablename__ = "agent_test_cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    session_history: Mapped[list] = mapped_column(JSONB, default=list)
    expected_tools: Mapped[list | None] = mapped_column(JSONB)
    expected_intent: Mapped[str | None] = mapped_column(Text)
    expected_keywords: Mapped[list | None] = mapped_column(JSONB)
    token_budget: Mapped[int | None] = mapped_column(Integer)
    human_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Flywheel fields
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)  # active | pending_review | rejected
    generated_from_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_run_snapshots.id", ondelete="SET NULL"), nullable=True)
    generation_reason: Mapped[str | None] = mapped_column(Text)

    # B4 metadata (Phase 1 of A1 tool-layer harness refactor)
    origin_badcase_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_run_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    target_dimension: Mapped[str] = mapped_column(Text, nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    added_by: Mapped[str] = mapped_column(Text, nullable=False)
    coverage_tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)

    __table_args__ = (
        Index("idx_agent_test_cases_target_dimension", "target_dimension"),
        Index("idx_agent_test_cases_origin_badcase_id", "origin_badcase_id"),
    )


class JudgeCalibrationLog(Base):
    __tablename__ = "judge_calibration_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    judge_model: Mapped[str] = mapped_column(String, nullable=False)
    mad_value: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    eval_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_eval_runs.id", ondelete="SET NULL"), nullable=True)


class OptimizationProposal(Base):
    __tablename__ = "optimization_proposals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    category: Mapped[str] = mapped_column(String, nullable=False)
    proposals: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)  # pending/approved/rejected
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String, ForeignKey("users.uid"), nullable=True)
