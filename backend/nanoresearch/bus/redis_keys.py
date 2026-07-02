"""Unified Redis key namespace and TTL constants."""
from __future__ import annotations


class RedisKeys:
    # Control signals — no TTL, manual DEL, non-evictable under volatile-lru
    @staticmethod
    def pending(session_key: str) -> str:
        return f"pending:{session_key}"

    @staticmethod
    def cancel(session_key: str) -> str:
        return f"cancel:{session_key}"

    @staticmethod
    def job(job_id: str) -> str:
        return f"job:{job_id}"

    # Run events stream — cross-process event delivery, 24h TTL
    RUN_EVENTS_TTL = 86400

    @staticmethod
    def run_events(run_id: str) -> str:
        return f"run_events:{run_id}"

    # Per-conversation live stream — server→client push for activity NOT tied to a run the
    # frontend started (e.g. a cron result delivered into the origin conversation). The web UI
    # holds one SSE per open conversation; cron delivery XADDs here so it appears live without
    # polling. Shares RUN_EVENTS_TTL (24h replay window).
    @staticmethod
    def conv_live(conversation_id: str) -> str:
        return f"conv_live:{conversation_id}"

    # Session short-term memory — 2 h TTL, MULTI/EXEC write
    SESSION_TTL = 7200

    @staticmethod
    def session_msg(uid: str, ch: str, chat_id: str) -> str:
        return f"session:msg:{uid}:{ch}:{chat_id}"

    @staticmethod
    def session_meta(uid: str, ch: str, chat_id: str) -> str:
        return f"session:meta:{uid}:{ch}:{chat_id}"

    # Config hot-cache — DEL on write
    AGENT_TTL = 1800
    USER_SETTINGS_TTL = 1800
    KB_META_TTL = 600

    @staticmethod
    def agent(agent_id: str) -> str:
        return f"agent:{agent_id}"

    @staticmethod
    def user_settings(uid: str) -> str:
        return f"user_settings:{uid}"

    @staticmethod
    def kb_meta(kb_id: str) -> str:
        return f"kb:meta:{kb_id}"

    # RAG cache — volatile-lru evictable
    CHUNK_TTL = 21600
    EMBEDDING_TTL = 3600

    @staticmethod
    def chunk(namespace: str, chunk_id: str) -> str:
        return f"chunk:{namespace}:{chunk_id}"

    @staticmethod
    def embedding(text_hash: str) -> str:
        return f"embedding:{text_hash}"

    # Agent inbox / dispatch (Phase 0) — per-(agent_id, conversation_id) addressing.
    # agent_id is "none" until Phase 2 fills real identity; structure is forward-compatible.
    AGENT_INBOX_TTL = 86400
    DISPATCH_NOTIFY = "dispatch_notify"
    DISPATCH_GROUP = "dispatch_cg"

    @staticmethod
    def agent_inbox(agent_id: str, conversation_id: str) -> str:
        return f"agent_inbox:{agent_id}:{conversation_id}"

    @staticmethod
    def agent_inbox_cursor(agent_id: str, conversation_id: str) -> str:
        return f"agent_inbox_cursor:{agent_id}:{conversation_id}"

    @staticmethod
    def agent_lock(agent_id: str, conversation_id: str) -> str:
        return f"agent_lock:{agent_id}:{conversation_id}"

    @staticmethod
    def continuation_lock(agent_id: str, conversation_id: str) -> str:
        # Phase 1: gate marker — "a subagent batch is completing / a continuation is pending or
        # running". Set atomically by the join when it empties pending; the parent run never
        # holds it (so the join never contends with the parent's agent_lock).
        return f"continuation_lock:{agent_id}:{conversation_id}"

    @staticmethod
    def subagent_results(session_key: str) -> str:
        # Phase 1: append-only staging list of subagent results for the continuation to drain.
        return f"subagent_results:{session_key}"

    # SSE stream (unchanged)
    @staticmethod
    def chat_events(chat_id: str) -> str:
        return f"chat_events:{chat_id}"

    # Pub/Sub channels
    INVALIDATE_SESSION = "invalidate:session"
    INVALIDATE_AGENT = "invalidate:agent"
    INVALIDATE_KB = "invalidate:kb"
