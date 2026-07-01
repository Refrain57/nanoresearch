"""Task 3: loop.py captures RAG citations, forwards via on_citations,
and embeds the accumulated deduped list in the assistant message as _citations.

Uses a scripted provider (two kb_search tool-call turns then a plain-text turn)
and a scripted tool layer whose kb_search results repeat chunk_id ``c1`` so the
merge/dedup + contiguous re-index behaviour is exercised.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanoresearch.agent.loop import AgentLoop
from nanoresearch.bus.queue import MessageBus
from nanoresearch.providers.base import GenerationSettings, LLMResponse, ToolCallRequest


def _citation(index: int, chunk_id: str, source: str, score: float,
              snippet: str, page: int | None, doc_id: str) -> dict:
    return {
        "index": index,
        "chunk_id": chunk_id,
        "source": source,
        "score": score,
        "snippet": snippet,
        "page": page,
        "doc_id": doc_id,
    }


@pytest.fixture
def loop_with_scripted_rag(tmp_path):
    """Build an AgentLoop whose LLM calls kb_search twice then answers, and
    whose kb_search tool returns JSON with overlapping citations (c1 twice, c2)."""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    # Keep prompt token estimate tiny so no consolidation fires mid-turn.
    provider.estimate_prompt_tokens.return_value = (50, "test-counter")

    calls = {"n": 0}

    async def chat_with_retry(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="t1", name="kb_search",
                                            arguments={"query": "alpha"})],
            )
        if calls["n"] == 2:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="t2", name="kb_search",
                                            arguments={"query": "beta"})],
            )
        return LLMResponse(content="Final answer [1][2].", tool_calls=[])

    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = AsyncMock()

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )
    loop.tools.get_definitions = MagicMock(return_value=[])

    # First kb_search returns c1; second returns c1 (dup) + c2.
    res1 = json.dumps({
        "success": True,
        "chunks": [],
        "citations": [_citation(1, "c1", "doc_a.pdf", 0.91, "snip-1", 3, "da")],
    })
    res2 = json.dumps({
        "success": True,
        "chunks": [],
        "citations": [
            _citation(1, "c1", "doc_a.pdf", 0.91, "snip-1", 3, "da"),
            _citation(2, "c2", "doc_b.pdf", 0.77, "snip-2", 5, "db"),
        ],
    })
    loop.tools.execute = AsyncMock(side_effect=[res1, res2])
    return loop


@pytest.mark.asyncio
async def test_after_iteration_forwards_deduped_citations(loop_with_scripted_rag):
    """Two kb_search hits on the same chunk_id -> on_citations receives a merged,
    deduped list with contiguous 1..N indices."""
    loop = loop_with_scripted_rag
    batches: list[list[dict]] = []

    async def on_citations(items):
        batches.append(items)

    await loop.process_direct("q", session_key="cli:cit1", on_citations=on_citations)

    assert batches, "on_citations should have been called at least once"
    final = batches[-1]
    cids = [c["chunk_id"] for c in final]
    assert cids == ["c1", "c2"]                        # deduped + merged
    assert len(cids) == len(set(cids))                 # no duplicates
    assert [c["index"] for c in final] == list(range(1, len(final) + 1))  # contiguous


@pytest.mark.asyncio
async def test_save_turn_embeds_citations_in_content(loop_with_scripted_rag):
    """The accumulated citations are persisted on the assistant message dict."""
    loop = loop_with_scripted_rag

    async def _noop(items):
        return None

    await loop.process_direct("q", session_key="cli:cit2", on_citations=_noop)

    session = await loop.sessions.get_or_create("cli:cit2")
    assistants = [m for m in session.messages if m.get("role") == "assistant"]
    assert assistants, "expected at least one assistant message persisted"
    final = assistants[-1]
    assert final.get("_citations"), "final assistant message should carry _citations"
    assert [c["chunk_id"] for c in final["_citations"]] == ["c1", "c2"]
    assert final["_citations"][0]["index"] == 1


@pytest.mark.asyncio
async def test_citations_captured_without_on_citations_callback(loop_with_scripted_rag):
    """Citation capture/embedding is independent of the on_citations callback."""
    loop = loop_with_scripted_rag

    await loop.process_direct("q", session_key="cli:cit3")  # no on_citations

    session = await loop.sessions.get_or_create("cli:cit3")
    final = [m for m in session.messages if m.get("role") == "assistant"][-1]
    assert [c["chunk_id"] for c in final.get("_citations", [])] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_turn_citations_reset_between_turns(loop_with_scripted_rag):
    """A turn with no RAG tool must not inherit the previous turn's citations."""
    loop = loop_with_scripted_rag

    # Turn 1: produces citations (consumes both scripted kb_search results).
    await loop.process_direct("q1", session_key="cli:cit4")

    # Turn 2: LLM answers directly with no tool calls -> no citations.
    async def chat_plain(**kwargs):
        return LLMResponse(content="direct answer", tool_calls=[])

    loop.provider.chat_with_retry = chat_plain

    await loop.process_direct("q2", session_key="cli:cit4")

    session = await loop.sessions.get_or_create("cli:cit4")
    assistants = [m for m in session.messages if m.get("role") == "assistant"]
    last = assistants[-1]
    assert last.get("content") == "direct answer"
    assert "_citations" not in last, "second turn must not inherit prior citations"
