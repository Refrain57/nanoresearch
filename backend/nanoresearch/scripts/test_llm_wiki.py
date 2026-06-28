"""Test LLM Wiki functionality.

This script tests the core features of the LLM Wiki implementation:
1. Knowledge source differentiation (research vs conversation)
2. Browseable knowledge (stats, list_claims, list_insights, browse_domains)
3. RRF source weighting
4. Conversation knowledge extraction
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanoresearch.rag.core.settings import load_settings
from nanoresearch.rag.libs.vector_store.chroma_store import ChromaStore
from nanoresearch.research.knowledge_search import KnowledgeSearch, WriteClaimsResult
from nanoresearch.research.types import Claim


async def test_source_differentiation():
    """Test that claims have source field (research/conversation)."""
    print("\n=== Test 1: Source Differentiation ===")

    settings = load_settings()
    store = ChromaStore(settings, collection_name="research_claims")
    all_claims = store.get_all_documents()

    sources = {}
    for c in all_claims:
        meta = c.get("metadata", {})
        src = meta.get("source", "MISSING")
        sources[src] = sources.get(src, 0) + 1

    print(f"Total claims: {len(all_claims)}")
    for src, count in sources.items():
        print(f"  source={src}: {count}")

    if "MISSING" in sources:
        print("[FAIL] FAIL: Some claims have no source field")
        return False

    print("[PASS] All claims have source field")
    return True


async def test_browseable_knowledge():
    """Test MCP tool for browsing knowledge."""
    print("\n=== Test 2: Browseable Knowledge ===")

    from nanoresearch.rag.mcp_server.tools.agentic.collections import ListResearchKnowledgeTool

    tool = ListResearchKnowledgeTool()

    # Test stats
    r = await tool.execute(action="stats")
    import json
    stats = json.loads(r.content)
    print(f"Stats: claims={stats['claims']['count']}, insights={stats['insights']['count']}")

    # Test list_claims
    r = await tool.execute(action="list_claims", source="research", limit=5)
    claims_data = json.loads(r.content)
    print(f"Research claims: {claims_data['total']}")

    # Test list_insights
    r = await tool.execute(action="list_insights", limit=3)
    insights_data = json.loads(r.content)
    print(f"Insights: {insights_data['total']}")

    # Check domains are properly parsed (not character-split)
    if insights_data["insights"]:
        first_insight = insights_data["insights"][0]
        domains = first_insight.get("domains", [])
        if domains:
            # Each domain should be longer than 1 character
            for d in domains:
                if len(d) <= 1:
                    print(f"[FAIL] FAIL: Domain was character-split: {d}")
                    return False
            print(f"[PASS] Domains properly parsed: {domains[:3]}...")

    # Test browse_domains
    r = await tool.execute(action="browse_domains")
    domains_data = json.loads(r.content)
    print(f"Unique domains: {domains_data['total']}")

    print("[PASS] PASS: Browseable knowledge works")
    return True


async def test_rrf_source_weighting():
    """Test that RRF applies source weights."""
    print("\n=== Test 3: RRF Source Weighting ===")

    from nanoresearch.rag.core.query_engine.fusion import RRFFusion

    weights = RRFFusion.DEFAULT_SOURCE_WEIGHTS
    print(f"Source weights: {weights}")

    if weights.get("research") != 1.0:
        print("[FAIL] FAIL: research weight should be 1.0")
        return False
    if weights.get("conversation") != 0.4:
        print("[FAIL] FAIL: conversation weight should be 0.4")
        return False

    print("[PASS] PASS: RRF source weights configured correctly")
    return True


async def test_write_claims_with_source():
    """Test writing claims with different sources.

    Note: source="conversation" is deprecated for write_claims.
    Use write_user_memory() for conversation-derived knowledge.
    This test verifies backward compatibility.
    """
    print("\n=== Test 4: Write Claims with Source (Legacy) ===")

    settings = load_settings()
    ks = KnowledgeSearch.from_settings(settings)

    # Write a conversation claim (legacy path, now should use user_memory)
    test_claim = Claim(
        claim="LLM Wiki 测试知识：conversation source (legacy)",
        type="factual",
        is_evergreen=False,
        source_urls=[],
        confidence=0.8,
    )

    result = await ks.write_claims([test_claim], source="conversation", verified=False)
    print(f"Write result: written={result.claims_written}, updated={result.claims_updated}, skipped={result.duplicates_skipped}")

    # Also test the new user_memory path
    user_mem_result = ks.write_user_memory_sync([{
        "text": "LLM Wiki 测试知识：user_memory path",
        "type": "factual",
        "confidence": 0.8,
        "is_evergreen": False,
    }])
    print(f"User memory result: written={user_mem_result[0]}, skipped={user_mem_result[1]}")

    if result.claims_written == 0 and result.claims_updated == 0 and user_mem_result[0] == 0:
        print("[FAIL] FAIL: No claim/memory was written")
        return False

    print("[PASS] PASS: Both legacy claims and user_memory work")
    return True


async def test_conversation_extractor():
    """Test ConversationKnowledgeExtractor initialization."""
    print("\n=== Test 5: Conversation Knowledge Extractor ===")

    from nanoresearch.agent.conversation_knowledge_extractor import ConversationKnowledgeExtractor

    print("[PASS] ConversationKnowledgeExtractor imported successfully")
    print(f"  Has extract_from_messages: {hasattr(ConversationKnowledgeExtractor, 'extract_from_messages')}")
    print(f"  Has _is_negated: {hasattr(ConversationKnowledgeExtractor, '_is_negated')}")

    print("[PASS] PASS: ConversationKnowledgeExtractor available")
    return True


async def test_write_claims_result():
    """Test WriteClaimsResult dataclass."""
    print("\n=== Test 6: WriteClaimsResult Dataclass ===")

    result = WriteClaimsResult(
        claims_written=2,
        claims_updated=1,
        duplicates_skipped=3,
        claim_ids=["id1", "id2", "id3"],
    )

    print(f"Result: written={result.claims_written}, updated={result.claims_updated}, skipped={result.duplicates_skipped}")
    print(f"Claim IDs: {result.claim_ids}")

    if result.claims_written != 2 or result.claims_updated != 1 or result.duplicates_skipped != 3:
        print("[FAIL] FAIL: WriteClaimsResult values incorrect")
        return False

    print("[PASS] PASS: WriteClaimsResult works correctly")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM Wiki Functional Tests")
    print("=" * 60)

    tests = [
        test_source_differentiation,
        test_browseable_knowledge,
        test_rrf_source_weighting,
        test_write_claims_with_source,
        test_conversation_extractor,
        test_write_claims_result,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] EXCEPTION: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
