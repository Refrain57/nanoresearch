"""Diagnose ThemesExtractor token usage on a real chunk."""
import asyncio, json, os

os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:123456@localhost:5432/nanoresearch"

with open("C:/Users/Augix/.nanoresearch/config.json") as f:
    cfg = json.load(f)

deepseek_key = cfg["providers"]["deepseek"]["apiKey"]
deepseek_url = cfg["providers"]["deepseek"]["apiBase"]
generator_model = cfg["agents"]["defaults"]["model"]

print(f"Model: {generator_model}")
print(f"Base URL: {deepseek_url}")

# Grab a real chunk from DB
from nanoresearch.storage.database import init_engine, get_session_factory
from sqlalchemy import text

async def get_chunks():
    init_engine()
    sf = get_session_factory()
    async with sf() as db:
        r = await db.execute(text("""
            SELECT c.content, length(c.content) as char_len
            FROM kb_chunks c
            JOIN kb_documents d ON c.document_id = d.id
            JOIN knowledge_bases kb ON c.kb_id = kb.id
            WHERE d.status = 'indexed'
            ORDER BY length(c.content) DESC
            LIMIT 5
        """))
        return r.fetchall()

chunks = asyncio.run(get_chunks())
print(f"\nTop 5 longest chunks:")
for i, (content, clen) in enumerate(chunks):
    print(f"  [{i}] {clen} chars: {repr(content[:80])}")

# Pick the longest chunk for testing
test_chunk_text = chunks[0][0]
print(f"\nTesting on chunk of {len(test_chunk_text)} chars")

# Patch OpenAI to intercept raw response
from openai import AsyncOpenAI
import openai

original_create = None

async def run_test():
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.testset.graph import Node, NodeType, KnowledgeGraph
    from ragas.testset.transforms.extractors.llm_based import ThemesExtractor

    gen_client = AsyncOpenAI(base_url=deepseek_url, api_key=deepseek_key)

    # Monkey-patch to capture raw response
    original = gen_client.chat.completions.create
    captured = {}

    async def capturing_create(*args, **kwargs):
        print(f"\n[API CALL] max_tokens={kwargs.get('max_tokens')}")
        print(f"[API CALL] model={kwargs.get('model')}")
        resp = await original(*args, **kwargs)
        captured["usage"] = resp.usage
        captured["finish_reason"] = resp.choices[0].finish_reason if resp.choices else None
        captured["content_len"] = len(resp.choices[0].message.content or "") if resp.choices else 0
        return resp

    gen_client.chat.completions.create = capturing_create

    gen_llm = llm_factory(generator_model, client=gen_client, max_tokens=8192)
    extractor = ThemesExtractor(llm=gen_llm)

    node = Node(
        type=NodeType.CHUNK,
        properties={"page_content": test_chunk_text, "document_metadata": {"chunk_id": "test"}}
    )

    try:
        prop_name, themes = await extractor.extract(node)
        print(f"\n[SUCCESS] themes count={len(themes)}")
        print(f"[SUCCESS] themes: {themes[:5]}")
    except Exception as e:
        print(f"\n[FAILED] {type(e).__name__}: {e}")

    if captured:
        u = captured.get("usage")
        print(f"\n[TOKEN USAGE]")
        print(f"  prompt_tokens:     {getattr(u, 'prompt_tokens', 'N/A')}")
        print(f"  completion_tokens: {getattr(u, 'completion_tokens', 'N/A')}")
        print(f"  total_tokens:      {getattr(u, 'total_tokens', 'N/A')}")
        if hasattr(u, 'completion_tokens_details'):
            print(f"  completion_details: {u.completion_tokens_details}")
        print(f"  finish_reason:     {captured.get('finish_reason')}")
        print(f"  content_len:       {captured.get('content_len')} chars")

asyncio.run(run_test())
