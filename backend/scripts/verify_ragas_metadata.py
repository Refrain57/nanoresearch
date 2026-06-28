"""Verify that RAGAS TestsetGenerator preserves chunk_id in reference_contexts metadata.

Run from backend/ directory:
    python -m scripts.verify_ragas_metadata

Expected output: each sample's reference_contexts[i].metadata contains 'chunk_id' ✓
If chunk_id is MISSING, the dataset generation endpoint needs a BM25 fallback.
"""

from __future__ import annotations

import sys
import uuid

from langchain_core.documents import Document


def build_fake_docs(n: int = 5) -> list[Document]:
    docs = []
    for i in range(n):
        chunk_id = str(uuid.uuid4())
        docs.append(Document(
            page_content=f"Sample chunk {i}. " * 20,
            metadata={"chunk_id": chunk_id, "source": f"doc_{i}.txt"},
        ))
    return docs


def main() -> None:
    try:
        from ragas.testset import TestsetGenerator
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from openai import AsyncOpenAI
    except ImportError as e:
        print(f"Import error: {e}")
        return

    from nanoresearch.rag.core.settings import load_settings
    settings = load_settings()

    base_url = settings.llm.base_url or "https://api.openai.com/v1"
    api_key = settings.llm.api_key or "sk-placeholder"
    gen_model = settings.llm.model or "qwen-plus"
    emb_model = settings.embedding.model or "text-embedding-v3"

    # Command-line overrides: python scripts/verify_ragas_metadata.py qwen-plus text-embedding-v3
    if len(sys.argv) > 1:
        gen_model = sys.argv[1]
    if len(sys.argv) > 2:
        emb_model = sys.argv[2]

    print(f"Using LLM: {gen_model}  base_url: {base_url}")
    print(f"Using embedding: {emb_model}")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    gen_llm = llm_factory(gen_model, client=client, max_tokens=4096)
    gen_emb = embedding_factory("openai", model=emb_model, client=client, interface="modern")

    docs = build_fake_docs(5)
    print(f"\nBuilt {len(docs)} fake docs. Sample metadata: {docs[0].metadata}")

    generator = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb)

    try:
        testset = generator.generate_with_langchain_docs(docs, testset_size=5)
    except Exception as e:
        print(f"\nGeneration failed: {e}")
        return

    samples = testset.to_list()
    print(f"\nGenerated {len(samples)} samples")
    passed = 0
    failed = 0
    for i, sample in enumerate(samples[:5]):
        contexts = sample.get("reference_contexts") or []
        print(f"\n--- Sample {i} ---")
        print(f"  user_input: {sample.get('user_input', '')[:80]}")
        print(f"  reference_contexts count: {len(contexts)}")
        for j, ctx in enumerate(contexts[:2]):
            meta = ctx.metadata if hasattr(ctx, "metadata") else {}
            print(f"  ctx[{j}] metadata keys: {list(meta.keys())}")
            if "chunk_id" in meta:
                print(f"    chunk_id: {meta['chunk_id']} ✓")
                passed += 1
            else:
                print(f"    chunk_id: MISSING ✗  (metadata={meta})")
                failed += 1

    print(f"\n{'='*40}")
    print(f"Result: {passed} ✓  {failed} ✗")
    if failed == 0:
        print("chunk_id 透传成功 → 现有 generate 端点逻辑无需修改")
    else:
        print("chunk_id 透传失败 → 需要在 dataset generation 里加 BM25 fallback")


if __name__ == "__main__":
    main()
