"""Quick smoke test: SummaryExtractor + EmbeddingExtractor + question synthesis."""
import asyncio
import json
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms_for_prechunked, apply_transforms
from ragas.run_config import RunConfig
from langchain_core.documents import Document as LCDocument

from nanoresearch.config.loader import get_nanoresearch_home

FAKE_CHUNKS = [
    "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的特征表示。",
    "Transformer 架构由 Attention Is All You Need 论文提出，已成为 NLP 的基础模型。",
    "RAG（检索增强生成）结合了检索系统和生成模型，能够利用外部知识库回答问题。",
    "向量数据库通过存储高维嵌入向量支持语义搜索，常见实现有 Chroma、Pinecone、Weaviate 等。",
    "BM25 是一种基于词频和逆文档频率的稀疏检索算法，在关键词匹配上效果优异。",
]

with open(get_nanoresearch_home() / "config.json") as f:
    cfg = json.load(f)

deepseek_key = cfg["providers"]["deepseek"]["apiKey"]
deepseek_url = cfg["providers"]["deepseek"]["apiBase"]
dashscope_key = cfg["providers"]["dashscope"]["apiKey"]
dashscope_url = cfg["providers"]["dashscope"]["apiBase"]

gen_client = AsyncOpenAI(base_url=deepseek_url, api_key=deepseek_key)
emb_client = AsyncOpenAI(base_url=dashscope_url, api_key=dashscope_key)

gen_llm = llm_factory("deepseek-v4-flash", client=gen_client, max_tokens=2048)
gen_emb = embedding_factory("openai", model="text-embedding-v4", client=emb_client, interface="modern")

run_cfg = RunConfig(max_workers=10, max_retries=3, max_wait=30, timeout=60)

# === Stage 1: Transforms ===
print("=== Stage 1: Transforms (Summary + Embedding) ===")
nodes = [
    Node(type=NodeType.CHUNK, properties={"page_content": c, "document_metadata": {"chunk_id": str(i)}})
    for i, c in enumerate(FAKE_CHUNKS)
]
kg = KnowledgeGraph(nodes=nodes)
transforms = default_transforms_for_prechunked(llm=gen_llm, embedding_model=gen_emb)
apply_transforms(kg, transforms, run_config=run_cfg)

ok_summary = sum(1 for n in kg.nodes if n.properties.get("summary"))
ok_emb = sum(1 for n in kg.nodes if n.properties.get("summary_embedding"))
print(f"Summary: {ok_summary}/{len(kg.nodes)}")
print(f"Embedding: {ok_emb}/{len(kg.nodes)}")

for node in kg.nodes:
    s = node.properties.get("summary", "")
    e = node.properties.get("summary_embedding", [])
    print(f"  summary={repr(s[:60]) if s else 'MISSING'}")
    print(f"  emb_dim={len(e)}")

# === Stage 2: Question synthesis (clients created fresh inside same context) ===
print("\n=== Stage 2: Question synthesis ===")
lc_docs = [LCDocument(page_content=c, metadata={"chunk_id": str(i)}) for i, c in enumerate(FAKE_CHUNKS)]

gen_client2 = AsyncOpenAI(base_url=deepseek_url, api_key=deepseek_key)
emb_client2 = AsyncOpenAI(base_url=dashscope_url, api_key=dashscope_key)
gen_llm2 = llm_factory("deepseek-v4-flash", client=gen_client2, max_tokens=2048)
gen_emb2 = embedding_factory("openai", model="text-embedding-v4", client=emb_client2, interface="modern")

generator = TestsetGenerator(llm=gen_llm2, embedding_model=gen_emb2)
dataset = generator.generate_with_chunks(
    lc_docs,
    testset_size=2,
    query_distribution=[(SingleHopSpecificQuerySynthesizer(llm=gen_llm2), 1.0)],
    run_config=run_cfg,
)

print(f"Generated {len(dataset.samples)} samples:")
for s in dataset.samples:
    es = s.eval_sample
    print(f"  Q: {getattr(es, 'user_input', '')}")
    print(f"  A: {getattr(es, 'reference', '')[:80]}")
    print()
