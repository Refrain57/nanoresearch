"""
轻量端到端测试：验证 Ragas testset 生成流水线的所有 monkey-patch 在真实 API 下正常工作。

运行方式（在项目根目录 D:/Code/nanobot/backend）：
    .venv/Scripts/python tests/eval/test_ragas_e2e.py

耗时预估：3-8 分钟（5 个 chunk，2 个 sample，~25 次 LLM/embedding 调用）
"""

import json
import logging
import pathlib
import queue
import sys
import threading
import types as _types

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("ragas_e2e")

# ---------------------------------------------------------------------------
# API 配置
# ---------------------------------------------------------------------------

def _load_api_config() -> dict:
    from nanoresearch.config.loader import get_nanoresearch_home
    cfg_path = get_nanoresearch_home() / "config.json"
    if not cfg_path.exists():
        sys.exit(f"[SKIP] {cfg_path} not found")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    providers = cfg.get("providers", {})
    deepseek  = providers.get("deepseek", {})
    dashscope = providers.get("dashscope", {})
    return {
        "gen_api_key":  deepseek.get("apiKey", ""),
        "gen_base_url": deepseek.get("apiBase", "https://api.deepseek.com"),
        "emb_api_key":  dashscope.get("apiKey", ""),
        "emb_base_url": dashscope.get("apiBase", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    }


# ---------------------------------------------------------------------------
# 5 个短文本 chunk
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    ("chunk-001",
     "BM25 是一种基于词频-逆文档频率的稀疏检索算法。"
     "它通过计算查询词与文档中词汇的匹配度来排序候选文档，"
     "擅长处理关键词精确匹配场景，在法律、医疗等专业领域效果突出。"
     "BM25 的超参数 k1 控制词频饱和度，b 控制文档长度归一化。"),
    ("chunk-002",
     "Dense Embedding 检索使用神经网络将查询和文档映射到同一高维向量空间，"
     "通过余弦相似度或内积衡量语义相关性。"
     "与 BM25 相比，Dense 检索能捕捉词汇之间的语义关联，"
     "对同义词、近义词具有更好的泛化能力，适合开放域问答场景。"),
    ("chunk-003",
     "混合检索将 BM25 稀疏检索与 Dense Embedding 检索的结果融合，"
     "通常采用 RRF（Reciprocal Rank Fusion）算法按排名倒数加权合并两路召回结果。"
     "RRF 公式为 score = 1/(k + rank)，其中 k 通常取 60。"
     "混合检索兼顾精确匹配与语义理解，在多数评测基准上优于单路检索。"),
    ("chunk-004",
     "Cross-Encoder 重排序在粗排阶段之后对候选文档进行精细打分。"
     "与 Bi-Encoder 不同，Cross-Encoder 将查询与文档拼接后联合编码，"
     "能够建模更深层的交互关系，精排效果显著优于 Bi-Encoder。"
     "常用模型包括 BAAI/bge-reranker-v2-m3，支持中英文双语重排。"),
    ("chunk-005",
     "RAG（Retrieval-Augmented Generation）将检索与生成结合。"
     "系统先从知识库中检索相关文档片段，再将其作为上下文输入大语言模型生成答案。"
     "RAG 可有效缓解大模型幻觉问题，并支持在不微调模型的情况下注入最新知识。"
     "评估 RAG 质量的常用指标包括 Faithfulness、Answer Relevancy 和 Context Recall。"),
]


# ---------------------------------------------------------------------------
# 流水线（在独立线程中运行，复制 eval_router 的完整 patch 逻辑）
# ---------------------------------------------------------------------------

def _run_pipeline(cfg: dict) -> object:
    from openai import AsyncOpenAI
    from langchain_core.documents import Document as LCDocument
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
    from ragas.testset.transforms import default_transforms_for_prechunked
    from ragas.testset.transforms.extractors.llm_based import (
        ThemesExtractor, SummaryExtractor, NERExtractor,
    )
    from ragas.testset.transforms.extractors.embeddings import EmbeddingExtractor
    from ragas.testset.transforms.relationship_builders.cosine import CosineSimilarityBuilder
    from ragas.testset.graph import Relationship as _Relationship
    import numpy as _np
    from ragas.run_config import RunConfig

    gen_client = AsyncOpenAI(base_url=cfg["gen_base_url"], api_key=cfg["gen_api_key"])
    emb_client = AsyncOpenAI(base_url=cfg["emb_base_url"], api_key=cfg["emb_api_key"])
    gen_llm = llm_factory("deepseek-v4-flash", client=gen_client, max_tokens=4096)
    gen_emb = embedding_factory("openai", model="text-embedding-v4",
                                client=emb_client, interface="modern")

    _transforms = default_transforms_for_prechunked(llm=gen_llm, embedding_model=gen_emb)

    s_skip=[0]; s_total=[0]; t_skip=[0]; t_total=[0]; n_skip=[0]; n_total=[0]

    def _patch(obj):
        items = obj if isinstance(obj, list) else getattr(obj, "transformations", None)
        if items is None:
            return
        for t in items:
            if isinstance(t, SummaryExtractor):
                _orig = t.extract.__func__
                async def _f(self, node, _orig=_orig):
                    s_total[0] += 1
                    try: return await _orig(self, node)
                    except Exception as e:
                        s_skip[0] += 1
                        log.warning("SummaryExtractor skip %s: %s", getattr(node,"id","?"), e)
                        return self.property_name, "[extraction skipped]"
                t.extract = _types.MethodType(_f, t)

            elif isinstance(t, ThemesExtractor):
                _orig = t.extract.__func__
                async def _f(self, node, _orig=_orig):
                    t_total[0] += 1
                    try: return await _orig(self, node)
                    except Exception as e:
                        t_skip[0] += 1
                        log.warning("ThemesExtractor skip %s: %s", getattr(node,"id","?"), e)
                        return self.property_name, []
                t.extract = _types.MethodType(_f, t)

            elif isinstance(t, EmbeddingExtractor):
                _orig = t.extract.__func__
                async def _f(self, node, _orig=_orig):
                    try: text = node.get_property(self.embed_property_name)
                    except Exception: text = node.properties.get(self.embed_property_name, "")
                    if not (text or "").strip():
                        log.warning("EmbeddingExtractor skip %s — empty '%s'",
                                    getattr(node,"id","?"), self.embed_property_name)
                        return self.property_name, []
                    return await _orig(self, node)
                t.extract = _types.MethodType(_f, t)

            elif isinstance(t, NERExtractor):
                _orig = t.extract.__func__
                async def _f(self, node, _orig=_orig):
                    n_total[0] += 1
                    try: return await _orig(self, node)
                    except Exception as e:
                        n_skip[0] += 1
                        log.warning("NERExtractor skip %s: %s", getattr(node,"id","?"), e)
                        return self.property_name, []
                t.extract = _types.MethodType(_f, t)

            elif isinstance(t, CosineSimilarityBuilder):
                def _plan(self, kg):
                    filtered = self.filter(kg)
                    pairs = [(n, n.get_property(self.property_name)) for n in filtered.nodes]
                    pairs = [(n, e) for n, e in pairs if e]
                    if len(pairs) < 2: return []
                    nodes_v, embs_v = zip(*pairs)
                    nodes_v, embs_v = list(nodes_v), list(embs_v)
                    if not all(len(e) == len(embs_v[0]) for e in embs_v):
                        log.warning("CosineSimilarityBuilder: inconsistent sizes, skip")
                        return []
                    async def _add(nodes_v=nodes_v, embs_v=embs_v):
                        for i, j, sim in self._find_similar_embedding_pairs(
                                _np.array(embs_v), self.threshold):
                            kg.relationships.append(_Relationship(
                                source=nodes_v[i], target=nodes_v[j],
                                type=self.new_property_name,
                                properties={self.new_property_name: sim},
                                bidirectional=True,
                            ))
                    return [_add()]
                t.generate_execution_plan = _types.MethodType(_plan, t)

            _patch(t)

    _patch(_transforms)

    lc_docs = [
        LCDocument(page_content=content, metadata={"chunk_id": cid, "source": cid})
        for cid, content in SAMPLE_CHUNKS
    ]

    result = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb).generate_with_chunks(
        lc_docs,
        testset_size=2,
        transforms=_transforms,
        query_distribution=[(SingleHopSpecificQuerySynthesizer(llm=gen_llm), 1.0)],
        run_config=RunConfig(max_workers=3, max_retries=3, max_wait=30, timeout=90),
    )

    log.info("Extractor stats: summary %d/%d skip, themes %d/%d skip, ner %d/%d skip",
             s_skip[0], s_total[0], t_skip[0], t_total[0], n_skip[0], n_total[0])
    return result


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    cfg = _load_api_config()
    log.info("Config loaded. Starting pipeline (est. 3-8 min)...")

    result_q: queue.Queue = queue.Queue()
    exc_q:    queue.Queue = queue.Queue()

    def _thread():
        try:
            result_q.put(_run_pipeline(cfg))
        except Exception as e:
            exc_q.put(e)

    t = threading.Thread(target=_thread, daemon=True)
    t.start()
    t.join(timeout=600)

    if not t.is_alive() and not exc_q.empty():
        err = exc_q.get()
        log.error("FAILED: %s", err)
        import traceback; traceback.print_exception(type(err), err, err.__traceback__)
        sys.exit(1)

    if t.is_alive():
        log.error("TIMEOUT: pipeline exceeded 10 minutes")
        sys.exit(1)

    result = result_q.get()
    samples = result.samples

    print(f"\n{'='*60}")
    print(f"RESULT: {len(samples)} sample(s) generated")
    print('='*60)
    for i, s in enumerate(samples):
        es = s.eval_sample
        q  = (getattr(es, "user_input",  "") or "")[:100]
        a  = (getattr(es, "reference",   "") or "")[:100]
        print(f"\n[{i+1}] Q: {q}")
        print(f"     A: {a}")

    if len(samples) < 1:
        log.error("FAILED: expected >=1 sample, got 0")
        sys.exit(1)

    print(f"\nPASS — pipeline completed without crash, {len(samples)} sample(s) produced")


if __name__ == "__main__":
    main()
