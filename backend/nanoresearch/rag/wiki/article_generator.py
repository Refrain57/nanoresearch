"""Wiki entity article generation (Phase 2 MVP): grounded free synthesis with [^n]."""
from __future__ import annotations

import hashlib


def evidence_signature(evidence: list[dict]) -> str:
    ids = sorted(str(e.get("chunk_id", "")) for e in evidence)
    return hashlib.sha256(",".join(ids).encode()).hexdigest()


def build_citations(evidence: list[dict]) -> list[dict]:
    out = []
    for i, e in enumerate(evidence, start=1):
        out.append({
            "index": i,
            "source": e.get("source", ""),
            "page": e.get("page"),
            "snippet": (e.get("content", "") or "")[:300],
        })
    return out


def build_article_prompt(name: str, facts: list[dict], evidence: list[dict]) -> tuple[str, str]:
    system = "你是知识库词条编写助手。只依据给定证据编写，不使用外部知识，不编造。"
    fact_lines = "\n".join(
        f"- {f.get('source')} —{f.get('label')}→ {f.get('target')}" for f in facts
    ) or "（无结构化事实）"
    ev_lines = "\n".join(
        f"[{i}] {e.get('content','')}" for i, e in enumerate(evidence, start=1)
    ) or "（无证据）"
    user = (
        f"实体：{name}\n\n"
        f"已知事实：\n{fact_lines}\n\n"
        f"证据（编号）：\n{ev_lines}\n\n"
        "请为该实体写一段简洁的中文词条正文（markdown）。要求：\n"
        "- 只综合上述证据，不确定或无证据支撑的内容不要写；\n"
        "- 每处引用在句末标 [^n]，n 为对应证据编号；\n"
        "- 不要输出证据列表本身，只输出词条正文。"
    )
    return system, user


async def generate_article(llm_settings, name: str, facts: list[dict], evidence: list[dict]) -> tuple[str, list[dict]]:
    """Call the configured LLM once (non-streaming); return (markdown, citations)."""
    from openai import AsyncOpenAI
    from nanoresearch.config.loader import env_key_or_raise

    system, user = build_article_prompt(name, facts, evidence)
    llm_cfg = getattr(llm_settings, "llm", None)
    client = AsyncOpenAI(
        base_url=getattr(llm_cfg, "base_url", None) or "https://api.openai.com/v1",
        api_key=getattr(llm_cfg, "api_key", None) or env_key_or_raise("OPENAI_API_KEY", role="ingestion_llm"),
    )
    model = getattr(llm_cfg, "model", None) or "gpt-4o-mini"
    resp = await client.chat.completions.create(
        model=model, temperature=0.3,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    markdown = (resp.choices[0].message.content or "").strip()
    return markdown, build_citations(evidence)
