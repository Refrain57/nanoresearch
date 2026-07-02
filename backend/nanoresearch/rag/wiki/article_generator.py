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


async def _complete(llm_settings, system: str, user: str) -> str:
    """Single non-streaming LLM completion (shared by all wiki generators)."""
    from openai import AsyncOpenAI
    from nanoresearch.config.loader import env_key_or_raise
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
    return (resp.choices[0].message.content or "").strip()


async def generate_article(llm_settings, name: str, facts: list[dict], evidence: list[dict]) -> tuple[str, list[dict]]:
    """Call the configured LLM once (non-streaming); return (markdown, citations)."""
    system, user = build_article_prompt(name, facts, evidence)
    markdown = await _complete(llm_settings, system, user)
    return markdown, build_citations(evidence)


def build_concept_prompt(topic: str, evidence: list[dict]) -> tuple[str, str]:
    system = "你是知识库词条编写助手。只依据给定的检索证据编写，不使用外部知识，不编造。"
    ev_lines = "\n".join(
        f"[{i}] {e.get('content','')}" for i, e in enumerate(evidence, start=1)
    ) or "（无证据）"
    user = (
        f"主题：{topic}\n\n"
        f"检索到的证据（编号）：\n{ev_lines}\n\n"
        "请围绕该主题写一段简洁的中文词条正文（markdown）。要求：\n"
        "- 只综合上述检索证据，不确定或无证据支撑的不要写；\n"
        "- 每处引用在句末标 [^n]，n 为对应证据编号；\n"
        "- 不要输出证据列表本身，只输出词条正文。"
    )
    return system, user


def build_overview_prompt(top_entities: list[dict], facts: list[dict]) -> tuple[str, str]:
    system = "你是知识库导览编写助手。只依据给定的实体与关系结构编写，不编造库中没有的内容。"
    ent_lines = "\n".join(
        f"- {e.get('name')}（被提及 {e.get('mentions', 0)} 次）" for e in top_entities
    ) or "（无实体）"
    rel_lines = "\n".join(
        f"- {f.get('source')} —{f.get('label')}→ {f.get('target')}" for f in facts
    ) or "（无关系）"
    user = (
        f"本知识库的主要实体：\n{ent_lines}\n\n"
        f"实体间关系：\n{rel_lines}\n\n"
        "请写一段中文总览/导览（markdown）：介绍本库有哪些主要主题、它们之间怎么关联。要求：\n"
        "- 只依据上面列出的实体与关系，不要编造未列出的内容；\n"
        "- 面向初次了解本库的读者，结构清晰。"
    )
    return system, user


def overview_signature(top_entities: list[dict], facts: list[dict]) -> str:
    ents = sorted(str(e.get("name", "")) for e in top_entities)
    rels = sorted(f"{f.get('source')}|{f.get('label')}|{f.get('target')}" for f in facts)
    return hashlib.sha256(("E:" + ",".join(ents) + ";R:" + ",".join(rels)).encode()).hexdigest()


async def generate_concept_article(llm_settings, topic: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    system, user = build_concept_prompt(topic, evidence)
    markdown = await _complete(llm_settings, system, user)
    return markdown, build_citations(evidence)


async def generate_overview_article(llm_settings, top_entities: list[dict], facts: list[dict]) -> tuple[str, list[dict]]:
    system, user = build_overview_prompt(top_entities, facts)
    markdown = await _complete(llm_settings, system, user)
    return markdown, []   # 总览引用实体/关系层，不逐句 [^n]；citations 留空
