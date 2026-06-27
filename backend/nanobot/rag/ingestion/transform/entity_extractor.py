"""Entity and relation extraction transform for Knowledge Graph construction."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from nanobot.rag.core.types import Chunk
from nanobot.rag.ingestion.transform.base_transform import BaseTransform
from nanobot.rag.libs.llm.base_llm import BaseLLM, Message
from nanobot.rag.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

_PROMPT = """请从下面文本中抽取实体和实体关系，返回严格 JSON，不要输出任何解释或 markdown。

JSON 格式：
{
  "nodes": [
    {"text": "实体文本", "label": "实体类型"}
  ],
  "relations": [
    {
      "source": {"text": "实体文本", "label": "实体类型"},
      "target": {"text": "实体文本", "label": "实体类型"},
      "text": "关系描述",
      "label": "关系类型"
    }
  ]
}

规则：
- nodes 列出文本中所有值得追踪的实体，包括没有显式关系的孤立实体
- relations 列出实体间有明确关系的三元组，可以为空数组
- 实体类型示例：Method, Model, Dataset, Concept, Organization, Person, Term
- 只抽取具体、有意义的实体，不要抽取"研究"、"方法"等过于宽泛的词
- 实体文本使用原文（不翻译），保持大小写

文本：
{chunk_text}"""


class EntityExtractor(BaseTransform):
    """Extract entities and relations from chunks for Knowledge Graph construction.

    Stores results in chunk.metadata["_kg_entities"] and ["_kg_relations"]
    (underscore-prefixed = temporary, will be consumed by persist_chunk_entities
    and removed before ChromaDB storage).
    """

    def __init__(self, settings=None, llm: Optional[BaseLLM] = None) -> None:
        self.settings = settings
        self._llm = llm
        self.use_llm = True
        self._max_workers = 4
        self._text_limit = 2000

    @property
    def llm(self) -> Optional[BaseLLM]:
        if self._llm is None and self.use_llm:
            try:
                self._llm = LLMFactory.create(self.settings)
            except Exception as e:
                logger.warning(f"[EntityExtractor] Failed to init LLM: {e}. KG extraction disabled.")
                self.use_llm = False
        return self._llm

    def transform(self, chunks: List[Chunk], trace=None) -> List[Chunk]:
        if not chunks or not self.use_llm:
            return chunks
        if not self.llm:
            return chunks

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._extract_single, c): c for c in chunks}
            results = {}
            for fut in as_completed(futures):
                chunk = futures[fut]
                try:
                    results[chunk.id] = fut.result()
                except Exception as e:
                    logger.warning(f"[EntityExtractor] chunk {chunk.id} failed: {e}")
                    results[chunk.id] = chunk

        return [results.get(c.id, c) for c in chunks]

    def _extract_single(self, chunk: Chunk) -> Chunk:
        try:
            prompt = _PROMPT.replace("{chunk_text}", chunk.text[: self._text_limit])
            response = self.llm.chat([Message(role="user", content=prompt)])

            text = response
            if hasattr(response, "content"):
                text = response.content
            elif hasattr(response, "text"):
                text = response.text
            elif not isinstance(response, str):
                text = str(response)

            parsed = _parse_response(text)
            if parsed is None:
                return chunk

            new_metadata = {
                **(chunk.metadata or {}),
                "_kg_entities": parsed.get("nodes", []),
                "_kg_relations": parsed.get("relations", []),
            }
            return Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=new_metadata,
                source_ref=chunk.source_ref,
            )
        except Exception as e:
            logger.warning(f"[EntityExtractor] extraction failed for chunk {chunk.id}: {e}")
            return chunk


def _parse_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, tolerating markdown fences."""
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in response
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    logger.debug(f"[EntityExtractor] Could not parse JSON from: {text[:200]}")
    return None
