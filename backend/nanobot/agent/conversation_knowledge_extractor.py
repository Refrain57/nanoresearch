"""Conversation Knowledge Extractor — extracts user information from conversations.

This module extracts user-specific information from conversations:
- Preferences: what the user likes/dislikes/prioritizes
- Habits: user's work style, common tools, workflows
- Decisions: choices, commitments, decisions made

It does NOT extract general knowledge or factual statements.
Extracted memories are written to user_memory collection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.research.knowledge_search import KnowledgeSearch


_EXTRACT_USER_INFO_PROMPT = """从以下对话中提取用户个人信息和偏好。

## 对话内容
{conversation_text}

## 任务
只提取以下类型的用户信息：

1. **偏好 (preference)**：用户明确表达喜欢/不喜欢/倾向/优先级
   - 例："我喜欢用 VS Code" → 类型：偏好，内容：编辑器偏好 VS Code
   - 例："优先考虑性能" → 类型：偏好，内容：优先考虑性能
   - 例："不喜欢用 Java" → 类型：偏好，内容：不喜欢 Java

2. **习惯 (habit)**：用户的工作方式、常用工具、工作流程
   - 例："我通常在早上写代码" → 类型：习惯，内容：早上写代码
   - 例："我们用 Git 做版本控制" → 类型：习惯，内容：使用 Git 版本控制
   - 例："习惯用 TDD 方式开发" → 类型：习惯，内容：TDD 开发方式

3. **决策 (decision)**：用户做出的选择、决定、承诺
   - 例："我决定用 Python 重写" → 类型：决策，内容：选择 Python 重写
   - 例："下周一定完成" → 类型：决策，内容：承诺下周完成
   - 例："选方案 A" → 类型：决策，内容：选择方案 A

## 不要提取
- 一般性知识陈述（"Python 是解释型语言"）
- 客观事实（"今天是周一"）
- 技术概念解释
- 代码逻辑说明
- Agent 的回复内容

## 输出格式
请输出 JSON 格式：
```json
{{
  "user_info": [
    {{
      "content": "偏好：编辑器偏好 VS Code",
      "type": "preference",
      "confidence": 0.9
    }}
  ]
}}
```

如果没有值得提取的用户信息，返回 {"user_info": []}

只返回 JSON，不要其他内容。"""


@dataclass
class ExtractedUserInfo:
    """User information extracted from conversation."""
    content: str
    type: str  # preference | habit | decision
    confidence: float


class ConversationKnowledgeExtractor:
    """Extracts user information from agent conversations.

    Key features:
    - Only extracts user preferences, habits, and decisions
    - Filters out general knowledge and factual statements
    - Writes to user_memory collection
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        knowledge_search: KnowledgeSearch,
    ):
        self.provider = provider
        self.model = model
        self.knowledge_search = knowledge_search

    async def extract_from_messages(
        self,
        messages: list[dict[str, Any]],
        uid: str | None = None,
    ) -> int:
        """Extract user information from conversation messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            uid: User ID to associate with extracted memories.

        Returns:
            Number of user info items written to user_memory.
        """
        if not messages:
            return 0

        # Format conversation for extraction
        conv_text = self._format_conversation(messages)

        # Extract user info using LLM
        user_infos = await self._extract_user_info(conv_text)
        if not user_infos:
            logger.debug("ConversationKnowledgeExtractor: no user info extracted")
            return 0

        # Write to user_memory
        memories = [
            {
                "text": ui.content,
                "type": ui.type,
                "confidence": ui.confidence,
                "is_evergreen": True,  # User preferences are evergreen
                "created_at": datetime.now().isoformat(),
            }
            for ui in user_infos
        ]

        written, skipped = await self.knowledge_search.write_user_memory(memories, uid=uid)

        logger.info(
            f"ConversationKnowledgeExtractor: wrote {written} user info items, skipped {skipped} duplicates"
        )

        return written

    def _format_conversation(self, messages: list[dict[str, Any]]) -> str:
        """Format messages into conversation text."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def _extract_user_info(self, conv_text: str) -> list[ExtractedUserInfo]:
        """Extract user information from conversation text using LLM.

        Args:
            conv_text: Formatted conversation text.

        Returns:
            List of extracted user information.
        """
        try:
            response = await self.provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "You are a user information extractor. Return only JSON."},
                    {"role": "user", "content": _EXTRACT_USER_INFO_PROMPT.format(
                        conversation_text=conv_text[:4000],  # Limit length
                    )},
                ],
                model=self.model,
                max_tokens=500,
                temperature=0.3,
            )

            if not response.content:
                return []

            # Parse JSON response
            import re
            json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response.content)
            try:
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    data = json.loads(response.content.strip())
            except json.JSONDecodeError:
                logger.debug("User info extraction: LLM returned invalid JSON: {:.200}", response.content)
                return []

            if not isinstance(data, dict):
                logger.debug("User info extraction: expected dict, got {}", type(data).__name__)
                return []

            user_infos = []
            for item in data.get("user_info", []):
                if not isinstance(item, dict):
                    continue
                user_infos.append(ExtractedUserInfo(
                    content=item.get("content", ""),
                    type=item.get("type", "preference"),
                    confidence=item.get("confidence", 0.8),
                ))

            return user_infos

        except Exception as e:
            logger.warning(f"User info extraction failed: {e}")
            return []
