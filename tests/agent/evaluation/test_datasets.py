"""Test datasets for agent evaluation.

This module provides ground truth data for testing:
- Research Agent evaluation
- RAG retrieval evaluation
- Memory consolidation evaluation
- Tool call evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.agent.evaluation.evaluator import (
    RetrievalTestCase,
    ToolCallTestCase,
)


# ============================================================================
# Research Agent Test Cases
# ============================================================================

@dataclass
class ResearchTestCase:
    """Test case for research agent evaluation."""
    topic: str
    expected_sub_questions: list[str]
    min_sub_questions: int = 3
    max_sub_questions: int = 6
    expected_sources_count: int = 5  # Minimum sources expected
    category: str = ""  # technical, factual, comparison, news
    depth: str = "normal"


RESEARCH_TEST_CASES: list[ResearchTestCase] = [
    ResearchTestCase(
        topic="3D Gaussian Splatting 技术原理与应用",
        expected_sub_questions=[
            "3D Gaussian Splatting 的核心原理",
            "与 NeRF 的对比",
            "主要应用场景",
            "性能与效果",
            "未来发展方向",
        ],
        min_sub_questions=3,
        max_sub_questions=6,
        category="technical",
    ),
    ResearchTestCase(
        topic="RAG 系统的最佳实践",
        expected_sub_questions=[
            "RAG 架构设计",
            "检索策略选择",
            "向量数据库对比",
            "生成质量控制",
        ],
        min_sub_questions=3,
        max_sub_questions=5,
        category="technical",
    ),
    ResearchTestCase(
        topic="2024年大模型发展趋势",
        expected_sub_questions=[
            "主流模型对比",
            "技术突破",
            "应用落地",
            "商业化进展",
        ],
        min_sub_questions=3,
        max_sub_questions=6,
        category="news",
    ),
]


# ============================================================================
# RAG Retrieval Test Cases (Ground Truth)
# ============================================================================

RAG_TEST_CASES: list[RetrievalTestCase] = [
    # Technical queries
    RetrievalTestCase(
        query="如何配置 Azure OpenAI 的 API 密钥",
        relevant_chunks=["azure_openai_config", "api_key_setup", "azure_credentials"],
        relevant_sources=["docs.azure.com", "learn.microsoft.com"],
        category="technical",
    ),
    RetrievalTestCase(
        query="向量数据库 ChromaDB 和 Milvus 的区别",
        relevant_chunks=["chromadb_intro", "milvus_intro", "vector_db_comparison"],
        relevant_sources=["docs.trychroma.com", "milvus.io"],
        category="comparison",
    ),
    RetrievalTestCase(
        query="RAG 系统中如何实现混合检索",
        relevant_chunks=["hybrid_search", "bm25_dense_fusion", "rrf_algorithm"],
        relevant_sources=["arxiv.org", "blog.langchain.dev"],
        category="technical",
    ),

    # Factual queries
    RetrievalTestCase(
        query="GPT-4o 的上下文窗口大小是多少",
        relevant_chunks=["gpt4o_specs", "context_window_limits", "model_comparison"],
        relevant_sources=["platform.openai.com", "help.openai.com"],
        category="factual",
    ),
    RetrievalTestCase(
        query="Python 的 GIL 是什么",
        relevant_chunks=["python_gil", "threading_gil", "concurrency_python"],
        relevant_sources=["docs.python.org", "realpython.com"],
        category="factual",
    ),

    # Comparison queries
    RetrievalTestCase(
        query="Anthropic Claude 和 OpenAI GPT 的优缺点对比",
        relevant_chunks=["claude_features", "gpt_features", "llm_comparison"],
        relevant_sources=["anthropic.com", "openai.com"],
        category="comparison",
    ),
]


# ============================================================================
# Memory Consolidation Test Cases
# ============================================================================

@dataclass
class MemoryTestCase:
    """Test case for memory consolidation evaluation."""
    conversation_id: str
    conversation: list[dict[str, str]]
    expected_facts: list[str]  # Key facts that should be extracted
    expected_events: list[str]  # Key events that should be logged
    not_expected: list[str]  # Things that should NOT appear in memory


MEMORY_TEST_CASES: list[MemoryTestCase] = [
    MemoryTestCase(
        conversation_id="user_preferences",
        conversation=[
            {"role": "user", "content": "你好，我叫小明"},
            {"role": "assistant", "content": "你好小明，很高兴认识你！有什么我可以帮你的吗？"},
            {"role": "user", "content": "我是一名软件工程师，主要用 Python 开发"},
            {"role": "assistant", "content": "Python 是一门很棒的语言！你主要开发什么类型的应用？"},
            {"role": "user", "content": "主要是后端服务和一些 AI 相关的项目"},
        ],
        expected_facts=[
            "用户名叫小明",
            "用户是软件工程师",
            "用户使用 Python 开发",
            "用户关注 AI 项目",
        ],
        expected_events=[
            "用户介绍了自己的身份和职业",
        ],
        not_expected=[
            "用户的密码",
            "用户的地址",
            "敏感个人信息",
        ],
    ),
    MemoryTestCase(
        conversation_id="task_context",
        conversation=[
            {"role": "user", "content": "帮我研究一下 RAG 技术"},
            {"role": "assistant", "content": "好的，我来帮你研究 RAG 技术..."},
            {"role": "user", "content": "我特别关心检索精度的问题"},
            {"role": "assistant", "content": "检索精度确实是 RAG 的核心问题之一..."},
            {"role": "user", "content": "下周三之前给我一个完整的报告"},
        ],
        expected_facts=[
            "用户关注 RAG 技术",
            "用户关心检索精度",
            "用户需要完整报告",
        ],
        expected_events=[
            "用户请求研究 RAG 技术",
            "用户要求下周三前完成报告",
        ],
        not_expected=[
            "具体的实现细节",  # 这些应该在对话历史中，不是长期记忆
        ],
    ),
]


# ============================================================================
# Tool Call Test Cases
# ============================================================================

TOOL_CALL_TEST_CASES: list[ToolCallTestCase] = [
    ToolCallTestCase(
        user_intent="读取文件内容",
        conversation_history=[
            {"role": "user", "content": "帮我看看 README.md 的内容"},
        ],
        expected_tools=["read_file"],
        expected_params={"path": "README.md"},
    ),
    ToolCallTestCase(
        user_intent="网络搜索",
        conversation_history=[
            {"role": "user", "content": "搜索一下最新的 AI 新闻"},
        ],
        expected_tools=["web_search"],
        expected_params={"query": "AI 新闻"},
    ),
    ToolCallTestCase(
        user_intent="写入文件",
        conversation_history=[
            {"role": "user", "content": "把这段代码保存到 test.py 文件中"},
        ],
        expected_tools=["write_file"],
        expected_params={"path": "test.py"},
    ),
    ToolCallTestCase(
        user_intent="执行命令",
        conversation_history=[
            {"role": "user", "content": "运行 pytest 测试"},
        ],
        expected_tools=["exec"],
        expected_params={"command": "pytest"},
    ),
    ToolCallTestCase(
        user_intent="研究任务",
        conversation_history=[
            {"role": "user", "content": "帮我研究一下大模型微调技术"},
        ],
        expected_tools=["research"],
        expected_params={"topic": "大模型微调"},
    ),
    ToolCallTestCase(
        user_intent="多工具调用",
        conversation_history=[
            {"role": "user", "content": "搜索 RAG 技术然后把结果保存到文件"},
        ],
        expected_tools=["web_search", "write_file"],
        expected_params=None,
    ),
]


# ============================================================================
# Benchmark Results (for comparison)
# ============================================================================

BENCHMARK_RESULTS: dict[str, dict[str, Any]] = {
    "research_agent": {
        "completeness_avg": 7.2,
        "accuracy_avg": 8.1,
        "readability_avg": 7.8,
        "depth_avg": 6.9,
        "overall_avg": 7.5,
    },
    "rag_retrieval": {
        "precision": 0.85,
        "recall": 0.72,
        "f1": 0.78,
        "mrr": 0.68,
        "ndcg": 0.75,
    },
    "memory_consolidation": {
        "accuracy_avg": 8.5,
        "completeness_avg": 7.8,
        "utility_avg": 8.2,
        "conciseness_avg": 7.5,
    },
    "tool_call": {
        "tool_match_rate": 0.82,
        "avg_param_score": 0.75,
    },
}