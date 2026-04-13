# Agent Evaluation Framework

评估框架用于测试和验证 nanobot Agent 各个模块的质量。

## 快速开始

```bash
# 运行所有基准测试
python -m tests.agent.evaluation.benchmark

# 运行单元测试
python -m pytest tests/agent/evaluation/ -v
```

## 测试模块

### 1. Research Agent (`test_research_agent.py`)

测试研究 Agent 的完整流程：
- 主题分解质量（子问题数量、覆盖度）
- 搜索编排（并行搜索、去重）
- 信息综合（结构化输出）
- 报告质量评估（LLM-as-Judge）

### 2. RAG 检索 (`test_rag_eval.py`)

基于 Ground Truth 评估检索质量：
- **Precision**：检索结果中相关文档的比例
- **Recall**：相关文档被检索到的比例
- **MRR**：首个相关结果的倒数排名
- **NDCG**：归一化折扣累积增益

### 3. 记忆提炼 (`test_memory_eval.py`)

评估记忆系统的质量：
- 事实提取准确性
- 记忆完整性
- 对话上下文保留
- 降级策略（LLM 失败时的容错）

### 4. 工具调用 (`test_tool_call.py`)

评估 Agent 工具选择能力：
- 工具匹配率
- 参数正确性
- 多工具协调

## 评估指标

| 指标 | 说明 | 阈值 |
|------|------|------|
| NDCG | 检索排序质量 | >= 0.6 |
| MRR | 首相关结果排名 | >= 0.5 |
| Tool Match Rate | 工具选择正确率 | >= 0.3 |
| Memory Quality | 记忆质量评分 | >= 6.0/10 |

## 添加新测试

### 添加 Research 测试用例

```python
# tests/agent/evaluation/test_datasets.py
RESEARCH_TEST_CASES.append(ResearchTestCase(
    topic="Your research topic",
    expected_sub_questions=["Q1", "Q2"],
    min_sub_questions=2,
    max_sub_questions=4,
    category="technical",
))
```

### 添加 RAG 测试用例

```python
# tests/agent/evaluation/test_datasets.py
RAG_TEST_CASES.append(RetrievalTestCase(
    query="Your search query",
    relevant_chunks=["chunk_id_1", "chunk_id_2"],
    relevant_sources=["source1.com"],
    category="factual",
))
```

### 添加记忆测试用例

```python
# tests/agent/evaluation/test_datasets.py
MEMORY_TEST_CASES.append(MemoryTestCase(
    conversation_id="test_id",
    conversation=[
        {"role": "user", "content": "I'm working on..."},
    ],
    expected_facts=["fact1", "fact2"],
    expected_events=["event1"],
    not_expected=["secret"],
))
```

## 架构

```
tests/agent/evaluation/
├── evaluator.py         # 核心评估类和接口
├── test_datasets.py    # 测试数据集
├── test_research_agent.py  # Research Agent 测试
├── test_rag_eval.py    # RAG 检索测试
├── test_memory_eval.py # 记忆系统测试
├── test_tool_call.py   # 工具调用测试
├── benchmark.py        # 基准测试运行器
└── run_evaluation.py   # 完整评估报告生成器
```

## 输出格式

```bash
# 运行基准测试
python -m tests.agent.evaluation.benchmark

# 输出示例：
# ============================================================
#  NANOBOT AGENT EVALUATION BENCHMARK
#  Started: 2026-04-06 10:29:43
# ============================================================
#
#  Research Agent Benchmark
#    [PASSED] 3D Gaussian Splatting... (score: 50.0%)
#    [PASSED] 2024年模型发展趋势... (score: 100.0%)
#
#  RAG Retrieval Benchmark
#    [PASSED] NDCG: 74.7%
#
#  Memory Consolidation Benchmark
#    [PASSED] 2/2 passed (100%)
#
#  Tool Call Benchmark
#    [PASSED] Tool Match Rate: 57.1%
```

## 后续改进方向

1. **引入真实 LLM 评估**：使用实际的 LLM-as-Judge 进行端到端评估
2. **自动化测试**：集成 CI/CD，定期运行评估并记录趋势
3. **可视化报告**：生成 HTML 报告展示指标趋势
4. **A/B 测试**：支持不同配置/策略的效果对比
