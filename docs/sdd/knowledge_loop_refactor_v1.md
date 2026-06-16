# SDD: Knowledge Loop Refactoring

## 概述

关闭 Research Knowledge Loop 的历史遗留问题，做三件事：

1. **Auto-ingest**：研究报告完成后自动写入 RAG 知识库（IngestionPipeline），移除用户确认 ingest 的提示
2. **`_get_existing_knowledge` → HybridSearch**：Research Planner 规划前从 RAG 检索已有研究报告作为上下文，替代当前返回空字符串的桩代码
3. **清理死代码**：删除废弃的 claims/insights 三 tier 架构代码及相关类型、文件、测试

## 前置条件：MarkdownLoader 增加 YAML frontmatter 解析

**问题**：`_save_report_md()` 写入的 MD 文件包含 `source: research` 等 YAML frontmatter，但 `MarkdownLoader.load()` 不解析 frontmatter，导致 `filters={"source": "research"}` 静默失效。

**方案**：在 `MarkdownLoader.load()` 中增加 frontmatter 解析：
- 文件首行是 `---` 时尝试 `yaml.safe_load`
- 解析到的字段 merge 进 chunk metadata
- frontmatter 块从 `document.text` 中 strip 掉，避免影响 chunk 质量
- 无 frontmatter 或解析失败的文件行为不变（静默降级）
- 系统字段优先：先写入 frontmatter 字段，再让 MarkdownLoader 的系统字段（`source_path`、`doc_hash`、`file_name` 等）覆盖同名 key

---

## Task 1: Auto-ingest Research Reports

### 现状

- `ResearchRunner.run()` 末尾调用 `_save_report_md()` 将报告写到 `~/.nanoresearch/research_notes/{rid}_{slug}.md`
- `ResearchTool.execute(action="start")` 在返回消息中提示用户输入 "是" 或 "ingest" 来触发知识库写入
- 实际 ingest 依赖 LLM 调用 MCP `ingest_document` tool，不是自动的

### 改动

#### 1a. ResearchRunner 增加 auto-ingest

在 `ResearchRunner.__init__` 新增两个可选参数：

```python
class ResearchRunner:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        web_search_tool: Any,
        web_fetch_tool: Any,
        config: ResearchConfig | None = None,
        knowledge_search: KnowledgeSearch | None = None,
        rag_store: Any = None,
        *,
        settings: Settings | None = None,         # NEW: RAG settings
    ) -> None:
```

在 `run()` 方法的 Phase 7（`_save_report_md` 之后）新增：

```python
# Phase 7b: Auto-ingest into RAG (fire-and-forget)
if self.settings and result.report_file_path:
    asyncio.create_task(self._auto_ingest(result.report_file_path))
```

`_auto_ingest` 实现：

```python
async def _auto_ingest(self, file_path: str) -> None:
    """Background-ingest the research report MD into the RAG pipeline."""
    try:
        from nanobot.rag.ingestion.pipeline import IngestionPipeline
        loop = asyncio.get_running_loop()
        pipeline = IngestionPipeline(
            self.settings,
            collection=self.settings.vector_store.collection_name,
            force=False,
        )
        result = await loop.run_in_executor(
            None, lambda: pipeline.run(file_path)
        )
        if result.success:
            logger.info("Auto-ingest OK: {} -> {} chunks", file_path, result.chunk_count)
        else:
            logger.warning("Auto-ingest failed: {}", result.error)
    except Exception as e:
        logger.warning("Auto-ingest error: {}", e)
```

#### 1b. ResearchTool 移除 ingest 提示

`research.py:101-107` 删掉以下行：

```python
# 删除：
if result.report_file_path:
    summary.append(f"**报告文件**: `{result.report_file_path}`")
    summary.append(f"")
    summary.append(f"> 是否将此研究报告 ingest 进知识库？回复 **\"是** 或 **\"ingest\"** 即可写入。")
```

替换为只保留文件路径提示（告知用户已自动写入）：

```python
if result.report_file_path:
    summary.append(f"**报告文件**: `{result.report_file_path}`")
```

#### 1c. Agent Loop 传入 settings

在 `loop.py` 中 `ResearchTool(` 调用处传入新的 `settings` 参数：

```python
ResearchTool(
    provider=self.provider,
    model=self.model,
    web_search_tool=self.tools.get("web_search"),
    web_fetch_tool=self.tools.get("web_fetch"),
    config=self.research_config,
    knowledge_search=self.knowledge_search,
    rag_store=self.rag_store,
    settings=self.rag_settings,     # NEW
)
```

同时在 `subagent.py` 对应位置做同样改动。

#### 错误处理策略

| 场景 | 行为 |
|------|------|
| settings 未配置 | `_auto_ingest` 跳过，静默无操作 |
| pipeline.run 抛异常 | `try/except` 捕获，`logger.warning` 记录 |
| 文件已处理（SHA256 match） | IngestionPipeline 自身的 integrity 检查会 skip，返回 success=true |
| 主流程不需要等待 ingest | `asyncio.create_task` 保证不阻塞 report 返回 |

---

## Task 2: `_get_existing_knowledge` → HybridSearch

### 现状

```python
async def _get_existing_knowledge(self, topic: str, token_budget: int = 1500) -> str:
    """Pre-query existing knowledge. Claims/insights deprecated; always returns empty."""
    return ""
```

### 改动

#### 2a. HybridSearch 懒初始化

ResearchRunner 不直接持有 HybridSearch 实例，而是在 `_get_existing_knowledge` 首次调用时通过 `create_hybrid_search(settings)` 懒创建。这样避免在 Agent 启动阶段做重初始化，且 `settings` 已经在 ResearchRunner 中可用。

#### 2b. 替换 `_get_existing_knowledge` 实现

```python
@property
def _hybrid_search(self):
    """Lazy-initialized HybridSearch from settings."""
    if self._hs is None and self.settings is not None:
        from nanobot.rag.core.query_engine.hybrid_search import create_hybrid_search
        self._hs = create_hybrid_search(self.settings)
    return self._hs

async def _get_existing_knowledge(self, topic: str, token_budget: int = 1500) -> str:
    """Retrieve relevant existing research reports from RAG."""
    hs = self._hybrid_search
    if not hs:
        return ""

    try:
        results = await hs.async_search(
            query=topic,
            top_k=5,
            filters={"source": "research"},
        )
        if not results:
            return ""

        # Group by document for readable formatting
        from collections import OrderedDict
        doc_groups: dict[str, list[tuple[float, str]]] = OrderedDict()
        for r in results:
            src = r.metadata.get("source_path", "unknown")
            doc_groups.setdefault(src, []).append((r.score, r.text))

        parts = []
        token_used = 0
        for src, snippets in doc_groups.items():
            head = f"### 来自 {Path(src).name}\n"
            body_parts: list[str] = []
            for score, text in snippets:
                line = f"[relevance={score:.2f}] {text.strip()}"
                # Rough token estimate: ~4 chars per token for CJK text
                estimated = len(line) // 4
                if token_used + estimated > token_budget:
                    break
                body_parts.append(line)
                token_used += estimated

            if body_parts:
                parts.append(head + "\n" + "\n".join(body_parts) + "\n")

        return "\n".join(parts) if parts else ""

    except Exception as e:
        logger.warning("_get_existing_knowledge failed: {}", e)
        return ""
```

关键设计决策：
- 使用 `filters={"source": "research"}` 只检索 auto-ingest 进来的研究报告（frontmatter 中 `source: research` 会被存为 chunk metadata）
- `top_k=5` 控制检索范围，`token_budget=1500`（约 6000 chars CJK）控制上下文窗口用量
- 按 source_path 分组显示，便于 Planner 理解信息来自哪份报告
- 异常静默降级返回 `""`，不影响主流程

#### 2c. 移除 `_get_document_context` 中的 `_embed` 依赖

当前 `_get_document_context` 使用了 `self.knowledge_search._embed(topic)`（私有方法）。既然我们要用 HybridSearch 替代 knowledge_search 的检索职责，需要把 `_get_document_context` 也改造为使用 hybrid_search。

但 Task 2 只覆盖 research report 检索，`_get_document_context`（用户上传文档的检索）暂时保持不动，后续可以统一迁移。

#### 错误处理策略

| 场景 | 行为 |
|------|------|
| settings 为 None / hybrid_search 创建失败 | 返回 `""` |
| async_search 抛异常 | `try/except` 捕获，`logger.warning` + 返回 `""` |
| 检索结果为空 | 返回 `""` |
| 超过 token_budget | 截断，只返回 budget 内的内容 |
| source:research 过滤无匹配 | 正常返回 `""` |

---

## Task 3: 清理死代码

### 删除的文件

| 文件 | 原因 |
|------|------|
| `research/knowledge_processor.py` | Knowledge Loop 三 tier 架构的核心编排器，所有方法依赖已删除的类型 |
| `research/insight_tracker.py` | JSON 文件持久化的 insight 候选队列，未被任何活跃代码调用 |
| `research/correction_tracker.py` | 事实修正追踪器，从未真正集成过 |
| `tests/research/test_knowledge_loop.py` | 测试上述三个模块的集成/单元测试 |

### 从 `research/types.py` 删除的数据类

| 数据类 | 行号范围 | 原因 |
|--------|----------|------|
| `Claim` | ~317 | claim/insight 三 tier 模型已废弃 |
| `Insight` | ~340 | 同上 |
| `KnowledgeProcessResult` | ~363 | knowledge_processor 的输出类型 |
| `BatchRefineResult` | ~372 | batch refine 的输出类型 |
| `Correction` | ~379 | correction_tracker 的类型 |

需要同时删除这些类的 `to_dict()/from_dict()` 方法（如果定义了）。

### `research/__init__.py` 的改动

删除以下 imports（第 55-58 行）：

```python
# 删除
from nanobot.research.knowledge_processor import KnowledgeProcessor
from nanobot.research.insight_tracker import InsightTracker
```

从 `__all__` 删除以下条目：

```python
# 从 __all__ 删除
"Claim",
"Insight", 
"KnowledgeProcessResult",
"BatchRefineResult",
"KnowledgeProcessor",
"InsightTracker",
# 但不删除 "KnowledgeSearch"（仍在被使用）
```

### Module docstring 清理

`research/__init__.py` 中的 Knowledge Loop 使用示例（第 18-32 行）需要更新或移除，因为 `KnowledgeProcessor`、`InsightTracker` 不再可用。

### 确认无其他引用

需要 grep 确认以下无引用后删除：

```bash
# 确认这些模块/类型没有被 import
grep -r "knowledge_processor" backend/ --include="*.py"
grep -r "insight_tracker" backend/ --include="*.py"
grep -r "correction_tracker" backend/ --include="*.py"
grep -r "from nanobot.research.types import.*Claim" backend/ --include="*.py"
grep -r "from nanobot.research.types import.*Insight" backend/ --include="*.py"
```

### 删除顺序

1. 先改 `research/types.py`（删除 5 个数据类）
2. 再改 `research/__init__.py`（删除 import 和 __all__）
3. 删除 3 个源文件 + 1 个测试文件
4. 删除对应的 `__pycache__` 目录（可选）
5. 运行测试确保没有 import 错误

---

## 全局改动清单

### `research/runner.py`

| 行号 | 改动 |
|------|------|
| `__init__` 签名 | 新增 `settings=None`，移除 `ingest_collection`（从 settings 读）|
| `__init__` 体 | 存储新参数到 `self` |
| `run()` Phase 7 后 | 新增 `asyncio.create_task(self._auto_ingest(...))` |
| 新增方法 | `_auto_ingest(file_path)` |
| `_get_existing_knowledge` | 替换实现为 HybridSearch 检索 |

### `agent/tools/research.py`

| 行号 | 改动 |
|------|------|
| `__init__` 签名 | 透传 `settings` 到 `ResearchRunner` |
| `execute()` ~L101-107 | 删除 ingest 提示，保留文件路径 |

### `agent/loop.py` + `agent/subagent.py`

| 行号 | 改动 |
|------|------|
| `ResearchTool(` 调用处 | 传入 `settings=self.rag_settings` 等新参数 |

### `research/__init__.py`

| 行号 | 改动 |
|------|------|
| L55-58 | 删除 2 个 import |
| L60-83 `__all__` | 删除 6 个条目 |
| L18-32 docstring | 删除 Knowledge Loop 使用示例 |

### `research/types.py`

| 行号 | 改动 |
|------|------|
| ~L317-407 | 删除 5 个数据类 |

### 删除的文件

```
backend/nanobot/research/knowledge_processor.py
backend/nanobot/research/insight_tracker.py
backend/nanobot/research/correction_tracker.py
tests/research/test_knowledge_loop.py
```

---

## 回滚计划

每项改动独立可逆：

1. **Auto-ingest**：注释掉 `asyncio.create_task` 一行即可禁用，IngestionPipeline 自身的 integrity 检查保证重新启用时不会重复处理同一文件
2. **`_get_existing_knowledge`**：将实现替换回 `return ""` 即可恢复原状
3. **死代码清理**：从 git 恢复被删除的文件即可

---

## 附件：数据流图

```
ResearchRunner.run()
  │
  ├─ Phase 0: _get_existing_knowledge(topic)
  │    └─ HybridSearch.async_search(filters={"source": "research"})
  │         └─ create_hybrid_search(settings) [懒初始化]
  │         └─ 返回已有报告摘要 → 注入 Planner prompt
  │
  ├─ Phase 1-6: 研究流水线 (不变)
  │
  └─ Phase 7: _save_report_md()
       └─ Phase 7b: _auto_ingest()  [fire-and-forget]
            └─ IngestionPipeline.run(file_path)
                 └─ chunk 写入 ChromaDB + BM25
                      └─ metadata.source = "research" (来自 frontmatter)
                           └─ 后续 Research 可通过 filter 检索到
```
