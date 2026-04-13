# RAG 结构感知改进进度报告

**日期**: 2026-04-08
**状态**: 开发完成，待测试验证

---

## 1. 已完成的功能

### 1.1 结构感知切分 (DocumentStructureChunker)

| 特性 | 文件 | 状态 |
|------|------|------|
| 按 Markdown 标题层级切分 | `nanobot/rag/ingestion/chunking/document_chunker.py` | ✅ 完成 |
| 代码块/表格/列表保护 | 同上 | ✅ 完成 |
| 结构元数据 (section_level, section_path, content_type) | 同上 | ✅ 完成 |
| 相邻链 (prev/next_chunk_id) | 同上 | ✅ 完成 |
| 单元测试 | `tests/rag/unit/test_document_chunker.py` | ✅ 38 通过 |

### 1.2 PDF 书签提取

| 特性 | 文件 | 状态 |
|------|------|------|
| PyMuPDF 书签提取 | `nanobot/rag/libs/loader/pdf_loader.py` | ✅ 完成 |
| 书签转 Markdown 标题 | 同上 | ✅ 完成 |

### 1.3 MCP 工具扩展

| 工具 | 功能 | 状态 |
|------|------|------|
| `fetch_section` | 按章节路径获取内容 | ✅ 完成 |
| `fetch_neighbors` | 获取相邻 chunk 上下文 | ✅ 完成 |
| `plan_query` | 结构感知检索规划 | ✅ 完成 |

**MCP 工具总数**: 19 个

### 1.4 Markdown 支持

| 特性 | 文件 | 状态 |
|------|------|------|
| MarkdownLoader | `nanobot/rag/libs/loader/markdown_loader.py` | ✅ 完成 |
| Pipeline 多格式支持 | `nanobot/rag/ingestion/pipeline.py` | ✅ 完成 |
| ingest_document 支持 .md | `nanobot/rag/mcp_server/tools/agentic/collections.py` | ✅ 完成 |

### 1.5 Bug 修复

| 问题 | 修复 | 状态 |
|------|------|------|
| Runtime Context 标签移除 | `nanobot/agent/context.py` | ✅ 完成 |
| loop.py 引用已删除属性 | `nanobot/agent/loop.py` | ✅ 完成 |
| ChromaDB 客户端单例 | `nanobot/rag/libs/vector_store/chroma_store.py` | ✅ 完成 |
| list_documents 异步 bug | `nanobot/rag/mcp_server/tools/agentic/collections.py` | ✅ 完成 |
| fetch_section 中文过滤问题 | `nanobot/rag/mcp_server/tools/agentic/retrieval.py` | ✅ 完成 |
| fetch_section schema 定义顺序 | 移动 schema 到工具类之前 | ✅ 完成 |

### 1.6 已知行为 (非 Bug)

| 行为 | 说明 | 状态 |
|------|------|------|
| ### 标题不生成独立 path | RAG 只为 #### 子节创建 path | ✅ 预期行为 |
| 路径格式示例 | `/NanoResearcher 项目简历/各模块详解/1. 三层记忆系统/设计初衷` | ✅ 正确 |

### 1.7 待修复 Bug

| 问题 | 说明 | 状态 |
|------|------|------|
| fetch_neighbors 邻居解析 | 返回中心 chunk 但未正确获取 prev/next 邻居 | ❌ 待修复 |

---

## 2. 配置

```yaml
# config/rag/settings.yaml
ingestion:
  chunk_strategy: "document_based"  # 启用结构感知切分
  min_chunk_length: 100
  max_chunk_length: 2000

vector_store:
  persist_directory: "./nanobot/workspace/rag_data/chroma"
  collection_name: "default"
```

---

## 3. 测试数据

- **RESUME_DRAFT.md** 已摄入到 `default` 集合
- **Chunk 总数**: 217 个 (173 旧 + 44 新)
- **结构元数据**: section_level, section_path, title, content_type, prev/next_chunk_id

---

## 4. 实测验证结果 (2026-04-08)

### 4.1 RESUME_DRAFT.md 摄入验证

- ✅ 成功摄入 default 集合
- ✅ 217 个 chunks (173 旧 + 44 新结构化)
- ✅ section_path 元数据正确生成

### 4.2 fetch_section 实测

| 测试用例 | 结果 | 说明 |
|---------|------|------|
| `"/NanoResearcher 项目简历/各模块详解"` | ✅ 成功 | 返回 8 个 chunks |
| `"/NanoResearcher 项目简历/各模块详解/1. 三层记忆系统"` | ❌ 失败 | ### 标题不生成独立 path |
| `"/NanoResearcher 项目简历/各模块详解/1. 三层记忆系统/层级关系"` | ✅ 成功 | #### 子节路径正确 |

**结论**: 使用 #### 子节路径 (如 `/.../层级关系`) 而非 ### 标题路径

### 4.3 fetch_neighbors 实测

| 测试用例 | 结果 | 说明 |
|---------|------|------|
| `chunk_id=b7a1a4f2_0002_e4f806fd` | ❌ 部分失败 | 返回中心 chunk，但邻居未正确获取 |

**问题**: 虽然 prev/next_chunk_id 元数据存在，但工具未正确解析和返回邻居

### 4.4 retrieve_hybrid 实测

| 测试用例 | 结果 |
|---------|------|
| `query="三层记忆系统"` | ✅ 成功返回相关 chunks |

---

## 5. 已知问题

1. **fetch_neighbors 邻居解析 bug**: prev/next_chunk_id 元数据存在但工具未正确获取邻居
2. **旧 chunks 没有结构元数据**: default 集合前 173 个 chunk 是旧的，缺少 section_level 等元数据 (非阻塞)

---

## 6. 待验证功能

- [x] `fetch_section` 按章节路径检索 — ✅ 已验证 (使用 #### 子节路径)
- [ ] `fetch_neighbors` 获取相邻上下文 — ❌ 邻居解析 bug
- [ ] `plan_query` 结构感知规划 — ⏳ 待测试
- [ ] PDF 书签提取 + 结构切分 — ⏳ 待测试

---

## 7. 下一步

1. [ ] 修复 fetch_neighbors 邻居解析 bug
2. [ ] 测试 plan_query 工具
3. [ ] 测试 PDF 书签提取
4. [ ] 清理无用的 test collection
5. [ ] 更新 SKILL.md 文档
