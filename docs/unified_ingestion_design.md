# 统一入库接口设计文档

**状态**：待审批（v3）
**日期**：2026-06-24

---

## 一、统一接口签名

```python
async def ingest_document(
    *,
    # ── 必填 ──
    kb_id: str,                    # KB UUID，不允许 None 或空字符串
    file_path: str,                # 绝对路径，已落到永久位置（见 4.5 节定义）
    original_filename: str,        # 展示用的原始文件名
    content_hash: str,             # SHA256 hex，调用方预先计算

    # ── 可选 ──
    pdf_parser: Literal["mineru", "marker", "markitdown"] = "mineru",
    chunk_strategy: str = "auto",
    force: bool = False,           # 必须显式 True 才能重处理，Agent 不暴露
    uid: str = "",
    metadata: dict[str, Any] | None = None,
) -> IngestResult:
```

约束：
- `kb_id` 必填，没有默认值
- `force` 默认 `False`，Agent 不可设置
- 不暴露 `collection` 参数给调用方
- `file_path` 必须是永久路径（接口内部校验，不满足就报错）

```python
@dataclass
class IngestResult:
    kb_id: str
    doc_id: str
    collection: str
    chunk_count: int
    status: Literal["created", "skipped_duplicate", "replaced"]
    duplicate_of: str | None
```

---

## 二、调用流程（含失败回滚）

```
ingest_document(kb_id, file_path, content_hash, ...)
  │
  ├─ Step 0：路径校验
  │     file_path 必须是绝对路径，不在系统临时目录下（/tmp, %TEMP%, %TMP%）
  │     不满足 → ValueError("file_path 必须是永久路径，当前值: {file_path}")
  │     （详见 4.5 节 —— 每个调用方如何保证传入永久路径）
  │
  ├─ Step A：kb_id → collection 解析
  │     SELECT chroma_collection FROM knowledge_bases WHERE id = kb_id
  │     NULL → RuntimeError（不兜底，详见 4.4 节）
  │     行不存在 → KbNotFoundError
  │
  ├─ Step B：去重判断
  │     SELECT id, status, content_hash FROM kb_documents
  │      WHERE kb_id = ? AND content_hash = ? AND status != 'failed'
  │
  │     命中 status='indexed', force=False → IngestResult(status="skipped_duplicate")
  │     命中 status='indexed', force=True  → 走替换分支
  │     命中 status='processing'           → 上一次崩溃残留，走替换分支（见表）
  │     未命中                              → 走新建分支
  │
  │     ┌───────────────┬──────────────────────────────────────────────┐
  │     │ 命中 status    │ 处理                                        │
  │     ├───────────────┼──────────────────────────────────────────────┤
  │     │ indexed       │ force=F→跳过; force=T→替换（先清 Chroma 旧 chunk） │
  │     │ processing    │ 上一次崩溃，先清 Chroma 该 source_path 残留， │
  │     │               │ 再走替换流程（保留 doc_id 不新建）               │
  │     │ failed        │ 不命中（WHERE 排除了），走新建                     │
  │     └───────────────┴──────────────────────────────────────────────┘
  │
  ├─ Step C：PG 记录
  │    新建 → INSERT (status='processing') → doc_id
  │    替换 → UPDATE status='reprocessing' → 保留原 doc_id
  │            → 先 ChromaDB.delete(where={"source_path": old_file_path})
  │
  ├─ Step D：IngestionPipeline
  │    pipeline = IngestionPipeline(settings, collection, force)
  │    result = pipeline.run(file_path)
  │    → result.chunk_payloads 包含 chroma_id, text, metadata 等（见 4.3）
  │
  │    ┌─ 成功 → 进 Step E
  │    └─ 失败 → 1. UPDATE kb_documents SET status='failed', error_msg=...
  │              2. ChromaDB.delete(where={"source_path": file_path})
  │              3. 抛出 IngestFailedError
  │
  ├─ Step E：批量写 KbChunk（用 result.chunk_payloads，不回读 ChromaDB）
  │    → INSERT KbChunk (...)
  │    → UPDATE kb_documents SET status='indexed', chunk_count=...
  │    → UPDATE knowledge_bases 计数
  │
  └─→ IngestResult
```

---

## 三、架构决定

### (a) MCP 写入 PG —— 采纳推荐，要同步写

### (b) kb_id 必填的三条边界

| 场景 | 策略 |
|---|---|
| **无 KB**（Agent 首次启动） | `worker.py` 构建 kb_map 时，检测绑空 → 调 API 创建 `name="个人知识库"` 的默认 KB → 写入 `agent_knowledge_bindings`。
| **单 KB** | `mcp.py` 自动注入 `kb_id = first(kb_map.keys())`，Agent 无需显式传 |
| **多 KB** | `mcp.py` 检测 `len(kb_map) > 1` 且 Agent 未传 `kb_id` → **报错**："ingest_document requires kb_id. Available: [list]"。不设主 KB、不选第一个、不兜底 |

### (c) 强制重处理入口 —— Web UI "重新处理" 按钮，Agent 不暴露 force

---

## 四、补的四个洞 + 一个追责

### 4.1 失败回滚

见流程 Step D 失败分支。关键设计：
- `status='failed'` 的 KbDocument 不参与去重拦截（Step B 的 WHERE 排除了它们）
- 用户重传同一文件 → content_hash 相同，但上次是 failed → 不命中 → 走新建
- `status='processing'` 残留：上次中途崩了，这次查到后走替换（清 Chroma + 重写），不新建 doc_id

### 4.2 UNIQUE(kb_id, content_hash) 上线时序

不能和接口代码一起上线。分三 Phase：

```
Phase 1：代码部署（统一接口 + source_path 修复）
         观察，确认无新重复
Phase 2：存量清洗脚本（content_hash 分组去重）
         洗完后同一 (kb_id, content_hash) 唯一
Phase 3：加约束 ALTER TABLE ... ADD CONSTRAINT uq_kb_content_hash
         UNIQUE (kb_id, content_hash)
```

Phase 2 必须先于 Phase 3，因为 `ReasonInChains.pdf` 已有重复，直接加约束会失败。

### 4.3 PG 和 ChromaDB 的一致性模型

**决定：Pipeline 输出为共同来源，PG 和 ChromaDB 同步写入。**

现状是 Step E 从 ChromaDB 回读 vector_ids → 逐条写 KbChunk。问题：一次额外的网络 IO；回读和写入之间无事务；和 (a) 里 PG 是数据的完整记录矛盾。

改为：**Pipeline 内部已有 chunk_id / text / metadata，直接带回来，不回读。**

`PipelineResult` 加一个字段：

```python
@dataclass
class ChunkPayload:
    chroma_id: str
    text: str
    token_count: int
    char_start: int | None
    char_end: int | None
    metadata: dict

# PipelineResult 加：
chunk_payloads: list[ChunkPayload]
```

`VectorUpserter.upsert()` 写 ChromaDB 的同时收集 payload，Pipeline 汇总返回。Step E 直接用这些数据批量 INSERT KbChunk。**PG 和 ChromaDB 是同一次 pipeline 产出的两份投影，无主从关系。**

改动量：PipelineResult（+1 字段）、VectorUpserter（收集 payload）、Pipeline.run（汇总返回）——约 30 行。

### 4.4 NULL collection 的兜底

chroma_collection 为 NULL → `RuntimeError`。不用 kb_id 兜底，不用默认值。NULL 意味着 KB 创建漏了写 chroma_collection（旧代码或迁移问题），应修复 DB 而非运行时掩盖。

### 4.5 file_path 追责：谁传 temp，谁保证它是永久的

追完全部三条通道后的事实：

| 通道 | file_path 来源 | 是否 temp | 是否稳定 |
|---|---|---|---|
| Web UI → worker | `knowledge_router.py:154` `tempfile.NamedTemporaryFile` → `tmp_path` → 传给 ARQ job → `worker.py:514` 原样传给 pipeline | **是，唯一来源** | 否 |
| Agent → MCP | `paper_fetch` 下载到 `workspace/papers/xxx.pdf` → Agent 调 `ingest_document(file_path="papers/xxx.pdf")` → `collections.py:800` 解析为 `~/.nanoresearch/workspace/papers/xxx.pdf` | 否 | 是 |
| 手动 → MCP | 调用者传绝对路径 | 取决于调用者 | 取决于调用者 |

**temp 路径只有一个来源：worker.py line 514。Agent 没有这个习惯。** `paper_fetch` 下载到 `workspace/papers/`，那是用户 home 下的永久目录，不是 `/tmp`。

修复分两层：

**第一层（治本）：每个调用方保证传入永久路径**

- **Worker**：文件已在 line 501 复制到 `perm_path = rag/documents/{kb_id}/{doc_id}.pdf`。改为调 `unified.ingest_document(file_path=perm_path, ...)`，**不再传 tmp_path**。line 514 的参数从 `file_path`（temp）改成 `perm_path`。
- **MCP**：在 `collections.py` 的 `execute` 里，路由到统一接口前加一步：如果路径在系统临时目录下，先复制到 `rag/documents/{kb_id}/{filename}` 再传入。

**第二层（兜底）：统一接口校验**

`ingest_document` 入口处（Step 0）校验 `file_path`：
- 必须是绝对路径
- 不在 `tempfile.gettempdir()` 下
- 不匹配 tmp/temp 命名模式（额外防御）
- 不满足 → `ValueError("file_path 必须是永久路径: {file_path}。请先将文件复制到永久目录再调用。")`

这样即便将来有新的调用方加进来、或者 Agent 将来某天拿到了 tmp 路径，接口层面也会直接拒绝，不会静默写入。

这两种情况不视为 temp 路径：
- `workspace/papers/` 下（有 `~/.nanoresearch` 前缀的绝对路径）
- `rag/documents/{kb_id}/` 下（KB 永久存储目录）
- 任何在用户 home 下、非 tmp 前缀的路径

---

## 五、三条通道改造清单

### 通道 A：Web UI

| 文件 | 位置 | 改动 |
|---|---|---|
| `knowledge_router.py:159` | `upload_document` | 加 `content_hash = SHA256(content)` 计算；KbDocument 创建时存入 |
| `worker.py:497-564` | `ingest_document_task` | **删除** `shutil.copy2`（文件已在调用方落位）。**删除** `IngestionPipeline` 直调。**删除** 手写 KbChunk 逻辑。**删除** line 534-535 source_path 覆写。改为：`file_path=perm_path`（不是 tmp_path），调 `unified.ingest_document(...)` |

### 通道 B：Agent MCP

| 文件 | 位置 | 改动 |
|---|---|---|
| `agent/tools/mcp.py:127` | `execute` | `ingest_document` 加入 kb_map 注入。单 KB 自动注入 kb_id；多 KB 且 Agent 未传 kb_id → 报错列出可选 KB |
| `collections.py:747-772` | `input_schema` | 加 `kb_id`（必填），`collection` 标记 deprecated，内部自动推导 |
| `collections.py:775-848` | `execute` | 加 temp 路径检查 → 如果在 temp 下，复制到 `rag/documents/{kb_id}/`；算 content_hash；调 `unified.ingest_document(...)` |

### 通道 C：手动/CLI

同通道 B，handler 自动兼容。

---

## 六、存量清洗时序

```
Phase 1：代码部署（统一接口 + worker 传 perm_path + mcp 注入 + PG/Chroma 同步写）
          ↓ 观察 1-2 天，确认增量不再产生
Phase 2：存量清洗
         每个 KB collection：
           1. 从 PG 拉 KbDocument → (content_hash, doc_id, file_path)
           2. 按 (kb_id, content_hash) 分组，找重复组
           3. 每组保留最早 doc，其余 → ChromaDB delete_by_source_path → PG delete KbChunk/Cocument
           4. "default" collection 的 MCP 直写数据无 PG 记录，直接扫 ChromaDB metadata 去重
          ↓ 清洗完毕后，同一 (kb_id, content_hash) 唯一
Phase 3：加 UNIQUE(kb_id, content_hash) 约束
```

三个 Phase 有严格先后顺序。Phase 2 必须在 source_path 固化后跑（否则 delete_by_source_path 打不中 tmp 路径的旧 chunk）。

---

## 七、v3 相对 v2 的变动

| 变动 | 位置 |
|---|---|
| 追责 temp 路径来源：只有 worker，Agent 没有 temp 习惯。修复 worker 传 perm_path + 接口加校验 | 4.5 |
| 失败回滚：Pipeline 失败 → PG 标 failed + 清 Chroma | Step D, 4.1 |
| UNIQUE 约束分 Phase，排在存量清洗后 | 4.2, 六 |
| PG/Chroma 同步写入，不回读 ChromaDB | 4.3, Step E |
| NULL collection 报错不兜底 | 4.4 |
| 多 KB 场景报错，不选默认 | 三(b) |
