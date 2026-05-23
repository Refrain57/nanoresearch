---
name: paper-digest
description: 每日自动搜集 arxiv 论文、精读摘要、推送给用户，并在用户确认后将论文 PDF 入库到知识库。
---

# Paper Digest Skill

## 概述

本 Skill 定义每日论文推送的完整工作流，分为两个独立阶段：

- **Phase A（定时触发）**：搜集 → 精读 → 推送摘要给用户
- **Phase B（用户回复触发）**：下载 PDF → 入库知识库 → 确认

---

## Phase A：搜集、精读、推送

> 触发时机：Cron 任务每日触发，或用户主动要求"执行论文推送"

### 步骤 1：搜索论文

用 `web_search` 搜索 arxiv 最新论文，关键词覆盖目标领域。每次搜索 15~20 条，提取 arxiv ID。

```
web_search("arxiv 2026 RAG retrieval augmented generation new paper", count=15)
web_search("arxiv 2026 LLM agent tool use latest", count=15)
```

从结果中提取 arxiv ID（格式：`2506.XXXXX`），去重后取前 **5 篇**相关度最高的。

**去重规则**：对比 `paper_inbox.json` 中已有的 arxiv_id，跳过已推送过的论文。

### 步骤 2：精读（spawn 后台执行）

精读属于耗时操作，用 `spawn` 后台执行，避免阻塞主 agent：

```
message("正在精读今日论文，稍后推送摘要...")

spawn(
    task="""精读以下 arxiv 论文并生成结构化摘要，完成后通知主 agent。

论文列表：
- https://arxiv.org/html/2506.XXXXX
- https://arxiv.org/html/2506.YYYYY
...

对每篇论文，用 web_fetch 抓取 HTML 全文（优先 arxiv.org/html/{id}，
不可用时降级到 arxiv.org/abs/{id} 摘要页），然后生成以下格式的 JSON：

{
  "arxiv_id": "2506.XXXXX",
  "title": "论文标题",
  "authors": ["作者1", "作者2"],
  "arxiv_url": "https://arxiv.org/abs/2506.XXXXX",
  "pdf_url": "https://arxiv.org/pdf/2506.XXXXX",
  "problem": "解决了什么核心问题（1句话）",
  "method": "核心方法/贡献（2~3句话）",
  "result": "主要实验结论（1~2句话）",
  "relevance": "与 RAG/LLM Agent 领域的关联（1句话）"
}

所有论文完成后，把完整的 JSON 数组返回。""",
    label="paper-digest-read"
)
```

### 步骤 3：接收精读结果，保存 inbox，推送用户

Subagent 完成后会自动把结果发回主 agent。收到后：

**3a. 写入 `paper_inbox.json`**（workspace 根目录）：

```json
{
  "date": "2026-05-28",
  "papers": [
    {
      "index": 1,
      "arxiv_id": "2506.XXXXX",
      "title": "...",
      "authors": ["..."],
      "arxiv_url": "https://arxiv.org/abs/2506.XXXXX",
      "pdf_url": "https://arxiv.org/pdf/2506.XXXXX",
      "problem": "...",
      "method": "...",
      "result": "...",
      "relevance": "...",
      "status": "pending"
    }
  ]
}
```

**3b. 通过 `message` 推送摘要给用户**，格式如下：

```
📄 今日论文精读（2026-05-28）共 5 篇

① <论文标题>
   问题：<problem>
   方法：<method>
   结论：<result>
   🔗 arxiv:2506.XXXXX

② ...

---
回复编号入库，例如：
  "入库 1 3"   → 将第1、3篇加入知识库
  "全部入库"   → 全部加入
  "跳过"       → 今日不入库
```

---

## Phase B：下载 PDF 并入库

> 触发时机：用户回复包含"入库"、编号或"全部"

### 步骤 1：解析用户意图

读取 `paper_inbox.json`，根据用户回复确定要入库的论文编号列表。

- "入库 1 3" → [1, 3]
- "全部入库" / "都要" → 所有 status=pending 的论文
- "跳过" / "不用了" → 更新全部 status=skipped，结束

### 步骤 2：逐篇下载 + 入库

对每篇要入库的论文：

**2a. 下载 PDF**

```
local_path = fetch_paper(
    url="https://arxiv.org/pdf/2506.XXXXX",
    filename="2506.XXXXX.pdf"
)
```

返回本地路径，例如 `/workspace/papers/2506.XXXXX.pdf`。

**2b. 调用 MCP ingest_document 入库**

```
mcp_rag_ingest_document(
    file_path=local_path,
    collection="papers",
    pdf_parser="marker"
)
```

返回 `task_id`，然后查询进度：

```
mcp_rag_get_task_status(task_id=task_id, wait=True)
```

等待完成（`status=completed`）后继续下一篇。

**2c. 更新 inbox 状态**

```json
"status": "accepted"
```

### 步骤 3：推送确认

```
message("✅ 已入库 2 篇论文：
  - 2506.XXXXX《...》→ papers collection，147 chunks
  - 2506.YYYYY《...》→ papers collection，203 chunks")
```

---

## 设置定时任务

用户首次使用时，让 Agent 注册 cron job：

```
cron(
    action="add",
    message="执行每日论文精读推送任务。按照 paper-digest skill 的指南完成完整的 Phase A 流程：搜索 arxiv 最新 RAG 和 LLM Agent 论文，精读后推送摘要给用户。目标领域：RAG、LLM Agent、Agentic System。每次推送 5 篇。",
    cron_expr="0 8 * * *",
    tz="Asia/Shanghai"
)
```

---

## 状态说明

| status | 含义 |
|--------|------|
| `pending` | 已推送，等待用户决策 |
| `accepted` | 用户已确认，已入库 |
| `skipped` | 用户跳过 |
| `expired` | 超过 48 小时未处理（下次 cron 触发时自动标记） |

过期处理：每次 Phase A 开始前，将 inbox 中超过 48 小时仍为 `pending` 的条目标记为 `expired`。

---

## 降级策略

| 情况 | 处理方式 |
|------|---------|
| `arxiv.org/html/{id}` 不可访问 | 降级到 `arxiv.org/abs/{id}` 摘要页 |
| `fetch_paper` 下载失败 | 告知用户，跳过该篇，继续其余 |
| `ingest_document` 失败 | 告知用户错误信息，PDF 文件保留在 `workspace/papers/` |
| Subagent 精读超时 | 推送已完成的部分，注明哪篇未能精读 |

---

## 工具速查

| 工具 | 用途 |
|------|------|
| `web_search` | 搜索 arxiv 论文列表 |
| `web_fetch` | 抓取论文 HTML 全文或摘要页 |
| `spawn` | 后台精读（避免阻塞） |
| `fetch_paper` | 下载论文 PDF 到本地 |
| `mcp_rag_ingest_document` | 将 PDF 入库到知识库 |
| `mcp_rag_get_task_status` | 查询入库进度 |
| `read_file` / `write_file` | 读写 paper_inbox.json |
| `message` | 向用户推送摘要或确认消息 |
| `cron` | 注册/管理定时任务 |
