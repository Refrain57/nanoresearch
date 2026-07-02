# Workspace 文件预览浮窗 + 删除功能 — 设计

- **日期**: 2026-07-02
- **分支**: `worktree-workspace-file-preview-delete`
- **来源**: `nanoresearch-待修清单.md` item 4 + item 5
- **规模**: 前端为主（1 个组件）+ 后端 1 个端点

## 背景 / 问题

### Item 4 — 工作区文件打不开（401）+ 开新网页（应为浮窗）
`web/src/components/WorkspaceFiles.vue:39-50` 把每个文件渲染成裸导航链接：

```html
<a :href="`/api/workspace/files/${entry.path}`" target="_blank" class="wf-row wf-file">
```

浏览器裸导航打开新标签、**不带 `Authorization` 头**；而后端 `download_file`
（`backend/nanoresearch/server/routers/workspace_router.py:84-89`）有
`Depends(get_current_user)` → 返回 `401 {"detail":"Not authenticated"}`。
且行为是「新开标签页」，而非期望的浮窗预览。

### Item 5 — 工作区没有删除文件功能
`workspace_router.py` 只有 `list(GET)` / `write(PUT)` / `download(GET)`，**无 DELETE 端点**；
前端也无删除入口。

## 目标

1. 点击工作区文件通过**带鉴权的 fetch** 拉取，在**浮窗（`a-modal`）**内预览，根治 401 与「开新标签」。
2. 新增带鉴权 + 路径越界防护的 **DELETE 端点**，前端加删除按钮 + 确认框，支持递归删目录。

## 非目标（YAGNI）

- 不做在线编辑（`PUT` 已存在，限 `EDITABLE_FILES`，本次不动）。
- 不做重命名 / 移动 / 新建文件夹。
- 不做批量选择删除；一次删一项。

## 设计

### Item 4 — 带鉴权浮窗预览（`WorkspaceFiles.vue`，前端为主）

**去掉裸链接。** 文件行由 `<a :href target="_blank">` 改为 `<div @click="openFile(entry)">`，
统一走 `fetch(url, { headers: userStore.getAuthHeaders() })`，不再依赖浏览器裸导航。

**按扩展名分流**（小写扩展名匹配集合）：

| 类型 | 扩展名 | 行为 |
|------|--------|------|
| 图片 | png, jpg, jpeg, gif, webp, svg, bmp, ico | fetch → blob → `objectURL`，modal 内 `<img :src>` |
| PDF  | pdf | fetch → blob → `objectURL`，modal 内 `<iframe :src>` |
| 文本 | md, txt, json, log, csv, yaml, yml, js, ts, jsx, tsx, vue, py, css, html, xml, sh, toml, ini | fetch → `blob.text()`，modal 内 `<pre>{{ text }}</pre>` |
| 其它 | 二进制等未列出扩展名 | 直接带鉴权 blob 下载，不开 modal（安全默认：不试图把二进制当文本渲染） |

**安全**：
- SVG 走 `<img>` 而非 `<iframe>`，`<img>` 上下文不执行内嵌脚本。
- 文本走 `<pre>{{ text }}</pre>`，Vue 插值自动 HTML 转义，防 XSS。

**Modal（复用项目已有 `a-modal` 模式）**：
- 标题 = 文件名；`:footer` 自定义一个「下载」按钮 → 预览类型也能下载。
- 关闭 modal 时 `URL.revokeObjectURL(previewUrl)` 回收对象 URL，避免内存泄漏。
- 组件卸载（`onBeforeUnmount`）时也回收。

**下载路径**：每行现有 `download-outlined` 图标保留，`@click.stop` 触发
`downloadFile(entry)` —— 带鉴权 fetch → blob → 临时 `<a download=name>` 点击 → `revokeObjectURL`。
（不论文件类型都走下载。）

**加载/错误反馈**：预览 fetch 期间 modal 内显示 `a-spin`；非 2xx 时 `message.error` 弹后端 `detail`。

### Item 5 — DELETE 端点 + 前端删除按钮

**后端** `workspace_router.py` 新增：

```python
import shutil  # 文件顶部

@router.delete("/api/workspace/files/{file_path:path}")
async def delete_file(
    file_path: str,
    request: Request,
    uid: str = Depends(get_current_user),
):
    # 1. 系统文件保护
    if Path(file_path).name in EDITABLE_FILES:
        raise HTTPException(status_code=403, detail="系统文件不允许删除")
    workspace = _user_workspace(request, uid)
    resolved = _safe_resolve(workspace, file_path)  # 复用路径越界防护 → 越界 403
    # 2. 不许删工作区根
    if resolved == workspace.resolve():
        raise HTTPException(status_code=403, detail="不允许删除工作区根目录")
    # 3. 不存在
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    # 4. 目录递归删 / 文件删
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()
    return {"deleted": file_path}
```

系统文件按**文件名**判定（与既有 `write_file` 的 `Path(file_path).name in EDITABLE_FILES` 判定一致）。

**前端**：
- 文件行与目录行都加删除图标（`delete-outlined`，hover 显示，`@click.stop` 防冒泡触发进目录/预览）。
- 点击 → `Modal.confirm` 确认框：文件提示「确认删除 `<name>`？」，目录额外提示「将递归删除该目录及全部内容」。
- 确认 → `fetch(url, { method: 'DELETE', headers: getAuthHeaders() })`。
- 成功 → `message.success` + `refresh()`（重新拉当前目录）；失败 → `message.error` 弹后端 `detail`。

## 数据流

```
[点击文件名] → openFile → fetch(+JWT) → blob
   ├─ 图片/PDF → objectURL → a-modal(<img>/<iframe>)
   ├─ 文本     → blob.text() → a-modal(<pre>)
   └─ 其它     → 触发下载
[点击下载图标] → downloadFile → fetch(+JWT) → blob → <a download> 点击
[点击删除图标] → Modal.confirm → DELETE(+JWT) → refresh()
```

## 错误处理

| 场景 | 后端 | 前端 |
|------|------|------|
| 未带 token | 401 | （已带 JWT，不再触发） |
| 路径越界 | 403「非法路径」 | `message.error` |
| 删系统文件 | 403「系统文件不允许删除」 | `message.error` |
| 删工作区根 | 403 | `message.error` |
| 文件不存在 | 404 | `message.error` + `refresh()` |
| 预览 fetch 失败 | — | modal 内 `message.error` |

## 测试 / 验证

**后端**（pytest，覆盖 delete 分支）：
- 删普通文件 → 200 + 文件消失。
- 删目录（含内容）→ 200 + 目录递归消失。
- 删 `SOUL.md` 等系统文件 → 403。
- 路径越界（`../` 逃逸）→ 403。
- 删不存在 → 404。

**前端**：
- `npm run build` 通过（无类型/编译错误）。
- 手动验证：图片/文本/PDF 三类预览；其它类型下载；删除确认流（文件 + 目录）。

## 落地方式

已按用户要求新建 git worktree（`worktree-workspace-file-preview-delete`），
spec 与实现代码都在该分支完成，保持 main 干净。
