"""Hybrid PDF 解析测试脚本。

策略：
1. PyMuPDF 检测布局（双栏、表格）
2. 双栏/表格 → PyMuPDF 逐页处理
3. 单栏简单文档 → MarkItDown
4. 表格 → LLM 辅助修正结构

用法：
    python nanoresearch/rag/ingestion/parsing/hybrid_parser_test.py <pdf_path>
"""

import sys
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# LLM 表格修正 Prompt
TABLE_FIX_PROMPT = """你是一个学术论文表格修复专家。请修复以下 Markdown 表格的结构问题。

原始表格数据来自 PDF 解析，可能存在以下问题：
1. 单元格内包含多个值（用空格分隔）
2. 单元格内有换行符
3. 有些单元格是空的（None）
4. 列对齐不正确

请根据以下规则修复表格：
1. 每个单元格只包含一个值
2. 清理单元格内容（去除多余空格）
3. 保持合理的列数
4. 空单元格用 "-" 或适当内容填充

输入表格（原始数据）：
{table_text}

请输出修复后的 Markdown 表格，只输出表格，不要有其他内容。
"""


def detect_layout(doc: fitz.Document) -> Dict:
    """检测 PDF 布局结构。

    Returns:
        {
            "is_two_column": bool,
            "has_tables": bool,
            "confidence": float,
            "page_count": int,
            "layout_type": "single" | "two_column" | "mixed"
        }
    """
    if len(doc) == 0:
        return {"is_two_column": False, "has_tables": False, "confidence": 0.0, "page_count": 0, "layout_type": "unknown"}

    # 检测双栏：分析前几页的文本块位置
    two_column_count = 0
    sample_pages = min(5, len(doc))

    for page_num in range(sample_pages):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        text_blocks = [b for b in blocks if b["type"] == 0]
        if len(text_blocks) < 2:
            continue

        # 收集文本块的 X 范围（bbox: x0, y0, x1, y1）
        block_x_ranges = []
        for block in text_blocks:
            bbox = block["bbox"]
            x0, x1 = bbox[0], bbox[2]
            block_x_ranges.append((x0, x1))

        if len(block_x_ranges) < 2:
            continue

        # 计算页面的 X 分布
        min_x = min(r[0] for r in block_x_ranges)
        max_x = max(r[1] for r in block_x_ranges)
        page_width = page.rect.width

        # 用页面的中点来判断
        mid_x = (min_x + max_x) / 2

        # 统计左栏和右栏的块数（块的中心位置）
        left_count = sum(1 for x0, x1 in block_x_ranges if (x0 + x1) / 2 < mid_x)
        right_count = sum(1 for x0, x1 in block_x_ranges if (x0 + x1) / 2 >= mid_x)

        # 如果左右都有块，且分布相对均匀，认为是双栏
        total = left_count + right_count
        if total > 0:
            ratio = min(left_count, right_count) / total
            # 降低阈值到 0.15，因为有些页右栏内容较少（如参考文献、图表说明）
            if ratio > 0.15:  # 两侧都有至少 15% 的块
                two_column_count += 1
                print(f"  Page {page_num + 1}: 左={left_count}, 右={right_count}, ratio={ratio:.2f} → 双栏")

    # 判断是否双栏
    is_two_column = two_column_count >= sample_pages * 0.5

    # 检测表格（尝试 find_tables）- 扫描全部页面
    has_tables = False
    table_count = 0
    table_pages = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()
            if tables and tables.tables:
                page_table_count = len(tables.tables)
                table_count += page_table_count
                table_pages.append((page_num + 1, page_table_count))
                has_tables = True
    except Exception as e:
        print(f"  表格检测错误: {e}")

    # 确定布局类型
    if is_two_column:
        layout_type = "two_column"
    elif has_tables:
        layout_type = "table_heavy"
    else:
        layout_type = "single"

    confidence = two_column_count / sample_pages if sample_pages > 0 else 0.0

    return {
        "is_two_column": is_two_column,
        "has_tables": has_tables,
        "confidence": confidence,
        "page_count": len(doc),
        "layout_type": layout_type,
        "two_column_pages": two_column_count,
        "sample_pages": sample_pages,
        "table_count": table_count,
        "table_pages": table_pages,
    }


def extract_text_pymupdf(doc: fitz.Document, page_num: int) -> str:
    """使用 PyMuPDF 提取单页文本。

    Args:
        doc: PyMuPDF 文档
        page_num: 页码（从 0 开始）

    Returns:
        提取的文本
    """
    page = doc[page_num]
    text = page.get_text("text")
    return text


def extract_with_layout_preservation(doc: fitz.Document, page_num: int) -> str:
    """使用 PyMuPDF 提取文本，保留布局信息。

    Args:
        doc: PyMuPDF 文档
        page_num: 页码

    Returns:
        带布局标记的文本
    """
    page = doc[page_num]
    blocks = page.get_text("dict")["blocks"]

    text_parts = []

    for block in blocks:
        if block["type"] == 0:  # 文本块
            block_text = ""
            for line in block["lines"]:
                line_text = " ".join(span["text"] for span in line["spans"])
                block_text += line_text + "\n"
            text_parts.append(block_text.strip())
        elif block["type"] == 1:  # 图片块
            text_parts.append("[IMAGE]")

    return "\n\n".join(text_parts)


def extract_tables_pymupdf(doc: fitz.Document, page_num: int) -> List[str]:
    """使用 PyMuPDF 检测并提取表格。

    Args:
        doc: PyMuPDF 文档
        page_num: 页码

    Returns:
        表格列表（每个表格为 CSV 格式字符串）
    """
    page = doc[page_num]
    tables = []

    try:
        table_settings = page.find_tables()
        if table_settings and table_settings.tables:
            for table in table_settings.tables:
                # 提取表格数据
                data = table.extract()
                if data:
                    # 转为 CSV 格式
                    csv_lines = []
                    for row in data:
                        csv_lines.append(",".join(str(cell).replace(",", ";") for cell in row))
                    tables.append("\n".join(csv_lines))
    except Exception as e:
        print(f"  表格提取失败: {e}")

    return tables


def extract_table_as_markdown(table, use_llm_fix: bool = True) -> Optional[str]:
    """将 PyMuPDF 表格对象转换为 Markdown 格式。

    Args:
        table: PyMuPDF 表格对象
        use_llm_fix: 是否使用 LLM 修正表格结构

    Returns:
        Markdown 格式的表格字符串
    """
    try:
        data = table.extract()
        if not data or len(data) == 0:
            return None

        # 先生成原始 Markdown
        raw_md = _data_to_raw_markdown(data)

        if use_llm_fix:
            # 用 LLM 修正
            fixed_md = _fix_table_with_llm(raw_md)
            if fixed_md:
                return fixed_md

        # 回退到原始 Markdown
        return raw_md
    except Exception as e:
        print(f"  表格转 Markdown 失败: {e}")
        return None


def _data_to_raw_markdown(data: List[List]) -> str:
    """将表格数据转为原始 Markdown。

    Args:
        data: 表格数据（二维列表）

    Returns:
        Markdown 格式字符串
    """
    lines = []
    for i, row in enumerate(data):
        # 处理每个单元格
        row_cells = []
        for cell in row:
            if cell is None:
                row_cells.append("")
            else:
                cell_str = str(cell).replace("|", "\\|").replace("\n", " ")
                row_cells.append(cell_str)

        lines.append("| " + " | ".join(row_cells) + " |")

        # 添加表头分隔符
        if i == 0:
            sep_cells = ["---"] * len(row)
            lines.append("|" + "|".join(sep_cells) + "|")

    return "\n".join(lines)


def _fix_table_with_llm(raw_table: str) -> Optional[str]:
    """使用 LLM 修正表格结构。

    Args:
        raw_table: 原始 Markdown 表格

    Returns:
        修正后的 Markdown 表格
    """
    try:
        from nanoresearch.rag.core.settings import get_settings

        settings = get_settings()
        if not hasattr(settings, "llm") or not settings.llm:
            return None

        # 获取 LLM 配置
        api_key = settings.llm.api_key
        base_url = settings.llm.base_url
        model = settings.llm.model

        # 调用 LLM
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        prompt = TABLE_FIX_PROMPT.format(table_text=raw_table)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )

        result = response.choices[0].message.content.strip()

        # 提取 Markdown 表格
        if "|" in result:
            return result

        return None

    except Exception as e:
        print(f"  LLM 表格修正失败: {e}")
        return None


def extract_two_column_text(doc: fitz.Document, page_num: int, page_tables: List) -> str:
    """双栏提取文本，同时处理表格区域。

    Args:
        doc: PyMuPDF 文档
        page_num: 页码
        page_tables: 当前页的表格对象列表

    Returns:
        双栏格式的文本
    """
    page = doc[page_num]
    blocks = page.get_text("dict")["blocks"]

    text_blocks = [b for b in blocks if b["type"] == 0]
    if len(text_blocks) < 2:
        return page.get_text("text")

    # 收集表格区域
    table_bboxes = []
    for t in page_tables:
        if hasattr(t, 'bbox'):
            table_bboxes.append(t.bbox)

    # 找中线
    min_x = min(b['bbox'][0] for b in text_blocks)
    max_x = max(b['bbox'][2] for b in text_blocks)
    mid_x = (min_x + max_x) / 2

    left_texts = []
    right_texts = []

    for block in text_blocks:
        bbox = block['bbox']
        block_x0, block_x1 = bbox[0], bbox[2]
        center_x = (block_x0 + block_x1) / 2

        # 检查是否在表格区域内
        is_in_table = any(
            tb[0] <= center_x <= tb[2] and tb[1] <= bbox[1] <= tb[3]
            for tb in table_bboxes
        )
        if is_in_table:
            continue  # 跳过表格内的文本

        block_text = ""
        for line in block["lines"]:
            line_text = " ".join(span["text"] for span in line["spans"])
            block_text += line_text + "\n"
        block_text = block_text.strip()

        if not block_text:
            continue

        if center_x < mid_x:
            left_texts.append(block_text)
        else:
            right_texts.append(block_text)

    # 合并：左栏在上，右栏在下
    return "\n\n".join(left_texts) + "\n\n" + "\n\n".join(right_texts)


def parse_markitdown(pdf_path: str) -> Tuple[str, Dict]:
    """使用 MarkItDown 解析 PDF。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        (markdown 文本, 解析元数据)
    """
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(pdf_path)

        return result.text_content, {
            "engine": "markitdown",
            "char_count": len(result.text_content),
        }
    except Exception as e:
        return f"[MarkItDown 解析失败: {e}]", {"engine": "markitdown", "error": str(e), "char_count": 0}


def parse_pymupdf(pdf_path: str) -> Tuple[str, Dict]:
    """使用 PyMuPDF 解析 PDF。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        (markdown 文本, 解析元数据)
    """
    doc = fitz.open(pdf_path)

    # 检测布局
    layout_info = detect_layout(doc)

    texts = []
    total_tables = 0

    for page_num in range(len(doc)):
        # 先检测当前页是否有表格
        page_tables = []
        try:
            tables = doc[page_num].find_tables()
            if tables and tables.tables:
                for t in tables.tables:
                    page_tables.append(t)
        except:
            pass

        # 提取文本
        if layout_info["is_two_column"]:
            # 双栏布局：按列提取
            page_text = extract_two_column_text(doc, page_num, page_tables)
        else:
            # 单栏布局：简单提取
            page_text = extract_text_pymupdf(doc, page_num)

        texts.append(f"## Page {page_num + 1}\n\n{page_text}")

        # 提取表格（转为 Markdown 格式）
        for table in page_tables:
            table_md = extract_table_as_markdown(table)
            if table_md:
                texts.append(f"\n### Table\n\n{table_md}\n")
                total_tables += 1

    doc.close()

    full_text = "\n\n---\n\n".join(texts)

    return full_text, {
        "engine": "pymupdf",
        "layout": layout_info,
        "total_tables": total_tables,
        "char_count": len(full_text),
    }


def hybrid_parse(pdf_path: str) -> Tuple[str, Dict]:
    """混合解析策略：先用 PyMuPDF 检测，再用合适的方式处理。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        (解析后的文本, 元数据)
    """
    doc = fitz.open(pdf_path)
    layout_info = detect_layout(doc)
    doc.close()

    print(f"布局检测结果: {layout_info}")

    # 根据布局选择解析策略
    if layout_info["layout_type"] == "single":
        print("→ 单栏布局，使用 MarkItDown")
        text, meta = parse_markitdown(pdf_path)
        meta["strategy"] = "markitdown_simple"
        return text, meta

    elif layout_info["layout_type"] == "two_column":
        print("→ 双栏布局，使用 PyMuPDF")
        text, meta = parse_pymupdf(pdf_path)
        meta["strategy"] = "pymupdf_two_column"
        return text, meta

    elif layout_info["layout_type"] == "table_heavy":
        print("→ 表格密集布局，使用 PyMuPDF")
        text, meta = parse_pymupdf(pdf_path)
        meta["strategy"] = "pymupdf_tables"
        return text, meta

    else:
        # 默认使用 MarkItDown
        print("→ 未知布局，使用 MarkItDown")
        text, meta = parse_markitdown(pdf_path)
        meta["strategy"] = "markitdown_default"
        return text, meta


def compare_engines(pdf_path: str) -> Dict:
    """对比两个引擎的解析结果。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        对比结果
    """
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print("=" * 60)

    # PyMuPDF 布局检测
    doc = fitz.open(pdf_path)
    layout_info = detect_layout(doc)
    doc.close()

    print("\n[1] PyMuPDF 布局检测:")
    print(f"    布局类型: {layout_info['layout_type']}")
    print(f"    页数: {layout_info['page_count']}")
    print(f"    双栏: {layout_info['is_two_column']} (置信度: {layout_info['confidence']:.2f})")
    print(f"    表格: {layout_info['has_tables']}")

    # MarkItDown
    print("\n[2] MarkItDown 解析:")
    try:
        md_text, md_meta = parse_markitdown(pdf_path)
        print(f"    字符数: {md_meta['char_count']}")
        if md_meta.get('error'):
            print(f"    错误: {md_meta['error']}")
        print(f"    前 200 字符:\n{md_text[:200]}...")
    except Exception as e:
        print(f"    MarkItDown 失败: {e}")
        md_text = ""
        md_meta = {"engine": "markitdown", "error": str(e), "char_count": 0}

    # PyMuPDF
    print("\n[3] PyMuPDF 解析:")
    pymupdf_text, pymupdf_meta = parse_pymupdf(pdf_path)
    print(f"    字符数: {pymupdf_meta['char_count']}")
    print(f"    前 200 字符:\n{pymupdf_text[:200]}...")

    # 混合策略
    print("\n[4] 混合策略建议:")
    text, meta = hybrid_parse(pdf_path)
    print(f"    选用: {meta['strategy']}")
    print(f"    字符数: {meta['char_count']}")

    return {
        "layout_info": layout_info,
        "markitdown": md_meta,
        "pymupdf": pymupdf_meta,
        "recommended": meta,
        "final_text": text,
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python hybrid_parser_test.py <pdf_path>")
        print("示例: python hybrid_parser_test.py ./papers/PGSR.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    result = compare_engines(pdf_path)

    # 输出最终结果
    print("\n" + "=" * 60)
    print("最终解析结果:")
    print("=" * 60)
    print(result["final_text"][:500])


if __name__ == "__main__":
    main()
