"""MinerU PDF 解析测试脚本。

测试 MinerU (magic-pdf) 对学术论文的解析效果。

用法：
    python nanobot/rag/ingestion/parsing/mineru_test.py <pdf_path>
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional


def test_mineru_parse(pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """使用 MinerU 解析 PDF。

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录（默认为 PDF 同目录）

    Returns:
        解析结果信息
    """
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

        print(f"正在解析: {pdf_path}")

        # 设置输出目录
        if output_dir is None:
            output_dir = str(Path(pdf_path).parent)

        # 读取 PDF
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)

        # 创建数据集
        ds = PymuDocDataset(pdf_bytes)

        # 分析文档
        if ds.classify() == "ocr":
            print("  检测到需要 OCR 处理...")
            infer_result = ds.apply_ocr()
        else:
            print("  使用文本提取模式...")
            infer_result = ds.apply()

        # 获取内容列表
        pipe_result = infer_result.pipe_ocr_mode if ds.classify() == "ocr" else infer_result.pipe_txt_mode

        # 导出为 Markdown
        content_list = pipe_result.get_content_list(
            FileBasedDataWriter(output_dir),
            os.path.basename(pdf_path).replace(".pdf", ""),
        )

        # 导出 Markdown 文件
        md_path = os.path.join(
            output_dir,
            os.path.basename(pdf_path).replace(".pdf", "") + ".md"
        )
        pipe_result.dump_md(
            FileBasedDataWriter(output_dir),
            md_path,
            os.path.basename(pdf_path).replace(".pdf", ""),
        )

        # 统计信息
        stats = {
            "pdf_path": pdf_path,
            "output_dir": output_dir,
            "md_path": md_path,
            "classify": ds.classify(),
            "success": True,
        }

        # 统计表格数量
        table_count = 0
        for item in content_list:
            if item.get("type") == "table":
                table_count += 1
        stats["table_count"] = table_count

        return stats

    except ImportError as e:
        print(f"MinerU 未安装: {e}")
        print("请运行: uv pip install magic-pdf[full]")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def test_simple_api(pdf_path: str) -> Dict[str, Any]:
    """使用 MinerU 简单 API 测试。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        解析结果
    """
    try:
        # 尝试使用高级 API
        from magic_pdf import MagicPDF

        print(f"使用 MagicPDF API 解析: {pdf_path}")

        mpdf = MagicPDF(pdf_path)
        result = mpdf.parse()

        print(f"解析完成: {len(result.pages)} 页")

        # 统计表格
        table_count = sum(1 for page in result.pages for block in page.blocks if block.type == "table")
        print(f"检测到 {table_count} 个表格")

        # 导出 Markdown
        output_dir = str(Path(pdf_path).parent)
        md_content = result.to_markdown()

        md_path = os.path.join(output_dir, Path(pdf_path).stem + "_mineru.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"已保存到: {md_path}")

        return {
            "success": True,
            "pages": len(result.pages),
            "table_count": table_count,
            "md_path": md_path,
        }

    except ImportError:
        # 回退到底层 API
        print("高级 API 不可用，使用底层 API...")
        return test_mineru_parse(pdf_path)
    except Exception as e:
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    if len(sys.argv) < 2:
        print("用法: python mineru_test.py <pdf_path>")
        print("示例: python mineru_test.py C:/Users/Augix/Downloads/GOF_arxiv_v2.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    print("=" * 60)
    print("MinerU PDF 解析测试")
    print("=" * 60)

    # 先尝试简单 API
    result = test_simple_api(pdf_path)

    if result.get("success"):
        print("\n解析成功!")
        print(f"  表格数量: {result.get('table_count', 0)}")
        print(f"  输出文件: {result.get('md_path', '')}")

        # 显示前几行
        md_path = result.get("md_path")
        if md_path and Path(md_path).exists():
            print("\n前 500 字符:")
            print("-" * 40)
            with open(md_path, "r", encoding="utf-8") as f:
                print(f.read(500))
    else:
        print(f"\n解析失败: {result.get('error')}")


if __name__ == "__main__":
    main()
