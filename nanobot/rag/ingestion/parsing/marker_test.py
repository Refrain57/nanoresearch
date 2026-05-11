"""Marker PDF 解析测试脚本。

测试 Marker 对学术论文的解析效果。

用法：
    python nanobot/rag/ingestion/parsing/marker_test.py <pdf_path>
"""

import sys
from pathlib import Path
from PIL import Image


def test_marker_parse(pdf_path: str) -> dict:
    """使用 Marker 解析 PDF（GPU 加速）。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        解析结果信息
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        print(f"正在解析: {pdf_path}")
        print("加载模型...")

        # 创建模型字典，使用 GPU
        models = create_model_dict(device="cuda")

        # 创建转换器
        converter = PdfConverter(artifact_dict=models)

        # 解析 PDF
        rendered = converter(pdf_path)

        # 提取文本和图片
        text, _, images = text_from_rendered(rendered)

        # 保存结果
        output_dir = Path(pdf_path).parent
        pdf_stem = Path(pdf_path).stem

        # 保存 Markdown
        output_path = output_dir / f"{pdf_stem}_marker.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        # 保存图片
        images_dir = output_dir / f"{pdf_stem}_marker_images"
        images_dir.mkdir(exist_ok=True)

        saved_images = 0
        for filename, img in images.items():
            if isinstance(img, Image.Image):
                # 保存为 PNG
                img_path = images_dir / filename.replace(".jpeg", ".png")
                img.save(img_path, "PNG")
                saved_images += 1

        print(f"\n已保存到: {output_path}")
        print(f"  字符数: {len(text)}")
        print(f"  图片数: {len(images)}")
        print(f"  图片目录: {images_dir}")

        return {
            "success": True,
            "text": text,
            "output_path": str(output_path),
            "images_dir": str(images_dir),
            "char_count": len(text),
            "image_count": len(images),
        }

    except Exception as e:
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    if len(sys.argv) < 2:
        print("用法: python marker_test.py <pdf_path>")
        print("示例: python marker_test.py C:/Users/Augix/Downloads/GOF_arxiv_v2.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    print("=" * 60)
    print("Marker PDF 解析测试")
    print("=" * 60)

    result = test_marker_parse(pdf_path)

    if result.get("success"):
        print("\n解析成功!")
        print(f"  字符数: {result.get('char_count', 0)}")
        print(f"  图片数: {result.get('image_count', 0)}")
        print(f"  输出文件: {result.get('output_path', '')}")

        # 显示前几行
        text = result.get("text", "")
        if text:
            print("\n前 500 字符:")
            print("-" * 40)
            print(text[:500])
    else:
        print(f"\n解析失败: {result.get('error')}")


if __name__ == "__main__":
    main()
