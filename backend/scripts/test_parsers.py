"""Test all three PDF parsers on a sample PDF."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 output on Windows to avoid gbk codec errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path

# Use first CLI arg as PDF path, or find any PDF in current dir
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    pdfs = list(Path(".").glob("**/*.pdf"))
    if not pdfs:
        print("Usage: python scripts/test_parsers.py <path/to/file.pdf>")
        sys.exit(1)
    pdf_path = str(pdfs[0])

print(f"Testing with: {pdf_path}\n")


def test_markitdown(path):
    from nanobot.rag.libs.loader.pdf_loader import PdfLoader
    loader = PdfLoader(extract_images=False)
    doc = loader.load(path)
    preview = doc.text[:200].replace("\n", " ")
    print(f"[markitdown] OK — {len(doc.text)} chars, preview: {preview!r}")


def test_marker(path):
    from nanobot.rag.libs.loader.marker_loader import MarkerLoader, MARKER_AVAILABLE, MARKER_API
    if not MARKER_AVAILABLE:
        print("[marker] SKIP — not installed")
        return
    print(f"[marker] API={MARKER_API}, loading models (may take a while)...")
    loader = MarkerLoader(device="cpu", extract_images=False)
    doc = loader.load(path)
    preview = doc.text[:200].replace("\n", " ")
    print(f"[marker] OK — {len(doc.text)} chars, preview: {preview!r}")


def test_mineru(path):
    from nanobot.rag.libs.loader.mineru_loader import MinerULoader
    print("[mineru] loading models (may take a while)...")
    loader = MinerULoader(mode="local")
    doc = loader.load(path)
    preview = doc.text[:200].replace("\n", " ")
    print(f"[mineru] OK — {len(doc.text)} chars, preview: {preview!r}")


for name, fn in [("markitdown", test_markitdown), ("marker", test_marker), ("mineru", test_mineru)]:
    print(f"--- {name} ---")
    try:
        fn(pdf_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[{name}] ERR: {e}")
    print()
