import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
if not pdf_path:
    print("Usage: python scripts/test_mineru.py <path/to/file.pdf>"); sys.exit(1)

print(f"Testing mineru with: {pdf_path}\n")
try:
    from nanoresearch.rag.libs.loader.mineru_loader import MinerULoader
    loader = MinerULoader(mode="local")
    doc = loader.load(pdf_path)
    print(f"[mineru] OK — {len(doc.text)} chars")
    print(f"[mineru] preview: {doc.text[:300].replace(chr(10), ' ')!r}")
except Exception as e:
    traceback.print_exc()
    print(f"[mineru] ERR: {e}")
