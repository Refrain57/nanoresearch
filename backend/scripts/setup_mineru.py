"""Setup script for MinerU conda environment.

Run once inside the mineru conda environment to:
  1. Write magic-pdf.json config pointing to local model weights
  2. Verify all required packages are importable
  3. Print the pip install command needed if anything is missing

Usage:
    conda activate mineru
    python scripts/setup_mineru.py
"""

import json
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models" / "mineru"
LAYOUTREADER_DIR = REPO_ROOT / "models" / "layoutreader"
CONFIG_FILE = Path.home() / "magic-pdf.json"

# ── Write magic-pdf.json ──────────────────────────────────────────────────────
config = {
    "bucket_info": {},
    "models-dir": str(MODELS_DIR),
    "device-mode": "cuda",
    "temp-output-dir": str(Path.home() / ".magic-pdf" / "tmp"),
    "layout-config": {
        "model": "doclayout_yolo"
    },
    "formula-config": {
        "mfd_model": "yolo_v8_mfd",
        "mfr_model": "unimernet_small",
        "enable": True
    },
    "table-config": {
        "model": "rapid_table",
        "enable": False
    },
    "layoutreader-model-dir": str(LAYOUTREADER_DIR),
}

CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False))
print(f"✓ magic-pdf.json written to {CONFIG_FILE}")
print(f"  models-dir : {MODELS_DIR}")
print(f"  device-mode: cuda")

# ── Verify packages ───────────────────────────────────────────────────────────
print("\nVerifying packages...")

checks = [
    # magic-pdf 1.x does NOT use the external paddleocr or unimernet packages;
    # it bundles its own unimernet_hf and uses paddle only optionally for OCR.
    ("magic_pdf",      "magic-pdf"),
    ("paddle",         "paddlepaddle-gpu"),
    ("ultralytics",    "ultralytics"),
    ("cv2",            "opencv-python"),
    ("torch",          "torch"),
    ("torchvision",    "torchvision"),
    ("fastapi",        "fastapi"),
    ("uvicorn",        "uvicorn"),
    ("omegaconf",      "omegaconf"),
]

missing = []
for module, pkg in checks:
    try:
        __import__(module)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg}  ← MISSING")
        missing.append(pkg)

if missing:
    print("\n⚠ Missing packages. Install with:")
    print(f"  pip install {' '.join(repr(p) for p in missing)}")
    sys.exit(1)
else:
    print("\n✓ All packages present. Run the server with:")
    print("  python scripts/mineru_server.py")
