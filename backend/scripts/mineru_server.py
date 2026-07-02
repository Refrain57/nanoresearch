"""MinerU HTTP Server — wraps magic-pdf as a REST API.

Run in a dedicated conda environment:

    conda activate mineru
    pip install fastapi uvicorn python-multipart "magic-pdf[full]" \
        paddlepaddle "paddleocr==2.7.3" ultralytics unimernet \
        --extra-index-url https://myhloli.github.io/wheels/
    python scripts/mineru_server.py

Endpoint:
    POST /api/v1/extract   multipart PDF →
        {"markdown": "...", "images": {...}, "middle_json": "..."}
    middle_json is MinerU's per-block layout (bbox+page) used for chunk grounding.
"""

import base64
import hashlib
import logging
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="MinerU HTTP Server")


def _run_magic_pdf_v1(pdf_bytes: bytes, img_dir: Path):
    """magic-pdf >= 1.0 API (PymuDocDataset)."""
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

    ds = PymuDocDataset(pdf_bytes)
    image_writer = FileBasedDataWriter(str(img_dir))

    try:
        from magic_pdf.config.enums import SupportedPdfParseMethod
        use_ocr = ds.classify() == SupportedPdfParseMethod.OCR
    except Exception:
        use_ocr = False

    infer_result = ds.apply(doc_analyze, ocr=use_ocr)
    pipe_result = (
        infer_result.pipe_ocr_mode(image_writer)
        if use_ocr
        else infer_result.pipe_txt_mode(image_writer)
    )
    markdown = pipe_result.get_markdown(str(img_dir))

    # Extract image types from content_list
    image_types: dict = {}
    try:
        for item in pipe_result.get_content_list(str(img_dir)):
            label = "table" if item.get("type") == "table" else "figure"
            img_path_str = (
                item.get("img_path") or item.get("img_name") or item.get("image_path") or ""
            )
            if img_path_str:
                p = Path(img_path_str)
                abs_p = p if p.is_absolute() else img_dir / p
                image_types[str(abs_p)] = label
                image_types[p.name] = label
    except Exception:
        pass

    # middle.json carries per-block bbox+page for chunk grounding (F2)
    middle_json = ""
    try:
        middle_json = pipe_result.get_middle_json()
    except Exception as exc:
        logger.warning(f"MinerU: get_middle_json failed: {exc}")

    return markdown, image_types, middle_json


def _run_magic_pdf_v06(pdf_bytes: bytes, img_dir: Path):
    """magic-pdf 0.6.x API (UNIPipe)."""
    import magic_pdf.model as model_config
    model_config.__use_inside_model__ = True

    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

    image_writer = DiskReaderWriter(str(img_dir))
    jso_useful_key = {"_pdf_type": "", "model_list": []}
    pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()
    middle_json = ""
    try:
        import json as _json
        mid = getattr(pipe, "pdf_mid_data", None)
        if mid:
            middle_json = mid if isinstance(mid, str) else _json.dumps(mid)
    except Exception as exc:
        logger.warning(f"MinerU (v0.6.x): middle json unavailable: {exc}")
    return pipe.get_markdown(str(img_dir)), {}, middle_json


def _run_magic_pdf(pdf_bytes: bytes, doc_hash: str):
    img_dir = Path(tempfile.gettempdir()) / "mineru_images" / doc_hash
    img_dir.mkdir(parents=True, exist_ok=True)

    # Try v1.x first
    try:
        markdown, image_types, middle_json = _run_magic_pdf_v1(pdf_bytes, img_dir)
        return markdown, image_types, img_dir, middle_json
    except (ImportError, ModuleNotFoundError):
        pass
    except SystemExit:
        pass

    # Fallback to v0.6.x
    markdown, image_types, middle_json = _run_magic_pdf_v06(pdf_bytes, img_dir)
    return markdown, image_types, img_dir, middle_json


def _collect_images(img_dir: Path, image_types: dict) -> dict:
    """Read extracted images and encode as base64 for transport."""
    images = {}
    for img_path in img_dir.glob("**/*"):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        img_type = (
            image_types.get(str(img_path))
            or image_types.get(img_path.name)
            or "figure"
        )
        images[img_path.name] = {
            "data": base64.b64encode(img_path.read_bytes()).decode(),
            "type": img_type,
        }
    return images


def _sanitize_surrogates(text: str, doc_hash: str) -> str:
    """Remove lone Unicode surrogates produced by broken PDF font encodings."""
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    cleaned = re.sub(r"[\ud800-\udfff]", "", text)
    removed = len(text) - len(cleaned)
    if removed:
        logger.warning(f"PDF {doc_hash[:16]}: removed {removed} lone surrogate(s) from broken font encoding")
    return cleaned


@app.post("/api/v1/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    doc_hash = hashlib.sha256(pdf_bytes).hexdigest()

    try:
        markdown, image_types, img_dir, middle_json = _run_magic_pdf(pdf_bytes, doc_hash)
        images = _collect_images(img_dir, image_types)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not markdown or not markdown.strip():
        raise HTTPException(status_code=500, detail="MinerU returned empty content")

    markdown = _sanitize_surrogates(markdown, doc_hash)

    return JSONResponse({"markdown": markdown, "images": images, "middle_json": middle_json})


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
