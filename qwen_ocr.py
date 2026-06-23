"""OCR PDFs to Markdown using local Qwen VLM (qwen2.5vl:3b via Ollama).

Renders each PDF page with PyMuPDF (no poppler needed) and sends the page PNG
to Ollama's qwen2.5vl model, which returns GitHub-flavored Markdown.

Usage:
    python3 qwen_ocr.py <src_dir> <out_dir> [--model qwen2.5vl:3b] [--dpi 200]
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import fitz  # PyMuPDF
import requests

OLLAMA_HOST = "http://localhost:11434"

PROMPT = """You are an OCR engine for legal/case documents.

Transcribe the page image to clean GitHub-flavored Markdown. Rules:
- Preserve headings, lists, tables, signatures, stamps, footnotes.
- Keep dates verbatim; do not normalize or interpret them.
- Use horizontal rules (`---`) for clear section breaks.
- Do NOT add commentary, summary, or any text not on the page.
- If a region is illegible, mark it as `[illegible]`.

Return ONLY the Markdown transcription."""


def transcribe(png: bytes, model: str) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def page_png(page: "fitz.Page", dpi: int) -> bytes:
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def ocr_pdf(path: Path, model: str, dpi: int) -> str:
    doc = fitz.open(path)
    chunks = []
    for i, page in enumerate(doc, start=1):
        print(f"    page {i}/{doc.page_count}...", flush=True)
        md = transcribe(page_png(page, dpi), model)
        chunks.append(f"## Page {i}\n\n{md}\n")
    doc.close()
    return "\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {src}", file=sys.stderr)
        return 1

    print(f"Model: {args.model}  |  {len(pdfs)} PDF(s) -> {out}")
    for pdf in pdfs:
        print(f"OCR {pdf.name}", flush=True)
        try:
            md = ocr_pdf(pdf, args.model, args.dpi)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        dest = out / f"{pdf.stem}.md"
        dest.write_text(f"# Source: {pdf.name}\n\n{md}", encoding="utf-8")
        print(f"  -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
