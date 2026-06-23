"""Stage 1: OCR PDF / DOC / DOCX files into raw Markdown using a local VLM.

A locally-hosted Vision-Language Model (served by Ollama) reads each
rendered page image and emits Markdown directly. This preserves layout
cues like headings, lists and tables better than classical OCR, and keeps
all data on the local machine — no cloud calls.

The endpoint and model are resolved from `config.load_config()`.
"""

from __future__ import annotations

import io
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from .config import load_config
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


_VLM_PROMPT = """You are an OCR engine for legal/case documents.

Transcribe the page image to clean GitHub-flavored Markdown. Rules:
- Preserve headings, lists, tables, signatures, stamps, footnotes.
- Keep dates verbatim; do not normalize or interpret them.
- Use horizontal rules (`---`) for clear section breaks.
- Do NOT add commentary, summary, or any text not on the page.
- If a region is illegible, mark it as `[illegible]`.

Return ONLY the Markdown transcription."""


class VLMClient(Protocol):
    """Any callable that turns a PNG image into Markdown text."""

    def transcribe(self, image_png: bytes) -> str: ...


# ---------------------------------------------------------------------------
# Local VLM backend (Ollama)

# Decoding options for OCR. Small VLMs (e.g. qwen2.5vl:3b) are prone to
# degenerate repetition loops with the model defaults, so we decode greedily
# (temperature 0), penalise repeats, and cap the token budget per page to bound
# any runaway. These keep transcription deterministic and stop the loops.
_VLM_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.3,
    "repeat_last_n": 256,
    "num_predict": 2048,
}


@dataclass
class OllamaVisionClient:
    """VLM backend served by a local Ollama instance."""

    model: str | None = None
    prompt: str = _VLM_PROMPT
    client: OllamaClient = field(default_factory=OllamaClient)
    options: dict = field(default_factory=lambda: dict(_VLM_OPTIONS))

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = load_config().vision_model

    def transcribe(self, image_png: bytes) -> str:
        return self.client.chat(
            prompt=self.prompt,
            model=self.model,
            images_png=[image_png],
            options=self.options,
        ).strip()


def default_vlm() -> VLMClient:
    """Return the local Ollama vision client."""
    return OllamaVisionClient()


# ---------------------------------------------------------------------------
# OCR engine

class OCREngine:
    """OCR documents into Markdown via a VLM."""

    def __init__(
        self,
        output_dir: str | Path,
        vlm: VLMClient | None = None,
        dpi: int = 200,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vlm = vlm or default_vlm()
        self.dpi = dpi

    def ocr_file(self, source: str | Path) -> Path:
        source = Path(source)
        suffix = source.suffix.lower()

        if suffix == ".pdf":
            markdown = self._ocr_pdf(source)
        elif suffix == ".docx":
            markdown = self._ocr_docx(source)
        elif suffix == ".doc":
            markdown = self._ocr_doc(source)
        elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            markdown = self._ocr_image(source)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        out_path = self.output_dir / f"{source.stem}.raw.md"
        header = f"# Source: {source.name}\n\n"
        out_path.write_text(header + markdown, encoding="utf-8")
        logger.info("OCR'd %s -> %s", source.name, out_path)
        return out_path

    def ocr_directory(self, src_dir: str | Path) -> list[Path]:
        src_dir = Path(src_dir)
        outputs: list[Path] = []
        supported = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        for path in sorted(src_dir.iterdir()):
            if path.suffix.lower() in supported:
                try:
                    outputs.append(self.ocr_file(path))
                except Exception as exc:
                    logger.exception("Failed to OCR %s: %s", path, exc)
        return outputs

    # ------------------------------------------------------------------
    # Per-format rendering -> VLM

    def _ocr_pdf(self, path: Path) -> str:
        page_pngs = _render_pdf_pages(path, self.dpi)
        chunks: list[str] = []
        for i, png in enumerate(page_pngs, start=1):
            md = self.vlm.transcribe(png)
            chunks.append(f"## Page {i}\n\n{md}\n")
        return "\n".join(chunks)

    def _ocr_image(self, path: Path) -> str:
        png = path.read_bytes() if path.suffix.lower() == ".png" else _to_png(path)
        return self.vlm.transcribe(png)

    def _ocr_docx(self, path: Path) -> str:
        # Render the docx to PDF first so the VLM sees the same layout
        # signatures, stamps, tables — that legal documents rely on.
        pdf_path = _libreoffice_convert(path, target="pdf")
        return self._ocr_pdf(pdf_path)

    def _ocr_doc(self, path: Path) -> str:
        pdf_path = _libreoffice_convert(path, target="pdf")
        return self._ocr_pdf(pdf_path)


# ---------------------------------------------------------------------------
# Helpers

def _render_pdf_pages(path: Path, dpi: int) -> list[bytes]:
    """Render each PDF page to PNG bytes.

    Prefers pypdfium2 (bundles its own PDFium binary, no system poppler needed).
    Falls back to pdf2image only if pypdfium2 is unavailable.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        from pdf2image import convert_from_path
        return [_pil_to_png_bytes(p) for p in convert_from_path(str(path), dpi=dpi)]

    pdf = pdfium.PdfDocument(str(path))
    try:
        scale = dpi / 72.0
        return [_pil_to_png_bytes(page.render(scale=scale).to_pil()) for page in pdf]
    finally:
        pdf.close()


def _pil_to_png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _to_png(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()


def _libreoffice_convert(path: Path, target: str) -> Path:
    out_dir = path.parent
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", target,
             "--outdir", str(out_dir), str(path)],
            check=True, capture_output=True, timeout=180,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"libreoffice is required to convert {path.suffix} files: {exc}"
        ) from exc
    return path.with_suffix(f".{target}")


def ocr_files(paths: Iterable[str | Path], output_dir: str | Path, **kwargs) -> list[Path]:
    engine = OCREngine(output_dir=output_dir, **kwargs)
    return [engine.ocr_file(p) for p in paths]
