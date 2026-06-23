"""Stage 2: clean raw OCR markdown.

Removes common OCR artefacts (page numbers, repeated headers, stray
whitespace, broken hyphenation) and normalizes the file so the event
extractor sees readable prose.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Common OCR character confusions inside words. Kept conservative so we
# don't corrupt legitimate text.
_CHAR_FIXES = [
    (re.compile(r"(?<=[a-z])\|(?=[a-z])"), "l"),
    (re.compile(r"\s+,"), ","),
    (re.compile(r"\s+\."), "."),
]

_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s+)?\d{1,4}\s*$", re.IGNORECASE)
_DEHYPHENATE = re.compile(r"(\w+)-\n(\w+)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def clean_markdown(text: str) -> str:
    text = _DEHYPHENATE.sub(r"\1\2", text)
    for pattern, replacement in _CHAR_FIXES:
        text = pattern.sub(replacement, text)

    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not _PAGE_NUMBER_LINE.match(ln)]

    text = "\n".join(lines)
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip() + "\n"


def clean_file(raw_path: str | Path, output_dir: str | Path | None = None) -> Path:
    raw_path = Path(raw_path)
    raw = raw_path.read_text(encoding="utf-8")
    cleaned = clean_markdown(raw)

    out_dir = Path(output_dir) if output_dir else raw_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = raw_path.name.replace(".raw.md", ".md")
    if out_name == raw_path.name:
        out_name = raw_path.stem + ".clean.md"
    out_path = out_dir / out_name
    out_path.write_text(cleaned, encoding="utf-8")
    logger.info("Cleaned %s -> %s", raw_path.name, out_path)
    return out_path
