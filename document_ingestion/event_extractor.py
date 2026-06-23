"""Stage 3: extract date-anchored events from cleaned markdown.

Uses a local Ollama text model when one is reachable, falling back to a
regex-based extractor so the pipeline still runs with no model present.
No cloud calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from . import config
from .ollama_client import OllamaClient, OllamaError

# Event-chunk window size (chars). Smaller chunks => smaller model responses,
# which helps dodge 503s/timeouts on dense or sensitive documents. Override
# per-run with DOCING_CHUNK_SIZE (e.g. =6000 for the 503-prone DV order).
_CHUNK_SIZE = int(os.getenv("DOCING_CHUNK_SIZE", "24000"))

logger = logging.getLogger(__name__)


UNDETERMINED = "undetermined"


@dataclass
class Event:
    """A single evidence-backed timeline entry.

    Field order keeps the legacy positional constructor
    ``Event(event_id, date, action, source_document)`` working; the
    remaining structured/evidence fields default to ``"undetermined"``
    so a partial extraction still produces a valid record.
    """

    event_id: str                       # evidence ID / ledger reference
    date: str                           # ISO yyyy-mm-dd or "undetermined"
    action: str                         # event / action
    source_document: str
    party: str = UNDETERMINED           # primary party or actor
    court: str = UNDETERMINED           # court / jurisdiction
    page_reference: str = UNDETERMINED  # page or chunk reference
    excerpt: str = ""                   # verbatim excerpt relied upon
    other_actors: str = ""              # other persons named in excerpt (semicolon-sep)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regex fallback extractor

_DATE_PATTERNS = [
    # ISO yyyy-mm-dd
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("y", "m", "d")),
    # d-m-yy / d-m-yyyy
    (re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b"), ("d", "m", "y")),
    # 12 January 2024 / January 12, 2024
    (re.compile(r"\b(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})\b"), ("d", "mon", "y")),
    (re.compile(r"\b([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})\b"), ("mon", "d", "y")),
]

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _normalize_date(groups: tuple[str, ...], order: tuple[str, ...]) -> str | None:
    mapping = dict(zip(order, groups))
    try:
        year = int(mapping["y"])
        if year < 100:
            year += 2000 if year < 50 else 1900
        if "mon" in mapping:
            month = _MONTHS.get(mapping["mon"].lower())
            if month is None:
                return None
        else:
            month = int(mapping["m"])
        day = int(mapping["d"])
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except (ValueError, KeyError):
        return None


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


# Page / chunk reference recovery -------------------------------------------

_PAGE_HEADER = re.compile(r"(?im)^\s*#*\s*page\s+(\d{1,4})\b")


def _page_index(markdown: str) -> list[tuple[int, str]]:
    """Return (char_offset, "Page N") for every page marker, in order."""
    return [(m.start(), f"Page {m.group(1)}") for m in _PAGE_HEADER.finditer(markdown)]


def _page_for(pages: list[tuple[int, str]], offset: int) -> str:
    label = UNDETERMINED
    for start, lab in pages:
        if start <= offset:
            label = lab
        else:
            break
    return label


# Lightweight court / jurisdiction heuristic --------------------------------

_COURT_PATTERNS = [
    (re.compile(r"district court", re.I), "District Court"),
    (re.compile(r"circuit court", re.I), "Circuit Court"),
    (re.compile(r"family court", re.I), "Family Court"),
    (re.compile(r"superior court", re.I), "Superior Court"),
    (re.compile(r"supreme court", re.I), "Supreme Court"),
    (re.compile(r"appeals court|court of appeals", re.I), "Court of Appeals"),
]


def _guess_court(*texts: str) -> str:
    blob = " ".join(t for t in texts if t)
    for pattern, label in _COURT_PATTERNS:
        if pattern.search(blob):
            return label
    return UNDETERMINED


def _regex_extract(markdown: str, source: str) -> list[Event]:
    events: list[Event] = []
    pages = _page_index(markdown)
    cursor = 0
    for sentence in _split_sentences(markdown):
        pos = markdown.find(sentence, cursor)
        if pos >= 0:
            cursor = pos + len(sentence)
        page_ref = _page_for(pages, pos if pos >= 0 else cursor)
        for pattern, order in _DATE_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            iso = _normalize_date(match.groups(), order)
            if not iso:
                continue
            events.append(Event(
                event_id=_make_event_id(),
                date=iso,
                action=sentence,
                source_document=source,
                court=_guess_court(sentence, source),
                page_reference=page_ref,
                excerpt=sentence,
            ))
            break
    return events


def _make_event_id() -> str:
    return "eid" + uuid.uuid4().hex[:9]


# ---------------------------------------------------------------------------
# Ollama-backed extractor (local)

_EXTRACTION_PROMPT = """You are building an evidence-backed legal timeline from a case document.

The document markdown may contain "## Page N" headers — use the nearest
preceding one as the page reference for an event.

Extract EVERY discrete event or action — be exhaustive and inclusive. Include
minor and procedural items: filings, motions, notices, orders, hearings,
service of process, certificates of service, emails/letters, scheduling,
deadlines, appearances, signatures, and exhibits. Err strongly toward
INCLUSION: when in doubt, include the event. Do not skip something because it
seems minor or routine.

Return ONLY JSON: a list of objects, each with these keys (use the string
"undetermined" for any field you cannot ground in the text — never guess):
  - "date": ISO date (yyyy-mm-dd), or "undetermined" if no explicit/derivable date.
  - "action": concise factual description of the event or action (one sentence).
  - "party": the SPECIFIC person who performed/is subject to the action — name
             the actual attorney, judge, clerk, or party by name when the text
             gives it. Only use a generic role ("the Court") if no name is present.
  - "other_actors": a list of EVERY other person named in the excerpt (attorneys,
             judges, clerks, parties, witnesses), by full name. [] if none.
  - "court": the court or jurisdiction involved.
  - "page": the page reference, e.g. "Page 2", from the nearest "## Page N" header.
  - "excerpt": a SHORT EXACT-VERBATIM quote (<=240 chars) copied character-for-
               character from the document. It MUST be a single contiguous span
               that appears verbatim in the text. DO NOT paraphrase, summarize,
               join distant passages, fix typos/OCR errors, or use "..."/ellipses.
               If you cannot copy a clean contiguous quote, omit the event.

Do not invent events. Every event MUST be supported by its verbatim excerpt —
this grounding requirement is absolute and is never relaxed for the sake of
including more events.

Document:
---
{markdown}
---
JSON:"""


_ROSTER_PROMPT = """List EVERY person named in the document below — attorneys,
judges, magistrates, clerks, parties, witnesses, social workers, officers.

Return ONLY JSON: a list of objects, each with:
  - "name": the person's full name exactly as it best appears.
  - "role": their role if stated (e.g. "Petitioner", "Judge", "counsel for
            Respondent", "clerk"), else "undetermined".
  - "mention": a SHORT EXACT-VERBATIM quote (<=160 chars) from the document that
               names this person. Copy it character-for-character; no ellipses.

Include a person only if their name actually appears in the text and you can
quote it verbatim. Do not invent or infer names that are not written.

Document:
---
{markdown}
---
JSON:"""


# Llama-optimized prompts: Llama-3 instruction-follows best with a forceful
# inclusive directive + a worked few-shot example anchoring the granularity, and
# an explicit "JSON array only, no prose/fences" instruction. The grounding
# (verbatim-excerpt) requirement is kept identical so the comparison stays fair.
_LLAMA_EXTRACTION_PROMPT = """You are a meticulous legal analyst extracting a COMPLETE timeline from a case document. Your objective is MAXIMUM RECALL: extract EVERY discrete event, action, filing, motion, order, ruling, hearing, service of process, notice, certificate, email, deadline, appearance, signature, and statement — however minor or routine. One page typically yields MANY events. Never summarize or merge: if a sentence describes two actions, output two separate events.

The document may contain "## Page N" headers — use the nearest preceding one as the page reference.

Output ONLY a JSON array — no prose, no explanation, no markdown code fences. Each object has these keys (use the string "undetermined" for any field you cannot ground in the text):
  - "date": ISO yyyy-mm-dd, or "undetermined".
  - "action": one short sentence describing the single event.
  - "party": the SPECIFIC named person who acted (attorney, judge, clerk, party). Use a generic role like "the Court" ONLY if no name is present.
  - "other_actors": a JSON list of EVERY other person named in the excerpt; [] if none.
  - "court": the court or jurisdiction, or "undetermined".
  - "page": e.g. "Page 2".
  - "excerpt": a SHORT EXACT-VERBATIM quote (<=240 chars) copied character-for-character from the document. It MUST appear verbatim and contiguously in the text. NEVER paraphrase, summarize, join separate passages, fix typos/OCR errors, or use "..."/ellipses. If you cannot copy a clean contiguous quote, omit that event.

EXAMPLE document:
---
## Page 1
On June 18, 2024, the Court held a final hearing. The Plaintiff, Jane Smith, appeared with counsel Robert Brown. The Defendant failed to appear. The Court entered a Default Judgment.
---
EXAMPLE output (note: four separate events from four sentences):
[
  {{"date":"2024-06-18","action":"The Court held a final hearing.","party":"the Court","other_actors":[],"court":"undetermined","page":"Page 1","excerpt":"On June 18, 2024, the Court held a final hearing."}},
  {{"date":"2024-06-18","action":"Plaintiff Jane Smith appeared with counsel Robert Brown.","party":"Jane Smith","other_actors":["Robert Brown"],"court":"undetermined","page":"Page 1","excerpt":"The Plaintiff, Jane Smith, appeared with counsel Robert Brown."}},
  {{"date":"2024-06-18","action":"The Defendant failed to appear.","party":"Defendant","other_actors":[],"court":"undetermined","page":"Page 1","excerpt":"The Defendant failed to appear."}},
  {{"date":"2024-06-18","action":"The Court entered a Default Judgment.","party":"the Court","other_actors":[],"court":"undetermined","page":"Page 1","excerpt":"The Court entered a Default Judgment."}}
]

Now be exhaustive and extract from the document below.
Document:
---
{markdown}
---
JSON array:"""


_LLAMA_ROSTER_PROMPT = """List EVERY person named in the document below — attorneys, judges, magistrates, clerks, parties, witnesses, social workers, police officers, guardians. Be exhaustive; include everyone named even once.

Output ONLY a JSON array — no prose, no markdown fences. Each object:
  - "name": the person's full name exactly as it appears.
  - "role": their role if stated (e.g. "Petitioner", "Judge", "counsel for Respondent", "clerk"), else "undetermined".
  - "mention": a SHORT EXACT-VERBATIM quote (<=160 chars) from the document naming this person. Copy it character-for-character; no ellipses.

Include a person ONLY if their name actually appears in the text and you can quote it verbatim. Do not invent or infer names.

Document:
---
{markdown}
---
JSON array:"""


def _extraction_prompt(backend: str) -> str:
    return _LLAMA_EXTRACTION_PROMPT if backend in ("openrouter", "llama") else _EXTRACTION_PROMPT


def _roster_prompt(backend: str) -> str:
    return _LLAMA_ROSTER_PROMPT if backend in ("openrouter", "llama") else _ROSTER_PROMPT


def _ollama_extract(markdown: str, source: str, model: str) -> list[Event]:
    client = OllamaClient()
    try:
        raw = client.chat(
            prompt=_EXTRACTION_PROMPT.format(markdown=markdown[:60_000]),
            model=model,
            options={"temperature": 0},
        )
    except OllamaError as exc:
        logger.warning("Ollama extract failed for %s (%s); using regex", source, exc)
        return _regex_extract(markdown, source)

    return _json_to_events(raw, source, markdown)


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for tolerant substring matching.

    OCR/markdown introduces irregular spacing and line breaks, so the
    grounding gate compares on normalized text rather than raw bytes — but it
    still requires the excerpt to be a single contiguous span of the source
    (no ellipses, no joined passages).
    """
    return " ".join(text.split()).lower()


def _is_grounded(excerpt: str, source_norm: str) -> bool:
    ex = _normalize(excerpt)
    if not ex:
        return False
    if "..." in excerpt or "…" in excerpt:
        return False
    return ex in source_norm


def _json_to_events(raw: str, source: str, source_text: str = "") -> list[Event]:
    """Parse a model's JSON reply into structured, evidence-backed Events.

    Each event passes a strict grounding gate: its excerpt must appear as a
    verbatim contiguous span of the source document (whitespace-normalized).
    Ungrounded events are dropped — a timeline entry the model cannot quote
    from the record is not trustworthy.
    """
    json_text = _extract_json_block(raw)
    if not json_text:
        logger.warning("LLM returned no parseable JSON for %s", source)
        return []

    try:
        items = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON for %s: %s", source, exc)
        return []

    source_norm = _normalize(source_text)
    events: list[Event] = []
    dropped = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        action = (item.get("action") or "").strip()
        if not action:
            continue
        excerpt = (item.get("excerpt") or "").strip()[:240]
        if source_norm and not _is_grounded(excerpt, source_norm):
            dropped += 1
            continue
        events.append(Event(
            event_id=_make_event_id(),
            date=_clean_date(item.get("date")),
            action=action,
            source_document=source,
            party=_clean_field(item.get("party")),
            court=_clean_field(item.get("court")) or _guess_court(action, source),
            page_reference=_clean_field(item.get("page")),
            excerpt=excerpt,
            other_actors=_clean_actor_list(item.get("other_actors")),
        ))
    if dropped:
        logger.info("Grounding gate dropped %d ungrounded event(s) for %s",
                    dropped, source)
    return events


# ---------------------------------------------------------------------------
# Cloud LLM extractor (Gemini or OpenRouter/Llama — same pipeline & grounding)

def _llm_client(backend: str, model: str | None):
    """Return an LLM client + its error type for the requested backend.

    All clients expose the same `.generate(prompt, temperature=...)` surface,
    so only the model changes — chunking, grounding, parsing, and rendering are
    identical across backends.
    """
    if backend == "bedrock":
        from .bedrock_client import BedrockClient, BedrockError, DEFAULT_MODEL
        return BedrockClient(model=model or DEFAULT_MODEL), BedrockError
    if backend in ("openrouter", "llama"):
        from .openrouter_client import OpenRouterClient, OpenRouterError, DEFAULT_MODEL
        return OpenRouterClient(model=model or DEFAULT_MODEL), OpenRouterError
    from .gemini_client import GeminiClient, GeminiError, DEFAULT_MODEL
    return GeminiClient(model=model or DEFAULT_MODEL), GeminiError


def _chunk_markdown(text: str, size: int | None = None, overlap: int = 2_000) -> list[str]:
    """Split a long document into overlapping windows.

    Chunking keeps the model focused on a smaller span so minor/procedural
    events aren't lost to whole-document summarization (recall lever). Splits
    are preferentially made at paragraph breaks near the window boundary.
    """
    size = size or _CHUNK_SIZE
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            brk = text.rfind("\n\n", start + size - overlap, end)
            if brk != -1 and brk > start:
                end = brk
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _llm_extract(markdown: str, source: str, backend: str,
                 model: str | None) -> list[Event]:
    """Extract events from every chunk; ground each against the FULL document.

    Grounding is enforced against the whole markdown (not just the chunk) so an
    excerpt is accepted only if it is a verbatim span of the real source — no
    fabrication, even though we extract chunk-by-chunk for higher recall. Works
    identically for Gemini and OpenRouter/Llama backends.
    """
    client, err_type = _llm_client(backend, model)
    chunks = _chunk_markdown(markdown)
    events: list[Event] = []
    for i, chunk in enumerate(chunks):
        try:
            raw = client.generate(
                _extraction_prompt(backend).format(markdown=chunk), temperature=0.0
            )
        except err_type as exc:
            logger.error("%s extract failed for %s chunk %d/%d: %s",
                         backend, source, i + 1, len(chunks), exc)
            raise
        events.extend(_json_to_events(raw, source, markdown))
    deduped = _dedup_events(events)
    if len(chunks) > 1:
        logger.info("%s [%s]: %d chunks -> %d events (%d after dedup)",
                    source, backend, len(chunks), len(events), len(deduped))
    return deduped


def _dedup_events(events: list[Event]) -> list[Event]:
    """Collapse chunk-overlap duplicates (same date + normalized excerpt)."""
    seen: set[tuple[str, str]] = set()
    out: list[Event] = []
    for ev in events:
        key = (ev.date, _normalize(ev.excerpt)[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def extract_roster(markdown_path: str | Path, model: str | None = None,
                   backend: str = "gemini") -> list[dict]:
    """Extract every person named in a document, each grounded by a verbatim mention.

    Returns a list of {name, role, mention}. Names whose mention cannot be
    verified verbatim in the source are dropped (no invented people).
    """
    path = Path(markdown_path)
    text = path.read_text(encoding="utf-8")
    source_norm = _normalize(text)
    client, err_type = _llm_client(backend, model)
    roster: list[dict] = []
    # Single whole-document call (not per-chunk) to economize on quota; the
    # model sees the full doc so name coverage stays high.
    try:
        raw = client.generate(_roster_prompt(backend).format(markdown=text[:200_000]),
                              temperature=0.0)
    except err_type as exc:
        logger.warning("roster extract failed for %s: %s", path.name, exc)
        return roster
    block = _extract_json_block(raw)
    if block:
        try:
            items = json.loads(block)
        except json.JSONDecodeError:
            items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or "").strip()
            mention = (it.get("mention") or "").strip()
            if not name or not mention:
                continue
            if _normalize(mention) not in source_norm:   # grounding gate for names
                continue
            roster.append({
                "name": name,
                "role": (it.get("role") or "undetermined").strip() or "undetermined",
                "mention": mention[:160],
                "source_document": path.name,
            })
    return roster


def _clean_field(value) -> str:
    text = (str(value).strip() if value is not None else "")
    return text if text else UNDETERMINED


def _clean_actor_list(value) -> str:
    """Normalize the LLM's other_actors (list or string) to 'a; b; c'."""
    if isinstance(value, list):
        names = [str(v).strip() for v in value if str(v).strip()]
    elif value:
        names = [s.strip() for s in re.split(r"[;,]", str(value)) if s.strip()]
    else:
        names = []
    # drop obvious non-persons / placeholders
    names = [n for n in names if n.lower() not in ("undetermined", "n/a", "none")]
    return "; ".join(dict.fromkeys(names))


def _clean_date(value) -> str:
    text = (str(value).strip() if value is not None else "")
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return text
    except ValueError:
        return UNDETERMINED


def _extract_json_block(text: str) -> str | None:
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    bracket = re.search(r"(\[.*\])", text, re.DOTALL)
    return bracket.group(1) if bracket else None


# ---------------------------------------------------------------------------
# Public API

def extract_events(
    markdown_path: str | Path,
    use_llm: bool = True,
    model: str | None = None,
    backend: str = "gemini",
) -> list[Event]:
    """Extract structured, evidence-backed events from a cleaned markdown file.

    Backends:
      - "gemini" (default): free-tier Google Gemini.
      - "openrouter" / "llama": OpenRouter (Llama-3-70B-Instruct by default).
        Both cloud backends run the identical chunking + grounding gate, so they
        differ only by model — for a fair Gemini-vs-Llama comparison.
      - "ollama": local Ollama text model, regex fallback if unreachable.
      - "regex":  offline regex extractor only.
    A cloud backend raises on error (no silent fallback) so failures are explicit.
    """
    path = Path(markdown_path)
    text = path.read_text(encoding="utf-8")

    if not use_llm:
        backend = "regex"

    if backend == "bedrock":
        from .bedrock_client import is_available as bedrock_available, BedrockError

        if not bedrock_available():
            raise BedrockError(
                "AWS credentials not configured; set up boto3 credentials to use the bedrock backend"
            )
        return _llm_extract(text, source=path.name, backend="bedrock", model=model)

    if backend == "gemini":
        from .gemini_client import is_available as gemini_available, GeminiError

        if not gemini_available():
            raise GeminiError(
                "GEMINI_API_KEY is not set; export it to use the gemini backend"
            )
        return _llm_extract(text, source=path.name, backend="gemini", model=model)

    if backend in ("openrouter", "llama"):
        from .openrouter_client import is_available as or_available, OpenRouterError

        if not or_available():
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set; export it to use the openrouter backend"
            )
        return _llm_extract(text, source=path.name, backend="openrouter", model=model)

    if backend == "ollama" and use_llm and config.is_available():
        model = model or config.load_config().text_model
        return _ollama_extract(text, source=path.name, model=model)
    return _regex_extract(text, source=path.name)


def extract_events_from_many(
    paths: Iterable[str | Path], **kwargs
) -> list[Event]:
    out: list[Event] = []
    for p in paths:
        out.extend(extract_events(p, **kwargs))
    return out
