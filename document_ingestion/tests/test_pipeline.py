"""Smoke tests that exercise cleaner -> extractor -> store -> timeline.

OCR and the VLM are not exercised against a live Ollama here; the OCR
module is tested via a fake VLMClient, and event extraction is forced
down the regex path (use_llm=False) so tests run with no model present.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from document_ingestion.cleaner import clean_markdown, clean_file
from document_ingestion.database import EventStore
from document_ingestion.event_extractor import Event, extract_events
from document_ingestion.ocr import OCREngine, VLMClient
from document_ingestion.timeline import generate_timeline, render_markdown


def test_clean_markdown_removes_page_numbers_and_dehyphenates():
    raw = "This is a sen-\ntence.\n\n\n12\n  trailing\n"
    out = clean_markdown(raw)
    assert "sen-\ntence" not in out
    assert "sentence" in out
    assert "\n12\n" not in out
    assert "\n\n\n" not in out


def test_regex_extractor_finds_dates(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text(
        "On 2024-01-15 the court denied the motion. "
        "The filing was made on March 3, 2024. "
        "No date here.\n",
        encoding="utf-8",
    )
    events = extract_events(md, use_llm=False)
    dates = sorted(e.date for e in events)
    assert dates == ["2024-01-15", "2024-03-03"]
    assert all(e.source_document == "doc.md" for e in events)
    assert all(e.event_id.startswith("eid") for e in events)


def test_event_store_roundtrip_and_rollup(tmp_path: Path):
    store = EventStore(tmp_path / "test.db")
    store.add_events([
        Event("eid001", "2024-01-15", "Motion denied.", "order1.pdf"),
        Event("eid002", "2024-01-15", "Hearing scheduled.", "order2.pdf"),
        Event("eid003", "2024-02-01", "Plea entered.", "order1.pdf"),
    ])

    dates = store.all_dates()
    assert [d["date"] for d in dates] == ["2024-01-15", "2024-02-01"]
    jan = next(d for d in dates if d["date"] == "2024-01-15")
    assert set(jan["event_ids"].split(",")) == {"eid001", "eid002"}
    assert set(jan["source_documents"].split(",")) == {"order1.pdf", "order2.pdf"}

    feb = store.events_on("2024-02-01")
    assert len(feb) == 1 and feb[0]["event_id"] == "eid003"


def test_export_ledger_lists_all_tables(tmp_path: Path):
    store = EventStore(tmp_path / "test.db")
    store.add_events([
        Event("eid001", "2024-01-15", "Motion denied.", "order1.pdf"),
        Event("eid002", "2024-01-15", "Hearing scheduled.", "order2.pdf"),
    ])
    ledger = store.export_ledger()

    assert ledger["ledger_type"] == "evidence_and_event_tracking"
    assert ledger["table_count"] == 4
    assert set(ledger["tables"]) == {"dates", "events", "case_actors", "event_actors"}

    events = ledger["tables"]["events"]
    assert events["row_count"] == 2
    assert {c["name"] for c in events["columns"]} >= {
        "event_id", "date", "action", "party", "court",
        "source_documents", "page_reference", "excerpt",
    }
    pk = [c["name"] for c in events["columns"] if c["primary_key"]]
    assert pk == ["event_id"]
    assert {r["event_id"] for r in events["rows"]} == {"eid001", "eid002"}


def test_person_name_filter_rejects_roles_and_institutions():
    from document_ingestion.entity_resolver import is_person_name
    for keep in ["Robert Brown", "Jane Smith", "Lauren Davis",
                 "Hon. Sarah Johnson"]:
        assert is_person_name(keep), keep
    for drop in ["the Court", "A parent", "both parents", "the parties",
                 "State of New York", "District Court", "Each party",
                 "the party incurring the same"]:
        assert not is_person_name(drop), drop


def test_actor_registry_captures_named_persons(tmp_path: Path):
    store = EventStore(tmp_path / "test.db")
    ev = Event("eid001", "2024-06-18", "Trial held.", "order.pdf")
    ev.party = "Robert Brown"
    ev.other_actors = "Jane Smith; Hon. Sarah Johnson"
    store.add_events([ev])
    actors = {a["name"]: a for a in store.all_actors()}
    assert len(actors) >= 3


def test_timeline_is_chronological(tmp_path: Path):
    store = EventStore(tmp_path / "test.db")
    store.add_events([
        Event("eid002", "2024-02-01", "Plea entered.", "order1.pdf"),
        Event("eid001", "2024-01-15", "Motion denied.", "order2.pdf"),
    ])
    entries = generate_timeline(store)
    assert [e.date for e in entries] == ["2024-01-15", "2024-02-01"]
    md = render_markdown(entries)
    assert md.index("2024-01-15") < md.index("2024-02-01")


def test_ocr_engine_uses_provided_vlm(tmp_path: Path):
    class StubVLM:
        def transcribe(self, image_png: bytes) -> str:
            return "stub markdown"

    image = tmp_path / "page.png"
    image.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c63f8cfc0000000030001011a3a8b8d"
        "0000000049454e44ae426082"
    ))

    engine = OCREngine(output_dir=tmp_path / "out", vlm=StubVLM())
    out = engine.ocr_file(image)
    text = out.read_text(encoding="utf-8")
    assert "stub markdown" in text
    assert "page.png" in text


def test_ollama_vision_client_sends_image_to_local_endpoint():
    from document_ingestion.ocr import OllamaVisionClient

    captured = {}

    class FakeOllamaClient:
        def chat(self, prompt, model, images_png=None, options=None):
            captured["prompt"] = prompt
            captured["model"] = model
            captured["n_images"] = len(images_png or [])
            return "  transcribed page  "

    vlm = OllamaVisionClient(model="llava", client=FakeOllamaClient())
    result = vlm.transcribe(b"\x89PNG-fake")

    assert result == "transcribed page"
    assert captured["model"] == "llava"
    assert captured["n_images"] == 1
    assert "OCR engine" in captured["prompt"]


def test_extract_events_falls_back_to_regex_when_ollama_down(tmp_path: Path, monkeypatch):
    from document_ingestion import config

    monkeypatch.setattr(config, "is_available", lambda *a, **k: False)
    md = tmp_path / "doc.md"
    md.write_text("Filed on 2024-05-09 in district court.\n", encoding="utf-8")

    events = extract_events(md, use_llm=True, backend="ollama")
    assert [e.date for e in events] == ["2024-05-09"]
