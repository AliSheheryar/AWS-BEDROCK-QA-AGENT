"""End-to-end orchestration: PDF/DOC -> Markdown -> Events -> SQLite (-> Parquet -> S3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .cleaner import clean_file
from .database import EventStore
from .event_extractor import Event, extract_events
from .ocr import OCREngine, VLMClient

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    raw_markdown: list[Path] = field(default_factory=list)
    clean_markdown: list[Path] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    parquet_files: dict[str, Path] = field(default_factory=dict)


class IngestionPipeline:
    """Run the four pipeline stages on a folder of source documents.

    Layout under `workdir`:
        raw_md/      -- VLM transcription per source file
        clean_md/    -- post-OCR cleanup
        case.db      -- SQLite store
        parquet/     -- exported Parquet files (for Athena)
    """

    def __init__(
        self,
        workdir: str | Path,
        db_path: str | Path | None = None,
        vlm: VLMClient | None = None,
        llm_model: str | None = None,
        use_llm_for_events: bool = True,
        event_backend: str = "gemini",
    ):
        self.workdir = Path(workdir)
        self.raw_dir = self.workdir / "raw_md"
        self.clean_dir = self.workdir / "clean_md"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        self.ocr = OCREngine(output_dir=self.raw_dir, vlm=vlm)
        self.db_path = db_path or self.workdir / "case.db"
        self.store = EventStore(self.db_path)
        self.llm_model = llm_model
        self.use_llm_for_events = use_llm_for_events
        self.event_backend = event_backend

    def ingest_directory(self, src_dir: str | Path) -> IngestionResult:
        result = IngestionResult()
        result.raw_markdown = self.ocr.ocr_directory(src_dir)
        for raw in result.raw_markdown:
            cleaned = clean_file(raw, self.clean_dir)
            result.clean_markdown.append(cleaned)
            events = extract_events(
                cleaned,
                use_llm=self.use_llm_for_events,
                model=self.llm_model,
                backend=self.event_backend,
            )
            result.events.extend(events)
            self.store.add_events(events)
        logger.info(
            "Ingested %d documents, extracted %d events",
            len(result.raw_markdown), len(result.events),
        )
        return result

    def ingest_file(self, source: str | Path) -> IngestionResult:
        result = IngestionResult()
        raw = self.ocr.ocr_file(source)
        cleaned = clean_file(raw, self.clean_dir)
        events = extract_events(
            cleaned,
            use_llm=self.use_llm_for_events,
            model=self.llm_model,
            backend=self.event_backend,
        )
        self.store.add_events(events)
        result.raw_markdown = [raw]
        result.clean_markdown = [cleaned]
        result.events = events
        return result

    def export_to_parquet(self) -> dict[str, Path]:
        """Export SQLite tables to Parquet for Athena."""
        from .athena_store import export_sqlite_to_parquet
        parquet_dir = self.workdir / "parquet"
        return export_sqlite_to_parquet(self.db_path, parquet_dir)

    def export_and_upload_parquet(self, s3_bucket: str) -> dict[str, str]:
        """Export to Parquet and upload to S3 for Athena."""
        from .athena_store import export_sqlite_to_parquet, upload_parquet_to_s3
        parquet_dir = self.workdir / "parquet"
        exported = export_sqlite_to_parquet(self.db_path, parquet_dir)
        return upload_parquet_to_s3(exported, s3_bucket)
