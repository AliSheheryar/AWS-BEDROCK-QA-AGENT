"""Command-line entrypoint for the ingestion pipeline.

Usage:
    python -m document_ingestion.cli ingest <src_dir> --workdir ./work
    python -m document_ingestion.cli ingest <src_dir> --workdir ./work --ocr-backend bedrock --event-backend bedrock
    python -m document_ingestion.cli timeline --workdir ./work [--out timeline.md]
    python -m document_ingestion.cli query "What motions were filed in 2024?"
    python -m document_ingestion.cli athena-query "how many events per court?"
    python -m document_ingestion.cli export-parquet --workdir ./work --s3-bucket my-bucket
    python -m document_ingestion.cli s3-sync --workdir ./work --s3-bucket my-bucket
    python -m document_ingestion.cli s3-ingest --s3-bucket my-bucket --workdir ./work
    python -m document_ingestion.cli kb-sync --data-source-id DS_ID
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .database import EventStore
from .pipeline import IngestionPipeline
from .timeline import generate_timeline, render_json, render_markdown


def _ingest(args: argparse.Namespace) -> int:
    vlm = None
    if args.ocr_backend == "anthropic":
        from .anthropic_vision import anthropic_vlm
        vlm = anthropic_vlm()
    elif args.ocr_backend == "bedrock":
        from .bedrock_vision import bedrock_vlm
        vlm = bedrock_vlm()

    pipeline = IngestionPipeline(
        workdir=args.workdir,
        vlm=vlm,
        use_llm_for_events=not args.no_llm,
        event_backend=args.event_backend,
    )
    result = pipeline.ingest_directory(args.src_dir)
    print(f"Ingested {len(result.raw_markdown)} docs, {len(result.events)} events")

    if getattr(args, "s3_bucket", None):
        from .s3_storage import S3Storage
        s3 = S3Storage(bucket=args.s3_bucket)
        uploaded = s3.sync_workdir_to_s3(args.workdir)
        print(f"Synced to S3: {uploaded}")
    return 0


def _timeline(args: argparse.Namespace) -> int:
    db_path = Path(args.workdir) / "case.db"
    store = EventStore(db_path)
    entries = generate_timeline(store)
    output = render_markdown(entries) if args.format == "markdown" else render_json(entries)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote {len(entries)} events to {args.out}")
    else:
        sys.stdout.write(output)

    if getattr(args, "s3_bucket", None) and args.out:
        from .s3_storage import S3Storage
        s3 = S3Storage(bucket=args.s3_bucket)
        s3.upload_timeline(output, Path(args.out).name)
        print(f"Uploaded timeline to S3")
    return 0


def _ledger(args: argparse.Namespace) -> int:
    db_path = Path(args.workdir) / "case.db"
    store = EventStore(db_path)
    ledger = store.export_ledger()
    output = json.dumps(ledger, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(
            f"Wrote evidence ledger ({ledger['table_count']} tables) to {args.out}"
        )
    else:
        sys.stdout.write(output)
    return 0


def _query(args: argparse.Namespace) -> int:
    from .rag_query import create_rag_engine

    engine = create_rag_engine(
        knowledge_base_id=getattr(args, "kb_id", None),
        athena_database=getattr(args, "athena_db", None),
        athena_workgroup=getattr(args, "athena_workgroup", None),
        model_arn=getattr(args, "model_arn", None),
    )

    if args.interactive:
        print("RAG Query Engine (Bedrock KB + Athena). Type 'quit' to exit.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in ("quit", "exit", "q"):
                break
            result = engine.query(question)
            print(f"\n{result.answer}")
            print(f"\n[backend: {result.backend}, "
                  f"sources: {len(result.source_documents)}, "
                  f"model: {result.model}]\n")
    else:
        question = args.question
        result = engine.query(question)
        print(result.answer)
        if args.verbose:
            print(f"\n--- Metadata ---")
            print(f"Backend: {result.backend}")
            print(f"Sources: {', '.join(result.source_documents)}")
            print(f"Citations: {len(result.citations)}")
            print(f"Athena rows: {len(result.athena_data)}")
            print(f"Model: {result.model}")
    return 0


def _athena_query(args: argparse.Namespace) -> int:
    from .athena_store import AthenaEventStore

    store = AthenaEventStore(
        database=getattr(args, "athena_db", None),
        workgroup=getattr(args, "athena_workgroup", None),
    )
    result = store.custom_query(args.sql)
    if args.format == "json":
        print(json.dumps(result.rows, indent=2))
    else:
        if result.columns:
            print(" | ".join(result.columns))
            print("-" * 40)
        for row in result.rows:
            print(" | ".join(str(v) for v in row.values()))
    print(f"\n({result.row_count} rows, {result.execution_time_ms}ms)")
    return 0


def _export_parquet(args: argparse.Namespace) -> int:
    from .athena_store import export_sqlite_to_parquet, upload_parquet_to_s3

    db_path = Path(args.workdir) / "case.db"
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    output_dir = Path(args.workdir) / "parquet"
    exported = export_sqlite_to_parquet(db_path, output_dir)
    print(f"Exported {len(exported)} tables to Parquet:")
    for table, path in exported.items():
        print(f"  {table} -> {path}")

    if args.s3_bucket:
        uploaded = upload_parquet_to_s3(exported, args.s3_bucket)
        print(f"\nUploaded to S3:")
        for table, uri in uploaded.items():
            print(f"  {table} -> {uri}")
    return 0


def _s3_sync(args: argparse.Namespace) -> int:
    from .s3_storage import S3Storage

    s3 = S3Storage(bucket=args.s3_bucket)
    uploaded = s3.sync_workdir_to_s3(args.workdir)
    print(f"Synced to s3://{args.s3_bucket}/")
    print(f"  Raw markdown: {len(uploaded['raw_markdown'])} files")
    print(f"  Clean markdown: {len(uploaded['clean_markdown'])} files")
    print(f"  Database: {uploaded['database'] or 'not found'}")
    return 0


def _s3_ingest(args: argparse.Namespace) -> int:
    """Download source docs from S3, run the pipeline, upload results back."""
    from .s3_storage import S3Storage

    s3 = S3Storage(bucket=args.s3_bucket)
    tmpdir, doc_paths = s3.download_source_to_tempdir()
    if not doc_paths:
        print("No source documents found in S3 bucket")
        return 1

    print(f"Downloaded {len(doc_paths)} documents from S3")

    vlm = None
    if args.ocr_backend == "bedrock":
        from .bedrock_vision import bedrock_vlm
        vlm = bedrock_vlm()
    elif args.ocr_backend == "anthropic":
        from .anthropic_vision import anthropic_vlm
        vlm = anthropic_vlm()

    pipeline = IngestionPipeline(
        workdir=args.workdir,
        vlm=vlm,
        use_llm_for_events=not args.no_llm,
        event_backend=args.event_backend,
    )
    result = pipeline.ingest_directory(tmpdir)
    print(f"Ingested {len(result.raw_markdown)} docs, {len(result.events)} events")

    uploaded = s3.sync_workdir_to_s3(args.workdir)
    print(f"Results synced back to S3: {uploaded}")

    if args.export_parquet:
        from .athena_store import export_sqlite_to_parquet, upload_parquet_to_s3
        db_path = Path(args.workdir) / "case.db"
        parquet_dir = Path(args.workdir) / "parquet"
        exported = export_sqlite_to_parquet(db_path, parquet_dir)
        upload_parquet_to_s3(exported, args.s3_bucket)
        print(f"Exported and uploaded {len(exported)} Parquet tables for Athena")
    return 0


def _kb_sync(args: argparse.Namespace) -> int:
    from .bedrock_knowledge_base import BedrockKnowledgeBase

    kb = BedrockKnowledgeBase(knowledge_base_id=getattr(args, "kb_id", None))
    job_id = kb.start_sync(args.data_source_id)
    print(f"Started Knowledge Base sync job: {job_id}")

    if args.wait:
        import time
        for _ in range(60):
            status = kb.get_sync_status(args.data_source_id, job_id)
            print(f"  Status: {status['status']}")
            if status["status"] in ("COMPLETE", "FAILED"):
                if status.get("stats"):
                    print(f"  Stats: {status['stats']}")
                break
            time.sleep(5)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="document_ingestion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="OCR + extract events from a folder")
    p_ingest.add_argument("src_dir", help="folder with .pdf/.doc/.docx files")
    p_ingest.add_argument("--workdir", default="./work")
    p_ingest.add_argument("--no-llm", action="store_true",
                          help="use regex-only event extraction")
    p_ingest.add_argument("--ocr-backend", choices=["ollama", "anthropic", "bedrock"],
                          default="ollama",
                          help="OCR backend: local Ollama, Anthropic Claude, or AWS Bedrock")
    p_ingest.add_argument("--event-backend",
                          choices=["gemini", "openrouter", "llama", "ollama", "bedrock", "regex"],
                          default="gemini",
                          help="Event extraction backend (default: gemini)")
    p_ingest.add_argument("--s3-bucket", default=None,
                          help="S3 bucket to sync results to after ingestion")
    p_ingest.set_defaults(func=_ingest)

    # --- timeline ---
    p_tl = sub.add_parser("timeline", help="render the chronological timeline")
    p_tl.add_argument("--workdir", default="./work")
    p_tl.add_argument("--format", choices=["json", "markdown"], default="json",
                      help="output format (default: json)")
    p_tl.add_argument("--out", help="write timeline to this file (default: stdout)")
    p_tl.add_argument("--s3-bucket", default=None,
                      help="also upload timeline to this S3 bucket")
    p_tl.set_defaults(func=_timeline)

    # --- ledger ---
    p_led = sub.add_parser("ledger",
                           help="export all DB tables as a JSON evidence ledger")
    p_led.add_argument("--workdir", default="./work")
    p_led.add_argument("--out", help="write JSON to this file (default: stdout)")
    p_led.set_defaults(func=_ledger)

    # --- query (Bedrock KB + Athena auto-routed) ---
    p_query = sub.add_parser("query",
                             help="Ask questions (auto-routes: KB for docs, Athena for analytics)")
    p_query.add_argument("question", nargs="?", default=None,
                         help="question to ask (omit for interactive mode)")
    p_query.add_argument("--kb-id", default=None,
                         help="Bedrock Knowledge Base ID (or set BEDROCK_KB_ID env)")
    p_query.add_argument("--athena-db", default=None,
                         help="Athena/Glue database name (or set ATHENA_DATABASE env)")
    p_query.add_argument("--athena-workgroup", default=None,
                         help="Athena workgroup (or set ATHENA_WORKGROUP env)")
    p_query.add_argument("--model-arn", default=None,
                         help="Bedrock model ARN override")
    p_query.add_argument("--interactive", "-i", action="store_true",
                         help="enter interactive query mode")
    p_query.add_argument("--verbose", "-v", action="store_true",
                         help="show metadata with answer")
    p_query.set_defaults(func=_query)

    # --- athena-query (direct SQL) ---
    p_aq = sub.add_parser("athena-query", help="Run a direct SQL query via Athena")
    p_aq.add_argument("sql", help="SQL query to execute")
    p_aq.add_argument("--athena-db", default=None)
    p_aq.add_argument("--athena-workgroup", default=None)
    p_aq.add_argument("--format", choices=["json", "table"], default="table")
    p_aq.set_defaults(func=_athena_query)

    # --- export-parquet ---
    p_ep = sub.add_parser("export-parquet",
                          help="Export SQLite to Parquet and optionally upload to S3")
    p_ep.add_argument("--workdir", default="./work")
    p_ep.add_argument("--s3-bucket", default=None,
                      help="Upload Parquet files to this S3 bucket for Athena")
    p_ep.set_defaults(func=_export_parquet)

    # --- s3-sync ---
    p_s3sync = sub.add_parser("s3-sync", help="Upload local workdir to S3")
    p_s3sync.add_argument("--workdir", default="./work")
    p_s3sync.add_argument("--s3-bucket", required=True, help="target S3 bucket")
    p_s3sync.set_defaults(func=_s3_sync)

    # --- s3-ingest ---
    p_s3i = sub.add_parser("s3-ingest",
                           help="Download docs from S3, ingest, upload results back")
    p_s3i.add_argument("--s3-bucket", required=True, help="S3 bucket with source_documents/")
    p_s3i.add_argument("--workdir", default="./work")
    p_s3i.add_argument("--no-llm", action="store_true")
    p_s3i.add_argument("--ocr-backend", choices=["ollama", "anthropic", "bedrock"],
                       default="bedrock",
                       help="OCR backend (default: bedrock for S3 ingest)")
    p_s3i.add_argument("--event-backend",
                       choices=["gemini", "openrouter", "llama", "ollama", "bedrock", "regex"],
                       default="bedrock",
                       help="Event extraction backend (default: bedrock for S3 ingest)")
    p_s3i.add_argument("--export-parquet", action="store_true",
                       help="Also export events to Parquet and upload for Athena")
    p_s3i.set_defaults(func=_s3_ingest)

    # --- kb-sync ---
    p_kb = sub.add_parser("kb-sync",
                          help="Trigger Bedrock Knowledge Base data source sync")
    p_kb.add_argument("--data-source-id", required=True,
                      help="KB data source ID to sync")
    p_kb.add_argument("--kb-id", default=None,
                      help="Knowledge Base ID (or set BEDROCK_KB_ID env)")
    p_kb.add_argument("--wait", action="store_true",
                      help="Wait for sync to complete")
    p_kb.set_defaults(func=_kb_sync)

    args = parser.parse_args(argv)

    if args.cmd == "query" and not args.question and not args.interactive:
        args.interactive = True

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
