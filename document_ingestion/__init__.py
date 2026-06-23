"""Document ingestion pipeline: PDF/DOC -> OCR -> Markdown -> Events -> SQLite -> Timeline.

Supports multiple backends:
  - OCR: Ollama (local), Anthropic (cloud), AWS Bedrock (cloud)
  - Event extraction: Gemini, OpenRouter/Llama, Ollama, AWS Bedrock, regex
  - Storage: local filesystem, AWS S3
  - Document queries: AWS Bedrock managed Knowledge Base (auto-chunking, embedding, retrieval)
  - Structured event queries: AWS Athena over Parquet via Glue Data Catalog
"""

from .pipeline import IngestionPipeline
from .database import EventStore
from .timeline import generate_timeline

__all__ = ["IngestionPipeline", "EventStore", "generate_timeline"]
