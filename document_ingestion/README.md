# document_ingestion

Pipeline: **PDF / DOC / DOCX → OCR → cleaned Markdown → date-anchored events → SQLite → S3 → Bedrock Knowledge Base + Athena**

## Architecture

- **OCR**: Ollama (local), Anthropic Claude, or AWS Bedrock (cloud)
- **Event extraction**: Gemini, OpenRouter/Llama, Ollama, AWS Bedrock, or regex fallback
- **Document storage**: Local filesystem or AWS S3
- **Document Q&A**: AWS Bedrock managed Knowledge Base (auto-chunking, embedding, vector retrieval via OpenSearch Serverless)
- **Structured queries**: AWS Athena over Parquet data via Glue Data Catalog
- **Foundation model**: Claude 3.5 Sonnet for generation

## Stages

1. **OCR (VLM).** Each page is rendered to PNG and sent to a vision model, which returns Markdown.
2. **Clean.** Strip page numbers, fix hyphenation, collapse whitespace.
3. **Extract.** An LLM identifies events with dates and returns structured `{date, action, party, court, excerpt}` records. A grounding gate enforces verbatim excerpt matching.
4. **Persist.** SQLite tables — `dates`, `events`, `case_actors`, `event_actors`.
5. **Export.** SQLite → Parquet → S3 for Athena. Clean markdown → S3 for Bedrock Knowledge Base.
6. **Query.** Bedrock KB for document-level Q&A; Athena SQL for structured event analytics.

## Prerequisites

1. AWS account with Bedrock, S3, Athena, Glue, and OpenSearch Serverless access
2. Terraform >= 1.5 for infrastructure provisioning
3. Python deps: `pip install -r document_ingestion/requirements.txt`
4. (Optional) Ollama for local OCR: `ollama serve && ollama pull llama3.2-vision`

## Infrastructure

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your bucket name and settings
terraform init && terraform apply
```

This provisions: S3 buckets, Bedrock Knowledge Base, OpenSearch Serverless, Glue Data Catalog, Athena workgroup, IAM roles.

## Quick start

```bash
# Ingest documents with Bedrock OCR + event extraction, export Parquet for Athena
python -m document_ingestion.cli s3-ingest --s3-bucket YOUR_BUCKET --workdir ./work --export-parquet

# Trigger Knowledge Base sync (after uploading clean markdown to S3)
python -m document_ingestion.cli kb-sync --data-source-id DS_ID --wait

# Query (auto-routes: document questions → KB, analytics → Athena)
BEDROCK_KB_ID=xxx ATHENA_DATABASE=doc_rag_db python -m document_ingestion.cli query -i

# Direct Athena SQL
python -m document_ingestion.cli athena-query "SELECT court, COUNT(*) FROM doc_rag_db.events GROUP BY court"
```

## Environment Variables

```bash
export BEDROCK_KB_ID="your-knowledge-base-id"
export ATHENA_DATABASE="doc_rag_db"
export ATHENA_WORKGROUP="doc-rag-workgroup"
export S3_BUCKET="your-bucket-name"
export AWS_DEFAULT_REGION="us-east-1"
```

## Configuration

Override Ollama settings via `.docing.json`:

```json
{
    "ollama_host": "http://localhost:11434",
    "vision_model": "llama3.2-vision",
    "text_model": "llama3.1"
}
```

Or via environment variables: `OLLAMA_HOST`, `DOCING_VISION_MODEL`, `DOCING_TEXT_MODEL`.

## Swapping the VLM

Any object implementing `VLMClient` protocol (`transcribe(image_png: bytes) -> str`) works:

```python
IngestionPipeline(workdir="./work", vlm=MyCustomVLM())
```
