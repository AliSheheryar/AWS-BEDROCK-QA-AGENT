# AWS Bedrock QA Agent

**Document QA Agent** powered by AWS Bedrock Knowledge Base + Athena NL-to-SQL over Parquet/Glue.

Ingest documents (PDF/DOC/DOCX) → OCR → extract structured events → query via natural language using two backends:
- **Bedrock Knowledge Base** — document-level Q&A with auto-chunking, embedding, and vector retrieval
- **Athena + Glue** — structured event analytics via natural language to SQL conversion

## Architecture

### Ingestion Pipeline
![Ingestion Pipeline](diagrams/ingestion_pipeline.svg)

**Local stages:** PDF → OCR → Clean Markdown → LLM Event Extraction → SQLite  
**Post-ingest sync:** SQLite → Parquet → S3 (for Athena) + Clean Markdown → S3 (for Bedrock KB)

### Query Flow — NL to SQL (Athena Path)
![NL-to-SQL Query Flow](diagrams/query_flow_nl_to_sql.svg)

User question → Bedrock Claude converts to SQL → Athena executes against Glue schema → reads Parquet from S3 → structured results.

### Query Flow — Document Q&A (Bedrock KB Path)
![Bedrock Knowledge Base Flow](diagrams/bedrock_knowledge_base_flow.svg)

User question → Bedrock KB `RetrieveAndGenerate` → vector search in OpenSearch Serverless → Claude generates cited answer from retrieved chunks.

### How Athena Uses Glue
![Athena-Glue Relationship](diagrams/athena_glue_relationship.svg)

- **Glue** = metadata only (column names, types, S3 locations)
- **Athena** = compute only (parses SQL, reads from S3)
- **S3** = storage only (Parquet files, columnar format)

## Tech Stack

| Component | Service |
|---|---|
| OCR | Ollama (local), Anthropic Claude, or AWS Bedrock |
| Event Extraction | Gemini, Llama, Ollama, Bedrock, or regex fallback |
| Document Storage | AWS S3 |
| Document Q&A | AWS Bedrock managed Knowledge Base (OpenSearch Serverless) |
| Embeddings | Amazon Titan Embeddings V2 (1024-dim) |
| Structured Queries | Amazon Athena over Parquet via Glue Data Catalog |
| Foundation Model | Claude 3.5 Sonnet |
| Infrastructure | Terraform |

## Quick Start

### 1. Deploy Infrastructure

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings
terraform init && terraform apply
```

This provisions: S3 buckets, Bedrock Knowledge Base, OpenSearch Serverless, Glue Data Catalog, Athena workgroup, IAM roles.

### 2. Install Dependencies

```bash
pip install -r document_ingestion/requirements.txt
```

### 3. Ingest Documents

```bash
# Ingest with Bedrock OCR + event extraction, export Parquet for Athena
python -m document_ingestion.cli s3-ingest \
  --s3-bucket YOUR_BUCKET \
  --workdir ./work \
  --export-parquet
```

### 4. Sync Knowledge Base

```bash
python -m document_ingestion.cli kb-sync --data-source-id DS_ID --wait
```

### 5. Query

```bash
# Interactive query (auto-routes between KB and Athena)
BEDROCK_KB_ID=xxx ATHENA_DATABASE=doc_rag_db \
  python -m document_ingestion.cli query -i

# Direct Athena SQL
python -m document_ingestion.cli athena-query \
  "SELECT court, COUNT(*) FROM doc_rag_db.events GROUP BY court"
```

## Environment Variables

```bash
export BEDROCK_KB_ID="your-knowledge-base-id"
export ATHENA_DATABASE="doc_rag_db"
export ATHENA_WORKGROUP="doc-rag-workgroup"
export S3_BUCKET="your-bucket-name"
export AWS_DEFAULT_REGION="us-east-1"
```

## CLI Commands

| Command | Description |
|---|---|
| `ingest` | Local PDF ingestion (OCR + extract + SQLite) |
| `s3-ingest` | Ingest + upload to S3 (with optional `--export-parquet`) |
| `export-parquet` | Export SQLite → Parquet → S3 |
| `kb-sync` | Trigger Bedrock Knowledge Base sync |
| `query` | Interactive Q&A (auto-routes KB vs Athena) |
| `athena-query` | Direct Athena SQL execution |
| `timeline` | Generate chronological event timeline |
| `ledger` | Export event ledger |
| `s3-sync` | Sync local files to S3 |

## Project Structure

```
├── document_ingestion/
│   ├── cli.py                    # CLI entry point (9 subcommands)
│   ├── pipeline.py               # Ingestion pipeline orchestrator
│   ├── ocr.py                    # OCR stage (multi-backend)
│   ├── cleaner.py                # Markdown cleaning
│   ├── event_extractor.py        # LLM event extraction + grounding gate
│   ├── database.py               # SQLite event store
│   ├── entity_resolver.py        # Entity resolution (configurable)
│   ├── timeline.py               # Timeline generation
│   ├── rag_query.py              # Unified query engine (KB + Athena router)
│   ├── bedrock_knowledge_base.py # Bedrock KB client (RetrieveAndGenerate)
│   ├── athena_store.py           # Athena client + Parquet export
│   ├── bedrock_client.py         # Bedrock foundation model client
│   ├── bedrock_vision.py         # Bedrock vision (OCR)
│   ├── s3_storage.py             # S3 upload/download
│   └── tests/
├── terraform/
│   ├── main.tf                   # All AWS resources
│   ├── variables.tf
│   ├── outputs.tf
│   └── terraform.tfvars.example
├── diagrams/                     # Architecture diagrams (SVG)
└── qwen_ocr.py                  # Standalone Qwen VLM OCR script
```
