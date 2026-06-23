"""Bedrock managed Knowledge Base client.

Uses the Bedrock Agent Runtime `RetrieveAndGenerate` API so all retrieval
(chunking, embedding, vector search) is handled by AWS — no custom scoring.

The Knowledge Base is backed by:
  - Data source: S3 bucket with clean markdown documents
  - Vector store: Amazon OpenSearch Serverless (provisioned by Terraform)
  - Embedding model: Amazon Titan Embeddings V2
  - Foundation model: Claude 3.5 Sonnet (for generation)

Configuration:
  BEDROCK_KB_ID           -- Knowledge Base ID (required)
  BEDROCK_MODEL_ARN       -- model ARN for generation (optional, defaults to Sonnet 3.5)
  AWS_DEFAULT_REGION      -- region (default: us-east-1)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_MODEL_ARN = os.getenv(
    "BEDROCK_MODEL_ARN",
    "arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
)


class KnowledgeBaseError(RuntimeError):
    pass


@dataclass
class KBRetrievalResult:
    """Single chunk returned by the Knowledge Base retrieval."""
    text: str
    score: float
    s3_uri: str
    metadata: dict = field(default_factory=dict)


@dataclass
class KBQueryResult:
    """Full result from a RetrieveAndGenerate call."""
    answer: str
    citations: list[dict]
    retrieved_chunks: list[KBRetrievalResult]
    model_arn: str


class BedrockKnowledgeBase:
    """Client for AWS Bedrock managed Knowledge Base."""

    def __init__(
        self,
        knowledge_base_id: str | None = None,
        model_arn: str | None = None,
        region: str | None = None,
    ):
        import boto3

        self.kb_id = knowledge_base_id or os.environ.get("BEDROCK_KB_ID")
        if not self.kb_id:
            raise KnowledgeBaseError(
                "BEDROCK_KB_ID env var or knowledge_base_id parameter is required"
            )
        self.region = region or DEFAULT_REGION
        self.model_arn = (model_arn or DEFAULT_MODEL_ARN).replace("{region}", self.region)
        self._client = boto3.client("bedrock-agent-runtime", region_name=self.region)
        self._agent_client = boto3.client("bedrock-agent", region_name=self.region)

    def query(self, question: str, max_results: int = 5) -> KBQueryResult:
        """Ask a question using RetrieveAndGenerate (retrieval + generation in one call)."""
        response = self._client.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": self.kb_id,
                    "modelArn": self.model_arn,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": max_results,
                        }
                    },
                    "generationConfiguration": {
                        "inferenceConfig": {
                            "textInferenceConfig": {
                                "temperature": 0.0,
                                "maxTokens": 4096,
                            }
                        },
                        "promptTemplate": {
                            "textPromptTemplate": (
                                "You are a document analyst. Answer the question "
                                "based ONLY on the provided search results. Cite "
                                "source documents. If the information is not in the "
                                "results, say so.\n\n$search_results$\n\n"
                                "Question: $query$\nAnswer:"
                            ),
                        },
                    },
                },
            },
        )

        answer = response["output"]["text"]
        citations = self._parse_citations(response.get("citations", []))
        chunks = self._extract_chunks(response.get("citations", []))

        return KBQueryResult(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            model_arn=self.model_arn,
        )

    def retrieve(self, question: str, max_results: int = 10) -> list[KBRetrievalResult]:
        """Retrieve relevant chunks without generation (for custom pipelines)."""
        response = self._client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": max_results,
                }
            },
        )

        results = []
        for item in response.get("retrievalResults", []):
            content = item.get("content", {}).get("text", "")
            score = item.get("score", 0.0)
            location = item.get("location", {})
            s3_uri = location.get("s3Location", {}).get("uri", "")
            metadata = item.get("metadata", {})
            results.append(KBRetrievalResult(
                text=content, score=score, s3_uri=s3_uri, metadata=metadata,
            ))
        return results

    def start_sync(self, data_source_id: str) -> str:
        """Trigger a Knowledge Base data source sync (after uploading new docs to S3)."""
        response = self._agent_client.start_ingestion_job(
            knowledgeBaseId=self.kb_id,
            dataSourceId=data_source_id,
        )
        job_id = response["ingestionJob"]["ingestionJobId"]
        logger.info("Started KB sync job %s for data source %s", job_id, data_source_id)
        return job_id

    def get_sync_status(self, data_source_id: str, job_id: str) -> dict:
        """Check the status of a sync job."""
        response = self._agent_client.get_ingestion_job(
            knowledgeBaseId=self.kb_id,
            dataSourceId=data_source_id,
            ingestionJobId=job_id,
        )
        job = response["ingestionJob"]
        return {
            "status": job["status"],
            "started": str(job.get("startedAt", "")),
            "updated": str(job.get("updatedAt", "")),
            "stats": job.get("statistics", {}),
        }

    def _parse_citations(self, citations: list) -> list[dict]:
        parsed = []
        for cit in citations:
            for ref in cit.get("retrievedReferences", []):
                loc = ref.get("location", {})
                s3_uri = loc.get("s3Location", {}).get("uri", "")
                text = ref.get("content", {}).get("text", "")[:200]
                parsed.append({"s3_uri": s3_uri, "excerpt": text})
        return parsed

    def _extract_chunks(self, citations: list) -> list[KBRetrievalResult]:
        chunks = []
        for cit in citations:
            for ref in cit.get("retrievedReferences", []):
                loc = ref.get("location", {})
                s3_uri = loc.get("s3Location", {}).get("uri", "")
                text = ref.get("content", {}).get("text", "")
                metadata = ref.get("metadata", {})
                chunks.append(KBRetrievalResult(
                    text=text, score=0.0, s3_uri=s3_uri, metadata=metadata,
                ))
        return chunks
