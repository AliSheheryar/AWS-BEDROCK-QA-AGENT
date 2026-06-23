"""RAG query engine: Bedrock Knowledge Base for documents, Athena for structured events.

Two query backends:
  1. Bedrock Knowledge Base — for document-level Q&A ("what did the motion say?")
     Uses RetrieveAndGenerate with auto-chunked/embedded clean markdown in S3.
  2. Athena — for structured event analytics ("how many motions in 2024?")
     Queries Parquet event data via Glue catalog + Athena SQL.

A router decides which backend to use based on the question type.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    backend: str
    citations: list[dict] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)
    athena_data: list[dict] = field(default_factory=list)
    model: str = ""


_ANALYTICS_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\blist all\b",
    r"\bhow often\b",
    r"\bfrequency\b",
    r"\bgroup by\b",
    r"\bbreak\s*down\b",
    r"\bsummar(y|ize)\b",
    r"\bstatistic\b",
    r"\btrend\b",
    r"\btimeline\b",
    r"\bper (month|year|court|party)\b",
    r"\bbetween .+ and .+\b",
    r"\brange\b",
    r"\btop \d+\b",
    r"\bmost (common|frequent)\b",
]


def _is_analytics_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _ANALYTICS_PATTERNS)


class RAGQueryEngine:
    """Unified query engine: routes to Bedrock KB or Athena based on question type."""

    def __init__(
        self,
        knowledge_base_id: str | None = None,
        athena_database: str | None = None,
        athena_workgroup: str | None = None,
        model_arn: str | None = None,
        region: str | None = None,
    ):
        self._kb = None
        self._athena = None
        self._kb_id = knowledge_base_id or os.environ.get("BEDROCK_KB_ID")
        self._athena_db = athena_database
        self._athena_wg = athena_workgroup
        self._model_arn = model_arn
        self._region = region

    def _get_kb(self):
        if self._kb is None:
            from .bedrock_knowledge_base import BedrockKnowledgeBase
            self._kb = BedrockKnowledgeBase(
                knowledge_base_id=self._kb_id,
                model_arn=self._model_arn,
                region=self._region,
            )
        return self._kb

    def _get_athena(self):
        if self._athena is None:
            from .athena_store import AthenaEventStore
            self._athena = AthenaEventStore(
                database=self._athena_db,
                workgroup=self._athena_wg,
                region=self._region,
            )
        return self._athena

    def query(self, question: str) -> RAGResult:
        if _is_analytics_question(question):
            return self._query_athena(question)
        return self._query_knowledge_base(question)

    def query_documents(self, question: str) -> RAGResult:
        return self._query_knowledge_base(question)

    def query_events(self, question: str) -> RAGResult:
        return self._query_athena(question)

    def _query_knowledge_base(self, question: str) -> RAGResult:
        kb = self._get_kb()
        result = kb.query(question)
        return RAGResult(
            answer=result.answer,
            backend="bedrock-kb",
            citations=result.citations,
            source_documents=[c["s3_uri"] for c in result.citations if c.get("s3_uri")],
            model=result.model_arn,
        )

    def _query_athena(self, question: str) -> RAGResult:
        athena = self._get_athena()
        sql = self._question_to_sql(question)
        try:
            result = athena.execute_query(sql)
            answer = self._format_athena_result(question, result.rows, result.columns)
            return RAGResult(
                answer=answer,
                backend="athena",
                athena_data=result.rows,
                model="athena-sql",
            )
        except Exception as exc:
            logger.warning("Athena query failed, falling back to KB: %s", exc)
            return self._query_knowledge_base(question)

    def _question_to_sql(self, question: str) -> str:
        """Convert a natural language question to an Athena SQL query.

        Uses the Bedrock model to generate SQL, falling back to a simple search.
        """
        from .bedrock_client import BedrockClient

        prompt = (
            "Convert this question to a SQL query for an events table in Athena.\n"
            "The database name is: " + (self._athena_db or "doc_rag_db") + "\n"
            "Table: events\n"
            "Columns: event_id, date, action, party, court, source_documents, "
            "excerpt, other_actors, role_gate, canonical_actor_id, "
            "confidence, forum_id, forum_role, raw_action, grounding_excerpt\n\n"
            "Also available: case_actors table with columns: "
            "name, role, mentions, canonical_id\n\n"
            "Return ONLY the SQL query, no explanation.\n\n"
            f"Question: {question}\nSQL:"
        )

        try:
            client = BedrockClient()
            raw = client.generate(prompt, temperature=0.0)
            sql = raw.strip().strip("```").strip("sql").strip()
            if sql.upper().startswith("SELECT"):
                return sql
        except Exception:
            pass

        db = self._athena_db or "doc_rag_db"
        return f"SELECT * FROM {db}.events ORDER BY date LIMIT 100"

    def _format_athena_result(
        self, question: str, rows: list[dict], columns: list[str],
    ) -> str:
        if not rows:
            return "No matching events found."
        if len(rows) == 1 and len(columns) == 1:
            col = columns[0]
            return f"{col}: {rows[0][col]}"

        parts = [f"Found {len(rows)} result(s):\n"]
        for row in rows[:30]:
            parts.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        if len(rows) > 30:
            parts.append(f"... and {len(rows) - 30} more rows")
        return "\n".join(parts)


def create_rag_engine(
    knowledge_base_id: str | None = None,
    athena_database: str | None = None,
    athena_workgroup: str | None = None,
    model_arn: str | None = None,
    region: str | None = None,
    **_kwargs,
) -> RAGQueryEngine:
    """Factory: create a RAG engine from config / env vars."""
    return RAGQueryEngine(
        knowledge_base_id=knowledge_base_id,
        athena_database=athena_database,
        athena_workgroup=athena_workgroup,
        model_arn=model_arn,
        region=region,
    )
