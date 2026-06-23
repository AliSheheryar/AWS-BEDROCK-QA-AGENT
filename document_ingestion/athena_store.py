"""Athena-backed event store: export events to Parquet in S3, query via Athena.

Replaces SQLite for analytics queries in the cloud deployment. The pipeline
still uses SQLite locally during ingestion, then exports to Parquet for Athena.

Architecture:
  SQLite (local ingestion) -> Parquet export -> S3 -> Glue Data Catalog -> Athena

Configuration:
  ATHENA_DATABASE       -- Glue database name (default: doc_rag_db)
  ATHENA_WORKGROUP      -- Athena workgroup (default: doc-rag-workgroup)
  ATHENA_RESULTS_BUCKET -- S3 bucket for query results
  AWS_DEFAULT_REGION    -- region (default: us-east-1)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_DATABASE = os.getenv("ATHENA_DATABASE", "doc_rag_db")
DEFAULT_WORKGROUP = os.getenv("ATHENA_WORKGROUP", "doc-rag-workgroup")
MAX_WAIT = 60


class AthenaError(RuntimeError):
    pass


@dataclass
class AthenaQueryResult:
    columns: list[str]
    rows: list[dict]
    row_count: int
    execution_time_ms: int


class AthenaEventStore:
    """Query the structured event data in S3 via Athena."""

    def __init__(
        self,
        database: str | None = None,
        workgroup: str | None = None,
        region: str | None = None,
    ):
        import boto3

        self.database = database or DEFAULT_DATABASE
        self.workgroup = workgroup or DEFAULT_WORKGROUP
        self.region = region or DEFAULT_REGION
        self._athena = boto3.client("athena", region_name=self.region)

    def execute_query(self, sql: str) -> AthenaQueryResult:
        """Run a SQL query against the events table and wait for results."""
        response = self._athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            WorkGroup=self.workgroup,
        )
        execution_id = response["QueryExecutionId"]

        elapsed = 0
        while elapsed < MAX_WAIT:
            status = self._athena.get_query_execution(QueryExecutionId=execution_id)
            state = status["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(1)
            elapsed += 1

        if state != "SUCCEEDED":
            reason = status["QueryExecution"]["Status"].get("StateChangeReason", "")
            raise AthenaError(f"Query {state}: {reason}")

        exec_ms = int(
            status["QueryExecution"]
            .get("Statistics", {})
            .get("EngineExecutionTimeInMillis", 0)
        )

        paginator = self._athena.get_paginator("get_query_results")
        all_rows: list[list[str]] = []
        for page in paginator.paginate(QueryExecutionId=execution_id):
            rs = page["ResultSet"]
            if not all_rows:
                all_rows.append([c["VarCharValue"] for c in rs["Rows"][0]["Data"]])
                data_rows = rs["Rows"][1:]
            else:
                data_rows = rs["Rows"]
            for row in data_rows:
                all_rows.append([c.get("VarCharValue", "") for c in row["Data"]])

        columns = all_rows[0] if all_rows else []
        records = [dict(zip(columns, r)) for r in all_rows[1:]]

        return AthenaQueryResult(
            columns=columns,
            rows=records,
            row_count=len(records),
            execution_time_ms=exec_ms,
        )

    def count_events(self, date: str | None = None, party: str | None = None) -> int:
        sql = f"SELECT COUNT(*) AS cnt FROM {self.database}.events WHERE 1=1"
        if date:
            sql += f" AND date = '{date}'"
        if party:
            sql += f" AND LOWER(party) LIKE '%{party.lower()}%'"
        result = self.execute_query(sql)
        return int(result.rows[0]["cnt"]) if result.rows else 0

    def events_by_date(self, date: str) -> list[dict]:
        sql = (
            f"SELECT event_id, date, action, party, court, source_documents, excerpt "
            f"FROM {self.database}.events "
            f"WHERE date = '{date}' ORDER BY event_id"
        )
        return self.execute_query(sql).rows

    def events_by_court(self) -> list[dict]:
        sql = (
            f"SELECT court, COUNT(*) AS count "
            f"FROM {self.database}.events "
            f"WHERE court != 'undetermined' "
            f"GROUP BY court ORDER BY count DESC"
        )
        return self.execute_query(sql).rows

    def events_by_party(self, limit: int = 20) -> list[dict]:
        sql = (
            f"SELECT party, COUNT(*) AS count "
            f"FROM {self.database}.events "
            f"WHERE party != 'undetermined' "
            f"GROUP BY party ORDER BY count DESC LIMIT {limit}"
        )
        return self.execute_query(sql).rows

    def events_in_range(self, start_date: str, end_date: str) -> list[dict]:
        sql = (
            f"SELECT event_id, date, action, party, court, source_documents "
            f"FROM {self.database}.events "
            f"WHERE date >= '{start_date}' AND date <= '{end_date}' "
            f"ORDER BY date"
        )
        return self.execute_query(sql).rows

    def timeline_summary(self) -> list[dict]:
        sql = (
            f"SELECT date, COUNT(*) AS event_count "
            f"FROM {self.database}.events "
            f"WHERE date != 'undetermined' "
            f"GROUP BY date ORDER BY date"
        )
        return self.execute_query(sql).rows

    def search_events(self, keyword: str) -> list[dict]:
        kw = keyword.lower().replace("'", "''")
        sql = (
            f"SELECT event_id, date, action, party, court, excerpt "
            f"FROM {self.database}.events "
            f"WHERE LOWER(action) LIKE '%{kw}%' "
            f"   OR LOWER(party) LIKE '%{kw}%' "
            f"   OR LOWER(excerpt) LIKE '%{kw}%' "
            f"ORDER BY date LIMIT 50"
        )
        return self.execute_query(sql).rows

    def actor_summary(self) -> list[dict]:
        sql = (
            f"SELECT name, role, mentions "
            f"FROM {self.database}.case_actors "
            f"ORDER BY mentions DESC"
        )
        return self.execute_query(sql).rows

    def custom_query(self, sql: str) -> AthenaQueryResult:
        return self.execute_query(sql)


def export_sqlite_to_parquet(
    db_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export all SQLite tables to Parquet files for S3/Athena.

    Returns a dict of {table_name: parquet_path}.
    """
    import sqlite3
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required for Parquet export: pip install pandas pyarrow")

    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    exported = {}
    for table in tables:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        parquet_path = output_dir / f"{table}.parquet"
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        exported[table] = parquet_path
        logger.info("Exported %s -> %s (%d rows)", table, parquet_path, len(df))

    conn.close()
    return exported


def upload_parquet_to_s3(
    parquet_files: dict[str, Path],
    s3_bucket: str,
    prefix: str = "athena_data",
    region: str | None = None,
) -> dict[str, str]:
    """Upload exported Parquet files to S3 under the Athena data prefix."""
    import boto3

    s3 = boto3.client("s3", region_name=region or DEFAULT_REGION)
    uploaded = {}
    for table_name, local_path in parquet_files.items():
        s3_key = f"{prefix}/{table_name}/{table_name}.parquet"
        s3.upload_file(str(local_path), s3_bucket, s3_key)
        uri = f"s3://{s3_bucket}/{s3_key}"
        uploaded[table_name] = uri
        logger.info("Uploaded %s -> %s", local_path.name, uri)
    return uploaded
