"""S3 storage layer for documents, markdown, and database files.

Replaces local filesystem operations with S3 for cloud-native deployment.
Documents are stored as:
    s3://<bucket>/source_documents/     -- original PDFs/DOCs
    s3://<bucket>/raw_markdown/         -- VLM OCR output
    s3://<bucket>/clean_markdown/       -- cleaned markdown
    s3://<bucket>/database/case.db      -- SQLite database (downloaded for local use)
    s3://<bucket>/timelines/            -- generated timeline outputs

Configuration via env vars:
    S3_BUCKET          -- bucket name (required)
    AWS_DEFAULT_REGION -- region (default: us-east-1)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


class S3StorageError(RuntimeError):
    pass


class S3Storage:
    def __init__(self, bucket: str | None = None, region: str | None = None):
        import boto3

        self.bucket = bucket or os.environ.get("S3_BUCKET")
        if not self.bucket:
            raise S3StorageError("S3_BUCKET env var or bucket parameter is required")
        self.region = region or DEFAULT_REGION
        self._s3 = boto3.client("s3", region_name=self.region)

    def upload_file(self, local_path: str | Path, s3_key: str) -> str:
        local_path = Path(local_path)
        self._s3.upload_file(str(local_path), self.bucket, s3_key)
        uri = f"s3://{self.bucket}/{s3_key}"
        logger.info("Uploaded %s -> %s", local_path.name, uri)
        return uri

    def download_file(self, s3_key: str, local_path: str | Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket, s3_key, str(local_path))
        logger.info("Downloaded s3://%s/%s -> %s", self.bucket, s3_key, local_path)
        return local_path

    def upload_text(self, text: str, s3_key: str) -> str:
        self._s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=text.encode("utf-8"),
            ContentType="text/markdown",
        )
        return f"s3://{self.bucket}/{s3_key}"

    def download_text(self, s3_key: str) -> str:
        response = self._s3.get_object(Bucket=self.bucket, Key=s3_key)
        return response["Body"].read().decode("utf-8")

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def key_exists(self, s3_key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except self._s3.exceptions.ClientError:
            return False

    def upload_source_document(self, local_path: str | Path) -> str:
        path = Path(local_path)
        return self.upload_file(path, f"source_documents/{path.name}")

    def upload_raw_markdown(self, local_path: str | Path) -> str:
        path = Path(local_path)
        return self.upload_file(path, f"raw_markdown/{path.name}")

    def upload_clean_markdown(self, local_path: str | Path) -> str:
        path = Path(local_path)
        return self.upload_file(path, f"clean_markdown/{path.name}")

    def upload_database(self, local_path: str | Path) -> str:
        return self.upload_file(local_path, "database/case.db")

    def download_database(self, local_path: str | Path) -> Path:
        return self.download_file("database/case.db", local_path)

    def upload_timeline(self, content: str, filename: str) -> str:
        return self.upload_text(content, f"timelines/{filename}")

    def list_source_documents(self) -> list[str]:
        return self.list_keys("source_documents/")

    def list_clean_markdown(self) -> list[str]:
        return self.list_keys("clean_markdown/")

    def download_source_to_tempdir(self) -> tuple[Path, list[Path]]:
        """Download all source documents to a temp directory for processing."""
        keys = self.list_source_documents()
        tmpdir = Path(tempfile.mkdtemp(prefix="s3_docs_"))
        paths = []
        for key in keys:
            filename = key.split("/")[-1]
            if not filename:
                continue
            local = tmpdir / filename
            self.download_file(key, local)
            paths.append(local)
        return tmpdir, paths

    def sync_workdir_to_s3(self, workdir: str | Path) -> dict:
        """Upload all pipeline output from a local workdir to S3."""
        workdir = Path(workdir)
        uploaded = {"raw_markdown": [], "clean_markdown": [], "database": None}

        raw_dir = workdir / "raw_md"
        if raw_dir.exists():
            for f in raw_dir.iterdir():
                if f.suffix == ".md":
                    self.upload_raw_markdown(f)
                    uploaded["raw_markdown"].append(f.name)

        clean_dir = workdir / "clean_md"
        if clean_dir.exists():
            for f in clean_dir.iterdir():
                if f.suffix == ".md":
                    self.upload_clean_markdown(f)
                    uploaded["clean_markdown"].append(f.name)

        db_path = workdir / "case.db"
        if db_path.exists():
            self.upload_database(db_path)
            uploaded["database"] = "case.db"

        logger.info(
            "Synced workdir to S3: %d raw, %d clean, db=%s",
            len(uploaded["raw_markdown"]),
            len(uploaded["clean_markdown"]),
            uploaded["database"],
        )
        return uploaded
